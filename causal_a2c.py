import copy

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor, nn

import networks
import tools
from tools import to_f32


def _frozen_copy(module: nn.Module) -> nn.Module:
    """Independent copy of `module` whose parameters are never updated."""
    frozen = copy.deepcopy(module)
    for param in frozen.parameters():
        param.requires_grad_(False)
    return frozen


def _shared_frozen_copy(module: nn.Module) -> nn.Module:
    """Frozen copy of `module` that shares parameter storage with it.

    Sharing `.data` is what makes the copy track in-place updates of the live
    module.  NOTE: "requires_grad" affects whether a parameter is updated, not
    whether gradients flow through its operations.
    """
    frozen = copy.deepcopy(module)
    for (name_orig, param_orig), (name_new, param_new) in zip(
        module.named_parameters(), frozen.named_parameters()
    ):
        assert name_orig == name_new
        param_new.data = param_orig.data
        param_new.requires_grad_(False)
    return frozen


class CausalBaseline(nn.Module):
    """Value head conditioned on the state and on the future exogenous noise.

    The exogenous window at step t is every remaining draw, t .. T_imag-2.  A GRU
    run backwards over the noise sequence gives all those summaries in one pass.
    """

    def __init__(self, config: DictConfig, feat_size: int, noise_size: int):
        super().__init__()
        self.rnn = nn.GRU(noise_size, int(config.rnn_units), batch_first=True)
        self.head = networks.MLPHead(config, feat_size + int(config.rnn_units))

    def forward(self, feat: Tensor, noise: Tensor):
        breakpoint()
        # feat: (B, T_imag, F), noise: (B, T_imag-1, S, K) aligned to feat[:, 1:]
        B, T, _ = feat.shape
        # reversed so the GRU consumes the sequence from the end backwards
        h, _ = self.rnn(noise.reshape(B, T - 1, -1).flip(1))
        # h[:, j] summarizes draws T-2-j .. T-2; unflipping puts the summary of
        # draws t .. T-2 at index t.  The last step's window is empty, which the
        # GRU's zero initial state already stands for.
        window = torch.cat([h.flip(1), torch.zeros_like(h[:, :1])], 1)
        # (B, T_imag, H)
        return self.head(torch.cat([feat, window], -1))


class CausalA2C(nn.Module):
    """Actor-critic learning on top of world model rollouts.

    Owns the actor, the value network and its slow-moving target.  Receives an
    already-simulated imagination rollout from the world model and turns it into
    losses; it never touches the world model itself.
    """

    def __init__(
        self,
        config: DictConfig,
        feat_size: int,
        act_space: gym.spaces.Space,
        noise_size: int,
    ):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)

        config.actor.shape = (
            (act_space.n,)
            if hasattr(act_space, "n")
            else tuple(map(int, act_space.shape))
        )
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        self.actor = networks.MLPHead(config.actor, feat_size)
        self.value = networks.MLPHead(config.critic, feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = _frozen_copy(self.value)
        self._slow_value_updates = 0

        self.causal_baseline = CausalBaseline(
            config.causal_baseline, feat_size, noise_size
        )
        self._slow_causal_baseline = _frozen_copy(self.causal_baseline)

        self.gamma = 1 - 1 / self.horizon

    def named_modules_to_optimize(self) -> dict[str, nn.Module]:
        """Modules the outer optimizer must see, under their original names."""
        return {
            "actor": self.actor,
            "value": self.value,
            "causal_baseline": self.causal_baseline,
        }

    def update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
                for v, s in zip(
                    self.causal_baseline.parameters(),
                    self._slow_causal_baseline.parameters(),
                ):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode: bool = True):
        super().train(mode)
        # slow targets should be always eval mode
        self._slow_value.train(False)
        self._slow_causal_baseline.train(False)
        return self

    def clone_and_freeze(self):
        self.frozen_actor = _shared_frozen_copy(self.actor)
        self.frozen_value = _shared_frozen_copy(self.value)
        self.frozen_slow_value = _shared_frozen_copy(self._slow_value)
        self.frozen_causal_baseline = _shared_frozen_copy(self.causal_baseline)
        self.frozen_slow_causal_baseline = _shared_frozen_copy(
            self._slow_causal_baseline
        )

    @torch.no_grad()
    def lambda_return(
        self,
        last: Tensor,
        term: Tensor,
        reward: Tensor,
        value: Tensor,
        boot: Tensor,
    ) -> Tensor:
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * self.gamma
        cont = (1 - to_f32(last))[:, 1:] * self.lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    def losses(
        self,
        data: TensorDict,
        feat: Tensor,
        imag_feat: Tensor,
        imag_actions: Tensor,
        imag_rewards: Tensor,
        imag_continuation: Tensor,
        imag_noise: Tensor,
        losses: dict[str, Tensor],
        metrics: dict[str, Tensor | float],
    ):
        """Actor and value losses from an imagination rollout, plus replay value learning.

        `feat` is the live posterior feature from the world model and is left
        attached on purpose, so the replay value loss can push gradients back
        into the world model.
        """
        # data: dict of (B, T, *), feat: (B, T, F)
        # imag_*: (B*T, T_imag, *), already detached by the caller
        # imag_noise: (B*T, T_imag-1, S, K), the exogenous draws of the rollout
        B, T, _ = feat.shape

        imag_values = self.frozen_value(imag_feat).mode()
        # (B*T, T_imag, 1)
        imag_slow_values = self.frozen_slow_value(imag_feat).mode()
        # (B*T, T_imag, 1)
        weights = torch.cumprod(imag_continuation * self.gamma, dim=1)
        # (B*T, T_imag, 1)
        weights = weights[:, :-1].detach()
        # (B*T, T_imag-1, 1)
        is_last = torch.zeros_like(imag_continuation)
        # (B*T, T_imag, 1)
        imag_termination = 1 - imag_continuation
        # (B*T, T_imag, 1)
        returns = self.lambda_return(
            is_last, imag_termination, imag_rewards, imag_values, imag_values
        )
        # (B*T, T_imag-1, 1)
        imag_baselines = self.frozen_causal_baseline(imag_feat, imag_noise).mode()
        # (B*T, T_imag, 1)
        imag_baselines = imag_baselines[:, :-1]
        # (B*T, T_imag-1, 1)
        advantages = returns - imag_baselines
        # (B*T, T_imag-1, 1)

        returns_offset, returns_scale = self.return_ema(returns)
        returns_normalized = (returns - returns_offset) / returns_scale
        advantages = advantages / returns_scale
        # (B*T, T_imag-1, 1)

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_actions)[:, :-1].unsqueeze(-1)
        # (B*T, T_imag-1, 1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        # (B*T, T_imag-1, 1)
        losses["policy"] = torch.mean(
            weights * -(logpi * advantages.detach() + self.act_entropy * entropy)
        )

        imag_value_dist = self.value(imag_feat)
        returns_padded = torch.cat([returns, 0 * returns[:, -1:]], 1)
        # (B*T, T_imag, 1)
        cross_entropy_returns = -imag_value_dist.log_prob(
            returns_padded.detach()
        ).unsqueeze(-1)
        # (B*T, T_imag, 1)
        cross_entropy_returns = cross_entropy_returns[:, :-1]
        # (B*T, T_imag-1, 1)
        cross_entropy_imag_slow_values = -imag_value_dist.log_prob(
            imag_slow_values.detach()
        ).unsqueeze(-1)
        # (B*T, T_imag, 1)
        cross_entropy_imag_slow_values = cross_entropy_imag_slow_values[:, :-1]
        # (B*T, T_imag-1, 1)
        losses["value"] = torch.mean(
            weights * (cross_entropy_returns + cross_entropy_imag_slow_values)
        )

        imag_baseline_dist = self.causal_baseline(imag_feat, imag_noise)
        # (B*T, T_imag, 1)
        imag_slow_baselines = self.frozen_slow_causal_baseline(imag_feat, imag_noise).mode()
        # (B*T, T_imag, 1)
        cross_entropy_returns = -imag_baseline_dist.log_prob(
            returns_padded.detach()
        ).unsqueeze(-1)
        # (B*T, T_imag, 1)
        cross_entropy_returns = cross_entropy_returns[:, :-1]
        # (B*T, T_imag-1, 1)
        cross_entropy_imag_slow_baselines = -imag_baseline_dist.log_prob(
            imag_slow_baselines.detach()
        ).unsqueeze(-1)
        # (B*T, T_imag, 1)
        cross_entropy_imag_slow_baselines = cross_entropy_imag_slow_baselines[:, :-1]
        # (B*T, T_imag-1, 1)
        losses["causal_baseline"] = torch.mean(
            weights * (cross_entropy_returns + cross_entropy_imag_slow_baselines)
        )

        # log
        metrics["ret"] = torch.mean(returns_normalized)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(advantages)
        metrics["adv_std"] = torch.std(advantages)
        metrics["con"] = torch.mean(imag_continuation)
        metrics["rew"] = torch.mean(imag_rewards)
        metrics["val"] = torch.mean(imag_values)
        metrics["bas"] = torch.mean(imag_baselines)
        metrics["tar"] = torch.mean(returns)
        metrics["slowval"] = torch.mean(imag_slow_values)
        metrics["slowbas"] = torch.mean(imag_slow_baselines)
        metrics["weights"] = torch.mean(weights)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_actions, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        is_last, termination, rewards = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        bootstrap = returns[:, 0].reshape(B, T, 1)
        values = self.frozen_value(feat).mode()
        slow_values = self.frozen_slow_value(feat).mode()
        weights = 1.0 - is_last
        weights = weights[:, :-1]
        returns = self.lambda_return(is_last, termination, rewards, values, bootstrap)
        returns_padded = torch.cat([returns, 0 * returns[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through.
        # No replay counterpart for the causal baseline: the exogenous noise it is
        # meant to condition on does not exist on replay data.
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weights
            * (
                -value_dist.log_prob(returns_padded.detach())
                - value_dist.log_prob(slow_values.detach())
            )[:, :-1].unsqueeze(-1)
        )
        # log
        metrics.update(tools.tensorstats(returns, "ret_replay"))
        metrics.update(tools.tensorstats(values, "value_replay"))
        metrics.update(tools.tensorstats(slow_values, "slow_value_replay"))
