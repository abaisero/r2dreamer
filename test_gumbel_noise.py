"""Checks for the exogenous transition noise threaded through the imagination rollout."""

import math

import pytest
import torch

import rssm
from distributions import OneHotDist, sample_exogenous_noise, uniform_to_gumbel


@pytest.mark.parametrize("kind", ["u", "g"])
def test_sample_exogenous_noise_is_one_tensor(kind):
    noise = sample_exogenous_noise((4, 6, 3, 5), torch.device("cpu"), kind)
    assert isinstance(noise, torch.Tensor)
    assert noise.shape == (4, 6, 3, 5)
    assert not noise.requires_grad

    # indexing the rollout depends on
    assert noise[:, 2].shape == (4, 3, 5)
    assert noise[:, :-1].shape == (4, 5, 3, 5)


def test_sample_exogenous_noise_rejects_unknown_kind():
    with pytest.raises(ValueError):
        sample_exogenous_noise((2,), torch.device("cpu"), "gumbel")


def test_u_is_uniform_and_g_is_gumbel():
    torch.manual_seed(0)
    u = sample_exogenous_noise((200_000,), torch.device("cpu"), "u")
    assert 0.0 < u.min().item() and u.max().item() < 1.0
    assert abs(u.mean().item() - 0.5) < 0.01
    assert abs(u.var().item() - 1 / 12) < 0.01

    g = sample_exogenous_noise((200_000,), torch.device("cpu"), "g")
    assert abs(g.mean().item() - 0.5772156649) < 0.01
    assert abs(g.var().item() - math.pi**2 / 6) < 0.05


def test_uniform_to_gumbel_matches_the_sampler():
    # same seed, so both calls draw the same underlying uniforms
    torch.manual_seed(0)
    u = sample_exogenous_noise((4, 6), torch.device("cpu"), "u")
    torch.manual_seed(0)
    g = sample_exogenous_noise((4, 6), torch.device("cpu"), "g")
    assert torch.equal(g, uniform_to_gumbel(u))


def test_rsample_with_noise():
    logits = torch.randn(4, 3, 5, requires_grad=True)
    g = sample_exogenous_noise((4, 3, 5), torch.device("cpu"), "g")

    # given the noise, sampling is deterministic
    s = OneHotDist(logits).rsample(noise=g)
    assert torch.equal(s, OneHotDist(logits).rsample(noise=g))

    # and it is the argmax of logits + g, one-hot
    unimixed = OneHotDist(logits).logits
    expected = torch.nn.functional.one_hot((unimixed + g).argmax(-1), 5).to(s)
    assert torch.equal(s, expected)
    assert torch.equal(s.sum(-1), torch.ones(4, 3))

    # straight-through: gradient still reaches the logits
    s.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0

    # without noise it still samples one-hot, and differs between calls
    d = OneHotDist(logits.detach())
    draws = torch.stack([d.rsample() for _ in range(50)])
    assert torch.equal(draws.sum(-1), torch.ones(50, 4, 3))
    assert not torch.equal(draws[0], draws[-1])


def test_img_step_is_determined_by_the_noise():
    torch.manual_seed(0)
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        dict(
            stoch=4, deter=32, hidden=16, discrete=5, act="SiLU", unimix_ratio=0.01,
            initial="zeros", device="cpu", obs_layers=1, img_layers=1, dyn_layers=1, blocks=2,
        )
    )
    model = rssm.RSSM(config, embed_size=8, act_dim=3)
    stoch, deter = model.initial(2)
    action = torch.zeros(2, 3)
    g = sample_exogenous_noise((2, 4, 5), torch.device("cpu"), "g")

    # same noise, same action -> same next state
    a = model.img_step(stoch, deter, action, g)
    b = model.img_step(stoch, deter, action, g)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    # same noise, different action -> different next state
    other = torch.ones(2, 3)
    c = model.img_step(stoch, deter, other, g)
    assert not torch.equal(a[1], c[1])


if __name__ == "__main__":
    test_sample_exogenous_noise_is_one_tensor("u")
    test_sample_exogenous_noise_is_one_tensor("g")
    test_sample_exogenous_noise_rejects_unknown_kind()
    test_u_is_uniform_and_g_is_gumbel()
    test_uniform_to_gumbel_matches_the_sampler()
    test_rsample_with_noise()
    test_img_step_is_determined_by_the_noise()
    print("ok")
