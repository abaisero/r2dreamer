import atexit
import pathlib
import sys
import warnings

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    if HydraConfig.get().mode == RunMode.MULTIRUN:
        raise RuntimeError(
            "--multirun is not supported: every job resolves the same ${now:...} "
            "logdir and overwrites the others' metrics and checkpoints.  Launch "
            "one `python train.py` per configuration instead, as runs/*.sh do."
        )

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    # wandb is configured entirely through env vars (WANDB_PROJECT, WANDB_ENTITY,
    # WANDB_TAGS, WANDB_RUN_GROUP, ...).  On by default; WANDB_MODE=offline or
    # WANDB_MODE=disabled opts out.  Used as a context manager so the run is
    # closed on the way out, which also marks it failed if training raises.
    with wandb.init(dir=str(logdir), config=OmegaConf.to_container(config, resolve=True)):
        # Chart everything against the logged "step" value instead of wandb's
        # internal counter, which must increase monotonically.
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

        logger = tools.Logger(logdir)
        # save config
        logger.log_hydra_config(config)

        replay_buffer = Buffer(config.buffer)

        print("Create envs.")
        train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

        print("Simulate agent.")
        agent = Dreamer(
            config.model,
            obs_space,
            act_space,
        ).to(config.device)

        policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
        policy_trainer.begin(agent)

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
