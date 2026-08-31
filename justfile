# Bare `just` lists recipes rather than running the first one.
default:
    @just --list

# Download the best-scoring eval gif per DMC task from wandb into media/.
mk-dmc-gifs:
    python scripts/pull-best-gifs.py

# Short run for interactive debugging; extra hydra overrides pass through.
debug *ARGS:
    PYTHONBREAKPOINT=ipdb.set_trace \
    WANDB_MODE=disabled MUJOCO_GL=egl \
    ipython train.py \
        env=dmc_proprio env.task=dmc_cartpole_balance \
        env.env_num=2 \
        model.rl=causal-a2c \
        model.compile=False \
        batch_size=4 batch_length=16 buffer.max_size=10000 \
        trainer.steps=2000 \
        trainer.eval_episode_num=0 \
        logdir=logdir/debug {{ARGS}}
