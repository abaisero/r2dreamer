# Bare `just` lists recipes rather than running the first one.
default:
    @just --list

# Download the best-scoring eval gif per DMC task from wandb into media/.
mk-dmc-gifs:
    python scripts/pull-best-gifs.py
