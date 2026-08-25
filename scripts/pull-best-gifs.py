"""Download, per DMC task, a full-episode gif from the best-performing run.

Reads the wandb project, finds the highest ``episode/eval_score`` ever recorded
for each task across all seeds of ``METHOD``, then saves a ``train_video`` gif
from around that point in training to ``media/``.

``train_video`` rather than ``eval_video`` because the eval recording stops
after ``batch_length`` (64) steps, roughly an eighth of an episode, while
training videos cover a whole episode.  The trade-off is that they show the
exploring policy rather than the greedy one, and that they are written whenever
the first training env finishes an episode -- never exactly on an eval step --
so the closest one is used.
"""

import re
import shutil
import tempfile
from pathlib import Path

import wandb

PROJECT = "indylab/r2dreamer-causal-a2c"
METHOD = "dreamer"  # model.rep_loss to keep; None for every method
OUT = Path("media")
VIDEO = "train_video"


def eval_rows(run):
    """Eval history rows, best score first.

    Uses ``history`` rather than ``scan_history``: since wandb 0.28 the latter
    needs a ``_step`` column in the run schema, which these runs lack because
    ``train.py`` makes ``step`` the step metric.  Non-eval rows carry no score,
    so drop them.
    """
    rows = [
        (row["episode/eval_score"], row["_step"], row.get("step"))
        for row in run.history(keys=["episode/eval_score", "step"], pandas=False, samples=100000)
        if row.get("episode/eval_score") is not None and row.get("_step") is not None
    ]
    return sorted(rows, reverse=True)


def videos(run):
    """Video files of interest, keyed by the wandb step in their name."""
    found = {}
    for f in run.files():
        match = re.match(rf"media/videos/{VIDEO}_(\d+)_", f.name)
        if match:
            found[int(match.group(1))] = f
    return found


def main():
    OUT.mkdir(exist_ok=True)
    api = wandb.Api()

    best = {}
    for run in api.runs(PROJECT):
        task = run.config.get("env", {}).get("task") or run.group
        if METHOD is not None and run.config.get("model", {}).get("rep_loss") != METHOD:
            continue
        rows = eval_rows(run)
        if not task or not rows:
            print(f"skip {run.name}: no task or no eval history")
            continue
        if task not in best or rows[0][0] > best[task][1][0][0]:
            best[task] = (run, rows)
        print(f"{task:<32} {run.name:<40} best={rows[0][0]:7.1f}")

    for task, (run, rows) in sorted(best.items()):
        seed = run.config.get("seed")
        method = run.config.get("model", {}).get("rep_loss")
        score, wstep, env_step = rows[0]
        found = videos(run)
        if not found:
            print(f"no {VIDEO} found for {task} ({run.name})")
            continue
        nearest = min(found, key=lambda s: abs(s - wstep))
        step = int(env_step) if env_step is not None else wstep
        dest = OUT / f"{task}_seed{seed}_{method}_step{step}_score{score:.1f}.gif"
        if dest.exists():
            print(f"have {dest.name}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            found[nearest].download(root=tmp, replace=True)
            shutil.move(f"{tmp}/{found[nearest].name}", dest)
        print(f"wrote {dest.name}  ({VIDEO} @ wandb step {nearest}, best eval @ {wstep})")


if __name__ == "__main__":
    main()
