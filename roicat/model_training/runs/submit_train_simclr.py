#!/usr/bin/env python3
"""
Example submitter for ROInet / SimCLR training (per ``/job-dispatch`` skill).

This script is **stdlib-only** so it runs under any base Python. It dispatches
the worker ``run_train_simclr.sh`` either locally or via SLURM, depending on
the module-level ``BACKEND`` constant.

To adapt for a project:
  1. Edit ``DATASETS`` with one entry per (run_name, path_config, dir_save).
  2. Set ``PYTHON_EXEC`` to the conda env's python.
  3. Flip ``BACKEND`` to "local" for a smoke run.
  4. Always test with ``--dry-run`` first.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Callable, TypedDict


## ---- Backend / env config -------------------------------------------------

BACKEND: str = "slurm"   ## "local" or "slurm"

## Absolute path to the conda env python that has roicat installed.
PYTHON_EXEC: str = "/n/holylabs/kempner_rhakim_lab/Lab/rhakim/envs/roicat2/bin/python"


## ---- Dataset specs --------------------------------------------------------

class TrainRunSpec(TypedDict):
    name: str          ## short label, also shown in SLURM job-name / wandb
    path_config: str   ## absolute path to params JSON
    dir_save: str      ## absolute directory for checkpoints, ONNX, logger.txt


DATASETS: list[TrainRunSpec] = [
    ## Example — replace with project-specific specs.
    {
        "name": "example_run",
        "path_config": "/abs/path/to/params_train.json",
        "dir_save": "/abs/path/to/output/example_run/",
    },
]


## ---- Backend dispatch -----------------------------------------------------

def dispatch_local(worker: Path, worker_args: list[str], _name: str, _logs: Path) -> list[str]:
    return ["bash", str(worker), *worker_args]


def dispatch_slurm(worker: Path, worker_args: list[str], name: str, logs: Path) -> list[str]:
    return [
        "sbatch",
        f"--job-name={name}",
        f"--output={logs}/{name}_%j.out",
        f"--error={logs}/{name}_%j.err",
        str(worker), *worker_args,
    ]


DISPATCH: dict[str, Callable[[Path, list[str], str, Path], list[str]]] = {
    "local": dispatch_local,
    "slurm": dispatch_slurm,
}


## ---- Main -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print commands without dispatching")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    worker = here / "run_train_simclr.sh"
    analysis = (here / ".." / "train_simclr.py").resolve()
    logs = here / "logs"
    assert worker.exists(), f"Worker not found: {worker}"
    assert analysis.exists(), f"Analysis script not found: {analysis}"
    logs.mkdir(exist_ok=True)

    env = {**os.environ, "PYTHON_EXEC": PYTHON_EXEC}
    build_cmd = DISPATCH[BACKEND]

    for i, spec in enumerate(DATASETS):
        worker_args = [str(analysis), spec["name"], spec["path_config"], spec["dir_save"]]
        cmd = build_cmd(worker, worker_args, spec["name"], logs)
        print(f"[{i+1}/{len(DATASETS)}] ({BACKEND}) {' '.join(cmd)}")
        if not args.dry_run:
            Path(spec["dir_save"]).mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, stdin=subprocess.DEVNULL, env=env)


if __name__ == "__main__":
    main()
