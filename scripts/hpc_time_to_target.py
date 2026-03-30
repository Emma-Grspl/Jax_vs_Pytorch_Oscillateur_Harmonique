"""Forward Jean Zay CLI arguments to the time-to-target runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Parse HPC wrapper arguments and launch the time-to-target subprocess."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--pytorch-repeats", type=int, default=None)
    parser.add_argument("--jax-repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--jax-epochs-multiplier", type=float, default=None)
    parser.add_argument("--jax-max-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--objective", choices=["physics_only", "physics_plus_data"], default=None)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--lambda-data", type=float, default=None)
    args = parser.parse_args()

    tag = args.tag or os.getenv("SLURM_JOB_ID") or args.device
    results_subdir = f"hpc_time_to_target_{tag}"

    command = [sys.executable, str(ROOT / "scripts" / "time_to_target.py")]
    command += ["--device", args.device, "--tag", tag, "--results-subdir", results_subdir]
    if args.pytorch_repeats is not None:
        command += ["--pytorch-repeats", str(args.pytorch_repeats)]
    if args.jax_repeats is not None:
        command += ["--jax-repeats", str(args.jax_repeats)]
    if args.warmup is not None:
        command += ["--warmup", str(args.warmup)]
    if args.jax_epochs_multiplier is not None:
        command += ["--jax-epochs-multiplier", str(args.jax_epochs_multiplier)]
    if args.jax_max_epochs is not None:
        command += ["--jax-max-epochs", str(args.jax_max_epochs)]
    if args.eval_every is not None:
        command += ["--eval-every", str(args.eval_every)]
    if args.objective is not None:
        command += ["--objective", args.objective]
    if args.n_data is not None:
        command += ["--n-data", str(args.n_data)]
    if args.lambda_data is not None:
        command += ["--lambda-data", str(args.lambda_data)]

    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
