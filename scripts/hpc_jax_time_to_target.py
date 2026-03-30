"""Forward Jean Zay CLI arguments to the JAX-only time-to-target runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Parse HPC wrapper arguments and launch the JAX-only subprocess."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--jax-epochs-multiplier", type=float, default=None)
    parser.add_argument("--jax-max-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    args = parser.parse_args()

    tag = args.tag or os.getenv("SLURM_JOB_ID") or args.device
    results_subdir = f"hpc_jax_time_to_target_{tag}"

    command = [sys.executable, str(ROOT / "scripts" / "jax_time_to_target_from_file.py")]
    command += [
        "--device",
        args.device,
        "--tag",
        tag,
        "--results-subdir",
        results_subdir,
        "--targets-file",
        args.targets_file,
    ]
    if args.repeats is not None:
        command += ["--repeats", str(args.repeats)]
    if args.warmup is not None:
        command += ["--warmup", str(args.warmup)]
    if args.jax_epochs_multiplier is not None:
        command += ["--jax-epochs-multiplier", str(args.jax_epochs_multiplier)]
    if args.jax_max_epochs is not None:
        command += ["--jax-max-epochs", str(args.jax_max_epochs)]
    if args.eval_every is not None:
        command += ["--eval-every", str(args.eval_every)]

    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
