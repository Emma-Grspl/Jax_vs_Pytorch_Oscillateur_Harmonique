"""Forward benchmark CLI arguments from an HPC job to the main benchmark runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Parse HPC wrapper arguments and launch the benchmark subprocess."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-collocation", type=int, default=None)
    parser.add_argument("--objective", choices=["physics_only", "physics_plus_data"], default=None)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--lambda-data", type=float, default=None)
    parser.add_argument("--frameworks", nargs="+", choices=["pytorch", "jax"], default=["pytorch", "jax"])
    args = parser.parse_args()

    tag = args.tag or os.getenv("SLURM_JOB_ID") or args.device
    results_subdir = f"hpc_{tag}"

    command = [sys.executable, str(ROOT / "scripts" / "run_benchmark.py")]
    command += ["--device", args.device, "--tag", tag, "--results-subdir", results_subdir]
    command += ["--frameworks", *args.frameworks]
    if args.repeats is not None:
        command += ["--repeats", str(args.repeats)]
    if args.warmup is not None:
        command += ["--warmup", str(args.warmup)]
    if args.epochs is not None:
        command += ["--epochs", str(args.epochs)]
    if args.n_collocation is not None:
        command += ["--n-collocation", str(args.n_collocation)]
    if args.objective is not None:
        command += ["--objective", args.objective]
    if args.n_data is not None:
        command += ["--n-data", str(args.n_data)]
    if args.lambda_data is not None:
        command += ["--lambda-data", str(args.lambda_data)]

    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
