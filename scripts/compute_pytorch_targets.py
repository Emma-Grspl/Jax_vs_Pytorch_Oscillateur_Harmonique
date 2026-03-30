"""Compute PyTorch reference targets for the separated time-to-target workflow."""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.runner import run_pytorch_once
from src.utils.benchmark import write_csv
from src.utils.config import load_config, resolve_framework_config
from src.utils.io import ensure_dir, write_json
from src.utils.system_info import get_system_info


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the PyTorch target computation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--objective", choices=["physics_only", "physics_plus_data"], default=None)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--lambda-data", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--tag", default="pytorch_targets")
    parser.add_argument("--results-subdir", default="time_to_target/pytorch_targets")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides to the base config."""
    updated = copy.deepcopy(config)
    if args.objective is not None:
        updated["training"]["objective"] = args.objective
    if args.n_data is not None:
        updated["training"]["n_supervision_points"] = args.n_data
    if args.lambda_data is not None:
        updated["training"]["lambda_data"] = args.lambda_data
    if args.device is not None:
        updated["training"]["device"] = args.device
    if args.tag is not None:
        updated["experiment"]["run_tag"] = args.tag
    return updated


def build_targets(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the median PyTorch error targets from measured runs."""
    return {
        "relative_l2_error_target": float(statistics.median(row["relative_l2_error"] for row in rows)),
        "absolute_energy_error_target": float(statistics.median(row["absolute_energy_error"] for row in rows)),
    }


def main() -> None:
    """Run PyTorch references only and export the target file for JAX."""
    args = parse_args()
    base_config = apply_overrides(load_config(ROOT / "config" / "quantum_oscillator.yaml"), args)
    config = resolve_framework_config(base_config, "pytorch")

    results_root = ensure_dir(ROOT / base_config["experiment"]["artifact_dir"] / args.results_subdir)
    write_json(results_root / "system_info.json", get_system_info())

    rows: list[dict[str, Any]] = []
    for warmup_idx in range(args.warmup):
        seed = int(config["experiment"]["seed"]) + warmup_idx
        print(f"[pytorch] warmup {warmup_idx + 1}/{args.warmup} seed={seed}")
        run_pytorch_once(copy.deepcopy(config), seed)

    for run_idx in range(args.repeats):
        seed = int(config["experiment"]["seed"]) + args.warmup + run_idx
        print(f"[pytorch] reference run {run_idx + 1}/{args.repeats} seed={seed}")
        metrics, _, _ = run_pytorch_once(copy.deepcopy(config), seed)
        metrics["run_index"] = run_idx
        rows.append(metrics)

    targets = build_targets(rows)
    payload = {
        "targets": targets,
        "reference_runs": rows,
        "objective": config["training"]["objective"],
        "n_supervision_points": int(config["training"].get("n_supervision_points", 0)),
        "lambda_data": float(config["training"].get("lambda_data", 0.0)),
        "tag": config["experiment"].get("run_tag", args.tag),
    }

    write_csv(results_root / "pytorch_reference_runs.csv", rows)
    write_json(results_root / "pytorch_targets.json", payload)

    print(
        f"PyTorch targets | L2={targets['relative_l2_error_target']:.3e} | "
        f"dE={targets['absolute_energy_error_target']:.3e}"
    )
    print(f"Target file written to {results_root / 'pytorch_targets.json'}")


if __name__ == "__main__":
    main()
