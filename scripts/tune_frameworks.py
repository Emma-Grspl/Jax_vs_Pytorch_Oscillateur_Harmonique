"""Grid-search a small set of framework-specific training overrides."""

from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.runner import run_jax_once, run_pytorch_once
from src.utils.benchmark import summarize_runs, write_csv
from src.utils.config import deep_update, load_config, resolve_framework_config
from src.utils.io import ensure_dir, write_json


def candidate_sets() -> dict[str, list[dict]]:
    """Return the predefined tuning candidates for each framework."""
    return {
        "pytorch": [
            {"name": "decay_clip1", "training": {"learning_rate": 1e-3, "min_learning_rate": 5e-5, "grad_clip_norm": 1.0, "early_stopping_patience": 800}},
            {"name": "lower_lr", "training": {"learning_rate": 7e-4, "min_learning_rate": 5e-5, "grad_clip_norm": 1.0, "early_stopping_patience": 800}},
            {"name": "stronger_clip", "training": {"learning_rate": 1e-3, "min_learning_rate": 5e-5, "grad_clip_norm": 0.5, "early_stopping_patience": 800}},
        ],
        "jax": [
            {"name": "constant_lr", "training": {"learning_rate": 1e-3, "min_learning_rate": 1e-3, "grad_clip_norm": 10.0, "early_stopping_patience": 4000}},
            {"name": "gentle_decay", "training": {"learning_rate": 1e-3, "min_learning_rate": 2e-4, "grad_clip_norm": 10.0, "early_stopping_patience": 4000}},
            {"name": "lower_constant_lr", "training": {"learning_rate": 7e-4, "min_learning_rate": 7e-4, "grad_clip_norm": 10.0, "early_stopping_patience": 4000}},
        ],
    }


def ranking_key(summary_row: dict) -> tuple:
    """Rank tuning candidates by success count, accuracy, then runtime."""
    return (
        -int(summary_row["successful_runs_l2"]),
        -int(summary_row["successful_runs_energy"]),
        float(summary_row["relative_l2_error_median"]),
        float(summary_row["absolute_energy_error_median"]),
        float(summary_row["training_seconds_mean"]),
    )


def main() -> None:
    """Evaluate the candidate overrides and write tuning summaries."""
    base_config = load_config(ROOT / "config" / "quantum_oscillator.yaml")
    tuning_root = ensure_dir(ROOT / "outputs" / "quantum_oscillator" / "tuning")
    all_rows = []
    best_overrides = {}

    for framework, candidates in candidate_sets().items():
        framework_base = resolve_framework_config(base_config, framework)
        framework_base["training"]["epochs"] = 2500
        framework_base["training"]["log_every"] = 500

        candidate_summaries = []
        for candidate in candidates:
            candidate_config = copy.deepcopy(framework_base)
            deep_update(candidate_config, candidate)
            run_rows = []
            for run_index, seed in enumerate((101, 102, 103)):
                print(f"[tune:{framework}] candidate={candidate['name']} seed={seed}")
                if framework == "pytorch":
                    metrics, _, _ = run_pytorch_once(copy.deepcopy(candidate_config), seed)
                else:
                    metrics, _, _ = run_jax_once(copy.deepcopy(candidate_config), seed)
                metrics["run_index"] = run_index
                metrics["candidate"] = candidate["name"]
                run_rows.append(metrics)
                all_rows.append(metrics)

            summary = summarize_runs(run_rows)[0]
            summary["candidate"] = candidate["name"]
            candidate_summaries.append(summary)

        candidate_summaries.sort(key=ranking_key)
        best = candidate_summaries[0]
        best_name = best["candidate"]
        best_overrides[framework] = next(candidate for candidate in candidates if candidate["name"] == best_name)
        print(f"[tune:{framework}] best={best_name}")

        write_csv(tuning_root / f"{framework}_candidate_summary.csv", candidate_summaries)

    write_csv(tuning_root / "tuning_runs.csv", all_rows)
    write_json(tuning_root / "best_overrides.json", best_overrides)
    print(best_overrides)


if __name__ == "__main__":
    main()
