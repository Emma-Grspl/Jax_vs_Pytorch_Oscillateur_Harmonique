"""Summarize the latest local benchmark outputs across frameworks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.benchmark import summarize_runs, write_csv
from src.utils.io import ensure_dir

ARTIFACT_ROOT = ROOT / "outputs" / "quantum_oscillator" / "artifacts"


def load_metrics(path: Path) -> dict:
    """Load a JSON metrics file from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_available_metrics() -> list[dict]:
    """Collect all available local benchmark rows for both frameworks."""
    rows = []
    for framework in ("pytorch", "jax"):
        summary_path = ARTIFACT_ROOT / framework / "benchmark_runs.json"
        metrics_path = ARTIFACT_ROOT / framework / "metrics.json"
        if summary_path.exists():
            payload = load_metrics(summary_path)
            rows.extend(payload.get("runs", []))
        elif metrics_path.exists():
            rows.append(load_metrics(metrics_path))
    return rows


def main() -> None:
    """Build and print a compact framework comparison report."""
    rows = collect_available_metrics()
    if not rows:
        raise SystemExit("No benchmark results found. Run at least one training script first.")

    ensure_dir(ARTIFACT_ROOT)
    summary_rows = summarize_runs(rows)
    summary_path = ARTIFACT_ROOT / "benchmark_summary.csv"
    write_csv(summary_path, summary_rows)

    print("Framework comparison")
    print(f"CSV summary: {summary_path}")
    print("")

    for row in summary_rows:
        print(
            f"{row['framework']:>7} | "
            f"time={row['training_seconds_mean']:.3f}s +/- {row['training_seconds_std']:.3f} | "
            f"s/epoch={row['seconds_per_epoch_mean']:.6f} | "
            f"L2={row['relative_l2_error_mean']:.3e} | "
            f"dE={row['absolute_energy_error_mean']:.3e} | "
            f"params={row['trainable_parameters']}"
        )

    by_framework = {row["framework"]: row for row in summary_rows}
    if "pytorch" in by_framework and "jax" in by_framework:
        pytorch_metrics = by_framework["pytorch"]
        jax_metrics = by_framework["jax"]
        speedup = pytorch_metrics["training_seconds_mean"] / jax_metrics["training_seconds_mean"]
        print("")
        print(f"JAX speedup vs PyTorch: {speedup:.3f}x")


if __name__ == "__main__":
    main()
