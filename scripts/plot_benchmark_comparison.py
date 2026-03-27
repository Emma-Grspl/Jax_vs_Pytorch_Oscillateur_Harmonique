"""Generate cross-experiment comparison figures for the final benchmark runs."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cache_dir = Path(tempfile.gettempdir()) / "qho_pinn_matplotlib"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    ("physics_only", ROOT / "outputs" / "quantum_oscillator" / "artifacts" / "hpc_jz_1405055"),
    ("physics_plus_data_32", ROOT / "outputs" / "quantum_oscillator" / "artifacts" / "hpc_jz_1407922"),
    ("physics_plus_data_64", ROOT / "outputs" / "quantum_oscillator" / "artifacts" / "hpc_jz_1408478"),
]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    """Convert CSV string values to numeric Python types when possible."""
    parsed = []
    for row in rows:
        parsed_row: dict[str, float | int | str] = {}
        for key, value in row.items():
            if value is None:
                parsed_row[key] = ""
                continue
            try:
                if value.isdigit():
                    parsed_row[key] = int(value)
                else:
                    parsed_row[key] = float(value)
            except ValueError:
                parsed_row[key] = value
        parsed.append(parsed_row)
    return parsed


def load_summary() -> dict[str, dict[str, dict[str, float | int | str]]]:
    """Load summary rows for the final benchmark experiments."""
    by_experiment: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for label, run_dir in RUNS:
        rows = parse_numeric_rows(load_csv_rows(run_dir / "benchmark_summary.csv"))
        by_experiment[label] = {str(row["framework"]): row for row in rows}
    return by_experiment


def plot_metric_bars(summary: dict[str, dict[str, dict[str, float | int | str]]], output_dir: Path) -> None:
    """Plot side-by-side bars for runtime and accuracy across experiments."""
    experiments = list(summary.keys())
    frameworks = ["pytorch", "jax"]
    colors = {"pytorch": "#1f77b4", "jax": "#e24a33"}
    metrics = [
        ("training_seconds_mean", "Total Time (s)"),
        ("relative_l2_error_mean", "Relative L2"),
        ("absolute_energy_error_mean", "Absolute Energy Error"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    x = np.arange(len(experiments))
    width = 0.35

    for ax, (metric_key, title) in zip(axes, metrics):
        for offset, framework in zip((-width / 2, width / 2), frameworks):
            values = [float(summary[experiment][framework][metric_key]) for experiment in experiments]
            bars = ax.bar(x + offset, values, width=width, label=framework, color=colors[framework], alpha=0.9)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3e}" if value < 0.1 else f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, experiments, rotation=12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Benchmark Comparison Across Training Regimes", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "1_comparison_bars.png", dpi=250)
    plt.close(fig)


def plot_speed_accuracy_tradeoff(summary: dict[str, dict[str, dict[str, float | int | str]]], output_dir: Path) -> None:
    """Plot the speed-versus-accuracy trade-off for all final experiments."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = {"pytorch": "#1f77b4", "jax": "#e24a33"}
    markers = {
        "physics_only": "o",
        "physics_plus_data_32": "s",
        "physics_plus_data_64": "^",
    }

    for experiment, framework_rows in summary.items():
        for framework, row in framework_rows.items():
            x_value = float(row["training_seconds_mean"])
            y_value = float(row["relative_l2_error_mean"])
            ax.scatter(x_value, y_value, s=90, color=colors[framework], marker=markers[experiment])
            ax.annotate(f"{framework} {experiment}", (x_value, y_value), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Total Time (s)")
    ax.set_ylabel("Relative L2")
    ax.set_title("Speed / Accuracy Trade-off")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "2_speed_accuracy_tradeoff.png", dpi=250)
    plt.close(fig)


def write_manifest(summary: dict[str, dict[str, dict[str, float | int | str]]], output_dir: Path) -> None:
    """Write the comparison metrics used to generate the final benchmark figures."""
    payload = {
        label: {
            framework: {
                "training_seconds_mean": float(row["training_seconds_mean"]),
                "relative_l2_error_mean": float(row["relative_l2_error_mean"]),
                "absolute_energy_error_mean": float(row["absolute_energy_error_mean"]),
            }
            for framework, row in framework_rows.items()
        }
        for label, framework_rows in summary.items()
    }
    with (output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    """Generate the final comparison figures across benchmark regimes."""
    figure_dir = ROOT / "results" / "quantum_oscillator" / "benchmark_comparison"
    analysis_dir = ROOT / "outputs" / "quantum_oscillator" / "analysis" / "benchmark_comparison"
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    plot_metric_bars(summary, figure_dir)
    plot_speed_accuracy_tradeoff(summary, figure_dir)
    write_manifest(summary, analysis_dir)
    print(f"Comparison figures written to: {figure_dir}")
    print(f"Comparison manifest written to: {analysis_dir}")


if __name__ == "__main__":
    main()
