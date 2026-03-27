"""Generate per-experiment plots from saved benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
cache_dir = Path(tempfile.gettempdir()) / "qho_pinn_matplotlib"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.problem import supervised_reference_data


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_latest_run(base_dir: Path) -> Path:
    """Find the latest complete HPC run directory under the benchmark results root."""
    candidates = []
    for path in base_dir.glob("hpc_jz_*"):
        summary = path / "benchmark_summary.csv"
        pytorch_dir = path / "pytorch"
        jax_dir = path / "jax"
        if summary.exists() and pytorch_dir.exists() and jax_dir.exists():
            try:
                job_id = int(path.name.split("_")[-1])
            except ValueError:
                continue
            candidates.append((job_id, path))
    if not candidates:
        raise FileNotFoundError(f"No complete HPC benchmark run found under {base_dir}")
    return max(candidates, key=lambda item: item[0])[1]


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
                if value in ("True", "False"):
                    parsed_row[key] = value == "True"
                else:
                    parsed_row[key] = value
        parsed.append(parsed_row)
    return parsed


def best_run(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    """Select the best run according to relative L2 error."""
    return min(rows, key=lambda row: float(row["relative_l2_error"]))


def load_json(path: Path) -> dict:
    """Load a JSON file from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_weight_values(framework: str, run_dir: Path) -> np.ndarray:
    """Load and flatten the saved weights for the selected framework and run."""
    if framework == "pytorch":
        state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
        leaves = [tensor.detach().cpu().numpy().ravel() for tensor in state_dict.values()]
        return np.concatenate(leaves)

    params = np.load(run_dir / "params.npz")
    leaves = [params[key].ravel() for key in sorted(params.files)]
    return np.concatenate(leaves)


def objective_label(run_rows: list[dict[str, float | int | str]]) -> str:
    """Build a directory-friendly label for the current training regime."""
    objective = str(run_rows[0].get("objective", "physics_only"))
    n_data = int(run_rows[0].get("n_supervision_points", 0))
    if objective == "physics_plus_data":
        return f"physics_plus_data_{n_data}"
    return objective


def display_title(run_rows: list[dict[str, float | int | str]]) -> str:
    """Build a plot title for the current training regime."""
    objective = str(run_rows[0].get("objective", "physics_only"))
    n_data = int(run_rows[0].get("n_supervision_points", 0))
    if objective == "physics_plus_data":
        return f"Physics + Data Benchmark (n={n_data})"
    return "Physics-Only Benchmark"


def plot_summary(summary_rows: list[dict[str, float | int | str]], output_dir: Path, title: str) -> None:
    """Plot summary bars for runtime and accuracy metrics."""
    frameworks = [str(row["framework"]) for row in summary_rows]
    total_times = [float(row["training_seconds_mean"]) for row in summary_rows]
    l2_values = [float(row["relative_l2_error_mean"]) for row in summary_rows]
    energy_values = [float(row["absolute_energy_error_mean"]) for row in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    titles = ["Total Time (s)", "Relative L2", "Absolute Energy Error"]
    series = [total_times, l2_values, energy_values]
    colors = ["#1f77b4", "#e24a33"]

    for ax, title, values in zip(axes, titles, series):
        bars = ax.bar(frameworks, values, color=colors[: len(frameworks)], alpha=0.9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3e}" if value < 0.1 else f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"{title} Summary", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "1_summary_metrics.png", dpi=250)
    plt.close(fig)


def plot_run_metrics(run_rows: dict[str, list[dict[str, float | int | str]]], output_dir: Path) -> None:
    """Plot run-by-run L2 and energy errors for both frameworks."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    colors = {"pytorch": "#1f77b4", "jax": "#e24a33"}

    for framework, rows in run_rows.items():
        seeds = [int(row["seed"]) for row in rows]
        l2_values = [float(row["relative_l2_error"]) for row in rows]
        de_values = [float(row["absolute_energy_error"]) for row in rows]
        axes[0].plot(seeds, l2_values, marker="o", linewidth=2, label=framework, color=colors[framework])
        axes[1].plot(seeds, de_values, marker="o", linewidth=2, label=framework, color=colors[framework])

    axes[0].set_title("Run-by-Run Relative L2")
    axes[1].set_title("Run-by-Run Absolute Energy Error")
    for ax in axes:
        ax.set_xlabel("Seed")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Relative L2")
    axes[1].set_ylabel("Absolute Energy Error")
    fig.savefig(output_dir / "2_run_metrics.png", dpi=250)
    plt.close(fig)


def plot_histories(best_runs: dict[str, dict[str, float | int | str]], run_root: Path, output_dir: Path) -> None:
    """Plot the loss and energy trajectories of the best run in each framework."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    colors = {"pytorch": "#1f77b4", "jax": "#e24a33"}

    for framework, row in best_runs.items():
        run_dir = run_root / framework / f"run_{int(row['run_index']):02d}_seed_{int(row['seed'])}"
        history = load_json(run_dir / "history.json")
        epochs = np.arange(1, len(history["total"]) + 1)
        axes[0].plot(epochs, history["total"], label=f"{framework} total", color=colors[framework], linewidth=2)
        axes[1].plot(epochs, history["energy"], label=f"{framework} energy", color=colors[framework], linewidth=2)
        axes[1].axhline(0.5, color=colors[framework], linestyle="--", alpha=0.35)

    axes[0].set_yscale("log")
    axes[0].set_title("Best-Run Training Loss")
    axes[1].set_title("Best-Run Energy Trajectory")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Energy")
    axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(output_dir / "3_training_history.png", dpi=250)
    plt.close(fig)


def plot_reconstruction(best_runs: dict[str, dict[str, float | int | str]], run_root: Path, output_dir: Path) -> None:
    """Plot best-run reconstructions and absolute errors for both frameworks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"pytorch": "#1f77b4", "jax": "#e24a33"}

    for row_idx, framework in enumerate(("pytorch", "jax")):
        row = best_runs[framework]
        run_dir = run_root / framework / f"run_{int(row['run_index']):02d}_seed_{int(row['seed'])}"
        predictions = np.load(run_dir / "predictions.npz")
        x_eval = predictions["x_eval"]
        psi_exact = predictions["psi_exact"]
        psi_pred = predictions["psi_pred"]
        abs_error = np.abs(psi_pred - psi_exact)

        axes[row_idx, 0].plot(x_eval, psi_exact, color="black", linestyle="--", linewidth=2, label="Analytical")
        axes[row_idx, 0].plot(x_eval, psi_pred, color=colors[framework], linewidth=2, label=framework)
        axes[row_idx, 0].set_title(f"{framework.capitalize()} Reconstruction")
        axes[row_idx, 0].grid(True, alpha=0.25)
        axes[row_idx, 0].legend()

        axes[row_idx, 1].plot(x_eval, abs_error, color=colors[framework], linewidth=2)
        axes[row_idx, 1].fill_between(x_eval, 0.0, abs_error, color=colors[framework], alpha=0.2)
        axes[row_idx, 1].set_title(f"{framework.capitalize()} Absolute Error")
        axes[row_idx, 1].grid(True, alpha=0.25)

    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[0, 0].set_ylabel("psi(x)")
    axes[1, 0].set_ylabel("psi(x)")
    axes[0, 1].set_ylabel("|error|")
    axes[1, 1].set_ylabel("|error|")
    fig.savefig(output_dir / "4_reconstruction_and_error.png", dpi=250)
    plt.close(fig)


def plot_error_heatmap(run_rows: dict[str, list[dict[str, float | int | str]]], run_root: Path, output_dir: Path) -> None:
    """Plot a heatmap of absolute errors across runs and spatial coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    for ax, framework in zip(axes, ("pytorch", "jax")):
        error_grid = []
        x_eval = None
        rows = sorted(run_rows[framework], key=lambda row: int(row["run_index"]))
        for row in rows:
            run_dir = run_root / framework / f"run_{int(row['run_index']):02d}_seed_{int(row['seed'])}"
            predictions = np.load(run_dir / "predictions.npz")
            x_eval = predictions["x_eval"]
            error_grid.append(np.abs(predictions["psi_pred"] - predictions["psi_exact"]))
        grid = np.vstack(error_grid)
        image = ax.imshow(grid, aspect="auto", origin="lower", extent=[float(x_eval.min()), float(x_eval.max()), 0, len(rows) - 1], cmap="magma")
        ax.set_title(f"{framework.capitalize()} Error Heatmap")
        ax.set_xlabel("x")
        ax.set_ylabel("Run index")
        fig.colorbar(image, ax=ax, shrink=0.9)

    fig.savefig(output_dir / "5_error_heatmaps.png", dpi=250)
    plt.close(fig)


def plot_weight_histograms(best_runs: dict[str, dict[str, float | int | str]], run_root: Path, output_dir: Path) -> None:
    """Plot weight histograms for the best PyTorch and JAX checkpoints."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {"pytorch": "#1f77b4", "jax": "#e24a33"}

    for ax, framework in zip(axes, ("pytorch", "jax")):
        row = best_runs[framework]
        run_dir = run_root / framework / f"run_{int(row['run_index']):02d}_seed_{int(row['seed'])}"
        weights = load_weight_values(framework, run_dir)
        ax.hist(weights, bins=50, color=colors[framework], alpha=0.85)
        ax.set_title(f"{framework.capitalize()} Weight Distribution")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    fig.savefig(output_dir / "6_weight_histograms.png", dpi=250)
    plt.close(fig)


def plot_snapshot_overlay(best_runs: dict[str, dict[str, float | int | str]], run_root: Path, output_dir: Path, title: str) -> None:
    """Plot the analytical solution, data points, and best learned curves on one axis."""
    colors = {"pytorch": "#1f77b4", "jax": "#e24a33"}
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)

    reference_x = None
    reference_psi = None
    best_predictions: dict[str, np.ndarray] = {}
    supervision_x = None
    supervision_psi = None
    for framework, row in best_runs.items():
        run_dir = run_root / framework / f"run_{int(row['run_index']):02d}_seed_{int(row['seed'])}"
        predictions = np.load(run_dir / "predictions.npz")
        reference_x = predictions["x_eval"]
        reference_psi = predictions["psi_exact"]
        best_predictions[framework] = predictions["psi_pred"]
        if supervision_x is None:
            config = load_json(run_dir / "config.json")
            n_points = int(config["training"].get("n_supervision_points", 0))
            if n_points > 0:
                supervision_x, supervision_psi = supervised_reference_data(config["problem"], n_points)

    ax.plot(reference_x, reference_psi, color="black", linestyle="--", linewidth=2.2, label="Analytical")
    if supervision_x is not None and supervision_psi is not None:
        ax.scatter(supervision_x, supervision_psi, s=28, color="#ff4d4d", alpha=0.6, label="Supervision data")
    for framework in ("pytorch", "jax"):
        psi_pred = best_predictions[framework]
        label = f"{framework} (L2={float(best_runs[framework]['relative_l2_error']):.3e})"
        ax.plot(reference_x, psi_pred, color=colors[framework], linewidth=2, label=label)

    ax.set_title("Best Snapshot With Data")
    ax.set_xlabel("x")
    ax.set_ylabel("psi(x)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.suptitle(f"{title} Snapshot", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "7_snapshot_overlay.png", dpi=250)
    plt.close(fig)


def write_manifest(run_dir: Path, best_runs: dict[str, dict[str, float | int | str]], output_dir: Path) -> None:
    """Write a small manifest describing the plotted source runs."""
    payload = {
        "source_run_dir": str(run_dir),
        "selected_best_runs": {
            framework: {
                "run_index": int(row["run_index"]),
                "seed": int(row["seed"]),
                "relative_l2_error": float(row["relative_l2_error"]),
                "absolute_energy_error": float(row["absolute_energy_error"]),
            }
            for framework, row in best_runs.items()
        },
    }
    with (output_dir / "plot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    """Generate the full per-experiment figure set for one saved benchmark run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific HPC run directory. Defaults to the latest complete hpc_jz_* run.",
    )
    args = parser.parse_args()

    results_base = ROOT / "outputs" / "quantum_oscillator" / "artifacts"
    legacy_base = results_base / "quantum_oscillator"
    if legacy_base.exists():
        results_base = legacy_base
    run_dir = args.run_dir if args.run_dir is not None else discover_latest_run(results_base)
    summary_rows = parse_numeric_rows(load_csv_rows(run_dir / "benchmark_summary.csv"))
    pytorch_rows = parse_numeric_rows(load_csv_rows(run_dir / "pytorch" / "benchmark_runs.csv"))
    jax_rows = parse_numeric_rows(load_csv_rows(run_dir / "jax" / "benchmark_runs.csv"))
    figure_dir = ROOT / "results" / "quantum_oscillator" / "benchmark_plots" / objective_label(pytorch_rows) / run_dir.name
    analysis_dir = ROOT / "outputs" / "quantum_oscillator" / "analysis" / "benchmark_plots" / objective_label(pytorch_rows) / run_dir.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_rows = {"pytorch": pytorch_rows, "jax": jax_rows}
    best_runs = {
        "pytorch": best_run(pytorch_rows),
        "jax": best_run(jax_rows),
    }
    title = display_title(pytorch_rows)

    plot_summary(summary_rows, figure_dir, title)
    plot_run_metrics(run_rows, figure_dir)
    plot_histories(best_runs, run_dir, figure_dir)
    plot_reconstruction(best_runs, run_dir, figure_dir)
    plot_error_heatmap(run_rows, run_dir, figure_dir)
    plot_weight_histograms(best_runs, run_dir, figure_dir)
    plot_snapshot_overlay(best_runs, run_dir, figure_dir, title)
    write_manifest(run_dir, best_runs, analysis_dir)

    print(f"Figures written to: {figure_dir}")
    print(f"Analysis manifest written to: {analysis_dir}")


if __name__ == "__main__":
    main()
