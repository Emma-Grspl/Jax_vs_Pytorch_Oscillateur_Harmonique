from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cache_dir = ROOT / ".matplotlib"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_latest_run(base_dir: Path) -> Path:
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
    return min(rows, key=lambda row: float(row["relative_l2_error"]))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_weight_values(framework: str, run_dir: Path) -> np.ndarray:
    if framework == "pytorch":
        state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
        leaves = [tensor.detach().cpu().numpy().ravel() for tensor in state_dict.values()]
        return np.concatenate(leaves)

    params = np.load(run_dir / "params.npz")
    leaves = [params[key].ravel() for key in sorted(params.files)]
    return np.concatenate(leaves)


def plot_summary(summary_rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
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

    fig.suptitle("Physics-Only Benchmark Summary", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "1_summary_metrics.png", dpi=250)
    plt.close(fig)


def plot_run_metrics(run_rows: dict[str, list[dict[str, float | int | str]]], output_dir: Path) -> None:
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


def write_manifest(run_dir: Path, best_runs: dict[str, dict[str, float | int | str]], output_dir: Path) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific HPC run directory. Defaults to the latest complete hpc_jz_* run.",
    )
    args = parser.parse_args()

    results_base = ROOT / "results" / "quantum_oscillator" / "quantum_oscillator"
    run_dir = args.run_dir if args.run_dir is not None else discover_latest_run(results_base)
    output_dir = ROOT / "outputs" / "quantum_oscillator" / "physics_only_plots" / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = parse_numeric_rows(load_csv_rows(run_dir / "benchmark_summary.csv"))
    pytorch_rows = parse_numeric_rows(load_csv_rows(run_dir / "pytorch" / "benchmark_runs.csv"))
    jax_rows = parse_numeric_rows(load_csv_rows(run_dir / "jax" / "benchmark_runs.csv"))
    run_rows = {"pytorch": pytorch_rows, "jax": jax_rows}
    best_runs = {
        "pytorch": best_run(pytorch_rows),
        "jax": best_run(jax_rows),
    }

    plot_summary(summary_rows, output_dir)
    plot_run_metrics(run_rows, output_dir)
    plot_histories(best_runs, run_dir, output_dir)
    plot_reconstruction(best_runs, run_dir, output_dir)
    plot_error_heatmap(run_rows, run_dir, output_dir)
    plot_weight_histograms(best_runs, run_dir, output_dir)
    write_manifest(run_dir, best_runs, output_dir)

    print(f"Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
