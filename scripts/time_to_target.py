"""Measure how long JAX needs to reach the PyTorch benchmark accuracy targets."""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.problem import reference_solution
from src.training.jax_trainer import JAXTrainer
from src.training.runner import run_pytorch_once
from src.utils.benchmark import build_run_metadata, count_jax_parameters, write_csv
from src.utils.config import load_config, resolve_framework_config
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import absolute_energy_error, align_sign, relative_l2_error
from src.utils.system_info import get_system_info


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the time-to-target experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-repeats", type=int, default=5)
    parser.add_argument("--jax-repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--jax-epochs-multiplier", type=float, default=5.0)
    parser.add_argument("--jax-max-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--objective", choices=["physics_only", "physics_plus_data"], default=None)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--lambda-data", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--tag", default="time_to_target")
    parser.add_argument("--results-subdir", default="time_to_target")
    return parser.parse_args()


def apply_common_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply objective-level overrides shared by both frameworks."""
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


def build_targets(pytorch_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Build the median PyTorch error targets used for the JAX runs."""
    return {
        "relative_l2_error_target": float(statistics.median(row["relative_l2_error"] for row in pytorch_rows)),
        "absolute_energy_error_target": float(statistics.median(row["absolute_energy_error"] for row in pytorch_rows)),
    }


def summarize_jax_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the JAX time-to-target runs into a compact summary table."""
    if not rows:
        return []

    summary = {
        "framework": "jax",
        "measured_runs": len(rows),
        "successful_runs_l2": sum(1 for row in rows if bool(row["hit_l2_target"])),
        "successful_runs_energy": sum(1 for row in rows if bool(row["hit_energy_target"])),
        "successful_runs_both": sum(1 for row in rows if bool(row["hit_both_targets"])),
        "relative_l2_error_target": rows[0]["relative_l2_error_target"],
        "absolute_energy_error_target": rows[0]["absolute_energy_error_target"],
    }
    metrics = (
        "training_seconds",
        "relative_l2_error",
        "absolute_energy_error",
        "epochs_ran",
        "best_epoch",
        "time_to_l2_target",
        "time_to_energy_target",
        "time_to_both_targets",
        "epoch_to_l2_target",
        "epoch_to_energy_target",
        "epoch_to_both_targets",
    )
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        summary[f"{metric}_mean"] = float(statistics.fmean(values)) if values else float("nan")
        summary[f"{metric}_median"] = float(statistics.median(values)) if values else float("nan")
        summary[f"{metric}_min"] = float(min(values)) if values else float("nan")
    return [summary]


def write_markdown_report(
    path: Path,
    targets: dict[str, float],
    pytorch_rows: list[dict[str, Any]],
    jax_rows: list[dict[str, Any]],
    jax_summary: list[dict[str, Any]],
) -> None:
    """Write a compact Markdown report for the time-to-target experiment."""
    summary = jax_summary[0] if jax_summary else {}
    lines = [
        "# Time-to-Target Report",
        "",
        "## PyTorch Targets",
        "",
        f"- Relative L2 target: {targets['relative_l2_error_target']:.3e}",
        f"- Absolute energy error target: {targets['absolute_energy_error_target']:.3e}",
        "",
        "## PyTorch Reference Runs",
        "",
        "| Run | Seed | Total (s) | L2 | dE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pytorch_rows:
        lines.append(
            f"| {row['run_index']} | {row['seed']} | {row['training_seconds']:.3f} | "
            f"{row['relative_l2_error']:.3e} | {row['absolute_energy_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## JAX Time to Target",
            "",
            "| Run | Seed | Hit L2 | Hit dE | Hit Both | Time to L2 (s) | Time to dE (s) | Time to Both (s) | Final L2 | Final dE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in jax_rows:
        def display(value: float | None) -> str:
            return f"{value:.3f}" if value is not None else "NA"

        lines.append(
            f"| {row['run_index']} | {row['seed']} | {int(row['hit_l2_target'])} | {int(row['hit_energy_target'])} | "
            f"{int(row['hit_both_targets'])} | {display(row['time_to_l2_target'])} | {display(row['time_to_energy_target'])} | "
            f"{display(row['time_to_both_targets'])} | {row['relative_l2_error']:.3e} | {row['absolute_energy_error']:.3e} |"
        )
    if summary:
        lines.extend(
            [
                "",
                "## JAX Summary",
                "",
                f"- L2 successes: {summary['successful_runs_l2']}/{summary['measured_runs']}",
                f"- dE successes: {summary['successful_runs_energy']}/{summary['measured_runs']}",
                f"- Both-target successes: {summary['successful_runs_both']}/{summary['measured_runs']}",
                f"- Median time to both targets: {summary['time_to_both_targets_median']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reference_pytorch(config: dict[str, Any], warmup: int, repeats: int) -> list[dict[str, Any]]:
    """Run the unchanged PyTorch benchmark to define the reference targets."""
    rows: list[dict[str, Any]] = []
    for warmup_idx in range(warmup):
        seed = int(config["experiment"]["seed"]) + warmup_idx
        print(f"[pytorch] warmup {warmup_idx + 1}/{warmup} seed={seed}")
        run_pytorch_once(copy.deepcopy(config), seed)

    for run_idx in range(repeats):
        seed = int(config["experiment"]["seed"]) + warmup + run_idx
        print(f"[pytorch] reference run {run_idx + 1}/{repeats} seed={seed}")
        metrics, _, _ = run_pytorch_once(copy.deepcopy(config), seed)
        metrics["run_index"] = run_idx
        rows.append(metrics)
    return rows


def run_jax_time_to_target(
    config: dict[str, Any],
    warmup: int,
    repeats: int,
    eval_every: int,
    targets: dict[str, float],
) -> list[dict[str, Any]]:
    """Run JAX with target tracking and report the first hit times."""
    rows: list[dict[str, Any]] = []
    x_eval, psi_exact, energy_exact = reference_solution(config["problem"])

    for warmup_idx in range(warmup):
        seed = int(config["experiment"]["seed"]) + warmup_idx
        warmup_config = copy.deepcopy(config)
        warmup_config["experiment"]["seed"] = seed
        print(f"[jax] warmup {warmup_idx + 1}/{warmup} seed={seed}")
        JAXTrainer(warmup_config).train()

    for run_idx in range(repeats):
        seed = int(config["experiment"]["seed"]) + warmup + run_idx
        run_config = copy.deepcopy(config)
        run_config["experiment"]["seed"] = seed
        trainer = JAXTrainer(run_config)
        tracking: dict[str, Any] = {
            "time_to_l2_target": None,
            "time_to_energy_target": None,
            "time_to_both_targets": None,
            "epoch_to_l2_target": None,
            "epoch_to_energy_target": None,
            "epoch_to_both_targets": None,
        }

        def epoch_callback(epoch: int, elapsed_seconds: float) -> None:
            psi_pred = align_sign(trainer.predict(x_eval), psi_exact)
            l2_error = relative_l2_error(psi_pred, psi_exact)
            energy_error = absolute_energy_error(float(trainer.params["energy"]), energy_exact)
            hit_l2 = l2_error <= targets["relative_l2_error_target"]
            hit_energy = energy_error <= targets["absolute_energy_error_target"]
            hit_both = hit_l2 and hit_energy

            if hit_l2 and tracking["time_to_l2_target"] is None:
                tracking["time_to_l2_target"] = float(elapsed_seconds)
                tracking["epoch_to_l2_target"] = int(epoch)
            if hit_energy and tracking["time_to_energy_target"] is None:
                tracking["time_to_energy_target"] = float(elapsed_seconds)
                tracking["epoch_to_energy_target"] = int(epoch)
            if hit_both and tracking["time_to_both_targets"] is None:
                tracking["time_to_both_targets"] = float(elapsed_seconds)
                tracking["epoch_to_both_targets"] = int(epoch)

        print(f"[jax] measured run {run_idx + 1}/{repeats} seed={seed}")
        params, history, timing = trainer.train(epoch_callback=epoch_callback, callback_every=eval_every)
        trainer.params = params
        psi_pred = align_sign(trainer.predict(x_eval), psi_exact)
        predicted_energy = float(params["energy"])
        epochs_ran = len(history["total"])

        row = {
            "compile_seconds": timing["compile_seconds"],
            "train_seconds": timing["train_seconds"],
            "training_seconds": timing["total_seconds"],
            "seconds_per_epoch": timing["train_seconds"] / max(epochs_ran, 1),
            "relative_l2_error": relative_l2_error(psi_pred, psi_exact),
            "absolute_energy_error": absolute_energy_error(predicted_energy, energy_exact),
            "predicted_energy": predicted_energy,
            "reference_energy": energy_exact,
            "final_total_loss": history["total"][-1],
            "trainable_parameters": count_jax_parameters(params),
            "seed": seed,
            "epochs_ran": epochs_ran,
            "best_epoch": int(timing["best_epoch"]),
            "resolved_device": str(timing["device"]),
            "run_index": run_idx,
            "relative_l2_error_target": targets["relative_l2_error_target"],
            "absolute_energy_error_target": targets["absolute_energy_error_target"],
            "hit_l2_target": tracking["time_to_l2_target"] is not None,
            "hit_energy_target": tracking["time_to_energy_target"] is not None,
            "hit_both_targets": tracking["time_to_both_targets"] is not None,
            **tracking,
        }
        row.update(build_run_metadata("jax", run_config))
        rows.append(row)

    return rows


def main() -> None:
    """Run the PyTorch-to-JAX time-to-target experiment."""
    args = parse_args()
    base_config = apply_common_overrides(load_config(ROOT / "config" / "quantum_oscillator.yaml"), args)

    pytorch_config = resolve_framework_config(base_config, "pytorch")
    jax_config = resolve_framework_config(base_config, "jax")
    if args.jax_max_epochs is not None:
        jax_config["training"]["epochs"] = int(args.jax_max_epochs)
    else:
        jax_config["training"]["epochs"] = max(
            int(jax_config["training"]["epochs"]),
            int(round(float(jax_config["training"]["epochs"]) * float(args.jax_epochs_multiplier))),
        )

    results_root = ensure_dir(ROOT / base_config["experiment"]["artifact_dir"] / args.results_subdir)
    write_json(results_root / "system_info.json", get_system_info())
    write_json(
        results_root / "experiment_config.json",
        {
            "base_config": base_config,
            "pytorch_config": pytorch_config,
            "jax_config": jax_config,
            "cli_args": vars(args),
        },
    )

    pytorch_rows = run_reference_pytorch(pytorch_config, warmup=args.warmup, repeats=args.pytorch_repeats)
    targets = build_targets(pytorch_rows)
    for row in pytorch_rows:
        row.update(targets)

    jax_rows = run_jax_time_to_target(
        jax_config,
        warmup=args.warmup,
        repeats=args.jax_repeats,
        eval_every=max(args.eval_every, 1),
        targets=targets,
    )
    jax_summary = summarize_jax_rows(jax_rows)

    write_csv(results_root / "pytorch_reference_runs.csv", pytorch_rows)
    write_json(results_root / "pytorch_reference_runs.json", {"runs": pytorch_rows, "targets": targets})
    write_csv(results_root / "jax_time_to_target_runs.csv", jax_rows)
    write_json(results_root / "jax_time_to_target_runs.json", {"runs": jax_rows})
    write_csv(results_root / "jax_time_to_target_summary.csv", jax_summary)
    write_json(results_root / "jax_time_to_target_summary.json", {"summary": jax_summary})
    write_markdown_report(results_root / "time_to_target_report.md", targets, pytorch_rows, jax_rows, jax_summary)

    print(
        f"PyTorch targets | L2={targets['relative_l2_error_target']:.3e} | "
        f"dE={targets['absolute_energy_error_target']:.3e}"
    )
    if jax_summary:
        summary = jax_summary[0]
        print(
            f"JAX summary | both-target hits={summary['successful_runs_both']}/{summary['measured_runs']} | "
            f"median time-to-both={summary['time_to_both_targets_median']}"
        )


if __name__ == "__main__":
    main()
