"""Run the JAX-only time-to-target benchmark from a saved PyTorch target file."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.problem import reference_solution
from src.training.jax_trainer import JAXTrainer
from src.utils.benchmark import build_run_metadata, count_jax_parameters, write_csv
from src.utils.config import load_config, resolve_framework_config
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import absolute_energy_error, align_sign, relative_l2_error
from src.utils.system_info import get_system_info


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the JAX-only time-to-target benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-file", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--jax-epochs-multiplier", type=float, default=5.0)
    parser.add_argument("--jax-max-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--tag", default="jax_time_to_target")
    parser.add_argument("--results-subdir", default="time_to_target/jax_runs")
    return parser.parse_args()


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate JAX time-to-target rows into a compact summary."""
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


def write_markdown_report(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    """Write a compact Markdown report for the JAX-only run."""
    summary = summary_rows[0] if summary_rows else {}
    targets = payload["targets"]
    lines = [
        "# JAX Time-to-Target Report",
        "",
        "## PyTorch Targets",
        "",
        f"- Relative L2 target: {targets['relative_l2_error_target']:.3e}",
        f"- Absolute energy error target: {targets['absolute_energy_error_target']:.3e}",
        f"- Source file: {payload['source_file']}",
        "",
        "## JAX Runs",
        "",
        "| Run | Seed | Hit L2 | Hit dE | Hit Both | Time to L2 (s) | Time to dE (s) | Time to Both (s) | Final L2 | Final dE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
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
                "## Summary",
                "",
                f"- L2 successes: {summary['successful_runs_l2']}/{summary['measured_runs']}",
                f"- dE successes: {summary['successful_runs_energy']}/{summary['measured_runs']}",
                f"- Both-target successes: {summary['successful_runs_both']}/{summary['measured_runs']}",
                f"- Median time to both targets: {summary['time_to_both_targets_median']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run JAX-only time-to-target from a saved PyTorch target file."""
    args = parse_args()
    with Path(args.targets_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    base_config = load_config(ROOT / "config" / "quantum_oscillator.yaml")
    base_config["training"]["objective"] = payload["objective"]
    base_config["training"]["n_supervision_points"] = int(payload["n_supervision_points"])
    base_config["training"]["lambda_data"] = float(payload["lambda_data"])
    if args.device is not None:
        base_config["training"]["device"] = args.device
    if args.tag is not None:
        base_config["experiment"]["run_tag"] = args.tag

    config = resolve_framework_config(base_config, "jax")
    if args.jax_max_epochs is not None:
        config["training"]["epochs"] = int(args.jax_max_epochs)
    else:
        config["training"]["epochs"] = max(
            int(config["training"]["epochs"]),
            int(round(float(config["training"]["epochs"]) * float(args.jax_epochs_multiplier))),
        )

    results_root = ensure_dir(ROOT / config["experiment"]["artifact_dir"] / args.results_subdir)
    write_json(results_root / "system_info.json", get_system_info())
    write_json(
        results_root / "experiment_config.json",
        {
            "jax_config": config,
            "targets_payload": payload,
            "cli_args": vars(args),
        },
    )

    rows: list[dict[str, Any]] = []
    targets = payload["targets"]
    x_eval, psi_exact, energy_exact = reference_solution(config["problem"])

    for warmup_idx in range(args.warmup):
        seed = int(config["experiment"]["seed"]) + warmup_idx
        warmup_config = copy.deepcopy(config)
        warmup_config["experiment"]["seed"] = seed
        print(f"[jax] warmup {warmup_idx + 1}/{args.warmup} seed={seed}")
        JAXTrainer(warmup_config).train()

    for run_idx in range(args.repeats):
        seed = int(config["experiment"]["seed"]) + args.warmup + run_idx
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

        print(f"[jax] measured run {run_idx + 1}/{args.repeats} seed={seed}")
        params, history, timing = trainer.train(epoch_callback=epoch_callback, callback_every=max(args.eval_every, 1))
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

    summary_rows = summarize_rows(rows)
    write_csv(results_root / "jax_time_to_target_runs.csv", rows)
    write_json(results_root / "jax_time_to_target_runs.json", {"runs": rows})
    write_csv(results_root / "jax_time_to_target_summary.csv", summary_rows)
    write_json(results_root / "jax_time_to_target_summary.json", {"summary": summary_rows, "source_file": args.targets_file})
    report_payload = dict(payload)
    report_payload["source_file"] = str(Path(args.targets_file).resolve())
    write_markdown_report(results_root / "jax_time_to_target_report.md", report_payload, rows, summary_rows)

    if summary_rows:
        summary = summary_rows[0]
        print(
            f"JAX summary | both-target hits={summary['successful_runs_both']}/{summary['measured_runs']} | "
            f"median time-to-both={summary['time_to_both_targets_median']}"
        )


if __name__ == "__main__":
    main()
