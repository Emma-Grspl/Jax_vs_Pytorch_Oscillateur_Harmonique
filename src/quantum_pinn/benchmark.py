from __future__ import annotations

import csv
import platform
import statistics
from pathlib import Path
from typing import Any


def count_pytorch_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def count_jax_parameters(params: Any) -> int:
    leaves = []

    def _collect(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _collect(value)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                _collect(value)
            return
        if hasattr(node, "size"):
            leaves.append(int(node.size))

    _collect(params)
    return int(sum(leaves))


def build_run_metadata(framework: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "framework": framework,
        "experiment_name": config["experiment"]["name"],
        "run_tag": config["experiment"].get("run_tag", "default"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": str(config["training"].get("device", "auto")),
        "epochs": int(config["training"]["epochs"]),
        "objective": str(config["training"].get("objective", "physics_only")),
        "n_supervision_points": int(config["training"].get("n_supervision_points", 0)),
        "lambda_data": float(config["training"].get("lambda_data", 0.0)),
        "n_collocation": int(config["problem"]["n_collocation"]),
        "domain_min": float(config["problem"]["domain_min"]),
        "domain_max": float(config["problem"]["domain_max"]),
        "state_index": int(config["problem"]["state_index"]),
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    metrics = (
        "compile_seconds",
        "train_seconds",
        "training_seconds",
        "seconds_per_epoch",
        "relative_l2_error",
        "absolute_energy_error",
        "final_total_loss",
        "epochs_ran",
        "best_epoch",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["framework"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for framework, framework_rows in grouped.items():
        summary = {
            "framework": framework,
            "measured_runs": len(framework_rows),
            "epochs": framework_rows[0]["epochs"],
            "n_collocation": framework_rows[0]["n_collocation"],
            "trainable_parameters": framework_rows[0]["trainable_parameters"],
            "successful_runs_l2": sum(1 for row in framework_rows if bool(row.get("success_l2", False))),
            "successful_runs_energy": sum(1 for row in framework_rows if bool(row.get("success_energy", False))),
        }
        for metric in metrics:
            values = [float(row.get(metric, 0.0 if metric == "compile_seconds" else row.get("training_seconds", 0.0))) for row in framework_rows]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_median"] = statistics.median(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = min(values)
        summaries.append(summary)

    return summaries


def write_markdown_report(path: str | Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
        "| Framework | Runs | Compile Mean (s) | Train Mean (s) | Total Mean (s) | L2 Mean | L2 Median | dE Mean | L2 Success | dE Success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['framework']} | {row['measured_runs']} | "
            f"{row['compile_seconds_mean']:.3f} | {row['train_seconds_mean']:.3f} | {row['training_seconds_mean']:.3f} | "
            f"{row['relative_l2_error_mean']:.3e} | {row['relative_l2_error_median']:.3e} | "
            f"{row['absolute_energy_error_mean']:.3e} | "
            f"{row['successful_runs_l2']}/{row['measured_runs']} | "
            f"{row['successful_runs_energy']}/{row['measured_runs']} |"
        )
    lines.extend(
        [
            "",
            "## Measured Runs",
            "",
            "| Framework | Run | Seed | Epochs | Best Epoch | Compile (s) | Train (s) | Total (s) | L2 | dE | Final Loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        compile_seconds = float(row.get("compile_seconds", 0.0))
        train_seconds = float(row.get("train_seconds", row.get("training_seconds", 0.0)))
        lines.append(
            f"| {row['framework']} | {row['run_index']} | {row['seed']} | {row.get('epochs_ran', 0)} | {row.get('best_epoch', 0)} | "
            f"{compile_seconds:.3f} | {train_seconds:.3f} | {row['training_seconds']:.3f} | "
            f"{row['relative_l2_error']:.3e} | {row['absolute_energy_error']:.3e} | {row['final_total_loss']:.3e} |"
        )

    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
