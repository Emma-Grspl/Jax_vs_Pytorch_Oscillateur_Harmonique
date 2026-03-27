"""Run repeated PyTorch and JAX benchmark experiments from one CLI entry point."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.runner import run_jax_once, run_pytorch_once
from src.utils.artifacts import save_benchmark_run_artifacts
from src.utils.benchmark import summarize_runs, write_csv, write_markdown_report
from src.utils.config import load_config, resolve_framework_config
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import align_sign
from src.data.problem import reference_solution
from src.utils.system_info import get_system_info


def _build_predictions(framework: str, config: dict, model_or_params):
    """Recompute aligned predictions for artifact export after a measured run."""
    x_eval, psi_exact, _ = reference_solution(config["problem"])
    if framework == "pytorch":
        model_or_params.eval()
        device = next(model_or_params.parameters()).device
        with torch.no_grad():
            prediction = model_or_params(
                torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32, device=device)
            ).detach().cpu().numpy().squeeze(-1)
        return x_eval, psi_exact, align_sign(prediction, psi_exact)

    if framework == "jax":
        import jax.numpy as jnp

        from src.models.jax_model import build_activation, mlp_forward

        activation = build_activation(config["model"]["activation"])
        prediction = mlp_forward(
            model_or_params["network"],
            jnp.asarray(x_eval.reshape(-1, 1), dtype=jnp.float32),
            activation,
        ).squeeze(-1)
        return x_eval, psi_exact, align_sign(np.asarray(prediction, dtype=np.float32), psi_exact)

    raise ValueError(f"Unsupported framework: {framework}")


def main() -> None:
    """Execute the configured benchmark and write summaries and per-run artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-collocation", type=int, default=None)
    parser.add_argument("--objective", choices=["physics_only", "physics_plus_data"], default=None)
    parser.add_argument("--n-data", type=int, default=None)
    parser.add_argument("--lambda-data", type=float, default=None)
    parser.add_argument("--frameworks", nargs="+", choices=["pytorch", "jax"], default=["pytorch", "jax"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--results-subdir", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / "config" / "quantum_oscillator.yaml")
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.n_collocation is not None:
        config["problem"]["n_collocation"] = args.n_collocation
    if args.objective is not None:
        config["training"]["objective"] = args.objective
    if args.n_data is not None:
        config["training"]["n_supervision_points"] = args.n_data
    if args.lambda_data is not None:
        config["training"]["lambda_data"] = args.lambda_data
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.tag is not None:
        config["experiment"]["run_tag"] = args.tag
    repeats = args.repeats if args.repeats is not None else int(config["benchmark"]["repeats"])
    warmup = args.warmup if args.warmup is not None else int(config["benchmark"]["warmup"])

    results_root = ROOT / config["experiment"]["artifact_dir"]
    if args.results_subdir:
        results_root = results_root / args.results_subdir
    results_root = ensure_dir(results_root)
    write_json(results_root / "system_info.json", get_system_info())
    rows: list[dict] = []

    frameworks = args.frameworks
    for framework in frameworks:
        framework_dir = ensure_dir(results_root / framework)
        run_rows: list[dict] = []
        framework_config = resolve_framework_config(config, framework)

        for warmup_idx in range(warmup):
            seed = int(framework_config["experiment"]["seed"]) + warmup_idx
            print(f"[{framework}] warmup {warmup_idx + 1}/{warmup} seed={seed}")
            if framework == "pytorch":
                run_pytorch_once(copy.deepcopy(framework_config), seed)
            else:
                run_jax_once(copy.deepcopy(framework_config), seed)

        for run_idx in range(repeats):
            seed = int(framework_config["experiment"]["seed"]) + warmup + run_idx
            print(f"[{framework}] measured run {run_idx + 1}/{repeats} seed={seed}")
            if framework == "pytorch":
                metrics, history, model_or_params = run_pytorch_once(copy.deepcopy(framework_config), seed)
            else:
                metrics, history, model_or_params = run_jax_once(copy.deepcopy(framework_config), seed)
            metrics["run_index"] = run_idx
            x_eval, psi_exact, psi_pred = _build_predictions(framework, framework_config, model_or_params)
            save_benchmark_run_artifacts(
                framework_dir=framework_dir,
                framework=framework,
                run_index=run_idx,
                seed=seed,
                metrics=metrics,
                history=history,
                model_or_params=model_or_params,
                config=framework_config,
                x_eval=x_eval,
                psi_exact=psi_exact,
                psi_pred=psi_pred,
            )
            run_rows.append(metrics)
            rows.append(metrics)

        write_csv(framework_dir / "benchmark_runs.csv", run_rows)
        write_json(framework_dir / "benchmark_runs.json", {"runs": run_rows})

    summary_rows = summarize_runs(rows)
    write_csv(results_root / "benchmark_summary.csv", summary_rows)
    write_json(results_root / "benchmark_summary.json", {"summary": summary_rows})
    write_markdown_report(results_root / "benchmark_report.md", rows, summary_rows)

    for row in summary_rows:
        print(
            f"{row['framework']:>7} | "
            f"compile={row['compile_seconds_mean']:.3f}s | "
            f"train={row['train_seconds_mean']:.3f}s +/- {row['train_seconds_std']:.3f} | "
            f"total={row['training_seconds_mean']:.3f}s | "
            f"L2={row['relative_l2_error_mean']:.3e} | "
            f"dE={row['absolute_energy_error_mean']:.3e}"
        )


if __name__ == "__main__":
    main()
