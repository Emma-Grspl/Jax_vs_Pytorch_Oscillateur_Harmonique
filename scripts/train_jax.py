"""Train a single JAX model and export the corresponding artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.analyse.plotting import plot_prediction, plot_training_history
from src.data.problem import reference_solution
from src.models.jax_model import build_activation, mlp_forward
from src.training.runner import run_jax_once
from src.utils.config import load_config, resolve_framework_config
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import align_sign


def main() -> None:
    """Run one JAX training job with plots and metrics export."""
    try:
        from src.training.jax_trainer import JAXTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "JAX is not installed in the current environment. "
            "Install `jax` before running `scripts/train_jax.py`."
        ) from exc

    config_path = ROOT / "config" / "quantum_oscillator.yaml"
    config = resolve_framework_config(load_config(config_path), "jax")

    figure_dir = ensure_dir(ROOT / config["experiment"]["figure_dir"] / "single_runs" / "jax")
    artifact_dir = ensure_dir(ROOT / config["experiment"]["artifact_dir"] / "single_runs" / "jax")

    metrics, history, params = run_jax_once(config, int(config["experiment"]["seed"]))

    write_json(artifact_dir / "metrics.json", metrics)
    import jax.numpy as jnp

    x_eval, psi_exact, _ = reference_solution(config["problem"])
    activation = build_activation(config["model"]["activation"])
    psi_pred = mlp_forward(
        params["network"],
        jnp.asarray(x_eval.reshape(-1, 1), dtype=jnp.float32),
        activation,
    ).squeeze(-1)
    psi_pred = align_sign(np.asarray(psi_pred, dtype=np.float32), psi_exact)
    plot_prediction(
        x=x_eval,
        reference=psi_exact,
        prediction=psi_pred,
        path=figure_dir / "prediction.png",
        title="Quantum Harmonic Oscillator Ground State - JAX PINN",
    )
    plot_training_history(history, figure_dir / "losses.png")

    print("JAX training completed.")
    print(metrics)


if __name__ == "__main__":
    main()
