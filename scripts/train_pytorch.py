"""Train a single PyTorch model and export the corresponding artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.analyse.plotting import plot_prediction, plot_training_history
from src.data.problem import reference_solution
from src.training.runner import run_pytorch_once
from src.utils.config import load_config, resolve_framework_config
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import align_sign


def main() -> None:
    """Run one PyTorch training job with plots and metrics export."""
    config_path = ROOT / "config" / "quantum_oscillator.yaml"
    config = resolve_framework_config(load_config(config_path), "pytorch")

    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    figure_dir = ensure_dir(ROOT / config["experiment"]["figure_dir"] / "single_runs" / "pytorch")
    artifact_dir = ensure_dir(ROOT / config["experiment"]["artifact_dir"] / "single_runs" / "pytorch")

    metrics, history, model = run_pytorch_once(config, seed)

    torch.save(model.state_dict(), artifact_dir / "model.pt")
    write_json(artifact_dir / "metrics.json", metrics)
    x_eval, psi_exact, _ = reference_solution(config["problem"])
    trainer_prediction = model(torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32)).detach().cpu().numpy().squeeze(-1)
    psi_pred = align_sign(trainer_prediction, psi_exact)
    plot_prediction(
        x=x_eval,
        reference=psi_exact,
        prediction=psi_pred,
        path=figure_dir / "prediction.png",
        title="Quantum Harmonic Oscillator Ground State - PyTorch PINN",
    )
    plot_training_history(history, figure_dir / "losses.png")

    print("PyTorch training completed.")
    print(metrics)


if __name__ == "__main__":
    main()
