"""Shared helpers for executing one benchmark run in each framework."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.data.problem import reference_solution
from src.training.pytorch_trainer import PyTorchTrainer
from src.utils.benchmark import build_run_metadata, count_jax_parameters, count_pytorch_parameters
from src.utils.metrics import absolute_energy_error, align_sign, relative_l2_error


def run_pytorch_once(config: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, list[float]], Any]:
    """Execute one PyTorch run and return metrics, history, and the trained model."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_eval, psi_exact, energy_exact = reference_solution(config["problem"])
    trainer = PyTorchTrainer(config)
    model, history, timing = trainer.train()
    psi_pred = align_sign(trainer.predict(x_eval), psi_exact)
    epochs_ran = len(history["total"])

    metrics = {
        "compile_seconds": timing["compile_seconds"],
        "train_seconds": timing["train_seconds"],
        "training_seconds": timing["total_seconds"],
        "seconds_per_epoch": timing["train_seconds"] / max(epochs_ran, 1),
        "relative_l2_error": relative_l2_error(psi_pred, psi_exact),
        "predicted_energy": float(model.energy.detach().cpu().item()),
        "reference_energy": energy_exact,
        "final_total_loss": history["total"][-1],
        "trainable_parameters": count_pytorch_parameters(model),
        "absolute_energy_error": absolute_energy_error(
            float(model.energy.detach().cpu().item()),
            energy_exact,
        ),
    }
    metrics.update(build_run_metadata("pytorch", config))
    metrics["seed"] = seed
    metrics["epochs_ran"] = epochs_ran
    metrics["best_epoch"] = int(timing["best_epoch"])
    metrics["resolved_device"] = str(timing["device"])
    thresholds = config["benchmark"]
    metrics["success_l2"] = metrics["relative_l2_error"] <= thresholds["l2_success_threshold"]
    metrics["success_energy"] = metrics["absolute_energy_error"] <= thresholds["energy_success_threshold"]
    return metrics, history, model


def run_jax_once(config: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, list[float]], Any]:
    """Execute one JAX run and return metrics, history, and the trained parameters."""
    from src.training.jax_trainer import JAXTrainer

    run_config = dict(config)
    run_config["experiment"] = dict(config["experiment"])
    run_config["experiment"]["seed"] = seed

    x_eval, psi_exact, energy_exact = reference_solution(run_config["problem"])
    trainer = JAXTrainer(run_config)
    params, history, timing = trainer.train()
    psi_pred = align_sign(trainer.predict(x_eval), psi_exact)
    epochs_ran = len(history["total"])

    predicted_energy = float(params["energy"])
    metrics = {
        "compile_seconds": timing["compile_seconds"],
        "train_seconds": timing["train_seconds"],
        "training_seconds": timing["total_seconds"],
        "seconds_per_epoch": timing["train_seconds"] / max(epochs_ran, 1),
        "relative_l2_error": relative_l2_error(psi_pred, psi_exact),
        "predicted_energy": predicted_energy,
        "reference_energy": energy_exact,
        "final_total_loss": history["total"][-1],
        "trainable_parameters": count_jax_parameters(params),
        "absolute_energy_error": absolute_energy_error(predicted_energy, energy_exact),
    }
    metrics.update(build_run_metadata("jax", run_config))
    metrics["seed"] = seed
    metrics["epochs_ran"] = epochs_ran
    metrics["best_epoch"] = int(timing["best_epoch"])
    metrics["resolved_device"] = str(timing["device"])
    thresholds = run_config["benchmark"]
    metrics["success_l2"] = metrics["relative_l2_error"] <= thresholds["l2_success_threshold"]
    metrics["success_energy"] = metrics["absolute_energy_error"] <= thresholds["energy_success_threshold"]
    return metrics, history, params
