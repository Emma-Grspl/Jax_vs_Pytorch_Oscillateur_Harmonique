"""Evaluation metrics for comparing analytical and learned solutions."""

from __future__ import annotations

import numpy as np


def align_sign(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip the prediction sign when needed to match the analytical branch."""
    if np.dot(prediction, reference) < 0.0:
        return -prediction
    return prediction


def relative_l2_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Compute the relative L2 error after sign alignment."""
    aligned = align_sign(prediction, reference)
    numerator = np.linalg.norm(aligned - reference)
    denominator = np.linalg.norm(reference)
    return float(numerator / denominator)


def absolute_energy_error(predicted_energy: float, reference_energy: float) -> float:
    """Compute the absolute error on the scalar energy estimate."""
    return float(abs(predicted_energy - reference_energy))
