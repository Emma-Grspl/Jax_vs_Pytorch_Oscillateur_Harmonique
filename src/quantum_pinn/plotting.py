from __future__ import annotations

import os
from pathlib import Path

cache_dir = Path(".matplotlib")
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(
    x: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x, reference, label="Analytical", color="black", linestyle="--")
    plt.plot(x, prediction, label="PINN", color="tab:blue", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("psi(x)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_history(history: dict[str, list[float]], path: str | Path) -> None:
    plt.figure(figsize=(10, 5))
    for key in ("total", "pde", "boundary", "norm", "center", "sign", "data"):
        if key in history and history[key]:
            plt.plot(history[key], label=key)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
