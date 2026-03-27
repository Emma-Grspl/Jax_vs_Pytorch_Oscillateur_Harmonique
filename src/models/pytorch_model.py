"""PyTorch model definition for the quantum harmonic oscillator PINN."""

from __future__ import annotations

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    """Create a PyTorch activation module from its configuration name."""
    activations = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class QuantumPINN(nn.Module):
    """Feed-forward PINN with a jointly optimized scalar energy parameter."""

    def __init__(self, hidden_layers: list[int], activation: str, energy_init: float) -> None:
        """Initialize the network layers and the trainable energy parameter."""
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 1

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(build_activation(activation))
            in_features = width

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)
        self.energy = nn.Parameter(torch.tensor(float(energy_init), dtype=torch.float32))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply Xavier initialization to linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network wavefunction on the input coordinates."""
        return self.network(x)
