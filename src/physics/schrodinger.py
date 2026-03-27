"""Residual and loss helpers for the stationary quantum harmonic oscillator."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import torch

from src.models.jax_model import mlp_forward


def torch_schrodinger_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    mass: float,
    omega: float,
    hbar: float,
) -> torch.Tensor:
    """Compute the stationary Schrödinger residual on a PyTorch collocation grid."""
    x = x.clone().detach().requires_grad_(True)
    psi = model(x)
    psi_x = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    psi_xx = torch.autograd.grad(psi_x, x, torch.ones_like(psi_x), create_graph=True)[0]
    potential = 0.5 * mass * (omega**2) * x**2
    kinetic = -(hbar**2 / (2.0 * mass)) * psi_xx
    return kinetic + potential * psi - model.energy * psi


def jax_trapezoidal_integral(y: jax.Array, x: jax.Array) -> jax.Array:
    """Integrate values over a one-dimensional grid with the trapezoidal rule."""
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * dx)


def jax_scalar_wavefunction(
    params,
    x_scalar: jax.Array,
    activation: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Evaluate the scalar wavefunction at one spatial coordinate in JAX."""
    x = jnp.array([[x_scalar]], dtype=jnp.float32)
    return mlp_forward(params["network"], x, activation).squeeze()


def jax_schrodinger_residual(
    params,
    x_collocation: jax.Array,
    mass: float,
    omega: float,
    hbar: float,
    activation: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Compute the JAX wavefunction values and Schrödinger residual."""
    psi_values = mlp_forward(params["network"], x_collocation, activation).squeeze(-1)
    d2psi = jax.vmap(
        jax.grad(jax.grad(lambda x_scalar: jax_scalar_wavefunction(params, x_scalar, activation)))
    )(x_collocation.squeeze(-1))
    potential = 0.5 * mass * (omega**2) * x_collocation.squeeze(-1) ** 2
    residual = -(hbar**2 / (2.0 * mass)) * d2psi + potential * psi_values - params["energy"] * psi_values
    return psi_values, residual
