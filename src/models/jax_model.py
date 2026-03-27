"""JAX model definition for the quantum harmonic oscillator PINN."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def build_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the configured JAX activation function."""
    activations = {
        "tanh": jnp.tanh,
        "silu": jax.nn.silu,
        "gelu": jax.nn.gelu,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def init_mlp(layer_sizes: list[int], key: jax.Array) -> list[dict[str, jax.Array]]:
    """Initialize an MLP parameter tree with Xavier-style uniform weights."""
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for in_dim, out_dim, layer_key in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        weight_key, _ = jax.random.split(layer_key)
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        weights = jax.random.uniform(
            weight_key,
            shape=(in_dim, out_dim),
            minval=-limit,
            maxval=limit,
        )
        bias = jnp.zeros((out_dim,))
        params.append({"w": weights, "b": bias})
    return params


def mlp_forward(
    params: list[dict[str, jax.Array]],
    x: jax.Array,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
) -> jax.Array:
    """Run a forward pass through the MLP parameter tree."""
    h = x
    for layer in params[:-1]:
        h = activation(h @ layer["w"] + layer["b"])
    output_layer = params[-1]
    return h @ output_layer["w"] + output_layer["b"]
