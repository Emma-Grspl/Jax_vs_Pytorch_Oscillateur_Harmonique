from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.quantum_pinn.jax.model import build_activation, init_mlp, mlp_forward


def adam_init(params):
    zeros_like = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0), "m": zeros_like, "v": zeros_like}


def adam_update(params, grads, state, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    step = state["step"] + 1
    m = jax.tree_util.tree_map(lambda m_prev, g: beta1 * m_prev + (1.0 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_prev, g: beta2 * v_prev + (1.0 - beta2) * (g**2), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**step), v)
    new_params = jax.tree_util.tree_map(
        lambda p, m_value, v_value: p - learning_rate * m_value / (jnp.sqrt(v_value) + eps),
        params,
        m_hat,
        v_hat,
    )
    return new_params, {"step": step, "m": m, "v": v}


def global_grad_clip(grads, max_norm: float):
    squared_norms = [jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(grads)]
    global_norm = jnp.sqrt(jnp.sum(jnp.stack(squared_norms)))
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-12))
    clipped = jax.tree_util.tree_map(lambda g: g * scale, grads)
    return clipped, global_norm


def trapezoidal_integral(y: jax.Array, x: jax.Array) -> jax.Array:
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * dx)


class JAXTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.problem_cfg = config["problem"]
        self.train_cfg = config["training"]
        self.model_cfg = config["model"]
        self.activation = build_activation(self.model_cfg["activation"])

        seed = int(config["experiment"]["seed"])
        layer_sizes = [1, *self.model_cfg["hidden_layers"], 1]
        self.params = {
            "network": init_mlp(layer_sizes, jax.random.PRNGKey(seed)),
            "energy": jnp.array(float(self.train_cfg["energy_init"]), dtype=jnp.float32),
        }
        self.optimizer_state = adam_init(self.params)

        self.x_collocation = jnp.linspace(
            self.problem_cfg["domain_min"],
            self.problem_cfg["domain_max"],
            self.problem_cfg["n_collocation"],
            dtype=jnp.float32,
        ).reshape(-1, 1)
        self.x_boundary = jnp.array(
            [[self.problem_cfg["domain_min"]], [self.problem_cfg["domain_max"]]],
            dtype=jnp.float32,
        )

    def _learning_rate_at_epoch(self, epoch: int) -> float:
        max_lr = float(self.train_cfg["learning_rate"])
        min_lr = float(self.train_cfg["min_learning_rate"])
        progress = min(max((epoch - 1) / max(self.train_cfg["epochs"] - 1, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine

    def _psi_scalar(self, params, x_scalar: jax.Array) -> jax.Array:
        x = jnp.array([[x_scalar]], dtype=jnp.float32)
        return mlp_forward(params["network"], x, self.activation).squeeze()

    def _loss_terms(self, params):
        mass = self.problem_cfg["mass"]
        omega = self.problem_cfg["omega"]
        hbar = self.problem_cfg["hbar"]

        psi_values = mlp_forward(params["network"], self.x_collocation, self.activation).squeeze(-1)
        d2psi = jax.vmap(jax.grad(jax.grad(lambda x_scalar: self._psi_scalar(params, x_scalar))))(self.x_collocation.squeeze(-1))
        potential = 0.5 * mass * (omega**2) * self.x_collocation.squeeze(-1) ** 2
        residual = -(hbar**2 / (2.0 * mass)) * d2psi + potential * psi_values - params["energy"] * psi_values
        loss_pde = jnp.mean(residual**2)

        psi_boundary = mlp_forward(params["network"], self.x_boundary, self.activation)
        loss_boundary = jnp.mean(psi_boundary**2)

        norm = trapezoidal_integral(psi_values**2, self.x_collocation.squeeze(-1))
        loss_norm = (norm - 1.0) ** 2

        psi_zero = self._psi_scalar(params, jnp.array(0.0, dtype=jnp.float32))
        psi_x_zero = jax.grad(lambda x_scalar: self._psi_scalar(params, x_scalar))(jnp.array(0.0, dtype=jnp.float32))
        loss_center = psi_x_zero**2
        loss_sign = jax.nn.relu(-psi_zero) ** 2

        total = (
            self.train_cfg["lambda_pde"] * loss_pde
            + self.train_cfg["lambda_boundary"] * loss_boundary
            + self.train_cfg["lambda_norm"] * loss_norm
            + self.train_cfg["lambda_center"] * loss_center
            + self.train_cfg["lambda_sign"] * loss_sign
        )
        return {
            "total": total,
            "pde": loss_pde,
            "boundary": loss_boundary,
            "norm": loss_norm,
            "center": loss_center,
            "sign": loss_sign,
        }

    def train(self):
        history = {key: [] for key in ("total", "pde", "boundary", "norm", "center", "sign", "energy", "learning_rate")}

        @jax.jit
        def train_step(params, optimizer_state, learning_rate):
            def objective(current_params):
                return self._loss_terms(current_params)["total"]

            grads = jax.grad(objective)(params)
            grads, _ = global_grad_clip(grads, jnp.asarray(self.train_cfg["grad_clip_norm"], dtype=jnp.float32))
            new_params, new_optimizer_state = adam_update(
                params=params,
                grads=grads,
                state=optimizer_state,
                learning_rate=learning_rate,
            )
            terms = self._loss_terms(new_params)
            return new_params, new_optimizer_state, terms

        best_loss = float("inf")
        best_epoch = 0
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), self.params)
        epochs_without_improvement = 0

        initial_lr = jnp.asarray(self._learning_rate_at_epoch(1), dtype=jnp.float32)
        compile_start = time.perf_counter()
        self.params, self.optimizer_state, terms = train_step(self.params, self.optimizer_state, initial_lr)
        compile_seconds = time.perf_counter() - compile_start
        for key in ("total", "pde", "boundary", "norm", "center", "sign"):
            history[key].append(float(terms[key]))
        history["energy"].append(float(self.params["energy"]))
        history["learning_rate"].append(float(initial_lr))
        best_loss = history["total"][-1]
        best_epoch = 1
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), self.params)

        start = time.perf_counter()
        for epoch in range(2, self.train_cfg["epochs"] + 1):
            current_lr = jnp.asarray(self._learning_rate_at_epoch(epoch), dtype=jnp.float32)
            self.params, self.optimizer_state, terms = train_step(self.params, self.optimizer_state, current_lr)
            for key in ("total", "pde", "boundary", "norm", "center", "sign"):
                history[key].append(float(terms[key]))
            history["energy"].append(float(self.params["energy"]))
            history["learning_rate"].append(float(current_lr))

            total_loss = history["total"][-1]
            if total_loss < best_loss - float(self.train_cfg["early_stopping_min_delta"]):
                best_loss = total_loss
                best_epoch = epoch
                best_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), self.params)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % self.train_cfg["log_every"] == 0:
                print(
                    f"[JAX] epoch={epoch:5d} "
                    f"loss={history['total'][-1]:.3e} "
                    f"E={history['energy'][-1]:.6f} "
                    f"lr={float(current_lr):.2e}"
                )

            if epochs_without_improvement >= int(self.train_cfg["early_stopping_patience"]):
                print(f"[JAX] early stop at epoch={epoch} best_epoch={best_epoch} best_loss={best_loss:.3e}")
                break

        self.params = best_params
        train_seconds = time.perf_counter() - start
        timing = {
            "compile_seconds": compile_seconds,
            "train_seconds": train_seconds,
            "total_seconds": compile_seconds + train_seconds,
            "best_epoch": float(best_epoch),
            "device": jax.devices()[0].platform if jax.devices() else "unknown",
        }
        return self.params, history, timing

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = mlp_forward(
            self.params["network"],
            jnp.asarray(x.reshape(-1, 1), dtype=jnp.float32),
            self.activation,
        ).squeeze(-1)
        return np.asarray(values, dtype=np.float32)
