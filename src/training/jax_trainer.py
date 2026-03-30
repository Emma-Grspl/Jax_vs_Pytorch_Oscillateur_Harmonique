"""JAX training loop for the quantum harmonic oscillator PINN benchmark."""

from __future__ import annotations

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from src.data.problem import supervised_reference_data
from src.models.jax_model import build_activation, init_mlp, mlp_forward
from src.physics.schrodinger import jax_scalar_wavefunction, jax_schrodinger_residual, jax_trapezoidal_integral
from src.training.scheduler import build_scheduler


def adam_init(params):
    """Initialize Adam optimizer state for a nested JAX parameter tree."""
    zeros_like = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.array(0), "m": zeros_like, "v": zeros_like}


def adam_update(params, grads, state, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    """Apply one Adam update step to a nested JAX parameter tree."""
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
    """Clip gradients by global norm and return the clipped tree and norm."""
    squared_norms = [jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(grads)]
    global_norm = jnp.sqrt(jnp.sum(jnp.stack(squared_norms)))
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-12))
    clipped = jax.tree_util.tree_map(lambda g: g * scale, grads)
    return clipped, global_norm


class JAXTrainer:
    """Train and evaluate the JAX PINN under shared benchmark settings."""

    def __init__(self, config: dict) -> None:
        """Store configuration, initialize parameters, and build training grids."""
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
        self.n_supervision_points = int(self.train_cfg.get("n_supervision_points", 0))
        self.lambda_data = float(self.train_cfg.get("lambda_data", 0.0))
        self.use_supervision = self.lambda_data > 0.0 and self.n_supervision_points > 0

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
        if self.use_supervision:
            x_supervision_np, psi_supervision_np = supervised_reference_data(self.problem_cfg, self.n_supervision_points)
            self.x_supervision = jnp.asarray(x_supervision_np.reshape(-1, 1), dtype=jnp.float32)
            self.psi_supervision = jnp.asarray(psi_supervision_np, dtype=jnp.float32)
        else:
            self.x_supervision = None
            self.psi_supervision = None

    def _loss_terms(self, params):
        """Compute all loss terms used by the JAX training objective."""
        psi_values, residual = jax_schrodinger_residual(
            params=params,
            x_collocation=self.x_collocation,
            mass=self.problem_cfg["mass"],
            omega=self.problem_cfg["omega"],
            hbar=self.problem_cfg["hbar"],
            activation=self.activation,
        )
        loss_pde = jnp.mean(residual**2)

        psi_boundary = mlp_forward(params["network"], self.x_boundary, self.activation)
        loss_boundary = jnp.mean(psi_boundary**2)

        norm = jax_trapezoidal_integral(psi_values**2, self.x_collocation.squeeze(-1))
        loss_norm = (norm - 1.0) ** 2

        psi_zero = jax_scalar_wavefunction(params, jnp.array(0.0, dtype=jnp.float32), self.activation)
        psi_x_zero = jax.grad(lambda x_scalar: jax_scalar_wavefunction(params, x_scalar, self.activation))(jnp.array(0.0, dtype=jnp.float32))
        loss_center = psi_x_zero**2
        loss_sign = jax.nn.relu(-psi_zero) ** 2
        if self.use_supervision:
            psi_data = mlp_forward(params["network"], self.x_supervision, self.activation).squeeze(-1)
            loss_data = jnp.mean((psi_data - self.psi_supervision) ** 2)
        else:
            loss_data = jnp.array(0.0, dtype=jnp.float32)

        total = (
            self.train_cfg["lambda_pde"] * loss_pde
            + self.train_cfg["lambda_boundary"] * loss_boundary
            + self.train_cfg["lambda_norm"] * loss_norm
            + self.train_cfg["lambda_center"] * loss_center
            + self.train_cfg["lambda_sign"] * loss_sign
            + self.lambda_data * loss_data
        )
        return {
            "total": total,
            "pde": loss_pde,
            "boundary": loss_boundary,
            "norm": loss_norm,
            "center": loss_center,
            "sign": loss_sign,
            "data": loss_data,
        }

    def train(
        self,
        epoch_callback: Callable[[int, float], None] | None = None,
        callback_every: int = 1,
    ):
        """Run training, keep the best parameter tree, and return logs and timing."""
        history = {key: [] for key in ("total", "pde", "boundary", "norm", "center", "sign", "data", "energy", "learning_rate")}

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
        scheduler = build_scheduler(self.train_cfg)

        initial_lr = jnp.asarray(scheduler.current_lr, dtype=jnp.float32)
        compile_start = time.perf_counter()
        self.params, self.optimizer_state, terms = train_step(self.params, self.optimizer_state, initial_lr)
        compile_seconds = time.perf_counter() - compile_start
        for key in ("total", "pde", "boundary", "norm", "center", "sign", "data"):
            history[key].append(float(terms[key]))
        history["energy"].append(float(self.params["energy"]))
        history["learning_rate"].append(float(initial_lr))
        best_loss = history["total"][-1]
        best_epoch = 1
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), self.params)
        scheduler.step(best_loss)
        callback_overhead_seconds = 0.0
        last_callback_epoch = 0

        if epoch_callback is not None:
            callback_start = time.perf_counter()
            epoch_callback(1, compile_seconds)
            callback_overhead_seconds += time.perf_counter() - callback_start
            last_callback_epoch = 1

        start = time.perf_counter()
        final_epoch = 1
        for epoch in range(2, self.train_cfg["epochs"] + 1):
            final_epoch = epoch
            current_lr = jnp.asarray(scheduler.current_lr, dtype=jnp.float32)
            self.params, self.optimizer_state, terms = train_step(self.params, self.optimizer_state, current_lr)
            for key in ("total", "pde", "boundary", "norm", "center", "sign", "data"):
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
            scheduler.step(total_loss)

            if epoch_callback is not None and (epoch % max(callback_every, 1) == 0 or epoch == self.train_cfg["epochs"]):
                elapsed_seconds = compile_seconds + (time.perf_counter() - start - callback_overhead_seconds)
                callback_start = time.perf_counter()
                epoch_callback(epoch, elapsed_seconds)
                callback_overhead_seconds += time.perf_counter() - callback_start
                last_callback_epoch = epoch

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

        if epoch_callback is not None and final_epoch != last_callback_epoch:
            elapsed_seconds = compile_seconds + (time.perf_counter() - start - callback_overhead_seconds)
            callback_start = time.perf_counter()
            epoch_callback(final_epoch, elapsed_seconds)
            callback_overhead_seconds += time.perf_counter() - callback_start

        self.params = best_params
        train_seconds = time.perf_counter() - start - callback_overhead_seconds
        timing = {
            "compile_seconds": compile_seconds,
            "train_seconds": train_seconds,
            "total_seconds": compile_seconds + train_seconds,
            "best_epoch": float(best_epoch),
            "device": jax.devices()[0].platform if jax.devices() else "unknown",
        }
        return self.params, history, timing

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a NumPy grid and return a NumPy prediction."""
        values = mlp_forward(
            self.params["network"],
            jnp.asarray(x.reshape(-1, 1), dtype=jnp.float32),
            self.activation,
        ).squeeze(-1)
        return np.asarray(values, dtype=np.float32)
