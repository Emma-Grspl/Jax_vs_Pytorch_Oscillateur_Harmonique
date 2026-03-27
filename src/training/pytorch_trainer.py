"""PyTorch training loop for the quantum harmonic oscillator PINN benchmark."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from src.data.problem import supervised_reference_data
from src.models.pytorch_model import QuantumPINN
from src.physics.schrodinger import torch_schrodinger_residual
from src.training.scheduler import build_scheduler


class PyTorchTrainer:
    """Train and evaluate the PyTorch PINN under shared benchmark settings."""

    def __init__(self, config: dict) -> None:
        """Store configuration, build the model, and initialize the optimizer."""
        self.config = config
        self.problem_cfg = config["problem"]
        self.train_cfg = config["training"]
        self.model_cfg = config["model"]

        self.device = self._resolve_device()
        self.model = QuantumPINN(
            hidden_layers=self.model_cfg["hidden_layers"],
            activation=self.model_cfg["activation"],
            energy_init=self.train_cfg["energy_init"],
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
        )
        self.n_supervision_points = int(self.train_cfg.get("n_supervision_points", 0))
        self.lambda_data = float(self.train_cfg.get("lambda_data", 0.0))
        self.use_supervision = self.lambda_data > 0.0 and self.n_supervision_points > 0

    def _resolve_device(self) -> torch.device:
        """Resolve the requested compute device from the training configuration."""
        requested = str(self.train_cfg.get("device", "auto")).lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
            return torch.device("cuda")
        if requested == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("Requested MPS device but it is unavailable.")
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported device setting: {requested}")

    def _set_learning_rate(self, value: float) -> None:
        """Apply the current learning rate to all optimizer parameter groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _loss_terms(
        self,
        x_collocation: torch.Tensor,
        x_boundary: torch.Tensor,
        x_supervision: torch.Tensor | None = None,
        psi_supervision: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss terms used by the PyTorch training objective."""
        residual = torch_schrodinger_residual(
            model=self.model,
            x=x_collocation,
            mass=self.problem_cfg["mass"],
            omega=self.problem_cfg["omega"],
            hbar=self.problem_cfg["hbar"],
        )
        loss_pde = torch.mean(residual**2)

        psi_boundary = self.model(x_boundary)
        loss_boundary = torch.mean(psi_boundary**2)

        psi_collocation = self.model(x_collocation)
        density = psi_collocation.squeeze(-1) ** 2
        norm = torch.trapezoid(density, x_collocation.squeeze(-1))
        loss_norm = (norm - 1.0) ** 2

        x_zero = torch.zeros((1, 1), dtype=torch.float32, device=self.device, requires_grad=True)
        psi_zero = self.model(x_zero)
        psi_x_zero = torch.autograd.grad(
            psi_zero,
            x_zero,
            torch.ones_like(psi_zero),
            create_graph=True,
        )[0]
        loss_center = torch.mean(psi_x_zero**2)
        loss_sign = torch.relu(-psi_zero).mean() ** 2
        if x_supervision is not None and psi_supervision is not None:
            psi_data = self.model(x_supervision).squeeze(-1)
            loss_data = torch.mean((psi_data - psi_supervision) ** 2)
        else:
            loss_data = torch.zeros((), dtype=torch.float32, device=self.device)

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

    def train(self) -> tuple[QuantumPINN, dict[str, list[float]], dict[str, float]]:
        """Run training, restore the best checkpoint, and return logs and timing."""
        domain_min = self.problem_cfg["domain_min"]
        domain_max = self.problem_cfg["domain_max"]
        n_collocation = self.problem_cfg["n_collocation"]

        x_collocation = torch.linspace(
            domain_min,
            domain_max,
            n_collocation,
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        x_boundary = torch.tensor(
            [[domain_min], [domain_max]],
            dtype=torch.float32,
            device=self.device,
        )
        x_supervision = None
        psi_supervision = None
        if self.use_supervision:
            x_supervision_np, psi_supervision_np = supervised_reference_data(self.problem_cfg, self.n_supervision_points)
            x_supervision = torch.tensor(x_supervision_np, dtype=torch.float32, device=self.device).view(-1, 1)
            psi_supervision = torch.tensor(psi_supervision_np, dtype=torch.float32, device=self.device)

        history = {key: [] for key in ("total", "pde", "boundary", "norm", "center", "sign", "data", "energy", "learning_rate")}
        best_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0
        scheduler = build_scheduler(self.train_cfg)
        start = time.perf_counter()

        for epoch in range(1, self.train_cfg["epochs"] + 1):
            current_lr = scheduler.current_lr
            self._set_learning_rate(current_lr)
            self.optimizer.zero_grad()
            losses = self._loss_terms(x_collocation, x_boundary, x_supervision, psi_supervision)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.train_cfg["grad_clip_norm"]))
            self.optimizer.step()

            for key in ("total", "pde", "boundary", "norm", "center", "sign", "data"):
                history[key].append(float(losses[key].detach().cpu().item()))
            history["energy"].append(float(self.model.energy.detach().cpu().item()))
            history["learning_rate"].append(current_lr)

            total_loss = history["total"][-1]
            if total_loss < best_loss - float(self.train_cfg["early_stopping_min_delta"]):
                best_loss = total_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            scheduler.step(total_loss)

            if epoch % self.train_cfg["log_every"] == 0:
                print(
                    f"[PyTorch] epoch={epoch:5d} "
                    f"loss={history['total'][-1]:.3e} "
                    f"E={history['energy'][-1]:.6f} "
                    f"lr={current_lr:.2e}"
                )

            if epochs_without_improvement >= int(self.train_cfg["early_stopping_patience"]):
                print(f"[PyTorch] early stop at epoch={epoch} best_epoch={best_epoch} best_loss={best_loss:.3e}")
                break

        self.model.load_state_dict(best_state)
        train_seconds = time.perf_counter() - start
        timing = {
            "compile_seconds": 0.0,
            "train_seconds": train_seconds,
            "total_seconds": train_seconds,
            "best_epoch": float(best_epoch),
            "device": str(self.device),
        }
        return self.model, history, timing

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a NumPy grid and return a NumPy prediction."""
        self.model.eval()
        tensor_x = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.model(tensor_x).cpu().numpy().squeeze(-1)
        return prediction.astype(np.float32)
