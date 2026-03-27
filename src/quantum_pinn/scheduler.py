from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlateauScheduler:
    current_lr: float
    min_lr: float
    factor: float
    patience: int
    threshold: float
    cooldown: int
    best_value: float = float("inf")
    bad_epochs: int = 0
    cooldown_counter: int = 0

    def step(self, metric: float) -> float:
        improved = metric < self.best_value - self.threshold
        if improved:
            self.best_value = metric
            self.bad_epochs = 0
            return self.current_lr

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_lr

        self.bad_epochs += 1
        if self.bad_epochs < self.patience:
            return self.current_lr

        next_lr = max(self.min_lr, self.current_lr * self.factor)
        self.bad_epochs = 0
        if next_lr < self.current_lr:
            self.current_lr = next_lr
            self.cooldown_counter = self.cooldown
        return self.current_lr


def build_scheduler(training_cfg: dict) -> PlateauScheduler:
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "plateau")).lower()
    if scheduler_type != "plateau":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return PlateauScheduler(
        current_lr=float(training_cfg["learning_rate"]),
        min_lr=float(training_cfg["min_learning_rate"]),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 200)),
        threshold=float(scheduler_cfg.get("threshold", training_cfg.get("early_stopping_min_delta", 1e-6))),
        cooldown=int(scheduler_cfg.get("cooldown", 0)),
    )
