"""Helpers for exporting benchmark artifacts for later analysis and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.io import ensure_dir, write_json


def _to_jsonable(value: Any) -> Any:
    """Convert nested values to JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_benchmark_run_artifacts(
    framework_dir: Path,
    framework: str,
    run_index: int,
    seed: int,
    metrics: dict[str, Any],
    history: dict[str, list[float]],
    model_or_params: Any,
    config: dict[str, Any],
    x_eval: np.ndarray,
    psi_exact: np.ndarray,
    psi_pred: np.ndarray,
) -> Path:
    """Persist metrics, history, predictions, config, and weights for one measured run."""
    run_dir = ensure_dir(framework_dir / f"run_{run_index:02d}_seed_{seed}")
    write_json(run_dir / "metrics.json", _to_jsonable(metrics))
    write_json(run_dir / "history.json", _to_jsonable(history))
    write_json(run_dir / "config.json", _to_jsonable(config))

    np.savez_compressed(
        run_dir / "predictions.npz",
        x_eval=np.asarray(x_eval, dtype=np.float32),
        psi_exact=np.asarray(psi_exact, dtype=np.float32),
        psi_pred=np.asarray(psi_pred, dtype=np.float32),
    )

    if framework == "pytorch":
        torch.save(model_or_params.state_dict(), run_dir / "model.pt")
    elif framework == "jax":
        flat_params = {}
        for key, value in _flatten_tree(model_or_params).items():
            flat_params[key] = np.asarray(value, dtype=np.float32)
        np.savez_compressed(run_dir / "params.npz", **flat_params)
        write_json(run_dir / "params_manifest.json", {"keys": sorted(flat_params)})
    else:
        raise ValueError(f"Unsupported framework for artifact export: {framework}")

    return run_dir


def _flatten_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested parameter tree into a path-keyed dictionary."""
    items: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            items.update(_flatten_tree(value, child_prefix))
        return items
    if isinstance(tree, list):
        for index, value in enumerate(tree):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            items.update(_flatten_tree(value, child_prefix))
        return items
    items[prefix or "value"] = tree
    return items
