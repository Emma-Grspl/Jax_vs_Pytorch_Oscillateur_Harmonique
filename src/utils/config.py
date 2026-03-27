"""Configuration loading and framework-specific override utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_framework_config(config: dict[str, Any], framework: str) -> dict[str, Any]:
    """Return a deep-copied config with the selected framework overrides applied."""
    resolved = copy.deepcopy(config)
    resolved["training"]["framework"] = framework
    framework_overrides = resolved.get("framework_overrides", {})
    if framework in framework_overrides:
        deep_update(resolved, framework_overrides[framework])
    return resolved
