"""Small filesystem and JSON helpers used throughout the benchmark code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to disk with indentation."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
