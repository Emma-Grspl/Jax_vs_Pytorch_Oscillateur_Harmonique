"""Runtime environment inspection helpers for benchmark metadata."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any


def _run_command(command: list[str]) -> str | None:
    """Run a shell command and return its output when available."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    output = completed.stdout.strip() or completed.stderr.strip()
    return output or None


def get_system_info() -> dict[str, Any]:
    """Collect hardware and software metadata for a benchmark run."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "jax_platform_name": os.getenv("JAX_PLATFORM_NAME"),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["torch_cuda_devices"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                }
                for idx in range(torch.cuda.device_count())
            ]
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["torch_mps_available"] = True
    except ModuleNotFoundError:
        info["torch_version"] = None

    try:
        import jax

        info["jax_version"] = jax.__version__
        info["jax_devices"] = [
            {
                "id": device.id,
                "kind": device.device_kind,
                "platform": device.platform,
            }
            for device in jax.devices()
        ]
    except ModuleNotFoundError:
        info["jax_version"] = None

    info["nvidia_smi"] = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return info
