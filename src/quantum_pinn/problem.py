from __future__ import annotations

import math
from typing import Any

import numpy as np


def create_grid(domain_min: float, domain_max: float, n_points: int) -> np.ndarray:
    return np.linspace(domain_min, domain_max, n_points, dtype=np.float32)


def potential(x: np.ndarray, mass: float, omega: float) -> np.ndarray:
    return 0.5 * mass * (omega**2) * x**2


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x

    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(2, n + 1):
        h_n = (2.0 * x * h_nm1) - (2.0 * (k - 1) * h_nm2)
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def analytical_energy(state_index: int, hbar: float, omega: float) -> float:
    return hbar * omega * (state_index + 0.5)


def analytical_wavefunction(
    x: np.ndarray,
    state_index: int,
    mass: float,
    omega: float,
    hbar: float,
) -> np.ndarray:
    xi = np.sqrt(mass * omega / hbar) * x
    prefactor = (mass * omega / (np.pi * hbar)) ** 0.25
    normalization = prefactor / np.sqrt((2.0**state_index) * math.factorial(state_index))
    return (
        normalization
        * hermite_polynomial(state_index, xi)
        * np.exp(-0.5 * xi**2)
    ).astype(np.float32)


def reference_solution(problem_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    x = create_grid(
        problem_cfg["domain_min"],
        problem_cfg["domain_max"],
        problem_cfg["n_eval"],
    )
    psi = analytical_wavefunction(
        x=x,
        state_index=problem_cfg["state_index"],
        mass=problem_cfg["mass"],
        omega=problem_cfg["omega"],
        hbar=problem_cfg["hbar"],
    )
    energy = analytical_energy(
        state_index=problem_cfg["state_index"],
        hbar=problem_cfg["hbar"],
        omega=problem_cfg["omega"],
    )
    return x, psi, energy


def supervised_reference_data(problem_cfg: dict[str, Any], n_points: int) -> tuple[np.ndarray, np.ndarray]:
    x = create_grid(
        problem_cfg["domain_min"],
        problem_cfg["domain_max"],
        n_points,
    )
    psi = analytical_wavefunction(
        x=x,
        state_index=problem_cfg["state_index"],
        mass=problem_cfg["mass"],
        omega=problem_cfg["omega"],
        hbar=problem_cfg["hbar"],
    )
    return x, psi
