# Contributing

## Scope

This repository benchmarks PyTorch and JAX on the same quantum PINN problem. Contributions should preserve that goal:

- keep the benchmark comparable across frameworks
- document any framework-specific deviation explicitly
- prefer reproducibility over ad hoc experiments

## Environment

Use a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository conventions

- Keep source code under `src/` following the current split:
  - `src/data`
  - `src/models`
  - `src/physics`
  - `src/training`
  - `src/utils`
  - `src/analyse`
- Keep raw benchmark artifacts in `outputs/quantum_oscillator/artifacts/`
- Keep non-figure analysis files in `outputs/quantum_oscillator/analysis/`
- Keep generated figures in `results/quantum_oscillator/`
- Keep only curated public figures in `assets/figures/`

## Code style

- Use English everywhere in code, comments, docstrings, and documentation
- Add concise module and function docstrings
- Prefer small, explicit functions over compact but opaque logic
- Avoid introducing notebook checkpoints, caches, or environment files into version control

## Benchmark changes

If you change the benchmark protocol, document:

- the exact configuration change
- whether the change applies to both frameworks or only one
- whether existing figures and conclusions must be regenerated

## Validation

Before opening a change, run:

```bash
python -m py_compile $(git ls-files '*.py')
```

If your change affects plots or benchmark outputs, regenerate the relevant figures and update `assets/figures/` only when the new figures are intended to represent the public-facing results.
