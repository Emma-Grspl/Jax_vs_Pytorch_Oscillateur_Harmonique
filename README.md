# Quantum Harmonic Oscillator PINN Benchmark

This repository provides a reproducible benchmark of **PyTorch** and **JAX** on the same physics-informed neural network (PINN) task: learning the one-dimensional ground-state wavefunction of the quantum harmonic oscillator.

The project now supports two training regimes:

- `physics_only`: the model is trained only from the stationary Schrödinger equation and auxiliary physical constraints.
- `physics_plus_data`: the same physical loss is augmented with analytical supervision points sampled from the closed-form solution.

The codebase is organized to keep the benchmark reproducible, inspectable, and easy to extend.

## Problem Statement

We solve the stationary Schrödinger equation for the one-dimensional harmonic oscillator:

$$
\left(-\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + \frac{1}{2}m\omega^2 x^2\right)\psi(x) = E \psi(x)
$$

The benchmark focuses on the ground state (`state_index = 0`) and compares how well PyTorch and JAX learn:

- the normalized wavefunction `psi(x)`
- the associated energy `E`

The analytical solution is available, which allows direct evaluation through:

- relative `L2` error on `psi(x)`
- absolute energy error `|E_pred - E_exact|`

## PINN Architecture

Both frameworks use the same conceptual model:

- a fully connected multilayer perceptron with input dimension `1` (`x`)
- configurable hidden layers from `config/quantum_oscillator.yaml`
- a scalar trainable energy parameter optimized jointly with the network weights

The training objective combines several terms:

- PDE residual loss from the stationary Schrödinger equation
- boundary loss at the domain edges
- normalization loss to enforce `||psi|| = 1`
- center derivative loss to encode ground-state symmetry
- sign loss to stabilize the ground-state branch
- optional supervised data loss in `physics_plus_data`

The optimization pipeline is also aligned across frameworks:

- `Adam`
- a shared plateau scheduler
- early stopping
- gradient clipping

## Repository Layout

### Source tree

- `src/data/problem.py`: analytical solution, evaluation grids, and supervision points
- `src/models/pytorch_model.py`: PyTorch PINN architecture
- `src/models/jax_model.py`: JAX PINN architecture
- `src/physics/schrodinger.py`: stationary Schrödinger residuals and physics operators
- `src/training/pytorch_trainer.py`: PyTorch training loop
- `src/training/jax_trainer.py`: JAX training loop
- `src/training/runner.py`: one-run execution helpers shared by scripts
- `src/training/scheduler.py`: shared plateau scheduler
- `src/utils/benchmark.py`: benchmark aggregation and reporting
- `src/utils/artifacts.py`: artifact export for histories, metrics, weights, and predictions
- `src/utils/metrics.py`: evaluation metrics and sign alignment
- `src/analyse/plotting.py`: generic plotting helpers used by the scripts

### Top-level directories

```text
.
├── assets/         curated public figures
├── config/         benchmark configuration
├── launch/         Slurm launch scripts
├── outputs/        raw artifacts and analysis manifests
├── results/        generated figures
├── scripts/        CLI entry points
└── src/            benchmark source code
```

### Configuration and launch

- `config/quantum_oscillator.yaml`: main benchmark configuration
- `launch/jz_submit.slurm`: Jean Zay submission script for benchmark runs

### Reproducible scripts

- `scripts/train_pytorch.py`: single PyTorch training run
- `scripts/train_jax.py`: single JAX training run
- `scripts/run_benchmark.py`: local benchmark driver
- `scripts/hpc_run_benchmark.py`: HPC wrapper around the benchmark driver
- `scripts/compare_frameworks.py`: summary view over available benchmark outputs
- `scripts/tune_frameworks.py`: small tuning utility for framework-specific experiments
- `scripts/plot_physics_only_results.py`: per-experiment plots from saved benchmark artifacts
- `scripts/plot_benchmark_comparison.py`: global comparison plots across final experiments

### Local artifacts

- `outputs/quantum_oscillator/artifacts/`: raw benchmark outputs, metrics, run histories, weights, and predictions
- `outputs/quantum_oscillator/analysis/`: JSON manifests and derived non-figure analysis files
- `results/quantum_oscillator/`: generated figures

These directories are intentionally kept out of version control. The repository should track **code, configs, launch scripts, and curated final figures**, not every raw run artifact.

### Curated assets

- `assets/figures/`: final repository-ready figures selected from the benchmark outputs

## Installation

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want GPU-backed JAX, install the appropriate JAX build for your machine or cluster.

## Reproducibility Workflow

The repository distinguishes three local artifact layers:

- `outputs/quantum_oscillator/artifacts/` for raw benchmark runs, weights, predictions, and metrics
- `outputs/quantum_oscillator/analysis/` for JSON manifests and non-figure derived files
- `results/quantum_oscillator/` for generated figures

For public-facing documentation, only selected figures should be copied to `assets/figures/`.

## How To Use The Repository

### Run a single framework locally

```bash
python scripts/train_pytorch.py
python scripts/train_jax.py
```

### Run a local benchmark

Physics-only:

```bash
python scripts/run_benchmark.py \
  --frameworks pytorch jax \
  --repeats 3 \
  --warmup 1 \
  --epochs 4000 \
  --objective physics_only \
  --results-subdir local_physics_only
```

Physics + analytical data:

```bash
python scripts/run_benchmark.py \
  --frameworks pytorch jax \
  --repeats 3 \
  --warmup 1 \
  --epochs 4000 \
  --objective physics_plus_data \
  --n-data 32 \
  --lambda-data 1.0 \
  --results-subdir local_physics_plus_data_32
```

### Run on Jean Zay

Edit `launch/jz_submit.slurm` if needed, then submit:

```bash
sbatch launch/jz_submit.slurm
```

### Generate plots from saved artifacts

Per experiment:

```bash
python scripts/plot_physics_only_results.py \
  --run-dir outputs/quantum_oscillator/artifacts/hpc_jz_<job_id>
```

Global comparison across the final experiments:

```bash
python scripts/plot_benchmark_comparison.py
```

## Key Figures

The most useful generated figures are:

- benchmark summaries for each experiment
- best-run reconstructions against the analytical solution
- `psi(x)` snapshots with supervision points for `physics_plus_data`
- run-by-run error plots
- global comparison bars across training regimes
- speed-versus-accuracy trade-off plots

Curated figures should live in `assets/figures/`.

## Main Findings

### Physics-only training

- PyTorch converges to a more accurate solution than JAX on this stationary PINN task.
- JAX remains significantly faster in total execution time.
- The shared scheduler improves fairness by aligning the optimization logic across frameworks.

### Physics + data training

- Adding analytical supervision points improves both frameworks substantially.
- JAX closes much of the precision gap once data is introduced.
- PyTorch remains slightly more accurate, while JAX keeps a clear time advantage.

### Effect of increasing the number of supervised points

- Moving from `n = 32` to `n = 64` gives a smaller incremental gain than the jump from `physics_only` to `physics_plus_data`.
- The ranking does not change: PyTorch is still more accurate, JAX is still faster.

## Conclusion

This repository supports a defensible benchmark story:

- **PyTorch** is the stronger framework for raw accuracy on this quantum PINN benchmark.
- **JAX** is the stronger framework for execution speed.
- Under pure physics supervision, the gap in accuracy is wider.
- Under mixed physics-and-data supervision, the gap narrows substantially and both frameworks become highly accurate.

That makes the project useful beyond a simple speed test. It shows that the framework ranking depends not only on the network architecture, but also on the training regime and the amount of analytical supervision.

## Contributing

See `CONTRIBUTING.md` for repository conventions, validation steps, and artifact policy.
