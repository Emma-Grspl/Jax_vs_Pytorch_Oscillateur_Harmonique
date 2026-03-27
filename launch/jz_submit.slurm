#!/bin/bash
#SBATCH --job-name=QHO_PINN_BENCH
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

mkdir -p slurm/log

echo "Job on ${SLURMD_NODENAME:-unknown-node}"
date
nvidia-smi

# Activate an additional JAX environment only if you provide it explicitly.
# Example:
# export JAX_ENV_PATH=/lustre/fswork/projects/rech/fdb/usv13rn/venvs/jax_pinn_gpu311
# sbatch launch/jz_submit.slurm
if [ -n "${JAX_ENV_PATH:-}" ]; then
  source "${JAX_ENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

# Override PROJECT_DIR if the repository is cloned elsewhere.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

# Use a Slurm-specific run tag to keep HPC outputs separate.
RUN_TAG="jz_${SLURM_JOB_ID:-manual}"

if [ -x "./project1/bin/python" ]; then
  PYTHON_BIN="./project1/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "Torch check:"
"${PYTHON_BIN}" -c "import torch; print(torch.__version__)"

echo "JAX check:"
"${PYTHON_BIN}" -c "import jax, jaxlib; print(jax.__version__); print(jaxlib.__version__); print(jax.devices())"

srun "${PYTHON_BIN}" -u scripts/hpc_run_benchmark.py \
  --device cuda \
  --frameworks pytorch jax \
  --repeats 3 \
  --warmup 1 \
  --epochs 4000 \
  --objective physics_plus_data \
  --n-data 64 \
  --lambda-data 1.0 \
  --tag "$RUN_TAG"
