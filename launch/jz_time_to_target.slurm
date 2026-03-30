#!/bin/bash
#SBATCH --job-name=QHO_T2T
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
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
# sbatch launch/jz_time_to_target.slurm
if [ -n "${JAX_ENV_PATH:-}" ]; then
  source "${JAX_ENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

RUN_TAG="${RUN_TAG:-jz_t2t_${SLURM_JOB_ID:-manual}}"
OBJECTIVE="${OBJECTIVE:-physics_only}"
PYTORCH_REPEATS="${PYTORCH_REPEATS:-5}"
JAX_REPEATS="${JAX_REPEATS:-5}"
WARMUP="${WARMUP:-1}"
JAX_EPOCHS_MULTIPLIER="${JAX_EPOCHS_MULTIPLIER:-5}"
EVAL_EVERY="${EVAL_EVERY:-10}"
N_DATA="${N_DATA:-64}"
LAMBDA_DATA="${LAMBDA_DATA:-1.0}"

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

COMMAND=(
  "${PYTHON_BIN}" -u scripts/hpc_time_to_target.py
  --device cuda
  --tag "${RUN_TAG}"
  --pytorch-repeats "${PYTORCH_REPEATS}"
  --jax-repeats "${JAX_REPEATS}"
  --warmup "${WARMUP}"
  --jax-epochs-multiplier "${JAX_EPOCHS_MULTIPLIER}"
  --eval-every "${EVAL_EVERY}"
  --objective "${OBJECTIVE}"
)

if [ "${OBJECTIVE}" = "physics_plus_data" ]; then
  COMMAND+=(--n-data "${N_DATA}" --lambda-data "${LAMBDA_DATA}")
fi

printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

srun "${COMMAND[@]}"
