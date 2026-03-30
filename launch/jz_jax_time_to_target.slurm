#!/bin/bash
#SBATCH --job-name=QHO_JT2T
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
source /lustre/fswork/projects/rech/fdb/usv13rn/jax_cuda12_test/bin/activate

mkdir -p slurm/log

echo "Job on ${SLURMD_NODENAME:-unknown-node}"
date
nvidia-smi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PROJECT_DIR="${PROJECT_DIR:-/lustre/fswork/projects/rech/fdb/usv13rn/Jax_vs_Pytorch_Oscillateur_Harmonique}"
cd "$PROJECT_DIR"

RUN_TAG="${RUN_TAG:-jz_jt2t_${SLURM_JOB_ID:-manual}}"
TARGETS_FILE="${TARGETS_FILE:?Set TARGETS_FILE to the pytorch_targets.json path}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-1}"
JAX_EPOCHS_MULTIPLIER="${JAX_EPOCHS_MULTIPLIER:-5}"
EVAL_EVERY="${EVAL_EVERY:-10}"

PYTHON_BIN="$(command -v python)"
echo "Using Python: ${PYTHON_BIN}"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c "import jax, jaxlib; print('jax=', jax.__version__); print('jaxlib=', jaxlib.__version__); print('jax_devices=', jax.devices()); print('jax_backend=', jax.default_backend())"

COMMAND=(
  "${PYTHON_BIN}" -u scripts/hpc_jax_time_to_target.py
  --device cuda
  --tag "${RUN_TAG}"
  --targets-file "${TARGETS_FILE}"
  --repeats "${REPEATS}"
  --warmup "${WARMUP}"
  --jax-epochs-multiplier "${JAX_EPOCHS_MULTIPLIER}"
  --eval-every "${EVAL_EVERY}"
)

printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

srun "${COMMAND[@]}"
