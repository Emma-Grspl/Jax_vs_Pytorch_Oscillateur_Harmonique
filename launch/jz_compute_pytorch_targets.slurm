#!/bin/bash
#SBATCH --job-name=QHO_PTGT
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

mkdir -p slurm/log

echo "Job on ${SLURMD_NODENAME:-unknown-node}"
date
nvidia-smi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

PROJECT_DIR="${PROJECT_DIR:-/lustre/fswork/projects/rech/fdb/usv13rn/Jax_vs_Pytorch_Oscillateur_Harmonique}"
cd "$PROJECT_DIR"

RUN_TAG="${RUN_TAG:-jz_ptgt_${SLURM_JOB_ID:-manual}}"
OBJECTIVE="${OBJECTIVE:-physics_only}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-1}"
N_DATA="${N_DATA:-64}"
LAMBDA_DATA="${LAMBDA_DATA:-1.0}"

PYTHON_BIN="$(command -v python)"
echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available())"

COMMAND=(
  "${PYTHON_BIN}" -u scripts/hpc_compute_pytorch_targets.py
  --device cuda
  --tag "${RUN_TAG}"
  --repeats "${REPEATS}"
  --warmup "${WARMUP}"
  --objective "${OBJECTIVE}"
)

if [ "${OBJECTIVE}" = "physics_plus_data" ]; then
  COMMAND+=(--n-data "${N_DATA}" --lambda-data "${LAMBDA_DATA}")
fi

printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

srun "${COMMAND[@]}"
