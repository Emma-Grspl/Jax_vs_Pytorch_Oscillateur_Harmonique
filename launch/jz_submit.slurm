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
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1
source /lustre/fswork/projects/rech/fdb/usv13rn/venvs/jax_adr/bin/activate

mkdir -p slurm/log

echo "Job on ${SLURMD_NODENAME:-unknown-node}"
date
nvidia-smi

# Si tu as un environnement JAX dedie sur Jean Zay, active-le ici.
# Sans cette ligne, le job fonctionnera pour PyTorch mais JAX risque d'etre absent.
# source /path/to/your/jax-env/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

# Adapte PROJECT_DIR si le repo est clone ailleurs.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

# Run tag dedie au job Slurm pour separer clairement les sorties HPC.
RUN_TAG="jz_${SLURM_JOB_ID:-manual}"

if [ -x "./project1/bin/python" ]; then
  PYTHON_BIN="./project1/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

srun "${PYTHON_BIN}" -u scripts/hpc_run_benchmark.py \
  --device cuda \
  --frameworks pytorch jax \
  --repeats 5 \
  --warmup 1 \
  --tag "$RUN_TAG"
