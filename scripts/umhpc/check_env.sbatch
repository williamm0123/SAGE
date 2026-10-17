#!/bin/bash
#SBATCH --job-name=sage-envcheck
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Submit from $PROJECT_ROOT:  mkdir -p logs && sbatch scripts/umhpc/check_env.sbatch
# Partition names come from DICC's "Basic HPC Usage" slides; confirm with `cluster-info` / `sinfo -s`.
# If your account needs it, add:  #SBATCH --account=<your_account>   or   #SBATCH --qos=<qos>

set -euo pipefail
source "${PROJECT_ROOT:-$HOME/SAGE}/scripts/umhpc/env.sh"

echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python "$PROJECT_ROOT/scripts/umhpc/env_check.py"
