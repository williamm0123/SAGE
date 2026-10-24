#!/bin/bash
#SBATCH --job-name=sage-envcheck
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --output=logs/sage-envcheck_%j.out
# Submit from $PROJECT_ROOT:  mkdir -p logs && sbatch scripts/umhpc/check_env.sbatch
# Partition names come from DICC's "Basic HPC Usage" slides; confirm with `cluster-info` / `sinfo -s`.
# DICC QoS wall-time limits: short 1 h, normal 8 h, long 3 days.

set -euo pipefail
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/scripts/umhpc/env.sh"

echo "host=$(hostname) job=${SLURM_JOB_ID:-interactive}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python "$PROJECT_ROOT/scripts/umhpc/env_check.py"
