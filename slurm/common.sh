# Shared setup sourced by every slurm/*.sbatch job.
set -euo pipefail
source "${PROJECT_ROOT:-$HOME/SAGE}/scripts/umhpc/env.sh"
cd "$PROJECT_ROOT"

export SAGE_DATA=${SAGE_DATA:-$DATA_ROOT/sage_v1}
export RESULTS=${RESULTS:-$PROJECT_ROOT/results}
export PRIOR_MIXED=${PRIOR_MIXED:-$PROJECT_ROOT/priors/prior_mixed.npz}
export PRIOR_SDV1=${PRIOR_SDV1:-$PROJECT_ROOT/priors/prior_sdv1.npz}

echo "job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-} host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
