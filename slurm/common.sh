# Shared setup sourced by every slurm/*.sbatch job.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/umhpc/env.sh"
cd "$PROJECT_ROOT"

export SAGE_DATA=${SAGE_DATA:-$DATA_ROOT/sage_v1}
export RESULTS=${RESULTS:-$PROJECT_ROOT/results}
export PRIOR_MIXED=${PRIOR_MIXED:-$PROJECT_ROOT/priors/prior_mixed.npz}
export PRIOR_SDV1=${PRIOR_SDV1:-$PROJECT_ROOT/priors/prior_sdv1.npz}
export GPU_SLOTS=${GPU_SLOTS:-1}
EXIT_INCOMPLETE=75            # run_attacks.py stopped before its time limit with work left (sage/experiment.py)

echo "job=${SLURM_JOB_ID:-local} name=${SLURM_JOB_NAME:-} array=${SLURM_ARRAY_TASK_ID:-} host=$(hostname) start=$(date -Is)" \
     "resubmission=${SAGE_RESUBMIT_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if [ -n "${SLURM_JOB_ID:-}" ] && command -v nvidia-smi >/dev/null; then
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ "${mem:-0}" -ge 70000 ] || echo "WARNING: GPU has ${mem} MiB; the protocol assumes an A100 80GB"
fi

# Submit the analysis once every primary / controls / transfer run is stored.
maybe_analyze() {
  [ "${SAGE_AUTO_ANALYZE:-1}" = 1 ] || return 0
  if python experiments/status.py --data "$SAGE_DATA" --results "$RESULTS" --detectors ${SAGE_DETECTORS:-ufd aeroblade} --quiet --require-complete; then
    echo "all experiments complete: submitting analysis $(sbatch --parsable slurm/05_analyze.sbatch)"
  fi
}

# finish <rc> <this sbatch script>: on exit 75 resubmit the same task (it resumes from its query logs).
finish() {
  local rc=$1 script=$2
  if [ "$rc" -eq "$EXIT_INCOMPLETE" ]; then
    local n=$(( ${SAGE_RESUBMIT_COUNT:-0} + 1 ))
    if [ "${SAGE_AUTO_RESUBMIT:-1}" = 1 ] && [ "$n" -le "${SAGE_MAX_RESUBMIT:-30}" ]; then
      local args=(--parsable --job-name="${SLURM_JOB_NAME:-sage-gpu}" --export="ALL,SAGE_RESUBMIT_COUNT=$n")
      [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && args+=(--array="$SLURM_ARRAY_TASK_ID")
      [ "$GPU_SLOTS" -le 1 ] && args+=(--dependency=singleton)
      echo "work left: resubmitted $script as job $(sbatch "${args[@]}" "$script") (resubmission $n)"
      exit 0
    fi
    echo "work left but not resubmitted (SAGE_AUTO_RESUBMIT=${SAGE_AUTO_RESUBMIT:-1}, resubmission $n)"
  fi
  [ "$rc" -eq 0 ] && maybe_analyze
  exit "$rc"
}
