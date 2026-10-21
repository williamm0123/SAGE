#!/usr/bin/env bash
# Run the test stages directly on a GPU node you already hold (salloc / srun --pty), without sbatch.
#
#   bash scripts/umhpc/run_interactive.sh            # check -> prepare -> pilot
#   bash scripts/umhpc/run_interactive.sh check      # GPU / weights / determinism self-check only
#   bash scripts/umhpc/run_interactive.sh prepare    # manifest, SAP priors, clean screening
#   bash scripts/umhpc/run_interactive.sh pilot      # 40 validation images, all methods, both detectors
#   bash scripts/umhpc/run_interactive.sh status     # pilot throughput + projected GPU hours
#
# It executes the same job bodies as scripts/umhpc/check_env.sbatch, slurm/01_prepare.sbatch and
# slurm/02_pilot.sbatch (the #SBATCH lines are only comments to bash), with two differences:
#   * nothing is resubmitted with sbatch (SAGE_AUTO_RESUBMIT=0) and no analysis job is queued;
#   * output goes to the terminal and to logs/interactive_<stage>_<time>.log.
# If the allocation is about to end, the pilot stops 30 min early (exit 75); run the same command
# again in a new allocation and it resumes from the stored results and query logs.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PROJECT_ROOT=$ROOT
export SAGE_AUTO_RESUBMIT=0 SAGE_AUTO_ANALYZE=0
mkdir -p logs

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No GPU visible here. Run this on the allocated GPU node, e.g.:" >&2
  echo "  srun --jobid=<your salloc job id> --pty bash    # or ssh to the node shown by squeue -u \$USER" >&2
  exit 1
fi
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "note: SLURM_JOB_ID is not set, so the pilot cannot see the allocation's time limit;"
  echo "      it will run until finished (use 'srun --jobid=... --pty bash' to get an early, resumable stop)."
fi

run_stage() {  # run_stage <name> <job script>
  local name=$1 script=$2 log rc=0
  log=logs/interactive_${name}_$(date +%Y%m%d_%H%M%S).log
  echo "=================== $name ($script) -> $log"
  bash "$script" 2>&1 | tee "$log" || rc=$?
  if [ "$rc" -eq 75 ]; then
    echo "=================== $name stopped before the time limit with work left (exit 75)."
    echo "Re-run:  bash scripts/umhpc/run_interactive.sh $name   (it resumes where it stopped)"
  elif [ "$rc" -ne 0 ]; then
    echo "=================== $name FAILED (exit $rc), see $log"
  else
    echo "=================== $name OK"
  fi
  return "$rc"
}

status() {
  # shellcheck disable=SC1091
  source scripts/umhpc/env.sh
  local data=${SAGE_DATA:-$DATA_ROOT/sage_v1}
  python experiments/status.py --data "$data" --results "${RESULTS:-$ROOT/results}" --experiments pilot
  python experiments/status.py --data "$data" --results "${RESULTS:-$ROOT/results}" || true
}

stage=${1:-all}
case "$stage" in
  check)   run_stage check   scripts/umhpc/check_env.sbatch ;;
  prepare) run_stage prepare slurm/01_prepare.sbatch ;;
  pilot)   run_stage pilot   slurm/02_pilot.sbatch ;;
  status)  status ;;
  all)
    run_stage check   scripts/umhpc/check_env.sbatch
    run_stage prepare slurm/01_prepare.sbatch
    run_stage pilot   slurm/02_pilot.sbatch
    ;;
  *) echo "usage: $0 [all|check|prepare|pilot|status]" >&2; exit 2 ;;
esac
