#!/usr/bin/env bash
# Submit the pipeline. Every GPU job requests ONE A100 for at most 3 days. With GPU_SLOTS=1 (default) all
# GPU jobs share the name "sage-gpu" with --dependency=singleton, so exactly one runs at a time, including
# the jobs that resubmit themselves to continue after their 3-day limit.
#   bash slurm/submit_all.sh            # prepare -> pilot (inspect throughput / projection first)
#   bash slurm/submit_all.sh --full     # prepare -> primary -> controls/transfer; analysis is submitted
#                                       # automatically once every run is stored
#   python experiments/status.py --data data/sage_v1 --results results    # progress at any time
# Overrides: GPU_SLOTS (parallel single-GPU tasks), NSHARD (shards per detector x method in primary),
#            NSHARD_CT (shards per block in controls/transfer), SAGE_AUTO_RESUBMIT=0, SAGE_MAX_RESUBMIT.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_ROOT=$PWD
mkdir -p logs
export GPU_SLOTS=${GPU_SLOTS:-1} NSHARD=${NSHARD:-2} NSHARD_CT=${NSHARD_CT:-1}

sub() {  # sub <dependency or ""> <sbatch args...>
  local dep=$1; shift
  local extra=()
  if [ "$GPU_SLOTS" -le 1 ]; then
    extra=(--job-name=sage-gpu --dependency="${dep:+$dep,}singleton")
  elif [ -n "$dep" ]; then
    extra=(--dependency="$dep")
  fi
  sbatch --parsable "${extra[@]}" "$@"
}

prep=$(sub "" slurm/01_prepare.sbatch)
echo "prepare: $prep"
if [ "${1:-}" != "--full" ]; then
  pilot=$(sub "afterok:$prep" slurm/02_pilot.sbatch)
  echo "pilot:   $pilot   (logs/sage-pilot_$pilot.out reports throughput and projected GPU hours)"
  exit 0
fi
prim=$(sub "afterok:$prep" --array=0-$(( 8 * NSHARD - 1 ))%$GPU_SLOTS slurm/03_primary.sbatch)
ctrl=$(sub "afterok:$prep" --array=0-$(( 4 * NSHARD_CT - 1 ))%$GPU_SLOTS slurm/04_controls_transfer.sbatch)
echo "primary: $prim  controls/transfer: $ctrl  (GPU_SLOTS=$GPU_SLOTS NSHARD=$NSHARD NSHARD_CT=$NSHARD_CT)"
echo "analysis is submitted by the job that completes the last run; progress: python experiments/status.py"
