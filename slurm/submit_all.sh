#!/usr/bin/env bash
# Submit the SAGE pipeline on UMHPC (DICC). QoS wall-time limits: short 1 h, normal 8 h, long 3 days;
# every job states its QoS in its #SBATCH header (prepare: normal, attacks: long, analysis: normal).
#
# Single GPU (default, GPU_SLOTS=1): only two jobs are ever queued - prepare, then ONE queue job
# (slurm/run_queue.sbatch) that runs all experiments in order and resubmits itself before its 3-day limit.
# All GPU jobs are named "sage-gpu" with --dependency=singleton, so exactly one runs at a time.
#   bash slurm/submit_all.sh                     # prepare -> pilot (40 validation images, seed 0)
#   bash slurm/submit_all.sh --full              # prepare -> primary, controls, transfer -> analysis
#   SAGE_DETECTORS=ufd bash slurm/submit_all.sh --full    # only some detectors
#   SKIP_PREPARE=1 bash slurm/submit_all.sh --full        # manifest / priors / screening already done
#
# Several GPUs (GPU_SLOTS>1): array jobs 03/04 instead of the queue (NSHARD, NSHARD_CT shards); check your
# QoS submit limits first (sacctmgr show qos long format=name,maxwall,maxsubmitpu,maxjobspu,maxtrespu).
#   python experiments/status.py --data data/sage_v1 --results results    # progress at any time
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_ROOT=$PWD
mkdir -p logs
export GPU_SLOTS=${GPU_SLOTS:-1} NSHARD=${NSHARD:-2} NSHARD_CT=${NSHARD_CT:-1}
export SAGE_DETECTORS=${SAGE_DETECTORS:-ufd aeroblade}
PHASE=pilot; [ "${1:-}" = "--full" ] && PHASE=full

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

dep=""
if [ "${SKIP_PREPARE:-0}" != 1 ]; then
  prep=$(sub "" slurm/01_prepare.sbatch)
  echo "prepare: $prep"
  dep="afterok:$prep"
fi

if [ "$GPU_SLOTS" -le 1 ] || [ "$PHASE" = pilot ]; then
  q=$(SAGE_PHASE=$PHASE sub "$dep" slurm/run_queue.sbatch)
  echo "queue ($PHASE, detectors: $SAGE_DETECTORS): $q   log: logs/sage-queue_$q.out"
  [ "$PHASE" = full ] && echo "analysis is submitted automatically when every run is stored"
  exit 0
fi
prim=$(sub "$dep" --array=0-$(( 8 * NSHARD - 1 ))%$GPU_SLOTS slurm/03_primary.sbatch)
ctrl=$(sub "$dep" --array=0-$(( 4 * NSHARD_CT - 1 ))%$GPU_SLOTS slurm/04_controls_transfer.sbatch)
echo "primary: $prim  controls/transfer: $ctrl  (GPU_SLOTS=$GPU_SLOTS NSHARD=$NSHARD NSHARD_CT=$NSHARD_CT)"
echo "analysis is submitted by the job that completes the last run; progress: python experiments/status.py"
