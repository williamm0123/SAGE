#!/bin/bash
#SBATCH --job-name=sage-gpu
#SBATCH --partition=gpu-a100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/sage-queue_%j.out
# Single-GPU work queue: runs every (experiment, detector, method) of the protocol one after another in
# ONE job, so at most one SAGE job is ever queued or running (UMHPC QoS / submit limits). Each step resumes
# from stored runs and query logs; when the 3-day limit approaches (exit 75) the job resubmits itself with
# --dependency=singleton and continues where it stopped. When everything is stored, the analysis is submitted.
#
#   SAGE_PHASE=pilot sbatch slurm/run_queue.sbatch    # 40 validation images, seed 0, all methods
#   SAGE_PHASE=full  sbatch slurm/run_queue.sbatch    # primary -> controls -> transfer, per detector
#   SAGE_DETECTORS="ufd" ...                          # restrict to some detectors (default: "ufd aeroblade")
#
# Billing on gpu-a100 is the maximum of GPU (150000/GPU), CPU (4687.5/core) and memory (600/GB), so 16 cores
# and 128 GB cost nothing beyond the one A100.
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/slurm/common.sh"

PHASE=${SAGE_PHASE:-full}
DETECTORS=${SAGE_DETECTORS:-ufd aeroblade}
echo "phase=$PHASE detectors=$DETECTORS"

items=()
for det in $DETECTORS; do
  if [ "$PHASE" = pilot ]; then
    for m in sage ba hsja qeba; do items+=("pilot $det $m"); done
  else
    for m in sage ba hsja qeba; do items+=("primary $det $m"); done
    items+=("controls $det sage" "transfer $det sage")
  fi
done

rc=0
for it in "${items[@]}"; do
  read -r exp det m <<<"$it"
  prior=$PRIOR_MIXED; extra=()
  [ "$exp" = transfer ] && prior=$PRIOR_SDV1
  [ "$exp" = pilot ] && extra=(--seeds 0 --save-images 4)
  [ "$exp" = primary ] && extra=(--save-images 2)
  echo "== $(date -Is) $exp / $det / $m"
  python experiments/run_attacks.py --experiment "$exp" --detector "$det" --method "$m" \
    --data "$SAGE_DATA" --prior "$prior" --out "$RESULTS" "${extra[@]}" || rc=$?
  [ "$rc" -eq 0 ] || break        # 75: time limit near -> resubmit; anything else: stop and inspect
done

if [ "$PHASE" = pilot ]; then
  python experiments/status.py --data "$SAGE_DATA" --results "$RESULTS" --experiments pilot || true
  python experiments/status.py --data "$SAGE_DATA" --results "$RESULTS" || true
  SAGE_AUTO_ANALYZE=0 finish "$rc" slurm/run_queue.sbatch
fi
python experiments/status.py --data "$SAGE_DATA" --results "$RESULTS" || true
finish "$rc" slurm/run_queue.sbatch
