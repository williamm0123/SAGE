#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
# One evaluation block of protocol SAGE-v1.0 (paper Sec. 4.3) on one A100, so the three blocks run in parallel:
#   SAGE_BLOCK=primary   block 1: SAGE / BA / HSJA / QEBA on all eligible test images, seeds 0/1/2, mixed prior
#   SAGE_BLOCK=controls  block 2: 100-image subset, shared start + uniform / low-frequency / learned sampling
#   SAGE_BLOCK=transfer  block 3: SAGE with the SD-v1-only prior on the SD-v2 test images
# Submit with slurm/submit_blocks.sh. Each block resubmits itself (same job name, --dependency=singleton)
# before its 3-day limit and resumes from stored runs and query logs; the job that completes the last run
# submits the analysis.
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/slurm/common.sh"

BLOCK=${SAGE_BLOCK:?set SAGE_BLOCK=primary|controls|transfer}
DETECTORS=${SAGE_DETECTORS:-ufd aeroblade}
echo "block=$BLOCK detectors=$DETECTORS"

items=()
for det in $DETECTORS; do
  case "$BLOCK" in
    primary)  for m in sage ba hsja qeba; do items+=("$det $m"); done ;;
    controls|transfer) items+=("$det sage") ;;
    *) echo "unknown SAGE_BLOCK=$BLOCK"; exit 2 ;;
  esac
done

prior=$PRIOR_MIXED; extra=()
[ "$BLOCK" = transfer ] && prior=$PRIOR_SDV1
[ "$BLOCK" = primary ] && extra=(--save-images 2)

rc=0
for it in "${items[@]}"; do
  read -r det m <<<"$it"
  echo "== $(date -Is) $BLOCK / $det / $m"
  python experiments/run_attacks.py --experiment "$BLOCK" --detector "$det" --method "$m" \
    --data "$SAGE_DATA" --prior "$prior" --out "$RESULTS" "${extra[@]}" || rc=$?
  [ "$rc" -eq 0 ] || break        # 75: time limit near -> resubmit; anything else: stop and inspect
done

python experiments/status.py --data "$SAGE_DATA" --results "$RESULTS" --experiments "$BLOCK" --detectors $DETECTORS || true
finish "$rc" slurm/run_block.sbatch
