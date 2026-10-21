#!/bin/bash
#SBATCH --job-name=sage-ctrl-xfer
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/sage-ctrl-xfer_%A_%a.out
#SBATCH --array=0-3%1
# Block 2 (component controls, 100-image subset) and block 3 (SD-v1 -> SD-v2 source-only transfer),
# each split into NSHARD_CT shards: array index -> (block, detector, shard), 4 x NSHARD_CT tasks.
#   blocks: 0 controls/ufd  1 controls/dire  2 transfer/ufd  3 transfer/dire
# Resumable and self-resubmitting like slurm/03_primary.sbatch.
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/slurm/common.sh"

NSHARD_CT=${NSHARD_CT:-1}
DETS=(ufd dire)
i=${SLURM_ARRAY_TASK_ID:-0}
shard=$(( i % NSHARD_CT )); b=$(( i / NSHARD_CT ))
det=${DETS[$(( b % 2 ))]}
rc=0
if [ "$b" -lt 2 ]; then
  echo "controls detector=$det shard=$shard/$NSHARD_CT"
  python experiments/run_attacks.py --experiment controls --detector "$det" \
    --data "$SAGE_DATA" --prior "$PRIOR_MIXED" --out "$RESULTS" \
    --shard "$shard" --num-shards "$NSHARD_CT" || rc=$?
else
  echo "transfer detector=$det shard=$shard/$NSHARD_CT"
  python experiments/run_attacks.py --experiment transfer --detector "$det" --method sage \
    --data "$SAGE_DATA" --prior "$PRIOR_SDV1" --out "$RESULTS" \
    --shard "$shard" --num-shards "$NSHARD_CT" || rc=$?
fi
finish "$rc" slurm/04_controls_transfer.sbatch
