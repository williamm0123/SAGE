#!/bin/bash
#SBATCH --job-name=sage-primary
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/sage-primary_%A_%a.out
#SBATCH --array=0-15%1
# Primary comparison (block 1): 2 detectors x 4 methods x NSHARD shards, one A100 per array task.
# Array index -> (detector, method, shard); UFD tasks come first. Submit via slurm/submit_all.sh, which
# sets the array size from NSHARD. A task stops 30 min before its 3-day limit and resubmits itself;
# unfinished attacks resume from their query logs. Changing NSHARD later is safe (finished runs are
# read from all shard files).
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/slurm/common.sh"

NSHARD=${NSHARD:-2}
DETS=(ufd dire); METHODS=(sage ba hsja qeba)
i=${SLURM_ARRAY_TASK_ID:-0}
shard=$(( i % NSHARD )); i=$(( i / NSHARD ))
method=${METHODS[$(( i % 4 ))]}; det=${DETS[$(( i / 4 ))]}
echo "detector=$det method=$method shard=$shard/$NSHARD"

rc=0
python experiments/run_attacks.py --experiment primary --detector "$det" --method "$method" \
  --data "$SAGE_DATA" --prior "$PRIOR_MIXED" --out "$RESULTS" \
  --shard "$shard" --num-shards "$NSHARD" --save-images 2 || rc=$?
finish "$rc" slurm/03_primary.sbatch
