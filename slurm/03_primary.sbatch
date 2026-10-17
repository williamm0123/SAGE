#!/bin/bash
#SBATCH --job-name=sage-primary
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-15%4
# Primary comparison (block 1): 2 detectors x 4 methods x NSHARD shards, one A100 per array task.
# Array index -> (detector, method, shard). Default: 2 shards per (detector, method) = 16 tasks;
# %4 caps how many single-GPU tasks run at once (be fair to other users). Re-submitting resumes.
#   NSHARD=4 sbatch --array=0-31%4 slurm/03_primary.sbatch
source "${PROJECT_ROOT:-$HOME/SAGE}/slurm/common.sh"

NSHARD=${NSHARD:-2}
DETS=(ufd dire); METHODS=(sage ba hsja qeba)
i=${SLURM_ARRAY_TASK_ID:-0}
shard=$(( i % NSHARD )); i=$(( i / NSHARD ))
method=${METHODS[$(( i % 4 ))]}; det=${DETS[$(( i / 4 ))]}
echo "detector=$det method=$method shard=$shard/$NSHARD"

python experiments/run_attacks.py --experiment primary --detector "$det" --method "$method" \
  --data "$SAGE_DATA" --prior "$PRIOR_MIXED" --out "$RESULTS" \
  --shard "$shard" --num-shards "$NSHARD" --save-images 2
