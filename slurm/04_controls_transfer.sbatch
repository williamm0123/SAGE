#!/bin/bash
#SBATCH --job-name=sage-ctrl-xfer
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-3%4
# Block 2 (component controls, 100-image subset) and block 3 (SD-v1 -> SD-v2 source-only transfer).
# 0: controls/ufd  1: controls/dire  2: transfer/ufd  3: transfer/dire
source "${PROJECT_ROOT:-$HOME/SAGE}/slurm/common.sh"

DETS=(ufd dire)
i=${SLURM_ARRAY_TASK_ID:-0}
det=${DETS[$(( i % 2 ))]}
if [ "$i" -lt 2 ]; then
  python experiments/run_attacks.py --experiment controls --detector "$det" \
    --data "$SAGE_DATA" --prior "$PRIOR_MIXED" --out "$RESULTS"
else
  python experiments/run_attacks.py --experiment transfer --detector "$det" --method sage \
    --data "$SAGE_DATA" --prior "$PRIOR_SDV1" --out "$RESULTS"
fi
