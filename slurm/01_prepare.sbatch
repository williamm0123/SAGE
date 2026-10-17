#!/bin/bash
#SBATCH --job-name=sage-prepare
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Offline SAP priors (no detector queries) + clean screening for both detectors (eligibility cost,
# reported separately, never charged to attack budgets).
source "${PROJECT_ROOT:-$HOME/SAGE}/slurm/common.sh"

python scripts/data/verify_manifest.py "$SAGE_DATA"
python experiments/build_prior.py --data "$SAGE_DATA" --generators sdv1 sdv2 --out "$PRIOR_MIXED"
python experiments/build_prior.py --data "$SAGE_DATA" --generators sdv1      --out "$PRIOR_SDV1"
python experiments/screen_clean.py --data "$SAGE_DATA" --detector ufd  --out "$RESULTS/screening"
python experiments/screen_clean.py --data "$SAGE_DATA" --detector dire --out "$RESULTS/screening" --batch-size 32
