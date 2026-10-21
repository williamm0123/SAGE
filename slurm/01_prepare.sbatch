#!/bin/bash
#SBATCH --job-name=sage-prepare
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sage-prepare_%j.out
# Frozen manifest (built once from $DIRE_ROOT), offline SAP priors (no detector queries) and clean
# screening for both detectors (eligibility cost, reported separately, never charged to attack budgets).
# Every step is skipped when its output already exists, so re-submitting is safe and keeps them frozen.
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/slurm/common.sh"

LSUN_TEST=$DIRE_ROOT/images/test/lsun_bedroom
SDV1_DIR=${SDV1_DIR:-sdv1_new}
if [ ! -f "$SAGE_DATA/manifest.csv" ]; then
  # test/lsun_bedroom/real has exactly 1000 images and 15 of them duplicate DIRE training images,
  # so val/lsun_bedroom/real joins the real pool (see docs/DATA.md).
  python scripts/data/build_manifest.py \
    --real "$LSUN_TEST/real" "$DIRE_ROOT/images/val/lsun_bedroom/real" \
    --sdv1 "$LSUN_TEST/$SDV1_DIR" \
    --sdv2 "$LSUN_TEST/sdv2" \
    --exclude "$DIRE_ROOT/images/train/lsun_bedroom/real" \
    --dataset-version "${DATASET_VERSION:-DiffusionForensics (DIRE benchmark) at $DIRE_ROOT}" \
    --out "$SAGE_DATA"
fi
python scripts/data/verify_manifest.py "$SAGE_DATA"

[ -f "$PRIOR_MIXED" ] || python experiments/build_prior.py --data "$SAGE_DATA" --generators sdv1 sdv2 --out "$PRIOR_MIXED"
[ -f "$PRIOR_SDV1" ]  || python experiments/build_prior.py --data "$SAGE_DATA" --generators sdv1      --out "$PRIOR_SDV1"
for det in ufd dire; do
  [ -f "$RESULTS/screening/$det.csv" ] || python experiments/screen_clean.py --data "$SAGE_DATA" --detector "$det" --out "$RESULTS/screening"
done
