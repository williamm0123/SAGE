#!/bin/bash
#SBATCH --job-name=sage-pilot
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --output=logs/sage-pilot_%j.out
# Validation-only pilot (paper Sec. 4.4): 40 generated validation images, seed 0, all methods,
# both detectors. Checks implementation, eligibility and throughput. Do NOT tune on its ASR.
# Resumable: on exit 75 the job resubmits itself and continues from the query logs.
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/slurm/common.sh"

rc=0
for det in ufd aeroblade; do
  for m in sage ba hsja qeba; do
    python experiments/run_attacks.py --experiment pilot --detector "$det" --method "$m" --seeds 0 \
      --data "$SAGE_DATA" --prior "$PRIOR_MIXED" --out "$RESULTS" --save-images 4 || rc=$?
    [ "$rc" -eq 0 ] || break 2
  done
done
# Throughput from process wall time and images sent (per-run wall_seconds overlap and are not summed),
# and the projected single-GPU hours of the full protocol.
python experiments/status.py --data "$SAGE_DATA" --results "$RESULTS" --experiments pilot || true
python experiments/status.py --data "$SAGE_DATA" --results "$RESULTS" || true
SAGE_AUTO_ANALYZE=0 finish "$rc" slurm/02_pilot.sbatch
