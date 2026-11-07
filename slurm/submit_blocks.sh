#!/usr/bin/env bash
# Submit the three evaluation blocks of protocol SAGE-v1.0 as three parallel single-A100 jobs
# (primary, controls, transfer). Assumes prepare already ran: manifest, priors, AEROBLADE threshold and
# clean screening exist under $RESULTS (the attack scripts abort if they are missing or inconsistent).
#   bash slurm/submit_blocks.sh                      # all three blocks, detectors "ufd aeroblade"
#   bash slurm/submit_blocks.sh primary transfer     # only some blocks
#   SAGE_DETECTORS=aeroblade bash slurm/submit_blocks.sh
#   python experiments/status.py --data data/sage_v1 --results results    # progress at any time
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT_ROOT=$PWD
mkdir -p logs
export SAGE_DETECTORS=${SAGE_DETECTORS:-ufd aeroblade}
# Each block is its own chain: resubmissions keep the job name and wait on --dependency=singleton.
export GPU_SLOTS=1

blocks=("$@"); [ ${#blocks[@]} -gt 0 ] || blocks=(primary controls transfer)
for b in "${blocks[@]}"; do
  j=$(SAGE_BLOCK=$b sbatch --parsable --job-name="sage-$b" --dependency=singleton slurm/run_block.sbatch)
  echo "$b: job $j   log: logs/sage-${b}_$j.out"
done
echo "analysis is submitted automatically when every run is stored (detectors: $SAGE_DETECTORS)"
