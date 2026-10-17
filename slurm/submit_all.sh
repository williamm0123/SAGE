#!/usr/bin/env bash
# Submit the whole pipeline with dependencies (each GPU job requests ONE A100):
#   bash slurm/submit_all.sh            # prepare -> pilot (stop here and inspect the pilot first)
#   bash slurm/submit_all.sh --full     # prepare -> primary + controls/transfer -> analyze
set -euo pipefail
cd "${PROJECT_ROOT:-$HOME/SAGE}"
mkdir -p logs

prep=$(sbatch --parsable slurm/01_prepare.sbatch)
echo "prepare: $prep"
if [ "${1:-}" != "--full" ]; then
  pilot=$(sbatch --parsable --dependency=afterok:$prep slurm/02_pilot.sbatch)
  echo "pilot:   $pilot   (check logs/sage-pilot_$pilot.out for wall time before --full)"
  exit 0
fi
prim=$(sbatch --parsable --dependency=afterok:$prep slurm/03_primary.sbatch)
ctrl=$(sbatch --parsable --dependency=afterok:$prep slurm/04_controls_transfer.sbatch)
ana=$(sbatch --parsable --dependency=afterany:$prim:$ctrl slurm/05_analyze.sbatch)
echo "primary: $prim  controls/transfer: $ctrl  analyze: $ana"
