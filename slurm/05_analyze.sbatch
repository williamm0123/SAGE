#!/bin/bash
#SBATCH --job-name=sage-analyze
#SBATCH --partition=cpu-epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sage-analyze_%j.out
# Tables (Table 2, Table 3 A/B, paired differences, common-success, diagnostics), figures, bootstrap CIs.
# CPU only. Refuses to run on incomplete or inconsistent results; for a draft:
#   ANALYZE_ARGS=--allow-incomplete sbatch slurm/05_analyze.sbatch
# Submitted automatically by the last attack job that completes the protocol.
source "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}/slurm/common.sh"
python experiments/analyze.py --data "$SAGE_DATA" --results "$RESULTS" --out "$RESULTS/analysis" ${ANALYZE_ARGS:-}
