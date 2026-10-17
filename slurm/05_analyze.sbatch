#!/bin/bash
#SBATCH --job-name=sage-analyze
#SBATCH --partition=cpu-epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Tables (Table 2, Table 3 A/B), ASR-vs-query figures, bootstrap CIs. CPU only.
source "${PROJECT_ROOT:-$HOME/SAGE}/slurm/common.sh"
python experiments/analyze.py --results "$RESULTS" --out "$RESULTS/analysis"
