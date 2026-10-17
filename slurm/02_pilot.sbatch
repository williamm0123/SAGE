#!/bin/bash
#SBATCH --job-name=sage-pilot
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
# Validation-only pilot (paper Sec. 4.4): 40 generated validation images, seed 0, all methods,
# both detectors. Checks implementation, eligibility and wall time. Do NOT tune on its ASR.
source "${PROJECT_ROOT:-$HOME/SAGE}/slurm/common.sh"

for det in ufd dire; do
  for m in sage ba hsja qeba; do
    python experiments/run_attacks.py --experiment pilot --detector "$det" --method "$m" --seeds 0 \
      --data "$SAGE_DATA" --prior "$PRIOR_MIXED" --out "$RESULTS" --save-images 4
  done
done
python - <<'PY'
import json, glob, collections
for f in sorted(glob.glob("results/pilot/*/*/*.jsonl")):
    rs = [json.loads(l) for l in open(f)]
    if not rs: continue
    s = sum(r["success"] for r in rs); q = sum(r["queries"] for r in rs); t = sum(r["wall_seconds"] for r in rs)
    print(f"{f}: runs={len(rs)} success={s} mean_queries={q/len(rs):.0f} sum_wall_s={t:.0f}")
PY
