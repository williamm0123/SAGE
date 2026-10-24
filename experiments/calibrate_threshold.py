"""Freeze a decision threshold for a detector that has none (AEROBLADE), using only the development split.

    python experiments/calibrate_threshold.py --data $DATA_ROOT/sage_v1 --detector aeroblade --out results

Scores all development images (500 real, 250 SD-v1, 250 SD-v2; never validation or test), then picks the
threshold maximising balanced accuracy, placed midway between neighbouring scores (lowest such threshold on
ties). Writes results/calibration/<detector>.json (+ scores CSV) and refuses to overwrite it: the threshold
is part of the frozen detector, so recalibrating means a new results directory.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.config import load_config  # noqa: E402
from sage.data import Manifest  # noqa: E402
from sage.detectors import build_detector, fixed_batch  # noqa: E402
from sage.experiment import DETECTOR_BATCH  # noqa: E402
from sage.runinfo import run_metadata  # noqa: E402


def best_threshold(real, fake):
    """Lowest threshold maximising ((real <= t) + (fake > t)) / 2, midway between neighbouring scores."""
    v = np.unique(np.concatenate([real, fake]))
    cand = np.concatenate([[v[0] - 1.0], (v[:-1] + v[1:]) / 2, [v[-1] + 1.0]])
    bal = np.array([((real <= t).mean() + (fake > t).mean()) / 2 for t in cand])
    i = int(np.argmax(bal))
    return float(cand[i]), float(bal[i])


def auc(neg, pos):
    allv = np.concatenate([neg, pos])
    r = allv.argsort(kind="stable").argsort() + 1
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--detector", default="aeroblade", choices=["aeroblade"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    out = Path(args.out) / "calibration"
    path = out / f"{args.detector}.json"
    if path.exists():
        print(f"{path} exists - the threshold is frozen; use a new --out to recalibrate")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    crit = cfg["detectors"][args.detector]["calibration"]
    if crit != {"split": "development", "criterion": "max_balanced_accuracy"}:
        sys.exit(f"unsupported calibration spec {crit}")
    m = Manifest(args.data)
    det = build_detector(args.detector, "cuda", cfg, threshold=float("nan"))
    bs = DETECTOR_BATCH[args.detector]
    score_fn = fixed_batch(det.prob_fake, bs)
    rows = m.select("development")
    t0 = time.time()
    scores = []
    for i in range(0, len(rows), bs):
        x = torch.stack([m.load(r) for r in rows[i:i + bs]]).cuda()
        scores += score_fn(x).cpu().tolist()
        print(f"{min(i + bs, len(rows))}/{len(rows)}", flush=True)
    seconds = time.time() - t0
    s = np.array(scores, dtype=np.float64)
    is_real = np.array([r["cls"] == "real" for r in rows])
    thr, bal = best_threshold(s[is_real], s[~is_real])

    with open(out / f"{args.detector}_development_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "cls", "score"])
        w.writerows((r["id"], r["cls"], f"{v:.9g}") for r, v in zip(rows, scores))
    per_class = {c: dict(n=int(sum(r["cls"] == c for r in rows)),
                         rate_fake=float(np.mean([v > thr for r, v in zip(rows, scores) if r["cls"] == c])))
                 for c in ("real", "sdv1", "sdv2")}
    desc = det.describe()
    desc.pop("threshold")
    result = dict(
        detector=args.detector, threshold=thr, criterion=crit, development_balanced_accuracy=bal,
        development_auc=auc(s[is_real], s[~is_real]), per_class=per_class, detector_batch=bs,
        scores_sha256=hashlib.sha256(np.round(s, 9).tobytes()).hexdigest(), detector_description=desc,
        manifest_sha256=m.sha256, seconds=seconds, meta=run_metadata(cfg),
    )
    path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("threshold", "development_balanced_accuracy", "development_auc",
                                             "per_class")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
