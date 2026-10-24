"""Clean-image screening (separate eligibility cost, excluded from every attack budget).

Writes results/screening/<detector>.csv (id, cls, subset, prob_fake, pred_real) and a summary with
the clean fake-detection rate per generator and the real-image false-positive rate.

    python experiments/screen_clean.py --data $DATA_ROOT/sage_v1 --detector ufd
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.config import load_config  # noqa: E402
from sage.data import Manifest  # noqa: E402
from sage.detectors import build_detector, fixed_batch  # noqa: E402
from sage.experiment import DETECTOR_BATCH, DETECTORS  # noqa: E402
from sage.runinfo import run_metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--detector", required=True, choices=list(DETECTORS))
    ap.add_argument("--subsets", nargs="+", default=["validation", "test"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help="fixed detector batch, zero-padded (default: DETECTOR_BATCH, same as run_attacks.py)")
    ap.add_argument("--out", default="results/screening")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = Manifest(args.data)
    device = "cuda"
    det = build_detector(args.detector, device, cfg, results=Path(args.out).parent)
    bs = args.batch_size or DETECTOR_BATCH[args.detector]
    prob_fake = fixed_batch(det.prob_fake, bs)
    rows = [r for s in args.subsets for r in m.select(s)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    records = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        x = torch.stack([m.load(r) for r in chunk]).to(device)
        p = prob_fake(x).cpu()
        for r, pf in zip(chunk, p.tolist()):
            records.append(dict(id=r["id"], cls=r["cls"], subset=r["subset"], prob_fake=pf,
                                pred_real=int(pf <= det.threshold)))
        print(f"{i + len(chunk)}/{len(rows)}", flush=True)
    seconds = time.time() - t0

    with open(out / f"{args.detector}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    summary = {}
    for s in args.subsets:
        for c in ("real", "sdv1", "sdv2"):
            rs = [r for r in records if r["subset"] == s and r["cls"] == c]
            if not rs:
                continue
            n_fake_pred = sum(1 - r["pred_real"] for r in rs)
            key = "false_positive_rate" if c == "real" else "fake_detection_rate"
            summary[f"{s}/{c}"] = {"n": len(rs), key: n_fake_pred / len(rs),
                                   **({} if c == "real" else {"eligible": n_fake_pred})}
    meta = run_metadata(cfg, dict(detector=det.describe(), manifest_sha256=m.sha256,
                                  seconds=seconds, seconds_per_image=seconds / len(rows),
                                  detector_batch=bs))
    (out / f"{args.detector}_summary.json").write_text(json.dumps(dict(summary=summary, meta=meta), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
