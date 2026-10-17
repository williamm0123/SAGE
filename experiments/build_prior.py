"""Offline Spectral Artifact Profiling from the development split (no detector queries).

    python experiments/build_prior.py --data $DATA_ROOT/sage_v1 --generators sdv1 sdv2 --out priors/prior_mixed.npz
    python experiments/build_prior.py --data $DATA_ROOT/sage_v1 --generators sdv1      --out priors/prior_sdv1.npz
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.config import load_config  # noqa: E402
from sage.data import Manifest  # noqa: E402
from sage.runinfo import run_metadata  # noqa: E402
from sage.sap import build_prior  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--generators", nargs="+", required=True, choices=["sdv1", "sdv2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = Manifest(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = m.select("development", ["real"])
    fake = m.select("development", args.generators)
    prior = build_prior(real, fake, lambda r: m.load(r), device,
                        description=f"development real vs {'+'.join(args.generators)}")
    prior.meta.update(run_metadata(cfg, dict(manifest_sha256=m.sha256)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prior.save(args.out)
    print(json.dumps({k: prior.meta[k] for k in ("description", "n_real", "n_fake", "ids_sha256")}, indent=2))


if __name__ == "__main__":
    main()
