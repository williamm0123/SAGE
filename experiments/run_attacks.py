"""Run attacks for one (experiment, detector, method) on one GPU, optionally sharded.

Experiments (paper Sec. 4.3):
  pilot     validation images flagged in_pilot (20 per generator), any method
  primary   all eligible generated test images, methods sage | ba | hsja | qeba
  controls  eligible test images flagged in_ablation; shared reference start + SBQR with
            uniform / lowfreq / learned weights (method is ignored)
  transfer  eligible SD-v2 test images, SAGE with the SD-v1-only prior

Results are appended as JSON lines; re-running the same command resumes and skips finished runs.

    python experiments/run_attacks.py --experiment primary --detector ufd --method sage \
        --data $DATA_ROOT/sage_v1 --prior priors/prior_mixed.npz --shard 0 --num-shards 4
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.attack import AttackContext, controls_task, make_sage_task  # noqa: E402
from sage.baselines import make_baseline_task  # noqa: E402
from sage.config import load_config  # noqa: E402
from sage.data import Manifest  # noqa: E402
from sage.detectors import build_detector  # noqa: E402
from sage.perceptual import LPIPSDistance  # noqa: E402
from sage.query import run_tasks  # noqa: E402
from sage.runinfo import run_metadata  # noqa: E402
from sage.sap import SpectralPrior  # noqa: E402
from sage.spectral import full_bands  # noqa: E402


def derive_seed(seed, image_id):
    return int(hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()[:15], 16)


def eligible_ids(screening_csv):
    import csv
    with open(screening_csv, newline="") as f:
        return {r["id"] for r in csv.DictReader(f) if r["cls"] != "real" and r["pred_real"] == "0"}


def select_targets(m, experiment):
    if experiment == "pilot":
        return m.select("validation", ["sdv1", "sdv2"], pilot=True)
    if experiment == "primary":
        return m.select("test", ["sdv1", "sdv2"])
    if experiment == "controls":
        return m.select("test", ["sdv1", "sdv2"], ablation=True)
    if experiment == "transfer":
        return m.select("test", ["sdv2"])
    raise ValueError(experiment)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=["pilot", "primary", "controls", "transfer"])
    ap.add_argument("--detector", required=True, choices=["ufd", "dire"])
    ap.add_argument("--method", default="sage", choices=["sage", "ba", "hsja", "qeba"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--prior", required=True, help="SAP prior .npz (mixed for primary/controls, sdv1-only for transfer)")
    ap.add_argument("--screening", default=None, help="default: results/screening/<detector>.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N targets")
    ap.add_argument("--concurrency", type=int, default=None, help="attacks run concurrently (default: 32 dire / 8 ufd)")
    ap.add_argument("--max-batch", type=int, default=None, help="max images per detector call (default: 32 dire / 64 ufd)")
    ap.add_argument("--fixed-budget", action="store_true", help="continue after first success (fixed-budget quality mode)")
    ap.add_argument("--save-images", type=int, default=0, help="store best candidates for the first N runs (figures)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]["attack"]
    method = "controls" if args.experiment == "controls" else args.method
    if args.experiment == "transfer" and method != "sage":
        sys.exit("transfer runs SAGE only; baselines are reused from the primary experiment")
    device = "cuda"
    m = Manifest(args.data)
    screening = Path(args.screening or f"{args.out}/screening/{args.detector}.csv")
    if not screening.exists():
        sys.exit(f"missing {screening}: run experiments/screen_clean.py first")
    ok = eligible_ids(screening)
    targets = [r for r in select_targets(m, args.experiment) if r["id"] in ok]
    targets = targets[: args.limit] if args.limit else targets
    runs = [(r, s) for r in targets for s in seeds]
    runs = runs[args.shard::args.num_shards]

    out_dir = Path(args.out) / args.experiment / args.detector / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"shard{args.shard:03d}of{args.num_shards:03d}.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            rec = json.loads(line)
            done.add((rec["id"], rec["seed"]))
    todo = [(r, s) for r, s in runs if (r["id"], s) not in done]
    print(f"{args.experiment}/{args.detector}/{method}: {len(targets)} eligible targets, "
          f"{len(runs)} runs in shard, {len(todo)} to do", flush=True)
    if not todo:
        return

    detector = build_detector(args.detector, device, cfg)
    perceptual = LPIPSDistance(device, cfg["perceptual"]["net"], cfg["perceptual"]["version"])
    prior = SpectralPrior.load(args.prior, device)
    bands = full_bands(cfg, cfg["canvas"]["size"], cfg["canvas"]["size"], device)
    refs = torch.stack([m.load(r) for r in m.references()[: cfg["init"]["max_references"]]]).to(device)
    ctx = AttackContext(cfg, prior, bands, perceptual, refs, device)

    meta = run_metadata(cfg, dict(
        experiment=args.experiment, detector=detector.describe(), method=method, seeds=seeds,
        shard=args.shard, num_shards=args.num_shards, manifest_sha256=m.sha256,
        prior=str(args.prior), prior_meta={k: prior.meta.get(k) for k in ("description", "ids_sha256", "n_real", "n_fake")},
        perceptual=perceptual.name, stop_on_success=not args.fixed_budget, argv=sys.argv))
    (out_dir / f"shard{args.shard:03d}of{args.num_shards:03d}.meta.json").write_text(json.dumps(meta, indent=2))

    stop_on_success = not args.fixed_budget
    started = {}

    def tasks():
        for r, s in todo:
            x_o = m.load(r, device)
            seed_gen = (lambda sd: (lambda: torch.Generator().manual_seed(sd)))(derive_seed(s, r["id"]))
            if method == "sage":
                gen = make_sage_task(x_o, ctx, seed_gen, stop_on_success)
            elif method == "controls":
                gen = controls_task(x_o, ctx, seed_gen, stop_on_success=stop_on_success)
            else:
                gen = make_baseline_task(method, x_o, ctx, seed_gen, stop_on_success)
            started[(r["id"], s)] = time.time()
            yield (r["id"], s), gen

    img_dir = out_dir / "images"
    saved = [0]
    fh = open(out_file, "a")

    def on_done(key, result):
        image_id, seed = key
        row = m.by_id[image_id]
        base = dict(id=image_id, cls=row["cls"], seed=seed, experiment=args.experiment,
                    detector=args.detector, wall_seconds=time.time() - started.pop(key))
        parts = result.values() if method == "controls" else [result]
        xs = [p.pop("_x", None) for p in parts]
        if saved[0] < args.save_images and xs[0] is not None:
            img_dir.mkdir(exist_ok=True)
            np.savez_compressed(img_dir / f"{image_id}_seed{seed}.npz", x_o=m.load(row).numpy(),
                                **{f"x_{i}": x.cpu().numpy() for i, x in enumerate(xs) if x is not None})
            saved[0] += 1
        if method == "controls":
            for variant, res in result.items():
                rec = dict(base, method=f"control_{variant}", **res)
                fh.write(json.dumps(rec, default=float) + "\n")
        else:
            fh.write(json.dumps(dict(base, method=method, **result), default=float) + "\n")
        fh.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {image_id} seed={seed} "
              f"{'OK ' if (result.get('success') if method != 'controls' else True) else 'FAIL'} "
              f"q={result.get('queries', '-')}", flush=True)

    conc = args.concurrency or (32 if args.detector == "dire" else 8)
    max_batch = args.max_batch or (32 if args.detector == "dire" else 64)
    t0 = time.time()
    stats = run_tasks(tasks(), detector.predict_real, max_batch=max_batch, concurrency=conc, on_done=on_done)
    fh.close()
    stats["seconds"] = time.time() - t0
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
