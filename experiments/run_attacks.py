"""Run attacks for one (experiment, detector, method) on one GPU, optionally sharded.

Experiments (paper Sec. 4.3):
  pilot     validation images flagged in_pilot (20 per generator), any method
  primary   all eligible generated test images, methods sage | ba | hsja | qeba
  controls  eligible test images flagged in_ablation; shared reference start + SBQR with
            uniform / lowfreq / learned weights (method is ignored)
  transfer  eligible SD-v2 test images, SAGE with the SD-v1-only prior

Integrity and resumption:
  * results/<experiment>/<detector>/<method>/LOCK.json freezes config, manifest, screening, prior,
    detector weights / settings / batch, code and baseline parameters; a mismatch aborts (exit 3).
  * finished runs are read from every shard file of the directory, so changing the shard count is safe;
    a controls run is finished only when all three variants are stored (they are written together).
  * every unfinished run keeps a query log (qlog/<id>_seed<s>.qlog); a resumed run replays it without
    new detector calls and continues exactly where it stopped.
  * --stop-margin-min stops before the Slurm time limit and exits with code 75 (resubmit to continue);
    exit 0 means the shard is complete.
  * each session appends its wall time and detector throughput to sessions.jsonl (per-run
    wall_seconds overlap between concurrent attacks and must not be summed).

    python experiments/run_attacks.py --experiment primary --detector ufd --method sage \
        --data $DATA_ROOT/sage_v1 --prior priors/prior_mixed.npz --shard 0 --num-shards 4
"""
import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.attack import AttackContext, controls_task, make_sage_task  # noqa: E402
from sage.baselines import BASELINE_PARAMS, make_baseline_task  # noqa: E402
from sage.config import load_config  # noqa: E402
from sage.data import Manifest  # noqa: E402
from sage.detectors import build_detector, fixed_batch  # noqa: E402
from sage.experiment import (EXIT_INCOMPLETE, EXIT_LOCK_MISMATCH, append_records, check_lock,  # noqa: E402
                             code_sha256, done_runs, eligible_ids, expected_runs, load_dir, method_dir,
                             qlog_path, repair_jsonl, resumable, screening_paths, sha256_file,
                             slurm_seconds_left)
from sage.perceptual import LPIPSDistance  # noqa: E402
from sage.query import run_tasks  # noqa: E402
from sage.runinfo import run_metadata  # noqa: E402
from sage.sap import SpectralPrior  # noqa: E402
from sage.spectral import full_bands  # noqa: E402

DEFAULT_BATCH = {"dire": 32, "ufd": 64}
DEFAULT_CONCURRENCY = {"dire": 32, "ufd": 8}


def derive_seed(seed, image_id):
    return int(hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()[:15], 16)


def check_screening(summary_path, manifest_sha, detector, batch):
    """The eligibility labels must come from this manifest and this exact detector function."""
    if not summary_path.exists():
        return [f"missing {summary_path}"]
    meta = json.loads(summary_path.read_text())["meta"]
    problems = []
    if meta.get("manifest_sha256") != manifest_sha:
        problems.append("screening was run on a different manifest")
    d, now = meta.get("detector", {}), detector.describe()
    if {k: v[1] for k, v in d.get("files", {}).items()} != {k: v[1] for k, v in now["files"].items()}:
        problems.append("screening used different detector weights")
    if d.get("threshold") != now["threshold"] or d.get("recon_args") != now.get("recon_args"):
        problems.append("screening used different detector settings")
    if meta.get("detector_batch") != batch:
        problems.append(f"screening detector batch {meta.get('detector_batch')} != attack batch {batch}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=["pilot", "primary", "controls", "transfer"])
    ap.add_argument("--detector", required=True, choices=["ufd", "dire"])
    ap.add_argument("--method", default="sage", choices=["sage", "ba", "hsja", "qeba"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--prior", required=True, help="SAP prior .npz (mixed for primary/controls, sdv1-only for transfer)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N targets (do not use for results)")
    ap.add_argument("--concurrency", type=int, default=None, help="attacks run concurrently (default: 32 dire / 8 ufd)")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="fixed images per detector call, zero-padded (default: 32 dire / 64 ufd; must equal screening)")
    ap.add_argument("--fixed-budget", action="store_true", help="continue after first success (fixed-budget quality mode)")
    ap.add_argument("--save-images", type=int, default=0,
                    help="store x_o and the reported candidate for up to N successes and N failures (figures)")
    ap.add_argument("--stop-margin-min", type=float, default=30.0,
                    help="inside Slurm: stop this many minutes before the time limit (exit 75)")
    ap.add_argument("--max-hours", type=float, default=None, help="stop after this many hours (exit 75)")
    ap.add_argument("--keep-qlogs", action="store_true", help="keep query logs of finished runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    t_start = time.time()
    left = slurm_seconds_left()
    deadlines = [t_start + left - 60 * args.stop_margin_min] if left is not None else []
    if args.max_hours:
        deadlines.append(t_start + 3600 * args.max_hours)
    deadline = min(deadlines) if deadlines else None

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]["attack"]
    method = "controls" if args.experiment == "controls" else args.method
    if args.experiment == "transfer" and method != "sage":
        sys.exit("transfer runs SAGE only; baselines are reused from the primary experiment")
    stop_on_success = not args.fixed_budget
    if stop_on_success != (cfg["budget"]["stopping_mode"] == "first_success"):
        print("NOTE: stopping mode differs from the protocol default; results go to their own directory")
    device = "cuda"
    m = Manifest(args.data)
    screening_csv, screening_json = screening_paths(args.out, args.detector)
    if not screening_csv.exists():
        sys.exit(f"missing {screening_csv}: run experiments/screen_clean.py first")
    runs = expected_runs(m, args.experiment, eligible_ids(screening_csv), seeds)
    if args.limit:
        keep = set(list(dict.fromkeys(r["id"] for r, _ in runs))[: args.limit])
        runs = [(r, s) for r, s in runs if r["id"] in keep]
    shard_runs = runs[args.shard::args.num_shards]

    sub = (method_dir(args.experiment, method) + ("" if stop_on_success else "_fixedbudget")
           + (f"_limit{args.limit}" if args.limit else ""))     # debug / non-protocol runs never mix with results
    out_dir = Path(args.out) / args.experiment / args.detector / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"shard{args.shard:03d}of{args.num_shards:03d}.jsonl"
    repair_jsonl(out_file)
    done = done_runs(load_dir(out_dir), args.experiment)
    for r, s in runs:                           # query logs of runs finished in an earlier session
        if (r["id"], s) in done and not args.keep_qlogs and qlog_path(out_dir, r["id"], s).exists():
            qlog_path(out_dir, r["id"], s).unlink()
    todo = [(r, s) for r, s in shard_runs if (r["id"], s) not in done]
    n_partial = sum(qlog_path(out_dir, r["id"], s).exists() for r, s in todo)
    print(f"{args.experiment}/{args.detector}/{sub}: {len(runs)} runs expected, {len(shard_runs)} in shard "
          f"{args.shard}/{args.num_shards}, {len(todo)} to do ({n_partial} resumable from query logs)", flush=True)
    if not todo:
        return 0

    max_batch = args.max_batch or DEFAULT_BATCH[args.detector]
    conc = args.concurrency or DEFAULT_CONCURRENCY[args.detector]
    detector = build_detector(args.detector, device, cfg)
    problems = check_screening(screening_json, m.sha256, detector, max_batch)
    if problems:
        print("ERROR screening does not match this run:\n  " + "\n  ".join(problems), flush=True)
        return EXIT_LOCK_MISMATCH
    perceptual = LPIPSDistance(device, cfg["perceptual"]["net"], cfg["perceptual"]["version"])
    prior = SpectralPrior.load(args.prior, device)
    bands = full_bands(cfg, cfg["canvas"]["size"], cfg["canvas"]["size"], device)
    refs = torch.stack([m.load(r) for r in m.references()[: cfg["init"]["max_references"]]]).to(device)
    ctx = AttackContext(cfg, prior, bands, perceptual, refs, device)

    info = run_metadata(cfg)
    desc = detector.describe()
    lock = dict(
        protocol=cfg["protocol_version"], config_sha256=cfg["_sha256"], experiment=args.experiment,
        method=method, seeds=seeds, stop_on_success=stop_on_success, limit=args.limit,
        manifest_sha256=m.sha256, screening_sha256=sha256_file(screening_csv),
        prior_sha256=prior.sha256, prior_description=prior.meta.get("description"),
        detector={k: v for k, v in desc.items() if k != "files"},
        detector_files={k: v[1] for k, v in desc["files"].items()}, detector_batch=max_batch,
        perceptual=perceptual.name, code_sha256=code_sha256(), third_party=info["third_party"],
        torch=info["torch"], cuda=info["cuda"],
        baseline_params=BASELINE_PARAMS.get(method),
    )
    diffs = check_lock(out_dir, lock)
    if diffs:
        print(f"ERROR {out_dir}/LOCK.json was created under a different configuration:\n  " + "\n  ".join(diffs) +
              "\nUse a new --out (and a new protocol version) instead of mixing results.", flush=True)
        return EXIT_LOCK_MISMATCH

    started = {}

    def tasks():
        for r, s in todo:
            x_o = m.load(r, device)
            seed_gen = (lambda sd: (lambda: torch.Generator().manual_seed(sd)))(derive_seed(s, r["id"]))
            if method == "sage":
                make = lambda log, x_o=x_o, g=seed_gen: make_sage_task(x_o, ctx, g, stop_on_success, log)  # noqa: E731
            elif method == "controls":
                make = lambda log, x_o=x_o, g=seed_gen: controls_task(x_o, ctx, g, stop_on_success=stop_on_success, log=log)  # noqa: E731
            else:
                make = lambda log, x_o=x_o, g=seed_gen: make_baseline_task(method, x_o, ctx, g, stop_on_success, log)  # noqa: E731
            started[(r["id"], s)] = time.time()
            yield (r["id"], s), resumable(make, qlog_path(out_dir, r["id"], s))

    img_dir = out_dir / "images"
    saved = {k: len(list(img_dir.glob(f"{k}_*.npz"))) if img_dir.exists() else 0 for k in ("succ", "fail")}
    fh = open(out_file, "a")

    def on_done(key, result):
        image_id, seed = key
        row = m.by_id[image_id]
        base = dict(id=image_id, cls=row["cls"], seed=seed, experiment=args.experiment,
                    detector=args.detector, wall_seconds=time.time() - started.pop(key))
        parts = list(result.items()) if method == "controls" else [(method, result)]
        xs = [p.pop("_x", None) for _, p in parts]
        first = parts[0][1]
        kind = "succ" if first.get("success") else "fail"
        if saved[kind] < args.save_images and xs[0] is not None:
            img_dir.mkdir(exist_ok=True)
            np.savez_compressed(img_dir / f"{kind}_{image_id}_seed{seed}.npz", x_o=m.load(row).numpy(),
                                **{f"x_{name}": x.cpu().numpy() for (name, _), x in zip(parts, xs) if x is not None})
            saved[kind] += 1
        recs = [dict(base, method=(f"control_{name}" if method == "controls" else name), **res) for name, res in parts]
        append_records(fh, recs)
        if not args.keep_qlogs:
            qlog_path(out_dir, image_id, seed).unlink(missing_ok=True)
        print(f"[{time.strftime('%H:%M:%S')}] {image_id} seed={seed} "
              + " ".join(f"{r['method']}:{'OK' if r['success'] else 'FAIL'}/q={r['queries']}" for r in recs), flush=True)

    t0 = time.time()
    stats = run_tasks(tasks(), fixed_batch(detector.predict_real, max_batch), max_batch=max_batch,
                      concurrency=conc, on_done=on_done, deadline=deadline)
    fh.close()
    remaining = len(todo) - stats["completed"]
    session = dict(run_metadata(cfg), host=socket.gethostname(), shard=args.shard, num_shards=args.num_shards,
                   concurrency=conc, detector_batch=max_batch, seconds=time.time() - t0,
                   detector_calls=stats["detector_calls"], detector_images=stats["images"],
                   runs_completed=stats["completed"], runs_remaining=remaining,
                   interrupted=stats["interrupted"], argv=sys.argv)
    session["images_per_second"] = stats["images"] / max(session["seconds"], 1e-9)
    with open(out_dir / "sessions.jsonl", "a") as f:
        f.write(json.dumps(session, default=str) + "\n")
    print(json.dumps({k: session[k] for k in ("seconds", "detector_calls", "detector_images", "images_per_second",
                                              "runs_completed", "runs_remaining")}), flush=True)
    return EXIT_INCOMPLETE if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
