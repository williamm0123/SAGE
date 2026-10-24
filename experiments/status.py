"""Completeness, throughput and remaining-time estimate for every result directory.

    python experiments/status.py --data $SAGE_DATA --results results                 # primary/controls/transfer
    python experiments/status.py --data $SAGE_DATA --results results --experiments pilot
    python experiments/status.py ... --require-complete --quiet    # exit 1 unless everything is finished

A run is complete when its record is stored (controls: all three variants). Expected runs are the
eligible targets (clean screening) x the seeds recorded in the directory's LOCK.json.

Throughput is measured per session (process wall time and images actually sent to the detector), not
from per-run wall_seconds, which overlap between concurrently scheduled attacks. Remaining GPU hours are
estimated as remaining runs x mean queries per finished run / detector images per second, using the
directory's own sessions when available and the pilot of the same detector and method otherwise. Replayed
queries cost no detector time, so the estimate is conservative for resumed runs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.config import load_config  # noqa: E402
from sage.data import Manifest  # noqa: E402
from sage.experiment import (DETECTORS, done_runs, eligible_ids, expected_runs, load_dir, method_dir, methods_of,  # noqa: E402
                             screening_paths)


def _sessions(od):
    p = od / "sessions.jsonl"
    if not p.exists():
        return 0.0, 0
    ss = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return sum(s["seconds"] for s in ss), sum(s["detector_images"] for s in ss)


def _dirs(experiment):
    return list(dict.fromkeys(method_dir(experiment, mm) for mm in methods_of(experiment)))


def collect_status(results, m, cfg, experiments=("primary", "controls", "transfer"), detectors=DETECTORS):
    results = Path(results)
    rows = []
    for exp in experiments:
        for det in detectors:
            csv_path, _ = screening_paths(results, det)
            elig = eligible_ids(csv_path) if csv_path.exists() else None
            for d in _dirs(exp):
                od = results / exp / det / d
                lock = json.loads((od / "LOCK.json").read_text()) if (od / "LOCK.json").exists() else None
                seeds = lock["seeds"] if lock else ([0] if exp == "pilot" else cfg["seeds"]["attack"])
                row = dict(experiment=exp, detector=det, method=d, dir=str(od), locked=lock is not None)
                if elig is None:
                    row.update(status="not screened", expected=None, done=0)
                    rows.append(row)
                    continue
                want = {(r["id"], s) for r, s in expected_runs(m, exp, elig, seeds)}
                recs = load_dir(od) if od.exists() else []
                done = done_runs(recs, exp)
                qs = {}
                for r in recs:
                    qs[(r["id"], r["seed"])] = qs.get((r["id"], r["seed"]), 0) + r["queries"]
                seconds, images = _sessions(od)
                row.update(expected=len(want), done=len(done & want), unexpected=len(done - want),
                           partial=len(list((od / "qlog").glob("*.qlog"))) if (od / "qlog").exists() else 0,
                           mean_queries=(sum(qs[k] for k in done) / len(done)) if done else None,
                           session_hours=seconds / 3600, images_per_second=images / seconds if seconds else None)
                row["status"] = ("complete" if row["done"] == row["expected"] and not row["unexpected"]
                                 else "inconsistent" if row["unexpected"] else "incomplete")
                rows.append(row)
    return rows


def add_estimates(rows, pilot_rows):
    pilot = {(r["detector"], r["method"]): r for r in pilot_rows}
    for r in rows:
        if r.get("expected") is None:
            continue
        remaining = r["expected"] - r["done"]
        q, ips = r.get("mean_queries"), r.get("images_per_second")
        src = "own"
        if q is None or ips is None:
            pm = "sage" if r["method"] in ("controls", "sage") else r["method"]
            p = pilot.get((r["detector"], pm))
            if p and p.get("mean_queries") and p.get("images_per_second"):
                q = q or p["mean_queries"] * (3 if r["method"] == "controls" else 1)
                ips = ips or p["images_per_second"]
                src = "pilot"
        r["est_gpu_hours_remaining"] = (remaining * q / ips / 3600) if (q and ips) else None
        r["estimate_from"] = src if r["est_gpu_hours_remaining"] is not None else None
    return rows


def format_rows(rows):
    out = ["| experiment | detector | method | done / expected | partial | status | mean q | img/s | "
           "session h | est. GPU-h left |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
    for r in rows:
        out.append(f"| {r['experiment']} | {r['detector']} | {r['method']} | {r['done']} / {r['expected']} | "
                   f"{r.get('partial', 0)} | {r['status']} | {f(r.get('mean_queries'), '.0f')} | "
                   f"{f(r.get('images_per_second'), '.1f')} | {f(r.get('session_hours'), '.2f')} | "
                   f"{f(r.get('est_gpu_hours_remaining'), '.1f')} {r.get('estimate_from') or ''} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--experiments", nargs="+", default=["primary", "controls", "transfer"])
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", default=None, help="also write the rows to this file")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = Manifest(args.data)
    rows = collect_status(args.results, m, cfg, args.experiments, args.detectors)
    pilot = collect_status(args.results, m, cfg, ["pilot"], args.detectors) if "pilot" not in args.experiments else rows
    add_estimates(rows, pilot)
    if not args.quiet:
        print(format_rows(rows))
        total = [r["est_gpu_hours_remaining"] for r in rows if r.get("est_gpu_hours_remaining") is not None]
        if total:
            print(f"\nestimated single-GPU hours remaining (listed rows): {sum(total):.1f} "
                  f"= {sum(total) / 72:.1f} jobs of 3 days")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    if args.require_complete and any(r["status"] != "complete" for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
