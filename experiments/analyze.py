"""Aggregate attack logs into the paper's tables and the ASR-versus-query figure (Sec. 4.5).

    python experiments/analyze.py --results results --out results/analysis

Definitions
  eligible set I_D     generated test images classified Fake by detector D (clean screening)
  ASR@Q                fraction of runs with a verified Real candidate with LPIPS <= delta_p
                       within Q attack-stage queries (first success carried forward)
  penalized queries    first-success query, failures replaced by Q_max + 1
  CIs                  10,000 generator-stratified image bootstrap replicates (seed 20261010);
                       all seeds of an image are kept together
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.config import load_config  # noqa: E402

METHODS = ["ba", "hsja", "qeba", "sage"]
CONTROLS = ["control_uniform", "control_lowfreq", "control_learned"]


def load_runs(root, experiment):
    runs = []
    for f in sorted((Path(root) / experiment).glob("*/*/*.jsonl")):
        runs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    # de-duplicate (a resumed shard never rewrites, but guard against copied files)
    uniq = {}
    for r in runs:
        uniq[(r["detector"], r["method"], r["id"], r["seed"])] = r
    return list(uniq.values())


def eligible(root, detector):
    p = Path(root) / "screening" / f"{detector}.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return {r["id"]: r["cls"] for r in csv.DictReader(f)
                if r["subset"] == "test" and r["cls"] != "real" and r["pred_real"] == "0"}


def per_image(runs, qmax, checkpoints):
    """Average over seeds within each image -> dict id -> metrics, plus generator label."""
    by_img = defaultdict(list)
    for r in runs:
        by_img[r["id"]].append(r)
    out = {}
    for i, rs in by_img.items():
        fs = [r["first_success"] for r in rs]
        succ = [f is not None for f in fs]
        rec = dict(cls=rs[0]["cls"], n_seeds=len(rs),
                   penalized=np.mean([f if f is not None else qmax + 1 for f in fs]),
                   init_failure=np.mean([bool(r.get("init_failure")) for r in rs]))
        for q in checkpoints:
            rec[f"asr@{q}"] = np.mean([f is not None and f <= q for f in fs])
        good = [r["best_feasible"] for r, s in zip(rs, succ) if s and r.get("best_feasible")]
        rec["lpips"] = np.mean([g["lpips"] for g in good]) if good else np.nan
        rec["psnr"] = np.mean([g["psnr"] for g in good]) if good else np.nan
        rec["succ_queries"] = [f for f in fs if f is not None]
        if "asa_success" in rs[0]:
            rec["asa_sr"] = np.mean([bool(r.get("asa_success")) for r in rs])
            rec["asa_flip"] = np.mean([bool(r.get("asa_real")) for r in rs])
        out[i] = rec
    return out


def bootstrap_ci(img, key, reps, seed, ci=0.95):
    ids = [i for i in img if not np.isnan(img[i][key])]
    if not ids:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for i in ids:
        strata[img[i]["cls"]].append(img[i][key])
    strata = [np.array(v) for v in strata.values()]
    stats = np.empty(reps)
    for b in range(reps):
        vals = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in strata])
        stats[b] = vals.mean()
    a = (1 - ci) / 2
    return tuple(np.quantile(stats, [a, 1 - a]))


def summarize(img, cfg, label):
    if not img:
        return None
    qmax = cfg["budget"]["max_queries"]
    reps, seed = cfg["bootstrap"]["replicates"], cfg["seeds"]["bootstrap"]
    row = dict(label=label, n_images=len(img), n_runs=sum(v["n_seeds"] for v in img.values()))
    for key in [f"asr@{q}" for q in cfg["budget"]["checkpoints"]] + ["penalized", "lpips", "psnr", "init_failure",
                                                                      "asa_sr", "asa_flip"]:
        vals = [v[key] for v in img.values() if key in v and not np.isnan(v[key])]
        if not vals:
            continue
        row[key] = float(np.mean(vals))
        if key.startswith("asr@") or key in ("penalized", "asa_sr"):
            row[key + "_ci"] = [float(c) for c in bootstrap_ci(img, key, reps, seed)]
    sq = [q for v in img.values() for q in v["succ_queries"]]
    row["median_queries_successful"] = float(np.median(sq)) if sq else None
    return row


def fmt(row, key, pct=True):
    if row is None or key not in row:
        return "-"
    v = row[key]
    s = f"{100 * v:.1f}" if pct else f"{v:.0f}" if key == "penalized" else f"{v:.3f}"
    if key + "_ci" in row:
        lo, hi = row[key + "_ci"]
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]" if pct else f" [{lo:.0f}, {hi:.0f}]"
    return s


def asr_curve(runs, qmax):
    fs = np.array([r["first_success"] if r["first_success"] is not None else qmax + 1 for r in runs])
    q = np.arange(1, qmax + 1)
    return q, (fs[None, :] <= q[:, None]).mean(1) if len(fs) else np.zeros_like(q, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    qmax, ckpts = cfg["budget"]["max_queries"], cfg["budget"]["checkpoints"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    primary = load_runs(args.results, "primary")
    controls = load_runs(args.results, "controls")
    transfer = load_runs(args.results, "transfer")
    all_rows, md = [], ["# SAGE results\n", f"Success = Real prediction and LPIPS <= {cfg['perceptual']['delta_p']}; "
                        f"penalized queries assign {qmax + 1} to failures. 95% CIs: generator-stratified image bootstrap.\n"]

    for det in ("ufd", "dire"):
        elig = eligible(args.results, det)
        md.append(f"\n## Detector: {det}\n")
        if elig is not None:
            md.append(f"Eligible generated test images: {len(elig)} "
                      f"(sdv1 {sum(c == 'sdv1' for c in elig.values())}, sdv2 {sum(c == 'sdv2' for c in elig.values())})\n")
        # ---------------------------------------------------------------- Table 2
        md.append("\n### Primary comparison (Table 2)\n")
        md.append("| Method | Generator | images | ASR@1k (%) | ASR@5k (%) | Penalized queries | Median q (succ.) | LPIPS* | init fail |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        curves = {}
        for meth in METHODS:
            rs = [r for r in primary if r["detector"] == det and r["method"] == meth]
            if not rs:
                continue
            curves[meth] = asr_curve(rs, qmax)
            for gen_label, gens in (("pooled", ("sdv1", "sdv2")), ("sdv1", ("sdv1",)), ("sdv2", ("sdv2",))):
                img = per_image([r for r in rs if r["cls"] in gens], qmax, ckpts)
                row = summarize(img, cfg, f"primary/{det}/{meth}/{gen_label}")
                if row is None:
                    continue
                all_rows.append(row)
                md.append(f"| {meth} | {gen_label} | {row['n_images']} | {fmt(row, 'asr@1000')} | {fmt(row, 'asr@5000')} | "
                          f"{fmt(row, 'penalized', False)} | {row['median_queries_successful'] or '-'} | "
                          f"{row.get('lpips', float('nan')):.4f} | {100 * row.get('init_failure', 0):.1f}% |")
            if meth == "sage":
                row = summarize(per_image(rs, qmax, ckpts), cfg, f"primary/{det}/sage/asa")
                md.append(f"\nASA one-query constrained success (ASA-SR): {fmt(row, 'asa_sr')}%; "
                          f"label flip rate: {100 * row.get('asa_flip', float('nan')):.1f}%\n")

        # ---------------------------------------------------------------- Table 3A
        md.append("\n### Component controls (Table 3, Panel A; 100-image ablation subset)\n")
        md.append("| Variant | images | ASR@1k (%) | ASR@5k (%) | Penalized queries |")
        md.append("|---|---|---|---|---|")
        abl_ids = {r["id"] for r in controls if r["detector"] == det}
        sage_primary = [r for r in primary if r["detector"] == det and r["method"] == "sage" and r["id"] in abl_ids]
        for name, rs in [(c, [r for r in controls if r["detector"] == det and r["method"] == c]) for c in CONTROLS] + \
                        [("complete SAGE (primary runs)", sage_primary)]:
            row = summarize(per_image(rs, qmax, ckpts), cfg, f"controls/{det}/{name}") if rs else None
            if row:
                all_rows.append(row)
            md.append(f"| {name} | {row['n_images'] if row else 0} | {fmt(row, 'asr@1000')} | {fmt(row, 'asr@5000')} | "
                      f"{fmt(row, 'penalized', False)} |")

        # ---------------------------------------------------------------- Table 3B
        md.append("\n### SD-v1 -> SD-v2 transfer (Table 3, Panel B; SD-v2 test images)\n")
        md.append("| Method | images | ASR@1k (%) | ASR@5k (%) | Penalized queries |")
        md.append("|---|---|---|---|---|")
        panel = [(m, [r for r in primary if r["detector"] == det and r["method"] == m and r["cls"] == "sdv2"])
                 for m in ("ba", "hsja", "qeba")]
        panel += [("SAGE: mixed-generator prior", [r for r in primary if r["detector"] == det and r["method"] == "sage"
                                                   and r["cls"] == "sdv2"]),
                  ("SAGE: SD-v1-only prior", [r for r in transfer if r["detector"] == det])]
        for name, rs in panel:
            row = summarize(per_image(rs, qmax, ckpts), cfg, f"transfer/{det}/{name}") if rs else None
            if row:
                all_rows.append(row)
            md.append(f"| {name} | {row['n_images'] if row else 0} | {fmt(row, 'asr@1000')} | {fmt(row, 'asr@5000')} | "
                      f"{fmt(row, 'penalized', False)} |")

        if curves:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for meth, (q, a) in curves.items():
                ax.plot(q, 100 * a, label=meth.upper())
            ax.set_xscale("log")
            ax.set_xlabel("attack-stage queries")
            ax.set_ylabel("ASR (%)")
            ax.set_title(f"{det.upper()}: ASR vs queries (LPIPS <= {cfg['perceptual']['delta_p']})")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"asr_vs_queries_{det}.pdf")
            fig.savefig(out / f"asr_vs_queries_{det}.png", dpi=200)
            plt.close(fig)

    (out / "summary.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(all_rows, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
