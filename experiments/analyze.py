"""Aggregate attack logs into the paper's tables and figures (Sec. 4.3, 4.5).

    python experiments/analyze.py --data $SAGE_DATA --results results --out results/analysis

Before anything is computed, every result directory is checked:
  * complete: all eligible targets x locked seeds present (controls: all three variants); otherwise the
    script stops (exit 2). --allow-incomplete produces a draft in which every table states done/expected;
    missing runs are neither dropped silently nor counted as failures - they are simply absent and flagged.
  * consistent: one manifest, config, code and attack-prior per experiment family; screening per detector.

Definitions
  eligible set I_D     generated test images classified Fake by detector D (clean screening)
  ASR@Q                fraction of runs with a verified Real candidate with LPIPS <= delta_p
                       within Q attack-stage queries (first success carried forward)
  penalized queries    first-success query, failures replaced by Q_max + 1
  CIs                  10,000 generator-stratified image bootstrap replicates (seed 20261010);
                       all seeds of an image are kept together; paired differences resample images
                       jointly for both methods
  common-success       (image, seed) runs in which every compared method succeeded
  quality              successes: LPIPS / PSNR / RMSE of the reported candidate; failures: reported
                       separately via the best real-classified diagnostic candidate (if any)
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.config import load_config  # noqa: E402
from sage.data import Manifest  # noqa: E402
from sage.experiment import (CONTROL_METHODS, PRIMARY_METHODS, eligible_ids, expected_runs, load_dir,  # noqa: E402
                             screening_paths)
from status import collect_status, format_rows  # noqa: E402

DETECTORS = ("ufd", "dire")


# ------------------------------------------------------------------------------------ loading
def load_experiment(results, m, cfg, experiment, det, eligible):
    """Records of one experiment/detector, restricted to the expected runs; dict method -> list."""
    out = defaultdict(list)
    base = Path(results) / experiment / det
    for d in sorted(p for p in base.glob("*") if p.is_dir() and "_" not in p.name):
        lock = d / "LOCK.json"
        seeds = json.loads(lock.read_text())["seeds"] if lock.exists() else cfg["seeds"]["attack"]
        want = {(r["id"], s) for r, s in expected_runs(m, experiment, eligible, seeds)}
        uniq = {}
        for r in load_dir(d):
            if (r["id"], r["seed"]) in want:
                uniq[(r["method"], r["id"], r["seed"])] = r
        for (meth, _, _), r in sorted(uniq.items()):
            out[meth].append(r)
    return out


def check_locks(results):
    """Same manifest / config / code everywhere; same screening per detector; mixed prior for primary+controls."""
    locks = {p.parent: json.loads(p.read_text()) for p in Path(results).glob("*/*/*/LOCK.json")
             if p.parts[-4] in ("primary", "controls", "transfer") and "_" not in p.parts[-2]}
    problems = []
    for key in ("manifest_sha256", "config_sha256", "code_sha256", "protocol"):
        vals = {str(v.get(key)) for v in locks.values()}
        if len(vals) > 1:
            problems.append(f"{key} differs between result directories: {sorted(vals)}")
    for det in DETECTORS:
        vals = {(v["screening_sha256"], json.dumps(v["detector_files"], sort_keys=True), v["detector_batch"])
                for d, v in locks.items() if d.parts[-2] == det}
        if len(vals) > 1:
            problems.append(f"{det}: screening / detector weights / batch differ between directories")
    mixed = {v["prior_sha256"] for d, v in locks.items() if d.parts[-3] in ("primary", "controls")
             and v["method"] in ("sage", "controls")}
    if len(mixed) > 1:
        problems.append("primary and controls used different SAP priors")
    for d, v in locks.items():
        if d.parts[-3] == "transfer" and "sdv2" in str(v.get("prior_description")):
            problems.append(f"{d}: transfer prior includes SD-v2 ({v.get('prior_description')})")
        if d.parts[-3] in ("primary", "controls") and v["method"] in ("sage", "controls") and \
                "sdv2" not in str(v.get("prior_description")):
            problems.append(f"{d}: expected the mixed prior, got {v.get('prior_description')}")
    return locks, problems


# ------------------------------------------------------------------------------------ metrics
def per_image(runs, qmax, checkpoints):
    """Average over seeds within each image -> dict id -> metrics."""
    by_img = defaultdict(list)
    for r in runs:
        by_img[r["id"]].append(r)
    out = {}
    for i, rs in by_img.items():
        fs = [r["first_success"] for r in rs]
        rec = dict(cls=rs[0]["cls"], n_seeds=len(rs),
                   penalized=np.mean([f if f is not None else qmax + 1 for f in fs]),
                   fail=np.mean([f is None for f in fs]),
                   init_failure=np.mean([bool(r.get("init_failure")) for r in rs]))
        for q in checkpoints:
            rec[f"asr@{q}"] = np.mean([f is not None and f <= q for f in fs])
        good = [r["best_feasible"] for r in rs if r["first_success"] is not None and r.get("best_feasible")]
        for k in ("lpips", "psnr", "rmse"):
            rec[k] = np.mean([g[k] for g in good]) if good else np.nan
        bad = [r.get("best_real") for r in rs if r["first_success"] is None]
        rec["fail_lpips"] = np.mean([b["lpips"] for b in bad if b]) if any(bad) else np.nan
        rec["fail_any_real"] = np.mean([b is not None for b in bad]) if bad else np.nan
        rec["succ_queries"] = [f for f in fs if f is not None]
        rec["after_success"] = np.mean([r.get("queries_after_success", 0) for r in rs])
        if "asa" in rs[0] and "asa_success" in rs[0]:
            rec["asa_sr"] = np.mean([bool(r.get("asa_success")) for r in rs])
            rec["asa_flip"] = np.mean([bool(r.get("asa_real")) for r in rs])
        out[i] = rec
    return out


def bootstrap_ci(img, key, reps, seed, ci=0.95):
    ids = [i for i in img if key in img[i] and not np.isnan(img[i][key])]
    if not ids:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for i in ids:
        strata[img[i]["cls"]].append(img[i][key])
    strata = [np.array(v) for v in strata.values()]
    stats = np.empty(reps)
    for b in range(reps):
        stats[b] = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in strata]).mean()
    a = (1 - ci) / 2
    return tuple(float(x) for x in np.quantile(stats, [a, 1 - a]))


def summarize(img, cfg, label, expected_images=None):
    if not img:
        return None
    reps, seed = cfg["bootstrap"]["replicates"], cfg["seeds"]["bootstrap"]
    row = dict(label=label, n_images=len(img), n_runs=int(sum(v["n_seeds"] for v in img.values())),
               expected_images=expected_images)
    for key in [f"asr@{q}" for q in cfg["budget"]["checkpoints"]] + \
               ["penalized", "fail", "init_failure", "lpips", "psnr", "rmse", "fail_lpips", "fail_any_real",
                "after_success", "asa_sr", "asa_flip"]:
        vals = [v[key] for v in img.values() if key in v and not np.isnan(v[key])]
        if not vals:
            continue
        row[key] = float(np.mean(vals))
        if key.startswith("asr@") or key in ("penalized", "asa_sr"):
            row[key + "_ci"] = list(bootstrap_ci(img, key, reps, seed))
    sq = [q for v in img.values() for q in v["succ_queries"]]
    row["median_queries_successful"] = float(np.median(sq)) if sq else None
    return row


def paired(img_a, img_b, cfg, keys, label):
    """Mean difference a - b over common images with a paired generator-stratified bootstrap CI."""
    common = sorted(set(img_a) & set(img_b))
    if not common:
        return None
    reps, seed = cfg["bootstrap"]["replicates"], cfg["seeds"]["bootstrap"]
    row = dict(label=label, n_images=len(common))
    for k in keys:
        diff = {i: {"cls": img_a[i]["cls"], k: img_a[i][k] - img_b[i][k]} for i in common}
        row[k] = float(np.mean([d[k] for d in diff.values()]))
        row[k + "_ci"] = list(bootstrap_ci(diff, k, reps, seed))
    return row


def common_success(groups):
    """groups: dict name -> runs. Per-method stats on runs (image, seed) where all methods succeeded."""
    succ = {n: {(r["id"], r["seed"]): r for r in rs if r["first_success"] is not None} for n, rs in groups.items()}
    keys = set.intersection(*(set(v) for v in succ.values())) if succ else set()
    out = {}
    for n, s in succ.items():
        rs = [s[k] for k in keys]
        out[n] = dict(n_runs=len(rs),
                      mean_queries=float(np.mean([r["first_success"] for r in rs])) if rs else None,
                      median_queries=float(np.median([r["first_success"] for r in rs])) if rs else None,
                      **{k: float(np.mean([r["best_feasible"][k] for r in rs])) if rs else None
                         for k in ("lpips", "psnr", "rmse")})
    return out


# ------------------------------------------------------------------------------------ formatting
def pct(row, key):
    if row is None or key not in row:
        return "-"
    s = f"{100 * row[key]:.1f}"
    if key + "_ci" in row:
        lo, hi = row[key + "_ci"]
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


def num(row, key, fmt=".0f"):
    if row is None or key not in row or row[key] is None:
        return "-"
    s = format(row[key], fmt)
    if key + "_ci" in row:
        lo, hi = row[key + "_ci"]
        s += f" [{format(lo, fmt)}, {format(hi, fmt)}]"
    return s


def n_of(row):
    if row is None:
        return "0"
    e = row.get("expected_images")
    return f"{row['n_images']}" + (f" / {e} INCOMPLETE" if e is not None and row["n_images"] < e else "")


def table(md, header, rows):
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "---|" * len(header))
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def asr_curve(runs, qmax):
    fs = np.array([r["first_success"] if r["first_success"] is not None else qmax + 1 for r in runs])
    q = np.arange(1, qmax + 1)
    return q, (fs[None, :] <= q[:, None]).mean(1) if len(fs) else np.zeros_like(q, dtype=float)


def example_figure(img_dir, path, title, per_kind=2):
    import matplotlib.pyplot as plt
    files = [(k, f) for k in ("succ", "fail") for f in sorted(Path(img_dir).glob(f"{k}_*.npz"))[:per_kind]]
    if not files:
        return False
    fig, axes = plt.subplots(len(files), 3, figsize=(7.5, 2.6 * len(files)), squeeze=False)
    for row, (kind, f) in zip(axes, files):
        z = np.load(f)
        x_o = z["x_o"]
        x = z[[k for k in z.files if k.startswith("x_") and k != "x_o"][0]]
        res = np.abs(x - x_o)
        for ax, im, t in zip(row, (x_o, x, np.clip(10 * res, 0, 1)),
                             ("source $x_o$", "success" if kind == "succ" else "failure (best real)", "|x - x_o| x10")):
            ax.imshow(np.transpose(im, (1, 2, 0)))
            ax.set_title(t, fontsize=8)
            ax.axis("off")
        row[0].set_ylabel(f.stem, fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="manifest directory (defines the expected runs)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-incomplete", action="store_true", help="draft tables; incomplete rows are flagged")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = Manifest(args.data)
    qmax, ckpts = cfg["budget"]["max_queries"], cfg["budget"]["checkpoints"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    status = collect_status(args.results, m, cfg)
    locks, problems = check_locks(args.results)
    incomplete = [r for r in status if r["status"] != "complete"]
    if problems:
        print("ERROR inconsistent result directories:\n  " + "\n  ".join(problems))
        return 2
    if incomplete and not args.allow_incomplete:
        print(format_rows(status))
        print(f"\nERROR {len(incomplete)} result directories are incomplete; finish them or use --allow-incomplete")
        return 2

    md = ["# SAGE results\n"]
    if incomplete:
        md.append("**DRAFT - INCOMPLETE RESULTS.** Missing runs are absent (not counted as failures); "
                  "every table states images present / expected.\n")
    md.append(f"Success = Real prediction and LPIPS <= {cfg['perceptual']['delta_p']}; penalized queries assign "
              f"{qmax + 1} to failures. 95% CIs: generator-stratified image bootstrap "
              f"({cfg['bootstrap']['replicates']} replicates); paired differences resample images jointly.\n")
    md.append("\n## Run status\n")
    md.append(format_rows(status))
    all_rows = []

    for det in DETECTORS:
        csv_path, summ_path = screening_paths(args.results, det)
        if not csv_path.exists():
            md.append(f"\n## Detector: {det}\n\nnot screened\n")
            continue
        elig = eligible_ids(csv_path)
        md.append(f"\n## Detector: {det}\n")
        if summ_path.exists():
            s = json.loads(summ_path.read_text())["summary"]
            md.append("\n### Clean screening (eligibility; not charged to attack budgets)\n")
            table(md, ["subset/class", "n", "fake-detection rate", "false-positive rate", "eligible"],
                  [(k, v["n"], f"{100 * v['fake_detection_rate']:.1f}%" if "fake_detection_rate" in v else "-",
                    f"{100 * v['false_positive_rate']:.1f}%" if "false_positive_rate" in v else "-",
                    v.get("eligible", "-")) for k, v in s.items()])
        primary = load_experiment(args.results, m, cfg, "primary", det, elig)
        controls = load_experiment(args.results, m, cfg, "controls", det, elig)
        transfer = load_experiment(args.results, m, cfg, "transfer", det, elig)
        exp_ids = {e: {r["id"]: r["cls"] for r in m.select("test", ["sdv1", "sdv2"]) if r["id"] in elig}
                   for e in ("primary",)}
        n_exp = {g: sum(1 for c in exp_ids["primary"].values() if c in gens)
                 for g, gens in (("pooled", ("sdv1", "sdv2")), ("sdv1", ("sdv1",)), ("sdv2", ("sdv2",)))}

        # ------------------------------------------------------------ Table 2
        md.append("\n### Primary comparison (Table 2)\n")
        rows, imgs, curves = [], {}, {}
        for meth in PRIMARY_METHODS:
            rs = primary.get(meth, [])
            if not rs:
                continue
            curves[meth] = asr_curve(rs, qmax)
            for g, gens in (("pooled", ("sdv1", "sdv2")), ("sdv1", ("sdv1",)), ("sdv2", ("sdv2",))):
                img = per_image([r for r in rs if r["cls"] in gens], qmax, ckpts)
                if g == "pooled":
                    imgs[meth] = img
                row = summarize(img, cfg, f"primary/{det}/{meth}/{g}", n_exp[g])
                if row is None:
                    continue
                all_rows.append(row)
                rows.append((meth, g, n_of(row), pct(row, "asr@500"), pct(row, "asr@1000"), pct(row, "asr@2000"),
                             pct(row, "asr@5000"), num(row, "penalized"), num(row, "median_queries_successful"),
                             num(row, "lpips", ".4f"), num(row, "psnr", ".2f"), num(row, "rmse", ".4f"),
                             pct(row, "init_failure")))
        table(md, ["method", "gen", "images", "ASR@500", "ASR@1k", "ASR@2k", "ASR@5k", "penalized q",
                   "median q (succ.)", "LPIPS*", "PSNR*", "RMSE*", "init fail"], rows)
        md.append("\n\\* successful runs only (reported candidate). Failures below.\n")

        md.append("\n#### Failures and charged queries\n")
        table(md, ["method", "fail rate", "failures with a real candidate", "best-real LPIPS (failures)",
                   "queries charged after first success (batch remainder)"],
              [(meth, pct(r, "fail"), pct(r, "fail_any_real"), num(r, "fail_lpips", ".4f"), num(r, "after_success", ".1f"))
               for meth in PRIMARY_METHODS if (r := summarize(imgs.get(meth, {}), cfg, ""))])

        md.append("\n#### Paired differences SAGE - baseline (common images)\n")
        prow = []
        for b in ("ba", "hsja", "qeba"):
            if "sage" in imgs and b in imgs:
                r = paired(imgs["sage"], imgs[b], cfg, ["asr@1000", "asr@5000", "penalized"], f"primary/{det}/sage-{b}")
                all_rows.append(r)
                prow.append((f"SAGE - {b.upper()}", r["n_images"], pct(r, "asr@1000"), pct(r, "asr@5000"),
                             num(r, "penalized")))
        table(md, ["comparison", "images", "dASR@1k (pp)", "dASR@5k (pp)", "d penalized q"], prow)

        cs = common_success({k: v for k, v in primary.items() if k in PRIMARY_METHODS})
        if cs:
            md.append(f"\n#### Common-success subset (runs where all {len(cs)} methods succeeded)\n")
            table(md, ["method", "runs", "mean q", "median q", "LPIPS", "PSNR", "RMSE"],
                  [(k, v["n_runs"], num(v, "mean_queries"), num(v, "median_queries"), num(v, "lpips", ".4f"),
                    num(v, "psnr", ".2f"), num(v, "rmse", ".4f")) for k, v in cs.items()])
            all_rows.append(dict(label=f"primary/{det}/common_success", **cs))

        sage_runs = primary.get("sage", [])
        if sage_runs:
            a = [r["asa"] for r in sage_runs if r.get("asa")]
            row = summarize(imgs["sage"], cfg, f"primary/{det}/sage/asa")
            starts = Counter(r.get("start", "none") for r in sage_runs)
            q = lambda k, f=np.mean: f([x[k] for x in a])  # noqa: E731
            md.append("\n#### ASA diagnostics (SAGE primary runs)\n")
            md.append(f"ASA-SR (Eq. 18, one verification query): {pct(row, 'asa_sr')}%; label flip rate "
                      f"{pct(row, 'asa_flip')}%; refinement start: {dict(starts)}.\n")
            table(md, ["diagnostic", "mean", "median", "max"],
                  [(k, f"{q(k):.4g}", f"{q(k, np.median):.4g}", f"{q(k, np.max):.4g}")
                   for k in ("lpips", "kl_before", "kl_pre_clip", "kl_post_clip", "clip_fraction", "imag_max",
                             "sweeps", "seconds")])
            md.append(f"\nmean alpha per band (low->high): {np.round(np.mean([x['alpha'] for x in a], 0), 3).tolist()}\n")
            all_rows.append(dict(label=f"primary/{det}/asa_diagnostics", starts=dict(starts),
                                 **{k: q(k) for k in ("lpips", "kl_before", "kl_pre_clip", "kl_post_clip",
                                                      "clip_fraction", "seconds")}))

        # ------------------------------------------------------------ Table 3A
        md.append("\n### Component controls (Table 3, Panel A; ablation subset)\n")
        abl = {r["id"] for r in m.select("test", ["sdv1", "sdv2"], ablation=True) if r["id"] in elig}
        cimgs, crow = {}, []
        groups = {c: controls.get(c, []) for c in CONTROL_METHODS}
        groups["complete SAGE"] = [r for r in sage_runs if r["id"] in abl]
        for name, rs in groups.items():
            img = per_image(rs, qmax, ckpts)
            cimgs[name] = img
            row = summarize(img, cfg, f"controls/{det}/{name}", len(abl))
            if row:
                all_rows.append(row)
            crow.append((name, n_of(row), pct(row, "asr@1000"), pct(row, "asr@5000"), num(row, "penalized"),
                         pct(row, "init_failure")))
        table(md, ["variant", "images", "ASR@1k", "ASR@5k", "penalized q", "init fail"], crow)
        prow = []
        for a_, b_ in (("control_learned", "control_uniform"), ("control_learned", "control_lowfreq"),
                       ("complete SAGE", "control_learned")):
            if cimgs.get(a_) and cimgs.get(b_):
                r = paired(cimgs[a_], cimgs[b_], cfg, ["asr@1000", "asr@5000", "penalized"], f"controls/{det}/{a_}-{b_}")
                all_rows.append(r)
                prow.append((f"{a_} - {b_}", r["n_images"], pct(r, "asr@1000"), pct(r, "asr@5000"), num(r, "penalized")))
        if prow:
            md.append("\nPaired differences:\n")
            table(md, ["comparison", "images", "dASR@1k (pp)", "dASR@5k (pp)", "d penalized q"], prow)
        cs = common_success({k: v for k, v in groups.items() if v})
        if len(cs) > 1:
            md.append("\nCommon-success subset:\n")
            table(md, ["variant", "runs", "mean q", "LPIPS", "PSNR", "RMSE"],
                  [(k, v["n_runs"], num(v, "mean_queries"), num(v, "lpips", ".4f"), num(v, "psnr", ".2f"),
                    num(v, "rmse", ".4f")) for k, v in cs.items()])

        # ------------------------------------------------------------ Table 3B
        md.append("\n### SD-v1 -> SD-v2 transfer (Table 3, Panel B; SD-v2 test images)\n")
        n_sdv2 = n_exp["sdv2"]
        panel = {b.upper(): [r for r in primary.get(b, []) if r["cls"] == "sdv2"] for b in ("ba", "hsja", "qeba")}
        panel["SAGE: mixed-generator prior"] = [r for r in sage_runs if r["cls"] == "sdv2"]
        panel["SAGE: SD-v1-only prior"] = transfer.get("sage", [])
        timgs, trow = {}, []
        for name, rs in panel.items():
            timgs[name] = per_image(rs, qmax, ckpts)
            row = summarize(timgs[name], cfg, f"transfer/{det}/{name}", n_sdv2)
            if row:
                all_rows.append(row)
            trow.append((name, n_of(row), pct(row, "asr@1000"), pct(row, "asr@5000"), num(row, "penalized")))
        table(md, ["method", "images", "ASR@1k", "ASR@5k", "penalized q"], trow)
        a_, b_ = "SAGE: SD-v1-only prior", "SAGE: mixed-generator prior"
        if timgs[a_] and timgs[b_]:
            r = paired(timgs[a_], timgs[b_], cfg, ["asr@1000", "asr@5000", "penalized"], f"transfer/{det}/sdv1only-mixed")
            all_rows.append(r)
            md.append(f"\nPaired SD-v1-only - mixed ({r['n_images']} images): dASR@1k {pct(r, 'asr@1000')} pp, "
                      f"dASR@5k {pct(r, 'asr@5000')} pp, d penalized q {num(r, 'penalized')}\n")

        # ------------------------------------------------------------ timing and figures
        md.append("\n### Compute (detector-inference sessions; per-run wall times overlap and are not summed)\n")
        table(md, ["experiment", "method", "session GPU-h", "detector img/s", "mean ASA local s"],
              [(r["experiment"], r["method"], num(r, "session_hours", ".2f"), num(r, "images_per_second", ".1f"),
                f"{np.mean([x['asa']['seconds'] for x in sage_runs if x.get('asa')]):.2f}"
                if r["method"] == "sage" and r["experiment"] == "primary" and sage_runs else "-")
               for r in status if r["detector"] == det])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if curves:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for meth, (qq, a) in curves.items():
                ax.plot(qq, 100 * a, label=meth.upper())
            for c in ckpts:
                ax.axvline(c, color="0.85", lw=0.8, zorder=0)
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
        for meth in PRIMARY_METHODS:
            d = Path(args.results) / "primary" / det / meth / "images"
            if d.exists() and example_figure(d, out / f"examples_{det}_{meth}.png", f"{det.upper()} / {meth.upper()}"):
                md.append(f"\nRepresentative successes / failures: `examples_{det}_{meth}.png`\n")

    md.append("\n## Execution locks\n")
    table(md, ["directory", "code", "config", "manifest", "prior", "detector batch"],
          [(str(d.relative_to(args.results)), v["code_sha256"][:12], v["config_sha256"][:12], v["manifest_sha256"][:12],
            v["prior_sha256"][:12], v["detector_batch"]) for d, v in sorted(locks.items())])
    (out / "summary.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(dict(status=status, rows=[r for r in all_rows if r],
                                                      locks={str(k): v for k, v in locks.items()}), indent=2,
                                                 default=float))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
