"""Build the SAGE-v1.0 data manifest (paper Sec. 4.1, Table 1).

Steps: scan -> sha256 -> exact / near-duplicate removal (pHash) -> optional exclusion of
detector-training images -> seeded split (20261008) -> 256x256 canvas (center crop + bicubic,
lossless PNG) -> manifest.csv + manifest_meta.json.

Example:
    python scripts/data/build_manifest.py \
        --real  $DATA_ROOT/raw/DiffusionForensics/images/test/lsun_bedroom/real \
        --sdv1  $DATA_ROOT/raw/DiffusionForensics/images/test/lsun_bedroom/sdv1_new \
        --sdv2  $DATA_ROOT/raw/DiffusionForensics/images/test/lsun_bedroom/sdv2 \
        --exclude $DATA_ROOT/raw/DiffusionForensics/images/train/lsun_bedroom/real \
        --dataset-version "DiffusionForensics (DIRE, RecDrive download 2026-10-xx)" \
        --out $DATA_ROOT/sage_v1
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CLASSES = ("real", "sdv1", "sdv2")
SUBSETS = ("development", "validation", "test")
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_canvas(img, size):
    """Center-crop to square, then bicubic resize to size x size (same order as DIRE's reshape_image)."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != size:
        img = img.resize((size, size), Image.BICUBIC)
    return img


def phash_int(img):
    return int(str(imagehash.phash(img, hash_size=8)), 16)


def hamming_matrix(a, b):
    """Pairwise Hamming distance between two uint64 arrays of 64-bit pHashes."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POPCOUNT[x.view(np.uint8).reshape(len(a), len(b), 8)].sum(-1)


def scan(folder, label, size):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"no images found under {folder}")
    rows = []
    for i, p in enumerate(files):
        with Image.open(p) as im:
            w, h = im.size
            ph = phash_int(to_canvas(im, size))
        rows.append(dict(cls=label, src_path=str(p), src_sha256=sha256(p), src_w=w, src_h=h, phash=ph))
        if (i + 1) % 500 == 0:
            print(f"  {label}: {i + 1}/{len(files)}", flush=True)
    print(f"{label}: {len(rows)} files", flush=True)
    return rows


def dedup(rows, thr):
    """Union near-duplicate images (pHash Hamming <= thr or identical sha256); keep one per group."""
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_sha = {}
    for i, r in enumerate(rows):
        j = by_sha.setdefault(r["src_sha256"], i)
        if j != i:
            parent[find(i)] = find(j)
    ph = np.array([r["phash"] for r in rows], dtype=np.uint64)
    for s in range(0, n, 1024):
        d = hamming_matrix(ph[s:s + 1024], ph)
        for a, b in zip(*np.nonzero(d <= thr)):
            if s + a < b:
                parent[find(s + a)] = find(b)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    keep, removed = [], []
    for members in groups.values():
        classes = {rows[i]["cls"] for i in members}
        if len(classes) > 1:  # same content labelled both real and fake: drop all, log it
            removed += [dict(rows[i], reason="cross_class_duplicate") for i in members]
            continue
        members.sort(key=lambda i: rows[i]["src_path"])
        keep.append(members[0])
        removed += [dict(rows[i], reason=f"duplicate_of:{rows[members[0]]['src_path']}") for i in members[1:]]
    return [rows[i] for i in sorted(keep)], removed


def exclude(rows, folders, thr, size):
    if not folders:
        return rows, []
    ex = []
    for f in folders:
        ex += scan(f, "exclude", size)
    ex_sha = {r["src_sha256"] for r in ex}
    ex_ph = np.array([r["phash"] for r in ex], dtype=np.uint64)
    keep, removed = [], []
    for s in range(0, len(rows), 256):
        chunk = rows[s:s + 256]
        ph = np.array([r["phash"] for r in chunk], dtype=np.uint64)
        near = np.zeros(len(chunk), bool)
        for t in range(0, len(ex_ph), 4096):
            near |= (hamming_matrix(ph, ex_ph[t:t + 4096]) <= thr).any(1)
        for r, nr in zip(chunk, near):
            if r["src_sha256"] in ex_sha or nr:
                removed.append(dict(r, reason="overlaps_excluded_set"))
            else:
                keep.append(r)
    return keep, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--sdv1", required=True)
    ap.add_argument("--sdv2", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="folders whose images must not appear (e.g. detector training data)")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/sage_v1.0.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-version", required=True, help="free text: source, link, download date")
    ap.add_argument("--phash-thr", type=int, default=4, help="near-duplicate Hamming threshold (64-bit pHash)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    size = cfg["canvas"]["size"]
    part = cfg["data"]["partition"]
    seeds = cfg["seeds"]
    out = Path(args.out)
    if (out / "manifest.csv").exists():
        sys.exit(f"{out}/manifest.csv exists - the split is frozen; use a new --out to rebuild")

    rows = []
    for cls, folder in (("real", args.real), ("sdv1", args.sdv1), ("sdv2", args.sdv2)):
        rows += scan(folder, cls, size)
    rows, removed_dup = dedup(rows, args.phash_thr)
    rows, removed_ex = exclude(rows, args.exclude, args.phash_thr, size)

    # Seeded split per class (paper: split seed 20261008)
    rng = np.random.default_rng(seeds["split"])
    by_cls = {c: sorted((r for r in rows if r["cls"] == c), key=lambda r: r["src_sha256"]) for c in CLASSES}
    selected = []
    for c in CLASSES:
        need = sum(part[s][c] for s in SUBSETS)
        pool = by_cls[c]
        if len(pool) < need:
            sys.exit(f"class {c}: need {need} images after dedup/exclusion, only {len(pool)} available")
        order = rng.permutation(len(pool))
        k = 0
        for s in SUBSETS:
            for idx in order[k:k + part[s][c]]:
                selected.append(dict(pool[idx], subset=s))
            k += part[s][c]

    # Reference pool: development real, one seeded order shared by every method
    ref_rng = np.random.default_rng([seeds["split"], 1])
    dev_real = sorted((r for r in selected if r["subset"] == "development" and r["cls"] == "real"),
                      key=lambda r: r["src_sha256"])
    for rank, i in enumerate(ref_rng.permutation(len(dev_real))):
        dev_real[i]["ref_order"] = rank

    # Ablation subset (test, 50/generator, seed 20261009) and pilot subset (validation, 20/generator)
    abl_rng = np.random.default_rng(seeds["ablation_selection"])
    pil_rng = np.random.default_rng([seeds["split"], 2])
    for flag, subset, counts, r_ in (("in_ablation", "test", cfg["data"]["ablation_subset"], abl_rng),
                                     ("in_pilot", "validation", cfg["data"]["pilot_subset"], pil_rng)):
        for c, n in counts.items():
            cand = sorted((r for r in selected if r["subset"] == subset and r["cls"] == c),
                          key=lambda r: r["src_sha256"])
            for i in r_.choice(len(cand), size=n, replace=False):
                cand[i][flag] = 1

    # Write 256x256 lossless canvases
    for r in selected:
        r["id"] = f"{r['cls']}_{r['src_sha256'][:16]}"
        dst = out / "images" / r["subset"] / r["cls"] / f"{r['id']}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(r["src_path"]) as im:
            to_canvas(im, size).save(dst, format="PNG", optimize=False)
        r["proc_path"] = str(dst.relative_to(out))
        r["proc_sha256"] = sha256(dst)
        r["label"] = 0 if r["cls"] == "real" else 1

    cols = ["id", "cls", "label", "subset", "ref_order", "in_ablation", "in_pilot",
            "proc_path", "proc_sha256", "src_path", "src_sha256", "src_w", "src_h", "phash"]
    selected.sort(key=lambda r: (SUBSETS.index(r["subset"]), CLASSES.index(r["cls"]), r["id"]))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in selected:
            w.writerow({**r, "phash": f"{r['phash']:016x}", "ref_order": r.get("ref_order", ""),
                        "in_ablation": r.get("in_ablation", 0), "in_pilot": r.get("in_pilot", 0)})
    with open(out / "removed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cls", "src_path", "src_sha256", "reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(removed_dup + removed_ex)

    counts = {s: {c: sum(1 for r in selected if r["subset"] == s and r["cls"] == c) for c in CLASSES} for s in SUBSETS}
    meta = dict(
        protocol=cfg["protocol_version"], created=datetime.now(timezone.utc).isoformat(),
        dataset_version=args.dataset_version,
        sources={"real": args.real, "sdv1": args.sdv1, "sdv2": args.sdv2, "exclude": args.exclude},
        canvas=cfg["canvas"], resize_impl=f"PIL {Image.__version__} BICUBIC after center crop",
        seeds=seeds, phash_threshold=args.phash_thr, counts=counts,
        removed={"duplicates": len(removed_dup), "excluded_overlap": len(removed_ex)},
        unused_after_split={c: len(by_cls[c]) - sum(counts[s][c] for s in SUBSETS) for c in CLASSES},
        manifest_sha256=sha256(out / "manifest.csv"),
        python=platform.python_version(), numpy=np.__version__,
    )
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["counts"], indent=2))
    print(f"removed: {meta['removed']} | manifest: {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
