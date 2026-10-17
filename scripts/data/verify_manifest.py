"""Verify that every processed image matches its manifest hash and that Table 1 counts hold.

    python scripts/data/verify_manifest.py $DATA_ROOT/sage_v1
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
meta = json.loads((root / "manifest_meta.json").read_text())
rows = list(csv.DictReader(open(root / "manifest.csv")))
bad = []
for r in rows:
    p = root / r["proc_path"]
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != r["proc_sha256"]:
        bad.append(r["proc_path"])

counts = Counter((r["subset"], r["cls"]) for r in rows)
for s, per_cls in meta["counts"].items():
    for c, n in per_cls.items():
        assert counts[(s, c)] == n, (s, c, counts[(s, c)], n)
assert len({r["src_sha256"] for r in rows}) == len(rows), "duplicate source hash in manifest"
assert hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest() == meta["manifest_sha256"], "manifest.csv changed"

print(f"{len(rows)} images | ablation={sum(r['in_ablation'] == '1' for r in rows)} "
      f"pilot={sum(r['in_pilot'] == '1' for r in rows)} "
      f"ref_pool={sum(r['ref_order'] != '' for r in rows)}")
if bad:
    sys.exit(f"{len(bad)} files missing or modified, e.g. {bad[:3]}")
print("manifest OK")
