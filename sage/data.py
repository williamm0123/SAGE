"""Manifest access (see scripts/data/build_manifest.py)."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Manifest:
    def __init__(self, root):
        self.root = Path(root)
        with open(self.root / "manifest.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))
        for r in self.rows:
            r["label"] = int(r["label"])
            r["in_ablation"] = r["in_ablation"] == "1"
            r["in_pilot"] = r["in_pilot"] == "1"
            r["ref_order"] = int(r["ref_order"]) if r["ref_order"] != "" else None
        self.by_id = {r["id"]: r for r in self.rows}
        self.sha256 = hashlib.sha256((self.root / "manifest.csv").read_bytes()).hexdigest()

    def select(self, subset=None, classes=None, ablation=None, pilot=None):
        out = []
        for r in self.rows:
            if subset and r["subset"] != subset:
                continue
            if classes and r["cls"] not in classes:
                continue
            if ablation is not None and r["in_ablation"] != ablation:
                continue
            if pilot is not None and r["in_pilot"] != pilot:
                continue
            out.append(r)
        return out

    def references(self):
        refs = [r for r in self.rows if r["ref_order"] is not None]
        return sorted(refs, key=lambda r: r["ref_order"])

    def load(self, row, device="cpu"):
        with Image.open(self.root / row["proc_path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
