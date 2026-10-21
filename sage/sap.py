"""Spectral Artifact Profiling (paper Eq. 3): mean real / generated power spectra and their difference."""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import torch

from .spectral import power_spectrum


class SpectralPrior:
    def __init__(self, P_r, P_f, meta=None):
        self.P_r = P_r            # (3, H, W) float64
        self.P_f = P_f
        self.delta = P_r - P_f
        self.meta = meta or {}

    @property
    def sha256(self):
        """Content hash of the two mean spectra (independent of file compression and metadata)."""
        h = hashlib.sha256()
        for P in (self.P_r, self.P_f):
            h.update(P.detach().to("cpu", torch.float64).contiguous().numpy().tobytes())
        return h.hexdigest()

    def to(self, device):
        return SpectralPrior(self.P_r.to(device), self.P_f.to(device), self.meta)

    def save(self, path):
        np.savez_compressed(path, P_r=self.P_r.cpu().numpy(), P_f=self.P_f.cpu().numpy(),
                            delta=self.delta.cpu().numpy(), meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path, device="cpu"):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(torch.from_numpy(z["P_r"]).to(device), torch.from_numpy(z["P_f"]).to(device), meta)


@torch.no_grad()
def mean_power(images, device, batch_size=64):
    """images: iterable of (3, H, W) float tensors. Returns the float64 mean power spectrum and count."""
    total, n, batch = None, 0, []

    def flush():
        nonlocal total, n
        x = torch.stack(batch).to(device)
        p = power_spectrum(x).sum(0)
        total = p if total is None else total + p
        n += len(batch)
        batch.clear()

    for x in images:
        batch.append(x)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return total / n, n


def build_prior(real_items, fake_items, load_fn, device, description):
    """real_items / fake_items: manifest rows; load_fn(row) -> (3, H, W) tensor in [0, 1]."""
    if {r["id"] for r in real_items} & {r["id"] for r in fake_items}:
        raise ValueError("real and generated profiling sets must be disjoint")
    P_r, n_r = mean_power((load_fn(r) for r in real_items), device)
    P_f, n_f = mean_power((load_fn(r) for r in fake_items), device)
    ids = sorted(r["id"] for r in real_items) + sorted(r["id"] for r in fake_items)
    meta = dict(
        description=description, created=datetime.now(timezone.utc).isoformat(),
        n_real=n_r, n_fake=n_f, fake_classes=sorted({r["cls"] for r in fake_items}),
        ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        fft="unitary fft2 (norm=ortho), unshifted, per-RGB-channel power, float64",
    )
    return SpectralPrior(P_r, P_f, meta)
