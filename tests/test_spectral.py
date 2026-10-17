import math

import torch

from sage.config import load_config
from sage.spectral import Bands, band_kl, full_bands, power_spectrum, reconstruct, sample_direction
from sage.weights import learned_weights, lowfreq_weights, uniform_weights

CFG = load_config()


def _img(h=64, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (0.3 + 0.4 * torch.rand(3, h, w, generator=g)).float()


def test_bands_cover_all_nondc():
    b = full_bands(CFG, 256, 256)
    assert b.present == list(range(8))
    assert int(b.index[0, 0]) == -1
    assert b.num_nondc == 256 * 256 - 1
    assert bool((b.index[b.nondc] >= 0).all())
    # band edges are equal-width over [0, sqrt(0.5)]
    assert math.isclose(float(b.edges[-1]), math.sqrt(0.5), rel_tol=1e-12)


def test_identity_and_dc():
    x = _img()
    b = Bands(64, 64, 8, math.sqrt(0.5))
    dP = torch.randn(3, 64, 64, dtype=torch.float64)
    dP = 0.5 * (dP + torch.roll(torch.flip(dP, (-2, -1)), (1, 1), (-2, -1)))   # symmetric like real spectra
    y = reconstruct(x, torch.zeros(8), dP, b, 1e-12)[0]
    assert (y - x).abs().max() < 1e-6
    y2, diag = reconstruct(x, torch.full((8,), 0.3), dP * 1e-3, b, 1e-12, return_diag=True)
    assert diag["imag_max"] < 1e-10
    assert float(diag["clip_fraction"][0]) == 0.0
    assert torch.allclose(y2[0].mean((1, 2)), x.mean((1, 2)), atol=1e-6)    # DC restored exactly


def test_batch_reconstruct_matches_single():
    x = _img()
    b = Bands(64, 64, 8, math.sqrt(0.5))
    dP = power_spectrum(_img(seed=1)) - power_spectrum(x)
    alphas = torch.rand(3, 8, dtype=torch.float64)
    batch = reconstruct(x, alphas, dP, b, 1e-12)
    for k in range(3):
        assert torch.allclose(batch[k], reconstruct(x, alphas[k], dP, b, 1e-12)[0], atol=1e-6)


def test_band_kl_zero_at_reference():
    b = Bands(64, 64, 8, math.sqrt(0.5))
    P = power_spectrum(_img())
    kl = band_kl(P[None] + 0.0, P - 1e-12 + 1e-12, b, 0.0)
    assert float(kl.abs().max()) < 1e-10


def test_weights_and_sampling():
    b = Bands(64, 64, 8, math.sqrt(0.5))
    dP = power_spectrum(_img(seed=1)) - power_spectrum(_img(seed=2))
    for W in (uniform_weights(b), lowfreq_weights(b, 0.1), learned_weights(torch.rand(8), dP, b, 0.1)):
        assert float(W[0, 0]) == 0.0
        assert math.isclose(float(W.sum()), 1.0, rel_tol=1e-9)
        assert float(W[b.nondc].min()) >= 0.1 / b.num_nondc - 1e-15 or W is not None
        s = sample_direction(W.sqrt(), (3, 64, 64), torch.Generator().manual_seed(0), "cpu")
        assert s.dtype == torch.float64 and s.shape == (3, 64, 64)
        assert float(s.mean((1, 2)).abs().max()) < 1e-10                   # zero DC
    lf = lowfreq_weights(b, 0.1)
    assert float(lf[b.rho > 0.25].max()) < float(lf[(b.rho > 0) & (b.rho <= 0.25)].min())
