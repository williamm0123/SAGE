"""Unitary FFT utilities, radial bands and the spectral reconstruction T(x; alpha) (paper Eq. 2, 4-6).

Conventions: arrays stay in the unshifted FFT layout; spectra are computed in float64; images are
(..., 3, H, W) float tensors in [0, 1].
"""
import math

import torch


def fft2(x):
    return torch.fft.fft2(x.to(torch.float64), norm="ortho")


def ifft2(X):
    return torch.fft.ifft2(X, norm="ortho")


def power_spectrum(x):
    return fft2(x).abs().square()


def radial_frequency(h, w, device=None):
    fh = torch.fft.fftfreq(h, device=device, dtype=torch.float64)
    fw = torch.fft.fftfreq(w, device=device, dtype=torch.float64)
    return torch.sqrt(fh[:, None] ** 2 + fw[None, :] ** 2)


def conj_flip(X):
    """Return Y with Y[k] = conj(X[-k mod N]) over the last two dims."""
    return torch.roll(torch.flip(X, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)).conj()


def hermitian(X):
    """Project onto the conjugate-symmetric subspace; self-conjugate coefficients become real."""
    return 0.5 * (X + conj_flip(X))


class Bands:
    """Equal-width radial bands r_{b-1} <= rho < r_b over [0, rho_max]; last band includes its endpoint.

    `index` is (H, W) long with -1 at DC (excluded from correction and sampling).
    `present` lists the non-empty band ids in low-to-high order.
    """

    def __init__(self, h, w, num_bands, rho_max, device=None):
        rho = radial_frequency(h, w, device)
        edges = torch.linspace(0.0, rho_max, num_bands + 1, dtype=torch.float64, device=device)
        idx = torch.searchsorted(edges, rho.contiguous(), right=True) - 1
        idx = idx.clamp(0, num_bands - 1)
        idx[rho > rho_max + 1e-12] = -1
        idx[0, 0] = -1
        self.h, self.w, self.num_bands = h, w, num_bands
        self.rho, self.edges, self.index = rho, edges, idx
        self.nondc = idx >= 0
        self.present = [b for b in range(num_bands) if bool((idx == b).any())]
        self.num_nondc = int(self.nondc.sum())

    def expand(self, alpha):
        """alpha (..., num_bands) -> per-frequency coefficient map (..., H, W), zero at DC."""
        alpha = alpha.to(torch.float64)
        padded = torch.cat([alpha, torch.zeros_like(alpha[..., :1])], dim=-1)  # slot -1 -> 0
        idx = torch.where(self.index >= 0, self.index, torch.full_like(self.index, self.num_bands))
        return padded[..., idx]


def reconstruct(x_o, alpha, delta_p, bands, eps, return_diag=False):
    """T(x_o; alpha): corrected magnitude, original phase, original DC, Hermitian pairing, clip to [0, 1].

    x_o: (3, H, W); alpha: (B,) or (K, B); delta_p: (3, H, W) float64.
    Returns (K, 3, H, W) float32 (K=1 when alpha is 1-D) and optional diagnostics.
    """
    if alpha.dim() == 1:
        alpha = alpha[None]
    X = fft2(x_o)                                   # (3, H, W)
    P = X.abs().square()
    coef = bands.expand(alpha)[:, None]             # (K, 1, H, W)
    P_new = torch.clamp(P[None] + coef * delta_p[None], min=eps)
    X_new = torch.sqrt(P_new) * torch.exp(1j * torch.angle(X))[None]
    X_new[..., 0, 0] = X[..., 0, 0]
    X_new = hermitian(X_new)
    y = ifft2(X_new)
    y_real = y.real
    out = y_real.clamp(0.0, 1.0).to(torch.float32)
    if not return_diag:
        return out
    diag = {
        "imag_max": float(y.imag.abs().max()),
        "clip_fraction": ((y_real < 0) | (y_real > 1)).float().mean(dim=(1, 2, 3)),
        "power_pre_clip": P_new,
    }
    return out, diag


def sample_direction(W_sqrt, shape, generator, device):
    """Eq. 12: s_c = Re F^-1(sqrt(W) * Z_c), Z_c conjugate-symmetric complex Gaussian.

    FFT of real white Gaussian noise (orthonormal) has unit expected |Z|^2 at every frequency,
    conjugate symmetry, and real Gaussian self-conjugate coefficients - exactly the required law.
    """
    n = torch.randn(shape, generator=generator, dtype=torch.float64).to(device)
    Z = fft2(n)
    return ifft2(W_sqrt * Z).real


def band_kl(power, ref_power, bands, eps):
    """Mean over channels and present bands of KL(q_{c,b} || q_{r,c,b}) (Eq. 7-8).

    power: (K, 3, H, W) corrected power (already >= eps); ref_power: (3, H, W) mean real power.
    """
    K = power.shape[0]
    B = bands.num_bands
    idx = bands.index[bands.nondc]                         # (M,)
    p = power[..., bands.nondc]                            # (K, 3, M)
    r = ref_power[..., bands.nondc] + eps                  # (3, M)
    sum_p = torch.zeros(K, 3, B, dtype=p.dtype, device=p.device).index_add_(2, idx, p)
    sum_r = torch.zeros(3, B, dtype=r.dtype, device=r.device).index_add_(1, idx, r)
    q = p / sum_p[..., idx]
    qr = r / sum_r[..., idx]
    terms = q * (torch.log(q) - torch.log(qr))
    kl = torch.zeros(K, 3, B, dtype=p.dtype, device=p.device).index_add_(2, idx, terms)
    present = torch.tensor(bands.present, device=p.device)
    return kl[..., present].mean(dim=(1, 2))


def full_bands(cfg, h, w, device=None):
    s = cfg["spectral"]
    rho_max = s.get("rho_max", math.sqrt(0.5))
    return Bands(h, w, s["num_bands"], rho_max, device)
