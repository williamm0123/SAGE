"""Spectral sampling weights W(u, v) (paper Eq. 10-11) and the controlled alternatives."""
import torch


def mixture_weights(a, bands, zeta):
    """W = (1 - zeta) a / sum(a) + zeta / M on non-DC frequencies, W(0,0) = 0; W = 1/M if sum(a) = 0."""
    a = torch.where(bands.nondc, a.to(torch.float64), torch.zeros_like(a, dtype=torch.float64))
    M = bands.num_nondc
    total = a.sum()
    if total <= 0:
        W = torch.full_like(a, 1.0 / M)
    else:
        W = (1 - zeta) * a / total + zeta / M
    return torch.where(bands.nondc, W, torch.zeros_like(W))


def learned_weights(alpha, delta_p, bands, zeta):
    """a(u, v) = mean_c |alpha_b(u,v) * dP_c(u, v)|."""
    a = (bands.expand(alpha)[None] * delta_p).abs().mean(0)
    return mixture_weights(a, bands, zeta)


def uniform_weights(bands):
    return mixture_weights(torch.zeros_like(bands.rho), bands, 1.0)


def lowfreq_weights(bands, zeta, rho_max=0.25):
    a = ((bands.rho > 0) & (bands.rho <= rho_max)).to(torch.float64)
    return mixture_weights(a, bands, zeta)
