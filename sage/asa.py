"""Adaptive Spectral Alignment (paper Eq. 7-8): band-wise coordinate descent over a coefficient grid."""
import time

import torch

from .spectral import band_kl, power_spectrum, reconstruct


@torch.no_grad()
def asa(x_o, prior, bands, perceptual, cfg):
    """Return x_spec, alpha* and diagnostics. Uses no detector queries.

    J(alpha) = mean_{c,b} KL(q_{c,b}(alpha) || q_{r,c,b}) + lambda_perc * d_p(T(x_o; alpha), x_o)
    """
    a_cfg, eps = cfg["asa"], cfg["spectral"]["eps"]
    grid = torch.tensor(a_cfg["alpha_grid"], dtype=torch.float64, device=x_o.device)
    lam = a_cfg["lambda_perc"]
    t0 = time.time()
    evals = 0

    def objective(alphas):
        nonlocal evals
        imgs, diag = reconstruct(x_o, alphas, prior.delta, bands, eps, return_diag=True)
        kl = band_kl(diag["power_pre_clip"], prior.P_r, bands, eps)
        dp = perceptual(imgs, x_o).to(torch.float64) if lam > 0 else torch.zeros_like(kl)
        evals += len(alphas)
        return kl + lam * dp, kl, dp

    alpha = torch.zeros(bands.num_bands, dtype=torch.float64, device=x_o.device)
    J, _, _ = objective(alpha[None])
    J = float(J[0])
    history = [J]
    sweeps = 0
    for _ in range(a_cfg["max_sweeps"]):
        J_start = J
        for b in bands.present:              # low-to-high
            cand = alpha.repeat(len(grid), 1)
            cand[:, b] = grid
            Jc, _, _ = objective(cand)
            k = int(torch.argmin(Jc))          # first minimum -> smaller coefficient on ties
            if float(Jc[k]) <= J:
                alpha, J = cand[k].clone(), float(Jc[k])
        sweeps += 1
        history.append(J)
        if (J_start - J) / max(abs(J_start), 1e-30) < a_cfg["rel_tol"]:
            break

    x_spec, diag = reconstruct(x_o, alpha, prior.delta, bands, eps, return_diag=True)
    _, kl0, _ = objective(torch.zeros_like(alpha)[None])
    _, kl1, dp1 = objective(alpha[None])
    # post-clip spectral discrepancy: KL of the clipped image's actual spectrum
    kl_post = band_kl(power_spectrum(x_spec).clamp_min(eps), prior.P_r, bands, eps)
    info = dict(
        alpha=alpha.tolist(), J=J, J_history=history, sweeps=sweeps, objective_evals=evals,
        kl_before=float(kl0[0]), kl_pre_clip=float(kl1[0]), kl_post_clip=float(kl_post[0]),
        lpips=float(dp1[0]), clip_fraction=float(diag["clip_fraction"][0]), imag_max=diag["imag_max"],
        seconds=time.time() - t0,
    )
    return x_spec[0], alpha, info
