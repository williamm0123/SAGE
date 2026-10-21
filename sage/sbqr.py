"""Spectral-Biased Query Refinement (paper Sec. 3.6, Eq. 12-14)."""
import torch

from .spectral import sample_direction


def sbqr(x_o, x_start, W, oracle, cfg, generator, info):
    """Refine a verified real-classified point toward x_o using spectrally weighted directions.

    W: (H, W) sampling weights (zero at DC). Mutates info["sbqr"] with counters.
    """
    s = cfg["sbqr"]
    lo, hi = s["step_bounds"]
    gamma, lam = s["gamma0"], s["lambda0"]
    W_sqrt = W.sqrt()
    x = x_start.clone()
    xo64 = x_o.to(torch.float64)
    stats = dict(proposals=0, local_rejections=0, detector_rejections=0, accepted=0,
                 degenerate_resamples=0, adaptations=0)
    info["sbqr"] = stats
    win_n = win_acc = 0
    while stats["proposals"] < s["max_local_proposals"]:
        d = xo64 - x.to(torch.float64)
        dn = d.norm()
        if dn < 1e-12:
            stats["stop"] = "converged"
            break
        eta = None
        for _ in range(s["max_degenerate_resample"] + 1):
            st = sample_direction(W_sqrt, x.shape, generator, x.device)
            proj = st - (st * d).sum() / dn ** 2 * d
            pn = proj.norm()
            if pn > 1e-12 * max(float(st.norm()), 1e-30):
                eta = proj / pn
                break
            stats["degenerate_resamples"] += 1
        if eta is None:
            stats["stop"] = "degenerate_direction"
            break
        x_perp = (x.to(torch.float64) + gamma * dn * eta).clamp(0, 1)
        x_new = (x_perp + lam * (xo64 - x_perp)).clamp(0, 1).to(torch.float32)   # the image actually queried
        stats["proposals"] += 1
        win_n += 1
        if (x_new.to(torch.float64) - xo64).norm() < dn * (1 - s["min_rel_l2_improvement"]):
            if (yield from oracle.query(x_new)):
                x = x_new
                win_acc += 1
                stats["accepted"] += 1
            else:
                stats["detector_rejections"] += 1
        else:
            stats["local_rejections"] += 1
        if win_n == s["adapt_window"]:
            rate = win_acc / win_n
            if rate < s["p_low"]:
                gamma, lam = gamma / s["a_step"], lam / s["a_step"]
            elif rate > s["p_high"]:
                gamma, lam = gamma * s["a_step"], lam * s["a_step"]
            gamma = min(max(gamma, lo), hi)
            lam = min(max(lam, lo), min(hi, 1 - 1e-12))
            stats["adaptations"] += 1
            win_n = win_acc = 0
    else:
        stats["stop"] = "proposal_cap"
    stats["gamma"], stats["lambda"] = gamma, lam
    return x
