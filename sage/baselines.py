"""Decision-based baselines under the shared query oracle: Boundary Attack, HSJA (L2) and QEBA-S.

All three start from the same reference initialisation as SAGE (shared reference pool and order),
and every detector call goes through the oracle. Default hyperparameters follow the reference
implementations:

* BA   - foolbox 3.3.4 `BoundaryAttack` defaults: spherical_step=1e-2, source_step=1e-2,
         source_step_convergance=1e-7, step_adaptation=1.5, update_stats_every_k=10,
         stats queues of 100 (spherical) / 30 (step).
* HSJA - Chen et al. official code, constraint='l2', gamma=1.0, init_num_evals=100,
         max_num_evals=1e4, stepsize_search='geometric_progression'.
* QEBA - QEBA-S (spatial resize subspace) of Li et al.: HSJA engine whose gradient-estimation
         noise is drawn at (3, H/4, W/4) and bilinearly upsampled (resize factor 4).
"""
import math

import torch
import torch.nn.functional as F

from .init_point import reference_init
from .query import QueryOracle, managed


def _randn(shape, generator, device):
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


# ----------------------------------------------------------------------------- Boundary Attack
class _Queue:
    def __init__(self, maxlen):
        self.maxlen, self.items = maxlen, []

    def append(self, v):
        self.items.append(float(v))
        self.items = self.items[-self.maxlen:]

    def full(self):
        return len(self.items) == self.maxlen

    def mean(self):
        return sum(self.items) / len(self.items)

    def clear(self):
        self.items = []


def boundary_attack(x_o, oracle, ctx, generator, info, steps=25000, spherical_step=1e-2, source_step=1e-2,
                    source_step_convergence=1e-7, step_adaptation=1.5, update_stats_every_k=10):
    best = yield from reference_init(x_o, ctx.refs, oracle, ctx.cfg, info)
    if best is None:
        return
    stats_sph, stats_step = _Queue(100), _Queue(30)
    xo = x_o.to(torch.float64)
    best = best.to(torch.float64)
    best_norm = (xo - best).norm()
    info["ba"] = dict(steps=0)
    for step in range(1, steps + 1):
        if source_step < source_step_convergence:
            info["ba"]["stop"] = "converged"
            break
        info["ba"]["steps"] = step
        unnormalized = xo - best
        source_norm = unnormalized.norm()
        source_dir = unnormalized / source_norm
        # draw proposals (foolbox 3 draw_proposals)
        eta = _randn(x_o.shape, generator, x_o.device).to(torch.float64)
        eta = eta - (eta * source_dir).sum() * source_dir
        eta = eta * (spherical_step * source_norm / eta.norm())
        distances = math.sqrt(spherical_step ** 2 + 1)
        directions = eta - unnormalized
        spherical = (xo + directions / distances).clamp(0, 1)
        new_src = xo - spherical
        new_src_norm = new_src.norm()
        length = (new_src_norm - source_step * source_norm).clamp_min(0) / new_src_norm
        candidate = (xo - new_src * length).clamp(0, 1)

        check = step % update_stats_every_k == 0
        is_adv = yield from oracle.query(candidate.to(torch.float32))
        if check:
            sph_adv = yield from oracle.query(spherical.to(torch.float32))
            stats_sph.append(sph_adv)
            stats_step.append(is_adv)
        dist = (xo - candidate).norm()
        if is_adv and dist < best_norm:
            best, best_norm = candidate, dist
        if check:
            if stats_sph.full():
                p = stats_sph.mean()
                if p > 0.5:
                    spherical_step *= step_adaptation
                    source_step *= step_adaptation
                    stats_sph.clear()
                elif p < 0.2:
                    spherical_step /= step_adaptation
                    source_step /= step_adaptation
                    stats_sph.clear()
            if stats_step.full():
                p = stats_step.mean()
                if p > 0.25:
                    source_step *= step_adaptation
                    stats_step.clear()
                elif p < 0.1:
                    source_step /= step_adaptation
                    stats_step.clear()
    info["ba"].update(spherical_step=spherical_step, source_step=source_step)


# ----------------------------------------------------------------------------- HSJA / QEBA-S
def _hsja(x_o, oracle, ctx, generator, info, subspace_factor=None, gamma=1.0, init_num_evals=100,
          max_num_evals=10000, num_iterations=100000, tag="hsja"):
    start = yield from reference_init(x_o, ctx.refs, oracle, ctx.cfg, info)
    if start is None:
        return
    shape = x_o.shape
    d = x_o.numel()
    theta = gamma / (math.sqrt(d) * d)
    xo = x_o

    def binary_search(perturbed):
        lo, hi = 0.0, 1.0
        while (hi - lo) / theta > 1:
            mid = (hi + lo) / 2.0
            if (yield from oracle.query(((1 - mid) * xo + mid * perturbed).clamp(0, 1))):
                hi = mid
            else:
                lo = mid
        out = ((1 - hi) * xo + hi * perturbed).clamp(0, 1)
        return out, float((out - xo).norm())

    def sample_noise(n):
        if subspace_factor is None:
            rv = _randn((n,) + tuple(shape), generator, xo.device)
        else:
            small = _randn((n, shape[0], shape[1] // subspace_factor, shape[2] // subspace_factor),
                           generator, xo.device)
            rv = F.interpolate(small, size=shape[1:], mode="bilinear", align_corners=False)
        return rv / rv.flatten(1).norm(dim=1).view(-1, 1, 1, 1)

    perturbed, dist_post = yield from binary_search(start)
    dist = dist_post
    stats = dict(iterations=0)
    info[tag] = stats
    for j in range(num_iterations):
        cur = j + 1
        stats["iterations"] = cur
        delta = 0.1 if cur == 1 else math.sqrt(d) * theta * dist_post
        num_evals = int(min(init_num_evals * math.sqrt(cur), max_num_evals))
        num_evals = min(num_evals, max(oracle.remaining(), 1))
        rv = sample_noise(num_evals)
        cand = (perturbed[None] + delta * rv).clamp(0, 1)
        rv = (cand - perturbed[None]) / delta
        dec = yield from oracle.query_batch(cand)
        fval = 2 * dec.to(xo.device, torch.float32).view(-1, 1, 1, 1) - 1
        m = float(fval.mean())
        if m == 1.0:
            grad = rv.mean(0)
        elif m == -1.0:
            grad = -rv.mean(0)
        else:
            grad = ((fval - m) * rv).mean(0)
        grad = grad / grad.norm().clamp_min(1e-30)
        # geometric progression step size
        eps = dist / math.sqrt(cur)
        for _ in range(60):
            if (yield from oracle.query((perturbed + eps * grad).clamp(0, 1))):
                break
            eps /= 2.0
        else:
            stats["stop"] = "step_search_failed"
            break
        perturbed = (perturbed + eps * grad).clamp(0, 1)
        perturbed, dist_post = yield from binary_search(perturbed)
        dist = float((perturbed - xo).norm())


def hsja(x_o, oracle, ctx, generator, info):
    yield from _hsja(x_o, oracle, ctx, generator, info, subspace_factor=None, tag="hsja")


def qeba_s(x_o, oracle, ctx, generator, info):
    yield from _hsja(x_o, oracle, ctx, generator, info, subspace_factor=4, tag="qeba")


BASELINES = {"ba": boundary_attack, "hsja": hsja, "qeba": qeba_s}


def make_baseline_task(name, x_o, ctx, seed_gen, stop_on_success=True):
    oracle = QueryOracle(x_o, ctx.perceptual, ctx.cfg, stop_on_success)
    info = {}
    return managed(BASELINES[name](x_o, oracle, ctx, seed_gen(), info), oracle, info)
