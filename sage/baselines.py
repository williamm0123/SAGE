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
         noise is drawn at (3, H/4, W/4) and bilinearly upsampled (resize factor 4). This is the one
         spatial variant used here, ported from github.com/AI-secure/QEBA @ 8ee4a586.

The authoritative values are BASELINE_PARAMS below (recorded in each result directory's LOCK.json).
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


def boundary_attack(x_o, oracle, ctx, generator, info, steps, spherical_step, source_step,
                    source_step_convergence, step_adaptation, update_stats_every_k, spherical_queue, step_queue):
    best = yield from reference_init(x_o, ctx.refs, oracle, ctx.cfg, info)
    if best is None:
        return
    stats_sph, stats_step = _Queue(spherical_queue), _Queue(step_queue)
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
        # step toward x_o, corrected for clipping having moved the spherical candidate off the sphere
        length = (source_step * source_norm + new_src_norm - source_norm).clamp_min(0) / new_src_norm
        candidate = (spherical + length * new_src).clamp(0, 1)

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
def _hsja(x_o, oracle, ctx, generator, info, subspace_factor, gamma, init_num_evals, max_num_evals,
          num_iterations, constraint, stepsize_search, eval_chunk, tag):
    assert constraint == "l2" and stepsize_search == "geometric_progression"
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
        # Monte-Carlo gradient estimate, streamed in chunks of eval_chunk images: the HSJA estimator
        # mean((f - mean f) rv) equals (sum f rv - mean f * sum rv) / n, so only running sums are kept
        # (a full 5,000-image batch would hold ~4 GB per attack). Queries stay in the same order.
        sum_rv = torch.zeros(shape, dtype=torch.float64, device=xo.device)
        sum_frv = torch.zeros_like(sum_rv)
        n_pos = 0
        for k0 in range(0, num_evals, eval_chunk):
            rv = sample_noise(min(eval_chunk, num_evals - k0))
            cand = (perturbed[None] + delta * rv).clamp(0, 1)
            rv = ((cand - perturbed[None]) / delta).to(torch.float64)
            dec = yield from oracle.query_batch(cand)
            fval = 2 * dec.to(xo.device, torch.float64).view(-1, 1, 1, 1) - 1
            sum_rv += rv.sum(0)
            sum_frv += (fval * rv).sum(0)
            n_pos += int(dec.sum())
        m = (2 * n_pos - num_evals) / num_evals
        if n_pos == num_evals:
            grad = sum_rv / num_evals
        elif n_pos == 0:
            grad = -sum_rv / num_evals
        else:
            grad = (sum_frv - m * sum_rv) / num_evals
        grad = (grad / grad.norm().clamp_min(1e-30)).to(torch.float32)
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


def hsja(x_o, oracle, ctx, generator, info, **params):
    yield from _hsja(x_o, oracle, ctx, generator, info, tag="hsja", **params)


def qeba_s(x_o, oracle, ctx, generator, info, **params):
    yield from _hsja(x_o, oracle, ctx, generator, info, tag="qeba", **params)


BASELINES = {"ba": boundary_attack, "hsja": hsja, "qeba": qeba_s}

# Full parameter dictionaries, frozen before test access and recorded in every result lock.
# Initialisation for all three: the shared reference pool / order and boundary bisection of Sec. 3.5.
# eval_chunk: gradient-estimation images per query call (memory bound only; in first-success mode the
# images of later chunks are not sent once a success is verified, earlier ones are all charged).
_HSJA = dict(constraint="l2", gamma=1.0, init_num_evals=100, max_num_evals=10000, num_iterations=100000,
             stepsize_search="geometric_progression", eval_chunk=32)
BASELINE_PARAMS = {
    "ba": dict(steps=25000, spherical_step=1e-2, source_step=1e-2, source_step_convergence=1e-7,
               step_adaptation=1.5, update_stats_every_k=10, spherical_queue=100, step_queue=30),
    "hsja": dict(_HSJA, subspace_factor=None),
    # QEBA-S: the spatial (bilinear resize, factor 4) subspace variant of QEBA; not QEBA-F / QEBA-I.
    "qeba": dict(_HSJA, subspace_factor=4),
}


def make_baseline_task(name, x_o, ctx, seed_gen, stop_on_success=True, log=None):
    oracle = QueryOracle(x_o, ctx.perceptual, ctx.cfg, stop_on_success, log)
    info = {}
    return managed(BASELINES[name](x_o, oracle, ctx, seed_gen(), info, **BASELINE_PARAMS[name]), oracle, info)
