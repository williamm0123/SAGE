"""Complete SAGE pipeline (paper Sec. 3.7) and the controlled refinement variants (Sec. 4.3, block 2)."""
import torch

from .asa import asa
from .init_point import reference_init
from .query import QueryOracle, StopAttack, managed
from .sbqr import sbqr
from .weights import learned_weights, lowfreq_weights, uniform_weights


class AttackContext:
    """Everything shared by all runs of one experiment on one device."""

    def __init__(self, cfg, prior, bands, perceptual, refs, device):
        self.cfg, self.prior, self.bands = cfg, prior, bands
        self.perceptual, self.refs, self.device = perceptual, refs, device


def sage_attack(x_o, oracle, ctx, generator, info):
    cfg = ctx.cfg
    x_spec, alpha, asa_info = asa(x_o, ctx.prior, ctx.bands, ctx.perceptual, cfg)
    info["asa"] = asa_info
    W = learned_weights(alpha, ctx.prior.delta, ctx.bands, cfg["sbqr"]["zeta"])
    info["asa_verified"] = False
    if oracle.remaining() <= 0:
        return
    info["asa_verified"] = True
    info["asa_real"], info["asa_success"] = None, False
    try:
        is_real = yield from oracle.query(x_spec)
    except StopAttack as e:
        if e.reason == "success":          # the only query so far was x_spec
            info["asa_real"], info["asa_success"] = True, True
        raise
    info["asa_real"] = is_real
    info["asa_success"] = bool(is_real and asa_info["lpips"] <= cfg["perceptual"]["delta_p"])
    if is_real:
        start = x_spec
        info["start"] = "asa"
    else:
        info["start"] = "reference"
        start = yield from reference_init(x_o, ctx.refs, oracle, cfg, info)
        if start is None:
            return
    yield from sbqr(x_o, start, W, oracle, cfg, generator, info)


def make_sage_task(x_o, ctx, seed_gen, stop_on_success=True, log=None):
    oracle = QueryOracle(x_o, ctx.perceptual, ctx.cfg, stop_on_success, log)
    info = {}
    return managed(sage_attack(x_o, oracle, ctx, seed_gen(), info), oracle, info)


CONTROL_VARIANTS = ("uniform", "lowfreq", "learned")


def controls_task(x_o, ctx, seed_gen, variants=CONTROL_VARIANTS, stop_on_success=True, log=None):
    """Shared verified start (reference initialisation, full cost charged to each variant), then the
    identical SBQR engine with uniform / fixed low-frequency / ASA-learned weights."""
    cfg = ctx.cfg
    base = QueryOracle(x_o, ctx.perceptual, cfg, stop_on_success, log)
    init_info = {}
    start, stop = None, None
    try:
        start = yield from reference_init(x_o, ctx.refs, base, cfg, init_info)
    except StopAttack as e:          # success already reached during initialisation
        stop = e.reason
    weights = {}
    if "uniform" in variants:
        weights["uniform"] = uniform_weights(ctx.bands)
    if "lowfreq" in variants:
        weights["lowfreq"] = lowfreq_weights(ctx.bands, cfg["sbqr"]["zeta"], cfg["lowfreq_control"]["rho_max"])
    if "learned" in variants:
        _, alpha, asa_info = asa(x_o, ctx.prior, ctx.bands, ctx.perceptual, cfg)
        init_info["asa"] = asa_info
        weights["learned"] = learned_weights(alpha, ctx.prior.delta, ctx.bands, cfg["sbqr"]["zeta"])

    results = {}
    for name in variants:
        oracle = base.snapshot()
        info = dict(init_info, variant=name)
        if start is None:
            results[name] = oracle.summary(stop or "init_failure")
            results[name].update(info)
            continue
        gen = managed(sbqr(x_o, start, weights[name], oracle, cfg, seed_gen(), info), oracle, info)
        results[name] = yield from gen
    return results
