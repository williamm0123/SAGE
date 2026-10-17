"""End-to-end checks with a deterministic linear toy detector (no downloads, CPU)."""
import copy

import pytest
import torch

from sage.attack import AttackContext, controls_task, make_sage_task
from sage.baselines import make_baseline_task
from sage.config import load_config
from sage.detectors import LinearToyDetector
from sage.perceptual import rmse
from sage.query import QueryOracle, StopAttack, run_tasks
from sage.sap import SpectralPrior
from sage.spectral import Bands, power_spectrum

H = 32


def toy_cfg(budget=400):
    cfg = copy.deepcopy(load_config())
    cfg["budget"]["max_queries"] = budget
    cfg["budget"]["checkpoints"] = [50, 100, budget]
    cfg["perceptual"]["delta_p"] = 0.01
    cfg["asa"]["alpha_grid"] = [0.0, 0.5, 1.0]
    return cfg


class RMSEPerceptual:
    name = "rmse"

    def __call__(self, x, ref):
        if ref.dim() == 3:
            ref = ref[None]
        return rmse(x, ref.expand_as(x)).float()


@pytest.fixture(scope="module")
def setup():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(3, H, H, generator=g)
    w = w / w.norm()
    b = torch.full((3, H, H), 0.5)
    det = LinearToyDetector(w, b)
    fakes = [(0.5 + 0.03 * w * H + 0.02 * torch.randn(3, H, H, generator=g)).clamp(0, 1) for _ in range(4)]
    fakes = [x for x in fakes if not bool(det.predict_real(x[None])[0])]
    # references far from the boundary normal, so bisection lands far from the closest boundary point
    reals = torch.stack([(0.5 + 0.15 * torch.randn(3, H, H, generator=g) - 0.01 * w * H).clamp(0, 1)
                         for _ in range(6)])
    reals = reals[det.predict_real(reals)][:3]
    assert bool(det.predict_real(reals).all())
    P_r = torch.stack([power_spectrum(x) for x in reals]).mean(0)
    P_f = torch.stack([power_spectrum(x) for x in fakes]).mean(0)
    prior = SpectralPrior(P_r, P_f)
    cfg = toy_cfg()
    bands = Bands(H, H, 8, 0.5 ** 0.5)
    ctx = AttackContext(cfg, prior, bands, RMSEPerceptual(), reals, "cpu")
    return det, fakes, ctx


def seed_gen(s=0):
    return lambda: torch.Generator().manual_seed(s)


def drive(det, gens, max_batch=16):
    results = {}
    run_tasks(gens, det.predict_real, max_batch=max_batch, concurrency=4,
              on_done=lambda k, r: results.__setitem__(k, r))
    return results


def test_oracle_cache_and_budget(setup):
    det, fakes, ctx = setup
    cfg = toy_cfg(budget=2)
    oracle = QueryOracle(fakes[0], ctx.perceptual, cfg)

    def attack():
        yield from oracle.query(ctx.refs[0])
        yield from oracle.query(ctx.refs[0])   # cached: not charged
        yield from oracle.query(ctx.refs[1])
        yield from oracle.query(ctx.refs[2])   # over budget

    gen = attack()
    with pytest.raises(StopAttack) as e:
        xs = next(gen)
        while True:
            xs = gen.send(det.predict_real(xs))
    assert e.value.reason == "budget"
    assert oracle.count == 2 and oracle.n_real == 2


@pytest.mark.parametrize("method", ["sage", "ba", "hsja", "qeba"])
def test_methods_respect_budget_and_improve(setup, method):
    det, fakes, ctx = setup
    gens = []
    for i, x in enumerate(fakes):
        if method == "sage":
            gens.append((i, make_sage_task(x, ctx, seed_gen(i), stop_on_success=False)))
        else:
            gens.append((i, make_baseline_task(method, x, ctx, seed_gen(i), stop_on_success=False)))
    res = drive(det, gens)
    assert len(res) == len(fakes)
    for i, r in res.items():
        assert r["queries"] <= ctx.cfg["budget"]["max_queries"]
        assert r["best_real"] is not None
        init_l2 = (ctx.refs[0] - fakes[i]).norm()
        assert r["best_real"]["l2"] < float(init_l2)
        assert set(r["checkpoints"]) == {"50", "100", "400"}
    if method == "sage":
        assert all("asa" in r and "sbqr" in r or r["start"] == "asa" for r in res.values())


def test_first_success_stops(setup):
    det, fakes, ctx = setup
    res = drive(det, [(i, make_sage_task(x, ctx, seed_gen(i))) for i, x in enumerate(fakes)])
    for r in res.values():
        if r["success"]:
            assert r["stop_reason"] == "success"
            assert r["queries"] == r["first_success"]


def test_controls_share_start(setup):
    det, fakes, ctx = setup
    res = drive(det, [(0, controls_task(fakes[0], ctx, seed_gen(0), stop_on_success=False))])[0]
    assert set(res) == {"uniform", "lowfreq", "learned"}
    init_q = res["uniform"]["init_queries"]
    for v in res.values():
        assert v["init_queries"] == init_q
        assert v["queries"] >= init_q


def test_sbqr_contracts_from_interior(setup):
    """Started deep inside the real region, SBQR must accept contractions and approach x_o."""
    from sage.query import managed
    from sage.sbqr import sbqr
    from sage.weights import uniform_weights

    det, fakes, ctx = setup
    x_o = fakes[0]
    oracle = QueryOracle(x_o, ctx.perceptual, ctx.cfg, stop_on_success=False)
    info = {}
    gen = managed(sbqr(x_o, ctx.refs[0], uniform_weights(ctx.bands), oracle, ctx.cfg,
                       torch.Generator().manual_seed(0), info), oracle, info)
    r = drive(det, [(0, gen)])[0]
    assert r["sbqr"]["accepted"] > 10
    assert r["best_real"]["l2"] < 0.95 * float((ctx.refs[0] - x_o).norm())   # boundary sits at ~0.88
