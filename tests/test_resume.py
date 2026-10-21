"""Resumption, first-success accounting and result-integrity checks (toy detector, CPU)."""
import json
import time

import pytest
import torch

from sage.attack import controls_task, make_sage_task
from sage.baselines import make_baseline_task
from sage.experiment import (_parse_slurm_time, append_records, check_lock, done_runs, read_jsonl,
                             repair_jsonl, resumable)
from sage.query import QueryLog, QueryOracle, StopAttack, run_tasks
from test_attacks import seed_gen, setup, toy_cfg  # noqa: F401  (fixture re-export)

SKIP = {"_x", "replayed_queries", "replay_restart"}


def make(method, x, ctx, log, stop=False):
    if method == "sage":
        return make_sage_task(x, ctx, seed_gen(1), stop_on_success=stop, log=log)
    if method == "controls":
        return controls_task(x, ctx, seed_gen(1), stop_on_success=stop, log=log)
    return make_baseline_task(method, x, ctx, seed_gen(1), stop_on_success=stop, log=log)


def drive(det, gen, max_calls=None):
    """Run one generator; stop (close) after max_calls detector calls. Returns (result, calls, images)."""
    calls = images = 0
    try:
        xs = next(gen)
        while True:
            if max_calls is not None and calls >= max_calls:
                gen.close()
                return None, calls, images
            calls += 1
            images += len(xs)
            xs = gen.send(det.predict_real(xs))
    except StopIteration as e:
        return e.value, calls, images


def strip(r):
    """Drop fields that legitimately differ between sessions (timings, replay bookkeeping)."""
    if isinstance(r, dict):
        return {k: strip(v) for k, v in r.items() if k not in SKIP and k != "seconds"}
    return r


def test_batch_first_success_is_frozen(setup):
    det, fakes, ctx = setup
    cfg = toy_cfg()
    cfg["perceptual"]["delta_p"] = 10.0                     # every real candidate is feasible
    x_o = fakes[0]
    oracle = QueryOracle(x_o, ctx.perceptual, cfg, stop_on_success=True)
    near, far = sorted(ctx.refs[:2], key=lambda r: float((r - x_o).norm()))  # both real-classified
    assert bool(det.predict_real(torch.stack([far, near])).all())

    def attack():
        yield from oracle.query_batch(torch.stack([far, near]))   # near is closer, but comes second

    with pytest.raises(StopAttack):
        drive(det, attack())
    s = oracle.summary("success")
    assert s["first_success"] == 1 and s["queries"] == 2 and s["queries_after_success"] == 1
    assert s["best_feasible"]["q"] == 1 and s["best_real"]["q"] == 1
    assert s["checkpoints"]["50"]["success"]


@pytest.mark.parametrize("method,cut", [("sage", 7), ("hsja", 3), ("ba", 25), ("controls", 30)])
def test_interrupted_run_resumes_identically(setup, tmp_path, method, cut):
    det, fakes, ctx = setup
    x = fakes[1]
    ref, _, ref_images = drive(det, make(method, x, ctx, None))
    path = tmp_path / "run.qlog"
    part, calls, images1 = drive(det, resumable(lambda log: make(method, x, ctx, log), path), max_calls=cut)
    assert part is None and calls == cut and path.stat().st_size == 17 * images1
    res, _, images2 = drive(det, resumable(lambda log: make(method, x, ctx, log), path))
    assert strip(res) == strip(ref)
    assert images1 + images2 == ref_images                      # nothing re-sent to the detector
    first = res if "queries" in res else res["uniform"]
    assert first["replayed_queries"] == images1 and not first["replay_restart"]


def test_replay_mismatch_restarts_and_torn_record(setup, tmp_path):
    det, fakes, ctx = setup
    x = fakes[1]
    ref, _, _ = drive(det, make("sage", x, ctx, None))
    path = tmp_path / "run.qlog"
    drive(det, resumable(lambda log: make("sage", x, ctx, log), path), max_calls=5)
    data = bytearray(path.read_bytes())
    data[17 * 2] ^= 0xFF                                        # corrupt the third recorded key
    path.write_bytes(bytes(data) + b"\x01\x02\x03")             # plus a torn trailing record
    res, _, _ = drive(det, resumable(lambda log: make("sage", x, ctx, log), path))
    assert res["replay_restart"] and strip(res) == strip(ref)

    log = QueryLog(path)
    assert path.stat().st_size % 17 == 0 and len(log.replay) == path.stat().st_size // 17


def test_jsonl_repair_and_controls_completeness(tmp_path):
    f = tmp_path / "shard000of001.jsonl"
    with open(f, "a") as fh:
        append_records(fh, [dict(id="a", seed=0, method=f"control_{v}", queries=1) for v in ("uniform", "lowfreq", "learned")])
        append_records(fh, [dict(id="b", seed=0, method="control_uniform", queries=1)])
    with open(f, "a") as fh:
        fh.write('{"id": "c", "seed": 0, "meth')                 # killed mid-write
    recs = read_jsonl(f)
    assert len(recs) == 4
    assert done_runs(recs, "controls") == {("a", 0)}
    assert done_runs(recs, "primary") == {("a", 0), ("b", 0)}
    repair_jsonl(f)
    assert f.read_text().endswith("\n") and len(read_jsonl(f)) == 4


def test_lock_mismatch(tmp_path):
    lock = dict(config_sha256="x", detector=dict(threshold=0.5), seeds=[0, 1, 2])
    assert check_lock(tmp_path, lock) == []
    assert check_lock(tmp_path, json.loads(json.dumps(lock))) == []
    diffs = check_lock(tmp_path, dict(lock, detector=dict(threshold=0.4)))
    assert diffs and "detector.threshold" in diffs[0]


def test_deadline_stops_scheduler(setup):
    det, fakes, ctx = setup
    gens = [(i, make_sage_task(x, ctx, seed_gen(i))) for i, x in enumerate(fakes)]
    stats = run_tasks(gens, det.predict_real, deadline=time.time() - 1)
    assert stats["completed"] == 0 and stats["detector_calls"] == 0


def test_slurm_time_parse():
    assert _parse_slurm_time("2-23:59:30") == ((2 * 24 + 23) * 60 + 59) * 60 + 30
    assert _parse_slurm_time("05:00") == 300
    assert _parse_slurm_time("1:00:00") == 3600
    assert _parse_slurm_time("UNLIMITED") is None
