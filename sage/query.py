"""Shared label-only query interface used by every attack (paper Sec. 3.5, 4.5).

Attacks are written as generators. To query, an attack does

    is_real = yield from oracle.query(x)            # one image (3, H, W)
    labels  = yield from oracle.query_batch(xs)     # (K, 3, H, W) -> bool tensor (K,)

The generator yields image batches to a scheduler (`run_tasks`), which concatenates pending
batches from many concurrent attacks into one detector call. This changes nothing about the
per-attack query count; it only improves GPU utilisation (crucial for DIRE).

The oracle owns the single query counter, the per-run label cache, the archive of verified
real-classified candidates and the checkpoint bookkeeping. It raises StopAttack when the budget
is exhausted or (first-success mode) as soon as a quality-constrained success is verified.
"""
import copy
import hashlib
import math

import torch

from .perceptual import l2, psnr


class StopAttack(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _key(x):
    return hashlib.blake2b(x.detach().contiguous().cpu().numpy().tobytes(), digest_size=16).digest()


class QueryOracle:
    def __init__(self, x_o, perceptual, cfg, stop_on_success=True):
        self.x_o = x_o
        self.perceptual = perceptual
        self.max_queries = cfg["budget"]["max_queries"]
        self.checkpoints = sorted(cfg["budget"]["checkpoints"])
        self.delta_p = cfg["perceptual"]["delta_p"]
        self.stop_on_success = stop_on_success
        self.count = 0
        self.cache = {}
        self.n_real = 0
        self.first_success = None
        self.best_feasible = None   # min-L2 real candidate with d_p <= delta_p
        self.best_real = None       # min-L2 real candidate (diagnostic)
        self.at_checkpoint = {}
        self.real_trace = []        # (query index, l2, lpips) for every verified real candidate

    # ------------------------------------------------------------------ queries
    def query(self, x):
        labels = yield from self.query_batch(x[None])
        return bool(labels[0])

    def query_batch(self, xs):
        keys = [_key(x) for x in xs]
        todo, seen = [], {}
        for i, k in enumerate(keys):
            if k not in self.cache and k not in seen:
                seen[k] = i
                todo.append(i)
        remaining = self.max_queries - self.count
        if todo and remaining <= 0:
            raise StopAttack("budget")
        send = todo[:remaining]
        if send:
            labels = yield xs[send]
            labels = labels.to("cpu", torch.bool)
            self._record(xs[send], labels, [keys[i] for i in send])
        if len(send) < len(todo):
            raise StopAttack("budget")
        if self.stop_on_success and self.first_success is not None:
            raise StopAttack("success")
        return torch.tensor([self.cache[k] for k in keys], dtype=torch.bool)

    def remaining(self):
        return self.max_queries - self.count

    # ------------------------------------------------------------- bookkeeping
    def _record(self, xs, labels, keys):
        real_idx = torch.nonzero(labels).flatten().tolist()
        d2 = l2(xs, self.x_o[None]) if real_idx else None
        dp = self.perceptual(xs[real_idx], self.x_o) if real_idx else None
        dp_of = dict(zip(real_idx, dp.tolist())) if real_idx else {}
        for j, (lab, k) in enumerate(zip(labels.tolist(), keys)):
            self.cache[k] = lab
            self.count += 1
            if lab:
                self.n_real += 1
                cand = dict(q=self.count, l2=float(d2[j]), lpips=dp_of[j], x=xs[j].detach())
                self.real_trace.append((cand["q"], cand["l2"], cand["lpips"]))
                if self.best_real is None or cand["l2"] < self.best_real["l2"]:
                    self.best_real = cand
                if cand["lpips"] <= self.delta_p:
                    if self.first_success is None:
                        self.first_success = self.count
                    if self.best_feasible is None or cand["l2"] < self.best_feasible["l2"]:
                        self.best_feasible = cand
            if self.count in self.checkpoints:
                self.at_checkpoint[self.count] = self._state()

    def _state(self):
        bf = self.best_feasible
        return dict(success=bf is not None, l2=bf["l2"] if bf else None, lpips=bf["lpips"] if bf else None)

    def snapshot(self):
        """Independent copy (shared starting point for controlled variants)."""
        other = copy.copy(self)
        other.cache = dict(self.cache)
        other.at_checkpoint = dict(self.at_checkpoint)
        other.real_trace = list(self.real_trace)
        return other

    def summary(self, stop_reason):
        final = self._state()
        ckpt = {}
        for q in self.checkpoints:
            # checkpoints beyond the last query inherit the final state (stopped early or attack ended)
            ckpt[str(q)] = self.at_checkpoint.get(q, final if q > self.count else None) or final
        out = dict(
            queries=self.count, stop_reason=stop_reason, first_success=self.first_success,
            success=self.first_success is not None, n_real=self.n_real, checkpoints=ckpt,
        )
        for name, c in (("best_feasible", self.best_feasible), ("best_real", self.best_real)):
            if c is None:
                out[name] = None
            else:
                out[name] = dict(q=c["q"], l2=c["l2"], lpips=c["lpips"],
                                 psnr=float(psnr(c["x"][None], self.x_o[None])[0]),
                                 rmse=c["l2"] / math.sqrt(self.x_o.numel()))
        return out


def managed(attack_gen, oracle, info):
    """Run an attack generator to completion, turning StopAttack into a normal return."""
    reason = "attack_finished"
    try:
        yield from attack_gen
    except StopAttack as e:
        reason = e.reason
    result = oracle.summary(reason)
    result.update(info)
    best = oracle.best_feasible or oracle.best_real
    result["_x"] = best["x"] if best else None      # popped by the runner (optional figure dumps)
    return result


def run_tasks(task_iter, predict_real, max_batch=64, concurrency=16, on_done=None):
    """Drive many attack generators, batching their pending queries into shared detector calls.

    task_iter yields (key, generator); predict_real(x (N,3,H,W)) -> bool tensor (N,).
    on_done(key, result) is called when a generator returns (result = its return value).
    Detector calls are chunked to at most max_batch images.
    """
    task_iter = iter(task_iter)
    active = {}
    stats = dict(detector_calls=0, images=0)

    def advance(key, gen, value):
        try:
            xs = gen.send(value)
        except StopIteration as e:
            if on_done:
                on_done(key, e.value)
            return
        active[key] = (gen, xs)

    def refill():
        while len(active) < concurrency:
            try:
                key, gen = next(task_iter)
            except StopIteration:
                return
            advance(key, gen, None)

    refill()
    while active:
        keys, total = [], 0
        for key, (_, xs) in active.items():
            if keys and total + len(xs) > max_batch:
                break
            keys.append(key)
            total += len(xs)
        X = torch.cat([active[k][1] for k in keys])
        labels = torch.cat([predict_real(X[i:i + max_batch]).to("cpu", torch.bool)
                            for i in range(0, len(X), max_batch)])
        stats["detector_calls"] += 1
        stats["images"] += len(X)
        off = 0
        for k in keys:
            gen, xs = active.pop(k)
            advance(k, gen, labels[off:off + len(xs)])
            off += len(xs)
        refill()
    return stats
