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

Batched queries and first-success stopping: a batch (HSJA / QEBA gradient estimation) is sent to the
detector as a whole, so every image in it is charged (`queries`), even those after the first success.
The success itself is located at its exact index in the query order (`first_success`, Eq. 17), and in
first-success mode the archive is frozen there: the reported candidate is the first success, and later
images of the same batch never replace it.

Resumption: an optional QueryLog stores every label obtained from the detector. A resumed run is
re-executed from its seed and answered from the log without new detector calls (the replayed queries
stay charged: they were actually sent). Attacks are deterministic given seed and labels, so the run
continues exactly where it stopped; a key mismatch raises ReplayMismatch and the caller restarts it.
"""
import copy
import hashlib
import math
import time
from pathlib import Path

import torch

from .perceptual import l2, psnr


class StopAttack(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class ReplayMismatch(Exception):
    """A resumed run asked for a different image than the one recorded at this position."""


def _key(x):
    return hashlib.blake2b(x.detach().contiguous().cpu().numpy().tobytes(), digest_size=16).digest()


class QueryLog:
    """Append-only record of (16-byte image key, label) for every image sent to the detector by one run."""
    REC = 17

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.replay, self.pos, self.fh = [], 0, None
        if self.path is not None and self.path.exists():
            data = self.path.read_bytes()
            n = len(data) // self.REC
            if n * self.REC != len(data):            # torn final record: drop it
                with open(self.path, "r+b") as f:
                    f.truncate(n * self.REC)
            self.replay = [(data[i * self.REC:i * self.REC + 16], data[i * self.REC + 16] == 1) for i in range(n)]

    @property
    def replayed(self):
        return self.pos

    def take(self, key):
        """Recorded label for the next query, None once the log is exhausted."""
        if self.pos >= len(self.replay):
            return None
        k, lab = self.replay[self.pos]
        if k != key:
            raise ReplayMismatch(f"{self.path}: query {self.pos + 1} differs from the recorded image")
        self.pos += 1
        return lab

    def append(self, keys, labels):
        if self.path is None:
            return
        if self.fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.path, "ab")
        self.fh.write(b"".join(k + bytes([int(bool(v))]) for k, v in zip(keys, labels)))
        self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None

    def discard(self):
        self.close()
        if self.path is not None and self.path.exists():
            self.path.unlink()
        self.replay, self.pos = [], 0


class QueryOracle:
    def __init__(self, x_o, perceptual, cfg, stop_on_success=True, log=None):
        self.x_o = x_o
        self.perceptual = perceptual
        self.max_queries = cfg["budget"]["max_queries"]
        self.checkpoints = sorted(cfg["budget"]["checkpoints"])
        self.delta_p = cfg["perceptual"]["delta_p"]
        self.stop_on_success = stop_on_success
        self.log = log              # shared by snapshots: one ordered record per run
        self.count = 0
        self.replayed = 0
        self.cache = {}
        self.n_real = 0
        self.first_success = None
        self.best_feasible = None   # min-L2 real candidate with d_p <= delta_p (first success in first-success mode)
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
            labels = []
            for i in send:
                lab = self.log.take(keys[i]) if self.log is not None else None
                if lab is None:
                    break
                labels.append(lab)
            self.replayed += len(labels)
            rest = send[len(labels):]
            if rest:
                new = (yield xs[rest]).to("cpu", torch.bool).tolist()
                if self.log is not None:
                    self.log.append([keys[i] for i in rest], new)
                labels += new
            self._record(xs[send], torch.tensor(labels, dtype=torch.bool), [keys[i] for i in send])
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
        d2_of, dp_of = {}, {}
        if real_idx:
            xr = xs[real_idx]
            d2_of = dict(zip(real_idx, l2(xr, self.x_o[None]).tolist()))
            dp_of = dict(zip(real_idx, self.perceptual(xr, self.x_o).tolist()))
        for j, (lab, k) in enumerate(zip(labels.tolist(), keys)):
            self.cache[k] = lab
            self.count += 1
            frozen = self.stop_on_success and self.first_success is not None
            if lab:
                self.n_real += 1
                cand = dict(q=self.count, l2=d2_of[j], lpips=dp_of[j], x=xs[j].detach())
                self.real_trace.append((cand["q"], cand["l2"], cand["lpips"]))
                if not frozen:
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
        """Independent copy (shared starting point for controlled variants); the query log stays shared."""
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
            queries_after_success=(self.count - self.first_success) if self.first_success else 0,
            replayed_queries=self.replayed,
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


def run_tasks(task_iter, predict_real, max_batch=64, concurrency=16, on_done=None, deadline=None):
    """Drive many attack generators, batching their pending queries into shared detector calls.

    task_iter yields (key, generator); predict_real(x (N,3,H,W)) -> bool tensor (N,).
    on_done(key, result) is called when a generator returns (result = its return value).
    Detector calls are chunked to at most max_batch images. When `deadline` (time.time()) passes, no
    new task is started and the unfinished ones are closed (resumable from their query logs);
    stats["interrupted"] is then True.
    """
    task_iter = iter(task_iter)
    active = {}
    stats = dict(detector_calls=0, images=0, completed=0, interrupted=False)

    def expired():
        return deadline is not None and time.time() >= deadline

    def advance(key, gen, value):
        try:
            xs = gen.send(value)
        except StopIteration as e:
            stats["completed"] += 1
            if on_done:
                on_done(key, e.value)
            return
        active[key] = (gen, xs)

    def refill():
        while len(active) < concurrency and not expired():
            try:
                key, gen = next(task_iter)
            except StopIteration:
                return
            advance(key, gen, None)

    refill()
    while active:
        if expired():
            stats["interrupted"] = True
            for gen, _ in active.values():
                gen.close()
            active.clear()
            break
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
