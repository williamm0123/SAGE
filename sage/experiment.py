"""Experiment bookkeeping shared by the runner, the status report and the analysis.

* which runs an experiment must contain (targets x seeds [x control variants]);
* crash-safe JSONL result files (torn last line repaired, control variants written as one unit);
* execution locks: a result directory accepts runs only from one frozen configuration;
* resumable run wrapper around the per-run QueryLog;
* Slurm time-left query for stopping before the job's time limit.
"""
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from .query import QueryLog, ReplayMismatch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("pilot", "primary", "controls", "transfer")
PRIMARY_METHODS = ("sage", "ba", "hsja", "qeba")
CONTROL_METHODS = ("control_uniform", "control_lowfreq", "control_learned")
EXIT_INCOMPLETE = 75          # runner stopped before its shard was finished (resubmit to continue)
EXIT_LOCK_MISMATCH = 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------- targets
def select_targets(m, experiment):
    if experiment == "pilot":
        return m.select("validation", ["sdv1", "sdv2"], pilot=True)
    if experiment == "primary":
        return m.select("test", ["sdv1", "sdv2"])
    if experiment == "controls":
        return m.select("test", ["sdv1", "sdv2"], ablation=True)
    if experiment == "transfer":
        return m.select("test", ["sdv2"])
    raise ValueError(experiment)


def methods_of(experiment):
    if experiment == "controls":
        return CONTROL_METHODS
    if experiment == "transfer":
        return ("sage",)
    return PRIMARY_METHODS


def method_dir(experiment, method):
    """Result directory name: all control variants share one directory (one run produces all three)."""
    return "controls" if experiment == "controls" else method


def screening_paths(results, detector):
    base = Path(results) / "screening"
    return base / f"{detector}.csv", base / f"{detector}_summary.json"


def eligible_ids(screening_csv):
    with open(screening_csv, newline="") as f:
        return {r["id"] for r in csv.DictReader(f) if r["cls"] != "real" and r["pred_real"] == "0"}


def expected_runs(m, experiment, eligible, seeds):
    """Ordered list of (row, seed) the experiment must contain for one detector."""
    targets = [r for r in select_targets(m, experiment) if r["id"] in eligible]
    return [(r, s) for r in targets for s in seeds]


# ---------------------------------------------------------------------------------- JSONL store
def read_jsonl(path):
    """Records of a result file; a torn final line (job killed mid-write) is ignored."""
    out = []
    lines = Path(path).read_text().split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    return out


def repair_jsonl(path):
    """Cut a torn final line so that appending starts on a fresh line."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    with open(path, "r+b") as f:
        f.truncate(data.rfind(b"\n") + 1)


def append_records(fh, records):
    """Write all records of one run in a single write and force them to disk."""
    fh.write("".join(json.dumps(r, default=float) + "\n" for r in records))
    fh.flush()
    os.fsync(fh.fileno())


def load_dir(out_dir):
    recs = []
    for f in sorted(Path(out_dir).glob("shard*.jsonl")):
        recs += read_jsonl(f)
    return recs


def done_runs(records, experiment):
    """(id, seed) pairs that are complete; a controls run counts only if all three variants are present."""
    if experiment != "controls":
        return {(r["id"], r["seed"]) for r in records}
    seen = {}
    for r in records:
        seen.setdefault((r["id"], r["seed"]), set()).add(r["method"])
    return {k for k, v in seen.items() if set(CONTROL_METHODS) <= v}


# --------------------------------------------------------------------------------------- locks
def code_sha256():
    """Hash of the code that determines attack behaviour (docs and analysis excluded)."""
    files = sorted((ROOT / "sage").glob("*.py")) + [ROOT / "experiments/run_attacks.py", ROOT / "configs/sage_v1.0.json"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _normalize(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def lock_diff(a, b, prefix=""):
    diffs = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs += lock_diff(va, vb, f"{prefix}{k}.")
        elif va != vb:
            diffs.append(f"{prefix}{k}: locked={va!r} now={vb!r}")
    return diffs


def check_lock(out_dir, lock):
    """Create LOCK.json on first use; afterwards refuse to add runs under a different configuration."""
    path = Path(out_dir) / "LOCK.json"
    lock = _normalize(lock)
    if not path.exists():
        path.write_text(json.dumps(lock, indent=2, sort_keys=True))
        return []
    return lock_diff(json.loads(path.read_text()), lock)


# ------------------------------------------------------------------------------- resumable runs
def qlog_path(out_dir, image_id, seed):
    return Path(out_dir) / "qlog" / f"{image_id}_seed{seed}.qlog"


def resumable(make_task, log_path):
    """Run make_task(log) so that it resumes from its query log; restart from scratch on replay mismatch."""
    log = QueryLog(log_path)
    restarted = False
    try:
        try:
            result = yield from make_task(log)
        except ReplayMismatch as e:
            print(f"WARNING {e}; restarting this run from scratch", flush=True)
            log.discard()
            log = QueryLog(log_path)
            restarted = True
            result = yield from make_task(log)
    finally:
        log.close()
    for r in (result.values() if "queries" not in result else [result]):
        r["replay_restart"] = restarted
    return result


# --------------------------------------------------------------------------------------- slurm
def _parse_slurm_time(s):
    """'D-HH:MM:SS', 'HH:MM:SS', 'MM:SS' -> seconds; None for UNLIMITED / unparsable."""
    m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", s.strip())
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def slurm_seconds_left():
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.check_output(["squeue", "-h", "-j", job, "-o", "%L"], text=True, timeout=30)
    except Exception:
        return None
    return _parse_slurm_time(out.splitlines()[0]) if out.strip() else None
