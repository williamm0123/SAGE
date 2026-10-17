"""Execution-lock metadata recorded next to every result file."""
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _git(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_metadata(cfg, extra=None):
    tp = Path(os.environ.get("THIRD_PARTY", ROOT / "third_party"))
    meta = dict(
        created=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array=os.environ.get("SLURM_ARRAY_TASK_ID"),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        sage_commit=_git(ROOT), third_party={d: _git(tp / d) for d in ("DIRE", "UniversalFakeDetect", "QEBA")},
        config_path=cfg["_path"], config_sha256=cfg["_sha256"], protocol=cfg["protocol_version"],
    )
    if extra:
        meta.update(extra)
    return meta
