import hashlib
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sage_v1.0.json"


def load_config(path=None):
    path = Path(path or DEFAULT_CONFIG)
    cfg = json.loads(path.read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg
