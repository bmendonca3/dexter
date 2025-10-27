import json
import os
import re
from pathlib import Path
from typing import Any, Optional


def is_offline() -> bool:
    """
    Return True when the agent should avoid live network calls.
    Controlled via the DEXTER_OFFLINE environment variable.
    """
    flag = os.getenv("DEXTER_OFFLINE", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _cache_root() -> Path:
    base = os.getenv("DEXTER_CACHE_DIR") or "cache"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_segment(segment: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", segment)


def cache_path(resource: str, key: str) -> Path:
    """
    Build a deterministic cache path for a resource/key pair.
    Resource should describe the tool (e.g., 'price_history').
    Key should encode arguments that affect the payload.
    """
    resource_dir = _cache_root() / _sanitize_segment(resource)
    resource_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitize_segment(key)}.json"
    return resource_dir / filename


def save_cache(resource: str, key: str, payload: Any) -> None:
    """
    Persist a JSON-serializable payload to disk so it can be reused in offline mode.
    """
    path = cache_path(resource, key)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def load_cache(resource: str, key: str) -> Optional[Any]:
    """
    Retrieve cached payload if it exists, otherwise return None.
    """
    path = cache_path(resource, key)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
