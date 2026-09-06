"""Load project configuration from config.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # repository root

_CONFIG = None


def resolve_path(p):
    """Resolve a possibly-relative path against the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path=None) -> dict:
    global _CONFIG
    if _CONFIG is None:
        p = Path(path) if path else ROOT / "config.yaml"
        with open(p, "r") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get(*keys, default=None):
    cfg = load_config()
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set(*keys, value):
    """Set a config value in-memory (runtime override, does not write yaml)."""
    cfg = load_config()
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
