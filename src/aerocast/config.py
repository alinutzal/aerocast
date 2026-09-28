"""Run configuration: YAML files with `base:` inheritance and dotted overrides."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

# Sections that name outputs or logging rather than the experiment; excluded from the hash.
_UNHASHED = ("name", "output", "logging")


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(_read(Path(path).parent / base), cfg)
    return cfg


def set_dotted(cfg, key, value):
    """Set cfg["a"]["b"] for key "a.b"; the key must already exist (catches typos)."""
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise KeyError(f"Unknown config section '{part}' in '{key}'")
        node = node[part]
    if leaf not in node:
        raise KeyError(f"Unknown config key '{key}'")
    node[leaf] = value


def load_config(path, overrides=None):
    """Load a config file. `overrides` are "section.key=value" strings; values are parsed as YAML."""
    path = Path(path)
    cfg = _read(path)
    cfg.setdefault("name", path.stem)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    # Round-trip through JSON so dates parsed by YAML become plain strings.
    return json.loads(json.dumps(cfg, default=str))


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def config_hash(cfg):
    """Short hash of the experiment settings (ignores run name, output paths and logging)."""
    settings = {k: v for k, v in cfg.items() if k not in _UNHASHED}
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
