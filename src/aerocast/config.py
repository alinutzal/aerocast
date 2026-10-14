"""Run configuration: YAML files with `base:` inheritance and dotted overrides."""
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

# Sections that name outputs or logging rather than the experiment; excluded from the hash.
_UNHASHED = ("name", "output", "logging", "tuning", "evaluate")
MODEL_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"


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


def model_file(model):
    """A model config: an existing YAML path, or a name in the repo's configs/models/."""
    path = Path(model)
    if path.suffix in (".yaml", ".yml") or path.exists():
        return path
    return MODEL_DIR / f"{model}.yaml"


def load_config(path, overrides=None, model=None):
    """Load a config file. `model` (a name in configs/models/ or a path) replaces the model section
    and merges the rest of that file; `overrides` are "section.key=value" strings parsed as YAML."""
    path = Path(path)
    cfg = _read(path)
    cfg.setdefault("name", path.stem)
    if model:
        model_cfg = _read(model_file(model))
        cfg["model"] = model_cfg.pop("model")
        cfg = deep_merge(cfg, model_cfg)
        cfg["name"] = f"{cfg['name']}-{cfg['model']['name']}"
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    # Job staging is applied after user overrides. Preserve scientific identity across
    # scratch/TMPDIR locations and retain original paths for later evaluation.
    if os.environ.get("AEROCAST_DATA_DIR"):
        original = cfg.setdefault("_storage_original", {
            key: cfg["data"].get(key) for key in ("data_dir", "cache_dir", "static_file")
        })
        cfg["data"]["data_dir"] = os.environ["AEROCAST_DATA_DIR"]
        cfg["data"]["cache_dir"] = os.environ["AEROCAST_CACHE_DIR"]
        if original.get("static_file") and Path(original["static_file"]).is_absolute():
            cfg["data"]["static_file"] = str(Path(os.environ["AEROCAST_DATA_DIR"]) / Path(original["static_file"]).name)
        root = Path(os.environ["AEROCAST_OUTPUT_DIR"])
        cfg["output"].update(runs_dir=str(root / "runs"),
                             results_csv=str(root / "results/results.csv"),
                             experiments_md=str(root / "EXPERIMENTS.md"))
    elif "_storage_original" in cfg:
        cfg["data"].update(cfg["_storage_original"])
    # Round-trip through JSON so dates parsed by YAML become plain strings.
    return json.loads(json.dumps(cfg, default=str))


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def config_hash(cfg):
    """Short hash of the experiment settings (ignores run name, output paths and logging)."""
    cfg = copy.deepcopy(cfg)
    original = cfg.pop("_storage_original", None)
    if original:
        for key, value in original.items():
            if value is None:
                cfg["data"].pop(key, None)
            else:
                cfg["data"][key] = value
    settings = {k: v for k, v in cfg.items() if k not in _UNHASHED}
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
