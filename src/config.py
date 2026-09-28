import copy
import os

import yaml


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    """Load a YAML config. A top-level `base:` key names another config
    (relative to this file) whose values are inherited and overridden."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("base", None)
    if base:
        base_path = os.path.join(os.path.dirname(path), base)
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg


def apply_overrides(cfg, overrides):
    """Apply `section.key=value` overrides from the command line.
    Values are parsed as YAML, so numbers, bools, null and lists work."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg
