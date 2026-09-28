"""Target modes: which fields the model predicts (config `target_mode`)."""

MODES = ("ox",)


def target_channels(cfg):
    """Names of the K output channels for the configured target mode."""
    mode = cfg.get("target_mode", "ox")
    if mode not in MODES:
        raise ValueError(f"Unknown target_mode {mode!r}; expected one of {MODES}")
    return ["Ox"]


def physical_ox(values, stats, target_names):
    """Ox in native units from normalized (B, T, K, H, W) outputs or targets: the Ox channel
    when there is one, otherwise the sum of the species, added after denormalizing."""
    names = list(target_names)
    if "Ox" in names:
        return stats.denormalize_target("Ox", values[:, :, names.index("Ox")])
    return sum(stats.denormalize_target(name, values[:, :, i]) for i, name in enumerate(names))
