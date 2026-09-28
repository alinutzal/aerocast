"""Target modes (config `target_mode`), implemented once in the output channels and the loss.

ox         one channel: Ox predicted directly.
species    NO2 and O3 (data.target). Ox = NO2 + O3 is formed after denormalizing, never in
           normalized space.
multitask  NO2, O3 and Ox. Loss = L(NO2) + L(O3) + L(Ox) + lambda * mean((Ox - (NO2 + O3))^2),
           the consistency term computed from denormalized heads and divided by the training
           Ox variance so lambda is unitless. Ox is scored from the Ox head; the species sum is
           reported as a second metric.
"""
import torch.nn as nn

MODES = ("ox", "species", "multitask")


def target_channels(cfg):
    """Names of the K output channels for the configured target mode."""
    mode = cfg.get("target_mode", "ox")
    species = [str(v) for v in cfg["data"]["target"]]
    channels = {"ox": ["Ox"], "species": species, "multitask": species + ["Ox"]}
    if mode not in channels:
        raise ValueError(f"Unknown target_mode {mode!r}; expected one of {MODES}")
    return channels[mode]


def physical(values, stats, target_names):
    """Native-unit fields from normalized (B, T, K, H, W) values: every channel by name, plus
    "Ox" (the Ox channel, or the species sum) and "Ox_sum" (species sum) when species are present."""
    names = list(target_names)
    fields = {name: stats.denormalize_target(name, values[:, :, i]) for i, name in enumerate(names)}
    species = [name for name in names if name != "Ox"]
    if species:
        fields["Ox_sum"] = sum(fields[name] for name in species)
        fields.setdefault("Ox", fields["Ox_sum"])
    return fields


def physical_ox(values, stats, target_names):
    """Ox in native units: the Ox channel when there is one, otherwise the species sum."""
    return physical(values, stats, target_names)["Ox"]


def base_loss(loss_cfg):
    """Per-channel loss on normalized values: Huber, or alpha * Huber + (1 - alpha) * MSE."""
    huber = nn.SmoothL1Loss(beta=loss_cfg["huber_beta"])
    if loss_cfg["type"] == "huber":
        return huber
    if loss_cfg["type"] == "mixed":
        mse, alpha = nn.MSELoss(), loss_cfg["alpha"]
        return lambda pred, target: alpha * huber(pred, target) + (1 - alpha) * mse(pred, target)
    raise ValueError(f"Unknown loss type {loss_cfg['type']!r}; expected huber or mixed")


class TargetLoss(nn.Module):
    """Sum of the per-channel losses, plus the multitask consistency term."""

    def __init__(self, loss_cfg, target_names, stats):
        super().__init__()
        self.base = base_loss(loss_cfg)
        self.names = list(target_names)
        self.stats = stats
        self.consistency = float(loss_cfg.get("consistency_weight", 0.0)) if "Ox" in self.names and len(self.names) > 1 else 0.0

    def forward(self, pred, target):
        loss = sum(self.base(pred[:, :, i], target[:, :, i]) for i in range(len(self.names)))
        if self.consistency:
            fields = physical(pred, self.stats, self.names)
            ox_std = self.stats.target_std["Ox"] if self.stats.target_std else 1.0
            loss = loss + self.consistency * ((fields["Ox"] - fields["Ox_sum"]) ** 2).mean() / ox_std ** 2
        return loss
