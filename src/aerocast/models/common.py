"""Helpers shared by the 2D models: stack hours into channels, pad to a size the network
needs, crop back, and split output channels into (T_out, K)."""
import torch
import torch.nn.functional as F


def stack_time(batch):
    """(B, T_in*C_state + (T_in+T_out)*C_forcing + C_static, H, W): state hours, then forcing
    hours, then static channels, each hour's channels kept together."""
    state, forcing, static = batch["state"], batch["forcing"], batch["static"]
    return torch.cat([state.flatten(1, 2), forcing.flatten(1, 2), static], dim=1)


def pad_to_multiple(x, multiple):
    """Pad (B, C, H, W) at the bottom/right by repeating edge values so H and W are multiples
    of `multiple` (an int or an (h, w) pair). Returns the padded tensor and the original (H, W)."""
    mh, mw = (multiple, multiple) if isinstance(multiple, int) else multiple
    height, width = x.shape[-2:]
    pad_h, pad_w = -height % mh, -width % mw
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (height, width)


def crop(x, size):
    """Undo pad_to_multiple on (..., H', W')."""
    height, width = size
    return x[..., :height, :width]


def unstack_output(y, t_out, k):
    """(B, T_out*K, H, W) -> (B, T_out, K, H, W)."""
    return y.reshape(y.shape[0], t_out, k, *y.shape[-2:])
