"""Train a forecast model with a fixed seed; early stopping and the best checkpoint use validation loss.

Each run is saved to <output.runs_dir>/<run_id>/: config.yaml (resolved, with the days used),
run.json (git commit, config hash, model, parameters, AMP, windows, target unit),
norm_stats.json, best.pt and history.csv.
"""
import argparse
import copy
import csv
import json
import random
import subprocess
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from aerocast.config import config_hash, load_config, save_config
from aerocast.data import WindowDataset
from aerocast.models import build_model, count_parameters, model_spec
from aerocast.normalize import fit_stats, normalized_arrays
from aerocast.splits import load_and_split
from aerocast.targets import TargetLoss, physical_ox

# Files that runs append to; changes there do not make the code "dirty".
OUTPUT_PATHS = ("results/", "EXPERIMENTS.md")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit():
    """Short HEAD hash of the repo holding this package, suffixed -dirty if tracked files other
    than the run logs (results/, EXPERIMENTS.md) have uncommitted changes."""
    repo = Path(__file__).resolve().parent

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout

    try:
        head = git("rev-parse", "--short=12", "HEAD").strip()
        changed = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = any(not path.startswith(OUTPUT_PATHS) for path in changed)
    return head + ("-dirty" if dirty else "")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def autocast(device, amp):
    """bf16 autocast when train.amp is 'bf16', otherwise full precision."""
    if amp in (None, False, "none"):
        return nullcontext()
    if amp == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"Unknown train.amp {amp!r}; expected none or bf16")


def model_state(model):
    """Tensors of the state dict only; libraries may add other entries (neuraloperator stores its
    constructor arguments, including functions, which weights-only loading rejects)."""
    return {key: value for key, value in model.state_dict().items() if isinstance(value, torch.Tensor)}


def window_datasets(hourly, stats, cfg, starts_by_name):
    """Normalize the hourly arrays once and build a WindowDataset per set of window starts."""
    spec = model_spec(cfg)
    arrays = normalized_arrays(hourly, stats, spec.target_channels)
    return {name: WindowDataset(**arrays, starts=starts, t_in=spec.t_in, t_out=spec.t_out,
                                future_forcings=cfg["features"]["future_forcings"])
            for name, starts in starts_by_name.items()}


def predict_batch(model, batch, device, amp=None):
    """Run the model on a WindowDataset batch; returns (pred, target), both (B, T_out, K, H, W) float32."""
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    with autocast(device, amp):
        pred = model(batch)
    return pred.float(), batch["target"]


def validate(model, loader, loss_fn, device, amp, stats, target_names):
    """Sample-weighted mean loss and Ox RMSE (native units) over a loader, in eval mode."""
    model.eval()
    loss_sum = sse = n_samples = n_values = 0.0
    with torch.no_grad():
        for batch in loader:
            pred, target = predict_batch(model, batch, device, amp)
            loss_sum += loss_fn(pred, target).item() * len(target)
            error = physical_ox(pred, stats, target_names) - physical_ox(target, stats, target_names)
            sse += float((error.double() ** 2).sum())
            n_samples += len(target)
            n_values += error.numel()
    return loss_sum / n_samples, (sse / n_values) ** 0.5


def new_run_dir(cfg):
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['name']}_{uuid.uuid4().hex[:4]}"
    run_dir = Path(cfg["output"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def train(cfg):
    """Train one run; returns its run directory."""
    cfg = copy.deepcopy(cfg)
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = train_cfg.get("amp", "none")

    hourly, splits = load_and_split(cfg)
    data_cfg["dates"] = sorted({str(d) for d in hourly.days})  # record exactly which days were used
    spec = model_spec(cfg)
    stats = fit_stats(hourly, splits.train, spec.t_in, spec.t_out, data_cfg["normalization"],
                      cfg["features"]["future_forcings"])
    datasets = window_datasets(hourly, stats, cfg, {"train": splits.train, "val": splits.val})
    train_loader = DataLoader(datasets["train"], batch_size=train_cfg["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(cfg["seed"]))
    val_loader = DataLoader(datasets["val"], batch_size=train_cfg["batch_size"])

    model = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **train_cfg["scheduler"])
    loss_fn = TargetLoss(train_cfg["loss"], spec.target_channels, stats)

    run_dir = new_run_dir(cfg)
    meta = {
        "run_id": run_dir.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config_name": cfg["name"],
        "model": cfg["model"]["name"],
        "parameters": count_parameters(model),
        "target_mode": cfg.get("target_mode", "ox"),
        "target_channels": list(spec.target_channels),
        "seed": cfg["seed"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": amp,
        "grad_checkpointing": bool(cfg["model"].get("grad_checkpointing", False)),
        "split_mode": splits.mode,
        "days": data_cfg["dates"],
        "hours_utc": [str(hourly.times[0]), str(hourly.times[-1])],
        "channels": hourly.channels,
        "target_unit": hourly.target_unit,
        "windows": {"train": len(splits.train), "val": len(splits.val),
                    **{f"eval_{label}": len(starts) for label, starts in splits.evaluate.items()}},
        "note": cfg["output"].get("note", ""),
    }
    for key in ("backend",):  # model-reported details (e.g. the Mamba scan backend)
        if hasattr(model, key):
            meta[key] = getattr(model, key)
    save_config(cfg, run_dir / "config.yaml")
    stats.save(run_dir / "norm_stats.json")
    write_json(run_dir / "run.json", meta)

    wandb_run = None
    if cfg["logging"]["wandb"]:
        import wandb
        wandb_run = wandb.init(project=cfg["logging"]["wandb_project"], entity=cfg["logging"]["wandb_entity"],
                               name=run_dir.name, config=cfg)

    print(f"Run {run_dir} on {device}: {meta['model']} ({meta['parameters'] / 1e6:.2f}M parameters, amp={amp}), "
          f"windows train={len(splits.train)} val={len(splits.val)} ({splits.mode}), normalization={stats.mode}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_val, best_epoch, wait, history = float("inf"), 0, 0, []
    started = time.time()
    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            pred, target = predict_batch(model, batch, device, amp)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip"])
            optimizer.step()
            loss_sum += loss.item()
        train_loss = loss_sum / len(train_loader)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_seconds = time.time() - epoch_start

        val_loss, val_rmse = validate(model, val_loader, loss_fn, device, amp, stats, spec.target_channels)
        if val_loss < best_val:
            best_val, best_epoch, wait = val_loss, epoch, 0
            torch.save({"model_state_dict": model_state(model), "epoch": epoch, "val_loss": val_loss,
                        "norm_stats": asdict(stats), "config": cfg, "run_id": run_dir.name,
                        "git_commit": meta["git_commit"], "config_hash": meta["config_hash"]},
                       run_dir / "best.pt")
        else:
            wait += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ox_rmse": val_rmse,
                        "lr": lr, "train_seconds": train_seconds})
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"Training loss became {train_loss} at epoch {epoch}")
        if wandb_run:
            wandb_run.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss, "val/ox_rmse": val_rmse, "lr": lr})
        if epoch % train_cfg["log_every"] == 0:
            print(f"Epoch {epoch:03d} - train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                  f"val Ox RMSE {val_rmse:.3f} {hourly.target_unit} | LR {lr:.6f} | best {best_val:.4f} @ {best_epoch}")
        if wait >= train_cfg["patience"]:
            print(f"Early stopping at epoch {epoch}: no val improvement for {train_cfg['patience']} epochs")
            break

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    meta.update(best_epoch=best_epoch, best_val_loss=best_val, epochs_run=len(history),
                train_seconds=round(time.time() - started, 1),
                seconds_per_epoch=round(float(np.mean([h["train_seconds"] for h in history])), 3))
    if device.type == "cuda":
        meta["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
    write_json(run_dir / "run.json", meta)
    if wandb_run:
        wandb_run.finish()
    print(f"Best val loss {best_val:.4f} at epoch {best_epoch}; checkpoint {run_dir / 'best.pt'}")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train an AeroCast model, then evaluate it and the baselines.")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--model", default=None, help="model config: a name in configs/models/ or a YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.epochs=2 (repeatable)")
    parser.add_argument("--note", default=None, help="note recorded in run.json and EXPERIMENTS.md")
    parser.add_argument("--no-eval", action="store_true", help="skip evaluation (run aerocast-evaluate later)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides, model=args.model)
    if args.note is not None:
        cfg["output"]["note"] = args.note
    run_dir = train(cfg)
    if not args.no_eval:
        from aerocast.evaluate import evaluate_run
        evaluate_run(run_dir)


if __name__ == "__main__":
    main()
