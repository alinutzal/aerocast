"""Train the ConvLSTM with a fixed seed; early stopping and the best checkpoint use validation loss.

Each run is saved to <output.runs_dir>/<run_id>/: config.yaml (resolved, with the days used),
run.json (git commit, config hash, windows, target unit), norm_stats.json, best.pt, history.csv.
"""
import argparse
import copy
import csv
import json
import random
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aerocast.config import config_hash, load_config, save_config
from aerocast.data import WindowDataset
from aerocast.models import build_model
from aerocast.normalize import fit_stats
from aerocast.splits import load_and_split


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit():
    """Short HEAD hash of the repo holding this package, suffixed -dirty if tracked files changed."""
    repo = Path(__file__).resolve().parent

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()

    try:
        head = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def build_loss(loss_cfg):
    huber = nn.SmoothL1Loss(beta=loss_cfg["huber_beta"])
    if loss_cfg["type"] == "huber":
        return huber
    if loss_cfg["type"] == "mixed":
        mse, alpha = nn.MSELoss(), loss_cfg["alpha"]
        return lambda pred, target: alpha * huber(pred, target) + (1 - alpha) * mse(pred, target)
    raise ValueError(f"Unknown loss type {loss_cfg['type']!r}; expected huber or mixed")


def window_datasets(hourly, stats, cfg, starts_by_name):
    """Normalize the hourly arrays once and build a WindowDataset per set of window starts."""
    x, y = stats.normalize_x(hourly.x), stats.normalize_y(hourly.y)
    forcing = hourly.forcing_channels if cfg["features"]["future_forcings"] else None
    seq_len, pred_len = cfg["data"]["seq_len"], cfg["data"]["pred_len"]
    return {name: WindowDataset(x, y, starts, seq_len, pred_len, forcing) for name, starts in starts_by_name.items()}


def predict_batch(model, batch, device, pred_len):
    """Roll the model out on a WindowDataset batch; returns (pred, y) on device, normalized units."""
    x, y, *future = (b.to(device) for b in batch)
    pred = model(x, pred_steps=pred_len, future_x=future[0] if future else None)
    return pred.reshape(y.shape), y


def mean_loss(model, loader, loss_fn, device, pred_len):
    """Sample-weighted mean loss and MSE over a loader, in eval mode."""
    model.eval()
    loss_sum = mse_sum = n = 0.0
    with torch.no_grad():
        for batch in loader:
            pred, y = predict_batch(model, batch, device, pred_len)
            loss_sum += loss_fn(pred, y).item() * len(y)
            mse_sum += torch.mean((pred - y) ** 2).item() * len(y)
            n += len(y)
    return loss_sum / n, mse_sum / n


def new_run_dir(cfg):
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['name']}_{uuid.uuid4().hex[:4]}"
    run_dir = Path(cfg["output"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def train(cfg):
    """Train one run; returns its run directory."""
    cfg = copy.deepcopy(cfg)
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    seq_len, pred_len = data_cfg["seq_len"], data_cfg["pred_len"]
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hourly, splits = load_and_split(cfg)
    data_cfg["dates"] = sorted({str(d) for d in hourly.days})  # record exactly which days were used
    stats = fit_stats(hourly, splits.train, seq_len, pred_len, data_cfg["normalization"])
    datasets = window_datasets(hourly, stats, cfg, {"train": splits.train, "val": splits.val})
    train_loader = DataLoader(datasets["train"], batch_size=train_cfg["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(cfg["seed"]))
    val_loader = DataLoader(datasets["val"], batch_size=train_cfg["batch_size"])

    model = build_model(cfg["model"], len(hourly.channels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **train_cfg["scheduler"])
    loss_fn = build_loss(train_cfg["loss"])

    run_dir = new_run_dir(cfg)
    meta = {
        "run_id": run_dir.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config_name": cfg["name"],
        "seed": cfg["seed"],
        "device": str(device),
        "split_mode": splits.mode,
        "days": data_cfg["dates"],
        "hours_utc": [str(hourly.times[0]), str(hourly.times[-1])],
        "channels": hourly.channels,
        "target": "+".join(data_cfg["target"]),
        "target_unit": hourly.target_unit,
        "windows": {"train": len(splits.train), "val": len(splits.val),
                    **{f"eval_{label}": len(starts) for label, starts in splits.evaluate.items()}},
    }
    save_config(cfg, run_dir / "config.yaml")
    stats.save(run_dir / "norm_stats.json")
    write_json(run_dir / "run.json", meta)

    wandb_run = None
    if cfg["logging"]["wandb"]:
        import wandb
        wandb_run = wandb.init(project=cfg["logging"]["wandb_project"], entity=cfg["logging"]["wandb_entity"],
                               name=run_dir.name, config=cfg)

    print(f"Run {run_dir} on {device}: {len(hourly.channels)} channels, "
          f"windows train={len(splits.train)} val={len(splits.val)} ({splits.mode}), normalization={stats.mode}")
    y_scale = stats.y_std or 1.0
    best_val, best_epoch, wait, history = float("inf"), 0, 0, []
    started = time.time()
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            pred, y = predict_batch(model, batch, device, pred_len)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip"])
            optimizer.step()
            loss_sum += loss.item()
        train_loss = loss_sum / len(train_loader)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_loss, val_mse = mean_loss(model, val_loader, loss_fn, device, pred_len)
        val_rmse = val_mse ** 0.5 * y_scale  # native units
        if val_loss < best_val:
            best_val, best_epoch, wait = val_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "norm_stats": asdict(stats), "config": cfg, "run_id": run_dir.name,
                        "git_commit": meta["git_commit"], "config_hash": meta["config_hash"]},
                       run_dir / "best.pt")
        else:
            wait += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rmse": val_rmse, "lr": lr})
        if wandb_run:
            wandb_run.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss, "val/rmse": val_rmse, "lr": lr})
        if epoch % train_cfg["log_every"] == 0:
            print(f"Epoch {epoch:03d} - train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                  f"val RMSE {val_rmse:.3f} {hourly.target_unit} | LR {lr:.6f} | best {best_val:.4f} @ {best_epoch}")
        if wait >= train_cfg["patience"]:
            print(f"Early stopping at epoch {epoch}: no val improvement for {train_cfg['patience']} epochs")
            break

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    meta.update(best_epoch=best_epoch, best_val_loss=best_val, epochs_run=len(history),
                train_seconds=round(time.time() - started, 1))
    write_json(run_dir / "run.json", meta)
    if wandb_run:
        wandb_run.finish()
    print(f"Best val loss {best_val:.4f} at epoch {best_epoch}; checkpoint {run_dir / 'best.pt'}")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the AeroCast ConvLSTM, then evaluate it and the baselines.")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.epochs=2 (repeatable)")
    parser.add_argument("--no-eval", action="store_true", help="skip evaluation (run aerocast-evaluate later)")
    args = parser.parse_args(argv)
    run_dir = train(load_config(args.config, args.overrides))
    if not args.no_eval:
        from aerocast.evaluate import evaluate_run
        evaluate_run(run_dir)


if __name__ == "__main__":
    main()
