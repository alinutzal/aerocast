"""Profile every model on one GPU with the benchmark settings of its config.

Records parameters, peak GPU memory and time per step while training (batch from
train.batch_size, the model's AMP and checkpointing settings), the training time per epoch
for a given number of training windows, and the inference time of one 10-hour forecast on
the full grid (batch 1, 50 runs after warm-up). The CMAQ wall-clock column is left empty to
fill in by hand for the same domain and horizon.

    uv run python scripts/profile_models.py --config configs/smoke.yaml
    uv run python scripts/profile_models.py --config configs/smoke.yaml --models unet fno --train-windows 700
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from aerocast.config import load_config
from aerocast.models import REGISTRY, build_model, count_parameters, model_spec
from aerocast.splits import load_and_split
from aerocast.train import autocast, seed_everything

COLUMNS = ["model", "parameters", "amp", "grad_checkpointing", "backend", "gpu", "grid", "batch_size",
           "peak_train_memory_gb", "train_seconds_per_step", "train_windows", "train_seconds_per_epoch",
           "inference_ms_mean", "inference_ms_std", "cmaq_wallclock_seconds"]


def random_batch(spec, batch, height, width, device):
    generator = torch.Generator(device="cpu").manual_seed(0)
    tensors = {"state": (batch, spec.t_in, len(spec.state_channels)),
               "forcing": (batch, spec.t_in + spec.t_out, len(spec.forcing_channels)),
               "target": (batch, spec.t_out, spec.k)}
    out = {key: torch.randn(*shape, height, width, generator=generator) for key, shape in tensors.items()}
    out["static"] = torch.rand(batch, len(spec.static_channels), height, width, generator=generator)
    return {key: value.to(device) for key, value in out.items()}


def timed(fn, runs, device):
    """Seconds per call, measured one call at a time with the GPU synchronized."""
    times = []
    for _ in range(runs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    return np.array(times)


def profile(name, args, height, width, train_windows, device):
    cfg = load_config(args.config, model=name)
    seed_everything(cfg["seed"])
    spec = model_spec(cfg)
    amp = cfg["train"].get("amp", "none")
    model = build_model(cfg).to(device)
    parameters, backend = count_parameters(model), getattr(model, "backend", "")
    batch_size = cfg["train"]["batch_size"]
    batch = random_batch(spec, batch_size, height, width, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train_step():
        with autocast(device, amp):
            pred = model(batch)
        loss = (pred.float() - batch["target"]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.train()
    timed(train_step, args.warmup_steps, device)
    torch.cuda.reset_peak_memory_stats(device)
    step = timed(train_step, args.train_steps, device).mean()
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30

    model.eval()
    single = random_batch(spec, 1, height, width, device)

    def forecast():
        with torch.no_grad(), autocast(device, amp):
            model(single)

    timed(forecast, args.warmup_runs, device)
    infer = timed(forecast, args.inference_runs, device) * 1e3
    del model, optimizer, batch
    torch.cuda.empty_cache()
    return {"model": name, "parameters": parameters, "amp": amp,
            "grad_checkpointing": bool(cfg["model"].get("grad_checkpointing", False)), "backend": backend,
            "gpu": torch.cuda.get_device_name(device), "grid": f"{height}x{width}", "batch_size": batch_size,
            "peak_train_memory_gb": round(peak, 2), "train_seconds_per_step": round(step, 4),
            "train_windows": train_windows,
            "train_seconds_per_epoch": round(step * math.ceil(train_windows / batch_size), 3),
            "inference_ms_mean": round(float(infer.mean()), 2), "inference_ms_std": round(float(infer.std()), 2),
            "cmaq_wallclock_seconds": ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/smoke.yaml", help="data config: grid size and training windows")
    parser.add_argument("--models", nargs="+", default=list(REGISTRY))
    parser.add_argument("--train-windows", type=int, default=None, help="default: training windows of --config")
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--inference-runs", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--out", default="results/profile.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("profile_models.py needs a GPU")
    device = torch.device("cuda")

    hourly, splits = load_and_split(load_config(args.config))
    height, width = hourly.state.shape[-2:]
    train_windows = args.train_windows or len(splits.train)
    rows = []
    for name in args.models:
        rows.append(profile(name, args, height, width, train_windows, device))
        row = rows[-1]
        print(f"{name:<10} {row['parameters'] / 1e6:5.2f}M  train {row['train_seconds_per_step']:.3f} s/step, "
              f"{row['peak_train_memory_gb']:.1f} GB  inference {row['inference_ms_mean']:.1f} ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if out.exists():  # keep hand-entered CMAQ times and rows for models not profiled now
        with open(out, newline="") as f:
            kept = list(csv.DictReader(f))
        cmaq = {r["model"]: r.get("cmaq_wallclock_seconds", "") for r in kept}
        for row in rows:
            row["cmaq_wallclock_seconds"] = cmaq.get(row["model"], "")
        kept = [r for r in kept if r["model"] not in args.models]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(kept + rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
