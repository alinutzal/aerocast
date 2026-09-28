import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from config import load_config, apply_overrides
from data import build_dataset, build_dataloader
from model import build_model
from evaluate import evaluate

try:
    import wandb
except ImportError:
    wandb = None


def build_loss(loss_cfg):
    """Returns (loss_fn, label). Types: "huber" or "mixed" (Huber*alpha + MSE*(1-alpha))."""
    loss_type = loss_cfg["type"]
    beta = loss_cfg["huber_beta"]  # 0.5 => more MAE-like, 1.0 => default SmoothL1
    alpha = loss_cfg["alpha"]
    huber_loss = nn.SmoothL1Loss(beta=beta)
    mse_loss = nn.MSELoss()

    if loss_type == "mixed":
        def compute_loss(pred, target):
            return alpha * huber_loss(pred, target) + (1 - alpha) * mse_loss(pred, target)
        return compute_loss, f"Mixed(Huber(beta={beta}), alpha={alpha}, + MSE)"
    if loss_type == "huber":
        return huber_loss, f"Huber(beta={beta})"
    raise ValueError(f"Unknown loss type '{loss_type}'. Choose 'huber' or 'mixed'.")


def train(model, dataset, dataloader, device, train_cfg, use_wandb=False):
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sched_cfg = train_cfg["scheduler"]
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=sched_cfg["T_0"], T_mult=sched_cfg["T_mult"], eta_min=sched_cfg["eta_min"]
    )
    compute_loss, loss_label = build_loss(train_cfg["loss"])

    # Training with early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = train_cfg["patience"]
    log_every = train_cfg.get("log_every", 5)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        sse_sum = 0.0  # sum of squared errors across all elements
        n_elem = 0     # total number of elements (B*H*W per batch)

        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)  # (B, T, C, H, W) - sequence data
            Y_batch = Y_batch.to(device)  # (B, pred_len, H, W)

            pred = model(X_batch, pred_steps=dataset.pred_len).reshape(Y_batch.shape)  # (B, pred_len, H, W)
            loss = compute_loss(pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip"])
            optimizer.step()

            total_loss += loss.item()
            # accumulate SSE and count for RMSE
            sse_sum += (pred - Y_batch).pow(2).sum().item()
            n_elem += Y_batch.numel()

        avg_loss = total_loss / len(dataloader)
        rmse_epoch = (sse_sum / max(1, n_elem)) ** 0.5
        scheduler.step()

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:03d} - {loss_label}: {avg_loss:.4f} | RMSE: {rmse_epoch:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Best: {best_loss:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/rmse": rmse_epoch,
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/best_loss": best_loss
            })

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} - no improvement for {patience} epochs")
            break

    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Train ConvLSTM grid forecast model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--skip-eval", action="store_true", help="Only train and save the checkpoint")
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. data.normalization_mode=none train.epochs=10")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    data_cfg, model_cfg, train_cfg, wandb_cfg = cfg["data"], cfg["model"], cfg["train"], cfg["wandb"]

    if cfg.get("seed") is not None:
        torch.manual_seed(cfg["seed"])

    print(data_cfg["predictor_vars"])

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = build_dataset(data_cfg)
    dataloader = build_dataloader(dataset, data_cfg)
    model = build_model(model_cfg, input_channels=len(data_cfg["predictor_vars"])).to(device)

    use_wandb = bool(wandb_cfg.get("enabled"))
    if use_wandb and wandb is None:
        raise ImportError("Weights & Biases is not installed. Install with: pip install wandb")

    if use_wandb:
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            config=cfg,
        )
        wandb.watch(model, log="gradients", log_freq=50)

    print(f"\nTraining on {device}")
    print(f"Dataset size: {len(dataset)} samples")
    if dataset.normalization_mode == "none":
        print("Normalization: none\n")
    elif dataset.normalization_mode == "predictors":
        print(f"Normalization: predictors only | X_mean={dataset.X_mean.mean():.4f}\n")
    else:
        print(f"Normalization: full | X_mean={dataset.X_mean.mean():.4f}, Y_mean={dataset.Y_mean:.4f}\n")

    train(model, dataset, dataloader, device, train_cfg, use_wandb=use_wandb)

    checkpoint_path = train_cfg["checkpoint"]
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    if not args.skip_eval:
        evaluate(model, dataset, device, cfg["eval"], use_wandb=use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
