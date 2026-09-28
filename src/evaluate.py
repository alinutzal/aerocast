import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import load_config, apply_overrides
from data import build_dataset
from model import build_model

try:
    import wandb
except ImportError:
    wandb = None


def plot_sample(target, pred, sample_idx, sample_rmse, output_path):
    pred_len = target.shape[0]
    fig, axes = plt.subplots(2, pred_len, figsize=(3*pred_len, 6), squeeze=False)

    # Use same color scale for all plots in this sample
    vmin = min(target.min().item(), pred.min().item())
    vmax = max(target.max().item(), pred.max().item())

    for step in range(pred_len):
        # Target row
        im1 = axes[0, step].imshow(target[step], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, step].set_title(f"Target t+{step+1}", fontsize=10)
        axes[0, step].axis('off')
        if step == pred_len - 1:
            plt.colorbar(im1, ax=axes[0, step])

        # Prediction row
        im2 = axes[1, step].imshow(pred[step], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, step].set_title(f"Pred t+{step+1}", fontsize=10)
        axes[1, step].axis('off')
        if step == pred_len - 1:
            plt.colorbar(im2, ax=axes[1, step])

    plt.suptitle(f"Multi-Step Forecast: NO2 + O3 - Sample {sample_idx}\nOverall RMSE: {sample_rmse:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def evaluate(model, dataset, device, eval_cfg, use_wandb=False):
    """Predict every sample, print per-step RMSE (physical units) and save one plot per sample."""
    output_dir = eval_cfg["output_dir"]
    print(f"\n[Visualization] Generating predictions for all {len(dataset)} samples")
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_sample_rmse = []
    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            X_sample, Y_sample = dataset[sample_idx]
            X_sample_batch = X_sample.unsqueeze(0).to(device)  # Add batch dim
            pred_sample = model(X_sample_batch, pred_steps=dataset.pred_len).cpu().reshape(Y_sample.shape)  # (pred_len, H, W)

            # Denormalize targets/predictions only when Y was normalized (full mode)
            Y_sample_denorm = dataset.denormalize_y(Y_sample)
            pred_sample_denorm = dataset.denormalize_y(pred_sample)

            # Compute overall RMSE for this sample
            sample_rmse = ((pred_sample_denorm - Y_sample_denorm) ** 2).mean().sqrt().item()
            all_sample_rmse.append(sample_rmse)

            print(f"\nSample {sample_idx}: Overall RMSE = {sample_rmse:.3f}")

            # Compute RMSE per time step
            for step in range(dataset.pred_len):
                step_rmse = ((pred_sample_denorm[step] - Y_sample_denorm[step]) ** 2).mean().sqrt().item()
                print(f"  t+{step+1}: {step_rmse:.3f}")

            if eval_cfg.get("save_plots", True):
                output_path = os.path.join(output_dir, f"grid_forecast_sample_{sample_idx:02d}.png")
                plot_sample(Y_sample_denorm, pred_sample_denorm, sample_idx, sample_rmse, output_path)
                print(f"  Plot saved to: {output_path}")

                if use_wandb:
                    wandb.log({f"eval/sample_{sample_idx:02d}_plot": wandb.Image(output_path)})

            if use_wandb:
                wandb.log({f"eval/sample_{sample_idx:02d}_rmse": sample_rmse})

    avg_rmse = float(np.mean(all_sample_rmse)) if all_sample_rmse else float("nan")
    print(f"\nAverage sample RMSE: {avg_rmse:.3f}")
    if use_wandb and all_sample_rmse:
        wandb.log({"eval/avg_sample_rmse": avg_rmse})

    print(f"All {len(dataset)} sample results saved to {output_dir}/")
    return all_sample_rmse


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained grid forecast checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (defaults to train.checkpoint in the config)")
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. eval.output_dir=results_tmp")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    checkpoint_path = args.checkpoint or cfg["train"]["checkpoint"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(cfg["data"])
    model = build_model(cfg["model"], input_channels=len(cfg["data"]["predictor_vars"])).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    evaluate(model, dataset, device, cfg["eval"])


if __name__ == "__main__":
    main()
