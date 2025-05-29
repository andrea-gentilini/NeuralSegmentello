import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data.config import DATA_ADAPTATION_DIR
from dataset import CoarseMaskDataset
from u_net import Coarse2FineTiny
from u_net_res_attention import Coarse2FineUNetAttention
from u_net_residual import Coarse2FineTinyRes


def plot_result(dataset, model, num_samples=5, threshold=0.5):
    n_rows = min(num_samples, len(dataset))
    n_cols = 4  # input image, input mask, target mask, predicted mask

    plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for i in range(n_rows):
        item, target = dataset[i + 89]  # FIXME
        # item = sample[0]
        # target = sample[1]
        # print(f"{item.shape = }")
        # print(f"{target.shape = }")
        # input()

        with torch.no_grad():
            predicted = model(item.unsqueeze(0))
            predicted_probs = torch.sigmoid(predicted)
            predicted_mask = (predicted_probs > threshold).float()

        images = [
            item[0].cpu(),  # input image
            item[1].cpu(),  # input coarse mask
            target[0].cpu(),  # ground truth mask
            predicted_mask[0][0].cpu(),  # predicted mask
        ]
        titles = [
            "Input image",
            "Input coarse mask",
            "Ground truth mask",
            "Predicted mask",
        ]

        for j in range(n_cols):
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            ax.imshow(images[j], cmap="gray")
            # ax.set_title(titles[j])
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_losses(best_model_metrics: str, worst_model_metrics: str) -> None:
    best_df = pd.read_csv(best_model_metrics)
    worst_df = pd.read_csv(worst_model_metrics)

    best_train_df = best_df[best_df["train_loss"].notnull()]
    best_valid_df = best_df[best_df["val_loss"].notnull()]
    worst_train_df = worst_df[worst_df["train_loss"].notnull()]
    worst_valid_df = worst_df[worst_df["val_loss"].notnull()]

    window_size = 80
    best_train_df["train_loss_smooth"] = (
        best_train_df["train_loss"].rolling(window=window_size, min_periods=1).mean()
    )
    worst_train_df["train_loss_smooth"] = (
        worst_train_df["train_loss"].rolling(window=window_size, min_periods=1).mean()
    )

    y_min = 0
    y_max = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Training and Validation Loss Comparison", fontsize=22)

    axes[0].plot(
        best_train_df["step"],
        best_train_df["train_loss_smooth"],
        label="Train Loss (smoothed)",
    )
    axes[0].plot(
        best_valid_df["step"],
        best_valid_df["val_loss"],
        label="Valid Loss",
        linewidth=2,
    )
    axes[0].set_title("'attn32-256' (Best Model)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Log Loss")
    axes[0].grid(True)
    axes[0].set_yscale("log")
    axes[0].set_ylim([y_min, y_max])
    axes[0].legend()

    axes[1].plot(
        worst_train_df["step"],
        worst_train_df["train_loss_smooth"],
        label="Train Loss (smoothed)",
    )
    axes[1].plot(
        worst_valid_df["step"],
        worst_valid_df["val_loss"],
        label="Valid Loss",
        linewidth=2,
    )
    axes[1].set_title("'res16-128' (Worst Model)")
    axes[1].set_xlabel("Step")
    axes[1].grid(True)
    axes[1].set_yscale("log")
    axes[1].set_ylim([y_min, y_max])
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def main() -> None:
    # dataset_gray = CoarseMaskDataset(DATA_ADAPTATION_DIR, transform_type="v2")
    # dataset_gray = CoarseMaskDataset(
    #     DATA_ADAPTATION_DIR,
    #     transform_type="erode",
    #     image_gradient=True,
    # )

    # model_path: str = "checkpoints/u_net_tiny_dice_bound/best-checkpoint.ckpt"
    # model = Coarse2FineTiny.load_from_checkpoint(model_path)

    # plot_result(dataset_gray, model, num_samples=5)

    plot_losses("best_metrics.csv", "worst_metrics.csv")


if __name__ == "__main__":
    main()
