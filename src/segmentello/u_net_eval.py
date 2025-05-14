from data.config import *
from u_net_training import CoarseMaskDataset, Coarse2FineUNet
from u_net_small import Coarse2FineUNetSmall
from u_net_tiny import Coarse2FineTiny
from u_net_tiny_res import Coarse2FineTinyRes
import torch
import pandas as pd
import matplotlib.pyplot as plt



def plot_result(
    dataset,
    model, 
    num_samples = 5, 
    threshold = 0.5
):
    n_rows = min(num_samples, len(dataset))
    n_cols = 4  # input image, input mask, target mask, predicted mask

    plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for i in range(n_rows):  # FIXME, sample in enumerate(dataset[:n_rows]):
        item, target = dataset[i+23]
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
            item[0].cpu(),                 # input image
            item[1].cpu(),                 # input coarse mask
            target[0].cpu(),               # ground truth mask
            predicted_mask[0][0].cpu()     # predicted mask
        ]
        titles = ["Input image", "Input coarse mask", "Ground truth mask", "Predicted mask"]

        for j in range(n_cols):
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            ax.imshow(images[j], cmap='gray')
            # ax.set_title(titles[j])
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_checkpoint(ckpt_dir: str) -> None:
    metrics_csv: str = os.path.join(ckpt_dir, "metrics.csv")
    df: pd.DataFrame = pd.read_csv(metrics_csv)

    train_df = df[df['train_loss'].notnull()]
    val_df = df[df['val_loss'].notnull()]

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['step'], train_df['train_loss'], label='Train Loss', marker='o')
    plt.plot(val_df['step'], val_df['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    # plt.yscale("log")
    plt.title('Train and Validation Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for loss in [
        "loss_bce",
        # "loss_boundary",
        "loss_dice",
        # "loss_refine",
        # "val_iou"
    ]:
        plt.plot(df["step"], df[loss])
        plt.title(loss)
        # plt.yscale("log")
        plt.show()

    plt.plot(val_df["step"], val_df["val_iou"])
    plt.xlabel("Step")
    plt.ylabel("Validation Intersection over Union")
    plt.show()


def main() -> None:

    # dataset_gray = CoarseMaskDataset(DATA_ADAPTATION_DIR, transform_type="v2")
    dataset_gray = CoarseMaskDataset(
        DATA_ADAPTATION_DIR, 
        transform_type="erode",
        image_gradient=True,
    )

    # model_path: str = "lightning_logs/version_1/checkpoints/epoch=9-step=490.ckpt"
    # model_path: str = "checkpoints/erode_13052025/best-checkpoint.ckpt"
    # model_path: str = "checkpoints/erode_14052025/best-checkpoint.ckpt"
    model_path: str = "checkpoints/u_net_tiny/best-checkpoint.ckpt"
    model_path: str = "checkpoints/u_net_tiny_res/best-checkpoint.ckpt"
    # model = Coarse2FineUNet.load_from_checkpoint(model_path)
    # model = Coarse2FineUNetSmall.load_from_checkpoint(model_path)
    # model = Coarse2FineTiny.load_from_checkpoint(model_path)
    model = Coarse2FineTinyRes.load_from_checkpoint(model_path)

    # sample = dataset_gray[0]
    # item = sample[0]
    # target = sample[1]
    # print(f"{item.shape = }")
    # print(f"{target.shape = }")
    # input()

    # with torch.no_grad():
        # predicted = model(item.unsqueeze(0))

    plot_result(dataset_gray, model, num_samples=5)

if __name__ == "__main__":
    main()
    evaluate_checkpoint("checkpoints/u_net_tiny_res")