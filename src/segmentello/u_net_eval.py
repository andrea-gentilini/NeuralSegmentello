from data.config import *
from u_net_training import CoarseMaskDataset, Coarse2FineUNet
import torch
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
        item, target = dataset[i]
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



def main() -> None:

    dataset_gray = CoarseMaskDataset(DATA_ADAPTATION_DIR, transform_type="v2")

    model_path: str = "lightning_logs/version_1/checkpoints/epoch=9-step=490.ckpt"
    model = Coarse2FineUNet.load_from_checkpoint(model_path)

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