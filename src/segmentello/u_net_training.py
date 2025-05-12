from data.config import *
from dataset import CoarseMaskDataset
from u_net_attention_model import Coarse2FineUNet

from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl



def main() -> None:
    pl.seed_everything(SEED)

    dataset_gray = CoarseMaskDataset(DATA_ADAPTATION_DIR, transform_type="v2")
    gray_dl = DataLoader(dataset_gray, batch_size=1, shuffle=True)  # batch_size=1 per immagini variabili

    # print(f"{dataset_gray[0][0].shape = }")
    # tmp_img = dataset_gray[0][0][0]
    # tmp_mask = dataset_gray[0][0][1]
    # tmp_ground_truth = dataset_gray[0][1][0]
    # plt.imshow(np.hstack([tmp_img, tmp_mask, tmp_ground_truth]))
    # plt.axis("off")
    # plt.show()

    # raise NotImplementedError

    model = Coarse2FineUNet()
    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1
    )
    trainer.fit(model, gray_dl)


if __name__ == "__main__":
    main()