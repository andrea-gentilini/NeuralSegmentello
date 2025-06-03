import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from data.config import (
    BATCH_SIZE,
    DIR_REDUCED_DSET,
    EPOCHS,
    IMG_GRADIENT,
    IMG_MODE,
    MONITOR_METRIC,
    NUM_WORKERS,
    PATIENCE,
    SAVE_TOP_K,
    SEED,
    TRAIN_VALID_SPLIT,
    TRANSFORM_MODE,
)
from dataset import CoarseMaskDataset, SingleSampleDataset, collate_fn
from u_net import Coarse2FineTiny
from u_net_res_attention import Coarse2FineUNetAttention
from u_net_residual import Coarse2FineTinyRes


def main() -> None:
    pl.seed_everything(SEED)

    full_dataset = CoarseMaskDataset(
        DIR_REDUCED_DSET,
        transform_type=TRANSFORM_MODE,
        image_gradient=IMG_GRADIENT,
        mode=IMG_MODE,
    )

    total_len = len(full_dataset)
    val_len = int(total_len * TRAIN_VALID_SPLIT)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(
        # dirpath=MODEL_CHECKPOINT_DIR,
        filename="best-checkpoint",
        save_top_k=SAVE_TOP_K,
        save_last=True,  # save last ckpt to resume training
        verbose=True,
        monitor=MONITOR_METRIC,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor=MONITOR_METRIC, patience=PATIENCE, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
    )

    # # 1 img dataset
    # dumb_dataset = SingleSampleDataset(full_dataset[0])
    # dumb_dl = DataLoader(
    #     dumb_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS
    # )

    model = Coarse2FineUNetAttention(
        losses=["bce", "dice", "boundary"],
        features=[32, 64, 128, 256],
        loss_weights=[0.4, 0.4, 0.2],
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path="checkpoints/best-checkpoint-v1.ckpt",
    )

    # # 1 img dataset overfitting test
    # model = Coarse2FineTiny()
    # trainer.fit(
    #     model,
    #     train_dataloaders=dumb_dl
    # )


if __name__ == "__main__":
    main()
