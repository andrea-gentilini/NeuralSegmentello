from data.config import *
from dataset import CoarseMaskDataset
from u_net_attention_model import Coarse2FineUNet
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

IMG_MODE = "gray"
IMG_GRADIENT = True
IN_CHANNELS = 1 + int(IMG_GRADIENT) + (3 if IMG_MODE == "RGB" else 1)

def main() -> None:
    pl.seed_everything(SEED)

    full_dataset = CoarseMaskDataset(
        DATA_ADAPTATION_DIR, 
        transform_type="paintbrush", 
        image_gradient=IMG_GRADIENT,
        mode=IMG_MODE
    )

    total_len = len(full_dataset)
    val_len = int(total_len * (1 - TRAIN_VALID_SPLIT))  # e.g., 10%
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CHECKPOINT_DIR,
        filename="best-checkpoint",
        save_top_k=SAVE_TOP_K,
        save_last=True,  # save last ckpt to resume training
        verbose=True,
        monitor=MONITOR_METRIC,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor=MONITOR_METRIC,
        patience=EPOCHS // 10,
        verbose=True,
        mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
    )

    model = Coarse2FineUNet(
        in_channels=IN_CHANNELS,
        lr=LR,
        starting_loss_weights=STARTING_LOSS_WEIGHTS,
        refinement_penalty=REFINEMENT_PENALTY,
        learnable_weights=False,
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path="checkpoints/best-checkpoint-v1.ckpt",
    )


if __name__ == "__main__":
    main()