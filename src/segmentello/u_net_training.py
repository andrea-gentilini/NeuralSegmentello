from data.config import *
from dataset import CoarseMaskDataset
from u_net_attention_model import Coarse2FineUNet
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping



def main() -> None:
    pl.seed_everything(SEED)

    dataset_gray = CoarseMaskDataset(DATA_ADAPTATION_DIR, transform_type="v2")
    gray_dl = DataLoader(dataset_gray, batch_size=1, shuffle=True)  # batch_size=1 per immagini variabili

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_CHECKPOINT_DIR,
        filename="best-checkpoint",
        save_top_k=SAVE_TOP_K,
        verbose=True,
        monitor=MONITOR_METRIC,
        mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor=MONITOR_METRIC,
        patience=10,
        verbose=True,
        mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
    )

    model = Coarse2FineUNet()
    
    trainer.fit(
        model,
        train_dataloaders=gray_dl,
        valid_dataloaders=gray_dl,
    )


if __name__ == "__main__":
    main()