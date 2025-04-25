# training config
BATCH_SIZE: int = 4
LR: float = 1e-3
NUM_WORKERS: int = 2
EPOCHS: int = 100
SEED: int = 42
TRAIN_VALID_SPLIT: float = 0.8

# lightning callbacks
SAVE_TOP_K: int = 1
MONITOR_METRIC: str = "val_loss"
MODEL_CHECKPOINT_DIR: str = "checkpoints/"

# TODO add train, valid folders and other constants here