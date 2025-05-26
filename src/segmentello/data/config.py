from pathlib import Path


# training config
BATCH_SIZE: int = 1
LR: float = 1e-3
NUM_WORKERS: int = 2
EPOCHS: int = 50
PATIENCE: int = EPOCHS // 10
SEED: int = 42
TRAIN_VALID_SPLIT: float = 0.1
STARTING_LOSS_WEIGHTS: list[float] = [0.3, 0.3, 0.3]
ORDER_LOSS_WEIGHTS: list[str] = ["bce", "dice", "boundary", "refine"]
TRANSFORM_MODE: str = "erode"
IMG_MODE: str = "gray"
IMG_GRADIENT: bool = True
IN_CHANNELS: int = 1 + int(IMG_GRADIENT) + (3 if IMG_MODE == "RGB" else 1)

# lightning callbacks
SAVE_TOP_K: int = 1
MONITOR_METRIC: str = "val_loss"
MODEL_CHECKPOINT_DIR: str = "checkpoints"

# dirs
DIR_ROOT: Path = Path(__file__).parent.parent.parent.parent
DATA_ADAPTATION_DIR: Path = DIR_ROOT / "COCO_dset_adaptation" / "reduced_dset_1000"
MODELS_DIR: Path = DIR_ROOT / "saved_models"