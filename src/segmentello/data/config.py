import os

# training config
BATCH_SIZE: int = 1
LR: float = 1e-3
NUM_WORKERS: int = 2
EPOCHS: int = 50
SEED: int = 42
TRAIN_VALID_SPLIT: float = 0.9
STARTING_LOSS_WEIGHTS = [0.3, 0.2, 0.2, 0.3]
ORDER_LOSS_WEIGHTS = ["bce", "dice", "boundary", "refine"]
TRANSFORM_MODE = 'erode'
IMG_MODE = "gray"
IMG_GRADIENT = True
IN_CHANNELS = 1 + int(IMG_GRADIENT) + (3 if IMG_MODE == "RGB" else 1)

# lightning callbacks
SAVE_TOP_K: int = 1
MONITOR_METRIC: str = "val_loss"
MODEL_CHECKPOINT_DIR: str = "checkpoints/"

# loss configs
REFINEMENT_PENALTY = {
    'recover': 0,         # GT=1, coarse=0
    'delete': 0,          # GT=0, coarse=1
    'hallucinate': 2.0,     # GT=0, coarse=0
    'soft_penalty': 1.0     # GT=0, coarse=1
}

DATA_ADAPTATION_DIR = os.path.join("COCO_dset_adaptation", "reduced_dset_1000")