"""Train with source-image-disjoint splits and fixed validation corruptions.

Run with ``python -m segmentello.training --help``. Saved split IDs describe new
training runs only; they do not establish provenance for the bundled checkpoints.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from .data.config import (
    BATCH_SIZE,
    DIR_REDUCED_DSET,
    EPOCHS,
    IMG_GRADIENT,
    IMG_MODE,
    NUM_WORKERS,
    PATIENCE,
    SEED,
    TRAIN_VALID_SPLIT,
    TRANSFORM_MODE,
)
from .dataset import CoarseMaskDataset, collate_fn, split_image_ids
from .inference import list_models
from .u_net import Coarse2FineTiny
from .u_net_res_attention import Coarse2FineUNetAttention
from .u_net_residual import Coarse2FineTinyRes


def main(argv=None):
    specs = {spec.name: spec for spec in list_models()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DIR_REDUCED_DSET)
    parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    parser.add_argument("--model", choices=list(specs), default="attn32-256_bce-dice-bound")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--validation-fraction", type=float, default=TRAIN_VALID_SPLIT)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory; choose a fresh path for each run")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be strictly between 0 and 1")
    if IMG_MODE.lower() != "gray" or not IMG_GRADIENT:
        parser.error("Bundled architectures require grayscale plus gradient (three input channels)")
    pl.seed_everything(args.seed, workers=True)
    shared = {
        "dataset_dir": args.dataset,
        "transform_type": TRANSFORM_MODE,
        "image_gradient": IMG_GRADIENT,
        "mode": IMG_MODE,
    }
    full = CoarseMaskDataset(**shared)
    train_ids, valid_ids = split_image_ids(full.ids, args.validation_fraction, args.seed)
    dimensions = {(full.coco.imgs[i]["height"], full.coco.imgs[i]["width"]) for i in full.ids}
    if args.batch_size > 1 and len(dimensions) > 1:
        parser.error(
            "Variable-resolution datasets require batch-size=1; no loss padding is applied"
        )
    spec = specs[args.model]
    losses = ["bce", "dice", "boundary"] if args.model.endswith("-bound") else ["bce", "dice"]
    weights = [0.4, 0.4, 0.2] if len(losses) == 3 else [0.5, 0.5]
    train = CoarseMaskDataset(**shared, image_ids=train_ids)
    valid = CoarseMaskDataset(**shared, image_ids=valid_ids, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": str(args.dataset.resolve()),
                "seed": args.seed,
                "train_image_ids": train_ids,
                "validation_image_ids": valid_ids,
                "validation_corruption": "fixed per source image: seed + image_id",
                "note": "This split belongs to this new run, not the bundled checkpoints.",
                "annotations_sha256": hashlib.sha256(Path(full.ann_file).read_bytes()).hexdigest(),
                "hyperparameters": {
                    "model": args.model,
                    "architecture": spec.architecture,
                    "features": list(spec.features),
                    "losses": losses,
                    "loss_weights": weights,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "coarse_transform": TRANSFORM_MODE,
                    "fast_dev_run": args.fast_dev_run,
                    "input_channels": ["coarse", "grayscale", "normalized_sobel"],
                },
            },
            indent=2,
        )
        + "\n"
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_args
    )
    valid_loader = DataLoader(valid, shuffle=False, **loader_args)
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output / "checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        log_every_n_steps=5,
        default_root_dir=str(args.output),
        accelerator=args.accelerator,
        devices=1,
        fast_dev_run=args.fast_dev_run,
    )
    model_class = {
        "tiny": Coarse2FineTiny,
        "residual": Coarse2FineTinyRes,
        "attention": Coarse2FineUNetAttention,
    }[spec.architecture]
    model = model_class(
        lr=args.lr, losses=losses, features=list(spec.features), loss_weights=weights
    )
    # Lightning checkpoints preserve the architecture choices for future loading.
    model.save_hyperparameters(
        {"features": list(spec.features), "losses": losses, "loss_weights": weights, "lr": args.lr}
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
