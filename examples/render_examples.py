"""Render illustrative comparisons with the same overlays as the browser demo.

By default this runs the demo’s default CPU checkpoint. To reuse a bundled benchmark:
  python examples/render_examples.py --predictions /path/to/benchmark-bundled/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from segmentello.demo import (
    COARSE_COLOR,
    DEFAULT_MODEL,
    REFINED_COLOR,
    binary_mask,
    load_example,
    overlay,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("Suitcase", "photo-1", "plot_dset", 0),
    ("Baseball", "photo-2", "plot_dset", 1),
)


def dice(mask: np.ndarray, target: np.ndarray) -> float:
    denominator = np.count_nonzero(mask) + np.count_nonzero(target)
    return 2 * np.count_nonzero(mask & target) / denominator if denominator else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--predictions", type=Path, help="Reuse saved predictions instead of running inference."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/readme_figs/refinement-examples.png"
    )
    args = parser.parse_args()
    refiner = None
    if args.predictions is None:
        import torch

        from segmentello.inference import Refiner

        torch.set_num_threads(4)
        refiner = Refiner(args.model, device="cpu")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 3, figsize=(11, 11), facecolor="#f8f9f5")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.85, bottom=0.075, hspace=0.23, wspace=0.07)
    fig.text(0.035, 0.959, "NEURALSEGMENTELLO", size=10, color="#65766b", weight="bold")
    fig.text(
        0.035, 0.925, "Refining an existing selection", size=26, color="#233d33", weight="medium"
    )
    fig.text(
        0.035,
        0.890,
        f"Illustrative examples · {args.model} · threshold 0.5",
        size=11,
        color="#627167",
    )
    for row, (label, key, dataset, index) in enumerate(CASES):
        image, mask_image = load_example(key)
        coarse = binary_mask(mask_image, image.size).astype(bool)
        if refiner is not None:
            prediction = refiner.predict(image, coarse, threshold=0.5, max_side=640).mask.astype(
                bool
            )
        else:
            prediction = np.load(
                args.predictions / args.model / f"{dataset}-{index}.npy", allow_pickle=False
            ).astype(bool)
        target_path = (
            ROOT
            / "dataset/COCO_dset_adaptation"
            / dataset
            / "gt"
            / f"ground_truth_mask_{index}.npy"
        )
        target = np.load(target_path, allow_pickle=False) > 0
        panels = [
            image,
            overlay(image, coarse, COARSE_COLOR),
            overlay(image, prediction, REFINED_COLOR),
        ]
        scores = [label, f"Dice {dice(coarse, target):.3f}", f"Dice {dice(prediction, target):.3f}"]
        for col, (panel, score) in enumerate(zip(panels, scores)):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(
                    ["Input image", "Coarse mask", "Predicted mask"][col],
                    pad=13,
                    loc="left",
                    color="#233d33",
                    size=12,
                    weight="bold",
                )
            ax.text(0, -0.065, score, transform=ax.transAxes, color="#627167", size=11)
    fig.text(
        0.035,
        0.027,
        "Predictions can improve or worsen a mask. These bundled cases have no verified held-out split.",
        color="#627167",
        size=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
