# Demo inputs

These PNGs are direct, deterministic conversions of arrays already present in this repository:

- `photo-1*`: `dataset/COCO_dset_adaptation/plot_dset/`, index 0.
- `photo-2*`: `dataset/COCO_dset_adaptation/plot_dset/`, index 1.

Floating-point images in the range 0–1 are scaled to 0–255; integer images retain their values. Coarse masks are stored as white foreground and black background. The pictures retain the original material's rights; the repository's code license does not confer additional image rights.

These are illustrative inputs, not a verified held-out test set. No model predictions are stored in these input files.

The original `goku_dset` is not included in the interactive examples: its positive class denotes the background, unlike the object selection used here.
