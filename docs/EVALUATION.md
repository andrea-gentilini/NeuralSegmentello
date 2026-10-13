# Evaluation notes

These measurements check the supplied checkpoints and evaluation pipeline. **They do not establish held-out generalization or reproduce the historical report.** No additional model training was used to produce them.

## Protocol

- Data: 49 annotated source images from the 50-image bundled COCO train2014 subset. One unannotated image is excluded. Each target is the largest annotated instance in that image.
- Coarse inputs: the `erode` corruption with a local RNG seeded by `42 + image_id`. Every model receives identical inputs, verified by hashes in the result files. Empty coarse masks remain in the benchmark.
- Inference: original resolution, no resizing, padding to the model stride, probability threshold 0.5. Three repeats per case, one first-case warm-up per model.
- Metrics: macro means over source images. Dice and IoU use binary foreground. Boundary F1 uses one-pixel inner contours and symmetric nearest-boundary matching within **2 pixels**; it is not boundary IoU. Both-empty masks score 1; one-empty scores 0. No background-dominated pixel accuracy is used as the headline metric.
- Timing: CPU wall time for `predict`, including preprocessing, device transfers, network, and postprocessing. Excludes checkpoint loading, metrics, and file I/O. Accelerator calls are synchronized when used. Reported p50 pools all 147 timed calls per model. Shape-dependent timings and local hardware limit portability.
- Environment: macOS 26.5.1, ARM64, Python 3.12.14, PyTorch 2.7.0, NumPy 2.2.5, four PyTorch threads. Models ran sequentially. These are local measurements, not a hardware-independent speed claim.

## Paired COCO subset results

The coarse baseline has mean Dice **0.8546**, IoU **0.8004**, and boundary F1 **0.5900**.

| Model | Dice ↑ | IoU ↑ | Boundary F1 ↑ | Dice improved / worse / tied | CPU p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `res16-32_bce-dice` | 0.8720 | 0.8221 | 0.5834 | 33 / 13 / 3 | 161.3 |
| `tiny16-32_bce-dice` | 0.8752 | 0.8254 | 0.5861 | 34 / 14 / 1 | 165.4 |
| `tiny16-128_bce-dice` | 0.8692 | 0.8186 | 0.5863 | 31 / 15 / 3 | 208.2 |
| `tiny16-128_bce-dice-bound` | 0.8905 | 0.8496 | 0.6834 | 45 / 3 / 1 | 208.6 |
| `res16-128_bce-dice` | 0.8711 | 0.8199 | 0.5735 | 30 / 17 / 2 | 208.3 |
| `res16-128_bce-dice-bound` | 0.8912 | 0.8475 | 0.6646 | 42 / 6 / 1 | 207.0 |
| `res32-256_bce-dice-bound` | 0.8913 | 0.8458 | 0.6564 | 46 / 3 / 0 | 485.3 |
| `attn32-256_bce-dice-bound` | 0.8972 | 0.8543 | 0.6740 | 47 / 2 / 0 | 520.9 |

The default `res16-128_bce-dice-bound` is a practical compromise between the boundary-loss models and the smaller models. This choice uses the included development examples; it is not an independently validated model-selection result. In this run, the attention model has the highest mean Dice, while the smaller boundary-loss Tiny model has the highest boundary F1. Some models improve overlap while slightly worsening boundaries, so these metrics should be read together.

[Full per-image JSON](benchmarks/coco50.json) · [CSV](benchmarks/coco50.csv)

## Bundled qualitative examples

The README shows both object-mask examples in `plot_dset`, using the default checkpoint at probability threshold 0.5. All displayed refined masks are actual predictions. The third stored example, `goku_dset`, marks the **background** as its positive class. It is preserved and scored as stored in the benchmark, but omitted from the object-selection demo to avoid implying the opposite mask convention.

The three-case JSON retains the original class convention. Its aggregate mixes object and background masks and must not be read as a uniform foreground-segmentation score. Each checkpoint improves Dice on two cases and worsens it on one. These examples are too few, and their source splits too uncertain, to support a general performance claim.

[Full per-case JSON](benchmarks/bundled.json) · [CSV](benchmarks/bundled.csv)

Regenerate the README figure with actual inference:

```sh
python examples/render_examples.py
```

## Reproduce the measurements

From an editable installation in the cloned repository:

```sh
python -m segmentello.benchmark \
  --dataset coco --seed 42 --corruption erode --device cpu \
  --threads 4 --repeats 3 \
  --models res16-32_bce-dice tiny16-32_bce-dice \
    tiny16-128_bce-dice tiny16-128_bce-dice-bound \
    res16-128_bce-dice res16-128_bce-dice-bound \
    res32-256_bce-dice-bound attn32-256_bce-dice-bound \
  --output outputs/benchmark-coco --save-predictions
```

Replace `--dataset coco` with `--dataset bundled` to run the three stored cases. Timings will vary. JSON stores hashes of inputs, targets, predictions, and checkpoint tensors so comparisons can check that they used the same data. Masks themselves are saved only with `--save-predictions`.

## Relationship to the 2025 report

The [original report](NeuralSegmentello.pdf) describes 1,000 COCO images, a 900/100 split, and its own results. The original split manifest is unavailable. The 49 images evaluated here could overlap with checkpoint training data, so none of the measurements above are labeled test-set accuracy.

The maintained code also fixes behavior that affects numerical comparisons:

- Tiny U-Net now concatenates the encoder skip feature before each decoder block, matching the existing checkpoint tensor shapes. The previously committed forward pass could not run those checkpoints.
- Validation now thresholds **sigmoid probability at 0.5**. The old validation thresholded the raw logit at 0.5, so archived training CSV metrics use a different decision rule.
- Dataset access keeps source-image identity: a small/empty coarse mask can no longer trigger a recursive replacement by another image.
- Validation corruptions are fixed by image ID and seed. New training runs save their own image-disjoint split and configuration; this does not retroactively establish the split of old checkpoints.

Checkpoint folder names identify the supplied architecture/loss labels. Tensor keys alone do not prove every detail of the historical forward pass or training setup. The inventory and archived weights remain available for inspection.

## Limitations

The data is small and uses synthetic corruptions. Performance on real brush strokes, new domains, tiny objects, or ambiguous selections is not established. The local benchmark does not retrain or fine-tune any model. CUDA and MPS can be selected by the API, but the published automated checks and these measurements exercise CPU execution. A proper future comparison needs a separately held-out, source-image-disjoint dataset and a fixed corruption protocol.
