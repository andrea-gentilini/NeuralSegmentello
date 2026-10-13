"""Paired coarse/refined-mask evaluation with explicit sample provenance.

Bundled examples and COCO train2014 images are demonstration data. Original
checkpoint training membership is not preserved, so this is not a held-out test.
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def mask_metrics(prediction, target, boundary_tolerance=2.0):
    """Binary Dice/IoU and inner-boundary F1 with Euclidean pixel tolerance.

    Both-empty comparisons score 1; exactly one empty mask/boundary scores 0.
    Boundary matching is symmetric nearest-neighbor matching, not one-to-one.
    """
    pred, truth = np.asarray(prediction) > 0, np.asarray(target) > 0
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("Masks must be two-dimensional with equal shape")
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    overlap = int(np.count_nonzero(pred & truth))
    size = int(pred.sum() + truth.sum())
    union = int(np.count_nonzero(pred | truth))
    dice = 2 * overlap / size if size else 1.0
    iou = overlap / union if union else 1.0

    def boundary(mask):
        eroded = cv2.erode(
            mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return mask & ~eroded.astype(bool)

    bp, bt = boundary(pred), boundary(truth)
    if not bp.any() or not bt.any():
        fscore = float(not bp.any() and not bt.any())
    else:
        to_truth = cv2.distanceTransform((~bt).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        to_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        precision = float((to_truth[bp] <= boundary_tolerance).mean())
        recall = float((to_pred[bt] <= boundary_tolerance).mean())
        fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"dice": dice, "iou": iou, "boundary_f1": fscore}


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_cases(dataset_root, dataset, seed=42, corruption="erode", limit=None):
    """Yield fixed inputs once, shared by every model in a benchmark run."""
    if dataset == "bundled":
        for group in ("plot_dset", "goku_dset"):
            for path in sorted((dataset_root / group / "coarse").glob("input_coarse_mask_*.npy")):
                index = path.stem.rsplit("_", 1)[1]
                image = np.load(dataset_root / group / "image" / f"input_image_{index}.npy")
                if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
                    image = np.rint(image * 255).astype(np.uint8)
                coarse = (np.load(path) > 0).astype(np.uint8)
                target = (
                    np.load(dataset_root / group / "gt" / f"ground_truth_mask_{index}.npy") > 0
                ).astype(np.uint8)
                yield {
                    "id": f"{group}-{index}",
                    "image": image,
                    "coarse": coarse,
                    "target": target,
                    "source": str(path.relative_to(dataset_root)),
                    "positive_class": "background (stored mask convention)"
                    if group == "goku_dset"
                    else "object foreground",
                    "image_id": None,
                }
    else:
        from .dataset import CoarseMaskDataset

        data = CoarseMaskDataset(
            dataset_root / "reduced_dset",
            transform_type=corruption,
            mode="rgb",
            image_gradient=False,
            seed=seed,
        )
        for index, image_id in enumerate(data.ids):
            if limit is not None and index >= limit:
                break
            inputs, target = data[index]
            yield {
                "id": f"coco-{image_id}",
                "image": np.rint(inputs[1:].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                "coarse": inputs[0].numpy().astype(np.uint8),
                "target": target[0].numpy().astype(np.uint8),
                "source": data.coco.imgs[image_id]["file_name"],
                "positive_class": "largest COCO object foreground",
                "image_id": image_id,
            }


def summarize(rows):
    summary = {}
    for metric in ("dice", "iou", "boundary_f1"):
        coarse = np.array([row[f"coarse_{metric}"] for row in rows])
        refined = np.array([row[f"refined_{metric}"] for row in rows])
        delta = refined - coarse
        summary[metric] = {
            "coarse_mean": float(coarse.mean()),
            "refined_mean": float(refined.mean()),
            "paired_delta_mean": float(delta.mean()),
            "paired_delta_median": float(np.median(delta)),
            "improved": int((delta > 1e-12).sum()),
            "degraded": int((delta < -1e-12).sum()),
            "tied": int((abs(delta) <= 1e-12).sum()),
        }
    latencies = [value for row in rows for value in row["latency_ms"]]
    summary["latency_ms"] = {
        "mean": float(np.mean(latencies)),
        "p50": float(np.median(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "observations": len(latencies),
    }
    return summary


def main(argv=None):
    from .inference import DEFAULT_MODEL, Refiner, synchronize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bundled", "coco"], default="bundled")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset/COCO_dset_adaptation",
    )
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corruption", choices=["v1", "erode", "paintbrush", "dist"], default="erode"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=float, default=2.0)
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeats, threads and limit must be positive")
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cases = list(
        load_cases(args.dataset_root, args.dataset, args.seed, args.corruption, args.limit)
    )
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        parser.error("No evaluation samples found")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_inventory = {"evaluated_cases": len(cases)}
    if args.dataset == "coco":
        annotation_path = args.dataset_root / "reduced_dset/annotations/instances_train2014.json"
        annotation_bytes = annotation_path.read_bytes()
        annotation_data = json.loads(annotation_bytes)
        annotated_ids = {annotation["image_id"] for annotation in annotation_data["annotations"]}
        all_ids = {image["id"] for image in annotation_data["images"]}
        dataset_inventory.update(
            {
                "source_image_count": len(all_ids),
                "annotated_image_count": len(all_ids & annotated_ids),
                "excluded_unannotated_image_ids": sorted(all_ids - annotated_ids),
                "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                "target_selection": "Largest-area annotation per source image",
            }
        )
    report = {
        "schema_version": 1,
        "dataset_inventory": dataset_inventory,
        "dataset": args.dataset,
        "sample_count": len(cases),
        "seed": args.seed,
        "requested_models": args.models,
        "status": "running",
        "positive_class_note": (
            "Scores follow each stored mask: plot examples mark object foreground; goku marks background. Inspect per-group scores rather than treating their mean as one foreground task."
            if args.dataset == "bundled"
            else "All targets mark the largest annotated COCO object."
        ),
        "provenance": (
            "Bundled COCO train2014 images; historical checkpoint training/validation membership is unavailable."
            if args.dataset == "coco"
            else "Three bundled illustrative examples; original source split is unverified."
        ),
        "held_out_test": False,
        "corruption": args.corruption if args.dataset == "coco" else "stored paired masks",
        "settings": {
            "threshold": args.threshold,
            "boundary_tolerance_pixels": args.boundary_tolerance,
            "max_side": args.max_side,
            "threads": args.threads,
            "device": args.device,
            "repeats": args.repeats,
        },
        "latency_definition": "Wall time per predict call, including preprocessing, device transfers, model and postprocessing; synchronized accelerator; excludes checkpoint load, metrics and file I/O; one unmeasured first-case warmup per model.",
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "models": {},
    }
    csv_rows = []
    for name in args.models:
        refiner = Refiner(model_name=name, device=args.device, models_dir=args.models_dir)
        first = cases[0]
        refiner.predict(
            first["image"], first["coarse"], threshold=args.threshold, max_side=args.max_side
        )
        synchronize(refiner.device)
        rows = []
        for case in cases:
            latencies = []
            for _ in range(args.repeats):
                synchronize(refiner.device)
                start = time.perf_counter()
                prediction = refiner.predict(
                    case["image"], case["coarse"], threshold=args.threshold, max_side=args.max_side
                )
                synchronize(refiner.device)
                latencies.append((time.perf_counter() - start) * 1000)
            coarse = mask_metrics(case["coarse"], case["target"], args.boundary_tolerance)
            refined = mask_metrics(prediction.mask, case["target"], args.boundary_tolerance)
            row = {
                "id": case["id"],
                "source": case["source"],
                "positive_class": case["positive_class"],
                "image_id": case["image_id"],
                "shape": list(case["target"].shape),
                "image_sha256": array_hash(case["image"]),
                "coarse_sha256": array_hash(case["coarse"]),
                "target_sha256": array_hash(case["target"]),
                "prediction_sha256": array_hash(prediction.mask),
                "latency_ms": latencies,
                **{f"coarse_{key}": value for key, value in coarse.items()},
                **{f"refined_{key}": value for key, value in refined.items()},
            }
            rows.append(row)
            csv_rows.append(
                {
                    "model": name,
                    "case": case["id"],
                    **{
                        key: row[key]
                        for key in row
                        if key.startswith(("coarse_", "refined_")) and not key.endswith("sha256")
                    },
                    "latency_ms_median": float(np.median(latencies)),
                }
            )
            if args.save_predictions:
                destination = args.output / "predictions" / name
                destination.mkdir(parents=True, exist_ok=True)
                np.save(destination / f"{case['id']}.npy", prediction.mask)
        report["models"][name] = {
            "model_info": refiner.info(),
            "summary": summarize(rows),
            "cases": rows,
        }
        if args.dataset == "bundled":
            report["models"][name]["group_summaries"] = {
                group: summarize([row for row in rows if row["id"].startswith(group)])
                for group in ("plot_dset", "goku_dset")
                if any(row["id"].startswith(group) for row in rows)
            }
        if len(report["models"]) == len(args.models):
            report["status"] = "complete"
        # Preserve completed models if a subsequent checkpoint fails.
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(name, json.dumps(report["models"][name]["summary"]))
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
