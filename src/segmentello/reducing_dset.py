"""Create a deterministic local COCO subset; never download data implicitly."""

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

from .data.config import DIR_COCO_DSET_ADAPTATION, SEED


def create_reduced_coco_dataset(full_dataset_path, N, output_dir=None, seed=SEED, overwrite=False):
    """Copy N source images and all their annotations, preserving COCO metadata.

    The two original positional arguments remain supported. Existing output is
    replaced only with explicit overwrite=True, and source data is never replaced.
    """
    if N < 1:
        raise ValueError("Subset size must be positive")
    source = Path(full_dataset_path).resolve()
    destination = (
        Path(output_dir)
        if output_dir is not None
        else DIR_COCO_DSET_ADAPTATION / f"reduced_dset_{N}"
    ).resolve()
    if source == destination or source.is_relative_to(destination):
        raise ValueError("Output must not replace the source dataset or one of its parents")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}; pass --overwrite explicitly")
    annotation_path = source / "annotations/instances_train2014.json"
    source_bytes = annotation_path.read_bytes()
    coco = json.loads(source_bytes)
    images = sorted(coco["images"], key=lambda image: image["id"])
    if len({image["id"] for image in images}) != len(images):
        raise ValueError("Source COCO metadata contains duplicate image IDs")
    if N > len(images):
        raise ValueError(f"Requested {N} images but source contains {len(images)}")
    selected = sorted(random.Random(seed).sample(images, N), key=lambda image: image["id"])
    for image in selected:
        name = Path(image["file_name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("COCO file_name must be a relative path without traversal")
        if not (source / "train2014" / name).is_file():
            raise FileNotFoundError(source / "train2014" / name)
    ids = {image["id"] for image in selected}
    reduced = {
        **coco,
        "images": selected,
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in ids],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "annotations").mkdir()
        (staging / "train2014").mkdir()
        for image in selected:
            target = staging / "train2014" / image["file_name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "train2014" / image["file_name"], target)
        (staging / "annotations/instances_train2014.json").write_text(json.dumps(reduced) + "\n")
        (staging / "subset_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "seed": seed,
                    "source": str(source),
                    "source_annotations_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "selected_image_ids": sorted(ids),
                    "sample_count": len(ids),
                    "note": "A source-image subset, not an independently verified held-out split.",
                },
                indent=2,
            )
            + "\n"
        )
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        staging.replace(destination)
    print(f"Created {N}-image subset at {destination}")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    create_reduced_coco_dataset(args.source, args.count, args.output, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
