from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from segmentello import dataset as datasets
from segmentello.benchmark import load_cases, mask_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/COCO_dset_adaptation"


def test_metrics_empty_identical_disjoint_and_boundary_tolerance():
    empty = np.zeros((20, 20), dtype=np.uint8)
    target = empty.copy()
    target[5:12, 5:12] = 1
    assert mask_metrics(empty, empty) == {"dice": 1, "iou": 1, "boundary_f1": 1}
    assert mask_metrics(empty, target) == {"dice": 0, "iou": 0, "boundary_f1": 0}
    assert mask_metrics(target, target) == {"dice": 1, "iou": 1, "boundary_f1": 1}
    shifted = np.roll(target, 1, axis=1)
    metrics = mask_metrics(shifted, target, boundary_tolerance=1)
    assert metrics["dice"] < 1 and metrics["boundary_f1"] == 1
    assert mask_metrics(shifted, target, boundary_tolerance=0)["boundary_f1"] < 1


def test_source_image_split_is_deterministic_and_disjoint():
    train, valid = datasets.split_image_ids([1, 2, 3, 4, 1, 2], 0.5, 42)
    assert not set(train) & set(valid)
    assert set(train + valid) == {1, 2, 3, 4}
    assert (train, valid) == datasets.split_image_ids([4, 3, 2, 1], 0.5, 42)
    with pytest.raises(ValueError):
        datasets.split_image_ids([1], 0.5, 42)


@pytest.mark.parametrize(
    "transform",
    [datasets.random_coarse, datasets.random_coarse_paintbrush, datasets.random_coarse_erode],
)
def test_coarse_transforms_always_return_binary(transform):
    for mask in (np.zeros((12, 12)), np.ones((12, 12))):
        assert set(np.unique(transform(mask))).issubset({0, 1})


def test_validation_is_repeatable_and_transform_choice_does_not_mutate():
    transforms = ["v1", "erode", "paintbrush", "dist"]
    data = datasets.CoarseMaskDataset(
        DATA / "reduced_dset", transform_type=transforms, seed=42, mode="rgb"
    )
    first = data[0]
    second = data[0]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert data.transform_type == transforms
    assert first[0].shape[0] == 4  # coarse + RGB; negative-stride NumPy inputs must work.


def test_empty_coarse_does_not_substitute_a_different_source_image(monkeypatch):
    monkeypatch.setattr(datasets, "random_coarse_erode", lambda mask, rng=None: np.zeros_like(mask))
    data = datasets.CoarseMaskDataset(DATA / "reduced_dset", transform_type="erode", seed=42)
    item, _ = data[0]
    image = cv2.imread(
        str(Path(data.img_dir) / data.coco.imgs[data.ids[0]]["file_name"]), cv2.IMREAD_COLOR
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert not item[0].any()
    assert np.allclose(item[1].numpy(), image / 255.0)


def test_stored_float_images_are_not_divided_by_255_twice():
    item, target = datasets.SimpleCoarseMaskDataset(DATA / "plot_dset", image_gradient=True)[0]
    original = np.load(DATA / "plot_dset/image/input_image_0.npy")
    assert np.allclose(item[1].numpy(), original)
    assert item.max() <= 1 and target.max() <= 1
    item, target = datasets.SimpleCoarseMaskDataset(DATA / "goku_dset", image_gradient=True)[0]
    assert item[0].max() == 1 and target.max() == 1


def test_bundled_cases_have_matching_spatial_shapes_and_binary_masks():
    cases = list(load_cases(DATA, "bundled"))
    assert len(cases) == 3
    for case in cases:
        assert case["image"].shape[:2] == case["coarse"].shape == case["target"].shape
        assert case["image"].dtype == np.uint8
        assert set(np.unique(case["coarse"])).issubset({0, 1})


def test_reduced_dataset_seed_provenance_and_explicit_overwrite(tmp_path):
    import json

    from segmentello.reducing_dset import create_reduced_coco_dataset

    source = tmp_path / "source"
    (source / "train2014").mkdir(parents=True)
    (source / "annotations").mkdir()
    images = [{"id": i, "file_name": f"{i}.jpg"} for i in range(8)]
    for image in images:
        (source / "train2014" / image["file_name"]).write_bytes(b"source image")
    metadata = {
        "images": images,
        "annotations": [{"id": i, "image_id": i} for i in range(8)],
        "categories": [],
        "info": {"description": "fixture"},
    }
    (source / "annotations/instances_train2014.json").write_text(json.dumps(metadata))
    one, two = tmp_path / "one", tmp_path / "two"
    create_reduced_coco_dataset(source, 3, one, seed=42)
    create_reduced_coco_dataset(source, 3, two, seed=42)
    first = json.loads((one / "subset_manifest.json").read_text())
    second = json.loads((two / "subset_manifest.json").read_text())
    assert first == second
    output = json.loads((one / "annotations/instances_train2014.json").read_text())
    assert output["info"] == metadata["info"]
    assert {ann["image_id"] for ann in output["annotations"]} == set(first["selected_image_ids"])
    assert len(list((one / "train2014").glob("*.jpg"))) == 3
    with pytest.raises(FileExistsError):
        create_reduced_coco_dataset(source, 3, one, seed=42)
    create_reduced_coco_dataset(source, 2, one, seed=42, overwrite=True)
    assert len(list((one / "train2014").glob("*.jpg"))) == 2
    with pytest.raises(ValueError, match="source dataset"):
        create_reduced_coco_dataset(source, 2, source, overwrite=True)
    with pytest.raises(ValueError, match="source contains"):
        create_reduced_coco_dataset(source, 100, tmp_path / "too_many")
    assert not (tmp_path / "too_many").exists()


@pytest.mark.parametrize("existing_kind", ["run", "file"])
def test_training_rejects_existing_output_before_dataset_access(
    tmp_path, monkeypatch, capsys, existing_kind
):
    from segmentello import training

    output = tmp_path / "training"
    if existing_kind == "run":
        (output / "checkpoints").mkdir(parents=True)
        files = {
            output / "split.json": b'{"seed": 17, "train_image_ids": [1]}',
            output / "checkpoints/best-checkpoint.ckpt": b"existing checkpoint",
        }
    else:
        files = {output: b"existing unrelated file"}
    for path, content in files.items():
        path.write_bytes(content)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Existing output must be rejected before dataset or training work")

    monkeypatch.setattr(training, "CoarseMaskDataset", unexpected_work)
    monkeypatch.setattr(training.pl, "Trainer", unexpected_work)
    with pytest.raises(SystemExit) as error:
        training.main(["--output", str(output), "--seed", "42", "--fast-dev-run"])
    assert error.value.code == 2
    assert "--output must be a new or empty directory" in capsys.readouterr().err
    for path, content in files.items():
        assert path.read_bytes() == content
