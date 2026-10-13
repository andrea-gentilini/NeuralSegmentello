"""Offline CPU tests using shipped checkpoints; no model downloads."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from segmentello.cli import main
from segmentello.inference import DEFAULT_MODEL, Refiner, list_models, resolve_device


@pytest.fixture(scope="module")
def refiner():
    return Refiner(DEFAULT_MODEL, device="cpu")


def inputs(height=31, width=47):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, width // 2 :] = 255
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
    return image, mask


@pytest.mark.parametrize("shape", [(31, 47), (32, 48), (1, 1), (7, 23)])
def test_original_size_and_probability_range(refiner, shape):
    image, coarse = inputs(*shape)
    result = refiner.predict(image, coarse)
    assert result.mask.shape == result.probability.shape == shape
    assert result.mask.dtype == np.uint8
    assert result.probability.dtype == np.float32
    assert np.isfinite(result.probability).all()
    assert ((result.probability >= 0) & (result.probability <= 1)).all()
    np.testing.assert_array_equal(result.mask, result.probability >= 0.5)


def test_pil_and_numpy_and_mask_scales_agree(refiner):
    image, coarse = inputs()
    a = refiner.predict(image, coarse)
    b = refiner.predict(Image.fromarray(image), Image.fromarray(coarse * 255))
    np.testing.assert_allclose(a.probability, b.probability, atol=0, rtol=0)
    c = refiner.predict(image.astype(np.float32) / 255, coarse.astype(bool))
    np.testing.assert_allclose(a.probability, c.probability, atol=0, rtol=0)


def test_resize_restores_shape_and_threshold_is_probability(refiner):
    image, coarse = inputs(53, 71)
    result = refiner.predict(image, coarse, max_side=24, threshold=0)
    assert result.mask.shape == (53, 71)
    assert max(result.metadata["inference_shape"]) == 24
    assert result.mask.all()
    assert refiner.predict(image, coarse, threshold=1).mask.sum() == 0


@pytest.mark.parametrize("case", ["mask_shape", "nonbinary", "nan", "rgba", "threshold", "resize"])
def test_invalid_inputs_fail_clearly(refiner, case):
    image, coarse = inputs()
    kwargs = {}
    if case == "mask_shape":
        coarse = coarse[:-1]
    elif case == "nonbinary":
        coarse = coarse.astype(float) + 0.25
    elif case == "nan":
        image = image.astype(float)
        image[0, 0, 0] = np.nan
    elif case == "rgba":
        image = np.zeros((*coarse.shape, 4), dtype=np.uint8)
    elif case == "threshold":
        kwargs["threshold"] = 1.1
    else:
        kwargs["max_side"] = 0
    with pytest.raises(ValueError):
        refiner.predict(image, coarse, **kwargs)


def test_registry_and_safe_loading(refiner, monkeypatch):
    assert len(list_models()) == 8
    assert all(s.checkpoint.is_file() for s in list_models())
    assert refiner.parameter_count > 0
    original = torch.load

    def load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        assert kwargs["map_location"] == "cpu"
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    second = Refiner(DEFAULT_MODEL, device="cpu")
    assert refiner.state_sha256 == second.state_sha256


def test_missing_and_incompatible_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError):
        Refiner(device="cpu", checkpoint=tmp_path / "missing.ckpt")
    bad = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"unexpected": torch.ones(1)}}, bad)
    with pytest.raises(RuntimeError):
        Refiner(device="cpu", checkpoint=bad)
    with pytest.raises(ValueError, match="Unknown model"):
        Refiner("not-a-model", device="cpu")


def test_tiny_forward_compatible_with_existing_checkpoint():
    model = Refiner("tiny16-32_bce-dice", device="cpu")
    image, mask = inputs(35, 41)
    assert model.predict(image, mask).mask.shape == mask.shape


def test_device_validation():
    assert str(resolve_device("cpu")) == "cpu"
    with pytest.raises(ValueError):
        resolve_device("meta")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda")


def test_cli_predict_and_info(tmp_path, capsys):
    image, coarse = inputs()
    image_path, mask_path = tmp_path / "image.png", tmp_path / "coarse.npy"
    Image.fromarray(image).save(image_path)
    np.save(mask_path, coarse)
    output, probability, metadata = (
        tmp_path / "mask.png",
        tmp_path / "probs.npy",
        tmp_path / "meta.json",
    )
    assert (
        main(
            [
                "predict",
                "--image",
                str(image_path),
                "--mask",
                str(mask_path),
                "--output",
                str(output),
                "--probability-output",
                str(probability),
                "--metadata-output",
                str(metadata),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert np.asarray(Image.open(output)).shape == coarse.shape
    assert np.load(probability).shape == coarse.shape
    assert json.loads(metadata.read_text())["threshold"] == 0.5
    capsys.readouterr()
    assert main(["info"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 8


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("u_net", "Coarse2FineTiny"),
        ("u_net_residual", "Coarse2FineTinyRes"),
        ("u_net_res_attention", "Coarse2FineUNetAttention"),
    ],
)
def test_validation_thresholds_probabilities_not_logits(monkeypatch, module_name, class_name):
    import importlib

    module = importlib.import_module(f"segmentello.{module_name}")
    model = getattr(module, class_name)(features=[16, 32])
    monkeypatch.setattr(model, "forward", lambda x: torch.full((1, 1, 4, 4), 0.25))
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    checked = []

    def iou(pred, target):
        # Positive logits represent p>0.5 even when the raw logit is below0.5.
        assert pred.sum().item() == 16
        checked.append(True)
        return torch.tensor(1.0)

    monkeypatch.setattr(module, "compute_iou", iou)
    model.validation_step((torch.zeros(1, 3, 4, 4), torch.ones(1, 1, 4, 4)), 0)
    assert checked


@pytest.mark.parametrize("case", ["same_output", "input_alias", "bad_probability", "hardlink"])
def test_cli_rejects_unsafe_outputs_before_loading_or_writing(tmp_path, monkeypatch, case):
    import os

    from segmentello import cli

    image = tmp_path / "image.png"
    mask = tmp_path / "coarse.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(image)
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(mask)
    output = tmp_path / "result.png"
    args = ["predict", "--image", str(image), "--mask", str(mask), "--output", str(output)]
    if case == "same_output":
        args += ["--metadata-output", str(output)]
    elif case == "input_alias":
        args += ["--metadata-output", str(image)]
    elif case == "bad_probability":
        args += ["--probability-output", str(tmp_path / "invalid.txt")]
    else:
        os.link(image, output)
    original_image = image.read_bytes()

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("Invalid output paths must be rejected before loading a model.")

    monkeypatch.setattr(cli, "Refiner", unexpected_model_load)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert image.read_bytes() == original_image
    assert output.exists() == (case == "hardlink")


def test_exif_orientation_is_respected_in_api_and_cli(tmp_path):
    from segmentello.cli import _load_array
    from segmentello.inference import _binary_mask, _image_uint8

    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(pixels)
    image.getexif()[274] = 6  # 90 degrees clockwise in display coordinates.
    expected = np.rot90(pixels, k=3)
    np.testing.assert_array_equal(_image_uint8(image), expected)
    image_path = tmp_path / "oriented.png"
    image.save(image_path, exif=image.getexif())
    np.testing.assert_array_equal(_load_array(str(image_path), True), expected)
    mask = Image.fromarray(np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8))
    mask.getexif()[274] = 6
    np.testing.assert_array_equal(_binary_mask(mask, (3, 2)), np.rot90(np.asarray(mask) > 0, k=3))


def test_complex_pixels_fail_without_silent_conversion():
    from segmentello.inference import _binary_mask, _image_uint8

    with pytest.raises(ValueError, match="numeric"):
        _image_uint8(np.zeros((2, 3, 3), dtype=np.complex64))
    with pytest.raises(ValueError, match="numeric"):
        _binary_mask(np.zeros((2, 3), dtype=np.complex64), (2, 3))


def test_state_digest_supports_bfloat16_and_scalar_tensors():
    from segmentello.inference import state_dict_digest

    state = {"weights": torch.tensor([1, 2], dtype=torch.bfloat16), "counter": torch.tensor(3)}
    assert state_dict_digest(state) == state_dict_digest(
        {key: value.clone() for key, value in state.items()}
    )
    assert state_dict_digest(state) != state_dict_digest({**state, "counter": torch.tensor(4)})


@pytest.mark.parametrize("model_name", [spec.name for spec in list_models()])
def test_every_bundled_checkpoint_produces_finite_probabilities(model_name):
    refiner = Refiner(model_name, device="cpu")
    image, coarse = inputs(17, 19)
    result = refiner.predict(image, coarse)
    assert result.probability.shape == coarse.shape
    assert np.isfinite(result.probability).all()
    assert set(np.unique(result.mask)).issubset({0, 1})
