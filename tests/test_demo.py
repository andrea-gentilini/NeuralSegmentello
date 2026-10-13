"""Demo boundaries: drawing never becomes model input and downloads stay binary."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from segmentello import demo


def sample():
    image = Image.new("RGB", (7, 5), (65, 92, 123))
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:5] = 1
    return image, mask


def test_editor_preserves_source_and_reads_only_painted_alpha():
    image, mask = sample()
    value = demo.editor_value(image, mask)
    value["composite"] = Image.new("RGBA", image.size, "red")
    actual_image, actual_mask = demo.editor_inputs(value)
    np.testing.assert_array_equal(actual_image, image)
    np.testing.assert_array_equal(actual_mask, mask)


def test_editor_combines_layers_and_rejects_empty_selection():
    image, mask = sample()
    value = demo.editor_value(image)
    with pytest.raises(ValueError, match="Paint over"):
        demo.editor_inputs(value)
    other = demo.editor_value(image, mask)["layers"][0]
    value["layers"].append(other)
    np.testing.assert_array_equal(demo.editor_inputs(value)[1], mask)


def test_editor_rejects_misaligned_or_opaque_layer():
    image, _ = sample()
    value = demo.editor_value(image)
    value["layers"] = [Image.new("RGBA", (1, 1))]
    with pytest.raises(ValueError, match="no longer matches"):
        demo.editor_inputs(value)
    value["layers"] = [Image.new("RGB", image.size)]
    with pytest.raises(ValueError, match="no transparency"):
        demo.editor_inputs(value)


def test_upload_mask_preserves_alignment_and_accepts_binary_formats():
    image, mask = sample()
    for value in (mask, mask * 255, mask.astype(float), Image.fromarray(mask * 255)):
        np.testing.assert_array_equal(demo.binary_mask(value, image.size), mask)
    with pytest.raises(ValueError, match="same width"):
        demo.binary_mask(mask, (8, 5))
    with pytest.raises(ValueError, match="invalid values"):
        demo.binary_mask(mask.astype(float) * np.nan, image.size)


def test_overlay_does_not_modify_image_or_mask():
    image, mask = sample()
    original = np.array(image)
    original_mask = mask.copy()
    result = demo.overlay(image, mask, demo.COARSE_COLOR)
    assert result.size == image.size
    assert not np.array_equal(result, image)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)


def test_refine_returns_aligned_binary_download(tmp_path, monkeypatch):
    image, mask = sample()
    prediction = np.flip(mask, axis=1).copy()
    calls = []

    class FakeRefiner:
        def predict(self, source, coarse, **kwargs):
            calls.append((source, coarse.copy(), kwargs))
            return SimpleNamespace(mask=prediction)

    monkeypatch.setattr(demo, "_refiner", lambda name, models_dir: FakeRefiner())
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    coarse, refined, png, path, status = demo.refine(image, mask, "test", 0.5)
    assert coarse.size == refined.size == png.size == image.size
    with Image.open(path) as saved:
        np.testing.assert_array_equal(saved, prediction * 255)
    assert calls[0][2] == {"threshold": 0.5, "max_side": 640}
    assert "CPU" in status and "7 × 5" in status


def test_refine_rejects_empty_mask_before_loading_model(monkeypatch):
    image, mask = sample()
    monkeypatch.setattr(demo, "_refiner", lambda *args: pytest.fail("Should not load a model"))
    with pytest.raises(ValueError, match="mask is empty"):
        demo.refine(image, mask * 0, "test", 0.5)


@pytest.mark.parametrize("name", [key for _, key in demo.EXAMPLES])
def test_bundled_example_is_aligned_and_nonempty(name):
    image, mask = demo.load_example(name)
    assert image.mode == "RGB"
    assert demo.binary_mask(mask, image.size).any()


def test_optional_gradio_demo_builds_without_running_inference(monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setattr(demo, "_refiner", lambda *args: pytest.fail("Should not load a model"))
    app = demo.build_demo()
    assert app.analytics_enabled is False
    assert app.title == "NeuralSegmentello"
