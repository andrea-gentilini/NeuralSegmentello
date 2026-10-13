"""Optional, local browser interface for the existing segmentation checkpoints.

Run ``python -m segmentello.demo`` after installing the ``demo`` extra.
The model and preprocessing live in :mod:`segmentello.inference`.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "demo"
EXAMPLES = (
    ("Suitcase", "photo-1"),
    ("Baseball", "photo-2"),
)
DEFAULT_MODEL = "res16-128_bce-dice-bound"
COARSE_COLOR = (222, 124, 75)
REFINED_COLOR = (39, 132, 121)
MAX_IMAGE_PIXELS = 20_000_000

CSS = """
.gradio-container {max-width: 1220px !important; margin: auto !important;}
body, .gradio-container {font-family: Arial, sans-serif;}
#masthead {padding: 24px 0 16px; border-bottom: 1px solid #d9ded8; margin-bottom: 18px;}
#masthead .eyebrow {font-size: 11px; letter-spacing: .14em; font-weight: 700; color: #65766b;}
#masthead h1 {font-size: clamp(30px, 4vw, 46px); letter-spacing: -.045em;
  line-height: 1.12; margin: 12px 0 !important; color: #233d33; font-weight: 600;}
#masthead p {font-size: 16px; line-height: 1.6; color: #627167; max-width: 650px;}
#refine-button {min-height: 49px; font-size: 15px; font-weight: 600;}
#settings {border: 1px solid #dde3dc; border-radius: 14px; padding: 20px; background: #f6f7f3;}
#result-status {padding: 10px 2px; min-height: 32px; color: #4e6659;}
#note {font-size: 12px; color: #758176; margin: 18px 0 8px;}
#example-row button {border-radius: 24px; min-width: 90px;}
.output-panel {border-radius: 12px !important;}
footer {display: none !important;}
"""


def _rgb(image: Image.Image | np.ndarray | None) -> Image.Image:
    if image is None:
        raise ValueError("Upload an image or choose one of the examples first.")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("This image is too large. Please use an image below 20 megapixels.")
    return ImageOps.exif_transpose(image).convert("RGB")


def binary_mask(mask: Image.Image | np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    """Read an uploaded white-foreground mask without silently stretching it."""
    if mask is None:
        raise ValueError("Upload a mask: white for the object, black for the background.")
    if isinstance(mask, Image.Image):
        mask = ImageOps.exif_transpose(mask)
        if mask.size != size:
            raise ValueError("The image and mask must have the same width and height.")
        values = np.asarray(mask.convert("L"))
    else:
        values = np.asarray(mask)
        if values.ndim == 3:
            values = values[..., :3].mean(axis=-1)
        if values.shape != (size[1], size[0]):
            raise ValueError("The image and mask must have the same width and height.")
    if not np.isfinite(values).all():
        raise ValueError("The mask contains invalid values.")
    return (values > (0.5 if values.max(initial=0) <= 1 else 127)).astype(np.uint8)


def editor_inputs(editor: dict[str, Any] | None) -> tuple[Image.Image, np.ndarray]:
    """Keep the source pixels untouched and use only painted-layer opacity."""
    if not editor or editor.get("background") is None:
        raise ValueError("Upload an image or choose one of the examples first.")
    image = _rgb(editor["background"])
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for layer in editor.get("layers", []):
        if layer is None:
            continue
        if not isinstance(layer, Image.Image):
            layer = Image.fromarray(np.asarray(layer))
        if layer.size != image.size:
            raise ValueError(
                "The painted mask no longer matches the image. Reload the image and try again."
            )
        if "A" not in layer.getbands():
            raise ValueError(
                "The drawing layer has no transparency. Use the mask-upload tab instead."
            )
        mask |= (np.asarray(layer.getchannel("A")) > 0).astype(np.uint8)
    if not mask.any():
        raise ValueError("Paint over the object first. A rough filled-in selection is enough.")
    return image, mask


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    """A translucent fill and crisp outline, identical for input and prediction."""
    base = _rgb(image)
    mask_image = Image.fromarray((np.asarray(mask).astype(bool) * 255).astype(np.uint8))
    if mask_image.size != base.size:
        raise ValueError("Overlay mask must match the image dimensions.")
    fill = Image.new("RGB", base.size, color)
    result = Image.composite(Image.blend(base, fill, 0.32), base, mask_image)
    edge = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3))) != np.asarray(
        mask_image.filter(ImageFilter.MinFilter(3))
    )
    pixels = np.array(result)
    pixels[edge] = color
    return Image.fromarray(pixels)


def editor_value(image: Image.Image, mask: np.ndarray | None = None) -> dict[str, Any]:
    background = _rgb(image).convert("RGBA")
    layer = Image.new("RGBA", background.size, (*COARSE_COLOR, 0))
    if mask is not None:
        alpha = Image.fromarray((np.asarray(mask).astype(bool) * 150).astype(np.uint8))
        layer.putalpha(alpha)
    return {
        "background": background,
        "layers": [layer],
        "composite": Image.alpha_composite(background, layer),
    }


def load_example(name: str, example_dir: Path = EXAMPLE_DIR) -> tuple[Image.Image, Image.Image]:
    if name not in {key for _, key in EXAMPLES}:
        raise ValueError("Unknown bundled example.")
    with Image.open(example_dir / f"{name}.png") as source:
        image = source.convert("RGB")
    with Image.open(example_dir / f"{name}-mask.png") as source:
        mask = source.convert("L")
    return image, mask


@lru_cache(maxsize=2)
def _refiner(model_name: str, models_dir: str | None):
    from segmentello.inference import Refiner

    return Refiner(model_name=model_name, device="cpu", models_dir=models_dir)


def refine(
    image: Image.Image,
    mask: np.ndarray,
    model_name: str,
    threshold: float,
    models_dir: str | None = None,
):
    if not np.any(mask):
        raise ValueError("The mask is empty. Mark at least part of the object before refining.")
    start = time.perf_counter()
    prediction = _refiner(model_name, models_dir).predict(
        image, mask, threshold=threshold, max_side=640
    )
    elapsed = time.perf_counter() - start
    result = (np.asarray(prediction.mask) > 0).astype(np.uint8)
    if result.shape != mask.shape:
        raise ValueError("The prediction does not match the input image dimensions.")
    out_root = Path(os.environ.get("GRADIO_TEMP_DIR", Path.cwd() / ".cache" / "segmentello-demo"))
    out_root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="result-", dir=out_root))
    path = folder / "refined-mask.png"
    Image.fromarray(result * 255).save(path)
    status = f"Completed in {elapsed:.2f} s · CPU · {image.width} × {image.height} pixels"
    if not result.any():
        status += (
            " · The model returned an empty mask. Try a broader selection or a lower threshold."
        )
    return (
        overlay(image, mask, COARSE_COLOR),
        overlay(image, result, REFINED_COLOR),
        Image.fromarray(result * 255),
        str(path),
        status,
    )


def build_demo(
    models_dir: str | None = None, default_model: str = DEFAULT_MODEL, hosted: bool = False
):
    """Build the optional UI without starting a public or local server."""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Install the demo extra first: pip install -e '.[demo]'") from error
    from segmentello.inference import list_models

    models = [spec for spec in list_models(models_dir) if spec.checkpoint.is_file()]
    if not models:
        raise ValueError(
            "No checkpoints found. Pass --models-dir with the repository’s models folder."
        )
    choices = [(spec.name, spec.name) for spec in models]
    if default_model not in {spec.name for spec in models}:
        raise ValueError(f"Unknown default model: {default_model}")
    first_image, first_mask = load_example(EXAMPLES[0][1])

    def run(mode, drawing, uploaded_image, uploaded_mask, model_name, threshold):
        try:
            if mode == "draw":
                image, mask = editor_inputs(drawing)
            else:
                image = _rgb(uploaded_image)
                mask = binary_mask(uploaded_mask, image.size)
            return refine(image, mask, model_name, threshold, models_dir)
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

    location = "INTERACTIVE DEMO" if hosted else "LOCAL DEMO"
    with gr.Blocks(
        analytics_enabled=False, title="NeuralSegmentello", delete_cache=(3600, 3600)
    ) as demo:
        gr.HTML(
            f'<div id="masthead"><div class="eyebrow">NEURALSEGMENTELLO / {location}</div>'
            "<h1>Refine a rough mask.</h1><p>Start with an example, or mark an object in your own image. "
            "Compare the selection with the model’s predicted boundary.</p></div>"
        )
        with gr.Row():
            with gr.Column(scale=3, min_width=320):
                with gr.Row(elem_id="example-row"):
                    example_buttons = [gr.Button(label, size="sm") for label, _ in EXAMPLES]
                mode = gr.State("draw")
                with gr.Tabs():
                    with gr.Tab("Draw a selection") as draw_tab:
                        drawing = gr.ImageEditor(
                            value=editor_value(
                                first_image, binary_mask(first_mask, first_image.size)
                            ),
                            label="Image & rough selection",
                            type="pil",
                            image_mode="RGBA",
                            sources=["upload"],
                            canvas_size=first_image.size,
                            fixed_canvas=False,
                            transforms=(),
                            layers=False,
                            brush=gr.Brush(
                                colors=[("#de7c4b", 0.6)], color_mode="fixed", default_size=22
                            ),
                            height=440,
                            format="png",
                            elem_classes="output-panel",
                        )
                        gr.Markdown(
                            "Paint **inside the object** with the brush. Use the eraser to remove marks."
                        )
                    with gr.Tab("Upload a mask") as upload_tab:
                        with gr.Row():
                            uploaded_image = gr.Image(
                                label="Image",
                                type="pil",
                                image_mode="RGB",
                                sources=["upload"],
                                height=330,
                            )
                            uploaded_mask = gr.Image(
                                label="Mask · white = object",
                                type="pil",
                                image_mode="L",
                                sources=["upload"],
                                height=330,
                            )
                        gr.Markdown(
                            "Use an image and a black-and-white mask of the **same dimensions**."
                        )
            with gr.Column(scale=1, min_width=270, elem_id="settings"):
                gr.Markdown("### Refinement")
                model = gr.Dropdown(choices, value=default_model, label="Model", interactive=True)
                threshold = gr.Slider(
                    0.1,
                    0.9,
                    value=0.5,
                    step=0.05,
                    label="Mask threshold",
                    info="Lower values include more pixels in the result.",
                )
                gr.Markdown(
                    "Inference runs on CPU. Large images are processed at up to 640 pixels on the longer edge, then restored to their original size."
                )
                button = gr.Button("Refine selection", variant="primary", elem_id="refine-button")
                gr.Markdown(
                    (
                        "Uploaded images are processed on this demo’s server. Use the examples if you prefer not to upload an image. "
                        if hosted
                        else "Your images stay on this computer. "
                    )
                    + "The first run also loads the selected model."
                )
        status = gr.Markdown(
            "Choose an example or mark an object, then refine the selection.",
            elem_id="result-status",
        )
        with gr.Row():
            coarse_preview = gr.Image(
                label="Your selection", interactive=False, height=320, elem_classes="output-panel"
            )
            refined_preview = gr.Image(
                label="Refined selection",
                interactive=False,
                height=320,
                elem_classes="output-panel",
            )
            mask_preview = gr.Image(
                label="Predicted mask", interactive=False, height=320, elem_classes="output-panel"
            )
        download = gr.File(label="Download mask · PNG", interactive=False, file_count="single")
        gr.Markdown(
            "Research demo using the repository’s existing checkpoints. The examples are illustrative; outputs are predictions, not ground truth.",
            elem_id="note",
        )
        outputs = [coarse_preview, refined_preview, mask_preview, download, status]
        button.click(
            run,
            [mode, drawing, uploaded_image, uploaded_mask, model, threshold],
            outputs,
            concurrency_limit=1,
            concurrency_id="inference",
            api_name="refine",
        )
        draw_tab.select(lambda: "draw", outputs=mode, api_visibility="private")
        upload_tab.select(lambda: "upload", outputs=mode, api_visibility="private")
        for example_button, (_, key) in zip(example_buttons, EXAMPLES):

            def select_example(example_key=key):
                image, mask = load_example(example_key)
                return (
                    gr.update(
                        value=editor_value(image, binary_mask(mask, image.size)),
                        canvas_size=image.size,
                    ),
                    image,
                    mask,
                    None,
                    None,
                    None,
                    None,
                    "Example loaded. Select Refine selection to run the model.",
                )

            example_button.click(
                select_example,
                outputs=[drawing, uploaded_image, uploaded_mask, *outputs],
                api_visibility="private",
            )
        for component in (drawing, uploaded_image, uploaded_mask, model, threshold):
            component.input(
                lambda: (
                    None,
                    None,
                    None,
                    None,
                    "Input changed. Refine again to update the result.",
                ),
                outputs=outputs,
                api_visibility="private",
                show_progress="hidden",
            )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NeuralSegmentello's local mask-refinement demo."
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    demo = build_demo(models_dir=args.models_dir, default_model=args.model)
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="gray",
        font=["Arial", "sans-serif"],
        font_mono=["Menlo", "monospace"],
    )
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        theme=theme,
        css=CSS,
        inbrowser=False,
        max_file_size="20mb",
        footer_links=[],
        mcp_server=False,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
