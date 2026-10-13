"""CPU entrypoint for the NeuralSegmentello Hugging Face Space."""

import os
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["DO_NOT_TRACK"] = "1"
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path.cwd() / ".cache" / "gradio"))

import gradio as gr
import torch

from segmentello.demo import CSS, build_demo
from segmentello.inference import DEFAULT_MODEL


def main():
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    demo = build_demo(
        models_dir=str(Path(__file__).resolve().parent / "models"),
        default_model=DEFAULT_MODEL,
        hosted=True,
    )
    theme = gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="gray",
        font=["Arial", "sans-serif"],
        font_mono=["Menlo", "monospace"],
    )
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
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
