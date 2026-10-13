---
title: NeuralSegmentello
emoji: 🎨
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
license: mit
short_description: Refine a rough object selection with compact U-Net models.
---
<!-- neuralsegmentello-space:v1 -->

# NeuralSegmentello

Upload an image and paint a rough object selection, or use an included example.
Compare the original selection with a compact U-Net prediction and download the
refined binary mask. All eight supplied checkpoints are available in the model
selector. Inference runs on CPU, with a maximum processing side of 640 pixels;
the result is returned at the original image dimensions.

Uploaded images are processed on this Space's server. Use the included examples
if you prefer not to upload an image. This application does not use uploads for
training; Gradio manages temporary input and output files.

This is a research demo. The examples are illustrative and predictions may be
worse than the input selection. See the [project repository](https://github.com/andrea-gentilini/NeuralSegmentello)
for evaluation scope, limitations, original authors, and local installation.
Example provenance is recorded in [examples/demo/README.md](examples/demo/README.md).

This bundle contains the source package, example assets, and the eight original
checkpoints. It does not include training datasets, notebooks, or training logs.
The project README is preserved as [PROJECT_README.md](PROJECT_README.md).
