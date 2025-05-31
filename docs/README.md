# Neural Segmentello

NeuralSegmentello is a deep learning framework for refining coarse object segmentation masks into high-precision segmentations. The task is performed via U-Net-based architectures, given a rough user-provided binary mask (simulating a brush stroke) and the original image. The system returns a refined mask with accurate object contours.

[***Project Report (PDF)***](./NeuralSegmentello.pdf)




# Dependencies Installation

### Using uv (recommended)
You can use `uv` to install dependencies: just run the following:
```sh
uv sync
```

--------------------------------------------------------------

### Using pip
You can install dependencies in a virtual environment e.g.:
```sh
python -m venv .venv

# then activate the environment according to your os:
source .venv/bin/activate  # linux/MacOS
.venv\Scripts\Activate  # windows
```
then you can use `pip` to install dependencies:
```sh
pip install -r requirements.txt
```
or install the whole project (-e for editable mode)
```sh
pip install -e .
```





