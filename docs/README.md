# NeuralSegmentello

NeuralSegmentello is a deep learning framework for refining coarse object segmentation masks into high-precision segmentations. The task is performed via U-Net-based architectures, given a rough user-provided binary mask (simulating a brush stroke) and the original image. The system returns a refined mask with accurate object contours.

Check out the full project explanation in the [**Project Report (PDF)**](./NeuralSegmentello.pdf).




# Dependencies Installation

### Using uv (recommended)

If you have [`uv`](https://docs.astral.sh/uv/) installed, simply run:
```sh
uv sync
```

--------------------------------------------------------------

### Using pip
You can also set up a virtual environment manually and install dependencies with pip.

Create a virtual environment:
```sh
python -m venv .venv
```

Activate it:
```sh
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies using pip:

```sh
pip install -r requirements.txt
```
or install the whole project (-e for editable mode)
```sh
pip install -e .
```

# Authors

| [<img src="https://github.com/andrea-gentilini.png" width="100px" style="border-radius:50%;" /><br><sub>**Andrea Gentilini**</sub>](https://github.com/andrea-gentilini) | [<img src="https://github.com/mich1803.png" width="100px" style="border-radius:50%;" /><br><sub>**Michele Magrini**</sub>](https://github.com/mich1803) | [<img src="https://github.com/leopetra20.png" width="100px" style="border-radius:50%;" /><br><sub>**Leo Vincenzo Petrarca**</sub>](https://github.com/leopetra20) | [<img src="https://github.com/IacopoScandale.png" width="100px" style="border-radius:50%;" /><br><sub>**Iacopo Scandale**</sub>](https://github.com/IacopoScandale) |
|:---:|:---:|:---:|:---:|




