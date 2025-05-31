# Neural Segmentello

NeuralSegmentello is a deep learning framework for refining coarse object segmentation masks into high-precision segmentations. The task is performed via U-Net-based architectures, given a rough user-provided binary mask (simulating a brush stroke) and the original image. The system returns a refined mask with accurate object contours.

Check out the full project explanation @ [***Project Report (PDF)***](./NeuralSegmentello.pdf)




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

# Authors

| [<img src="https://github.com/andrea-gentilini.png" width="100px" style="border-radius:50%;" /><br><sub>**Andrea Gentilini**</sub>](https://github.com/andrea-gentilini) | [<img src="https://github.com/mich1803.png" width="100px" style="border-radius:50%;" /><br><sub>**Michele Magrini**</sub>](https://github.com/mich1803) | [<img src="https://github.com/leopetra20.png" width="100px" style="border-radius:50%;" /><br><sub>**Leo Vincenzo Petrarca**</sub>](https://github.com/leopetra20) | [<img src="https://github.com/IacopoScandale.png" width="100px" style="border-radius:50%;" /><br><sub>**Iacopo Scandale**</sub>](https://github.com/IacopoScandale) |
|:---:|:---:|:---:|:---:|




