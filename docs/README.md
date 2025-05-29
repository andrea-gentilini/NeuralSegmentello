# Neural Segmentello

[***Project Report (PDF)***](./NeuralSegmentello.pdf)




## Installation

### pip
You can install dependencies in a virtual environment e.g.:
```sh
python -m venv .venv

# then activate the environment according to your os:
source .venv/bin/activate  # linux
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

### uv
You can use `uv` to install dependencies: just run the following:
```sh
uv sync
```