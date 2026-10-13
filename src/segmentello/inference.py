"""Local checkpoint loading and image-size-preserving coarse-mask refinement.

Checkpoint inputs are [binary coarse mask, grayscale image, normalized Sobel
magnitude]. RGB is accepted at the interface, then converted to grayscale.
"""

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

DEFAULT_MODEL = "res16-128_bce-dice-bound"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    features: tuple[int, ...]
    checkpoint: Path


_CONFIGS = {
    "tiny16-32_bce-dice": ("tiny", (16, 32)),
    "tiny16-128_bce-dice": ("tiny", (16, 32, 64, 128)),
    "tiny16-128_bce-dice-bound": ("tiny", (16, 32, 64, 128)),
    "res16-32_bce-dice": ("residual", (16, 32)),
    "res16-128_bce-dice": ("residual", (16, 32, 64, 128)),
    "res16-128_bce-dice-bound": ("residual", (16, 32, 64, 128)),
    "res32-256_bce-dice-bound": ("residual", (32, 64, 128, 256)),
    "attn32-256_bce-dice-bound": ("attention", (32, 64, 128, 256)),
}


def list_models(models_dir: str | Path | None = None) -> list[ModelSpec]:
    root = Path(models_dir) if models_dir is not None else REPO_ROOT / "models"
    return [
        ModelSpec(name, arch, features, root / name / "checkpoints" / "best-checkpoint.ckpt")
        for name, (arch, features) in _CONFIGS.items()
    ]


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    result = torch.device(device)
    if result.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("Device must be auto, cpu, mps, or cuda[:index].")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def state_dict_digest(state: dict[str, torch.Tensor]) -> str:
    """Fingerprint tensor content, excluding optimizer state and file packaging."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _read_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Use models_dir or --checkpoint; no models are downloaded."
        )
    # Never fall back to unrestricted pickle loading for an unknown checkpoint.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            "Checkpoint cannot be loaded with weights_only=True; unrestricted pickle is disabled."
        ) from error
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a state dictionary or Lightning checkpoint dictionary.")
    state = checkpoint.get("state_dict", checkpoint)
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(v, torch.Tensor) for v in state.values())
    ):
        raise ValueError("Checkpoint state_dict must contain only tensors.")
    metadata = {
        key: checkpoint[key]
        for key in ("epoch", "global_step", "pytorch-lightning_version")
        if key in checkpoint
    }
    return state, metadata


def _make_model(spec: ModelSpec):
    if spec.architecture == "tiny":
        from .u_net import Coarse2FineTiny

        return Coarse2FineTiny(features=list(spec.features))
    if spec.architecture == "residual":
        from .u_net_residual import Coarse2FineTinyRes

        return Coarse2FineTinyRes(features=list(spec.features))
    from .u_net_res_attention import Coarse2FineUNetAttention

    return Coarse2FineUNetAttention(features=list(spec.features))


def _image_uint8(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] != 3):
        raise ValueError("Image must be HxW grayscale or HxWx3 RGB.")
    if (
        min(array.shape[:2]) < 1
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError("Image must contain nonempty numeric pixels.")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
        raise ValueError("Image pixels must be finite and in [0, 255].")
    if np.issubdtype(array.dtype, np.floating) and array.max() <= 1:
        array = array * 255
    return np.ascontiguousarray(np.rint(array).astype(np.uint8))


def _binary_mask(mask, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask = np.asarray(ImageOps.exif_transpose(mask).convert("L"))
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != shape:
        raise ValueError(f"Coarse mask shape {array.shape} must match image shape {shape}.")
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError("Coarse mask must contain numeric or boolean pixels.")
    if not np.isfinite(array).all() or not np.isin(array, [0, 1, 255]).all():
        raise ValueError("Coarse mask must be binary (0/1, bool, or 0/255).")
    return np.ascontiguousarray((array > 0).astype(np.float32))


@dataclass
class Prediction:
    probability: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any]


class Refiner:
    """Load once, then refine multiple images without downloads or training.

    Registry architecture names preserve the supplied folder labels. Weights
    alone cannot establish whether the original forward added a residual mask.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        models_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
    ):
        choices = {spec.name: spec for spec in list_models(models_dir)}
        if model_name not in choices:
            raise ValueError(f"Unknown model {model_name!r}; choose one of {', '.join(choices)}.")
        self.spec = choices[model_name]
        self.checkpoint = Path(checkpoint) if checkpoint is not None else self.spec.checkpoint
        self.device = resolve_device(device)
        state, self.checkpoint_metadata = _read_state(self.checkpoint)
        self.model = _make_model(self.spec)
        # Strict shape/key checks prevent a mislabeled or incompatible model from silently loading.
        self.model.load_state_dict(state, strict=True)
        self.state_sha256 = state_dict_digest(state)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self, image, coarse_mask, threshold: float = 0.5, max_side: int | None = None
    ) -> Prediction:
        if not 0 <= threshold <= 1:
            raise ValueError("Probability threshold must be in [0, 1].")
        if max_side is not None and (not isinstance(max_side, int) or max_side < 1):
            raise ValueError("max_side must be a positive integer or None.")
        image = _image_uint8(image)
        original_shape = image.shape[:2]
        coarse = _binary_mask(coarse_mask, original_shape)
        if max_side is not None and max(original_shape) > max_side:
            ratio = max_side / max(original_shape)
            shape = (
                max(1, round(original_shape[1] * ratio)),
                max(1, round(original_shape[0] * ratio)),
            )
            image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
            coarse = cv2.resize(coarse, shape, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient /= gradient.max() + 1e-8
        data = np.stack([coarse, gray.astype(np.float32) / 255, gradient])
        inputs = torch.from_numpy(data).unsqueeze(0).to(self.device)
        height, width = gray.shape
        divisor = 2 ** len(self.spec.features)
        pad_h, pad_w = (-height) % divisor, (-width) % divisor
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode="replicate")
        logits = self.model(inputs)
        if logits.shape != (1, 1, height + pad_h, width + pad_w):
            raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}.")
        probabilities = torch.sigmoid(logits[..., :height, :width])
        if (height, width) != original_shape:
            probabilities = F.interpolate(
                probabilities, size=original_shape, mode="bilinear", align_corners=False
            )
        probability = probabilities[0, 0].cpu().numpy().astype(np.float32)
        if not np.isfinite(probability).all():
            raise RuntimeError("Model produced non-finite probabilities.")
        return Prediction(
            probability,
            (probability >= threshold).astype(np.uint8),
            {
                "model": self.spec.name,
                "architecture": self.spec.architecture,
                "device": str(self.device),
                "threshold": threshold,
                "original_shape": list(original_shape),
                "inference_shape": [height, width],
                "padding": [pad_h, pad_w],
                "state_sha256": self.state_sha256,
                "parameter_count": self.parameter_count,
                "input_channels": ["coarse_mask", "grayscale", "sobel_magnitude"],
            },
        )

    def info(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "architecture": self.spec.architecture,
            "features": list(self.spec.features),
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "parameter_count": self.parameter_count,
            "state_sha256": self.state_sha256,
            "checkpoint_metadata": self.checkpoint_metadata,
        }
