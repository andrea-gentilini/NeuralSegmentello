"""Command line interface for local, download-free inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .inference import DEFAULT_MODEL, Refiner, list_models


def _load_array(path: str, image: bool):
    file = Path(path)
    if file.suffix.lower() == ".npy":
        return np.load(file, allow_pickle=False)
    with Image.open(file) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB" if image else "L"))


def _output_paths(args):
    """Validate all destinations before inference or writing any result."""
    output = Path(args.output)
    probability = Path(args.probability_output) if args.probability_output else None
    metadata = Path(args.metadata_output) if args.metadata_output else None
    if output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png so the binary mask is saved losslessly.")
    if probability is not None and probability.suffix.lower() != ".npy":
        raise ValueError("--probability-output must end in .npy.")
    destinations = [path for path in (output, probability, metadata) if path is not None]
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Mask, probability, and metadata outputs must use different paths.")
    sources = [Path(args.image), Path(args.mask)]
    checkpoint = args.checkpoint or next(
        spec.checkpoint for spec in list_models(args.models_dir) if spec.name == args.model
    )
    sources.append(Path(checkpoint))
    if set(resolved).intersection(path.resolve() for path in sources):
        raise ValueError("Output paths must not overwrite an input image, mask, or checkpoint.")
    for index, destination in enumerate(destinations):
        if destination.exists() and any(
            other.exists() and destination.samefile(other) for other in destinations[index + 1 :]
        ):
            raise ValueError("Mask, probability, and metadata outputs must use different files.")
        if destination.exists() and any(
            source.exists() and destination.samefile(source) for source in sources
        ):
            raise ValueError("Output paths must not overwrite an input image, mask, or checkpoint.")
        if destination.exists() and destination.is_dir():
            raise ValueError(f"Output path is a directory: {destination}")
    return output, probability, metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Refine a coarse binary mask with a local NeuralSegmentello checkpoint."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser(
        "info", help="List bundled model names, or inspect one loaded checkpoint."
    )
    info.add_argument("--model", choices=[m.name for m in list_models()])
    info.add_argument("--models-dir")
    info.add_argument("--checkpoint")
    info.add_argument("--device", default="cpu")
    predict = commands.add_parser(
        "predict", aliases=["infer"], help="Save an original-size refined mask and probability map."
    )
    predict.add_argument("--image", required=True, help="RGB/grayscale image or .npy array.")
    predict.add_argument("--mask", required=True, help="Binary coarse mask image or .npy array.")
    predict.add_argument("--output", required=True, help="Destination binary mask PNG.")
    predict.add_argument("--probability-output", help="Optional float32 .npy probability map.")
    predict.add_argument("--metadata-output", help="Optional JSON inference metadata.")
    predict.add_argument("--model", default=DEFAULT_MODEL, choices=[m.name for m in list_models()])
    predict.add_argument("--models-dir")
    predict.add_argument("--checkpoint")
    predict.add_argument("--device", default="auto")
    predict.add_argument("--threshold", type=float, default=0.5)
    predict.add_argument("--max-side", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            if args.model is None:
                result = [
                    {
                        "name": m.name,
                        "architecture": m.architecture,
                        "features": list(m.features),
                        "checkpoint": str(m.checkpoint),
                        "available": m.checkpoint.is_file(),
                    }
                    for m in list_models(args.models_dir)
                ]
            else:
                result = Refiner(args.model, args.device, args.models_dir, args.checkpoint).info()
            print(json.dumps(result, indent=2))
            return 0
        output, probability, metadata = _output_paths(args)
        refiner = Refiner(args.model, args.device, args.models_dir, args.checkpoint)
        result = refiner.predict(
            _load_array(args.image, True),
            _load_array(args.mask, False),
            args.threshold,
            args.max_side,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.mask * 255).save(output)
        if probability is not None:
            probability.parent.mkdir(parents=True, exist_ok=True)
            np.save(probability, result.probability, allow_pickle=False)
        if metadata is not None:
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps(result.metadata, indent=2) + "\n")
        print(json.dumps({"output": str(output), **result.metadata}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
