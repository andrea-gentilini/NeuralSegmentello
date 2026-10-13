# Hugging Face deployment

This directory contains the Gradio Space entrypoint and configuration. The Space
runs on `cpu-basic`; it does not need GPU hardware or inference-provider keys.
The hosted interface tells visitors that uploads are processed on the server.

As of 13 October 2026, Hugging Face requires a **PRO subscription** for personal
accounts creating Gradio or Docker Spaces on `cpu-basic`. The hardware has no
hourly charge, but the subscription is a prerequisite. See the official
[Space management documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces).
A local demo remains available without that subscription.

`deploy_space.py` stages only Python source, packaging metadata, license, demo
examples, and the eight checkpoint files. It excludes datasets, notebooks, and
training logs. Checkpoints use hard links when possible to avoid duplicating
large files; on a different filesystem, staging stops unless `--copy-checkpoints`
is supplied. Inspect the generated `space-manifest.json` before uploading.

From the repository root, with the project environment and `huggingface_hub`:

```sh
python deploy/huggingface/deploy_space.py stage
python deploy/huggingface/deploy_space.py inspect
# Only after the account meets the hosting requirements:
python deploy/huggingface/deploy_space.py create
python deploy/huggingface/deploy_space.py upload
python deploy/huggingface/deploy_space.py status
```

The staging directory defaults to `../space`. Pass `--stage-dir` to change it.
`stage` never makes network requests. `upload` uploads the reviewed staging
snapshot; run `stage` again to include later source changes. Creation and upload
are separate explicit operations. The script refuses to overwrite an existing
Space unless its card contains this deployment's marker or it was created by a
previous successful invocation recorded in the local creation receipt.

Authentication comes from the configured Git credential helper for
`huggingface.co`. The script never writes the credential to a file or environment
variable. It does not log in, request paid hardware, or create a subscription.
Only `andrea-gentilini/NeuralSegmentello` on `cpu-basic` is supported by this
project-specific helper.

The runtime pins Gradio 6.30.0 and Python 3.12 via the
[Space YAML configuration](https://huggingface.co/docs/hub/spaces-config-reference).
PyTorch and TorchVision use the official CPU wheel index, following
[PyTorch's installation instructions](https://github.com/pytorch/pytorch/releases/tag/v2.14.1).
