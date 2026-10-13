"""Stage or explicitly publish the project Space; never persist credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ID = "andrea-gentilini/NeuralSegmentello"
MARKER = "<!-- neuralsegmentello-space:v1 -->"
STAGE_MARKER = ".neuralsegmentello-stage"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def default_repo():
    location = Path(__file__).resolve()
    if location.parent.name == "artifacts":
        return location.parents[1] / "repo"
    return location.parents[2]


def paths(args):
    repo = args.repo_dir.resolve()
    stage = (args.stage_dir or repo.parent / "space").resolve()
    if stage == repo or repo in stage.parents:
        raise ValueError("The staging directory must be outside the source repository.")
    artifacts = stage.parent / "artifacts"
    return repo, stage, artifacts


def stage_files(args):
    repo, stage, artifacts = paths(args)
    template = repo / "deploy" / "huggingface"
    if stage.exists() and any(stage.iterdir()) and not (stage / STAGE_MARKER).is_file():
        raise ValueError("Refusing to replace a staging folder without the project marker.")
    selected = {
        Path("app.py"): template / "app.py",
        Path("README.md"): template / "README.space.md",
        Path("requirements.txt"): template / "requirements.txt",
        Path("pyproject.toml"): repo / "pyproject.toml",
        Path("PROJECT_README.md"): repo / "README.md",
        Path("LICENSE"): repo / "LICENSE",
    }
    for source in (repo / "src").rglob("*.py"):
        selected[source.relative_to(repo)] = source
    for source in (repo / "examples" / "demo").iterdir():
        if source.suffix.lower() in {".png", ".md"}:
            selected[source.relative_to(repo)] = source
    checkpoints = sorted((repo / "models").glob("*/checkpoints/best-checkpoint.ckpt"))
    if len(checkpoints) != 8:
        raise ValueError(f"Expected eight supplied checkpoints, found {len(checkpoints)}.")
    for source in checkpoints:
        selected[source.relative_to(repo)] = source
    for source in selected.values():
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Expected a regular source file: {source.relative_to(repo)}")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / STAGE_MARKER).write_text(REPO_ID + "\n")
    for previous in stage.rglob("*"):
        if previous.is_file() and previous.name != STAGE_MARKER:
            previous.unlink()
    for relative, source in selected.items():
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.suffix == ".ckpt":
            try:
                os.link(source, target)
            except OSError:
                if not args.copy_checkpoints:
                    raise ValueError(
                        "Checkpoint hard links unavailable. Use --copy-checkpoints to allow copies."
                    ) from None
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
    records = [
        {
            "path": str(relative),
            "bytes": (stage / relative).stat().st_size,
            "sha256": digest(stage / relative),
        }
        for relative in sorted(selected)
    ]
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = {"repo_id": REPO_ID, "files": records}
    (artifacts / "space-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": str(stage),
                "files": len(records),
                "checkpoints": 8,
                "bytes": sum(record["bytes"] for record in records),
            }
        )
    )


def authenticated_api():
    from huggingface_hub import HfApi

    completed = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=huggingface.co\n\n",
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise ValueError("No Hugging Face credential is available from the Git helper.")
    api = HfApi(token=token)
    who = api.whoami()
    if who.get("name") != REPO_ID.split("/")[0]:
        raise ValueError("The authenticated account is not the expected Space owner.")
    return api, who


def remote_info(api):
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return api.space_info(REPO_ID)
    except RepositoryNotFoundError as error:
        if error.response.status_code == 404:
            return None
        raise


def runtime_summary(api):
    runtime = api.get_space_runtime(REPO_ID)
    return {
        "stage": runtime.stage,
        "hardware": runtime.hardware,
        "requested_hardware": runtime.requested_hardware,
    }


def validate_stage(stage, artifacts):
    manifest = json.loads((artifacts / "space-manifest.json").read_text())
    if manifest.get("repo_id") != REPO_ID:
        raise ValueError("Staging manifest does not match the target Space.")
    expected = {record["path"] for record in manifest["files"]}
    actual = {
        str(path.relative_to(stage))
        for path in stage.rglob("*")
        if path.is_file() and path.name != STAGE_MARKER
    }
    if actual != expected:
        raise ValueError("Staging file list changed. Run stage and review the new manifest first.")
    for record in manifest["files"]:
        file = stage / record["path"]
        if (
            file.is_symlink()
            or file.stat().st_size != record["bytes"]
            or digest(file) != record["sha256"]
        ):
            raise ValueError(f"Staging content changed: {record['path']}. Run stage again.")


def verify_existing_space(api, info, artifacts):
    from huggingface_hub import hf_hub_download

    if info.private or info.sdk != "gradio":
        raise ValueError("Existing Space is not this public Gradio deployment.")
    card_path = hf_hub_download(REPO_ID, "README.md", repo_type="space", token=api.token)
    related = MARKER in Path(card_path).read_text()
    receipt = artifacts / "space-creation.json"
    if not related and receipt.is_file():
        previous = json.loads(receipt.read_text())
        related = previous.get("repo_id") == REPO_ID and previous.get("sha") == info.sha
    if not related:
        raise ValueError("Existing Space lacks the project marker or matching creation receipt.")
    runtime = runtime_summary(api)
    if runtime["hardware"] not in (None, "cpu-basic") or runtime["requested_hardware"] not in (
        None,
        "cpu-basic",
    ):
        raise ValueError("Refusing to deploy onto hardware other than cpu-basic.")


def remote_operation(args):
    _, stage, artifacts = paths(args)
    api, who = authenticated_api()
    info = remote_info(api)
    if args.action == "inspect":
        print(
            json.dumps(
                {
                    "account": who["name"],
                    "is_pro": who.get("isPro"),
                    "exists": info is not None,
                    **(runtime_summary(api) if info else {}),
                }
            )
        )
    elif args.action == "create":
        if info is not None:
            raise ValueError("The Space already exists; no repository was changed.")
        api.create_repo(
            REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="cpu-basic",
            private=False,
            exist_ok=False,
        )
        info = api.space_info(REPO_ID)
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "space-creation.json").write_text(
            json.dumps({"repo_id": REPO_ID, "sha": info.sha}, indent=2) + "\n"
        )
        print(json.dumps({"created": True, "url": f"https://huggingface.co/spaces/{REPO_ID}"}))
    elif args.action == "upload":
        if info is None:
            raise ValueError("Space does not exist. Creation is a separate explicit operation.")
        validate_stage(stage, artifacts)
        verify_existing_space(api, info, artifacts)
        result = api.upload_folder(
            repo_id=REPO_ID,
            repo_type="space",
            folder_path=stage,
            ignore_patterns=[STAGE_MARKER],
            commit_message="Publish NeuralSegmentello CPU demo and bundled checkpoints",
        )
        print(json.dumps({"uploaded": True, "commit": result.oid}))
    else:
        if info is None:
            raise ValueError("Space does not exist.")
        print(json.dumps(runtime_summary(api)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["stage", "inspect", "create", "upload", "status"])
    parser.add_argument("--repo-dir", type=Path, default=default_repo())
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--copy-checkpoints", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "stage":
            stage_files(args)
        else:
            remote_operation(args)
    except Exception as error:  # noqa: BLE001 -- do not expose credential-bearing exceptions
        # HTTP/credential exceptions may contain request details. Never print them.
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if status == 402:
            parser.exit(
                1,
                "Hugging Face requires PRO for this cpu-basic Space. No paid hardware was requested.\n",
            )
        if status is not None:
            parser.exit(1, f"Hugging Face request failed (HTTP {status}).\n")
        if isinstance(error, ValueError):
            parser.exit(1, str(error) + "\n")
        parser.exit(1, f"Operation failed ({type(error).__name__}); credentials were not logged.\n")


if __name__ == "__main__":
    main()
