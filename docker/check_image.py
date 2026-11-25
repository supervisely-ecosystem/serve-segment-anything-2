"""Checks that config.json names the image of the release and that docker/Dockerfile starts the app as config.json does.

Every release vX.Y.Z of the app has its own image, supervisely/segment-anything-2:X.Y.Z, which carries the app's code: the
Release workflow builds it from the release's commit. config.json's "version" is X.Y.Z too.

    python3 docker/check_image.py                   # pull requests: config.json and docker/Dockerfile agree
    python3 docker/check_image.py --release v2.2.0  # the Release workflow, before it builds anything
"""

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "supervisely/segment-anything-2"
# Where the agents run the app (the release they download is mounted over /app): the image's copy lives there too.
APP_DIR = "/app/repo"
VERSION = re.compile(r"\d+\.\d+\.\d+")


def instructions(dockerfile: Path) -> list:
    """(INSTRUCTION, arguments) of every instruction, continuation lines joined and comments dropped."""
    result, current = [], ""
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        keyword, _, arguments = (current + stripped).partition(" ")
        result.append((keyword.upper(), arguments.strip()))
        current = ""
    return result


def last(found: list, keyword: str):
    values = [arguments for name, arguments in found if name == keyword]
    return values[-1] if values else None


def check(config: dict, release) -> list:
    errors = []
    version = config.get("version", "")
    image = f"{IMAGE}:{version}"
    if not VERSION.fullmatch(version):
        errors.append(f'config.json: "version" is {version!r}, not X.Y.Z')
    elif config.get("docker_image") != image:
        errors.append(f'config.json: "docker_image" is {config.get("docker_image")!r}; version {version} runs on {image!r}')
    if release is not None and release != f"v{version}":
        wanted = release[1:] if release.startswith("v") and VERSION.fullmatch(release[1:]) else "X.Y.Z"
        errors.append(
            f"release {release!r} is not config.json's version {version}: release a commit whose config.json has "
            f'"version": "{wanted}" and "docker_image": "{IMAGE}:{wanted}" (releases are vX.Y.Z)'
        )

    found = instructions(ROOT / "docker" / "Dockerfile")
    workdir = last(found, "WORKDIR")
    if workdir != APP_DIR:
        errors.append(f"docker/Dockerfile: the last WORKDIR is {workdir!r}, not {APP_DIR!r}")
    cmd = last(found, "CMD")
    entrypoint = shlex.split(config.get("entrypoint", ""))
    try:
        cmd = json.loads(cmd) if cmd is not None else None
    except json.JSONDecodeError:
        pass
    if cmd != entrypoint:
        errors.append(f"docker/Dockerfile: the last CMD is {cmd!r}, not config.json's entrypoint {json.dumps(entrypoint)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", help="the release's tag, vX.Y.Z")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    errors = check(config, parser.parse_args().release)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"OK: {config['docker_image']}, WORKDIR {APP_DIR}, CMD {config['entrypoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
