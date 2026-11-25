"""Checks that config.json names the image of the release and that docker/Dockerfile starts the app as config.json does.

Every release vX.Y.Z of the app has its own image, supervisely/segment-anything-2:X.Y.Z, which carries the app's code: the
Release workflow builds it from the release's commit. config.json's "version" is not the release: the platform reads it as
the app engine version, and an app below 2.0.0 is served as a v1 app.

    python3 docker/check_image.py                    # pull requests: config.json and docker/Dockerfile agree
    python3 docker/check_image.py --release v1.0.65  # the Release workflow, before it builds anything

It prints the image's tag (X.Y.Z) last, on its own line, for the workflow to build.
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
    engine = config.get("version", "")
    if not VERSION.fullmatch(engine) or int(engine.split(".")[0]) < 2:
        errors.append(f'config.json: "version" is the app engine version, 2.x.y for this app, not {engine!r}')
    image = config.get("docker_image", "")
    name, _, tag = image.partition(":")
    if name != IMAGE or not VERSION.fullmatch(tag):
        errors.append(f'config.json: "docker_image" is {image!r}, not {IMAGE}:X.Y.Z')
    elif release is not None and release != f"v{tag}":
        wanted = release[1:] if release.startswith("v") and VERSION.fullmatch(release[1:]) else "X.Y.Z"
        errors.append(
            f"release {release!r} does not run on config.json's image {image}: release a commit whose config.json has "
            f'"docker_image": "{IMAGE}:{wanted}" (releases are vX.Y.Z)'
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
    print(f"OK: {config['docker_image']}, WORKDIR {APP_DIR}, CMD {config['entrypoint']}", file=sys.stderr)
    print(config["docker_image"].partition(":")[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
