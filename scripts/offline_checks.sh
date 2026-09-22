#!/usr/bin/env bash
# Offline CPU checks for the Smart Tool init mask (tests/unit).
#
# The app normally runs inside the image named in config.json; this builds a
# throwaway virtualenv with the SDK version docker/Dockerfile pins so the checks
# run in a plain checkout. No GPU, no SAM 2 weights and no platform are needed:
# src/main.py is imported and its /smart_segmentation route is called directly.
#
# Reuses $SAM2_CHECK_VENV (default .venv-offline-checks) when it already exists.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
VENV="${SAM2_CHECK_VENV:-.venv-offline-checks}"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet "supervisely==6.74.36" mock pytest
    # CPU build; the image installs the CUDA one, but nothing here loads a model.
    "$PY" -m pip install --quiet "torch==2.5.1" \
        --index-url https://download.pytorch.org/whl/cpu
    # src/main.py imports sam2 at module level, which is what the image builds
    # from source; the CUDA extension is not needed to import it.
    SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1 \
        "$PY" -m pip install --quiet "git+https://github.com/facebookresearch/sam2.git"
fi

# supervisely imports python-magic, which dlopens libmagic. The image gets it
# from the base distro; fetch it next to the virtualenv when the host has none.
if ! "$PY" -c "import magic" >/dev/null 2>&1; then
    MAGIC_DIR="$VENV/libmagic"
    if [ ! -d "$MAGIC_DIR" ]; then
        mkdir -p "$MAGIC_DIR"
        for deb in libmagic1t64_5.46-5_amd64.deb libmagic-mgc_5.46-5_amd64.deb; do
            curl -sSfL -o "$MAGIC_DIR/$deb" \
                "https://deb.debian.org/debian/pool/main/f/file/$deb"
            dpkg-deb -x "$MAGIC_DIR/$deb" "$MAGIC_DIR"
        done
    fi
    MAGIC_LIB="$(cd "$MAGIC_DIR/usr/lib/x86_64-linux-gnu" && pwd)"
    export LD_LIBRARY_PATH="$MAGIC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PY" -m pytest tests/unit -q -p no:warnings "$@"
