# Release images build on image 1.0.21, pinned by digest: its CUDA, torch (2.9.1+cu128), SAM 2
# (facebookresearch/sam2 2b90b9f) and SAM 2.1 weights are what production runs. The recipe that
# built it cloned SAM 2 unpinned and let SAM 2 pull the latest torch, so rebuilding it would not
# reproduce it; see this file's history for that recipe.
FROM supervisely/segment-anything-2:1.0.21@sha256:acb431c25ea03ea32c26f994792725e71d4f5717030fba30e40fd147162923f0

RUN python3 -m pip install --no-cache-dir supervisely[video-av]==6.74.62

# The app, last, so that a release that changes only code adds only these small layers to an image
# an agent already has. --link keeps them independent of the layers below. Dockerfile.dockerignore
# lets only these into the build context.
COPY --link models/ /app/repo/models/
COPY --link custom_settings.yaml /app/repo/
COPY --link src/ /app/repo/src/
COPY --link config.json /app/repo/
WORKDIR /app/repo
CMD ["python3", "-m", "uvicorn", "src.main:m.app", "--app-dir", "./serve", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]

LABEL python_sdk_version=6.74.62
