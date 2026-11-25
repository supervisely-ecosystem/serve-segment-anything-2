# Builds and pushes config.json's image by hand (the Release workflow builds it for each release in CI). Run from the
# repository's root. A tag already on Docker Hub is never pushed over: agents keep the image they pulled under it.
set -e
image=$(python3 -c 'import json; print(json.load(open("config.json"))["docker_image"])')
if docker buildx imagetools inspect "$image" > /dev/null 2>&1; then
    echo "$image is on Docker Hub already: set a new version in config.json (python3 docker/check_image.py)" >&2
    exit 1
fi
docker buildx build -f docker/Dockerfile -t "$image" \
    --output "type=image,push=true,compression=zstd,compression-level=7,force-compression=true,oci-mediatypes=true" .
