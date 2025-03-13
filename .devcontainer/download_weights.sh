#!/bin/bash
set -e

WEIGHTS_DIR="${HOME}/sam2_weights"
mkdir -p "${WEIGHTS_DIR}"

if [ ! -f "${WEIGHTS_DIR}/sam2.1_hiera_tiny.pt" ]; then
  echo "Downloading sam2.1_hiera_tiny.pt..."
  curl -L -o "${WEIGHTS_DIR}/sam2.1_hiera_tiny.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
fi

if [ ! -f "${WEIGHTS_DIR}/sam2.1_hiera_small.pt" ]; then
  echo "Downloading sam2.1_hiera_small.pt..."
  curl -L -o "${WEIGHTS_DIR}/sam2.1_hiera_small.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
fi

if [ ! -f "${WEIGHTS_DIR}/sam2.1_hiera_base_plus.pt" ]; then
  echo "Downloading sam2.1_hiera_base_plus.pt..."
  curl -L -o "${WEIGHTS_DIR}/sam2.1_hiera_base_plus.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
fi

if [ ! -f "${WEIGHTS_DIR}/sam2.1_hiera_large.pt" ]; then
  echo "Downloading sam2.1_hiera_large.pt..."
  curl -L -o "${WEIGHTS_DIR}/sam2.1_hiera_large.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi

echo "All weights downloaded to ${WEIGHTS_DIR}"