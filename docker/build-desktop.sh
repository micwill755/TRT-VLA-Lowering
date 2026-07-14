#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/docker/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

IMAGE_NAME="${DESKTOP_IMAGE_NAME:-trt-vla-desktop}"
IMAGE_TAG="${DESKTOP_IMAGE_TAG:-2.10-cu130-trt10.14}"
BASE_IMAGE="${DESKTOP_BASE_IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (BASE_IMAGE=${BASE_IMAGE}) ..."
docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "HOST_UID=${HOST_UID}" \
  --build-arg "HOST_GID=${HOST_GID}" \
  --build-arg "HOST_USER=${HOST_USER}" \
  -f "${ROOT_DIR}/docker/Dockerfile.desktop" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo
echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Next: cp docker/env.desktop.example docker/.env && edit paths, then ./docker/run-desktop.sh"
