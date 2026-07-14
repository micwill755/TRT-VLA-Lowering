#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
LEGACY_ENV_FILE="${ROOT_DIR}/docker/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
elif [[ -f "${LEGACY_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LEGACY_ENV_FILE}"
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
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo
echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Next: cp docker/desktop/env.example docker/desktop/.env && edit paths, then ./docker/desktop/run.sh"
