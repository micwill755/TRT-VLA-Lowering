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

IMAGE_NAME="${IMAGE_NAME:-trt-vla-thor}"
IMAGE_TAG="${IMAGE_TAG:-7.0.5-cu130-trt10.14}"
BASE_IMAGE="${BASE_IMAGE:-mlperf-automotive:jsuh-aarch64-base}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"

# Drive OS sets default-runtime=nvidia but often omits nvidia-container-runtime.
# Symlink to runc so docker build/run work with host lib mounts instead.
if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
  echo "nvidia-container-runtime not found; linking to runc ..."
  sudo ln -sf "$(command -v runc)" /usr/local/sbin/nvidia-container-runtime
fi

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (BASE_IMAGE=${BASE_IMAGE}) ..."
sudo docker build \
  --network=host \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "HOST_UID=${HOST_UID}" \
  --build-arg "HOST_GID=${HOST_GID}" \
  --build-arg "HOST_USER=${HOST_USER}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo
echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Next: cp docker/thor/env.example docker/thor/.env && edit paths, then ./docker/thor/run.sh"
