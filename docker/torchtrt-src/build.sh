#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

IMAGE_NAME="${IMAGE_NAME:-trt-vla-torchtrt-src}"
IMAGE_TAG="${IMAGE_TAG:-thor-opt}"
BASE_IMAGE="${BASE_IMAGE:-trt-vla-thor:7.0.5-cu130-trt10.14}"

if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
  echo "nvidia-container-runtime not found; linking to runc ..."
  sudo ln -sf "$(command -v runc)" /usr/local/sbin/nvidia-container-runtime
fi

if ! sudo docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "Base image missing: ${BASE_IMAGE}" >&2
  echo "Build it first: ./docker/thor/build.sh" >&2
  exit 1
fi

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (BASE_IMAGE=${BASE_IMAGE}) ..."
sudo docker build \
  --network=host \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  "${SCRIPT_DIR}"

echo
echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Next: ./docker/torchtrt-src/run.sh /usr/local/bin/torchtrt-build-opt.sh"
echo "      (clones pytorch/TensorRT into a temp dir, then pip install . → opt)"
