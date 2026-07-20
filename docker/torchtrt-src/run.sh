#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

IMAGE_NAME="${IMAGE_NAME:-trt-vla-torchtrt-src}"
IMAGE_TAG="${IMAGE_TAG:-thor-opt}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"
LIBCUDA_GID="${LIBCUDA_GID:-$(getent group libcuda | cut -d: -f3)}"
LIBCUDA_GID="${LIBCUDA_GID:-2885}"
RUN_AS_ROOT="${TRT_VLA_RUN_AS_ROOT:-1}"
USE_HOST_TENSORRT="${USE_HOST_TENSORRT:-1}"
TORCH_TRT_REPO="${TORCH_TRT_REPO:-https://github.com/pytorch/TensorRT.git}"
TORCH_TRT_REF="${TORCH_TRT_REF:-v2.10.0}"
TORCH_TRT_KEEP_SRC="${TORCH_TRT_KEEP_SRC:-0}"
# Packaged TensorRT tree with include/ + lib/ (DriveOS headers are not under /usr/include).
TENSORRT_PKG_ROOT="${TENSORRT_PKG_ROOT:-/gtl/managed/builds/TensorRT-10.14.2.2}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/artifacts}"

if [[ ! -d "${TENSORRT_PKG_ROOT}" ]]; then
  echo "TENSORRT_PKG_ROOT does not exist: ${TENSORRT_PKG_ROOT}" >&2
  exit 1
fi
mkdir -p "${ARTIFACT_DIR}"

if [[ "${RUN_AS_ROOT}" == "1" ]]; then
  CONTAINER_USER="0:0"
else
  CONTAINER_USER="${HOST_UID}:${HOST_GID}"
fi

THOR_CUDA_LIB="${THOR_CUDA_LIB:-/usr/local/cuda-13.0/thor/targets/aarch64-linux/lib}"
THOR_LD_PRELOAD="/host-lib/libcudnn.so.9:/host-lib/libcudnn_cnn.so.9:/host-lib/libcudnn_ops.so.9:/host-lib/libcudnn_adv.so.9:${THOR_CUDA_LIB}/libcublas.so.13:${THOR_CUDA_LIB}/libcublasLt.so.13"

DOCKER_ARGS=(
  --net=host
  --runtime=runc
  --privileged
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -u "${CONTAINER_USER}"
  --group-add "${LIBCUDA_GID}"
  -w /tmp
  --entrypoint ""
  -e HOME="/home/${HOST_USER}"
  -e CUDA_HOME=/usr/local/cuda-13.0
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/av.libs:${THOR_CUDA_LIB}:/usr/local/cuda-13.0/lib64:/host-usr-lib:/host-lib
  -e LD_PRELOAD="${THOR_LD_PRELOAD}"
  -e USE_HOST_TENSORRT="${USE_HOST_TENSORRT}"
  -e TENSORRT_PKG_IN_CONTAINER=/host-trt-pkg
  -e TORCH_TRT_REPO="${TORCH_TRT_REPO}"
  -e TORCH_TRT_REF="${TORCH_TRT_REF}"
  -e TORCH_TRT_KEEP_SRC="${TORCH_TRT_KEEP_SRC}"
  -e ARTIFACT_DIR=/artifacts
  -e COMPILATION_MODE=opt
  -v "${SCRIPT_DIR}/build-opt.sh:/usr/local/bin/torchtrt-build-opt.sh:ro"
  -v "${ARTIFACT_DIR}:/artifacts"
  -v "${TENSORRT_PKG_ROOT}:/host-trt-pkg:ro"
  -v /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro
  -v /usr/local/cuda:/usr/local/cuda:ro
  -v /usr/local/lib/python3.12/dist-packages/tensorrt:/usr/local/lib/python3.12/dist-packages/tensorrt:ro
  -v /usr/local/lib/python3.12/dist-packages/tensorrt-10.14.1.31.dist-info:/usr/local/lib/python3.12/dist-packages/tensorrt-10.14.1.31.dist-info:ro
  -v /usr/lib:/host-usr-lib:ro
  -v /lib/aarch64-linux-gnu:/host-lib:ro
  -v /dev:/dev
  --rm
)

if [[ -d "/home/${HOST_USER}" ]]; then
  DOCKER_ARGS+=(-v "/home/${HOST_USER}:/home/${HOST_USER}")
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

if [[ -t 0 ]]; then
  DOCKER_ARGS+=(-it)
fi

echo "Starting ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  clone ${TORCH_TRT_REPO} @ ${TORCH_TRT_REF} into a temp dir, then opt-build"
echo "  TensorRT package: ${TENSORRT_PKG_ROOT}"
echo "  Wheel output: ${ARTIFACT_DIR}"
sudo docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}:${IMAGE_TAG}" "$@"
