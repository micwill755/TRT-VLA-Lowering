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
TRT_VLA_ROOT="${TRT_VLA_ROOT:-${ROOT_DIR}}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(dirname "${ROOT_DIR}")/lerobot}"
ENGINE_DIR="${ENGINE_DIR:-/gtl/pi05_edge_llm}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"
# GID of the libcuda group so the container can read /usr/lib/libcuda.so.1
LIBCUDA_GID="${LIBCUDA_GID:-$(getent group libcuda | cut -d: -f3)}"
LIBCUDA_GID="${LIBCUDA_GID:-2885}"
# Thor driver libs (libcuda, libnvrm_*) are group-restricted; root in the
# container can load them. Set TRT_VLA_RUN_AS_ROOT=1 in docker/thor/.env.
RUN_AS_ROOT="${TRT_VLA_RUN_AS_ROOT:-0}"

if [[ ! -d "${TRT_VLA_ROOT}" ]]; then
  echo "TRT_VLA_ROOT does not exist: ${TRT_VLA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${LEROBOT_ROOT}" ]]; then
  echo "WARNING: LEROBOT_ROOT does not exist yet: ${LEROBOT_ROOT}" >&2
  echo "Clone LeRobot before running VLA scripts (see docker/RUNBOOK.md)." >&2
fi

mkdir -p "${ENGINE_DIR}"

if [[ "${RUN_AS_ROOT}" == "1" ]]; then
  CONTAINER_USER="0:0"
else
  CONTAINER_USER="${HOST_UID}:${HOST_GID}"
fi

THOR_CUDA_LIB="${THOR_CUDA_LIB:-/usr/local/cuda-13.0/thor/targets/aarch64-linux/lib}"
# Pip torch bundles generic nvidia/cu13 libs (wrong for Thor). LD_PRELOAD forces
# DriveOS cuBLAS + cuDNN from /host-lib (bind-mount of /lib/aarch64-linux-gnu).
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
  -w /workspace/TRT-VLA-Lowering
  -e HOME="/home/${HOST_USER}"
  -e CUDA_HOME=/usr/local/cuda-13.0
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/av.libs:${THOR_CUDA_LIB}:/usr/local/cuda-13.0/lib64:/host-usr-lib:/host-lib
  -e LD_PRELOAD="${THOR_LD_PRELOAD}"
  -e TRT_VLA_THOR=1
  -e EDGE_LLM_PLUGIN_SO="${EDGE_LLM_PLUGIN_SO:-}"
  -e EDGELLM_PLUGIN_PATH="${EDGE_LLM_PLUGIN_SO:-}"
  -e ENGINE_DIR="${ENGINE_DIR}"
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
  -v "${TRT_VLA_ROOT}:/workspace/TRT-VLA-Lowering"
  -v "${ROOT_DIR}/docker/common/entrypoint.sh:/usr/local/bin/trt-vla-entrypoint.sh:ro"
  -v "${ENGINE_DIR}:${ENGINE_DIR}"
  -v /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro
  -v /usr/local/cuda:/usr/local/cuda:ro
  -v /usr/local/lib/python3.12/dist-packages/tensorrt:/usr/local/lib/python3.12/dist-packages/tensorrt:ro
  -v /usr/local/lib/python3.12/dist-packages/tensorrt-10.14.1.31.dist-info:/usr/local/lib/python3.12/dist-packages/tensorrt-10.14.1.31.dist-info:ro
  -v /usr/lib:/host-usr-lib:ro
  -v /lib/aarch64-linux-gnu:/host-lib:ro
  -v /dev:/dev
)

if [[ -d "${LEROBOT_ROOT}" ]]; then
  DOCKER_ARGS+=(-v "${LEROBOT_ROOT}:/workspace/lerobot")
fi

if [[ -n "${EDGE_LLM_PLUGIN_SO:-}" && -f "${EDGE_LLM_PLUGIN_SO}" ]]; then
  PLUGIN_DIR="$(cd "$(dirname "${EDGE_LLM_PLUGIN_SO}")" && pwd)"
  DOCKER_ARGS+=(-v "${PLUGIN_DIR}:${PLUGIN_DIR}:ro")
fi

if [[ -d "/home/${HOST_USER}" ]]; then
  DOCKER_ARGS+=(-v "/home/${HOST_USER}:/home/${HOST_USER}")
fi

HF_CACHE="${HF_HOME:-/home/${HOST_USER}/.cache/huggingface}"
if [[ -d "${HF_CACHE}" ]]; then
  DOCKER_ARGS+=(-v "${HF_CACHE}:${HF_CACHE}")
elif [[ -n "${HF_TOKEN:-}" ]]; then
  mkdir -p "${HF_CACHE}"
  DOCKER_ARGS+=(-v "${HF_CACHE}:${HF_CACHE}")
fi

DOCKER_ARGS+=(--rm)

if [[ $# -eq 0 ]]; then
  set -- bash
fi

if [[ -t 0 ]]; then
  DOCKER_ARGS+=(-it)
fi

echo "Starting ${IMAGE_NAME}:${IMAGE_TAG} as ${HOST_USER} (${HOST_UID}:${HOST_GID}) ..."
sudo docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}:${IMAGE_TAG}" "$@"
