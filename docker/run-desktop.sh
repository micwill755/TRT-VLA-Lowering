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
TRT_VLA_ROOT="${TRT_VLA_ROOT:-${ROOT_DIR}}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(dirname "${ROOT_DIR}")/lerobot}"
ENGINE_DIR="${ENGINE_DIR:-/tmp/pi05_edge_llm}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"
# Match Thor script behavior when set to 1 (disables cuDNN). Default off on desktop.
TRT_VLA_THOR="${TRT_VLA_THOR:-0}"

if [[ ! -d "${TRT_VLA_ROOT}" ]]; then
  echo "TRT_VLA_ROOT does not exist: ${TRT_VLA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${LEROBOT_ROOT}" ]]; then
  echo "WARNING: LEROBOT_ROOT does not exist yet: ${LEROBOT_ROOT}" >&2
fi

mkdir -p "${ENGINE_DIR}"

DOCKER_ARGS=(
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -u "${HOST_UID}:${HOST_GID}"
  -w /workspace/TRT-VLA-Lowering
  -e HOME="/tmp/home-${HOST_USER}"
  -e TRT_VLA_THOR="${TRT_VLA_THOR}"
  -e EDGE_LLM_PLUGIN_SO="${EDGE_LLM_PLUGIN_SO:-}"
  -e EDGELLM_PLUGIN_PATH="${EDGE_LLM_PLUGIN_SO:-}"
  -e ENGINE_DIR="${ENGINE_DIR}"
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
  -v "${TRT_VLA_ROOT}:/workspace/TRT-VLA-Lowering"
  -v "${ROOT_DIR}/docker/entrypoint.sh:/usr/local/bin/trt-vla-entrypoint.sh:ro"
  -v "${ENGINE_DIR}:${ENGINE_DIR}"
)

if [[ -d "${LEROBOT_ROOT}" ]]; then
  DOCKER_ARGS+=(-v "${LEROBOT_ROOT}:/workspace/lerobot")
fi

if [[ -n "${EDGE_LLM_PLUGIN_SO:-}" && -f "${EDGE_LLM_PLUGIN_SO}" ]]; then
  PLUGIN_DIR="$(cd "$(dirname "${EDGE_LLM_PLUGIN_SO}")" && pwd)"
  DOCKER_ARGS+=(-v "${PLUGIN_DIR}:${PLUGIN_DIR}:ro")
fi

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
if [[ -d "${HF_CACHE}" ]]; then
  DOCKER_ARGS+=(-v "${HF_CACHE}:${HF_CACHE}")
  DOCKER_ARGS+=(-e HF_HOME="${HF_CACHE}")
elif [[ -n "${HF_TOKEN:-}" ]]; then
  mkdir -p "${HF_CACHE}"
  DOCKER_ARGS+=(-v "${HF_CACHE}:${HF_CACHE}")
  DOCKER_ARGS+=(-e HF_HOME="${HF_CACHE}")
fi

DOCKER_ARGS+=(--rm)

if [[ $# -eq 0 ]]; then
  set -- bash
fi

if [[ -t 0 ]]; then
  DOCKER_ARGS+=(-it)
fi

echo "Starting ${IMAGE_NAME}:${IMAGE_TAG} (TRT_VLA_THOR=${TRT_VLA_THOR}) ..."
docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}:${IMAGE_TAG}" "$@"
