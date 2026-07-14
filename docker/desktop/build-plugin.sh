#!/usr/bin/env bash
# Build the TensorRT Edge-LLM plugin *inside* the desktop parity container so it
# links the same TensorRT 10.14 libs (libnvinfer.so.10) shipped by the pip wheel.
#
# Why: the host may only have TensorRT 11 (libnvinfer.so.11). Building here avoids
# the soname mismatch and does not require the gated NVIDIA TensorRT tarball —
# C++ headers come from the open-source TensorRT repo, libs come from pip.
#
# Usage:
#   EDGE_LLM_SRC=/home/micwilliams/workspace/TensorRT-Edge-LLM \
#     ./docker/desktop/build-plugin.sh
#
# Optional env:
#   TRT_INCLUDE_DIR   Prebuilt TensorRT 10.14 headers dir (skips the git clone).
#   TRT_OSS_REF       TensorRT OSS git ref for headers (default: release/10.14).
#   CUDA_ARCHS        SM archs for cubins (default: 89;120 for Ada + Blackwell).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

IMAGE_NAME="${DESKTOP_IMAGE_NAME:-trt-vla-desktop}"
IMAGE_TAG="${DESKTOP_IMAGE_TAG:-2.10-cu130-trt10.14}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
TRT_OSS_REF="${TRT_OSS_REF:-release/10.14}"
CUDA_ARCHS="${CUDA_ARCHS:-89;120}"

EDGE_LLM_SRC="${EDGE_LLM_SRC:-}"
if [[ -z "${EDGE_LLM_SRC}" ]]; then
  # Derive from EDGE_LLM_PLUGIN_SO if it points inside a checkout.
  if [[ -n "${EDGE_LLM_PLUGIN_SO:-}" ]]; then
    guess="$(cd "$(dirname "${EDGE_LLM_PLUGIN_SO}")/.." 2>/dev/null && pwd || true)"
    if [[ -f "${guess}/CMakeLists.txt" ]]; then
      EDGE_LLM_SRC="${guess}"
    fi
  fi
fi

if [[ -z "${EDGE_LLM_SRC}" || ! -f "${EDGE_LLM_SRC}/CMakeLists.txt" ]]; then
  echo "Set EDGE_LLM_SRC to the TensorRT-Edge-LLM source checkout." >&2
  echo "  EDGE_LLM_SRC=/path/to/TensorRT-Edge-LLM ./docker/desktop/build-plugin.sh" >&2
  exit 1
fi
EDGE_LLM_SRC="$(cd "${EDGE_LLM_SRC}" && pwd)"

# Acquire TensorRT 10.14 headers on the host (no gated download): clone the OSS repo.
HEADER_CACHE="${TRT_INCLUDE_DIR:-}"
if [[ -z "${HEADER_CACHE}" ]]; then
  HEADER_CACHE="${ROOT_DIR}/docker/desktop/.trt-oss-include"
  if [[ ! -f "${HEADER_CACHE}/NvInfer.h" ]]; then
    TMP_OSS="$(mktemp -d)"
    echo "Fetching TensorRT headers (${TRT_OSS_REF}) from github.com/NVIDIA/TensorRT ..."
    git clone --depth 1 --branch "${TRT_OSS_REF}" \
      https://github.com/NVIDIA/TensorRT.git "${TMP_OSS}"
    rm -rf "${HEADER_CACHE}"
    mkdir -p "${HEADER_CACHE}"
    cp -a "${TMP_OSS}/include/." "${HEADER_CACHE}/"
    rm -rf "${TMP_OSS}"
  fi
fi

if [[ ! -f "${HEADER_CACHE}/NvInfer.h" ]]; then
  echo "TensorRT headers not found in ${HEADER_CACHE} (no NvInfer.h)." >&2
  echo "Set TRT_INCLUDE_DIR to a TensorRT 10.14 include/ dir." >&2
  exit 1
fi
HEADER_CACHE="$(cd "${HEADER_CACHE}" && pwd)"

# In-container build: synthesize a TRT package dir (headers + symlinked pip libs),
# then run cmake/make. Output lands in the mounted source's build-plugin-trt10/.
IN_CONTAINER_BUILD=$(cat <<'EOS'
set -euo pipefail

PIP_TRT_LIB="$(python3 -c 'import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))')"
echo "pip TensorRT libs: ${PIP_TRT_LIB}"

PKG=/tmp/trt10pkg
rm -rf "${PKG}"
mkdir -p "${PKG}/include" "${PKG}/lib"
cp -a /trt-include/. "${PKG}/include/"

# Symlink versioned pip libs and create the unversioned names the linker needs.
for base in libnvinfer libnvinfer_plugin libnvonnxparser; do
  if [[ -f "${PIP_TRT_LIB}/${base}.so.10" ]]; then
    ln -sf "${PIP_TRT_LIB}/${base}.so.10" "${PKG}/lib/${base}.so.10"
    ln -sf "${base}.so.10" "${PKG}/lib/${base}.so"
  fi
done
ln -sf "${PIP_TRT_LIB}"/libnvinfer_builder_resource*.so* "${PKG}/lib/" 2>/dev/null || true

export LD_LIBRARY_PATH="${PKG}/lib:${PIP_TRT_LIB}:${LD_LIBRARY_PATH:-}"

BUILD_DIR=/src/build-plugin-trt10
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cmake -S /src -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR="${PKG}" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -DCUDA_CTK_VERSION=13.0 \
  -DENABLE_CUTE_DSL=OFF

cmake --build "${BUILD_DIR}" --target NvInfer_edgellm_plugin -j"$(nproc)"

echo "--- ldd (expect libnvinfer.so.10) ---"
SO="$(ls "${BUILD_DIR}"/libNvInfer_edgellm_plugin.so* 2>/dev/null | head -n1 || true)"
if [[ -n "${SO}" ]]; then
  ldd "${SO}" | grep -i nvinfer || true
  chown -R "${HOST_UID}:${HOST_GID}" "${BUILD_DIR}" 2>/dev/null || true
  echo "Built: ${SO}"
else
  echo "Build did not produce a plugin .so" >&2
  exit 1
fi
EOS
)

echo "Building Edge-LLM plugin inside ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker run --rm \
  --gpus all \
  -u 0:0 \
  -e HOST_UID="${HOST_UID}" \
  -e HOST_GID="${HOST_GID}" \
  -e CUDA_ARCHS="${CUDA_ARCHS}" \
  -v "${EDGE_LLM_SRC}:/src" \
  -v "${HEADER_CACHE}:/trt-include:ro" \
  --entrypoint bash \
  "${IMAGE_NAME}:${IMAGE_TAG}" -lc "${IN_CONTAINER_BUILD}"

echo
echo "Plugin built at: ${EDGE_LLM_SRC}/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0"
echo "Set in docker/desktop/.env:"
echo "  EDGE_LLM_PLUGIN_SO=${EDGE_LLM_SRC}/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0"
