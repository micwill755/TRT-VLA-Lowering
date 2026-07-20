#!/usr/bin/env bash
# Clone pytorch/TensorRT into a temp dir and install with Bazel --compilation_mode=opt.
#
# setup.py maps:
#   pip install -e . / develop  → --compilation_mode=dbg
#   pip install .    / install  → --compilation_mode=opt   ← this script
#
# Usage (from host):
#   ./docker/torchtrt-src/run.sh /usr/local/bin/torchtrt-build-opt.sh
set -euo pipefail

REPO_URL="${TORCH_TRT_REPO:-https://github.com/pytorch/TensorRT.git}"
REPO_REF="${TORCH_TRT_REF:-v2.10.0}"
KEEP_SRC="${TORCH_TRT_KEEP_SRC:-0}"
# Bind-mounted DriveOS / packaged TensorRT tree (include/, lib/).
TRT_PKG="${TENSORRT_PKG_IN_CONTAINER:-/host-trt-pkg}"

if ! command -v bazel >/dev/null 2>&1 && ! command -v bazelisk >/dev/null 2>&1; then
  echo "bazel/bazelisk not found in PATH" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git not found in PATH" >&2
  exit 1
fi

if ! python3 -c "import torch" >/dev/null 2>&1; then
  echo "torch is not importable in this container" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d /tmp/torchtrt-src.XXXXXX)"
SRC_DIR="${TMP_ROOT}/TensorRT"
TRT_STAGE="${TMP_ROOT}/trt-local"
CUDA_STAGE="${TMP_ROOT}/cuda-local"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-13.0}"
THOR_CUDA_ROOT="${THOR_CUDA_ROOT:-${CUDA_ROOT}/thor}"
cleanup() {
  if [[ "${KEEP_SRC}" == "1" ]]; then
    echo "Keeping temp tree at ${TMP_ROOT} (TORCH_TRT_KEEP_SRC=1)"
  else
    rm -rf "${TMP_ROOT}"
  fi
}
trap cleanup EXIT

echo "Cloning ${REPO_URL} @ ${REPO_REF} -> ${SRC_DIR}"
if git ls-remote --exit-code --heads --tags "${REPO_URL}" "${REPO_REF}" >/dev/null 2>&1; then
  git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${SRC_DIR}"
else
  git clone --filter=blob:none --no-checkout "${REPO_URL}" "${SRC_DIR}"
  git -C "${SRC_DIR}" fetch --depth 1 origin "${REPO_REF}"
  git -C "${SRC_DIR}" checkout FETCH_HEAD
fi

cd "${SRC_DIR}"
echo "Source revision: $(git rev-parse --short HEAD) ($(git describe --tags --always 2>/dev/null || true))"

export TORCH_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
echo "Using TORCH_PATH=${TORCH_PATH}"
echo "torch=$(python3 -c 'import torch; print(torch.__version__)')"
python3 -c "import tensorrt as trt; print('tensorrt', trt.__version__)" || true

if [[ ! -f "${TORCH_PATH}/lib/libtorch.so" ]]; then
  echo "Installed torch package is missing lib/libtorch.so under ${TORCH_PATH}" >&2
  exit 1
fi

stage_host_cuda() {
  # Thor splits the toolkit: core CUDA under CUDA_ROOT, math libs (cublas,
  # cusparse, cufft, curand, cusolver, ...) under CUDA_ROOT/thor. Bazel and
  # LibTorch headers expect a unified include/ + lib64/ tree.
  if [[ ! -d "${CUDA_ROOT}/include" ]]; then
    echo "CUDA include not found under ${CUDA_ROOT}/include" >&2
    exit 1
  fi
  if [[ ! -d "${THOR_CUDA_ROOT}/targets/aarch64-linux/include" ]]; then
    echo "Thor CUDA include not found under ${THOR_CUDA_ROOT}/targets/aarch64-linux/include" >&2
    exit 1
  fi

  mkdir -p "${CUDA_STAGE}/include" "${CUDA_STAGE}/lib64"
  ln -sft "${CUDA_STAGE}/include" "${CUDA_ROOT}/include/"*
  ln -sft "${CUDA_STAGE}/include" "${THOR_CUDA_ROOT}/targets/aarch64-linux/include/"*
  ln -sft "${CUDA_STAGE}/lib64" "${CUDA_ROOT}/lib64/"* 2>/dev/null || true
  ln -sft "${CUDA_STAGE}/lib64" "${THOR_CUDA_ROOT}/targets/aarch64-linux/lib/"*

  if [[ ! -e "${CUDA_STAGE}/include/cusparse.h" ]]; then
    echo "cusparse.h missing from staged CUDA include tree" >&2
    exit 1
  fi

  echo "Staged host CUDA for Bazel at ${CUDA_STAGE}"
  ls "${CUDA_STAGE}/include/"{cublas,cusparse,cufft,curand,cusolver}*.h 2>/dev/null | head -20
  ls "${CUDA_STAGE}/lib64/"libcudart.so* | head -3
}

stage_host_tensorrt() {
  if [[ ! -f "${TRT_PKG}/include/NvInfer.h" ]]; then
    echo "TensorRT headers not found under ${TRT_PKG}/include" >&2
    echo "Set TENSORRT_PKG_ROOT in docker/torchtrt-src/.env" >&2
    exit 1
  fi
  if [[ ! -e "${TRT_PKG}/lib/libnvinfer.so" && ! -e "${TRT_PKG}/lib/libnvinfer.so.10" ]]; then
    echo "libnvinfer not found under ${TRT_PKG}/lib" >&2
    exit 1
  fi

  # third_party/tensorrt/local/BUILD expects Debian-style aarch64 paths.
  mkdir -p "${TRT_STAGE}/include/aarch64-linux-gnu" "${TRT_STAGE}/lib/aarch64-linux-gnu"
  ln -sft "${TRT_STAGE}/include/aarch64-linux-gnu" "${TRT_PKG}/include/"*.h
  ln -sft "${TRT_STAGE}/lib/aarch64-linux-gnu" "${TRT_PKG}/lib/"libnvinfer* \
    "${TRT_PKG}/lib/"libnvonnxparser* \
    "${TRT_PKG}/lib/"libnvinfer_plugin* 2>/dev/null || true

  if [[ ! -e "${TRT_STAGE}/lib/aarch64-linux-gnu/libnvinfer.so" ]]; then
    local so
    so="$(ls -1 "${TRT_STAGE}/lib/aarch64-linux-gnu"/libnvinfer.so.* 2>/dev/null | head -1 || true)"
    if [[ -n "${so}" ]]; then
      ln -sf "$(basename "${so}")" "${TRT_STAGE}/lib/aarch64-linux-gnu/libnvinfer.so"
    fi
  fi

  echo "Staged host TensorRT for Bazel at ${TRT_STAGE}"
  ls -la "${TRT_STAGE}/include/aarch64-linux-gnu/NvInfer.h"
  ls -la "${TRT_STAGE}/lib/aarch64-linux-gnu/libnvinfer.so"* | head -5
}

patch_module_bazel() {
  local module_file="MODULE.bazel"
  local host_trt="${1:-}"
  local host_cuda="${2:-}"
  python3 - "${module_file}" "${TORCH_PATH}" "${host_trt}" "${host_cuda}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
torch_path = sys.argv[2]
host_trt = sys.argv[3]
host_cuda = sys.argv[4]
text = path.read_text()

def comment_http_archive(text: str, name: str) -> str:
    pattern = re.compile(
        rf'(?ms)^http_archive\(\n(?:^[^\n]*\n)*?^[ \t]*name = "{name}",\n(?:^[^\n]*\n)*?^\)\n'
    )

    def _comment(m: re.Match[str]) -> str:
        return "".join("#" + line if line.strip() else line for line in m.group(0).splitlines(True))

    new_text, n = pattern.subn(_comment, text)
    if n:
        print(f"Commented out http_archive name={name}")
    return new_text

def has_active_local_repo(text: str, name: str) -> bool:
    return re.search(
        rf'(?m)^new_local_repository\(\n(?:^[^\n]*\n)*?^[ \t]*name = "{name}",',
        text,
    ) is not None

def replace_local_repo_path(text: str, name: str, new_path: str) -> str:
    pattern = re.compile(
        rf'(?ms)^(new_local_repository\(\n(?:^[^\n]*\n)*?^[ \t]*name = "{name}",\n)((?:^[^\n]*\n)*?)(^\))',
    )

    def _repl(m: re.Match[str]) -> str:
        body = m.group(2)
        if re.search(r'^[ \t]*path = "', body, re.M):
            body = re.sub(r'(?m)^[ \t]*path = ".*"', f'    path = "{new_path}"', body, count=1)
        else:
            body = f'    path = "{new_path}",\n' + body
        return m.group(1) + body + m.group(3)

    new_text, n = pattern.subn(_repl, text, count=1)
    if n:
        print(f"Updated @{name} path -> {new_path}")
        return new_text
    return text

# Always prefer the installed Python torch package as @libtorch on aarch64.
# The default http_archive points at an x86_64 nightly zip.
text = comment_http_archive(text, "libtorch")
if not has_active_local_repo(text, "libtorch"):
    text += f'''
new_local_repository(
    name = "libtorch",
    path = "{torch_path}",
    build_file = "@//third_party/libtorch:BUILD"
)
'''
    print(f"Enabled local @libtorch -> {torch_path}")
else:
    print("local @libtorch already present")

if host_cuda:
    if has_active_local_repo(text, "cuda"):
        text = replace_local_repo_path(text, "cuda", host_cuda)
    else:
        text += f'''
new_local_repository(
    name = "cuda",
    path = "{host_cuda}",
    build_file = "@//third_party/cuda:BUILD"
)
'''
        print(f"Enabled local @cuda -> {host_cuda}")

if host_trt:
    for name in ("tensorrt", "tensorrt_sbsa", "tensorrt_l4t"):
        text = comment_http_archive(text, name)

    for name in ("tensorrt_sbsa", "tensorrt", "tensorrt_l4t"):
        if has_active_local_repo(text, name):
            continue
        text += f'''
new_local_repository(
   name = "{name}",
   path = "{host_trt}",
   build_file = "@//third_party/tensorrt/local:BUILD"
)
'''
        print(f"Enabled local @{name} -> {host_trt}")

path.write_text(text)
print(f"Patched {path}")
PY
}

stage_host_cuda
if [[ "${USE_HOST_TENSORRT:-1}" == "1" ]]; then
  stage_host_tensorrt
  patch_module_bazel "${TRT_STAGE}/" "${CUDA_STAGE}/"
else
  patch_module_bazel "" "${CUDA_STAGE}/"
fi

# Thor/mlperf base removes /etc/pip/constraint.txt but pip.conf may still
# point at it. Match the Thor entrypoint workaround.
export PIP_CONSTRAINT=/dev/null
ARTIFACT_DIR="${ARTIFACT_DIR:-/artifacts}"
mkdir -p "${ARTIFACT_DIR}"

echo "Uninstalling any existing torch-tensorrt wheel ..."
python3 -m pip uninstall -y torch-tensorrt torch_tensorrt 2>/dev/null || true

echo
echo "Building persistent opt wheel into ${ARTIFACT_DIR} ..."
# No editable install: that would force dbg via DevelopCommand / EditableWheelCommand.
python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir "${ARTIFACT_DIR}" .

WHEEL="$(ls -1t "${ARTIFACT_DIR}"/torch_tensorrt-*.whl | head -1)"
echo "Installing built wheel: ${WHEEL}"
python3 -m pip install --no-deps "${WHEEL}"

echo
python3 - <<'PY'
import torch_tensorrt
print("Installed torch_tensorrt", torch_tensorrt.__version__)
PY

echo
echo "Done. Opt source build is installed in this container's Python env."
echo "Persistent wheel: ${WHEEL}"
