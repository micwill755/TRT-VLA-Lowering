# Desktop x86_64 container matching the Thor stack for parity benchmarking:
#   PyTorch 2.10+cu130, torch-tensorrt 2.10, TensorRT 10.14 (pip wheels).
#
# Build:
#   ./docker/desktop/build.sh
#
# This is NOT the native 5090 dev stack (torch 2.14 / TRT 11). Use it when you
# want apples-to-apples comparison with DRIVE AGX Thor compile behavior.

ARG BASE_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_TENSORRT_VERSION=2.10.0
# 10.14.1.48 (no suffix) shipped broken bindings/libs and is excluded by
# torch-tensorrt 2.10 (!=10.14.1.48). NVIDIA re-released it as .post1.
ARG TENSORRT_VERSION=10.14.1.48.post1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=dev

# Ubuntu 24.04 base ships a default "ubuntu" user at UID 1000, which can collide
# with HOST_UID. Reuse any existing account for that UID instead of failing.
RUN groupadd -f -g "${HOST_GID}" "${HOST_USER}" && \
    if ! id -u "${HOST_USER}" >/dev/null 2>&1; then \
      existing_user="$(getent passwd "${HOST_UID}" | cut -d: -f1)"; \
      if [ -n "${existing_user}" ]; then \
        usermod -l "${HOST_USER}" -g "${HOST_GID}" "${existing_user}" && \
        usermod -d "/home/${HOST_USER}" -m "${HOST_USER}" 2>/dev/null || true; \
      else \
        useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}"; \
      fi; \
    fi

WORKDIR /workspace/TRT-VLA-Lowering

# build-essential + cmake let you compile the Edge-LLM plugin inside the
# container against the pip TensorRT 10.14 libs (see docker/desktop/build-plugin.sh).
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      git \
      ca-certificates \
      build-essential \
      cmake \
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade --ignore-installed pip setuptools wheel && \
    python3 -m pip install --upgrade \
      "torch==${TORCH_VERSION}+cu130" \
      torchvision \
      --index-url "${TORCH_INDEX}" && \
    python3 -m pip install --no-deps "torch-tensorrt==${TORCH_TENSORRT_VERSION}" && \
    python3 -m pip install \
      packaging typing-extensions dllist psutil "nvidia-cuda-runtime-cu13==0.0.0a0" && \
    python3 -m pip install "tensorrt==${TENSORRT_VERSION}" "tensorrt-cu13==${TENSORRT_VERSION}"

COPY docker/common/requirements.container.txt /tmp/requirements.container.txt
RUN python3 -m pip install -r /tmp/requirements.container.txt

# torch+cu130 bundles nvidia-cublas/cudnn 13.1 via pip. The base cuda:13.0 image
# ships older libcublasLt 13.0 — if it wins LD_LIBRARY_PATH, large GEMMs on
# Blackwell (sm_120) fail with CUBLAS_STATUS_INVALID_VALUE / NOT_INITIALIZED.
RUN python3 - <<'PY'
import os
import nvidia.cublas
import nvidia.cudnn
cu13 = os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib")
cudnn = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
assert os.path.isfile(os.path.join(cu13, "libcublas.so.13")), cu13
assert os.path.isfile(os.path.join(cudnn, "libcudnn.so.9")), cudnn
with open("/etc/trt-vla-ldpath", "w") as f:
    f.write(f"{cu13}:{cudnn}:/usr/local/cuda/lib64\n")
print(open("/etc/trt-vla-ldpath").read().strip())
PY
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY docker/common/entrypoint.sh /usr/local/bin/trt-vla-entrypoint.sh
RUN chmod +x /usr/local/bin/trt-vla-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/trt-vla-entrypoint.sh"]
CMD ["bash"]
