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
ARG TENSORRT_VERSION=10.14.1.48

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

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

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      git \
      ca-certificates \
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

COPY docker/common/entrypoint.sh /usr/local/bin/trt-vla-entrypoint.sh
RUN chmod +x /usr/local/bin/trt-vla-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/trt-vla-entrypoint.sh"]
CMD ["bash"]
