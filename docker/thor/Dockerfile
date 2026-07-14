# Thor / Drive OS 7.0.5 container for TRT-VLA-Lowering.
#
# Design: keep CUDA 13 + TensorRT 10.14 aligned with the host DOS image by
# mounting host libraries at runtime (see run.sh). The image only carries the
# Python ML stack.
#
# Build:
#   ./docker/thor/build.sh
#
# Default base is the local mlperf image already on this device. Override with:
#   docker build --build-arg BASE_IMAGE=ubuntu:24.04 -f docker/thor/Dockerfile .

ARG BASE_IMAGE=mlperf-automotive:jsuh-aarch64-base
FROM ${BASE_IMAGE}

# torch-tensorrt 2.10.x is the newest release that supports TensorRT 10.14.x
# shipped with Drive OS 7.0.5. Newer torch-tensorrt requires TRT >= 10.16.
ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_TENSORRT_VERSION=2.10.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep container CUDA paths consistent with the Thor host mount layout.
ENV CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:${PATH}

ARG HOST_UID=1216
ARG HOST_GID=1501
ARG HOST_USER=mwilliams

# Map the Thor host user so torch-tensorrt and file permissions work correctly.
RUN groupadd -f -g "${HOST_GID}" "${HOST_USER}" && \
    id -u "${HOST_USER}" >/dev/null 2>&1 || useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}"

WORKDIR /workspace/TRT-VLA-Lowering

# Pin torch-tensorrt 2.10.x to match TensorRT 10.14 on Drive OS 7.0.5.
# TensorRT pip wheels do not support Tegra/Thor — use the host-mounted Python
# package at runtime (see run.sh).
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --upgrade \
      "torch==${TORCH_VERSION}+cu130" \
      torchvision \
      --index-url "${TORCH_INDEX}" && \
    python3 -m pip install --no-deps "torch-tensorrt==${TORCH_TENSORRT_VERSION}" && \
    python3 -m pip install \
      packaging typing-extensions dllist psutil "nvidia-cuda-runtime-cu13==0.0.0a0"

COPY docker/common/requirements.container.txt /tmp/requirements.container.txt
RUN python3 -m pip install -r /tmp/requirements.container.txt

# LeRobot needs numpy 2.x; clear the mlperf base image pip constraint for our installs.
RUN rm -f /etc/pip/constraint.txt

# Convenience entrypoint used by run.sh
COPY docker/common/entrypoint.sh /usr/local/bin/trt-vla-entrypoint.sh
RUN chmod +x /usr/local/bin/trt-vla-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/trt-vla-entrypoint.sh"]
CMD ["bash"]
