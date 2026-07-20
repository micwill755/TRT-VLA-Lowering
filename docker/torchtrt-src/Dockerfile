# Thor image layer for building pytorch/TensorRT from source (opt).
#
# Base already has torch + torch-tensorrt wheels; this layer adds Bazelisk
# and C++ build tools so `pip install .` can run Bazel --compilation_mode=opt.
#
# Build:
#   ./docker/torchtrt-src/build.sh

ARG BASE_IMAGE=trt-vla-thor:7.0.5-cu130-trt10.14
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Bazelisk + common native build deps used by Torch-TensorRT's setup.py/Bazel.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
      build-essential \
      git \
      python3-dev \
      zip unzip \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    arch="$(uname -m)"; \
    case "${arch}" in \
      aarch64|arm64) BAZELISK_ARCH=arm64 ;; \
      x86_64|amd64)  BAZELISK_ARCH=amd64 ;; \
      *) echo "unsupported arch: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /usr/local/bin/bazelisk \
      "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-${BAZELISK_ARCH}"; \
    chmod +x /usr/local/bin/bazelisk; \
    ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel; \
    bazel version

# Do not inherit the Thor VLA entrypoint (it installs LeRobot on start).
ENTRYPOINT []
WORKDIR /tmp
CMD ["bash"]
