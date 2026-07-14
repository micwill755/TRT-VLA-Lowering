# TRT-VLA-Lowering Container Runbook (Thor / Drive OS 7.0.5)

Reusable guide for running `test_vla_pi05_e2e.py` and related scripts in an isolated Docker environment on **DRIVE AGX Thor** with:

- **Drive OS**: `7.0.5.0-43328940`
- **TensorRT**: `10.14.2.2`
- **CUDA**: `13.0`

## Why containers on Thor?

Docker on Thor aarch64 is **experimental** (upstream Ubuntu runtime, not fully validated by NVIDIA for production). This runbook uses a **host-library bind-mount** pattern so the container reuses the DOS CUDA/TRT stack instead of bundling mismatched versions.

Validated on this device:

| Check | Result |
|-------|--------|
| Docker installed | Yes (`28.2.2`) |
| Default runtime `nvidia` configured | Yes, but `nvidia-container-runtime` binary is **missing** |
| Bridge networking | **Broken** (`operation not supported`) — use `--net=host` |
| Existing `mlperf-automotive` image | CUDA **12.8** — mismatched with host CUDA **13.0** |
| Host `libcuda.so.1` | `/usr/lib/libcuda.so.1` (requires correct mount + privileges) |

## Architecture

```text
Host (Thor DOS 7.0.5)
├── CUDA 13.0          (/usr/local/cuda-13.0)  ─┐
├── libcuda + GPU devs (/usr/lib, /dev/nvgpu)  ─┼─ bind-mount ─► Container
├── TensorRT 10.14.2   (system .deb packages)  ─┘
└── Docker image carries only:
    ├── Python 3.12
    ├── PyTorch 2.10+cu130 (aarch64)
    ├── torch-tensorrt 2.10.x (last release supporting TRT 10.14)
    └── pip deps (transformers, etc.)

Mounted at runtime:
├── TRT-VLA-Lowering repo
├── LeRobot repo
└── libNvInfer_edgellm_plugin.so (Edge-LLM plugin)
```

## One-time prerequisites

### 1. Confirm host versions

```bash
cat /etc/nvidia/version-ubuntu-rootfs.txt
# Expected: 7.0.5.0-43328940

dpkg -l | grep libnvinfer10
# Expected: 10.14.2.2-1+cuda13.0
```

### 2. Docker access

Docker requires `sudo` on this device (user is not in the `docker` group).

```bash
sudo docker info
```

Optional (admin): add your user to the `docker` group to avoid `sudo`.

### 3. Fix GPU runtime (required on this device)

`/etc/docker/daemon.json` sets `"default-runtime": "nvidia"`, but `nvidia-container-runtime` is not installed. Both **build** and **run** will fail until this is addressed.

`docker/thor/build.sh` auto-creates a symlink as a workaround:

```bash
sudo ln -sf "$(command -v runc)" /usr/local/sbin/nvidia-container-runtime
```

For runs, also use `--runtime=runc` with host library mounts (handled by `docker/thor/run.sh`):

Workaround used by this runbook:

```bash
--runtime=runc --net=host --privileged \
  -v /dev:/dev \
  -v /usr/lib:/host-usr-lib:ro \
  -v /lib/aarch64-linux-gnu:/host-lib:ro \
  -v /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro
```

### 4. Clone LeRobot

The project `requirements.txt` expects LeRobot as a sibling directory:

```bash
cd /home/mwilliams/test
git clone https://github.com/huggingface/lerobot.git
# Or your internal fork with PI0.5 support
```

### 5. Build the Edge-LLM TensorRT plugin

Export scripts call `load_plugins_for_trt()` and require:

```bash
export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
```

Built natively on this Thor device from `/home/mwilliams/tensorrt-edge-llm`
(branch `cosmos`). Result:

```text
/home/mwilliams/tensorrt-edge-llm/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0
```

**Build steps (reproducible):**

```bash
cd /home/mwilliams/tensorrt-edge-llm

# 1. Init the only submodule the plugin needs (headers only; NVTX not needed)
git submodule update --init 3rdParty/nlohmannJson

# 2. Configure for DriveOS Thor (auto-thor, CUDA 13.0, SM110).
#    TensorRT dev headers are NOT installed system-wide — only runtime libs.
#    Point TRT_PACKAGE_DIR at the full TRT package under /gtl (headers + .so).
rm -rf build-plugin-trt10 && mkdir build-plugin-trt10 && cd build-plugin-trt10
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/gtl/managed/builds/TensorRT-10.14.2.2 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=auto-thor \
  -DCUDA_CTK_VERSION=13.0 \
  -DENABLE_CUTE_DSL=OFF

# 3. Build only the plugin target (~2-3 min)
make NvInfer_edgellm_plugin -j"$(nproc)"
```

Then set in `docker/.env`:

```bash
EDGE_LLM_PLUGIN_SO=/home/mwilliams/tensorrt-edge-llm/build-plugin-trt10/libNvInfer_edgellm_plugin.so.1.0
```

**Notes / gotchas:**

- TensorRT **dev headers** (`NvInfer.h`, `NvOnnxParser.h`) and unversioned
  `.so` symlinks are not installed under `/usr`. The full package lives at
  `/gtl/managed/builds/TensorRT-10.14.2.2/` (matches runtime `10.14.2.2`).
- `ENABLE_CUTE_DSL=OFF` is sufficient for the VLA attention plugin. Use `ALL`
  only if you need Qwen3.5 GDN / NVFP4 MoE runtime paths.
- Verified the `.so` registers **`AttentionPlugin`** and **`ViTAttentionPlugin`**
  (the two the VLA scripts patch in), plus 58 other TRT creators.
- The toolchain uses `aarch64-linux-gnu-gcc`, which on this native aarch64
  device is the system GCC 13.3 — no cross-compile setup needed.

### GPU memory carveout (Thor — required for Pi0.5)

On Thor, CUDA `total_memory` is **not fixed VRAM**. It equals the **GPU
carveout** reserved from unified system RAM via 2 MB hugetlbfs huge pages.
The boot default is **3072 pages = 6 GB** (`/etc/systemd/scripts/nv_hugetlbfs_init.sh`),
which is too small for Pi0.5 TRT compile.

#### What it does

Thor has **unified memory** (~58 GB total). A slice is reserved for the GPU
via hugetlbfs; CUDA only sees that slice as `torch.cuda.get_device_properties().total_memory`.
The rest is **host RAM** for Linux, Docker, and TensorRT's CPU-side builder.

```
58 GB total RAM
├── GPU carveout (hugetlbfs)  → CUDA / GPU weights / TRT GPU builder
└── Host RAM pool             → OS, Docker, TRT graph on CPU (~35 GB peak for language)
```

**Trade-off:** more carveout → more GPU memory, **less** host RAM. Language TRT
compile needs both (~20 GB GPU **and** ~35 GB host peak), so you cannot max out
only one side.

#### Check current size

```bash
cat /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
# pages × 2 MB = GPU memory seen by CUDA

free -h
```

Inside the container (restart containers after changing carveout):

```bash
./docker/thor/run.sh python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
```

#### Increase at runtime (until reboot)

Stop GPU containers first:

```bash
sudo docker stop $(sudo docker ps -q) 2>/dev/null
```

Set pages (`pages = target_GB × 512`):

```bash
# 12 GB — vision + load; language compile may GPU-OOM
sudo bash -c 'echo 6144 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'

# 16 GB — language compile GPU-OOM'd here (builder needs ~9 GB on top of weights)
sudo bash -c 'echo 8192 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'

# 20 GB — recommended for full Pi0.5 e2e compile (validated)
sudo bash -c 'echo 10240 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'
```

Verify:

```bash
python3 -c "p=int(open('/sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages').read()); print(f'GPU carveout: {p*2/1024:.1f} GB, host pool: ~{58-p*2/1024:.1f} GB')"
```

#### Make permanent

Edit `NumPages=10240` (20 GB) in `/etc/systemd/scripts/nv_hugetlbfs_init.sh`, then reboot.

Official reference: [DriveOS GPU Carveout](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/platform-customization/Carveout_Customization_and_Profiling/carveout_customization.html)

### Host swap file (Thor — required for language TRT compile)

#### What it does

A **swap file** is disk space used as overflow when **host RAM** is full. It does
**not** add GPU memory. During language TRT compile, host RSS peaks at **~35 GB**.
With a 20 GB carveout only **~38 GB** host RAM remains — tight once Docker and
the OS are included. Swap prevents the kernel OOM-killer (`Killed` with no Python
traceback) during that spike.

Swap is slower than RAM but acceptable for a **one-time engine build**. Inference
of finished engines uses far less memory.

#### Why `/gtl` and not `/`

The root filesystem is only **~26 GB** and fills up when writing multi-GB
`.engine` files. Use the NVMe mount at `/gtl` (~1.8 TB) for both swap and engine output.

Set in `docker/.env`:

```bash
ENGINE_DIR=/gtl/pi05_edge_llm
```

#### Create swap (one-time)

```bash
# 32 GB swap on NVMe (adjust size as needed)
sudo fallocate -l 32G /gtl/swapfile
sudo chmod 600 /gtl/swapfile
sudo mkswap /gtl/swapfile
sudo swapon /gtl/swapfile

# Persist across reboot
grep -q '/gtl/swapfile' /etc/fstab || echo '/gtl/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Verify:

```bash
free -h
swapon --show
```

During compile, watch usage:

```bash
watch -n2 'free -h; swapon --show'
```

If `Swap: used` climbs during language compile, swap is doing its job.

#### Disable swap (optional test)

To confirm host RAM alone is insufficient without swap:

```bash
sudo swapoff /gtl/swapfile
# run compile — expect Killed during language TRT build
sudo swapon /gtl/swapfile
```

### Pi0.5 compile memory requirements (Thor, `test_vla_pi05_e2e.py`)

Measured on DRIVE AGX Thor (~58 GB RAM, DriveOS 7.0.5, TRT 10.14). Values are
**compile-time** peaks for the default libero frame / fp16 policy — not inference.

| Stage | GPU carveout (min) | Host RAM peak | Engine on disk | Failure mode if short |
|-------|-------------------:|--------------:|---------------:|------------------------|
| **Policy load** | 6 GB (boot default) too small; **12 GB+** to fit fp16 weights | ~8 GB | — | CUDA OOM loading weights |
| **Vision TRT compile** | **12–16 GB** | ~15–20 GB | (in-process; not saved by default) | CUDA OOM or builder warning |
| **Language TRT compile** | **20 GB** recommended (14 GB → GPU OOM: builder needs **~9.2 GB** on top of weights) | **~35 GB** RSS | **~9.2 GB** (`language.engine`) | `Killed` (host OOM) without swap; GPU OOM if carveout under ~18 GB |
| **Diffusion TRT compile** | **20 GB** (after freeing language TRT runtime) | ~12–18 GB | varies | CUDA OOM if language TRT still on GPU |
| **Full e2e (all stages)** | **20 GB carveout** | **~35 GB host** + swap headroom | **10+ GB** total engines | See rows above |

**Recommended Thor compile config (validated end-to-end, `close%=100.0` all stages):**

| Setting | Value |
|---------|-------|
| GPU carveout | **10240 pages = 20 GB** |
| Host swap | **32 GB** at `/gtl/swapfile` |
| Engine output | `ENGINE_DIR=/gtl/pi05_edge_llm` |
| Root disk | Keep free; do not write engines to `/tmp` on `/` |

**Carveout vs swap — do you need both?**

| Config | Vision | Language compile | Notes |
|--------|--------|------------------|-------|
| 20 GB carveout, no swap | OK | **Killed** (~35 GB host RSS) | GPU OK, host OOM |
| 14 GB carveout, no swap | OK | **GPU OOM** (builder ~9.2 GB) | More host RAM, not enough GPU |
| **20 GB carveout + 32 GB swap** | OK | **OK** | Validated full e2e |

Inference of saved engines needs substantially less than compile peaks.

### GPU access: the `libcuda` group (important)

`/usr/lib/libcuda.so.1` is mode `----r-----`, owner `root:libcuda`. Only members
of the **`libcuda`** group can load it. User `mwilliams` is currently only in
`ml-perf`, which is why:

- `ctypes.CDLL("libcuda.so.1")` fails with **`Permission denied`** as your user
- `torch.cuda.is_available()` is `False` in the container

Fix (needs admin, then re-login):

```bash
sudo usermod -aG libcuda mwilliams
# log out / back in, or: newgrp libcuda
```

Until then, GPU work requires `sudo`. In the container, also add the group:
add `--group-add libcuda` (or the numeric GID) to `docker/thor/run.sh`.

### TensorRT Python bindings (Thor-specific)

`pip install tensorrt` **fails on Tegra/Thor** (`TensorRT does not currently build wheels for Tegra systems`). The container mounts the host-installed package instead:

```text
/usr/local/lib/python3.12/dist-packages/tensorrt  →  container site-packages
```

If your host TensorRT dist-info version differs from `10.14.1.31`, update the second mount in `docker/thor/run.sh`.

An empty `/home/mwilliams/torch/` directory can shadow the real PyTorch package on the **host**. Inside the container this is not an issue, but remove or rename it on the host if you also run scripts outside Docker:

```bash
# Only if this directory is empty and unintentional:
rmdir ~/torch 2>/dev/null || true
```

## Quick start

```bash
cd /home/mwilliams/test/TRT-VLA-Lowering

# 1. Configure paths
cp docker/thor/env.example docker/thor/.env
# Edit docker/thor/.env:
#   LEROBOT_ROOT=...
#   EDGE_LLM_PLUGIN_SO=...

# 2. Build the image (reuses local mlperf base; upgrades PyTorch to cu130)
chmod +x docker/thor/build.sh docker/thor/run.sh docker/common/entrypoint.sh
./docker/thor/build.sh

# 3. Open an interactive shell in the container
./docker/thor/run.sh

# 4. Inside the container — verify stack
python3 - <<'PY'
import torch, torch_tensorrt, tensorrt as trt
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_tensorrt", torch_tensorrt.__version__)
print("tensorrt", trt.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

# 5. Run the Pi0.5 parity script
cd vla
python3 test_vla_pi05_e2e.py
```

### Run a one-off command without interactive shell

```bash
./docker/thor/run.sh python3 vla/test_vla_pi05_e2e.py
```

## Configuration reference (`docker/thor/.env`)

| Variable | Purpose |
|----------|---------|
| `TRT_VLA_ROOT` | Path to this repo on the host |
| `LEROBOT_ROOT` | Path to cloned LeRobot repo |
| `EDGE_LLM_PLUGIN_SO` | Path to `libNvInfer_edgellm_plugin.so` |
| `ENGINE_DIR` | Where compiled `.engine` files are written |
| `IMAGE_NAME` / `IMAGE_TAG` | Docker image name for rebuilds |
| `HOST_UID` / `HOST_GID` / `HOST_USER` | Map container user to host user |

## Validation checklist

Run after `./docker/thor/run.sh` opens a shell:

```bash
# Versions aligned with Thor stack
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

# GPU visible
python3 -c "import torch; print(torch.cuda.is_available())"

# Edge-LLM plugin path
test -n "$EDGE_LLM_PLUGIN_SO" && test -f "$EDGE_LLM_PLUGIN_SO" && echo "plugin ok"

# LeRobot importable
python3 -c "from lerobot.policies.pi05 import PI05Policy; print('lerobot ok')"
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch_tensorrt'`

Image not built yet, or running `python3` on the **host** instead of inside the container. Use `./docker/thor/run.sh`.

### `nvidia-container-runtime: executable file not found`

Expected on this device. `docker/thor/run.sh` forces `--runtime=runc` with host lib mounts.

### `operation not supported` (bridge networking)

Use `--net=host` (already in `docker/thor/run.sh`).

### `libcuda.so.1: cannot open shared object file`

Missing `/usr/lib` bind mount or container started without `--privileged`. Use `docker/thor/run.sh`.

### `libcublas.so.* not found` inside container

PyTorch in the image is not aligned with host CUDA. Rebuild with `./docker/thor/build.sh` (upgrades to `torch+cu130`).

### Segfault on GPU `Conv2d` / vision forward (`GridVisionExportModule`)

Pip PyTorch ships generic `nvidia/cu13` cuBLAS and cuDNN wheels that do **not**
match DriveOS Thor. Symptoms: `matmul` may work after cuBLAS preload, but
`nn.Conv2d(...).cuda()` still segfaults.

`docker/thor/run.sh` sets `LD_PRELOAD` for DriveOS cuBLAS (required for GEMM) and
`TRT_VLA_THOR=1` so test scripts disable the pip cuDNN backend (preloading host
cuDNN alone is not enough — `Conv2d` still segfaults). VLA scripts call
`configure_thor_pytorch()` from `trt/utils.py` automatically.

**Restart the container** after changing `run.sh` — an existing shell keeps the
old environment.

Quick check inside the container:

```bash
python3 -c "
from trt.utils import configure_thor_pytorch
import torch, torch.nn as nn
configure_thor_pytorch()
conv = nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(torch.randn(2, 3, 224, 224, device='cuda'))
print('conv OK:', y.shape)
"
```

In an existing shell without restarting, disable cuDNN manually before GPU work:

```bash
python3 -c "import torch; torch.backends.cudnn.enabled = False"
# or export TRT_VLA_THOR=1 and use scripts that call configure_thor_pytorch()
```


Plugin not built or path not set in `docker/thor/.env`.

### `No module named 'lerobot'`

Clone LeRobot to `LEROBOT_ROOT` and restart the container (entrypoint auto-installs on first start).

### `apt` "Release file is not valid yet" when building from `ubuntu:24.04`

Host clock may be skewed. Default build uses the local `mlperf-automotive:jsuh-aarch64-base` image to avoid apt during build. Sync NTP if you need a fresh Ubuntu base:

```bash
timedatectl status
```

### PyTorch not officially supported on DRIVE

NVIDIA recommends ONNX → TensorRT for deployment inference. This project uses PyTorch + Torch-TensorRT for **export and parity** during development — expect best-effort support on Thor.

### `Killed` during language TRT compile (host OOM)

Language TRT build peaks at **~35 GB host RSS**. With a 20 GB GPU carveout only
**~38 GB** host RAM remains — not enough without swap. `dmesg` shows
`global_oom` / `Out of memory: Killed process python3`.

Fix: add swap (see **Host swap file** above) and set `ENGINE_DIR=/gtl/pi05_edge_llm`
so engine writes do not fill root `/`.

### `OSError: [Errno 28] No space left on device` writing `.engine`

Root disk (`/`) is only ~26 GB. Language engine alone is **~9.2 GB**. Use
`ENGINE_DIR=/gtl/pi05_edge_llm` in `docker/thor/.env` (NVMe mount).

### `torch.OutOfMemoryError` / CUDA OOM with plenty of system RAM

Thor defaults to a **6 GB GPU carveout** while system RAM may be 60 GB. CUDA only
sees the carveout pool, not all of unified memory. Language TRT compile also needs
**~9 GB GPU** for the TensorRT builder on top of model weights — **14 GB carveout
is not enough**; use **20 GB** (see **GPU memory carveout** and **Pi0.5 compile
memory requirements** above).

```bash
sudo docker stop $(sudo docker ps -q) 2>/dev/null
sudo bash -c 'echo 6144 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'
./docker/thor/run.sh python3 vla/test_vla_pi05_e2e.py
```

## Rebuild / update

```bash
# After changing docker/thor/Dockerfile or requirements
./docker/thor/build.sh

# Force LeRobot reinstall inside a fresh container
./docker/thor/run.sh python3 -m pip install -e /workspace/lerobot[dataset,pi]
```

## Alternative: conda env (fallback)

If Docker GPU access remains blocked, use a host-side conda env instead:

1. Create env with Python 3.12
2. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
3. `pip install torch-tensorrt tensorrt==10.14.1.48`
4. `pip install -e ../lerobot[dataset,pi]`
5. `pip install -r docker/common/requirements.container.txt`
6. Set `EDGE_LLM_PLUGIN_SO` and run from `vla/`

See `docker/common/requirements.container.txt` for the shared pip dependency list.

## Files in this directory

```text
docker/
├── common/
│   ├── entrypoint.sh              # Auto-install LeRobot on first start
│   └── requirements.container.txt   # Pip deps (excluding torch/lerobot)
├── thor/                          # DRIVE AGX Thor (aarch64, host TRT mount)
│   ├── Dockerfile
│   ├── build.sh
│   ├── run.sh
│   ├── env.example
│   └── .env                       # Local config (not committed)
├── desktop/                       # Desktop parity (x86_64, pip TRT 10.14)
│   ├── Dockerfile
│   ├── build.sh
│   ├── run.sh
│   ├── env.example
│   └── .env                       # Local config (not committed)
├── build.sh                       # Wrapper → thor/build.sh
├── run.sh                         # Wrapper → thor/run.sh
├── build-desktop.sh               # Wrapper → desktop/build.sh
├── run-desktop.sh                 # Wrapper → desktop/run.sh
└── RUNBOOK.md                     # This document
```

## Desktop parity container (RTX 5090 / x86_64)

Use this when benchmarking against Thor with the **same** PyTorch / TensorRT /
torch-tensorrt versions. Your native conda env (torch 2.14, TRT 11) is faster
but not comparable to Thor compile behavior.

### Prerequisites

1. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (`docker run --gpus all` works).
2. Edge-LLM plugin built for **TRT 10.14 on x86_64** (not the TRT 11 plugin used for native 5090 dev).
3. LeRobot cloned and `docker/desktop/.env` paths set.

### Quick start

```bash
cd /path/to/TRT-VLA-Lowering
cp docker/desktop/env.example docker/desktop/.env
# edit TRT_VLA_ROOT, LEROBOT_ROOT, EDGE_LLM_PLUGIN_SO

chmod +x docker/desktop/build.sh docker/desktop/run.sh
./docker/desktop/build.sh
./docker/desktop/run.sh bash
```

Inside the container:

```bash
python3 -c "import torch, torch_tensorrt, tensorrt as trt; print(torch.__version__, torch_tensorrt.__version__, trt.__version__)"
python3 vla/test_vla_pi05_e2e.py
```

### Thor vs desktop parity vs native 5090

| | Thor `thor/run.sh` | Desktop `desktop/run.sh` | Native 5090 conda |
|---|---|---|---|
| Arch | aarch64 | x86_64 | x86_64 |
| PyTorch | 2.10+cu130 | 2.10+cu130 | 2.14+cu132 |
| TensorRT | 10.14 host | 10.14 pip | 11.0 |
| torch-tensorrt | 2.10 | 2.10 | 2.14 dev |
| `TRT_VLA_THOR` | `1` | `0` (set `1` to match Thor) | N/A |
| GPU memory | 20 GB carveout + swap | Full VRAM | Full VRAM |
| cuDNN | disabled | enabled (unless `TRT_VLA_THOR=1`) | enabled |

Set `TRT_VLA_THOR=1` when calling `desktop/run.sh` to test Thor's cuDNN-disabled
path inside the otherwise identical stack. On 5090 this barely changes timings;
the Thor gap vs desktop is mostly TRT version, memory, and SoC bandwidth.

### Edge-LLM plugin on x86_64

Build the TRT **10.14** plugin (same branch/cmake as Thor, x86 toolchain):

```bash
cd /path/to/tensorrt-edge-llm
git submodule update --init 3rdParty/nlohmannJson
rm -rf build-plugin-trt10 && mkdir build-plugin-trt10 && cd build-plugin-trt10
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT-10.14.x \
  -DENABLE_CUTE_DSL=OFF
make NvInfer_edgellm_plugin -j"$(nproc)"
```

Point `EDGE_LLM_PLUGIN_SO` in `docker/.env` at the resulting `.so`.
