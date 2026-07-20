# Building Torch-TensorRT from source (opt) on Thor

This folder builds [`pytorch/TensorRT`](https://github.com/pytorch/TensorRT)
inside Docker using Bazel **`--compilation_mode=opt`**.

That mapping comes from `setup.py`:

| Install command | setup.py path | Bazel mode |
|-----------------|---------------|------------|
| `pip install -e .` | `develop=True` | **dbg** |
| `pip install .` | `develop=False` | **opt** ← used here |

It reuses the existing Thor VLA image (`trt-vla-thor:…`), adds Bazelisk, then
**clones the repo into a temp directory** for each build (no local checkout
required).

## Check the Thor clock first

The Docker build uses the **host's system clock**. A Thor device can boot
without a battery-backed real-time clock or before network time synchronization
has completed. If its clock is behind, APT sees signed repository metadata as
coming from the future and rejects it with errors such as:

```text
Release file ... is not valid yet
```

Check that the clock and NTP synchronization are correct:

```bash
date -u
timedatectl status
```

The expected status is:

```text
System clock synchronized: yes
NTP service: active
```

If synchronization is disabled, enable and restart it:

```bash
sudo timedatectl set-ntp true
sudo systemctl restart systemd-timesyncd
timedatectl status
```

If NTP is unavailable, temporarily set the correct UTC time manually:

```bash
sudo timedatectl set-ntp false
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS UTC"
```

Fix the host clock rather than disabling APT date validation. Repository
metadata is signed and time-bounded; those checks help prevent stale or
not-yet-valid package indexes from being accepted.

## One-time setup

```bash
cd /home/mwilliams/test/TRT-VLA-Lowering

# Thor runtime image must exist first
./docker/thor/build.sh

cp docker/torchtrt-src/env.example docker/torchtrt-src/.env
# Optional: pin TORCH_TRT_REF (default: v2.10.0 to match torch 2.10)

chmod +x docker/torchtrt-src/*.sh
./docker/torchtrt-src/build.sh
```

## Build opt Torch-TensorRT (clone → patch → install)

```bash
# Rebuild once after Dockerfile changes (clears inherited Thor entrypoint)
./docker/torchtrt-src/build.sh

./docker/torchtrt-src/run.sh /usr/local/bin/torchtrt-build-opt.sh
```

What the script does:

1. `git clone` `${TORCH_TRT_REPO}` @ `${TORCH_TRT_REF}` into `/tmp/torchtrt-src.*`
2. Stage Thor-local deps and patch the clone’s `MODULE.bazel` so:
   - `@libtorch` → installed Python `torch` package (`TORCH_PATH`)
   - `@cuda` → combined CUDA+cuBLAS tree (Thor keeps cuBLAS under `cuda/thor/`)
   - `@tensorrt_sbsa` (and aliases) → local DriveOS TensorRT
   and comment out downloaded x86 libtorch / mismatched TRT archives
   (on aarch64, `setup.py` queries `@tensorrt_sbsa`, not `@tensorrt`)
3. Set `TORCH_PATH` from the container’s installed `torch`
4. Build a wheel → Bazel **`--compilation_mode=opt`**
5. Save it under `docker/torchtrt-src/artifacts/` on the host
6. Delete the temp clone (set `TORCH_TRT_KEEP_SRC=1` to keep it)

Or interactive:

```bash
./docker/torchtrt-src/run.sh bash
/usr/local/bin/torchtrt-build-opt.sh
```

### Pin a release / TensorRT package

In `docker/torchtrt-src/.env`:

```bash
TORCH_TRT_REPO=https://github.com/pytorch/TensorRT.git
TORCH_TRT_REF=v2.10.0   # must match container torch (default: v2.10.0)
TENSORRT_PKG_ROOT=/gtl/managed/builds/TensorRT-10.14.2.2
```

## Verify

```bash
./docker/torchtrt-src/run.sh python3 -c "import torch, torch_tensorrt, tensorrt as trt; print(torch.__version__, torch_tensorrt.__version__, trt.__version__)"
```

## Notes

- **Do not** use `pip install -e .` if you want opt — that forces **dbg**.
- The built wheel persists under `docker/torchtrt-src/artifacts/` even though
  the build container uses `--rm`.
- Thor/desktop `run.sh` uses `docker/common/entrypoint.sh`, which auto-installs
  the newest `artifacts/torch_tensorrt-*.whl` when present (set
  `TRT_VLA_USE_ARTIFACT_TORCHTRT=0` to keep the image pip wheel).
- For Pi0.5 e2e testing with the prebuilt wheel, keep using `./docker/thor/run.sh`.
  Use this folder only when you intentionally need a source **opt** build.
