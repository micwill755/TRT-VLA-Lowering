#!/usr/bin/env bash
set -euo pipefail

# Install LeRobot from the mounted workspace on first container start.
if [[ -f /workspace/lerobot/pyproject.toml || -f /workspace/lerobot/setup.py ]]; then
  if ! python3 -c "import lerobot" >/dev/null 2>&1; then
    echo "[entrypoint] Installing LeRobot from /workspace/lerobot ..."
    PIP_CONSTRAINT=/dev/null python3 -m pip install "numpy>=2.0.0,<2.3.0"
    python3 -m pip uninstall -y opencv opencv-python opencv-python-headless 2>/dev/null || true
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2 /usr/local/lib/python3.12/dist-packages/opencv*
    PIP_CONSTRAINT=/dev/null python3 -m pip install --no-cache-dir "opencv-python-headless>=4.9.0,<4.14.0"
    PIP_CONSTRAINT=/dev/null python3 -m pip install -e "/workspace/lerobot[dataset,pi]"
    # LeRobot's pi extra pins transformers<5.6; TRT-VLA-Lowering expects 5.13.x.
    PIP_CONSTRAINT=/dev/null python3 -m pip install --no-cache-dir "transformers==5.13.1"
  fi
elif [[ -d /workspace/lerobot ]]; then
  echo "[entrypoint] WARNING: /workspace/lerobot is mounted but is not a LeRobot checkout." >&2
else
  echo "[entrypoint] WARNING: /workspace/lerobot not mounted. LeRobot-dependent scripts will fail." >&2
fi

if [[ -z "${EDGE_LLM_PLUGIN_SO:-}" ]]; then
  echo "[entrypoint] WARNING: EDGE_LLM_PLUGIN_SO is unset. Export scripts will fail at load_plugins_for_trt()." >&2
elif [[ ! -f "${EDGE_LLM_PLUGIN_SO}" ]]; then
  echo "[entrypoint] WARNING: EDGE_LLM_PLUGIN_SO='${EDGE_LLM_PLUGIN_SO}' does not exist." >&2
fi

# Ensure transformers 5.13.x even when LeRobot was installed in a prior container run.
if python3 -c "import lerobot" >/dev/null 2>&1; then
  PIP_CONSTRAINT=/dev/null python3 -m pip install -q "transformers==5.13.1"
fi

exec "$@"
