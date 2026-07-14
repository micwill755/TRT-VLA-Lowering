#!/usr/bin/env bash
set -euo pipefail

# Best-effort passwd entry for a nicer shell prompt. Never fail the container
# if the image already has a build-time user with the same name but wrong UID.
setup_host_user() {
  if [[ "$(id -u)" != "0" || "${TRT_VLA_SETUP_HOST_USER:-0}" != "1" ]]; then
    return 0
  fi

  local uid="${HOST_UID:?HOST_UID required}"
  local gid="${HOST_GID:?HOST_GID required}"
  local user="${HOST_USER:-user${uid}}"
  local home="${CONTAINER_HOME:-/tmp/home-${user}}"

  if ! getent group "${gid}" >/dev/null 2>&1; then
    groupadd -g "${gid}" "gid${gid}" 2>/dev/null || true
  fi

  local group_name
  group_name="$(getent group "${gid}" | cut -d: -f1)"

  # Image build may have created ${user} at UID 1000; host UID can differ.
  if getent passwd "${user}" >/dev/null 2>&1; then
    local image_uid
    image_uid="$(getent passwd "${user}" | cut -d: -f3)"
    if [[ "${image_uid}" != "${uid}" ]]; then
      usermod -l "_img_${user}" "${user}" 2>/dev/null || \
        userdel "${user}" 2>/dev/null || true
    fi
  fi

  if getent passwd "${uid}" >/dev/null 2>&1; then
    local existing_user
    existing_user="$(getent passwd "${uid}" | cut -d: -f1)"
    if [[ "${existing_user}" != "${user}" ]]; then
      usermod -l "${user}" -g "${group_name}" -d "${home}" -s /bin/bash "${existing_user}" 2>/dev/null || true
    fi
  elif ! getent passwd "${user}" >/dev/null 2>&1; then
    useradd -u "${uid}" -g "${group_name}" -m -d "${home}" -s /bin/bash "${user}" 2>/dev/null || \
      useradd -u "${uid}" -g "${gid}" -m -d "${home}" -s /bin/bash "${user}" 2>/dev/null || true
  fi

  mkdir -p "${home}"
  chown -R "${uid}:${gid}" "${home}" 2>/dev/null || true
}

run_as_container_user() {
  if [[ "$(id -u)" == "0" && "${TRT_VLA_SETUP_HOST_USER:-0}" == "1" ]]; then
    local home="${CONTAINER_HOME:-/tmp/home-${HOST_USER:-user${HOST_UID}}}"
    mkdir -p "${home}"
    chown -R "${HOST_UID}:${HOST_GID}" "${home}" 2>/dev/null || true
    # Numeric UID/GID via setpriv works without a passwd entry (corporate AD UIDs).
    setpriv --reuid="${HOST_UID}" --regid="${HOST_GID}" --clear-groups -- \
      env HOME="${home}" "$@"
  else
    "$@"
  fi
}

exec_as_container_user() {
  if [[ "$(id -u)" == "0" && "${TRT_VLA_SETUP_HOST_USER:-0}" == "1" ]]; then
    setup_host_user
    local home="${CONTAINER_HOME:-/tmp/home-${HOST_USER:-user${HOST_UID}}}"
    mkdir -p "${home}"
    chown -R "${HOST_UID}:${HOST_GID}" "${home}" 2>/dev/null || true
    exec setpriv --reuid="${HOST_UID}" --regid="${HOST_GID}" --clear-groups -- \
      env HOME="${home}" "$@"
  fi
  exec "$@"
}

setup_host_user

# Install LeRobot from the mounted workspace on first container start.
if [[ -f /workspace/lerobot/pyproject.toml || -f /workspace/lerobot/setup.py ]]; then
  if ! run_as_container_user python3 -c "import lerobot" >/dev/null 2>&1; then
    echo "[entrypoint] Installing LeRobot from /workspace/lerobot ..."
    run_as_container_user env PIP_CONSTRAINT=/dev/null python3 -m pip install "numpy>=2.0.0,<2.3.0"
    run_as_container_user python3 -m pip uninstall -y opencv opencv-python opencv-python-headless 2>/dev/null || true
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2 /usr/local/lib/python3.12/dist-packages/opencv*
    run_as_container_user env PIP_CONSTRAINT=/dev/null python3 -m pip install --no-cache-dir "opencv-python-headless>=4.9.0,<4.14.0"
    run_as_container_user env PIP_CONSTRAINT=/dev/null python3 -m pip install -e "/workspace/lerobot[dataset,pi]"
    # LeRobot's pi extra pins transformers<5.6; TRT-VLA-Lowering expects 5.13.x.
    run_as_container_user env PIP_CONSTRAINT=/dev/null python3 -m pip install --no-cache-dir "transformers==5.13.1"
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
  echo "[entrypoint]          Check the path on the host and update docker/desktop/.env." >&2
fi

# Ensure transformers 5.13.x even when LeRobot was installed in a prior container run.
if run_as_container_user python3 -c "import lerobot" >/dev/null 2>&1; then
  run_as_container_user env PIP_CONSTRAINT=/dev/null python3 -m pip install -q "transformers==5.13.1"
fi

exec_as_container_user "$@"
