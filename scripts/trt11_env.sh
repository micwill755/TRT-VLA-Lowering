#!/usr/bin/env bash
# Source before GROOT / Edge-LLM compile or runtime:
#   source Test/scripts/trt11_env.sh
#
# Do NOT run as ./trt11_env.sh — that starts a subshell and exits.
# Aligns on TensorRT 11.0.0.114 (pip tensorrt-cu13 + build-plugin-trt11).

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export EDGE_LLM_PLUGIN_SO="${EDGE_LLM_PLUGIN_SO:-${WORKSPACE_ROOT}/gitlab/TensorRT-Edge-LLM/build-plugin-trt11/libNvInfer_edgellm_plugin.so}"
export EDGELLM_TRT_PLUGIN_SO="${EDGELLM_TRT_PLUGIN_SO:-${EDGE_LLM_PLUGIN_SO}}"

echo "EDGE_LLM_PLUGIN_SO=${EDGE_LLM_PLUGIN_SO}"

if command -v python3 >/dev/null 2>&1; then
  python3 -c "import tensorrt as tr; print('tensorrt', tr.__version__)" || true
fi
