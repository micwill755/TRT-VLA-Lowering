#!/usr/bin/env bash
# Backward-compatible wrapper — prefer ./docker/thor/run.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/thor/run.sh" "$@"
