#!/usr/bin/env bash
# Backward-compatible wrapper — prefer ./docker/desktop/run.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/desktop/run.sh" "$@"
