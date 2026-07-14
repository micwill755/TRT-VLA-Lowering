#!/usr/bin/env bash
# Backward-compatible wrapper — prefer ./docker/desktop/build.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/desktop/build.sh" "$@"
