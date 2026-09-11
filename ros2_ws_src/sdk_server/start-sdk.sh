#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi
exec "$PYTHON" -m hypercorn "$@" main:app --bind 0.0.0.0:8000
