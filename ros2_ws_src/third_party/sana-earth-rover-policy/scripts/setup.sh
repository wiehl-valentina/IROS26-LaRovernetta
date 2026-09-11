#!/usr/bin/env bash
# Full setup: venv + torch + the vendored packages (genie, traversability)
# + the SDK cloned into lib/ as a black box. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

SDK_REPO_URL="${SDK_REPO_URL:-https://github.com/frodobots-org/earth-rovers-sdk-v2.git}"

echo "==> Python environment"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  # An environment is already active (e.g. `pyenv activate venv313`): use it.
  VENV="$VIRTUAL_ENV"
  echo "    using the active environment: $VENV"
else
  # Python 3.10-3.13: newer breaks the SDK deps build (pydantic-core) and
  # torch may not have wheels yet. Override with PYTHON=<interpreter>.
  if [ -z "${PYTHON:-}" ]; then
    for cand in python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v "$cand" >/dev/null 2>&1 \
         && "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,14) else 1)' 2>/dev/null; then
        PYTHON="$cand"
        break
      fi
    done
  fi
  [ -n "${PYTHON:-}" ] || { echo "No Python 3.10-3.13 found (install one or pass PYTHON=)"; exit 1; }
  "$PYTHON" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,14) else 1)' \
    || { echo "Python 3.10-3.13 required (PYTHON=$PYTHON)"; exit 1; }
  echo "    creating .venv with $PYTHON ($("$PYTHON" --version 2>&1))"
  [ -d .venv ] || "$PYTHON" -m venv .venv
  VENV="$PWD/.venv"
fi
PIP="$VENV/bin/pip"
"$PIP" install --quiet --upgrade pip

echo "==> SDK in lib/ (black box)"
mkdir -p lib
if [ ! -d lib/earth-rovers-sdk ]; then
  git clone --depth 1 "$SDK_REPO_URL" lib/earth-rovers-sdk
else
  git -C lib/earth-rovers-sdk pull --ff-only || true
fi

echo "==> dependencies (torch takes a while the first time)"
"$PIP" install torch torchvision
"$PIP" install -r lib/earth-rovers-sdk/requirements.txt
"$PIP" install -e ./genie
"$PIP" install -e "./traversability[hf]"

echo
echo "Done (environment: $VENV). Next steps:"
echo "  1. Configure the bot credentials in lib/earth-rovers-sdk/.env"
echo "  2. ./scripts/run_sdk.sh                       # terminal 1"
echo "  3. $VENV/bin/python missions/level2.py --dry-run   # terminal 2, no movement"
