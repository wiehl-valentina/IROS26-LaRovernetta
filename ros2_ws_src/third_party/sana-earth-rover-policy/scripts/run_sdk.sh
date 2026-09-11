#!/usr/bin/env bash
# Start the Earth Rovers SDK (black box) on localhost:8000.
set -euo pipefail
cd "$(dirname "$0")/../lib/earth-rovers-sdk"

[ -f .env ] || { echo "Missing lib/earth-rovers-sdk/.env with the bot credentials."; exit 1; }

# With an activated environment (pyenv activate venv313) use its hypercorn;
# otherwise fall back to the local .venv created by setup.sh.
HYPERCORN="${VIRTUAL_ENV:-../../.venv}/bin/hypercorn"
[ -x "$HYPERCORN" ] || HYPERCORN="../../.venv/bin/hypercorn"
exec "$HYPERCORN" main:app --bind 127.0.0.1:8000
