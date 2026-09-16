#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/earth-rovers-sdk"

PYTHON="/home/marian/miniconda3/envs/erc_sdk/bin/python3"
HYPERCORN="/home/marian/miniconda3/envs/erc_sdk/bin/hypercorn"

echo "[start_sdk] Iniciando el servidor del SDK localmente en puerto 8000..."
exec "$HYPERCORN" main:app --reload
