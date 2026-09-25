#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$DIR/.env"

if [ -f "$ENV_FILE" ]; then
    # Cargar variables de entorno ignorando comentarios
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "[error] No se encontró el archivo .env"
    exit 1
fi

API_URL="${FRODOBOTS_API_URL:-https://frodobots-web-api.onrender.com/api/v1}"
CMD="${1:-help}"

case "$CMD" in
    start)
        echo "=== INICIANDO MISIÓN EN SDK LOCAL (http://localhost:8000/start-mission) ==="
        curl -s -X POST http://localhost:8000/start-mission | jq . 2>/dev/null || curl -s -X POST http://localhost:8000/start-mission
        echo ""
        ;;
    end)
        echo "=== CORTANDO MISIÓN EN SDK LOCAL (http://localhost:8000/end-mission) ==="
        curl -s -X POST http://localhost:8000/end-mission | jq . 2>/dev/null || curl -s -X POST http://localhost:8000/end-mission
        echo ""
        ;;
    release|force-end)
        echo "=== LIBERANDO BOT DIRECTAMENTE EN LA NUBE FRODOBOTS (/sdk/end_ride) ==="
        echo "  • Bot:     $BOT_SLUG"
        echo "  • Misión:  $MISSION_SLUG"
        curl -s -X POST "$API_URL/sdk/end_ride" \
            -H "Authorization: Bearer $SDK_API_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{\"bot_slug\": \"$BOT_SLUG\", \"mission_slug\": \"$MISSION_SLUG\"}" | jq . 2>/dev/null || \
        curl -s -X POST "$API_URL/sdk/end_ride" \
            -H "Authorization: Bearer $SDK_API_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{\"bot_slug\": \"$BOT_SLUG\", \"mission_slug\": \"$MISSION_SLUG\"}"
        echo ""
        ;;
    status)
        echo "=== ESTADO DEL SDK LOCAL ==="
        if curl -s -f http://localhost:8000/status >/dev/null 2>&1; then
            echo "[ok] Servidor SDK corriendo en http://localhost:8000"
            curl -s http://localhost:8000/status | jq . 2>/dev/null || curl -s http://localhost:8000/status
        else
            echo "[info] Servidor SDK local apagado o no responde en puerto 8000."
        fi
        echo ""
        echo "=== ESTADO DE MISIÓN EN NUBE FRODOBOTS ==="
        curl -s "$API_URL/sdk/missions?bot_slug=$BOT_SLUG" \
            -H "Authorization: Bearer $SDK_API_TOKEN" | jq . 2>/dev/null || \
        curl -s "$API_URL/sdk/missions?bot_slug=$BOT_SLUG" \
            -H "Authorization: Bearer $SDK_API_TOKEN"
        echo ""
        ;;
    help|*)
        echo "Uso: ./mission_control.sh [start|end|release|status]"
        echo ""
        echo "Comandos:"
        echo "  start    : Inicia la misión en el SDK local (POST http://localhost:8000/start-mission)"
        echo "  end      : Corta la misión en el SDK local (POST http://localhost:8000/end-mission)"
        echo "  release  : Fuerza el corte directo en la nube FrodoBots (útil si dice 'Bot unavailable')"
        echo "  status   : Muestra el estado del SDK local y de las misiones del bot en la nube"
        echo ""
        ;;
esac
