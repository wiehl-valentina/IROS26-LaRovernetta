#!/usr/bin/env bash
set -e

# Script de lanzamiento unificado para GeNIE Rover

MODE="${1:-indoor-semantic}"
SHIFT_ARGS="${@:2}"

case "$MODE" in
  indoor-semantic)
    echo "[LAUNCH] Iniciando Misión Semántica Indoor (VLM + BEV)..."
    python -m genie_rover.Indoor.indoor_bridge \
      --config configs/indoor_semantic_tour.yaml \
      $SHIFT_ARGS
    ;;

  indoor-cone)
    echo "[LAUNCH] Iniciando Misión de Búsqueda de Cono (Indoor)..."
    python -m genie_rover.Indoor.indoor_bridge \
      --config configs/indoor_cone_search.yaml \
      $SHIFT_ARGS
    ;;

  indoor-mapping)
    echo "[LAUNCH] Iniciando Sesión de Mapeo Indoor (Frontier)..."
    python -m genie_rover.Indoor.map_session \
      --config configs/indoor_mapping.yaml \
      $SHIFT_ARGS
    ;;

  outdoor)
    echo "[LAUNCH] Iniciando Misión Outdoor (GPS / Checkpoints)..."
    python -m genie_rover.bridge \
      --config configs/frodobot_rover.yaml \
      $SHIFT_ARGS
    ;;

  vlm-server)
    echo "[LAUNCH] Levanto servidor vLLM (Qwen2-VL-2B)..."
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2-VL-2B-Instruct \
      --port 8001 \
      --max-model-len 4096 \
      --limit-mm-per-prompt image=1
    ;;

  *)
    echo "Uso: $0 {indoor-semantic|indoor-cone|indoor-mapping|outdoor|vlm-server} [opciones adicionales]"
    exit 1
    ;;
esac