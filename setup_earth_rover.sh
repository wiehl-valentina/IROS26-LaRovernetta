#!/usr/bin/env bash

# SYNOPSIS
#     Instala y configura los dos entornos del proyecto Earth Rover Challenge:
#     - genie  (planificador GeNIE / SAM-TP, necesita PyTorch + GPU)
#     - sdk    (earth-rovers-sdk, controla el robot vía FastAPI + Chrome headless)
#
# USAGE
#     Parado en la carpeta raíz "Earth Rover":
#         ./setup_earth_rover.sh
#
#     Con rutas personalizadas:
#         ./setup_earth_rover.sh --genie-path "/ruta/a/genie" --sdk-path "/ruta/a/sdk"
#
#     Para saltear la instalación de PyTorch:
#         ./setup_earth_rover.sh --skip-torch

# Frenar la ejecución si ocurre un error
set -e

# Parámetros por defecto
GENIE_PATH="./genie"
SDK_PATH="./earth-rovers-sdk"
SKIP_TORCH=false

# Parseo de argumentos
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --genie-path) GENIE_PATH="$2"; shift ;;
        --sdk-path) SDK_PATH="$2"; shift ;;
        --skip-torch) SKIP_TORCH=true ;;
        -h|--help)
            echo "Uso: $0 [--genie-path <ruta>] [--sdk-path <ruta>] [--skip-torch]"
            exit 0
            ;;
        *) echo "Parámetro desconocido: $1"; exit 1 ;;
    esac
    shift
done

# Funciones de salida con colores
write_step() { echo -e "\n\e[36m==> $1\e[0m"; }
write_warn() { echo -e "\e[33m!!  $1\e[0m"; }
write_ok()   { echo -e "\e[32mOK  $1\e[0m"; }

# ---------------------------------------------------------------------------
# 0. Chequeos previos
# ---------------------------------------------------------------------------
write_step "Chequeando rutas"

if [ ! -d "$GENIE_PATH" ]; then
    echo "No encuentro la carpeta genie en '$GENIE_PATH'. Pasala con --genie-path '/ruta/completa'."
    exit 1
fi
if [ ! -d "$SDK_PATH" ]; then
    echo "No encuentro la carpeta sdk/raíz en '$SDK_PATH'. Pasala con --sdk-path '/ruta/completa'."
    exit 1
fi

GENIE_PATH=$(realpath "$GENIE_PATH")
SDK_PATH=$(realpath "$SDK_PATH")

write_ok "genie -> $GENIE_PATH"
write_ok "sdk   -> $SDK_PATH"

write_step "Chequeando GPU NVIDIA"
if command -v nvidia-smi &> /dev/null; then
    write_ok "nvidia-smi responde, hay driver instalado"
else
    write_warn "nvidia-smi no funciona. Instalá/actualizá el driver de NVIDIA antes de seguir."
fi

# ---------------------------------------------------------------------------
# 1. Entorno de genie (sam_tp)
# ---------------------------------------------------------------------------
write_step "Configurando entorno de genie"

pushd "$GENIE_PATH" > /dev/null

if [ ! -d ".venv_genie" ]; then
    echo "Creando .venv_genie en genie..."
    python3 -m venv .venv_genie
else
    write_ok ".venv_genie ya existe, lo reuso"
fi

GENIE_VENV_PYTHON="./.venv_genie/bin/python"
"$GENIE_VENV_PYTHON" -m pip install --upgrade pip > /dev/null

if [ -f "requirements.txt" ]; then
    echo "Instalando requirements.txt de genie..."
    "$GENIE_VENV_PYTHON" -m pip install -r requirements.txt
else
    write_warn "No encontré requirements.txt en genie, sigo igual."
fi

if [ "$SKIP_TORCH" = false ]; then
    echo "Instalando PyTorch (cu126)..."
    "$GENIE_VENV_PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
else
    write_warn "Salteando instalación de PyTorch (--skip-torch)."
fi

if [ -f "pyproject.toml" ]; then
    echo "Instalando genie en modo editable (pip install -e .)..."
    "$GENIE_VENV_PYTHON" -m pip install -e .
fi

echo "Instalando dependencias del puente (requests, pyyaml)..."
"$GENIE_VENV_PYTHON" -m pip install requests pyyaml

write_step "Verificando PyTorch + GPU"
if "$GENIE_VENV_PYTHON" -c "import torch; print('torch:', torch.__version__, '| build CUDA:', torch.version.cuda); print('GPU visible:', torch.cuda.is_available())" 2>/dev/null; then
    write_ok "PyTorch verificado correctamente"
else
    write_warn "No pude verificar torch. Si falló la instalación, revisá manualmente."
fi

popd > /dev/null
write_ok "Entorno de genie listo (.venv_genie dentro de $GENIE_PATH)"

# ---------------------------------------------------------------------------
# 2. Entorno de sdk (raíz del proyecto)
# ---------------------------------------------------------------------------
write_step "Configurando entorno de sdk (raíz)"

pushd "$SDK_PATH" > /dev/null

if [ ! -d ".venv_sdk" ]; then
    echo "Creando .venv_sdk en sdk..."
    python3 -m venv .venv_sdk
else
    write_ok ".venv_sdk ya existe, lo reuso"
fi

SDK_VENV_PYTHON="./.venv_sdk/bin/python"
"$SDK_VENV_PYTHON" -m pip install --upgrade pip > /dev/null

if [ -f "requirements.txt" ]; then
    echo "Instalando requirements.txt de sdk..."
    "$SDK_VENV_PYTHON" -m pip install -r requirements.txt
else
    write_warn "No encontré requirements.txt en sdk, sigo igual."
fi

echo "Instalando google-genai..."
"$SDK_VENV_PYTHON" -m pip install google-genai

# --- .env ---
write_step "Configurando .env"
if [ -f ".env" ]; then
    write_ok ".env ya existe, no lo toco."
elif [ -f ".env.sample" ]; then
    cp ".env.sample" ".env"
    write_ok "Creé .env a partir de .env.sample. Faltan completar los valores reales:"
    echo "    - SDK_API_TOKEN"
    echo "    - BOT_SLUG"
    echo "    - CHROME_EXECUTABLE_PATH"
else
    write_warn "No encontré .env.sample, no pude generar el .env."
fi

# --- Chrome ---
write_step "Chequeando Google Chrome"
CHROME_FOUND=""
for path in "/usr/bin/google-chrome" "/usr/bin/google-chrome-stable" "/usr/bin/chromium-browser" "/usr/bin/chromium"; do
    if [ -f "$path" ]; then
        CHROME_FOUND="$path"
        break
    fi
done

if [ -n "$CHROME_FOUND" ]; then
    write_ok "Chrome/Chromium encontrado en: $CHROME_FOUND"
else
    write_warn "No encontré Chrome instalado."
fi

popd > /dev/null
write_ok "Entorno de sdk listo (.venv_sdk dentro de $SDK_PATH)"

# ---------------------------------------------------------------------------
# 3. Resumen
# ---------------------------------------------------------------------------
write_step "Resumen"
echo "genie: $GENIE_PATH/.venv_genie"
echo "sdk:   $SDK_PATH/.venv_sdk"
echo ""
echo "Para arrancar (en dos terminales separadas):"
echo "  Terminal 1:  cd \"$SDK_PATH\"   ; source .venv_sdk/bin/activate   ; hypercorn main:app --reload"
echo "  Terminal 2:  cd \"$GENIE_PATH\" ; source .venv_genie/bin/activate ; python -m genie_rover.sdk_client"