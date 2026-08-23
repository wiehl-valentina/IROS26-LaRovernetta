#!/usr/bin/env bash
# rover_launch.sh
#
# Launcher unico para el proyecto: sabe donde vive cada componente, que
# entorno (.venv / sistema+ROS2) activar, y te deja pasar configs por flags
# en vez de tener que acordarte de rutas y "source .venv/bin/activate" cada
# vez.
#
# Uso:
#   ./rover_launch.sh <comando> [opciones]
#
# Comandos:
#   sdk                          Levanta el SDK (hypercorn main:app)
#   genie-bridge [--go] [--config PATH] [--max-seconds N] [--start-mission]
#                                 Corre genie_rover.bridge (dry-run por default)
#   sdk-client                    Prueba de conexion SDK <-> genie (sin mover)
#   perception --image PATH [--config PATH]
#                                 Prueba de percepcion SAM-TP sobre una imagen
#   mapping-ros2 [--db PATH]     Levanta rtabmap_mapping.launch.py (ROS2)
#   map-session [--go] [--config PATH] [--max-seconds N] [--map-out PATH]
#                                 Corre genie_rover.Indoor.map_session
#   indoor-bridge [--go] [--config PATH] [--max-seconds N] [--debug-dir PATH]
#                                 Corre genie_rover.Indoor.indoor_bridge (busca cono)
#   traversability <predict|live|drive|mission> [opciones propias]
#                                 Corre rover_traversability.demo <nivel>
#   ros2-check                    Test basico ROS2 (talker), Ctrl+C para cortar
#   all                            Levanta sdk + mapping-ros2 + genie-bridge
#                                 juntos en paneles de tmux (requiere tmux)
#   doctor                        Chequea que cada pieza este instalada/OK
#
# Configuracion de rutas: editar el bloque "RUTAS" mas abajo una sola vez,
# o exportar las variables antes de llamar al script (REPO, SDK_DIR, etc.)
#
# Nota importante: este script se encarga de desactivar conda antes de
# activar cualquier .venv, porque conda pisa el "python3" del sistema y
# rompe la creacion/uso de venvs (problema que ya nos paso una vez).

set -uo pipefail

# --------------------------------------------------------------- RUTAS ----
REPO="${REPO:-$HOME/IROS26-LaRovernetta}"
SDK_DIR="${SDK_DIR:-$REPO/earth-rovers-sdk}"
SDK_VENV="${SDK_VENV:-$REPO/.venv}"
GENIE_DIR="${GENIE_DIR:-$REPO/genie}"
GENIE_VENV="${GENIE_VENV:-$GENIE_DIR/.venv}"
MAPPING_DIR="${MAPPING_DIR:-$SDK_DIR/examples/ros2/mapping}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
MAPS_DIR="${MAPS_DIR:-$HOME/maps}"
# ----------------------------------------------------------------------------

c_red()   { echo -e "\033[1;31m$*\033[0m"; }
c_green() { echo -e "\033[1;32m$*\033[0m"; }
c_yellow(){ echo -e "\033[1;33m$*\033[0m"; }
c_blue()  { echo -e "\033[1;34m$*\033[0m"; }

die() { c_red "ERROR: $*"; exit 1; }

# Conda pisa python3/pip del sistema y rompe todo lo que dependa de rclpy
# o de los venvs armados sobre el python del sistema. Lo desactivamos
# siempre antes de tocar cualquier venv o ROS2, sin importar si el usuario
# lo tiene activado o no.
neutralize_conda() {
    if command -v conda >/dev/null 2>&1; then
        conda deactivate >/dev/null 2>&1 || true
        conda config --set auto_activate_base false >/dev/null 2>&1 || true
    fi
}

need_dir() {
    [ -d "$1" ] || die "No encuentro la carpeta: $1 (revisa las RUTAS al principio del script, o exporta la variable correspondiente)"
}

need_file() {
    [ -f "$1" ] || die "No encuentro el archivo: $1"
}

activate_venv() {
    local venv_dir="$1"
    need_dir "$venv_dir"
    need_file "$venv_dir/bin/activate"
    # shellcheck disable=SC1090
    source "$venv_dir/bin/activate"
}

source_ros2() {
    need_file "$ROS_SETUP"
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
}

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

# ------------------------------------------------------------- comandos ---

cmd_sdk() {
    neutralize_conda
    need_dir "$SDK_DIR"
    activate_venv "$SDK_VENV"
    cd "$SDK_DIR" || exit 1
    need_file "main.py"
    c_blue "==> SDK: hypercorn main:app  (dashboard en http://localhost:8000)"
    exec hypercorn main:app
}

cmd_genie_bridge() {
    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1

    local config="configs/frodobot_rover.yaml"
    local go=0
    local max_seconds=""
    local start_mission=0

    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --go) go=1; shift ;;
            --max-seconds) max_seconds="$2"; shift 2 ;;
            --start-mission) start_mission=1; shift ;;
            *) die "Opcion desconocida para genie-bridge: $1" ;;
        esac
    done

    need_file "$config"

    local args=(--config "$config")
    if [ "$go" -eq 1 ]; then
        args+=(--go)
        [ -n "$max_seconds" ] && args+=(--max-seconds "$max_seconds")
        [ "$start_mission" -eq 1 ] && args+=(--start-mission)
        c_yellow "==> genie_rover.bridge en MODO REAL — el rover se va a mover"
    else
        c_blue "==> genie_rover.bridge en modo simulacro (dry-run, no mueve el rover)"
    fi

    exec python -m genie_rover.bridge "${args[@]}"
}

cmd_sdk_client() {
    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1
    c_blue "==> genie_rover.sdk_client — prueba de conexion, no mueve el rover"
    exec python -m genie_rover.sdk_client
}

cmd_perception() {
    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1

    local image=""
    local config="configs/frodobot_rover.yaml"
    local out="debug/"

    while [ $# -gt 0 ]; do
        case "$1" in
            --image) image="$2"; shift 2 ;;
            --config) config="$2"; shift 2 ;;
            --out) out="$2"; shift 2 ;;
            *) die "Opcion desconocida para perception: $1" ;;
        esac
    done

    [ -n "$image" ] || die "Falta --image <ruta_a_foto.jpg>"
    need_file "$image"
    need_file "$config"

    c_blue "==> genie_rover.perception sobre $image"
    exec python -m genie_rover.perception --config "$config" --image "$image" --out "$out"
}

cmd_mapping_ros2() {
    neutralize_conda
    need_dir "$MAPPING_DIR"
    source_ros2
    cd "$MAPPING_DIR" || exit 1
    need_file "rtabmap_mapping.launch.py"

    local db="$MAPS_DIR/sesion_$(date +%s 2>/dev/null || echo manual).db"
    while [ $# -gt 0 ]; do
        case "$1" in
            --db) db="$2"; shift 2 ;;
            *) die "Opcion desconocida para mapping-ros2: $1" ;;
        esac
    done

    mkdir -p "$MAPS_DIR"
    c_blue "==> RTAB-Map: database_path=$db"
    exec ros2 launch rtabmap_mapping.launch.py database_path:="$db"
}

cmd_map_session() {
    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1

    local config="configs/indoor_mapping.yaml"
    local go=0
    local max_seconds=""
    local map_out=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --go) go=1; shift ;;
            --max-seconds) max_seconds="$2"; shift 2 ;;
            --map-out) map_out="$2"; shift 2 ;;
            *) die "Opcion desconocida para map-session: $1" ;;
        esac
    done

    need_file "$config"

    local args=(--config "$config")
    if [ "$go" -eq 1 ]; then
        args+=(--go)
        [ -n "$max_seconds" ] && args+=(--max-seconds "$max_seconds")
        [ -n "$map_out" ] && args+=(--map-out "$map_out")
        c_yellow "==> map_session en MODO REAL — el rover se va a mover y explorar solo"
    else
        c_blue "==> map_session en modo simulacro (dry-run)"
    fi

    exec python -m genie_rover.Indoor.map_session "${args[@]}"
}

cmd_indoor_bridge() {
    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1

    local config="configs/indoor_cone_search.yaml"
    local go=0
    local max_seconds=""
    local debug_dir=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --go) go=1; shift ;;
            --max-seconds) max_seconds="$2"; shift 2 ;;
            --debug-dir) debug_dir="$2"; shift 2 ;;
            *) die "Opcion desconocida para indoor-bridge: $1" ;;
        esac
    done

    need_file "$config"

    # Nota: el modulo real vive en genie_rover/Indoor/indoor_bridge.py. El
    # docstring del propio archivo usa "genie_rover.indoor_bridge" en el
    # ejemplo, pero la ruta de paquete real (misma que map_session, que ya
    # confirmamos que funciona) es genie_rover.Indoor.indoor_bridge. Si esto
    # da ModuleNotFoundError, probar el modulo sin ".Indoor" como alternativa.
    local args=(--config "$config")
    if [ "$go" -eq 1 ]; then
        args+=(--go)
        [ -n "$max_seconds" ] && args+=(--max-seconds "$max_seconds")
        [ -n "$debug_dir" ] && args+=(--debug-dir "$debug_dir")
        c_yellow "==> indoor_bridge en MODO REAL — el rover se va a mover buscando el cono"
    else
        c_blue "==> indoor_bridge en modo simulacro (dry-run)"
    fi

    exec python -m genie_rover.Indoor.indoor_bridge "${args[@]}"
}

cmd_traversability() {
    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$REPO" || exit 1

    local level="${1:-}"
    [ -n "$level" ] || die "Falta el nivel: predict | live | drive | mission"
    shift || true

    case "$level" in
        predict)
            local image="" out="overlay.png"
            while [ $# -gt 0 ]; do
                case "$1" in
                    --image) image="$2"; shift 2 ;;
                    --out) out="$2"; shift 2 ;;
                    *) die "Opcion desconocida: $1" ;;
                esac
            done
            [ -n "$image" ] || die "Falta --image <ruta_a_foto.jpg>"
            need_file "$image"
            c_blue "==> rover_traversability predict sobre $image"
            exec python -m rover_traversability.demo predict "$image" --out "$out"
            ;;
        live)
            local save_dir="trav_out"
            while [ $# -gt 0 ]; do
                case "$1" in
                    --save-dir) save_dir="$2"; shift 2 ;;
                    *) die "Opcion desconocida: $1" ;;
                esac
            done
            c_blue "==> rover_traversability live — NO mueve el rover, guarda overlays en $save_dir"
            exec python -m rover_traversability.demo live --save-dir "$save_dir"
            ;;
        drive)
            c_yellow "==> rover_traversability drive — MODO REAL, esquiva obstaculos sin meta GPS"
            exec python -m rover_traversability.demo drive --yes-i-want-the-rover-to-move
            ;;
        mission)
            c_yellow "==> rover_traversability mission — MODO REAL, navega checkpoints GPS"
            exec python -m rover_traversability.demo mission --start-mission --yes-i-want-the-rover-to-move
            ;;
        *) die "Nivel desconocido: $level (usar predict | live | drive | mission)" ;;
    esac
}

cmd_ros2_check() {
    neutralize_conda
    source_ros2
    c_blue "==> talker de prueba — Ctrl+C para cortar. Abri OTRA terminal y corre:"
    c_blue "      source $ROS_SETUP && ros2 run demo_nodes_py listener"
    exec ros2 run demo_nodes_cpp talker
}

cmd_doctor() {
    echo "Chequeo rapido de instalacion:"
    echo

    command -v conda >/dev/null 2>&1 && c_yellow "conda: instalado (recorda que este script lo desactiva solo antes de cada comando)"

    if [ -f "$SDK_VENV/bin/activate" ]; then c_green "OK  venv del SDK: $SDK_VENV"; else c_red "FALTA venv del SDK: $SDK_VENV"; fi
    if [ -f "$GENIE_VENV/bin/activate" ]; then c_green "OK  venv de genie: $GENIE_VENV"; else c_red "FALTA venv de genie: $GENIE_VENV"; fi
    if [ -f "$SDK_DIR/main.py" ]; then c_green "OK  SDK main.py encontrado"; else c_red "FALTA $SDK_DIR/main.py"; fi
    if [ -f "$MAPPING_DIR/rtabmap_mapping.launch.py" ]; then c_green "OK  launch de mapeo encontrado"; else c_red "FALTA $MAPPING_DIR/rtabmap_mapping.launch.py"; fi
    if [ -f "$ROS_SETUP" ]; then c_green "OK  ROS2 Humble instalado"; else c_red "FALTA $ROS_SETUP (ROS2 no instalado)"; fi

    local ckpt="$GENIE_DIR/sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt"
    if [ -f "$ckpt" ]; then
        c_green "OK  checkpoint_2.pt encontrado ($(du -h "$ckpt" | cut -f1))"
    else
        c_red "FALTA checkpoint_2.pt en $ckpt"
    fi

    echo
    which google-chrome-stable >/dev/null 2>&1 && c_green "OK  google-chrome-stable: $(which google-chrome-stable)" || c_red "FALTA google-chrome-stable"

    if [ -f "$GENIE_DIR/configs/indoor_cone_search.yaml" ]; then c_green "OK  config indoor_cone_search.yaml"; else c_yellow "falta configs/indoor_cone_search.yaml (necesario para indoor-bridge)"; fi
    if [ -f "$GENIE_VENV/lib/python3.10/site-packages/rover_traversability" ] || [ -d "$GENIE_VENV/lib/python3.10/site-packages/rover_traversability" ]; then
        c_green "OK  rover_traversability instalado en el venv de genie"
    else
        c_yellow "rover_traversability no parece instalado (necesario para 'traversability') — pip install -e './traversability[hf]' parado en $REPO con el venv de genie activo"
    fi

    echo
    echo "Rutas configuradas actualmente:"
    echo "  REPO=$REPO"
    echo "  SDK_DIR=$SDK_DIR"
    echo "  GENIE_DIR=$GENIE_DIR"
    echo "  MAPPING_DIR=$MAPPING_DIR"
}

cmd_all() {
    command -v tmux >/dev/null 2>&1 || die "Necesitas tmux para 'all' (sudo apt install -y tmux) — o corre cada comando en su propia terminal por separado."

    local session="rover"
    tmux kill-session -t "$session" 2>/dev/null || true

    tmux new-session -d -s "$session" -n main
    tmux send-keys -t "$session:main" "$0 sdk" C-m
    tmux split-window -h -t "$session:main"
    tmux send-keys -t "$session:main" "$0 mapping-ros2" C-m
    tmux split-window -v -t "$session:main"
    tmux send-keys -t "$session:main" "echo 'Esperando que el SDK levante...'; sleep 5; $0 genie-bridge" C-m

    c_green "Sesion tmux '$session' armada con 3 paneles: sdk / mapping-ros2 / genie-bridge"
    c_blue  "Attach con:  tmux attach -t $session"
    c_blue  "Cambiar de panel: Ctrl+B luego flechas. Salir sin matar: Ctrl+B luego D."
    tmux attach -t "$session"
}

# ------------------------------------------------------------------ main --

[ $# -ge 1 ] || usage
cmd="$1"; shift

case "$cmd" in
    sdk)          cmd_sdk "$@" ;;
    genie-bridge) cmd_genie_bridge "$@" ;;
    sdk-client)   cmd_sdk_client "$@" ;;
    perception)   cmd_perception "$@" ;;
    mapping-ros2) cmd_mapping_ros2 "$@" ;;
    map-session)  cmd_map_session "$@" ;;
    indoor-bridge) cmd_indoor_bridge "$@" ;;
    traversability) cmd_traversability "$@" ;;
    ros2-check)   cmd_ros2_check "$@" ;;
    all)          cmd_all "$@" ;;
    doctor)       cmd_doctor "$@" ;;
    -h|--help|help) usage ;;
    *) die "Comando desconocido: $cmd (corre '$0 help' para ver la lista)" ;;
esac
