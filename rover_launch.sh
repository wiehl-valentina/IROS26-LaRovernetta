#!/usr/bin/env bash
# rover_launch.sh — launcher unico del proyecto La Rovernetta (v2)
#
# Sabe donde vive cada componente, que entorno activar (.venv / sistema+ROS2)
# y te deja pasar configs por flags en vez de acordarte rutas y "source
# .venv/bin/activate" cada vez.
#
# Novedades v2:
#   * Las rutas se autodetectan desde la ubicacion de ESTE archivo (ya no
#     asume ~/IROS26-LaRovernetta). Se pueden pisar con variables de entorno.
#   * ROS2 se autodetecta (humble / jazzy / iron / foxy).
#   * El stack ROS2 de mapeo se busca en el SDK y, si no esta, en la copia
#     canonica del repo (Indoor_Instalacion_SDK_SLAM/ros2). "sync-ros2"
#     copia una en la otra.
#   * "mapping-ros2" deriva intrinsecos, distorsion, odometria y extrinsecos
#     de camara del MISMO yaml de genie que usa el bridge, para que la pose
#     de RTAB-Map y la de genie_rover no diverjan.
#   * Todos los flags reales de cada modulo estan expuestos, y cada comando
#     tiene su propio --help.
#
# Uso:
#   ./rover_launch.sh <comando> [opciones]
#   ./rover_launch.sh <comando> --help
#   ./rover_launch.sh help
#
# Nota: este script desactiva conda antes de activar cualquier .venv, porque
# conda pisa el "python3" del sistema y rompe la creacion/uso de venvs y de
# rclpy.

set -uo pipefail

# --------------------------------------------------------------- RUTAS ----
# Todo se deriva de donde vive este archivo. Cualquiera se puede pisar
# exportando la variable antes de llamar al script.
_SELF="${BASH_SOURCE[0]}"
while [ -L "$_SELF" ]; do _SELF="$(readlink "$_SELF")"; done
SCRIPT_DIR="$(cd "$(dirname "$_SELF")" && pwd)"

REPO="${REPO:-$SCRIPT_DIR}"
SDK_DIR="${SDK_DIR:-$REPO/earth-rovers-sdk}"
GENIE_DIR="${GENIE_DIR:-$REPO/genie}"
TRAV_DIR="${TRAV_DIR:-$REPO/traversability}"
INDOOR_DIR="${INDOOR_DIR:-$REPO/Indoor_Instalacion_SDK_SLAM}"
MAPS_DIR="${MAPS_DIR:-$HOME/maps}"
DASHBOARD="${DASHBOARD:-$REPO/dashboard_server.py}"

c_red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
c_green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
c_yellow(){ printf '\033[1;33m%s\033[0m\n' "$*"; }
c_blue()  { printf '\033[1;34m%s\033[0m\n' "$*"; }

die() { c_red "ERROR: $*"; exit 1; }
warn(){ c_yellow "AVISO: $*"; }

# Primer candidato que exista (archivo o carpeta). Vacio si ninguno.
first_existing() {
    local c
    for c in "$@"; do [ -e "$c" ] && { printf '%s' "$c"; return 0; }; done
    return 1
}

# --- venvs -----------------------------------------------------------------
# Historicamente el venv del SDK estuvo en la raiz del repo y el de genie
# adentro de genie/. Aceptamos ambos ordenes por si el repo se reorganiza.
SDK_VENV="${SDK_VENV:-$(first_existing "$REPO/.venv" "$SDK_DIR/.venv" || echo "$REPO/.venv")}"
GENIE_VENV="${GENIE_VENV:-$(first_existing "$GENIE_DIR/.venv" "$REPO/.venv" || echo "$GENIE_DIR/.venv")}"

# --- ROS2 ------------------------------------------------------------------
# Respeta ROS_SETUP explicito; si no, respeta $ROS_DISTRO; si no, busca.
detect_ros_setup() {
    [ -n "${ROS_SETUP:-}" ] && { printf '%s' "$ROS_SETUP"; return; }
    if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
        printf '/opt/ros/%s/setup.bash' "$ROS_DISTRO"; return
    fi
    local d
    for d in jazzy humble iron foxy rolling; do
        [ -f "/opt/ros/$d/setup.bash" ] && { printf '/opt/ros/%s/setup.bash' "$d"; return; }
    done
    printf '/opt/ros/humble/setup.bash'   # ultimo recurso: mensaje de error claro
}
ROS_SETUP="$(detect_ros_setup)"

# --- stack ROS2 de mapeo ---------------------------------------------------
# Canonico en el repo: Indoor_Instalacion_SDK_SLAM/ros2
# Instalado (lo que el launch espera): earth-rovers-sdk/examples/ros2
ROS2_SRC_DIR="${ROS2_SRC_DIR:-$INDOOR_DIR/ros2}"
ROS2_SDK_DIR="${ROS2_SDK_DIR:-$SDK_DIR/examples/ros2}"
resolve_ros2_dir() {
    local d
    d="$(first_existing "$ROS2_SDK_DIR/mapping/rtabmap_mapping.launch.py" \
                        "$ROS2_SRC_DIR/mapping/rtabmap_mapping.launch.py")" || return 1
    printf '%s' "$(cd "$(dirname "$(dirname "$d")")" && pwd)"
}
ROS2_DIR="${ROS2_DIR:-$(resolve_ros2_dir || echo "$ROS2_SRC_DIR")}"
MAPPING_DIR="${MAPPING_DIR:-$ROS2_DIR/mapping}"

# Config por defecto de cada comando (pisables por flag)
CFG_OUTDOOR="${CFG_OUTDOOR:-configs/frodobot_rover.yaml}"
CFG_INDOOR="${CFG_INDOOR:-configs/indoor_cone_search.yaml}"
CFG_MAPPING="${CFG_MAPPING:-configs/indoor_mapping.yaml}"
# ----------------------------------------------------------------------------

neutralize_conda() {
    if command -v conda >/dev/null 2>&1; then
        conda deactivate >/dev/null 2>&1 || true
        conda config --set auto_activate_base false >/dev/null 2>&1 || true
    fi
}

need_dir()  { [ -d "$1" ] || die "No encuentro la carpeta: $1 (exporta la variable correspondiente o revisa el bloque RUTAS)"; }
need_file() { [ -f "$1" ] || die "No encuentro el archivo: $1"; }

activate_venv() {
    local venv_dir="$1"
    [ -f "$venv_dir/bin/activate" ] || die "No encuentro el venv: $venv_dir (falta bin/activate). Crealo con: python3 -m venv '$venv_dir'"
    # shellcheck disable=SC1090
    source "$venv_dir/bin/activate"
}

source_ros2() {
    [ -f "$ROS_SETUP" ] || die "No encuentro ROS2 en $ROS_SETUP. Instala ros-<distro>-desktop o exporta ROS_SETUP=/opt/ros/<distro>/setup.bash"
    export AMENT_TRACE_SETUP_FILES=0   # bug conocido de ament con 'set -u'
    set +u
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
    set -u
}

# expanduser + mkdir del directorio padre (para --db, --map-out, --debug-dir)
expand_path() {
    local p="$1"
    # shellcheck disable=SC2088  # el patron "~/"* es un tilde LITERAL, a proposito
    case "$p" in
        "~") p="$HOME" ;;
        "~/"*) p="$HOME/${p#\~/}" ;;
    esac
    printf '%s' "$p"
}
ensure_parent_dir() { local p; p="$(expand_path "$1")"; mkdir -p "$(dirname "$p")" 2>/dev/null || true; printf '%s' "$p"; }

# Ayuda por comando: imprime el texto y sale 0.
show_help() { printf '%s\n' "$1"; exit 0; }

# Detecta "--help/-h" en cualquier posicion de los argumentos de un comando.
wants_help() {
    local a
    for a in "$@"; do [ "$a" = "--help" ] || [ "$a" = "-h" ] && return 0; done
    return 1
}

# ------------------------------------------------------------- comandos ---

HELP_SDK='sdk — levanta el Earth Rovers SDK (hypercorn main:app, dashboard en :8000)

  ./rover_launch.sh sdk [--port N] [--reload]'
cmd_sdk() {
    wants_help "$@" && show_help "$HELP_SDK"
    local port="8000" reload=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            --reload) reload=1; shift ;;
            *) die "Opcion desconocida para sdk: $1" ;;
        esac
    done
    neutralize_conda
    need_dir "$SDK_DIR"
    activate_venv "$SDK_VENV"
    cd "$SDK_DIR" || exit 1
    need_file "main.py"
    [ -f ".env" ] || warn "No hay .env en $SDK_DIR — el SDK va a arrancar sin token/bot_slug (copiá .env.sample)."
    local args=(main:app --bind "127.0.0.1:$port")
    [ "$reload" -eq 1 ] && args+=(--reload)
    c_blue "==> SDK: hypercorn ${args[*]}  (dashboard en http://localhost:$port)"
    exec hypercorn "${args[@]}"
}

HELP_DASH='dashboard — levanta el panel web local que maneja este mismo launcher

  ./rover_launch.sh dashboard [--port 8765] [--host 127.0.0.1]

Abrí http://localhost:8765 en el navegador (desde Windows funciona directo
con WSL2). Usa FastAPI+uvicorn si estan instalados (logs por WebSocket) y
si no cae solo a la version de libreria estandar (polling).'
cmd_dashboard() {
    wants_help "$@" && show_help "$HELP_DASH"
    neutralize_conda
    need_file "$DASHBOARD"
    # El dashboard solo necesita stdlib; si hay un venv con fastapi, mejor.
    local py="python3"
    if [ -x "$SDK_VENV/bin/python" ] && "$SDK_VENV/bin/python" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
        py="$SDK_VENV/bin/python"
    elif [ -x "$GENIE_VENV/bin/python" ] && "$GENIE_VENV/bin/python" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
        py="$GENIE_VENV/bin/python"
    fi
    cd "$REPO" || exit 1
    c_blue "==> dashboard ($py) — http://localhost:8765"
    exec "$py" "$DASHBOARD" --script "$REPO/rover_launch.sh" "$@"
}

HELP_GENIE='genie-bridge — bridge outdoor (GPS + SAM-TP), genie_rover.bridge

  ./rover_launch.sh genie-bridge [--config PATH] [--debug-dir DIR]
                                 [--go] [--max-seconds N] [--start-mission]

Sin --go corre en simulacro (no manda comandos al rover).'
cmd_genie_bridge() {
    wants_help "$@" && show_help "$HELP_GENIE"
    local config="$CFG_OUTDOOR" go=0 max_seconds="" start_mission=0 debug_dir=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --go) go=1; shift ;;
            --max-seconds) max_seconds="$2"; shift 2 ;;
            --start-mission) start_mission=1; shift ;;
            --debug-dir) debug_dir="$2"; shift 2 ;;
            *) die "Opcion desconocida para genie-bridge: $1 (--help para la lista)" ;;
        esac
    done

    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1
    need_file "$config"

    local args=(--config "$config")
    # --max-seconds y --debug-dir tambien sirven en simulacro (v1 los ataba a --go)
    [ -n "$max_seconds" ] && args+=(--max-seconds "$max_seconds")
    [ -n "$debug_dir" ] && args+=(--debug-dir "$(ensure_parent_dir "$debug_dir")")
    if [ "$go" -eq 1 ]; then
        args+=(--go)
        [ "$start_mission" -eq 1 ] && args+=(--start-mission)
        c_yellow "==> genie_rover.bridge en MODO REAL — el rover se va a mover"
    else
        [ "$start_mission" -eq 1 ] && warn "--start-mission sin --go no hace nada (el simulacro no arranca la mision)"
        c_blue "==> genie_rover.bridge en simulacro (dry-run, no mueve el rover)"
    fi
    exec python -m genie_rover.bridge "${args[@]}"
}

HELP_INDOOR='indoor-bridge — tour de checkpoints indoor sin GPS (busca conos y los fotografia)

  ./rover_launch.sh indoor-bridge [--config PATH] [--debug-dir DIR]
                                  [--search-mode wander|frontier|waypoints]
                                  [--waypoints-path PATH]
                                  [--go] [--max-seconds N]

--search-mode pisa mission.search_mode del config sin editarlo.
Modulo real: genie_rover.Indoor.indoor_bridge'
cmd_indoor_bridge() {
    wants_help "$@" && show_help "$HELP_INDOOR"
    local config="$CFG_INDOOR" go=0 max_seconds="" debug_dir="" search_mode="" waypoints_path=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --go) go=1; shift ;;
            --max-seconds) max_seconds="$2"; shift 2 ;;
            --debug-dir) debug_dir="$2"; shift 2 ;;
            --search-mode) search_mode="$2"; shift 2 ;;
            --waypoints-path) waypoints_path="$2"; shift 2 ;;
            *) die "Opcion desconocida para indoor-bridge: $1 (--help para la lista)" ;;
        esac
    done

    case "$search_mode" in
        ""|wander|frontier|waypoints) ;;
        *) die "--search-mode invalido: $search_mode (usar wander | frontier | waypoints)" ;;
    esac
    if [ "$search_mode" = "waypoints" ] && [ -z "$waypoints_path" ]; then
        warn "--search-mode waypoints sin --waypoints-path: se usa mission.waypoints_path del config (si es null, el modulo va a fallar)"
    fi

    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1
    need_file "$config"
    [ -n "$waypoints_path" ] && need_file "$waypoints_path"

    local args=(--config "$config")
    [ -n "$search_mode" ] && args+=(--search-mode "$search_mode")
    [ -n "$waypoints_path" ] && args+=(--waypoints-path "$waypoints_path")
    [ -n "$max_seconds" ] && args+=(--max-seconds "$max_seconds")
    [ -n "$debug_dir" ] && args+=(--debug-dir "$(ensure_parent_dir "$debug_dir")")
    if [ "$go" -eq 1 ]; then
        args+=(--go)
        c_yellow "==> indoor_bridge en MODO REAL — el rover se mueve buscando conos"
    else
        c_blue "==> indoor_bridge en simulacro (dry-run)"
    fi
    [ -n "$search_mode" ] && c_blue "    modo de busqueda: $search_mode"
    exec python -m genie_rover.Indoor.indoor_bridge "${args[@]}"
}

HELP_MAPSESS='map-session — sesion de mapeo por frontera + export a maps/*.yaml+.pgm

  ./rover_launch.sh map-session [--config PATH] [--map-out PREFIJO]
                                [--debug-dir DIR] [--export-every-s N]
                                [--go] [--max-seconds N]

Si en el config mapping.rtabmap_correction.enabled esta en true, levanta
antes "./rover_launch.sh mapping-ros2" en otra terminal.
Modulo real: genie_rover.Indoor.map_session'
cmd_map_session() {
    wants_help "$@" && show_help "$HELP_MAPSESS"
    local config="$CFG_MAPPING" go=0 max_seconds="" map_out="" debug_dir="" export_every_s=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --go) go=1; shift ;;
            --max-seconds) max_seconds="$2"; shift 2 ;;
            --map-out) map_out="$2"; shift 2 ;;
            --debug-dir) debug_dir="$2"; shift 2 ;;
            --export-every-s) export_every_s="$2"; shift 2 ;;
            *) die "Opcion desconocida para map-session: $1 (--help para la lista)" ;;
        esac
    done

    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1
    need_file "$config"

    local args=(--config "$config")
    [ -n "$debug_dir" ] && args+=(--debug-dir "$(ensure_parent_dir "$debug_dir")")
    [ -n "$export_every_s" ] && args+=(--export-every-s "$export_every_s")
    [ -n "$max_seconds" ] && args+=(--max-seconds "$max_seconds")
    # el export tambien es util en simulacro (v1 lo ataba a --go)
    [ -n "$map_out" ] && args+=(--map-out "$(ensure_parent_dir "$map_out")")
    if [ "$go" -eq 1 ]; then
        args+=(--go)
        c_yellow "==> map_session en MODO REAL — el rover explora solo"
        if grep -qE '^\s*enabled:\s*true' "$config" 2>/dev/null; then
            c_yellow "    (el config parece pedir correccion RTAB-Map: levantá './rover_launch.sh mapping-ros2' en otra terminal)"
        fi
    else
        c_blue "==> map_session en simulacro (dry-run)"
    fi
    exec python -m genie_rover.Indoor.map_session "${args[@]}"
}

HELP_MAPROS='mapping-ros2 — RTAB-Map + bridge ROS2 (correccion de pose por cierre de bucles)

  ./rover_launch.sh mapping-ros2 [--db PATH] [--config PATH] [--sdk-url URL]
                                 [--feed-fps N] [--raw]

Por defecto deriva intrinsecos, distorsion, radio de rueda, ancho de trocha,
signo de rotacion y extrinsecos de camara del yaml de genie (--config, por
defecto genie/'"$CFG_MAPPING"'), para que la odometria del bridge ROS2 y la de
genie_rover NO diverjan. Con --raw usa los defaults del launch file.'
cmd_mapping_ros2() {
    wants_help "$@" && show_help "$HELP_MAPROS"
    local db="" config="$GENIE_DIR/$CFG_MAPPING" sdk_url="" feed_fps="" raw=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --db) db="$2"; shift 2 ;;
            --config) config="$2"; shift 2 ;;
            --sdk-url) sdk_url="$2"; shift 2 ;;
            --feed-fps) feed_fps="$2"; shift 2 ;;
            --raw) raw=1; shift ;;
            *) die "Opcion desconocida para mapping-ros2: $1 (--help para la lista)" ;;
        esac
    done

    neutralize_conda
    need_dir "$MAPPING_DIR"
    need_file "$MAPPING_DIR/rtabmap_mapping.launch.py"
    [ -f "$ROS2_DIR/earth_rover_bridge.py" ] || die "Falta $ROS2_DIR/earth_rover_bridge.py — corre './rover_launch.sh sync-ros2' o revisa ROS2_DIR"

    if [ -z "$db" ]; then
        mkdir -p "$MAPS_DIR"
        db="$MAPS_DIR/sesion_$(date +%Y%m%d_%H%M%S).db"
    else
        db="$(ensure_parent_dir "$db")"
    fi

    source_ros2
    cd "$MAPPING_DIR" || exit 1

    local launch_args=(database_path:="$db")
    [ -n "$sdk_url" ]  && launch_args+=(sdk_url:="$sdk_url")
    [ -n "$feed_fps" ] && launch_args+=(feed_fps:="$feed_fps")

    if [ "$raw" -eq 0 ]; then
        local helper="$ROS2_DIR/config_to_ros_params.py"
        if [ -f "$helper" ] && [ -f "$config" ]; then
            local derived
            derived="$(python3 "$helper" "$config" 2>/dev/null)"
            if [ -n "$derived" ]; then
                # shellcheck disable=SC2206
                local extra=($derived)
                launch_args+=("${extra[@]}")
                c_green "==> parametros derivados de $(basename "$config"): ${#extra[@]} valores"
            else
                warn "no pude derivar parametros de $config — uso los defaults del launch file"
            fi
        else
            warn "sin config_to_ros_params.py / config ($config) — uso los defaults del launch file"
        fi
    fi

    c_blue "==> RTAB-Map: database_path=$db"
    c_blue "    (al cortar con Ctrl+C revisa que el .db haya quedado ahi y no en ~/.ros/)"
    exec ros2 launch rtabmap_mapping.launch.py "${launch_args[@]}"
}

HELP_ROSBRIDGE='ros2-bridge — solo el bridge ROS2 del SDK (topicos, TF y odometria), sin RTAB-Map

  ./rover_launch.sh ros2-bridge [--sdk-url URL] [--feed-fps N] [--config PATH] [--raw]

Util para ver topicos con "ros2 topic list" / rviz2 sin levantar todo el
mapeo. Mismos parametros derivados que mapping-ros2.'
cmd_ros2_bridge() {
    wants_help "$@" && show_help "$HELP_ROSBRIDGE"
    local sdk_url="http://localhost:8000" feed_fps="" config="$GENIE_DIR/$CFG_MAPPING" raw=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --sdk-url) sdk_url="$2"; shift 2 ;;
            --feed-fps) feed_fps="$2"; shift 2 ;;
            --config) config="$2"; shift 2 ;;
            --raw) raw=1; shift ;;
            *) die "Opcion desconocida para ros2-bridge: $1 (--help para la lista)" ;;
        esac
    done

    neutralize_conda
    need_file "$ROS2_DIR/earth_rover_bridge.py"
    source_ros2
    cd "$ROS2_DIR" || exit 1

    local ros_args=(--ros-args -p "sdk_url:=$sdk_url")
    [ -n "$feed_fps" ] && ros_args+=(-p "feed_fps:=$feed_fps")
    if [ "$raw" -eq 0 ] && [ -f "$ROS2_DIR/config_to_ros_params.py" ] && [ -f "$config" ]; then
        local derived
        derived="$(python3 "$ROS2_DIR/config_to_ros_params.py" --style ros-args "$config" 2>/dev/null)"
        if [ -n "$derived" ]; then
            # shellcheck disable=SC2206
            local extra=($derived)
            ros_args+=("${extra[@]}")
            c_green "==> parametros derivados de $(basename "$config")"
        fi
    fi
    c_blue "==> earth_rover_bridge.py -> topicos /earth_rover/*  (Ctrl+C para cortar)"
    exec python3 earth_rover_bridge.py "${ros_args[@]}"
}

HELP_SYNC='sync-ros2 — copia el stack ROS2 canonico del repo dentro del SDK

  ./rover_launch.sh sync-ros2 [--force] [--dry-run]

Copia Indoor_Instalacion_SDK_SLAM/ros2/*  ->  earth-rovers-sdk/examples/ros2/
que es donde lo espera la documentacion del SDK. Sin --force no pisa
archivos que ya existan y sean distintos (te los lista).'
cmd_sync_ros2() {
    wants_help "$@" && show_help "$HELP_SYNC"
    local force=0 dry=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --force) force=1; shift ;;
            --dry-run) dry=1; shift ;;
            *) die "Opcion desconocida para sync-ros2: $1" ;;
        esac
    done

    need_dir "$ROS2_SRC_DIR"
    [ -d "$SDK_DIR" ] || die "No encuentro el SDK en $SDK_DIR (¿corriste 'git submodule update --init'?)"

    local rel dst src conflicts=0
    while IFS= read -r src; do
        rel="${src#"$ROS2_SRC_DIR"/}"
        dst="$ROS2_SDK_DIR/$rel"
        if [ -f "$dst" ] && ! cmp -s "$src" "$dst"; then
            if [ "$force" -eq 0 ]; then
                c_yellow "  distinto (no lo piso): examples/ros2/$rel"
                conflicts=$((conflicts + 1))
                continue
            fi
        fi
        if [ "$dry" -eq 1 ]; then
            echo "  copiaria: $rel"
        else
            mkdir -p "$(dirname "$dst")"
            cp "$src" "$dst" && c_green "  ok: examples/ros2/$rel"
        fi
    done < <(find "$ROS2_SRC_DIR" -type f \( -name '*.py' -o -name '*.md' -o -name 'Dockerfile' \))

    if [ "$conflicts" -gt 0 ]; then
        echo
        warn "$conflicts archivo(s) ya existian con contenido distinto. Revisalos y volve a correr con --force si querés pisarlos."
    fi
    echo
    c_blue "Destino: $ROS2_SDK_DIR"
}

HELP_SDKCLIENT='sdk-client — prueba de conexion SDK <-> genie (solo lectura, no mueve el rover)

  ./rover_launch.sh sdk-client [--base-url http://localhost:8000]'
cmd_sdk_client() {
    wants_help "$@" && show_help "$HELP_SDKCLIENT"
    local base_url=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --base-url) base_url="$2"; shift 2 ;;
            *) die "Opcion desconocida para sdk-client: $1" ;;
        esac
    done
    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1
    c_blue "==> genie_rover.sdk_client — prueba de conexion, no mueve el rover"
    if [ -n "$base_url" ]; then
        exec python -m genie_rover.sdk_client --base-url "$base_url"
    fi
    exec python -m genie_rover.sdk_client
}

HELP_PERCEPTION='perception — prueba de percepcion SAM-TP sobre una imagen

  ./rover_launch.sh perception --image FOTO.jpg [--config PATH] [--out DIR]'
cmd_perception() {
    wants_help "$@" && show_help "$HELP_PERCEPTION"
    local image="" config="$CFG_OUTDOOR" out="debug/"
    while [ $# -gt 0 ]; do
        case "$1" in
            --image) image="$2"; shift 2 ;;
            --config) config="$2"; shift 2 ;;
            --out) out="$2"; shift 2 ;;
            *) die "Opcion desconocida para perception: $1" ;;
        esac
    done
    [ -n "$image" ] || die "Falta --image <ruta_a_foto.jpg>"

    neutralize_conda
    need_dir "$GENIE_DIR"
    activate_venv "$GENIE_VENV"
    cd "$GENIE_DIR" || exit 1
    need_file "$image"
    need_file "$config"
    mkdir -p "$out" 2>/dev/null || true
    c_blue "==> genie_rover.perception sobre $image"
    exec python -m genie_rover.perception --config "$config" --image "$image" --out "$out"
}

HELP_TRAV='traversability — stack rover_traversability (SAM-TP standalone)

  ./rover_launch.sh traversability <nivel> [opciones]

Niveles que NO mueven el rover:
  predict --image FOTO [--out overlay.png]
  live    [--save-dir DIR] [--interval S] [--max-frames N]
  capture --save-dir DIR [--interval S] [--max-frames N] [--with-policy]
  policy-test --images DIR --out DIR [--configs JSON] [--cache-dir DIR] [--no-overlays]
  tune    --images DIR --out DIR [--base-config JSON] [--labels CSV] [--cache-dir DIR]

Niveles que SI mueven el rover (piden confirmacion):
  drive   [--save-dir DIR] [--interval S] [--max-iterations N]
  mission [--start-mission] [--arrive-attempt-m M] [--interval S] [--max-steps N]

Opciones globales del modelo (validas en cualquier nivel):
  --checkpoint PATH   --device cuda|mps|cpu   --no-refine'
cmd_traversability() {
    wants_help "$@" && show_help "$HELP_TRAV"
    local level="${1:-}"
    [ -n "$level" ] || die "Falta el nivel (--help para la lista)"
    shift || true

    # --- flags globales del modelo (van ANTES del subcomando en demo.py) ---
    local g_checkpoint="" g_device="" g_norefine=0
    local rest=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --checkpoint) g_checkpoint="$2"; shift 2 ;;
            --device) g_device="$2"; shift 2 ;;
            --no-refine) g_norefine=1; shift ;;
            *) rest+=("$1"); shift ;;
        esac
    done
    set -- ${rest[@]+"${rest[@]}"}

    neutralize_conda
    need_dir "$TRAV_DIR"
    activate_venv "$GENIE_VENV"

    # OJO: --no-refine solo existe en rover_traversability.demo. Los modulos
    # de testing (capture_test / policy_test / policy_tuner) aceptan
    # --checkpoint y --device pero NO --no-refine: pasarselo los hace abortar.
    local globals=()
    [ -n "$g_checkpoint" ] && globals+=(--checkpoint "$g_checkpoint")
    [ -n "$g_device" ] && globals+=(--device "$g_device")
    if [ "$g_norefine" -eq 1 ]; then
        case "$level" in
            predict|live|drive|mission) globals+=(--no-refine) ;;
            *) warn "--no-refine no aplica a '$level' (solo a predict/live/drive/mission), lo ignoro" ;;
        esac
    fi

    # policy-test / tune / capture viven en el paquete "testing", que se
    # importa parado en traversability/. El resto se corre desde el repo.
    case "$level" in
        predict|live|drive|mission)
            cd "$REPO" || exit 1
            python -c "import rover_traversability" 2>/dev/null || \
                die "rover_traversability no esta instalado en $GENIE_VENV. Corre, con ese venv activo y parado en $REPO:  pip install -e './traversability[hf]'"
            ;;
        capture|policy-test|tune)
            cd "$TRAV_DIR" || exit 1
            ;;
    esac

    case "$level" in
        predict)
            local image="" out="overlay.png"
            while [ $# -gt 0 ]; do
                case "$1" in
                    --image) image="$2"; shift 2 ;;
                    --out) out="$2"; shift 2 ;;
                    *) die "Opcion desconocida para predict: $1" ;;
                esac
            done
            [ -n "$image" ] || die "Falta --image <ruta_a_foto.jpg>"
            need_file "$image"
            c_blue "==> traversability predict sobre $image"
            exec python -m rover_traversability.demo ${globals[@]+"${globals[@]}"} predict "$image" --out "$(ensure_parent_dir "$out")"
            ;;
        live)
            local save_dir="trav_out" interval="" max_frames=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    --save-dir) save_dir="$2"; shift 2 ;;
                    --interval) interval="$2"; shift 2 ;;
                    --max-frames) max_frames="$2"; shift 2 ;;
                    *) die "Opcion desconocida para live: $1" ;;
                esac
            done
            local a=(--save-dir "$save_dir")
            [ -n "$interval" ] && a+=(--interval "$interval")
            [ -n "$max_frames" ] && a+=(--max-frames "$max_frames")
            c_blue "==> traversability live — NO mueve el rover, overlays en $save_dir"
            exec python -m rover_traversability.demo ${globals[@]+"${globals[@]}"} live "${a[@]}"
            ;;
        drive)
            local save_dir="" interval="" max_iterations=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    --save-dir) save_dir="$2"; shift 2 ;;
                    --interval) interval="$2"; shift 2 ;;
                    --max-iterations) max_iterations="$2"; shift 2 ;;
                    *) die "Opcion desconocida para drive: $1" ;;
                esac
            done
            local a=(--yes-i-want-the-rover-to-move)
            [ -n "$save_dir" ] && a+=(--save-dir "$save_dir")
            [ -n "$interval" ] && a+=(--interval "$interval")
            [ -n "$max_iterations" ] && a+=(--max-iterations "$max_iterations")
            c_yellow "==> traversability drive — MODO REAL, esquiva obstaculos sin meta GPS"
            exec python -m rover_traversability.demo ${globals[@]+"${globals[@]}"} drive "${a[@]}"
            ;;
        mission)
            local start_mission=0 arrive="" interval="" max_steps=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    --start-mission) start_mission=1; shift ;;
                    --arrive-attempt-m) arrive="$2"; shift 2 ;;
                    --interval) interval="$2"; shift 2 ;;
                    --max-steps) max_steps="$2"; shift 2 ;;
                    *) die "Opcion desconocida para mission: $1" ;;
                esac
            done
            local a=(--yes-i-want-the-rover-to-move)
            # v1 mandaba --start-mission SIEMPRE; ahora es opt-in, para poder
            # retomar una mision ya arrancada sin resetearla.
            [ "$start_mission" -eq 1 ] && a+=(--start-mission)
            [ -n "$arrive" ] && a+=(--arrive-attempt-m "$arrive")
            [ -n "$interval" ] && a+=(--interval "$interval")
            [ -n "$max_steps" ] && a+=(--max-steps "$max_steps")
            c_yellow "==> traversability mission — MODO REAL, navega checkpoints GPS"
            [ "$start_mission" -eq 0 ] && c_blue "    (sin --start-mission: asume que la mision ya esta arrancada)"
            exec python -m rover_traversability.demo ${globals[@]+"${globals[@]}"} mission "${a[@]}"
            ;;
        capture)
            local save_dir="" interval="" max_frames="" with_policy=0
            while [ $# -gt 0 ]; do
                case "$1" in
                    --save-dir) save_dir="$2"; shift 2 ;;
                    --interval) interval="$2"; shift 2 ;;
                    --max-frames) max_frames="$2"; shift 2 ;;
                    --with-policy) with_policy=1; shift ;;
                    *) die "Opcion desconocida para capture: $1" ;;
                esac
            done
            [ -n "$save_dir" ] || die "Falta --save-dir <carpeta>"
            local a=(--save-dir "$save_dir")
            [ -n "$interval" ] && a+=(--interval "$interval")
            [ -n "$max_frames" ] && a+=(--max-frames "$max_frames")
            [ "$with_policy" -eq 1 ] && a+=(--with-policy)
            c_blue "==> testing.capture_test — junta frames del SDK, no mueve el rover"
            exec python -m testing.capture_test ${globals[@]+"${globals[@]}"} "${a[@]}"
            ;;
        policy-test)
            local images="" out="" configs="" cache_dir="" no_overlays=0
            while [ $# -gt 0 ]; do
                case "$1" in
                    --images) images="$2"; shift 2 ;;
                    --out) out="$2"; shift 2 ;;
                    --configs) configs="$2"; shift 2 ;;
                    --cache-dir) cache_dir="$2"; shift 2 ;;
                    --no-overlays) no_overlays=1; shift ;;
                    *) die "Opcion desconocida para policy-test: $1" ;;
                esac
            done
            [ -n "$images" ] || die "Falta --images <carpeta con frames>"
            [ -n "$out" ] || die "Falta --out <carpeta de salida>"
            local a=(--images "$images" --out "$out")
            [ -n "$configs" ] && a+=(--configs "$configs")
            [ -n "$cache_dir" ] && a+=(--cache-dir "$cache_dir")
            [ "$no_overlays" -eq 1 ] && a+=(--no-overlays)
            c_blue "==> testing.policy_test — barrido de configs de policy sobre frames guardados"
            exec python -m testing.policy_test ${globals[@]+"${globals[@]}"} "${a[@]}"
            ;;
        tune)
            local images="" out="" base_config="" labels="" cache_dir=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    --images) images="$2"; shift 2 ;;
                    --out) out="$2"; shift 2 ;;
                    --base-config) base_config="$2"; shift 2 ;;
                    --labels) labels="$2"; shift 2 ;;
                    --cache-dir) cache_dir="$2"; shift 2 ;;
                    *) die "Opcion desconocida para tune: $1" ;;
                esac
            done
            [ -n "$images" ] || die "Falta --images <carpeta con frames>"
            [ -n "$out" ] || die "Falta --out <carpeta de salida>"
            local a=(--images "$images" --out "$out")
            [ -n "$base_config" ] && a+=(--base-config "$base_config")
            [ -n "$labels" ] && a+=(--labels "$labels")
            [ -n "$cache_dir" ] && a+=(--cache-dir "$cache_dir")
            c_blue "==> testing.policy_tuner — busca la mejor config de policy"
            exec python -m testing.policy_tuner ${globals[@]+"${globals[@]}"} "${a[@]}"
            ;;
        *) die "Nivel desconocido: $level (--help para la lista)" ;;
    esac
}

HELP_ROSCHECK='ros2-check — talker de prueba de ROS2 (Ctrl+C para cortar)'
cmd_ros2_check() {
    wants_help "$@" && show_help "$HELP_ROSCHECK"
    neutralize_conda
    source_ros2
    c_blue "==> talker de prueba. En OTRA terminal:"
    c_blue "      source $ROS_SETUP && ros2 run demo_nodes_py listener"
    exec ros2 run demo_nodes_cpp talker
}

HELP_MAPS='maps — lista los mapas y bases de datos de mapeo que hay en el disco

  ./rover_launch.sh maps [--dir CARPETA]'
cmd_maps() {
    wants_help "$@" && show_help "$HELP_MAPS"
    local dir="$MAPS_DIR"
    while [ $# -gt 0 ]; do
        case "$1" in
            --dir) dir="$(expand_path "$2")"; shift 2 ;;
            *) die "Opcion desconocida para maps: $1" ;;
        esac
    done
    c_blue "Bases de RTAB-Map (.db)"
    find "$dir" "$HOME/.ros" -maxdepth 1 -name '*.db' -printf '  %10s  %TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -k2 || echo "  (ninguna)"
    echo
    c_blue "Mapas exportados (yaml+pgm) de PersistentMap"
    find "$dir" "$GENIE_DIR/maps" "$REPO/maps" -maxdepth 2 -name '*.yaml' -printf '  %TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort || echo "  (ninguno)"
}

HELP_DOCTOR='doctor — chequea que cada pieza este instalada y que las rutas resuelvan

  ./rover_launch.sh doctor [--quiet]'
cmd_doctor() {
    wants_help "$@" && show_help "$HELP_DOCTOR"
    local problems=0
    ok()   { c_green "OK    $*"; }
    bad()  { c_red   "FALTA $*"; problems=$((problems + 1)); }
    soft() { c_yellow "aviso $*"; }

    echo "== Entornos =="
    command -v conda >/dev/null 2>&1 && soft "conda instalado (este script lo desactiva solo antes de cada comando)"
    [ -f "$SDK_VENV/bin/activate" ]   && ok "venv del SDK: $SDK_VENV"     || bad "venv del SDK: $SDK_VENV"
    [ -f "$GENIE_VENV/bin/activate" ] && ok "venv de genie: $GENIE_VENV"  || bad "venv de genie: $GENIE_VENV"
    if [ -x "$GENIE_VENV/bin/python" ]; then
        echo "      python de genie: $("$GENIE_VENV/bin/python" -V 2>&1)"
    fi

    echo
    echo "== Codigo =="
    [ -f "$SDK_DIR/main.py" ] && ok "SDK main.py" || bad "$SDK_DIR/main.py (¿git submodule update --init?)"
    [ -f "$SDK_DIR/.env" ] && ok "SDK .env" || soft "no hay $SDK_DIR/.env (copia .env.sample y completá SDK_API_TOKEN / BOT_SLUG)"
    [ -f "$GENIE_DIR/genie_rover/bridge.py" ] && ok "genie_rover" || bad "$GENIE_DIR/genie_rover"
    [ -f "$GENIE_DIR/genie_rover/Indoor/indoor_bridge.py" ] && ok "genie_rover.Indoor" || bad "genie_rover/Indoor"
    [ -f "$TRAV_DIR/rover_traversability/demo.py" ] && ok "rover_traversability (fuente)" || bad "$TRAV_DIR"
    [ -f "$DASHBOARD" ] && ok "dashboard_server.py" || soft "falta $DASHBOARD"

    # v1 miraba site-packages/python3.10 a mano: falla con -e (deja un .pth)
    # y con cualquier otra version de python. Preguntarle al interprete es lo
    # unico que sirve.
    if [ -x "$GENIE_VENV/bin/python" ]; then
        if "$GENIE_VENV/bin/python" -c "import rover_traversability" >/dev/null 2>&1; then
            ok "rover_traversability importable desde el venv de genie"
        else
            soft "rover_traversability NO importable — con el venv de genie activo y parado en $REPO:  pip install -e './traversability[hf]'"
        fi
        "$GENIE_VENV/bin/python" -c "import torch" >/dev/null 2>&1 \
            && ok "torch: $("$GENIE_VENV/bin/python" -c 'import torch;print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null)" \
            || soft "torch no instalado en el venv de genie (SAM-TP no va a correr)"
        "$GENIE_VENV/bin/python" -c "import fastapi, uvicorn" >/dev/null 2>&1 \
            && ok "fastapi+uvicorn (dashboard con logs por WebSocket)" \
            || soft "sin fastapi/uvicorn: el dashboard cae al modo stdlib (polling). pip install -r requirements-dashboard.txt"
    fi

    echo
    echo "== Configs =="
    local c
    for c in "$CFG_OUTDOOR" "$CFG_INDOOR" "$CFG_MAPPING"; do
        [ -f "$GENIE_DIR/$c" ] && ok "genie/$c" || soft "falta genie/$c"
    done

    echo
    echo "== Modelo =="
    local ckpt="$GENIE_DIR/sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt"
    if [ -f "$ckpt" ]; then ok "checkpoint SAM-TP ($(du -h "$ckpt" | cut -f1))"
    else bad "checkpoint SAM-TP en $ckpt"; fi
    [ -d "$HOME/.cache/rover_traversability" ] && ok "cache de pesos de rover_traversability" || soft "sin cache en ~/.cache/rover_traversability (se baja solo la primera vez)"

    echo
    echo "== ROS2 / mapeo =="
    if [ -f "$ROS_SETUP" ]; then
        ok "ROS2: $ROS_SETUP"
        # shellcheck disable=SC1090
        if (set +u; source "$ROS_SETUP" >/dev/null 2>&1; ros2 pkg list 2>/dev/null | grep -q '^rtabmap_slam$'); then
            ok "rtabmap_slam instalado"
        else
            soft "rtabmap_slam no encontrado (sudo apt install ros-\$ROS_DISTRO-rtabmap-ros)"
        fi
    else
        bad "ROS2 no encontrado (buscado en $ROS_SETUP)"
    fi
    [ -f "$MAPPING_DIR/rtabmap_mapping.launch.py" ] && ok "launch de mapeo: $MAPPING_DIR" || bad "rtabmap_mapping.launch.py (ni en el SDK ni en el repo)"
    [ -f "$ROS2_DIR/earth_rover_bridge.py" ] && ok "bridge ROS2: $ROS2_DIR/earth_rover_bridge.py" || bad "earth_rover_bridge.py"
    if [ -d "$ROS2_SDK_DIR" ] && [ -d "$ROS2_SRC_DIR" ]; then
        if diff -rq "$ROS2_SRC_DIR" "$ROS2_SDK_DIR" >/dev/null 2>&1; then
            ok "copia del stack ROS2 en el SDK sincronizada con la del repo"
        else
            soft "el stack ROS2 del SDK difiere del canonico del repo — './rover_launch.sh sync-ros2 --dry-run' para ver que cambia"
        fi
    elif [ ! -d "$ROS2_SDK_DIR" ]; then
        soft "el SDK todavia no tiene examples/ros2 — './rover_launch.sh sync-ros2' lo arma"
    fi

    echo
    echo "== Otros =="
    command -v tmux >/dev/null 2>&1 && ok "tmux (necesario para 'all')" || soft "sin tmux: el comando 'all' no va a funcionar"
    command -v google-chrome-stable >/dev/null 2>&1 && ok "google-chrome-stable" \
        || { command -v chromium >/dev/null 2>&1 && ok "chromium" || soft "sin Chrome/Chromium (el SDK lo necesita para el feed)"; }

    echo
    echo "== Rutas resueltas =="
    printf '  REPO=%s\n  SDK_DIR=%s\n  GENIE_DIR=%s\n  TRAV_DIR=%s\n  ROS2_DIR=%s\n  MAPPING_DIR=%s\n  MAPS_DIR=%s\n  ROS_SETUP=%s\n' \
        "$REPO" "$SDK_DIR" "$GENIE_DIR" "$TRAV_DIR" "$ROS2_DIR" "$MAPPING_DIR" "$MAPS_DIR" "$ROS_SETUP"

    echo
    if [ "$problems" -eq 0 ]; then c_green "Sin problemas criticos."
    else c_red "$problems problema(s) critico(s) — revisá las lineas FALTA."; fi
    return 0
}

HELP_ALL='all / all-indoor — levanta el stack completo en paneles de tmux

  ./rover_launch.sh all           sdk + dashboard + genie-bridge (outdoor)
  ./rover_launch.sh all-indoor    sdk + mapping-ros2 + dashboard (indoor/mapeo)

Attach: tmux attach -t rover   ·   cambiar panel: Ctrl+B y flechas
Salir sin matar: Ctrl+B luego D   ·   matar todo: tmux kill-session -t rover'
cmd_all() {
    wants_help "$@" && show_help "$HELP_ALL"
    local flavor="${1:-outdoor}"
    command -v tmux >/dev/null 2>&1 || die "Necesitas tmux para 'all' (sudo apt install -y tmux), o corre cada comando en su propia terminal."

    local session="rover"
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" -n main -c "$REPO"

    tmux send-keys -t "$session:main" "$0 sdk" C-m
    tmux split-window -h -t "$session:main" -c "$REPO"
    if [ "$flavor" = "indoor" ]; then
        tmux send-keys -t "$session:main" "echo 'Esperando al SDK...'; sleep 6; $0 mapping-ros2" C-m
    else
        tmux send-keys -t "$session:main" "echo 'Esperando al SDK...'; sleep 6; $0 genie-bridge" C-m
    fi
    tmux split-window -v -t "$session:main" -c "$REPO"
    tmux send-keys -t "$session:main" "$0 dashboard" C-m
    tmux select-layout -t "$session:main" tiled

    c_green "Sesion tmux '$session' armada (sdk / $( [ "$flavor" = indoor ] && echo mapping-ros2 || echo genie-bridge) / dashboard)"
    c_blue  "Dashboard: http://localhost:8765"
    tmux attach -t "$session"
}

usage() {
    cat <<'USAGE'
rover_launch.sh — launcher unico de La Rovernetta

  ./rover_launch.sh <comando> [opciones]
  ./rover_launch.sh <comando> --help      ayuda detallada de ese comando

Operacion
  sdk                Earth Rovers SDK (hypercorn main:app, :8000)
  dashboard          panel web local que maneja este launcher (:8765)
  genie-bridge       bridge outdoor con GPS  (genie_rover.bridge)
  indoor-bridge      tour de conos sin GPS   (genie_rover.Indoor.indoor_bridge)
  map-session        mapeo por frontera + export ROS (genie_rover.Indoor.map_session)
  traversability     stack rover_traversability: predict | live | drive | mission
                     | capture | policy-test | tune

ROS2 / mapeo
  mapping-ros2       RTAB-Map + bridge ROS2 (correccion de pose)
  ros2-bridge        solo el bridge ROS2 del SDK (topicos y TF)
  sync-ros2          copia el stack ROS2 del repo dentro del SDK
  ros2-check         talker de prueba de ROS2

Diagnostico
  doctor             chequeo de instalacion y de rutas
  sdk-client         prueba de conexion (solo lectura)
  perception         SAM-TP sobre una foto
  maps               lista .db de RTAB-Map y mapas exportados

Combos
  all                sdk + genie-bridge + dashboard en tmux
  all-indoor         sdk + mapping-ros2 + dashboard en tmux

Variables de entorno que pisan las rutas autodetectadas:
  REPO SDK_DIR GENIE_DIR TRAV_DIR ROS2_DIR MAPPING_DIR MAPS_DIR
  SDK_VENV GENIE_VENV ROS_SETUP
USAGE
    exit "${1:-1}"
}

# ------------------------------------------------------------------ main --

[ $# -ge 1 ] || usage
cmd="$1"; shift

case "$cmd" in
    sdk)            cmd_sdk "$@" ;;
    dashboard)      cmd_dashboard "$@" ;;
    genie-bridge)   cmd_genie_bridge "$@" ;;
    indoor-bridge)  cmd_indoor_bridge "$@" ;;
    map-session)    cmd_map_session "$@" ;;
    traversability) cmd_traversability "$@" ;;
    mapping-ros2)   cmd_mapping_ros2 "$@" ;;
    ros2-bridge)    cmd_ros2_bridge "$@" ;;
    sync-ros2)      cmd_sync_ros2 "$@" ;;
    ros2-check)     cmd_ros2_check "$@" ;;
    doctor)         cmd_doctor "$@" ;;
    sdk-client)     cmd_sdk_client "$@" ;;
    perception)     cmd_perception "$@" ;;
    maps)           cmd_maps "$@" ;;
    all)            cmd_all outdoor ;;
    all-indoor)     cmd_all indoor ;;
    -h|--help|help) usage 0 ;;
    *) die "Comando desconocido: $cmd (corre '$0 help' para ver la lista)" ;;
esac
