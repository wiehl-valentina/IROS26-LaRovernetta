#!/usr/bin/env python3
"""dashboard_server.py — panel de control local de La Rovernetta.

Le pone botones a `rover_launch.sh`: lanzas, parás y ves el log en vivo de
cada componente desde el navegador, sin acordarte flags ni rutas. Corre
DENTRO de tu WSL/Linux y solo escucha en localhost, asi que no es alcanzable
desde otras maquinas de la red (es un servidor que ejecuta comandos de tu
sistema: eso no es negociable).

    cd ~/IROS26-LaRovernetta
    ./rover_launch.sh dashboard          # recomendado (elige el mejor python)
    # o directo:
    python3 dashboard_server.py
    # abrir http://localhost:8765

DOS MOTORES, MISMA APP
----------------------
  * Si hay fastapi + uvicorn:  se usa ese. Los logs llegan por WebSocket
    (empuje del servidor, sin polling), asi que la salida aparece al
    instante y no hay una request cada 1.5 s por cada job vivo.
  * Si no los hay:             cae solo a http.server de la libreria
    estandar, con los MISMOS endpoints. El frontend detecta que no hay
    WebSocket y hace polling incremental.

En los dos casos el frontend es el mismo archivo y el manejo de procesos es
el mismo objeto (JobManager), asi que no hay dos comportamientos que
mantener sincronizados a mano.

    pip install -r requirements-dashboard.txt   # para el modo WebSocket

QUE CAMBIO RESPECTO DE LA VERSION ANTERIOR
------------------------------------------
  * Los comandos y sus flags se declaran UNA vez (COMMANDS) y de ahi salen
    tanto la whitelist del backend como los formularios del frontend. Antes
    estaban duplicados y ya habian quedado desincronizados con el launcher
    (faltaba --debug-dir del bridge, sobraba --start-mission forzado en
    traversability mission, no existian los niveles capture/policy-test/tune,
    ni los comandos ros2-bridge / sync-ros2 / maps).
  * El log ya no se manda entero en cada refresh: cada linea tiene numero de
    secuencia y el cliente pide solo lo nuevo. Con una corrida larga de ROS2
    la version anterior reenviaba megabytes por segundo.
  * Buffer acotado por job (BUFFER_LINES) — antes crecia sin limite en RAM.
  * Boton de PARADA DE EMERGENCIA que frena todos los procesos a la vez.
  * Estado del SDK y de las piezas en la barra superior.
  * Las preferencias de cada formulario se guardan del lado del servidor
    (.dashboard_state.json), asi que sobreviven a cerrar el navegador.
  * Descarga del log completo de un job a archivo.

NOVEDAD v3: BOTONES DE UN CLIC (STACKS)
---------------------------------------
Ademas de los comandos sueltos hay "stacks": un solo boton que levanta
VARIOS procesos en el orden correcto, esperando a que el anterior este
arriba. Cada paso sigue siendo un job normal (su panel, su log, su boton de
parar) y el stack agrega un panel coordinador que narra que lanzo y por que.
Parar el coordinador corta en cascada todo lo que lanzo.

  * "Grabar recorrido"  -> SDK + RTAB-Map (SLAM) + map-session por frontera.
  * "Mision indoor"     -> selector de modo ANTES de lanzar:
        checkpoints  ruta pregrabada (waypoints)
        map          correccion en funcion del mapa (RTAB-Map corrige la pose)
        both         las dos a la vez (lo recomendado si ya mapeaste)

Los stacks se declaran igual que los comandos (STACKS, mas abajo): de ahi
salen el formulario, la vista previa de lo que se va a ejecutar y la
validacion del backend. No hay strings de shell en ningun lado: cada paso
pasa por el mismo build_argv() con whitelist de los comandos sueltos.
"""

# NOTA: a proposito NO usamos `from __future__ import annotations`. FastAPI
# resuelve las anotaciones de cada endpoint contra los globals del modulo, y
# los nombres que importamos dentro de run_fastapi() (Request, WebSocket) no
# viven ahi: con las anotaciones diferidas los tomaba como query params y
# devolvia 422 en cada POST y 403 en el WebSocket. Requiere Python >= 3.10,
# que es lo que ya pide el resto del repo.

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------- ajustes ---

BUFFER_LINES = 5000          # lineas de log que se guardan por job
DEFAULT_PORT = 8765
STATE_FILE = ".dashboard_state.json"

SCRIPT_PATH = "./rover_launch.sh"
STATE_PATH = Path(STATE_FILE)


# ------------------------------------------------------- catalogo de comandos

def _extra(field, default, show_when):
    """`default`: valor inicial si el usuario todavia no toco nada.
    `show_when`: condicion para mostrar el campo (ver _cond_ok) — asi el
    formulario del stack muestra solo lo que aplica al modo elegido."""
    if default is not None:
        field["default"] = default
    if show_when is not None:
        field["show_when"] = show_when
    return field


def _txt(key, label, placeholder="", help="", default=None, show_when=None):
    return _extra({"key": key, "label": label, "type": "text",
                   "placeholder": placeholder, "help": help}, default, show_when)


def _sel(key, label, options, help="", default=None, show_when=None):
    # `options` acepta "valor" o ["valor", "etiqueta linda"]
    return _extra({"key": key, "label": label, "type": "select",
                   "options": options, "help": help}, default, show_when)


def _bool(key, label, help="", default=None, show_when=None):
    return _extra({"key": key, "label": label, "type": "bool", "help": help},
                  default, show_when)


# Fuente unica de verdad: de aca salen la validacion del argv Y los
# formularios del navegador. `moves` marca los comandos que mueven el rover
# (el front pide confirmacion y pinta la tarjeta).
COMMANDS = [
    # ---------------------------------------------------------- operacion ---
    {
        "cmd": "sdk", "group": "Operacion", "title": "SDK (hypercorn)",
        "desc": "Servidor del rover en :8000. Todo lo demas habla con esto.",
        "fields": [_txt("port", "Puerto", "8000"), _bool("reload", "--reload (autorecarga)")],
    },
    {
        "cmd": "genie-bridge", "group": "Operacion", "title": "Bridge outdoor (GPS)",
        "desc": "genie_rover.bridge — navega checkpoints GPS con SAM-TP.",
        "go": True, "moves": "go",
        "fields": [
            _txt("config", "Config", "configs/frodobot_rover.yaml"),
            _txt("max-seconds", "Max segundos", "300"),
            _txt("debug-dir", "Carpeta debug", "debug/run1"),
            _bool("start-mission", "--start-mission (arranca la mision en el SDK)"),
        ],
    },
    {
        "cmd": "indoor-bridge", "group": "Operacion", "title": "Bridge indoor (tour de conos)",
        "desc": "genie_rover.Indoor.indoor_bridge — sin GPS: busca conos, saca foto y sigue.",
        "go": True, "moves": "go",
        "fields": [
            _txt("config", "Config", "configs/indoor_cone_search.yaml"),
            _sel("search-mode", "Modo de busqueda", ["", "wander", "frontier", "waypoints"],
                 "vacio = el que diga el config"),
            _txt("waypoints-path", "Ruta de waypoints", "configs/waypoints_example.yaml",
                 "solo si el modo es waypoints"),
            _txt("max-seconds", "Max segundos", "180"),
            _txt("debug-dir", "Carpeta debug", "debug/indoor_run1"),
        ],
    },
    {
        "cmd": "map-session", "group": "Operacion", "title": "Mapeo por frontera",
        "desc": "genie_rover.Indoor.map_session — explora y exporta maps/*.yaml+.pgm.",
        "go": True, "moves": "go",
        "fields": [
            _txt("config", "Config", "configs/indoor_mapping.yaml"),
            _txt("map-out", "Prefijo de salida", "maps/sesion1"),
            _txt("export-every-s", "Exportar cada N s", "30"),
            _txt("max-seconds", "Max segundos", "300"),
            _txt("debug-dir", "Carpeta debug", "debug/map_run1"),
        ],
    },
    {
        "cmd": "traversability", "group": "Operacion", "title": "Traversability",
        "desc": "rover_traversability: prediccion, manejo reactivo y barridos de tuning.",
        "level_key": "level", "moves_levels": ["drive", "mission"],
        "fields": [
            _sel("level", "Nivel",
                 ["predict", "live", "drive", "mission", "capture", "policy-test", "tune"]),
            _txt("checkpoint", "Checkpoint (global)", "", "vacio = el default del paquete"),
            _sel("device", "Device (global)", ["", "cuda", "mps", "cpu"]),
            _bool("no-refine", "--no-refine (sin refinado por contraste)"),
            _txt("image", "Imagen", "screenshots/foto.jpg", "predict"),
            _txt("out", "Salida", "overlay.png", "predict / policy-test / tune"),
            _txt("save-dir", "Carpeta de salida", "trav_out", "live / drive / capture"),
            _txt("interval", "Intervalo (s)", "0.5", "live / drive / mission / capture"),
            _txt("max-frames", "Max frames", "", "live / capture"),
            _txt("max-iterations", "Max iteraciones", "", "drive"),
            _txt("max-steps", "Max pasos", "", "mission"),
            _txt("arrive-attempt-m", "Radio de llegada (m)", "8.0", "mission"),
            _bool("start-mission", "--start-mission", "mission: arranca la mision en el SDK"),
            _bool("with-policy", "--with-policy", "capture: guarda tambien la decision"),
            _txt("images", "Carpeta de frames", "trav_out", "policy-test / tune"),
            _txt("configs", "Configs a barrer (json)", "configs.json", "policy-test"),
            _txt("base-config", "Config base (json)", "", "tune"),
            _txt("labels", "Etiquetas (csv)", "", "tune"),
            _txt("cache-dir", "Cache de inferencia", "", "policy-test / tune"),
            _bool("no-overlays", "--no-overlays", "policy-test"),
        ],
        # que flags aplican a cada nivel (el resto se ignora al armar el argv)
        "level_fields": {
            "predict": ["image", "out"],
            "live": ["save-dir", "interval", "max-frames"],
            "drive": ["save-dir", "interval", "max-iterations"],
            "mission": ["start-mission", "arrive-attempt-m", "interval", "max-steps"],
            "capture": ["save-dir", "interval", "max-frames", "with-policy"],
            "policy-test": ["images", "out", "configs", "cache-dir", "no-overlays"],
            "tune": ["images", "out", "base-config", "labels", "cache-dir"],
        },
        "global_fields": ["checkpoint", "device", "no-refine"],
        # --no-refine solo existe en rover_traversability.demo; los modulos de
        # testing aceptan --checkpoint/--device pero no eso.
        "global_fields_by_level": {
            "capture": ["checkpoint", "device"],
            "policy-test": ["checkpoint", "device"],
            "tune": ["checkpoint", "device"],
        },
    },

    # --------------------------------------------------------- ros2/mapeo ---
    {
        "cmd": "mapping-ros2", "group": "ROS2 y mapeo", "title": "RTAB-Map (sesion de mapeo)",
        "desc": "Bridge ROS2 + camera_info + RTAB-Map. Corrige la deriva por cierre de bucles.",
        "fields": [
            _txt("db", "database_path", "~/maps/sesion1.db", "vacio = ~/maps/sesion_<fecha>.db"),
            _txt("config", "Config de genie del que derivar parametros", "genie/configs/indoor_mapping.yaml"),
            _txt("sdk-url", "URL del SDK", "http://localhost:8000"),
            _txt("feed-fps", "FPS del feed", "15"),
            _bool("raw", "--raw (usar los defaults del launch, no derivar del config)"),
        ],
    },
    {
        "cmd": "ros2-bridge", "group": "ROS2 y mapeo", "title": "Bridge ROS2 solo",
        "desc": "Publica /earth_rover/* y el TF, sin RTAB-Map. Para mirar con rviz2.",
        "fields": [
            _txt("sdk-url", "URL del SDK", "http://localhost:8000"),
            _txt("feed-fps", "FPS del feed", "15"),
            _txt("config", "Config de genie", "genie/configs/indoor_mapping.yaml"),
            _bool("raw", "--raw"),
        ],
    },
    {
        "cmd": "sync-ros2", "group": "ROS2 y mapeo", "title": "Sincronizar stack ROS2 al SDK",
        "desc": "Copia Indoor_Instalacion_SDK_SLAM/ros2 -> earth-rovers-sdk/examples/ros2.",
        "fields": [_bool("dry-run", "--dry-run (solo mostrar)"), _bool("force", "--force (pisar diferencias)")],
    },

    # -------------------------------------------------------- diagnostico ---
    {
        "cmd": "sdk-client", "group": "Diagnostico", "title": "Prueba de conexion",
        "desc": "Solo lectura: telemetria, GPS, frame y checkpoints. No mueve el rover.",
        "fields": [_txt("base-url", "URL del SDK", "http://localhost:8000")],
    },
    {
        "cmd": "perception", "group": "Diagnostico", "title": "Percepcion sobre una foto",
        "desc": "SAM-TP sobre una imagen guardada.",
        "fields": [
            _txt("image", "Ruta a la imagen", "screenshots/foto.jpg"),
            _txt("config", "Config", "configs/frodobot_rover.yaml"),
            _txt("out", "Carpeta de salida", "debug/"),
        ],
    },
    {
        "cmd": "maps", "group": "Diagnostico", "title": "Mapas y bases en disco",
        "desc": "Lista .db de RTAB-Map y mapas exportados.",
        "fields": [_txt("dir", "Carpeta", "~/maps")],
    },
    {
        "cmd": "ros2-check", "group": "Diagnostico", "title": "Test basico de ROS2",
        "desc": "Talker de prueba. Sirve para confirmar que ROS2 esta bien sourceado.",
        "fields": [],
    },
    {
        "cmd": "doctor", "group": "Diagnostico", "title": "Chequeo de instalacion",
        "desc": "Rutas resueltas, venvs, ROS2, checkpoint, configs.",
        "fields": [],
    },
]

COMMANDS_BY_NAME = {c["cmd"]: c for c in COMMANDS}


def build_argv(cmd: str, opts: dict) -> list[str]:
    """argv seguro para subprocess. NUNCA concatena texto libre a un shell:
    arma la lista termino a termino y rechaza cualquier flag que no este
    declarada en COMMANDS."""
    spec = COMMANDS_BY_NAME.get(cmd)
    if spec is None:
        raise ValueError(f"comando no permitido: {cmd}")

    by_key = {f["key"]: f for f in spec["fields"]}
    argv = [SCRIPT_PATH, cmd]

    def add(key: str, value) -> None:
        # --go no vive en `fields` (lo dibuja el front aparte, con su aviso)
        if key == "go" and spec.get("go"):
            if value:
                argv.append("--go")
            return
        field = by_key.get(key)
        if field is None:
            raise ValueError(f"flag no permitida para {cmd}: {key}")
        if field["type"] == "bool":
            if value:
                argv.append(f"--{key}")
            return
        text = str(value).strip()
        if not text:
            return
        if "\n" in text or "\r" in text:
            raise ValueError(f"valor invalido para --{key}")
        argv.extend([f"--{key}", text])

    # traversability: el nivel es posicional y cada nivel acepta otras flags
    if spec.get("level_key"):
        level = str(opts.get(spec["level_key"], "")).strip()
        valid = by_key[spec["level_key"]]["options"]
        if level not in valid:
            raise ValueError(f"nivel invalido: {level!r}")
        # El launcher lee el NIVEL como primer posicional y despues acepta las
        # flags globales del modelo en cualquier posicion, asi que el orden
        # correcto es: <nivel> <globales> <flags del nivel>.
        argv.append(level)
        globales = (spec.get("global_fields_by_level", {}).get(level)
                    or spec.get("global_fields", []))
        for key in globales:
            if key in opts:
                add(key, opts[key])
        for key in spec["level_fields"].get(level, []):
            if key in opts:
                add(key, opts[key])
        return argv

    for key, value in opts.items():
        add(key, value)
    return argv


def command_moves_rover(cmd: str, opts: dict) -> bool:
    spec = COMMANDS_BY_NAME.get(cmd) or {}
    if spec.get("moves") == "go":
        return bool(opts.get("go"))
    if spec.get("moves_levels"):
        return str(opts.get(spec.get("level_key", "level"), "")) in spec["moves_levels"]
    return False


# ------------------------------------------------------------- STACKS ------
# Un stack es UN boton que lanza VARIOS comandos en orden. Se declara igual
# que los comandos: `fields` arma el formulario, `steps` dice que se lanza y
# `when` decide que pasos aplican segun lo que eligio el usuario.
#
# Cada paso puede esperar:
#   wait_port "127.0.0.1:8000"  -> hasta que ese puerto acepte conexiones
#                                  (el SDK tarda unos segundos en levantar y
#                                   RTAB-Map se cae si arranca antes)
#   wait_s N                    -> N segundos fijos (para lo que no tiene
#                                  puerto que mirar, como RTAB-Map cargando)
#
# Las opciones de cada paso NO son texto libre: son referencias a campos del
# formulario ({"from": "campo"}) o constantes ({"const": "valor"}), y todas
# terminan pasando por build_argv(), que rechaza cualquier flag que el
# comando no declare. Un stack no puede ejecutar nada que no se pueda
# ejecutar con un boton suelto.

def _step(cmd, label, opts=None, when=None, wait_s=0, wait_port=None):
    return {"cmd": cmd, "label": label, "opts": opts or {},
            "when": when, "wait_s": wait_s, "wait_port": wait_port}


def _from(field):
    return {"from": field}


def _const(value):
    return {"const": value}


SDK_PORT = "127.0.0.1:8000"

STACKS = [
    # ------------------------------------------------- grabar un recorrido ---
    {
        "id": "record-run",
        "title": "Grabar recorrido (SLAM)",
        "button": "GRABAR RECORRIDO",
        "desc": ("Un clic: SDK + RTAB-Map + sesion de mapeo por frontera. El rover "
                 "explora, arma el mapa 2D y lo exporta a maps/<prefijo>.yaml+.pgm. "
                 "El control manual sigue siendo del SDK: esto solo orquesta."),
        "go": True,
        "fields": [
            _bool("sdk", "Levantar el SDK", "destildalo si ya lo tenes corriendo", default=True),
            _bool("slam", "Correccion visual RTAB-Map (SLAM)",
                  "recomendado: sin esto la pose es solo rueda+giro y deriva", default=True),
            _txt("map_out", "Prefijo del mapa", "maps/sesion1",
                 "sin extension: escribe .yaml y .pgm", default="maps/sesion1"),
            _txt("db", "Base de RTAB-Map (.db)", "maps/sesion1.db",
                 "vacio = maps/sesion_<fecha>.db", show_when={"field": "slam", "truthy": True}),
            _txt("config", "Config de mapeo", "configs/indoor_mapping.yaml",
                 default="configs/indoor_mapping.yaml"),
            _txt("export_every", "Exportar el mapa cada N s", "30",
                 "seguro ante cortes: si se corta, el mapa hasta ahi ya esta escrito",
                 default="30"),
            _txt("max_seconds", "Max segundos", "300", "vacio = hasta Ctrl+C"),
            _txt("debug_dir", "Carpeta debug", "debug/mapeo1",
                 "guarda RGB, plan y BEV por frame — pesa"),
        ],
        "steps": [
            _step("sdk", "SDK (:8000)", when={"field": "sdk", "truthy": True}),
            _step("mapping-ros2", "RTAB-Map + bridge ROS2",
                  opts={"db": _from("db")},
                  when={"field": "slam", "truthy": True},
                  wait_port=SDK_PORT),
            _step("map-session", "Sesion de mapeo por frontera",
                  opts={"config": _from("config"), "map-out": _from("map_out"),
                        "export-every-s": _from("export_every"),
                        "max-seconds": _from("max_seconds"),
                        "debug-dir": _from("debug_dir"), "go": _from("go")},
                  wait_port=SDK_PORT, wait_s=8),
        ],
        "requires": [
            {"field": "map_out", "error": "Falta el prefijo del mapa (ej. maps/sesion1)"},
        ],
        "after": ("Cuando termine: sacale los waypoints al mapa con "
                  "poses_to_waypoints.py y usalos en la mision indoor."),
    },

    # -------------------------------------------------------- mision indoor ---
    {
        "id": "indoor-run",
        "title": "Mision indoor (tour de conos)",
        "button": "LANZAR MISION",
        "desc": ("Un clic: levanta todo lo que la mision necesita. Elegi primero "
                 "COMO se ubica el rover mientras no ve un cono."),
        "go": True,
        "fields": [
            _sel("mode", "Modo de ejecucion", [
                    ["checkpoints", "A · Ruta pregrabada (checkpoints)"],
                    ["map", "B · Correccion en funcion del mapa"],
                    ["both", "C · Ambas (ruta + correccion)"],
                 ],
                 "A: sigue el yaml de waypoints. B: RTAB-Map corrige la pose y "
                 "explora por frontera. C: las dos — lo recomendado si ya mapeaste.",
                 default="both"),
            _bool("sdk", "Levantar el SDK", "destildalo si ya lo tenes corriendo", default=True),
            _txt("waypoints", "Ruta de waypoints (yaml)", "configs/waypoints_example.yaml",
                 "la que sacaste del mapeo con poses_to_waypoints.py",
                 default="configs/waypoints_example.yaml",
                 show_when={"field": "mode", "in": ["checkpoints", "both"]}),
            _txt("db", "Base de RTAB-Map (.db)", "maps/sesion1.db",
                 "la del mapeo previo: con esa base cierra bucles contra lo ya visto",
                 show_when={"field": "mode", "in": ["map", "both"]}),
            _txt("config", "Config de la mision", "configs/indoor_cone_search.yaml",
                 default="configs/indoor_cone_search.yaml"),
            _txt("max_seconds", "Max segundos", "300", "vacio = hasta Ctrl+C"),
            _txt("debug_dir", "Carpeta debug", "debug/indoor1"),
        ],
        "steps": [
            _step("sdk", "SDK (:8000)", when={"field": "sdk", "truthy": True}),
            _step("mapping-ros2", "RTAB-Map (correccion de pose)",
                  opts={"db": _from("db")},
                  when={"field": "mode", "in": ["map", "both"]},
                  wait_port=SDK_PORT),
            # A y C: ruta pregrabada
            _step("indoor-bridge", "Mision sobre la ruta pregrabada",
                  opts={"config": _from("config"),
                        "search-mode": _const("waypoints"),
                        "waypoints-path": _from("waypoints"),
                        "max-seconds": _from("max_seconds"),
                        "debug-dir": _from("debug_dir"), "go": _from("go")},
                  when={"field": "mode", "in": ["checkpoints", "both"]},
                  wait_port=SDK_PORT, wait_s=8),
            # B: sin ruta fija, explora por frontera con la pose corregida
            _step("indoor-bridge", "Mision explorando por frontera (pose corregida)",
                  opts={"config": _from("config"),
                        "search-mode": _const("frontier"),
                        "max-seconds": _from("max_seconds"),
                        "debug-dir": _from("debug_dir"), "go": _from("go")},
                  when={"field": "mode", "in": ["map"]},
                  wait_port=SDK_PORT, wait_s=8),
        ],
        "requires": [
            {"field": "waypoints",
             "when": {"field": "mode", "in": ["checkpoints", "both"]},
             "error": ("El modo elegido necesita la ruta de waypoints. Generala del "
                       "mapeo con poses_to_waypoints.py o cambia a 'B · Correccion "
                       "en funcion del mapa'.")},
        ],
        "notes": {
            "map": ("Modo B/C: la correccion de pose solo se aplica si el config tiene "
                    "mapping.rtabmap_correction.enabled: true. Si RTAB-Map no esta "
                    "arriba, el bridge avisa una vez y sigue con rueda+giro."),
            "both": ("Modo B/C: la correccion de pose solo se aplica si el config tiene "
                     "mapping.rtabmap_correction.enabled: true. Si RTAB-Map no esta "
                     "arriba, el bridge avisa una vez y sigue con rueda+giro."),
        },
    },
]

STACKS_BY_ID = {st["id"]: st for st in STACKS}


def _cond_ok(cond: dict | None, form: dict) -> bool:
    """Evalua un `when` / `show_when` contra los valores del formulario."""
    if not cond:
        return True
    value = form.get(cond["field"])
    if "in" in cond:
        return str(value) in [str(v) for v in cond["in"]]
    if "equals" in cond:
        return str(value) == str(cond["equals"])
    if cond.get("truthy"):
        return bool(value) and value not in ("0", "false", "no")
    return True


def build_stack_plan(stack_id: str, form: dict) -> list[dict]:
    """De un stack + lo que eligio el usuario, saca la lista de pasos con su
    argv ya resuelto. Se usa para EJECUTAR y tambien para la vista previa que
    muestra el navegador antes de apretar el boton: lo que se ve es
    exactamente lo que se va a correr."""
    stack = STACKS_BY_ID.get(stack_id)
    if stack is None:
        raise ValueError(f"stack no permitido: {stack_id}")

    for req in stack.get("requires", []):
        if not _cond_ok(req.get("when"), form):
            continue
        if not str(form.get(req["field"], "") or "").strip():
            raise ValueError(req["error"])

    plan: list[dict] = []
    for step in stack["steps"]:
        if not _cond_ok(step.get("when"), form):
            continue
        opts: dict = {}
        for flag, src in step["opts"].items():
            value = src["const"] if "const" in src else form.get(src["from"])
            if value is None or value is False or value == "":
                continue
            opts[flag] = value
        plan.append({
            "label": step["label"],
            "cmd": step["cmd"],
            "opts": opts,
            "argv": build_argv(step["cmd"], opts),      # valida la whitelist
            "wait_s": step.get("wait_s", 0),
            "wait_port": step.get("wait_port"),
        })

    if not plan:
        raise ValueError("con esas opciones no queda ningun proceso para lanzar")
    return plan


# --------------------------------------------------- entorno de los hijos ---

def child_env() -> dict:
    """Entorno para los procesos que lanza el dashboard.

    Dos cosas que NO se pueden dejar en el default, porque el stdout del hijo
    es un pipe y no una terminal:

    PYTHONUNBUFFERED
        Python bufferea por bloques (~8 KB) cuando stdout no es un tty, y solo
        hace flush al salir. Con la consola en tabla de las misiones (~100
        bytes por fila) eso son ~80 iteraciones acumuladas en RAM: el log
        aparecia entero recien al terminar el proceso, en vez de fila a fila.
        Con esto stdout queda sin buffer y cada linea llega apenas se imprime.
        Tambien lo heredan los subprocesos que lance el hijo.

    FORCE_COLOR / NO_COLOR
        Por el mismo motivo (`isatty()` falso), console_report.py y las
        funciones c_red/c_green/... del launcher apagan solos los colores
        ANSI. El frontend SI sabe interpretarlos (ansiToFragment), asi que
        los forzamos y se pierde menos informacion: el color del estado y el
        verde/rojo de la barra de traversabilidad. Si alguien exporta
        NO_COLOR a proposito, se respeta y no se fuerza nada.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if not env.get("NO_COLOR"):
        env["FORCE_COLOR"] = "1"
    return env


# ------------------------------------------------------------- JobManager ---

class Job:
    """Un proceso lanzado (kind="proc") o el coordinador de un stack
    (kind="stack", sin proceso propio: su trabajo es lanzar a los otros y
    dejar escrito por que)."""

    __slots__ = ("id", "cmd", "argv", "proc", "lines", "next_seq", "status",
                 "returncode", "started_at", "finished_at", "moves", "lock",
                 "kind", "title", "children", "cancel")

    def __init__(self, job_id: str, cmd: str, argv: list[str], proc, moves: bool,
                 kind: str = "proc", title: str = ""):
        self.id = job_id
        self.cmd = cmd
        self.argv = argv
        self.proc = proc
        self.kind = kind
        self.title = title or cmd
        self.children: list[str] = []          # jobs que lanzo (solo stacks)
        self.cancel = threading.Event()
        self.lines: deque[tuple[int, str]] = deque(maxlen=BUFFER_LINES)
        self.next_seq = 0
        self.status = "running"
        self.returncode: int | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.moves = moves
        self.lock = threading.Lock()

    def append(self, text: str) -> tuple[int, str]:
        with self.lock:
            entry = (self.next_seq, text)
            self.lines.append(entry)
            self.next_seq += 1
            return entry

    def snapshot(self, since: int = -1) -> dict:
        with self.lock:
            # cada linea viaja con su numero de secuencia: si dos refrescos se
            # cruzan (uno del POST de lanzamiento y otro del evento WebSocket),
            # el cliente puede descartar lo que ya pinto en vez de duplicarlo.
            nuevas = [[seq, t] for seq, t in self.lines if seq > since]
            first = self.lines[0][0] if self.lines else 0
            return {
                "id": self.id, "cmd": self.cmd, "argv": self.argv,
                "kind": self.kind, "title": self.title,
                "children": list(self.children),
                "status": self.status, "returncode": self.returncode,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "moves": self.moves,
                "lines": nuevas,
                "next_seq": self.next_seq,
                # si el buffer descarto lineas viejas, el cliente tiene que
                # saberlo para no creer que su historial esta completo
                "truncated_before": first,
            }


class JobManager:
    """Todo el manejo de procesos, independiente del framework web."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self._subscribers: list = []           # callables(evento: dict)
        self._sub_lock = threading.Lock()

    # --- pub/sub (lo usa el modo WebSocket; el modo stdlib simplemente no
    #     se suscribe y consulta por HTTP) -----------------------------------
    def subscribe(self, fn) -> None:
        with self._sub_lock:
            self._subscribers.append(fn)

    def unsubscribe(self, fn) -> None:
        with self._sub_lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def _emit(self, event: dict) -> None:
        with self._sub_lock:
            subs = list(self._subscribers)
        for fn in subs:
            try:
                fn(event)
            except Exception:
                pass

    # --- ciclo de vida ------------------------------------------------------
    def start(self, cmd: str, opts: dict) -> str:
        argv = build_argv(cmd, opts)
        moves = command_moves_rover(cmd, opts)

        with self.lock:
            for job in self.jobs.values():
                if job.cmd == cmd and job.status == "running":
                    raise ValueError(
                        f"'{cmd}' ya esta corriendo (job {job.id}). Paralo antes de relanzarlo.")

        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env(),
            # grupo de procesos propio: asi podemos cortar todo lo que lance
            # (ros2 launch arranca varios hijos) y no solo el proceso padre
            start_new_session=True,
        )

        job = Job(uuid.uuid4().hex[:8], cmd, argv, proc, moves)
        with self.lock:
            self.jobs[job.id] = job
        job.append(f"[dashboard] $ {' '.join(shlex.quote(a) for a in argv)}")
        self._emit({"type": "job_started", "job": job.snapshot()})

        threading.Thread(target=self._reader, args=(job,), daemon=True).start()
        return job.id

    def _log(self, job: Job, text: str) -> None:
        seq, line = job.append(text)
        self._emit({"type": "line", "id": job.id, "seq": seq, "text": line})

    # --- stacks -------------------------------------------------------------
    def start_stack(self, stack_id: str, form: dict) -> str:
        """Lanza un stack: valida TODO el plan primero (asi no queda medio
        stack arriba por un error en el ultimo paso) y despues lo ejecuta en
        un hilo, respetando las esperas de cada paso."""
        stack = STACKS_BY_ID.get(stack_id)
        if stack is None:
            raise ValueError(f"stack no permitido: {stack_id}")
        plan = build_stack_plan(stack_id, form)
        moves = any(p["opts"].get("go") for p in plan)

        with self.lock:
            for job in self.jobs.values():
                if job.kind == "stack" and job.cmd == stack_id and job.status == "running":
                    raise ValueError(
                        f"'{stack['title']}' ya se esta levantando (job {job.id}). "
                        "Espera a que termine de lanzar o paralo.")

        job = Job(uuid.uuid4().hex[:8], stack_id,
                  [p["cmd"] for p in plan], None, moves,
                  kind="stack", title=stack["title"])
        with self.lock:
            self.jobs[job.id] = job
        job.append(f"[stack] {stack['title']} — {len(plan)} paso(s)")
        for i, step in enumerate(plan, 1):
            job.append(f"[stack]   {i}. {step['label']}: "
                       + " ".join(shlex.quote(a) for a in step["argv"]))
        if moves:
            job.append("[stack] MODO REAL (--go): el rover se va a mover")
        self._emit({"type": "job_started", "job": job.snapshot()})

        threading.Thread(target=self._stack_runner, args=(job, plan), daemon=True).start()
        return job.id

    def _wait_port(self, job: Job, hostport: str, timeout: float = 45.0) -> bool:
        host, _, port = hostport.partition(":")
        port_i = int(port or 80)
        if _port_open(host, port_i):
            return True
        self._log(job, f"[stack] esperando a que {hostport} responda (hasta {int(timeout)} s)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if job.cancel.is_set():
                return False
            if _port_open(host, port_i):
                self._log(job, f"[stack] {hostport} arriba")
                return True
            time.sleep(0.5)
        self._log(job, f"[stack] {hostport} no respondio a tiempo — sigo igual "
                       "(el paso puede fallar solo si de verdad hacia falta)")
        return False

    def _sleep(self, job: Job, seconds: float) -> bool:
        """Espera cortable: si paran el stack, no hay que aguantar el sleep."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if job.cancel.is_set():
                return False
            time.sleep(0.2)
        return True

    def _stack_runner(self, job: Job, plan: list[dict]) -> None:
        rc = 0
        try:
            for i, step in enumerate(plan, 1):
                if job.cancel.is_set():
                    self._log(job, "[stack] cancelado antes de terminar de lanzar")
                    rc = 130
                    break

                if step["wait_port"]:
                    self._wait_port(job, step["wait_port"])
                if job.cancel.is_set():
                    rc = 130
                    break
                if step["wait_s"]:
                    self._log(job, f"[stack] margen de {step['wait_s']} s para que "
                                   "termine de levantar el paso anterior")
                    if not self._sleep(job, step["wait_s"]):
                        rc = 130
                        break

                # Si el comando ya esta corriendo (lo lanzaste a mano, o quedo
                # de un stack anterior), no se relanza: se reusa y se avisa.
                with self.lock:
                    ya = next((j for j in self.jobs.values()
                               if j.kind == "proc" and j.cmd == step["cmd"]
                               and j.status == "running"), None)
                if ya is not None:
                    self._log(job, f"[stack] {i}. {step['label']}: '{step['cmd']}' ya "
                                   f"estaba corriendo (job {ya.id}) — lo reuso")
                    continue

                try:
                    child = self.start(step["cmd"], step["opts"])
                except Exception as exc:
                    self._log(job, f"[stack] {i}. {step['label']}: NO se pudo lanzar — {exc}")
                    rc = 1
                    break
                job.children.append(child)
                self._log(job, f"[stack] {i}. {step['label']}: lanzado (job {child})")
            else:
                self._log(job, "[stack] todo arriba. Los logs de cada pieza estan en "
                               "su propio panel; 'Detener' aca corta el stack completo.")
        except Exception as exc:                    # pragma: no cover
            self._log(job, f"[stack] error inesperado: {exc}")
            rc = 1
        finally:
            with job.lock:
                job.status = "exited"
                job.returncode = rc
                job.finished_at = time.time()
            self._emit({"type": "job_exited", "id": job.id, "returncode": rc,
                        "finished_at": job.finished_at})

    def _reader(self, job: Job) -> None:
        try:
            for line in job.proc.stdout:            # type: ignore[union-attr]
                seq, text = job.append(line.rstrip("\n"))
                self._emit({"type": "line", "id": job.id, "seq": seq, "text": text})
        except Exception as exc:                    # pragma: no cover
            seq, text = job.append(f"[dashboard] error leyendo salida: {exc}")
            self._emit({"type": "line", "id": job.id, "seq": seq, "text": text})
        finally:
            rc = job.proc.wait()
            with job.lock:
                job.status = "exited"
                job.returncode = rc
                job.finished_at = time.time()
            seq, text = job.append(f"[dashboard] proceso terminado (codigo {rc})")
            self._emit({"type": "line", "id": job.id, "seq": seq, "text": text})
            self._emit({"type": "job_exited", "id": job.id, "returncode": rc,
                        "finished_at": job.finished_at})

    def stop(self, job_id: str, force: bool = False) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
        if not job:
            return
        # Parar un stack = cancelar lo que falte lanzar Y frenar lo ya lanzado.
        # Si no, "Detener" en el coordinador dejaba el SDK y RTAB-Map vivos.
        if job.kind == "stack":
            job.cancel.set()
            for child in list(job.children):
                self.stop(child, force=force)
            return
        if job.status != "running" or job.proc is None:
            return
        sig = signal.SIGKILL if force else signal.SIGINT
        try:
            os.killpg(os.getpgid(job.proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def stop_all(self, force: bool = False) -> int:
        with self.lock:
            stacks = [j.id for j in self.jobs.values()
                      if j.kind == "stack" and j.status == "running"]
            procs = [j.id for j in self.jobs.values()
                     if j.kind == "proc" and j.status == "running"]
        # primero los coordinadores, para que no sigan lanzando pasos nuevos
        for job_id in stacks:
            self.stop(job_id, force=force)
        for job_id in procs:
            self.stop(job_id, force=force)
        return len(procs)

    def clear(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status != "running":
                del self.jobs[job_id]
        self._emit({"type": "job_cleared", "id": job_id})

    def clear_finished(self) -> int:
        with self.lock:
            ids = [j.id for j in self.jobs.values() if j.status != "running"]
            for job_id in ids:
                del self.jobs[job_id]
        for job_id in ids:
            self._emit({"type": "job_cleared", "id": job_id})
        return len(ids)

    def snapshot(self, since: dict[str, int] | None = None) -> list[dict]:
        since = since or {}
        with self.lock:
            jobs = list(self.jobs.values())
        jobs.sort(key=lambda j: j.started_at)
        return [j.snapshot(since.get(j.id, -1)) for j in jobs]

    def full_log(self, job_id: str) -> str:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            return ""
        with job.lock:
            return "\n".join(text for _seq, text in job.lines)

    def shutdown(self) -> None:
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            job.cancel.set()
            if job.status == "running" and job.proc is not None:
                try:
                    os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass


JOBS = JobManager()


# ------------------------------------------------------------- utilidades ---

def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def sdk_status(base_url: str = "http://localhost:8000") -> dict:
    """Estado del SDK sin depender de requests (stdlib pura)."""
    host, _, port = base_url.split("//", 1)[1].partition(":")
    port_i = int(port or 80)
    if not _port_open(host, port_i):
        return {"up": False, "detail": "puerto cerrado"}
    for endpoint in ("/v2/data", "/data"):
        try:
            with urllib.request.urlopen(base_url + endpoint, timeout=1.5) as resp:
                if resp.status == 200:
                    try:
                        payload = json.loads(resp.read().decode("utf-8"))
                    except Exception:
                        payload = {}
                    battery = payload.get("battery")
                    signal_level = payload.get("signal_level")
                    return {"up": True, "detail": "telemetria OK",
                            "battery": battery, "signal_level": signal_level}
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return {"up": True, "detail": "responde, pero sin telemetria"}


def run_doctor() -> str:
    try:
        res = subprocess.run([SCRIPT_PATH, "doctor"], capture_output=True,
                             text=True, timeout=60)
        return res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return "doctor tardo mas de 60 s y se corto."
    except Exception as exc:
        return f"error corriendo doctor: {exc}"


def load_prefs() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_prefs(data: dict) -> None:
    try:
        STATE_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------------ HTML ----

INDEX_HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rover Launcher</title>
<style>
  :root{
    --bg:#0f1115; --panel:#161a20; --panel2:#12151a; --ink:#e6e6e6; --dim:#8a8f98;
    --line:#262b33; --accent:#8ab4f0; --accent-bg:#2b6cb0; --warn:#b7791f;
    --danger:#c53030; --ok:#7ee2a8; --armed-bg:#1c150a;
    color-scheme: dark;
  }
  *{box-sizing:border-box}
  body{background:var(--bg);color:var(--ink);font-family:ui-monospace,Menlo,Consolas,monospace;
       margin:0;padding:0 0 60px;font-size:13px}
  a{color:var(--accent)}

  header{position:sticky;top:0;z-index:20;background:rgba(15,17,21,.94);
         backdrop-filter:blur(6px);border-bottom:1px solid var(--line);
         padding:12px 24px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;letter-spacing:.02em}
  .chips{display:flex;gap:8px;flex-wrap:wrap;flex:1}
  .chip{font-size:11px;padding:3px 10px;border-radius:20px;border:1px solid var(--line);
        background:var(--panel);color:var(--dim);white-space:nowrap}
  .chip.ok{border-color:#1d5c39;color:var(--ok)}
  .chip.bad{border-color:#5c1d1d;color:#ff8080}
  .chip.busy{border-color:#1d3a5c;color:var(--accent)}

  button{background:var(--accent-bg);color:#fff;border:none;border-radius:6px;
         padding:8px 12px;font-size:12.5px;font-family:inherit;cursor:pointer}
  button:disabled{opacity:.4;cursor:not-allowed}
  button.small{padding:4px 10px;font-size:11px}
  button.ghost{background:#232833}
  button.ghost.active{background:#1d3a5c;color:var(--accent)}
  button.warn{background:var(--warn)}
  button.danger{background:var(--danger)}
  button.block{width:100%;margin-top:12px}

  .layout{display:grid;grid-template-columns:minmax(430px,640px) 1fr;gap:24px;
          align-items:start;padding:20px 24px}
  @media (max-width:1100px){.layout{grid-template-columns:1fr}}
  .col-forms{position:sticky;top:70px;max-height:calc(100vh - 90px);overflow-y:auto;padding-right:6px}

  h2.section{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#7ea6d6;
             margin:20px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
  h2.section:first-child{margin-top:0}

  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:10px;align-items:start}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
        padding:12px 14px;margin-bottom:10px;transition:background-color .15s,border-color .15s}
  .card.armed{background:var(--armed-bg);border-color:var(--warn)}
  .card>summary{cursor:pointer;list-style:none;display:flex;align-items:baseline;gap:8px}
  .card>summary::-webkit-details-marker{display:none}
  .card .name{font-size:12.5px;color:#9fd3ff;text-transform:uppercase;letter-spacing:.03em}
  .card .desc{color:var(--dim);font-size:11.5px;margin:6px 0 2px;line-height:1.45}
  .card .caret{color:#5a606b;font-size:10px}
  .card[open] .caret{transform:rotate(90deg);display:inline-block}

  label{display:block;font-size:11px;color:#b7bcc4;margin:9px 0 3px}
  label .hint{color:#5f6570;font-weight:normal}
  input[type=text],select{width:100%;background:#0f1115;border:1px solid #2a2f38;color:var(--ink);
        border-radius:6px;padding:6px 8px;font-family:inherit;font-size:12.5px}
  input[type=text]:focus,select:focus{outline:none;border-color:var(--accent)}
  .row-check{display:flex;align-items:center;gap:8px;margin-top:9px}
  .row-check input{width:auto}
  .row-check label{margin:0;font-size:12px}
  .go-warning{display:none;background:#3a2410;border:1px solid var(--warn);color:#f0c46a;
              font-size:11.5px;padding:8px;border-radius:6px;margin-top:8px;line-height:1.45}
  .hidden{display:none}

  #doctorOut{display:none;margin-top:10px;height:240px;overflow:auto;resize:vertical;
             background:#0d0f13;border:1px solid var(--line);border-radius:8px;padding:10px;
             font-size:11.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word}

  .panel{background:#0d0f13;border:1px solid var(--line);border-radius:10px;margin-bottom:12px;overflow:hidden}
  .panel.moves{border-color:#7a4a12}
  .panel-head{display:flex;justify-content:space-between;align-items:center;gap:10px;
              background:var(--panel);padding:7px 12px;cursor:pointer;user-select:none;flex-wrap:wrap}
  .panel-head .left{display:flex;align-items:center;gap:8px;min-width:0}
  .panel-head .caret{color:#666;font-size:11px;width:10px;flex:none}
  .panel-head .name{font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .panel-head .meta{font-size:10.5px;color:var(--dim)}
  .status{font-size:10.5px;padding:2px 8px;border-radius:10px;flex:none}
  .status.running{background:#134e2c;color:var(--ok)}
  .status.exited{background:#3a2410;color:#f0c46a}
  .status.exited.zero{background:#1b2a20;color:#8fbf9f}
  .actions{display:flex;gap:6px;flex:none;flex-wrap:wrap}
  .panel.collapsed .panel-body{display:none}
  .panel.collapsed .caret{transform:rotate(-90deg);display:inline-block}
  .panel-body{border-top:1px solid #1b1f27}
  .filterbar{display:flex;gap:8px;padding:6px 10px;border-bottom:1px solid #1b1f27;align-items:center}
  .filterbar input{flex:1;font-size:11.5px;padding:4px 8px}
  pre.log{margin:0;padding:10px 12px;height:300px;min-height:90px;max-height:75vh;overflow:auto;
          resize:vertical;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
  pre.log .hit{background:#3a3410}
  .empty{color:#565c66;font-size:13px;padding:24px;text-align:center;border:1px dashed var(--line);border-radius:10px}

  /* --- stacks (botones de un clic) --- */
  .stack{background:linear-gradient(180deg,#1a212b,#151920);border:1px solid #2f4a63;
         border-radius:12px;padding:14px 16px;margin-bottom:12px}
  .stack.armed{border-color:var(--warn);background:linear-gradient(180deg,#241a0e,#191410)}
  .stack .stitle{font-size:14px;color:#a9d6ff;letter-spacing:.02em;margin-bottom:4px}
  .stack .sdesc{color:var(--dim);font-size:11.5px;line-height:1.5;margin-bottom:4px}
  .stack .note{background:#131a22;border-left:2px solid #2f4a63;color:#96a4b4;
               font-size:11px;line-height:1.5;padding:7px 9px;border-radius:4px;margin-top:10px}
  .stack button.go{width:100%;margin-top:12px;padding:12px;font-size:13.5px;letter-spacing:.04em}
  .stack.armed button.go{background:var(--warn)}
  .plan{margin-top:10px;background:#0b0e12;border:1px solid var(--line);border-radius:8px;
        padding:8px 10px;font-size:11px;line-height:1.6;max-height:190px;overflow:auto;
        white-space:pre-wrap;word-break:break-all;color:#8fa3b8}
  .plan .step{color:#cfe3f7}
  .plan .wait{color:#6f7b88}
  .plan .err{color:#ff8080}
  .panel.stackjob{border-color:#2f4a63}
  .ansi-red{color:#ff8080}.ansi-green{color:var(--ok)}.ansi-yellow{color:#f0c46a}.ansi-blue{color:var(--accent)}
  .ansi-magenta{color:#e0a0ff}.ansi-cyan{color:#79d4d4}.ansi-white{color:#f2f4f7}
  .ansi-gray{color:var(--dim)}.ansi-black{color:#4a4f58}.ansi-bold{font-weight:700}
</style>
</head>
<body>
<header>
  <h1>Rover Launcher</h1>
  <div class="chips" id="chips"></div>
  <button class="danger" id="stopAll" onclick="stopAll()">PARAR TODO</button>
</header>

<div class="layout">
  <div class="col-forms">
    <h2 class="section">Un clic</h2>
    <div id="stacks"></div>

    <h2 class="section">Instalacion</h2>
    <div class="card" style="padding:12px 14px">
      <button class="ghost block" style="margin-top:0" onclick="checkDoctor()">Chequear instalacion (doctor)</button>
      <pre id="doctorOut"></pre>
    </div>
    <div id="forms"></div>
  </div>

  <div>
    <h2 class="section" style="display:flex;justify-content:space-between;align-items:center">
      <span>Procesos</span>
      <button class="small ghost" onclick="clearFinished()">Limpiar terminados</button>
    </h2>
    <div id="jobs"><div class="empty">Todavia no lanzaste nada.</div></div>
  </div>
</div>

<script>
// ---------------------------------------------------------------- helpers ---
function el(tag, attrs, ...kids){
  const e = document.createElement(tag);
  for (const k in (attrs||{})){
    if (k === "class") e.className = attrs[k];
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
  }
  for (const c of kids) if (c != null) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return e;
}

// El launcher (c_red/c_green/...) y console_report.py colorean con codigos
// ANSI pensados para una terminal; en el navegador se verian como basura si
// no se interpretan.
//
// La version anterior solo entendia la forma "1;31" (negrita+color en UN
// solo escape), que es la que emite el shell. console_report.py emite el
// atributo y el color por separado ("\x1b[1m\x1b[32m") y usa colores que no
// estaban en la tabla (gray 90, cyan 36, white 97), asi que sus filas salian
// sin color. Ahora se parsea SGR de verdad: se acumulan los parametros y se
// mantiene el estado (negrita + color de frente) hasta el reset.
const ANSI_FG = {30:"ansi-black",31:"ansi-red",32:"ansi-green",33:"ansi-yellow",
                 34:"ansi-blue",35:"ansi-magenta",36:"ansi-cyan",37:"ansi-white",
                 90:"ansi-gray",91:"ansi-red",92:"ansi-green",93:"ansi-yellow",
                 94:"ansi-blue",95:"ansi-magenta",96:"ansi-cyan",97:"ansi-white"};
const ANSI_RE = /\x1b\[([0-9;]*)m/g;
const stripAnsi = s => s.replace(/\x1b\[[0-9;]*m/g, "");

function ansiToFragment(text){
  const frag = document.createDocumentFragment();
  const re = new RegExp(ANSI_RE.source, "g");
  let last = 0, m, fg = null, bold = false;
  const push = s => {
    if (!s) return;
    if (!fg && !bold){ frag.appendChild(document.createTextNode(s)); return; }
    const sp = document.createElement("span");
    sp.className = [fg, bold ? "ansi-bold" : null].filter(Boolean).join(" ");
    sp.textContent = s;
    frag.appendChild(sp);
  };
  while ((m = re.exec(text)) !== null){
    push(text.slice(last, m.index));
    last = re.lastIndex;
    // "\x1b[m" vacio == reset, igual que "\x1b[0m"
    for (const raw of (m[1] === "" ? "0" : m[1]).split(";")){
      const n = parseInt(raw || "0", 10);
      if (n === 0){ fg = null; bold = false; }
      else if (n === 1) bold = true;
      else if (n === 22) bold = false;
      else if (n === 39) fg = null;
      else if (ANSI_FG[n]) fg = ANSI_FG[n];
    }
  }
  push(text.slice(last));
  return frag;
}

function fmtElapsed(sec){
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec/3600), m = Math.floor(sec%3600/60), s = sec%60;
  return h ? `${h}h${String(m).padStart(2,"0")}m` : (m ? `${m}m${String(s).padStart(2,"0")}s` : `${s}s`);
}

// ------------------------------------------------------------ formularios ---
let SPEC = [], PREFS = {};
const formState = {};   // cmd -> {get(), btn, card, fields}

function buildCard(spec){
  const abrir = ["sdk", "genie-bridge", "indoor-bridge", "mapping-ros2"].includes(spec.cmd);
  const card = el("details", {class:"card", open: abrir ? "" : null});
  const sum = el("summary", {},
    el("span", {class:"caret"}, "▸"),
    el("span", {class:"name"}, spec.title));
  card.appendChild(sum);
  if (spec.desc) card.appendChild(el("div", {class:"desc"}, spec.desc));

  const getters = {}, wrappers = {}, elements = {};
  const saved = PREFS[spec.cmd] || {};

  for (const f of spec.fields){
    const wrap = el("div", {});
    if (f.type === "bool"){
      const row = el("div", {class:"row-check"});
      const cb = el("input", {type:"checkbox", id:`${spec.cmd}-${f.key}`});
      cb.checked = !!saved[f.key];
      row.appendChild(cb);
      row.appendChild(el("label", {for:`${spec.cmd}-${f.key}`}, f.label));
      wrap.appendChild(row);
      if (f.help) wrap.appendChild(el("div", {class:"desc"}, f.help));
      getters[f.key] = () => cb.checked;
      elements[f.key] = cb;
      cb.addEventListener("change", schedulePrefs);
    } else if (f.type === "select"){
      wrap.appendChild(el("label", {}, f.label, f.help ? el("span",{class:"hint"}, "  — "+f.help) : null));
      const sel = el("select", {id:`${spec.cmd}-${f.key}`});
      for (const o of f.options) sel.appendChild(el("option", {value:o}, o === "" ? "(default)" : o));
      if (saved[f.key] !== undefined) sel.value = saved[f.key];
      wrap.appendChild(sel);
      getters[f.key] = () => sel.value;
      elements[f.key] = sel;
      sel.addEventListener("change", () => { schedulePrefs(); applyLevelVisibility(spec); });
    } else {
      wrap.appendChild(el("label", {}, f.label, f.help ? el("span",{class:"hint"}, "  — "+f.help) : null));
      const inp = el("input", {type:"text", id:`${spec.cmd}-${f.key}`, placeholder:f.placeholder||""});
      if (saved[f.key] !== undefined) inp.value = saved[f.key];
      wrap.appendChild(inp);
      getters[f.key] = () => inp.value.trim();
      elements[f.key] = inp;
      inp.addEventListener("input", schedulePrefs);
    }
    wrappers[f.key] = wrap;
    card.appendChild(wrap);
  }

  let goCb = null;
  if (spec.go){
    const row = el("div", {class:"row-check"});
    goCb = el("input", {type:"checkbox", id:`${spec.cmd}-go`});
    row.appendChild(goCb);
    row.appendChild(el("label", {for:`${spec.cmd}-go`}, "--go (MODO REAL: el rover se mueve)"));
    card.appendChild(row);
    const warn = el("div", {class:"go-warning"},
      "Modo real. Confirma que el area esta despejada y tene a mano PARAR TODO antes de lanzar.");
    card.appendChild(warn);
    goCb.addEventListener("change", () => {
      warn.style.display = goCb.checked ? "block" : "none";
      card.classList.toggle("armed", goCb.checked);
    });
  }

  const btn = el("button", {class:"block", onclick: () => launch(spec)}, "Lanzar");
  card.appendChild(btn);

  formState[spec.cmd] = {
    spec, card, btn, wrappers, elements,
    get(){
      const o = {};
      for (const k in getters){ const v = getters[k](); if (v !== "" && v !== false) o[k] = v; }
      if (spec.go) o.go = !!(goCb && goCb.checked);
      return o;
    },
  };
  applyLevelVisibility(spec);
  return card;
}

// traversability: cada nivel usa un subconjunto de flags. Mostrar solo esas
// evita mandar (y que el launcher rechace) flags que no aplican.
function applyLevelVisibility(spec){
  if (!spec.level_key) return;
  const st = formState[spec.cmd];
  if (!st) return;
  const sel = st.elements[spec.level_key];
  if (!sel) return;
  const level = sel.value;
  const globales = (spec.global_fields_by_level||{})[level] || spec.global_fields || [];
  const visibles = new Set([spec.level_key, ...globales, ...((spec.level_fields||{})[level]||[])]);
  for (const key in st.wrappers) st.wrappers[key].classList.toggle("hidden", !visibles.has(key));
  const mueve = (spec.moves_levels||[]).includes(level);
  st.card.classList.toggle("armed", mueve);
}

function renderForms(){
  const root = document.getElementById("forms");
  root.innerHTML = "";
  const groups = [];
  for (const s of SPEC){ if (!groups.includes(s.group)) groups.push(s.group); }
  for (const g of groups){
    root.appendChild(el("h2", {class:"section"}, g));
    const grid = el("div", {class:"cards"});
    for (const s of SPEC.filter(x => x.group === g)) grid.appendChild(buildCard(s));
    root.appendChild(grid);
  }
}

// ----------------------------------------------------------------- stacks ---
// Un stack es un boton que lanza varios comandos. El formulario, la
// condicion de visibilidad de cada campo y la vista previa salen del mismo
// spec que usa el backend, asi que no hay dos listas que mantener a mano.
let STACKS = [];
const stackState = {};

function condOk(cond, form){
  if (!cond) return true;
  const v = form[cond.field];
  if (cond.in) return cond.in.map(String).includes(String(v));
  if (cond.equals !== undefined) return String(v) === String(cond.equals);
  if (cond.truthy) return !!v && v !== "0" && v !== "false" && v !== "no";
  return true;
}

// Campo generico: devuelve el nodo y como leerlo. Soporta `default` (valor
// inicial) y opciones con etiqueta: ["valor", "Etiqueta linda"].
function makeField(prefix, f, saved, onChange){
  const id = `${prefix}-${f.key}`;
  const wrap = el("div", {});
  let get;
  if (f.type === "bool"){
    const row = el("div", {class:"row-check"});
    const cb = el("input", {type:"checkbox", id});
    cb.checked = saved[f.key] !== undefined ? !!saved[f.key] : !!f.default;
    row.appendChild(cb); row.appendChild(el("label", {for:id}, f.label));
    wrap.appendChild(row);
    if (f.help) wrap.appendChild(el("div", {class:"desc"}, f.help));
    cb.addEventListener("change", onChange);
    get = () => cb.checked;
  } else if (f.type === "select"){
    wrap.appendChild(el("label", {}, f.label));
    const sel = el("select", {id});
    for (const o of f.options){
      const val = Array.isArray(o) ? o[0] : o;
      const txt = Array.isArray(o) ? o[1] : (o === "" ? "(default)" : o);
      sel.appendChild(el("option", {value:val}, txt));
    }
    sel.value = saved[f.key] !== undefined ? saved[f.key] : (f.default ?? sel.value);
    wrap.appendChild(sel);
    if (f.help) wrap.appendChild(el("div", {class:"desc"}, f.help));
    sel.addEventListener("change", onChange);
    get = () => sel.value;
  } else {
    wrap.appendChild(el("label", {}, f.label, f.help ? el("span",{class:"hint"}, "  — "+f.help) : null));
    const inp = el("input", {type:"text", id, placeholder:f.placeholder||""});
    inp.value = saved[f.key] !== undefined ? saved[f.key] : (f.default ?? "");
    wrap.appendChild(inp);
    inp.addEventListener("input", onChange);
    get = () => inp.value.trim();
  }
  return {wrap, get};
}

function buildStackCard(stack){
  const card = el("div", {class:"stack"});
  card.appendChild(el("div", {class:"stitle"}, stack.title));
  if (stack.desc) card.appendChild(el("div", {class:"sdesc"}, stack.desc));

  const saved = PREFS["stack:" + stack.id] || {};
  const getters = {}, wrappers = {};
  const onChange = () => { schedulePrefs(); syncStack(stack); };

  for (const f of stack.fields){
    const {wrap, get} = makeField("st-" + stack.id, f, saved, onChange);
    getters[f.key] = get; wrappers[f.key] = wrap;
    card.appendChild(wrap);
  }

  let goCb = null;
  if (stack.go){
    const row = el("div", {class:"row-check"});
    goCb = el("input", {type:"checkbox", id:`st-${stack.id}-go`});
    goCb.checked = !!saved.go;
    row.appendChild(goCb);
    row.appendChild(el("label", {for:`st-${stack.id}-go`}, "--go (MODO REAL: el rover se mueve)"));
    card.appendChild(row);
    goCb.addEventListener("change", onChange);
  }

  const note = el("div", {class:"note"});
  note.style.display = "none";
  card.appendChild(note);

  const plan = el("div", {class:"plan"}, "…");
  card.appendChild(plan);

  const btn = el("button", {class:"go", onclick: () => launchStack(stack)}, stack.button || "LANZAR");
  card.appendChild(btn);

  stackState[stack.id] = {
    stack, card, btn, plan, note, wrappers, goCb,
    get(){
      const o = {};
      for (const k in getters) o[k] = getters[k]();
      if (stack.go) o.go = !!(goCb && goCb.checked);
      return o;
    },
  };
  syncStack(stack);
  return card;
}

// Muestra solo los campos que aplican al modo elegido, pinta la tarjeta si
// va a mover el rover, y pide al backend la vista previa del plan.
function syncStack(stack){
  const st = stackState[stack.id];
  if (!st) return;
  const form = st.get();
  for (const f of stack.fields)
    if (st.wrappers[f.key])
      st.wrappers[f.key].classList.toggle("hidden", !condOk(f.show_when, form));

  st.card.classList.toggle("armed", !!form.go);

  const nota = (stack.notes || {})[String(form[ (stack.fields.find(f=>f.type==="select")||{}).key ])];
  if (nota){ st.note.textContent = nota; st.note.style.display = "block"; }
  else st.note.style.display = "none";

  schedulePlan(stack);
}

const planTimers = {};
function schedulePlan(stack){
  clearTimeout(planTimers[stack.id]);
  planTimers[stack.id] = setTimeout(() => refreshPlan(stack), 250);
}

async function refreshPlan(stack){
  const st = stackState[stack.id];
  if (!st) return;
  let data;
  try{
    const res = await fetch("/api/stack-plan", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({stack: stack.id, form: st.get()})});
    data = await res.json();
  }catch(e){ return; }
  st.plan.textContent = "";
  const levantando = st.btn.textContent === "Levantando...";
  if (data.error){
    st.plan.appendChild(el("div", {class:"err"}, "⚠ " + data.error));
    st.btn.disabled = true;
    return;
  }
  if (!levantando) st.btn.disabled = false;
  data.steps.forEach((step, i) => {
    const espera = [step.wait_port ? `espera ${step.wait_port}` : null,
                    step.wait_s ? `+${step.wait_s}s` : null].filter(Boolean).join(", ");
    st.plan.appendChild(el("div", {class:"step"}, `${i+1}. ${step.label}`));
    st.plan.appendChild(el("div", {}, "   " + step.line));
    if (espera) st.plan.appendChild(el("div", {class:"wait"}, "   (" + espera + ")"));
  });
  if (stack.after) st.plan.appendChild(el("div", {class:"wait"}, "\n" + stack.after));
}

async function launchStack(stack){
  const st = stackState[stack.id];
  const form = st.get();
  if (form.go && !confirm("Esto MUEVE el rover de verdad.\n\nArea despejada? Tenes a mano PARAR TODO?")) return;
  st.btn.disabled = true; const antes = st.btn.textContent; st.btn.textContent = "Levantando...";
  try{
    const res = await fetch("/api/run-stack", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({stack: stack.id, form})});
    if (!res.ok){ alert("Error: " + await res.text()); st.btn.disabled = false; st.btn.textContent = antes; return; }
    await refreshJobs();
  }catch(e){
    alert("No pude hablar con el dashboard: " + e);
    st.btn.disabled = false; st.btn.textContent = antes;
  }
}

function renderStacks(){
  const root = document.getElementById("stacks");
  root.innerHTML = "";
  for (const st of STACKS) root.appendChild(buildStackCard(st));
}

let prefsTimer = null;
function schedulePrefs(){
  clearTimeout(prefsTimer);
  prefsTimer = setTimeout(() => {
    const data = {};
    for (const cmd in formState) data[cmd] = formState[cmd].get();
    for (const id in stackState) data["stack:" + id] = stackState[id].get();
    fetch("/api/prefs", {method:"POST", headers:{"Content-Type":"application/json"},
                         body: JSON.stringify(data)}).catch(()=>{});
  }, 600);
}

async function launch(spec){
  const st = formState[spec.cmd];
  const opts = st.get();
  const mueve = spec.moves === "go" ? !!opts.go
              : (spec.moves_levels||[]).includes(opts[spec.level_key]);
  if (mueve && !confirm("Esto MUEVE el rover de verdad.\n\nArea despejada? Tenes a mano PARAR TODO?")) return;
  st.btn.disabled = true; st.btn.textContent = "Lanzando...";
  try{
    const res = await fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"},
                                         body: JSON.stringify({cmd: spec.cmd, opts})});
    if (!res.ok){ alert("Error: " + await res.text()); st.btn.disabled = false; st.btn.textContent = "Lanzar"; return; }
    await refreshJobs();
  }catch(e){
    alert("No pude hablar con el dashboard: " + e);
    st.btn.disabled = false; st.btn.textContent = "Lanzar";
  }
}

// ----------------------------------------------------------------- paneles ---
const panels = {};   // id -> {lines, collapsed, autoscroll, filter, el, pre, ...}

function ensurePanel(job, container){
  if (panels[job.id]) return panels[job.id];
  const panel = el("div", {class:"panel" + (job.moves ? " moves" : "")
                          + (job.kind === "stack" ? " stackjob" : "")});
  const state = {seq:-1, collapsed:false, autoscroll:true, filter:"", el:panel, job};

  const head = el("div", {class:"panel-head", onclick:(ev)=>{
    if (ev.target.closest("button") || ev.target.closest("input")) return;
    state.collapsed = !state.collapsed;
    panel.classList.toggle("collapsed", state.collapsed);
  }});
  const left = el("div", {class:"left"},
    el("span", {class:"caret"}, "▾"),
    el("span", {class:"name"}, `${job.kind === "stack" ? "◆ " + (job.title || job.cmd) : job.cmd}  #${job.id}`),
    el("span", {class:"meta", id:`meta-${job.id}`}, ""));
  const statusEl = el("span", {class:"status running"}, "corriendo");
  const stopBtn = el("button", {class:"small warn", onclick:ev=>{ev.stopPropagation();
    stopBtn.disabled = killBtn.disabled = true; stopBtn.textContent="Deteniendo..."; stopJob(job.id,false);}}, "Detener");
  const killBtn = el("button", {class:"small danger", onclick:ev=>{ev.stopPropagation();
    stopBtn.disabled = killBtn.disabled = true; killBtn.textContent="Matando..."; stopJob(job.id,true);}}, "Forzar");
  const autoBtn = el("button", {class:"small ghost active", onclick:ev=>{ev.stopPropagation();
    state.autoscroll = !state.autoscroll;
    autoBtn.textContent = state.autoscroll ? "Auto-scroll: ON" : "Auto-scroll: OFF";
    autoBtn.classList.toggle("active", state.autoscroll);
    if (state.autoscroll) state.pre.scrollTop = state.pre.scrollHeight;
  }}, "Auto-scroll: ON");
  const copyBtn = el("button", {class:"small ghost", onclick:ev=>{ev.stopPropagation(); copyLog(job.id);}}, "Copiar");
  const dlBtn = el("a", {class:"small", href:`/api/log?job=${job.id}`, download:`${job.cmd}-${job.id}.log`,
                         style:"text-decoration:none"},
                     el("button", {class:"small ghost", onclick:ev=>ev.stopPropagation()}, "Descargar"));
  const clearBtn = el("button", {class:"small ghost", onclick:ev=>{ev.stopPropagation(); clearJob(job.id);}}, "Quitar");

  const actions = el("div", {class:"actions"}, autoBtn, copyBtn, dlBtn, stopBtn, killBtn, clearBtn);
  head.appendChild(left);
  head.appendChild(el("div", {class:"actions"}, statusEl, actions));

  const filterInput = el("input", {type:"text", placeholder:"filtrar lineas (vacio = todas)"});
  filterInput.addEventListener("input", () => { state.filter = filterInput.value.toLowerCase(); repaint(state); });
  // (el filtro compara contra el texto SIN codigos ANSI: ver repaint)
  const body = el("div", {class:"panel-body"},
    el("div", {class:"filterbar"}, filterInput),
    el("pre", {class:"log", id:`log-${job.id}`}));

  panel.appendChild(head); panel.appendChild(body);
  container.appendChild(panel);

  Object.assign(state, {pre: body.querySelector("pre"), statusEl, stopBtn, killBtn,
                        autoBtn, metaEl: left.querySelector(`#meta-${job.id}`), all: []});
  panels[job.id] = state;
  return state;
}

// `entries` son pares [seq, texto]. Se descarta lo que ya se pinto: los
// refrescos pueden cruzarse y sin esto la misma linea aparecia dos veces.
function appendLines(state, entries){
  const nuevas = [];
  for (const [seq, text] of entries){
    if (seq <= state.seq) continue;
    state.seq = seq;
    nuevas.push(text);
  }
  const lines = nuevas;
  if (!lines.length) return;
  state.all.push(...lines);
  const nearBottom = state.pre.scrollTop + state.pre.clientHeight >= state.pre.scrollHeight - 24;
  if (state.filter){ repaint(state); return; }
  const text = (state.pre.childNodes.length ? "\n" : "") + lines.join("\n");
  state.pre.appendChild(ansiToFragment(text));
  if (state.autoscroll && nearBottom) state.pre.scrollTop = state.pre.scrollHeight;
}

function repaint(state){
  const shown = state.filter
    ? state.all.filter(l => stripAnsi(l).toLowerCase().includes(state.filter))
    : state.all;
  state.pre.textContent = "";
  state.pre.appendChild(ansiToFragment(shown.join("\n")));
  if (state.autoscroll) state.pre.scrollTop = state.pre.scrollHeight;
}

function paintStatus(state, job){
  if (job.status === "running"){
    state.statusEl.textContent = "corriendo";
    state.statusEl.className = "status running";
    state.stopBtn.style.display = ""; state.killBtn.style.display = "";
    state.stopBtn.disabled = state.killBtn.disabled = false;
    state.stopBtn.textContent = "Detener"; state.killBtn.textContent = "Forzar";
  } else {
    const zero = job.returncode === 0;
    state.statusEl.textContent = zero ? "termino (0)" : `salio (${job.returncode})`;
    state.statusEl.className = "status exited" + (zero ? " zero" : "");
    state.stopBtn.style.display = "none"; state.killBtn.style.display = "none";
  }
}

function tickMeta(){
  const now = Date.now()/1000;
  for (const id in panels){
    const st = panels[id], j = st.job;
    const end = j.status === "running" ? now : (j.finished_at || now);
    st.metaEl.textContent = "· " + fmtElapsed(end - j.started_at);
  }
  renderChips();
}

async function stopJob(id, force){ await fetch(`/api/stop?job=${encodeURIComponent(id)}&force=${force?1:0}`, {method:"POST"}); }
async function clearJob(id){
  await fetch(`/api/clear?job=${encodeURIComponent(id)}`, {method:"POST"});
  if (panels[id]){ panels[id].el.remove(); delete panels[id]; }
  refreshJobs();
}
async function clearFinished(){ await fetch("/api/clear-finished", {method:"POST"}); refreshJobs(); }
async function stopAll(){
  if (!confirm("Frenar TODOS los procesos que esten corriendo?")) return;
  await fetch("/api/stop-all", {method:"POST"});
}

function copyLog(id){
  const pre = document.getElementById(`log-${id}`);
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).catch(()=>{
    const r = document.createRange(); r.selectNodeContents(pre);
    const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
  });
}

// ------------------------------------------------------------------ estado ---
let STATUS = {}, TRANSPORT = "…";

function renderChips(){
  const chips = document.getElementById("chips");
  const running = Object.values(panels).filter(p => p.job.status === "running");
  const items = [];
  const sdk = STATUS.sdk || {};
  items.push([`SDK ${sdk.up ? "conectado" : "caido"}`, sdk.up ? "ok" : "bad"]);
  if (sdk.battery !== undefined && sdk.battery !== null) items.push([`bateria ${sdk.battery}%`, ""]);
  if (sdk.signal_level !== undefined && sdk.signal_level !== null) items.push([`senal ${sdk.signal_level}`, ""]);
  items.push([`${running.length} corriendo`, running.length ? "busy" : ""]);
  if (running.some(p => p.job.moves)) items.push(["MODO REAL activo", "bad"]);
  items.push([`logs: ${TRANSPORT}`, ""]);
  chips.innerHTML = "";
  for (const [txt, cls] of items) chips.appendChild(el("span", {class:"chip " + cls}, txt));
  document.getElementById("stopAll").disabled = running.length === 0;
}

function syncLaunchButtons(){
  const vivos = Object.values(panels).filter(p=>p.job.status==="running");
  const runningCmds = new Set(vivos.filter(p=>p.job.kind !== "stack").map(p=>p.job.cmd));
  const runningStacks = new Set(vivos.filter(p=>p.job.kind === "stack").map(p=>p.job.cmd));
  for (const id in stackState){
    const st = stackState[id];
    if (runningStacks.has(id)){ st.btn.disabled = true; st.btn.textContent = "Levantando..."; }
    else if (st.btn.textContent === "Levantando..."){
      st.btn.disabled = false; st.btn.textContent = st.stack.button || "LANZAR";
    }
  }
  for (const cmd in formState){
    const st = formState[cmd];
    if (runningCmds.has(cmd)){ st.btn.disabled = true; st.btn.textContent = "Ya esta corriendo"; }
    else { st.btn.disabled = false; st.btn.textContent = "Lanzar"; }
  }
}

let refreshInFlight = null;
function refreshJobs(){
  // Un solo refresco a la vez: el POST de lanzamiento y el evento WebSocket
  // disparaban dos en paralelo y los dos pedian el log desde el mismo punto.
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = doRefreshJobs().finally(() => { refreshInFlight = null; });
  return refreshInFlight;
}

async function doRefreshJobs(){
  const since = {};
  for (const id in panels) since[id] = panels[id].seq;
  const res = await fetch("/api/jobs?since=" + encodeURIComponent(JSON.stringify(since)));
  const jobs = await res.json();
  const root = document.getElementById("jobs");
  if (!jobs.length){
    for (const id in panels){ panels[id].el.remove(); delete panels[id]; }
    root.innerHTML = '<div class="empty">Todavia no lanzaste nada.</div>';
    syncLaunchButtons(); renderChips();
    return;
  }
  if (root.querySelector(".empty")) root.innerHTML = "";
  const seen = new Set();
  for (const j of jobs){
    seen.add(j.id);
    const st = ensurePanel(j, root);
    st.job = j;
    if (j.lines && j.lines.length) appendLines(st, j.lines);
    paintStatus(st, j);
  }
  for (const id in panels) if (!seen.has(id)){ panels[id].el.remove(); delete panels[id]; }
  syncLaunchButtons(); renderChips();
}

async function refreshStatus(){
  try{ STATUS = await (await fetch("/api/status")).json(); }catch(e){ STATUS = {}; }
  renderChips();
}

async function checkDoctor(){
  const out = document.getElementById("doctorOut");
  out.style.display = "block"; out.textContent = "corriendo...";
  const text = await (await fetch("/api/doctor")).text();
  out.textContent = ""; out.appendChild(ansiToFragment(text));
}

// -------------------------------------------------- transporte de eventos ---
// WebSocket cuando el backend es FastAPI; polling incremental si no.
function connectWS(){
  let ws;
  try{ ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws"); }
  catch(e){ startPolling(); return; }
  let opened = false;
  ws.onopen = () => { opened = true; TRANSPORT = "websocket"; refreshJobs(); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "line"){
      const st = panels[msg.id];
      if (!st){ refreshJobs(); return; }
      appendLines(st, [[msg.seq, msg.text]]);
    } else if (msg.type === "job_exited"){
      const st = panels[msg.id];
      if (st){ st.job.status = "exited"; st.job.returncode = msg.returncode;
               st.job.finished_at = msg.finished_at; paintStatus(st, st.job); syncLaunchButtons(); renderChips(); }
    } else {
      refreshJobs();
    }
  };
  ws.onclose = () => { if (opened) { TRANSPORT = "reconectando"; setTimeout(connectWS, 1500); } else startPolling(); };
  ws.onerror = () => { try{ ws.close(); }catch(e){} };
}

let pollTimer = null;
function startPolling(){
  if (pollTimer) return;
  TRANSPORT = "polling";
  pollTimer = setInterval(refreshJobs, 1200);
  refreshJobs();
}

// ------------------------------------------------------------------- init ---
(async function init(){
  SPEC = await (await fetch("/api/spec")).json();
  try{ STACKS = await (await fetch("/api/stacks")).json(); }catch(e){ STACKS = []; }
  try{ PREFS = await (await fetch("/api/prefs")).json(); }catch(e){ PREFS = {}; }
  renderStacks();
  renderForms();
  await refreshStatus();
  await refreshJobs();
  connectWS();
  setInterval(tickMeta, 1000);
  setInterval(refreshStatus, 5000);
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------- API compartida ---
# Las dos implementaciones (FastAPI y http.server) llaman a estas funciones,
# asi que la logica de cada endpoint existe UNA sola vez.

def api_spec() -> list[dict]:
    return COMMANDS


def api_jobs(since_raw: str | None) -> list[dict]:
    since: dict[str, int] = {}
    if since_raw:
        try:
            parsed = json.loads(since_raw)
            if isinstance(parsed, dict):
                since = {str(k): int(v) for k, v in parsed.items()}
        except (ValueError, TypeError):
            since = {}
    return JOBS.snapshot(since)


def api_run(payload: dict) -> dict:
    cmd = str(payload.get("cmd", ""))
    opts = payload.get("opts") or {}
    if not isinstance(opts, dict):
        raise ValueError("opts tiene que ser un objeto")
    return {"job": JOBS.start(cmd, opts)}


def api_stacks() -> list[dict]:
    return STACKS


def api_run_stack(payload: dict) -> dict:
    stack_id = str(payload.get("stack", ""))
    form = payload.get("form") or {}
    if not isinstance(form, dict):
        raise ValueError("form tiene que ser un objeto")
    return {"job": JOBS.start_stack(stack_id, form)}


def api_stack_plan(payload: dict) -> dict:
    """Vista previa: que se va a lanzar, en que orden y con que flags.
    Devuelve 200 con {"error": ...} en vez de 400 porque el front la pide en
    cada tecla y un formulario a medio llenar no es un error del usuario."""
    stack_id = str(payload.get("stack", ""))
    form = payload.get("form") or {}
    try:
        plan = build_stack_plan(stack_id, form if isinstance(form, dict) else {})
    except ValueError as exc:
        return {"error": str(exc), "steps": []}
    return {"steps": [
        {"label": p["label"], "cmd": p["cmd"], "wait_port": p["wait_port"],
         "wait_s": p["wait_s"],
         "line": " ".join(shlex.quote(a) for a in p["argv"])}
        for p in plan]}


def api_status() -> dict:
    running = [j for j in JOBS.snapshot() if j["status"] == "running"]
    return {
        "sdk": sdk_status(),
        "running": len(running),
        "moving": any(j["moves"] for j in running),
        "script": SCRIPT_PATH,
        "cwd": os.getcwd(),
    }


# --------------------------------------------------------- backend FastAPI --

def run_fastapi(host: str, port: int) -> None:
    import asyncio

    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    import uvicorn

    loop = None                    # asyncio.AbstractEventLoop, se setea al arrancar
    clients: set = set()           # una cola por cliente WebSocket

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal loop
        loop = asyncio.get_running_loop()
        yield
        JOBS.unsubscribe(on_event)

    app = FastAPI(title="Rover Launcher", docs_url=None, redoc_url=None, lifespan=lifespan)

    def on_event(event: dict) -> None:
        """Se llama desde el hilo lector de cada job (no del loop asyncio)."""
        if loop is None:
            return
        for q in list(clients):
            loop.call_soon_threadsafe(q.put_nowait, event)

    JOBS.subscribe(on_event)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(INDEX_HTML)

    @app.get("/favicon.ico")
    async def favicon():
        return PlainTextResponse("", status_code=204)

    @app.get("/api/spec")
    async def spec():
        return JSONResponse(api_spec())

    @app.get("/api/jobs")
    async def jobs(since: str | None = None):
        return JSONResponse(api_jobs(since))

    @app.get("/api/stacks")
    async def stacks():
        return JSONResponse(api_stacks())

    @app.post("/api/run-stack")
    async def run_stack(request: Request):
        try:
            return JSONResponse(api_run_stack(await request.json()))
        except ValueError as exc:
            return PlainTextResponse(str(exc), status_code=400)

    @app.post("/api/stack-plan")
    async def stack_plan(request: Request):
        return JSONResponse(api_stack_plan(await request.json()))

    @app.get("/api/status")
    async def status():
        return JSONResponse(await asyncio.to_thread(api_status))

    @app.get("/api/doctor", response_class=PlainTextResponse)
    async def doctor():
        return PlainTextResponse(await asyncio.to_thread(run_doctor))

    @app.get("/api/log", response_class=PlainTextResponse)
    async def log(job: str):
        return PlainTextResponse(JOBS.full_log(job))

    @app.get("/api/prefs")
    async def prefs_get():
        return JSONResponse(load_prefs())

    @app.post("/api/prefs")
    async def prefs_set(request: Request):
        save_prefs(await request.json())
        return JSONResponse({"ok": True})

    @app.post("/api/run")
    async def run(request: Request):
        try:
            return JSONResponse(api_run(await request.json()))
        except ValueError as exc:
            return PlainTextResponse(str(exc), status_code=400)

    @app.post("/api/stop")
    async def stop(job: str, force: int = 0):
        JOBS.stop(job, force=bool(force))
        return JSONResponse({"ok": True})

    @app.post("/api/stop-all")
    async def stop_all(force: int = 0):
        return JSONResponse({"stopped": JOBS.stop_all(force=bool(force))})

    @app.post("/api/clear")
    async def clear(job: str):
        JOBS.clear(job)
        return JSONResponse({"ok": True})

    @app.post("/api/clear-finished")
    async def clear_finished():
        return JSONResponse({"cleared": JOBS.clear_finished()})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=4000)
        clients.add(q)
        try:
            while True:
                event = await q.get()
                await ws.send_text(json.dumps(event))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            clients.discard(q)

    print(f"Dashboard (FastAPI + WebSocket) en http://localhost:{port}   Ctrl+C para cortar")
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------- backend stdlib (fallback)

def run_stdlib(host: str, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):   # silenciar el log por default
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj).encode("utf-8"), "application/json", code)

        def _text(self, text, code=200):
            self._send(text.encode("utf-8"), "text/plain; charset=utf-8", code)

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            path = parsed.path
            if path == "/":
                self._send(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(b"", "image/x-icon", 204)
            elif path == "/api/spec":
                self._json(api_spec())
            elif path == "/api/jobs":
                self._json(api_jobs((qs.get("since") or [None])[0]))
            elif path == "/api/stacks":
                self._json(api_stacks())
            elif path == "/api/status":
                self._json(api_status())
            elif path == "/api/doctor":
                self._text(run_doctor())
            elif path == "/api/log":
                self._text(JOBS.full_log((qs.get("job") or [""])[0]))
            elif path == "/api/prefs":
                self._json(load_prefs())
            else:
                self._text("not found", 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            path = parsed.path
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if path == "/api/run":
                try:
                    self._json(api_run(json.loads(raw or b"{}")))
                except ValueError as exc:
                    self._text(str(exc), 400)
                except Exception as exc:
                    self._text(f"error: {exc}", 500)
            elif path == "/api/run-stack":
                try:
                    self._json(api_run_stack(json.loads(raw or b"{}")))
                except ValueError as exc:
                    self._text(str(exc), 400)
                except Exception as exc:
                    self._text(f"error: {exc}", 500)
            elif path == "/api/stack-plan":
                try:
                    self._json(api_stack_plan(json.loads(raw or b"{}")))
                except Exception as exc:
                    self._json({"error": str(exc), "steps": []})
            elif path == "/api/prefs":
                try:
                    save_prefs(json.loads(raw or b"{}"))
                except ValueError:
                    pass
                self._json({"ok": True})
            elif path == "/api/stop":
                JOBS.stop((qs.get("job") or [""])[0], force=(qs.get("force") or ["0"])[0] == "1")
                self._json({"ok": True})
            elif path == "/api/stop-all":
                self._json({"stopped": JOBS.stop_all()})
            elif path == "/api/clear":
                JOBS.clear((qs.get("job") or [""])[0])
                self._json({"ok": True})
            elif path == "/api/clear-finished":
                self._json({"cleared": JOBS.clear_finished()})
            else:
                self._text("not found", 404)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard (libreria estandar, polling) en http://localhost:{port}   Ctrl+C para cortar")
    print("  tip: pip install -r requirements-dashboard.txt para logs en vivo por WebSocket")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ------------------------------------------------------------------- main ---

def main() -> None:
    global SCRIPT_PATH, STATE_PATH

    ap = argparse.ArgumentParser(description="Panel de control local de La Rovernetta")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1",
                    help="por defecto solo localhost. Cambialo SOLO si sabes lo que haces: "
                         "este servidor ejecuta comandos de tu sistema.")
    ap.add_argument("--script", default="./rover_launch.sh")
    ap.add_argument("--stdlib", action="store_true",
                    help="forzar el backend de libreria estandar aunque haya fastapi")
    args = ap.parse_args()

    SCRIPT_PATH = os.path.abspath(args.script)
    STATE_PATH = Path(os.path.dirname(SCRIPT_PATH) or ".") / STATE_FILE

    if not os.path.isfile(SCRIPT_PATH):
        raise SystemExit(
            f"No encuentro {SCRIPT_PATH}. Corre este server parado en la misma carpeta que "
            f"rover_launch.sh, o pasale --script /ruta/completa/rover_launch.sh")
    if not os.access(SCRIPT_PATH, os.X_OK):
        raise SystemExit(f"{SCRIPT_PATH} no tiene permiso de ejecucion — corre: chmod +x {SCRIPT_PATH}")

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  AVISO: escuchando en {args.host}, no solo en localhost.\n"
              f"  Cualquiera que llegue a este puerto puede mover tu rover.\n")

    use_fastapi = not args.stdlib
    if use_fastapi:
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError:
            use_fastapi = False

    try:
        if use_fastapi:
            run_fastapi(args.host, args.port)
        else:
            run_stdlib(args.host, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        JOBS.shutdown()


if __name__ == "__main__":
    main()
