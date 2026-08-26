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

def _txt(key, label, placeholder="", help=""):
    return {"key": key, "label": label, "type": "text", "placeholder": placeholder, "help": help}


def _sel(key, label, options, help=""):
    return {"key": key, "label": label, "type": "select", "options": options, "help": help}


def _bool(key, label, help=""):
    return {"key": key, "label": label, "type": "bool", "help": help}


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
    __slots__ = ("id", "cmd", "argv", "proc", "lines", "next_seq", "status",
                 "returncode", "started_at", "finished_at", "moves", "lock")

    def __init__(self, job_id: str, cmd: str, argv: list[str], proc, moves: bool):
        self.id = job_id
        self.cmd = cmd
        self.argv = argv
        self.proc = proc
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
        if not job or job.status != "running":
            return
        sig = signal.SIGKILL if force else signal.SIGINT
        try:
            os.killpg(os.getpgid(job.proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def stop_all(self, force: bool = False) -> int:
        with self.lock:
            ids = [j.id for j in self.jobs.values() if j.status == "running"]
        for job_id in ids:
            self.stop(job_id, force=force)
        return len(ids)

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
            if job.status == "running":
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

let prefsTimer = null;
function schedulePrefs(){
  clearTimeout(prefsTimer);
  prefsTimer = setTimeout(() => {
    const data = {};
    for (const cmd in formState) data[cmd] = formState[cmd].get();
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
  const panel = el("div", {class:"panel" + (job.moves ? " moves" : "")});
  const state = {seq:-1, collapsed:false, autoscroll:true, filter:"", el:panel, job};

  const head = el("div", {class:"panel-head", onclick:(ev)=>{
    if (ev.target.closest("button") || ev.target.closest("input")) return;
    state.collapsed = !state.collapsed;
    panel.classList.toggle("collapsed", state.collapsed);
  }});
  const left = el("div", {class:"left"},
    el("span", {class:"caret"}, "▾"),
    el("span", {class:"name"}, `${job.cmd}  #${job.id}`),
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
  const runningCmds = new Set(Object.values(panels).filter(p=>p.job.status==="running").map(p=>p.job.cmd));
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
  try{ PREFS = await (await fetch("/api/prefs")).json(); }catch(e){ PREFS = {}; }
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
