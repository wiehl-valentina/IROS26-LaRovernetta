#!/usr/bin/env python3
"""
dashboard_server.py
--------------------
Dashboard HTML local para el proyecto: un servidor chiquito (solo libreria
estandar de Python, no hace falta instalar nada mas) que corre DENTRO de tu
WSL, y le da botones a rover_launch.sh. Abrís http://localhost:8765 desde
el navegador de Windows (funciona directo, WSL2 expone localhost solo) y
desde ahi lanzas/parás cada componente y ves su log en vivo.

Como es un servidor que ejecuta comandos de tu sistema, SOLO escucha en
localhost (127.0.0.1) — no es alcanzable desde otras maquinas de tu red.

Uso:
    cd ~/IROS26-LaRovernetta
    python3 dashboard_server.py
    # abrir http://localhost:8765 en el navegador

Requiere que rover_launch.sh este en el mismo directorio (o pasale la ruta
con --script /ruta/a/rover_launch.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --------------------------------------------------------------- estado ---

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
SCRIPT_PATH = "./rover_launch.sh"

# Comandos permitidos y que flags acepta cada uno. Es una whitelist a
# proposito: el servidor arma la lista de argv el mismo (nunca concatena
# texto libre a un shell), asi que no hay injection posible aunque esto
# quede escuchando mientras trabajas.
ALLOWED = {
    "sdk": [],
    "sdk-client": [],
    "ros2-check": [],
    "doctor": [],
    "perception": ["image", "config", "out"],
    "genie-bridge": ["config", "go", "max-seconds", "start-mission"],
    "mapping-ros2": ["db"],
    "map-session": ["config", "go", "max-seconds", "map-out", "debug-dir", "export-every-s"],
    "indoor-bridge": ["config", "go", "max-seconds", "debug-dir", "search-mode", "waypoints-path"],
    "traversability": ["level", "image", "out", "save-dir"],
}

BOOL_FLAGS = {"go", "start-mission"}


def build_argv(cmd: str, opts: dict) -> list[str]:
    if cmd not in ALLOWED:
        raise ValueError(f"comando no permitido: {cmd}")
    allowed_keys = set(ALLOWED[cmd])
    argv = [SCRIPT_PATH, cmd]

    if cmd == "traversability":
        level = opts.get("level")
        if level not in ("predict", "live", "drive", "mission"):
            raise ValueError("nivel invalido para traversability")
        argv.append(level)
        for key in ("image", "out", "save-dir"):
            val = opts.get(key)
            if val:
                argv += [f"--{key}", str(val)]
        return argv

    for key, val in opts.items():
        if key not in allowed_keys:
            raise ValueError(f"flag no permitida para {cmd}: {key}")
        if key in BOOL_FLAGS:
            if val:
                argv.append(f"--{key}")
        elif val:
            argv += [f"--{key}", str(val)]
    return argv


def start_job(cmd: str, opts: dict) -> str:
    argv = build_argv(cmd, opts)
    job_id = uuid.uuid4().hex[:8]

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # process group propio -> podemos cortar todo lo que lance (ros2 launch arranca varios hijos)
    )

    job = {
        "id": job_id,
        "cmd": cmd,
        "argv": argv,
        "proc": proc,
        "output": [],
        "status": "running",
        "returncode": None,
        "started_at": time.time(),
        "lock": threading.Lock(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    def reader():
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                with job["lock"]:
                    job["output"].append(line.rstrip("\n"))
        except Exception as exc:  # pragma: no cover
            with job["lock"]:
                job["output"].append(f"[dashboard] error leyendo salida: {exc}")
        finally:
            rc = proc.wait()
            with job["lock"]:
                job["status"] = "exited"
                job["returncode"] = rc
                job["output"].append(f"[dashboard] proceso terminado (codigo {rc})")

    threading.Thread(target=reader, daemon=True).start()
    return job_id


def stop_job(job_id: str, force: bool = False) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "running":
        return
    pgid = os.getpgid(job["proc"].pid)
    try:
        os.killpg(pgid, signal.SIGKILL if force else signal.SIGINT)
    except ProcessLookupError:
        pass


def clear_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job and job["status"] != "running":
            del JOBS[job_id]


# ------------------------------------------------------------------ HTML --

INDEX_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Rover Launcher</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { background:#0f1115; color:#e6e6e6; font-family: ui-monospace, Menlo, Consolas, monospace; margin:0; padding:24px 28px 60px; }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:#8a8f98; font-size:13px; margin-bottom:20px; }
  h2.section-title { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:#7ea6d6; margin:22px 0 10px; border-bottom:1px solid #22262e; padding-bottom:6px; }
  .layout { display:grid; grid-template-columns: 680px 1fr; gap:24px; align-items:start; }
  .col-forms { position:sticky; top:24px; }
  .primary-grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; align-items:start; }
  .card { background:#161a20; border:1px solid #262b33; border-radius:10px; padding:14px; margin-bottom:12px; transition: background-color .15s, border-color .15s; }
  .card.armed { background:#1c150a; border-color:#b7791f; }
  .card h3 { margin:0 0 10px; font-size:12.5px; color:#9fd3ff; text-transform:uppercase; letter-spacing:.03em; }
  label { display:block; font-size:11.5px; color:#b7bcc4; margin:8px 0 3px; }
  input[type=text], select { width:100%; background:#0f1115; border:1px solid #2a2f38; color:#e6e6e6; border-radius:6px; padding:6px 8px; font-family:inherit; font-size:12.5px; }
  .checkbox-row { display:flex; align-items:center; gap:8px; margin-top:10px; }
  .checkbox-row input { width:auto; }
  .checkbox-row label { margin:0; font-size:12px; }
  button { background:#2b6cb0; color:white; border:none; border-radius:6px; padding:8px 12px; font-size:12.5px; cursor:pointer; margin-top:12px; width:100%; }
  button.small { width:auto; margin:0; padding:4px 10px; font-size:11px; }
  button.danger { background:#c53030; }
  button.warn { background:#b7791f; }
  button.ghost { background:#232833; }
  button.small.ghost.active { background:#1d3a5c; color:#9fd3ff; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .go-warning { display:none; background:#3a2410; border:1px solid #b7791f; color:#f0c46a; font-size:11.5px; padding:8px; border-radius:6px; margin-top:8px; line-height:1.4; }

  details.secondary { background:#12151a; border:1px solid #20242c; border-radius:10px; padding:10px 14px; margin-top:6px; }
  details.secondary summary { cursor:pointer; font-size:12px; color:#8a8f98; text-transform:uppercase; letter-spacing:.05em; padding:4px 0; }
  details.secondary[open] summary { color:#b7bcc4; margin-bottom:8px; }

  #doctorOut { display:none; margin-top:10px; height:220px; min-height:100px; max-height:70vh; overflow-y:auto; resize:vertical; background:#0d0f13; border:1px solid #262b33; border-radius:8px; padding:10px; font-size:11.5px; line-height:1.5; user-select:text; white-space:pre-wrap; word-break:break-word; }

  .panel { background:#0d0f13; border:1px solid #262b33; border-radius:10px; margin-bottom:14px; overflow:hidden; }
  .panel-head { display:flex; justify-content:space-between; align-items:center; background:#161a20; padding:8px 12px; cursor:pointer; user-select:none; }
  .panel-head .left { display:flex; align-items:center; gap:8px; min-width:0; }
  .panel-head .caret { color:#666; font-size:11px; width:10px; flex:none; }
  .panel-head .name { font-size:13px; color:#e6e6e6; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .panel-head .status { font-size:10.5px; padding:2px 8px; border-radius:10px; flex:none; }
  .status.running { background:#134e2c; color:#7ee2a8; }
  .status.exited { background:#3a2410; color:#f0c46a; }
  .panel-actions { display:flex; gap:6px; flex:none; }
  .panel-body { border-top:1px solid #1b1f27; }
  .panel.collapsed .panel-body { display:none; }
  .panel.collapsed .caret { transform: rotate(-90deg); display:inline-block; }
  pre.log { margin:0; padding:10px 12px; height:260px; min-height:80px; max-height:70vh; overflow-y:auto; overflow-x:hidden; resize:vertical; font-size:12px; line-height:1.5; white-space:pre-wrap; word-break:break-word; user-select:text; cursor:text; }
  .empty { color:#565c66; font-size:13px; padding:20px; text-align:center; border:1px dashed #262b33; border-radius:10px; }
  .ansi-red { color:#ff8080; }
  .ansi-green { color:#7ee2a8; }
  .ansi-yellow { color:#f0c46a; }
  .ansi-blue { color:#8ab4f0; }
  .btn-busy { opacity:.5; cursor:default; }
</style>
</head>
<body>
<h1>Rover Launcher</h1>
<div class="sub">Dashboard local sobre rover_launch.sh &mdash; corre en tu WSL, solo accesible desde esta maquina.</div>

<div class="layout">
  <div class="col-forms">
    <div class="card">
      <h3>Instalacion</h3>
      <button class="ghost" onclick="checkDoctor()">Chequear instalacion (doctor)</button>
      <pre id="doctorOut"></pre>
    </div>
    <div id="primaryForms" class="primary-grid"></div>
    <details class="secondary">
      <summary>Pruebas / diagnostico</summary>
      <div id="secondaryForms"></div>
    </details>
  </div>

  <div>
    <h2 class="section-title">Procesos en vivo</h2>
    <div id="jobs"><div class="empty">Todavia no lanzaste nada.</div></div>
  </div>
</div>

<script>
// Cada entrada: cmd, title, fields, goFlag (agrega checkbox --go con aviso),
// levelMoves (para traversability: niveles que mueven el rover de verdad).
const PRIMARY = [
  { cmd: "sdk", title: "SDK (hypercorn)", fields: [] },
  { cmd: "genie-bridge", title: "Bridge: genie_rover.bridge", fields: [
      {key:"config", label:"Config", type:"text", placeholder:"configs/frodobot_rover.yaml"},
      {key:"max-seconds", label:"Max segundos (con --go)", type:"text", placeholder:"300"},
      {key:"start-mission", label:"--start-mission", type:"bool"},
  ], goFlag:true },
  { cmd: "indoor-bridge", title: "Bridge: indoor (tour de checkpoints, busca conos)", fields: [
      {key:"config", label:"Config", type:"text", placeholder:"configs/indoor_cone_search.yaml"},
      {key:"search-mode", label:"Modo de busqueda (vacio = el del config)", type:"select", options:["","wander","frontier","waypoints"]},
      {key:"waypoints-path", label:"Ruta de waypoints (solo si el modo es waypoints)", type:"text", placeholder:"configs/waypoints_example.yaml"},
      {key:"max-seconds", label:"Max segundos (con --go)", type:"text", placeholder:"180"},
      {key:"debug-dir", label:"Carpeta debug (con --go)", type:"text", placeholder:"debug/indoor_run1"},
  ], goFlag:true },
  { cmd: "traversability", title: "Traversability", fields: [
      {key:"level", label:"Nivel", type:"select", options:["predict","live","drive","mission"]},
      {key:"image", label:"Imagen (solo predict)", type:"text", placeholder:"screenshots/imagen.jpg"},
      {key:"out", label:"Salida overlay (solo predict)", type:"text", placeholder:"overlay.png"},
      {key:"save-dir", label:"Carpeta overlays (solo live)", type:"text", placeholder:"trav_out"},
  ], levelMoves:["drive","mission"] },
  { cmd: "mapping-ros2", title: "RTAB-Map (ROS2)", fields: [
      {key:"db", label:"database_path", type:"text", placeholder:"~/maps/sesion1.db"},
  ]},
  { cmd: "map-session", title: "Mapeo: genie_rover.Indoor.map_session", fields: [
      {key:"config", label:"Config", type:"text", placeholder:"configs/indoor_mapping.yaml"},
      {key:"max-seconds", label:"Max segundos (con --go)", type:"text", placeholder:"300"},
      {key:"map-out", label:"Prefijo de salida (con --go)", type:"text", placeholder:"maps/sesion1"},
      {key:"export-every-s", label:"Exportar cada N segundos", type:"text", placeholder:"30"},
      {key:"debug-dir", label:"Carpeta debug (con --go)", type:"text", placeholder:"debug/map_run1"},
  ], goFlag:true },
];

const SECONDARY = [
  { cmd: "sdk-client", title: "Prueba de conexion (sdk-client)", fields: [] },
  { cmd: "perception", title: "Percepcion (SAM-TP sobre foto)", fields: [
      {key:"image", label:"Ruta a la imagen", type:"text", placeholder:"screenshots/foto.jpg"},
      {key:"config", label:"Config (opcional)", type:"text", placeholder:"configs/frodobot_rover.yaml"},
      {key:"out", label:"Carpeta de salida", type:"text", placeholder:"debug/"},
  ]},
  { cmd: "ros2-check", title: "Test basico ROS2 (talker)", fields: [] },
];

// El script bash colorea su salida con codigos ANSI pensados para una
// terminal (\\x1b[1;32m...\\x1b[0m). En el navegador esos codigos se ven
// como texto basura si no se interpretan — este parser los convierte en
// spans con clase de color, y arma texto plano para todo lo demas. Solo
// contempla los codigos que emite nuestro propio script (rojo/verde/
// amarillo/azul en negrita + reset), es a proposito minimo.
const ANSI_MAP = { "1;31":"ansi-red", "1;32":"ansi-green", "1;33":"ansi-yellow", "1;34":"ansi-blue", "0":null };

function ansiToFragment(text) {
  const frag = document.createDocumentFragment();
  const re = /\x1b\[([0-9;]*)m/g;
  let lastIndex = 0, currentClass = null, match;
  const push = (str) => {
    if (!str) return;
    if (currentClass) {
      const span = document.createElement("span");
      span.className = currentClass;
      span.textContent = str;
      frag.appendChild(span);
    } else {
      frag.appendChild(document.createTextNode(str));
    }
  };
  while ((match = re.exec(text)) !== null) {
    push(text.slice(lastIndex, match.index));
    lastIndex = re.lastIndex;
    currentClass = ANSI_MAP[match[1]] ?? null;
  }
  push(text.slice(lastIndex));
  return frag;
}

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  for (const k in (attrs||{})) {
    if (k === "class") e.className = attrs[k];
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  for (const c of children) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return e;
}

const formButtons = {}; // cmd -> {btn, card}

function buildForm(f, compact) {
  const card = el("div", {class:"card"});
  card.appendChild(el("h3", {}, f.title));
  const inputs = {};
  f.fields.forEach(field => {
    card.appendChild(el("label", {}, field.label));
    if (field.type === "bool") {
      const row = el("div", {class:"checkbox-row"});
      const cb = el("input", {type:"checkbox", id:`${f.cmd}-${field.key}`});
      row.appendChild(cb);
      row.appendChild(el("label", {for:`${f.cmd}-${field.key}`}, field.key));
      card.appendChild(row);
      inputs[field.key] = () => cb.checked;
    } else if (field.type === "select") {
      const sel = el("select", {id:`${f.cmd}-${field.key}`});
      field.options.forEach(o => sel.appendChild(el("option", {value:o}, o)));
      card.appendChild(sel);
      inputs[field.key] = () => sel.value;
    } else {
      const inp = el("input", {type:"text", id:`${f.cmd}-${field.key}`, placeholder: field.placeholder||""});
      card.appendChild(inp);
      inputs[field.key] = () => inp.value.trim();
    }
  });

  let goCb = null;
  const warn = el("div", {class:"go-warning"}, "Esto va a mover el rover de verdad. Confirma que el area esta despejada y tenes a mano el boton Detener.");
  if (f.goFlag) {
    const row = el("div", {class:"checkbox-row"});
    goCb = el("input", {type:"checkbox", id:`${f.cmd}-go`});
    row.appendChild(goCb);
    row.appendChild(el("label", {for:`${f.cmd}-go`}, "--go (modo real, mueve el rover)"));
    card.appendChild(row);
    card.appendChild(warn);
    goCb.addEventListener("change", () => {
      warn.style.display = goCb.checked ? "block" : "none";
      card.classList.toggle("armed", goCb.checked);
    });
  }

  const btn = el("button", {onclick: async () => {
    if (btn.disabled) return;
    const opts = {};
    f.fields.forEach(field => { const v = inputs[field.key](); if (v !== "" && v !== false) opts[field.key] = v; });
    if (f.goFlag) opts.go = !!goCb.checked;
    if (f.levelMoves && f.levelMoves.includes(opts.level)) {
      if (!confirm("Este nivel mueve el rover de verdad. Confirmas?")) return;
    }
    btn.disabled = true;
    btn.textContent = "Lanzando...";
    try {
      await runCmd(f.cmd, opts);
    } finally {
      // el propio refreshJobs() decide si lo vuelve a habilitar (queda
      // deshabilitado mientras ese comando siga corriendo)
    }
  }}, "Lanzar");
  card.appendChild(btn);
  formButtons[f.cmd] = btn;
  return card;
}

function renderForms() {
  const p = document.getElementById("primaryForms");
  PRIMARY.forEach(f => p.appendChild(buildForm(f)));
  const s = document.getElementById("secondaryForms");
  SECONDARY.forEach(f => s.appendChild(buildForm(f)));
}

async function runCmd(cmd, opts) {
  const res = await fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({cmd, opts})});
  if (!res.ok) { alert("Error: " + await res.text()); return; }
  refreshJobs();
}

async function stopJob(id, force) {
  await fetch(`/api/stop?job=${encodeURIComponent(id)}&force=${force?1:0}`, {method:"POST"});
}

async function clearJob(id) {
  await fetch(`/api/clear?job=${encodeURIComponent(id)}`, {method:"POST"});
  delete panelState[id];
  refreshJobs();
}

function copyLog(id) {
  const pre = document.getElementById(`log-${id}`);
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).catch(() => {
    // fallback: seleccionar el texto para que Ctrl+C funcione a mano
    const range = document.createRange();
    range.selectNodeContents(pre);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  });
}

// Estado por panel que sobrevive a los refresh: cuantas lineas ya se
// pintaron (para solo *agregar* texto nuevo, nunca reconstruir el <pre> —
// asi no se pierde una seleccion de texto en curso), y si esta colapsado.
const panelState = {}; // id -> {lines, collapsed, el, pre, statusEl}

function ensurePanel(job, container) {
  if (panelState[job.id]) return panelState[job.id];

  const panel = el("div", {class:"panel"});
  const head = el("div", {class:"panel-head", onclick: (ev) => {
    if (ev.target.closest("button")) return;
    state.collapsed = !state.collapsed;
    panel.classList.toggle("collapsed", state.collapsed);
  }});
  const left = el("div", {class:"left"});
  const caret = el("span", {class:"caret"}, "▾");
  const nameEl = el("span", {class:"name"}, `${job.cmd}  #${job.id}`);
  left.appendChild(caret);
  left.appendChild(nameEl);
  const statusEl = el("span", {class:"status running"}, "corriendo");
  const actions = el("div", {class:"panel-actions"});
  const stopBtn = el("button", {class:"small warn", onclick:(ev)=>{
    ev.stopPropagation();
    stopBtn.disabled = true; killBtn.disabled = true;
    stopBtn.textContent = "Deteniendo...";
    stopJob(job.id,false);
  }}, "Detener");
  const killBtn = el("button", {class:"small danger", onclick:(ev)=>{
    ev.stopPropagation();
    stopBtn.disabled = true; killBtn.disabled = true;
    killBtn.textContent = "Matando...";
    stopJob(job.id,true);
  }}, "Forzar");
  const copyBtn = el("button", {class:"small ghost", onclick:(ev)=>{ev.stopPropagation(); copyLog(job.id);}}, "Copiar log");
  const clearBtn = el("button", {class:"small ghost", onclick:(ev)=>{ev.stopPropagation(); clearJob(job.id);}}, "Quitar");
  // Auto-scroll: ON por defecto (baja sola mientras el usuario ya este cerca
  // del final). Un click lo apaga para poder leer un error historico sin que
  // cada linea nueva tire el scroll para abajo; otro click lo prende de
  // vuelta y salta al final de una. Util con procesos que escupen mucho
  // (ROS2, mapeo).
  const autoBtn = el("button", {class:"small ghost active", onclick:(ev)=>{
    ev.stopPropagation();
    state.autoscroll = !state.autoscroll;
    autoBtn.textContent = state.autoscroll ? "Auto-scroll: ON" : "Auto-scroll: OFF";
    autoBtn.classList.toggle("active", state.autoscroll);
    if (state.autoscroll) state.pre.scrollTop = state.pre.scrollHeight;
  }}, "Auto-scroll: ON");
  actions.appendChild(autoBtn);
  actions.appendChild(copyBtn);
  actions.appendChild(stopBtn);
  actions.appendChild(killBtn);
  actions.appendChild(clearBtn);
  head.appendChild(left);
  const right = el("div", {class:"panel-actions"});
  right.appendChild(statusEl);
  right.appendChild(actions);
  head.appendChild(right);

  const body = el("div", {class:"panel-body"});
  const pre = el("pre", {class:"log", id:`log-${job.id}`});
  body.appendChild(pre);

  panel.appendChild(head);
  panel.appendChild(body);
  container.appendChild(panel);

  const state = { lines: 0, collapsed: false, autoscroll: true, el: panel, pre, statusEl, stopBtn, killBtn, autoBtn };
  panelState[job.id] = state;
  return state;
}

async function refreshJobs() {
  const res = await fetch("/api/jobs");
  const jobs = await res.json();
  const root = document.getElementById("jobs");

  if (jobs.length === 0) {
    root.innerHTML = '<div class="empty">Todavia no lanzaste nada.</div>';
    return;
  }
  if (root.querySelector(".empty")) root.innerHTML = "";

  const seen = new Set();
  for (const j of jobs) {
    seen.add(j.id);
    const st = ensurePanel(j, root);

    // Solo agregar lineas nuevas — nunca tocar el texto ya pintado, para no
    // romper una seleccion de texto que el usuario tenga en curso.
    if (j.output.length > st.lines) {
      const nuevas = j.output.slice(st.lines);
      const nearBottom = st.pre.scrollTop + st.pre.clientHeight >= st.pre.scrollHeight - 20;
      const text = (st.lines > 0 ? "\\n" : "") + nuevas.join("\\n");
      st.pre.appendChild(ansiToFragment(text));
      st.lines = j.output.length;
      // Con auto-scroll apagado (el usuario lo apago para leer algo mas
      // arriba) no tocamos el scroll aunque llegue mucha salida nueva.
      if (st.autoscroll && nearBottom) st.pre.scrollTop = st.pre.scrollHeight;
    }

    if (j.status === "running") {
      st.statusEl.textContent = "corriendo";
      st.statusEl.className = "status running";
      st.stopBtn.style.display = "";
      st.killBtn.style.display = "";
      st.stopBtn.disabled = false; st.killBtn.disabled = false;
      st.stopBtn.textContent = "Detener"; st.killBtn.textContent = "Forzar";
    } else {
      st.statusEl.textContent = `salio (${j.returncode})`;
      st.statusEl.className = "status exited";
      st.stopBtn.style.display = "none";
      st.killBtn.style.display = "none";
    }
  }

  // sacar del DOM paneles de jobs que el server ya no reporta (limpiados)
  for (const id of Object.keys(panelState)) {
    if (!seen.has(id)) { panelState[id].el.remove(); delete panelState[id]; }
  }

  // "Lanzar" queda deshabilitado mientras ya haya un job de ese mismo
  // comando corriendo — evita pisar dos instancias del mismo proceso
  // (dos hypercorn peleando el puerto, dos bridges mandando comandos
  // contradictorios al rover, etc.)
  const runningCmds = new Set(jobs.filter(j => j.status === "running").map(j => j.cmd));
  for (const [cmd, btn] of Object.entries(formButtons)) {
    if (runningCmds.has(cmd)) {
      btn.disabled = true;
      btn.classList.add("btn-busy");
      btn.textContent = "Ya esta corriendo";
    } else {
      btn.disabled = false;
      btn.classList.remove("btn-busy");
      btn.textContent = "Lanzar";
    }
  }
}

async function checkDoctor() {
  const out = document.getElementById("doctorOut");
  out.style.display = "block";
  out.textContent = "corriendo...";
  const res = await fetch("/api/doctor");
  const text = await res.text();
  out.textContent = "";
  out.appendChild(ansiToFragment(text));
}

renderForms();
refreshJobs();
setInterval(refreshJobs, 1500);
</script>
</body>
</html>
"""


# --------------------------------------------------------------- server ---

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silenciar el log por default de http.server
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text, code=200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/jobs":
            with JOBS_LOCK:
                out = []
                for j in JOBS.values():
                    with j["lock"]:
                        out.append({
                            "id": j["id"], "cmd": j["cmd"], "status": j["status"],
                            "returncode": j["returncode"], "output": j["output"],
                        })
                out.sort(key=lambda j: j["id"])
            self._json(out)
        elif parsed.path == "/api/doctor":
            try:
                res = subprocess.run([SCRIPT_PATH, "doctor"], capture_output=True, text=True, timeout=30)
                self._text(res.stdout + res.stderr)
            except Exception as exc:
                self._text(f"error corriendo doctor: {exc}", 500)
        else:
            self._text("not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                job_id = start_job(data.get("cmd", ""), data.get("opts", {}))
                self._json({"job": job_id})
            except Exception as exc:
                self._text(str(exc), 400)
        elif parsed.path == "/api/stop":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("job") or [""])[0]
            force = (qs.get("force") or ["0"])[0] == "1"
            stop_job(job_id, force=force)
            self._json({"ok": True})
        elif parsed.path == "/api/clear":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("job") or [""])[0]
            clear_job(job_id)
            self._json({"ok": True})
        else:
            self._text("not found", 404)


def main():
    global SCRIPT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--script", default="./rover_launch.sh")
    args = ap.parse_args()

    SCRIPT_PATH = args.script
    if not os.path.isfile(SCRIPT_PATH):
        raise SystemExit(f"No encuentro {SCRIPT_PATH} — corre este server parado en la misma carpeta que rover_launch.sh, o pasa --script /ruta/completa")
    if not os.access(SCRIPT_PATH, os.X_OK):
        raise SystemExit(f"{SCRIPT_PATH} no tiene permiso de ejecucion — corre: chmod +x {SCRIPT_PATH}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard corriendo en http://localhost:{args.port}  (Ctrl+C para cortar)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with JOBS_LOCK:
            for j in JOBS.values():
                if j["status"] == "running":
                    try:
                        os.killpg(os.getpgid(j["proc"].pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    main()
