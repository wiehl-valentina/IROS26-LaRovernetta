"""Live mission viewer: stdlib-only HTTP server for the demo.

Serves three routes on localhost while the mission runs:

    /               auto-refreshing page: overlay image + decision table
    /overlay.jpg    latest SAM-TP overlay (pre-encoded by perception)
    /decision.json  latest control-tick record

Zero dependencies, zero work per request — the handlers serve bytes that the
perception and control threads already produced. Runs in a daemon thread via
``start_viewer``; the runner calls ``server.shutdown()`` on mission end.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .shared import SharedState

log = logging.getLogger(__name__)

_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Sana Rover — live</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #111;
         color: #eee; margin: 0; padding: 1rem; }
  h1 { font-size: 1.1rem; margin: 0 0 .75rem; color: #7c4; }
  .wrap { display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-start; }
  img { max-width: min(720px, 100%); border-radius: 6px; background: #000; }
  table { border-collapse: collapse; font-size: .9rem; min-width: 260px; }
  td { padding: .25rem .6rem; border-bottom: 1px solid #333; }
  td:first-child { color: #999; }
  .state { font-weight: 700; color: #7c4; }
  .stopped { color: #e66; }
</style>
</head>
<body>
<h1>Sana Rover Policy — SAM-TP live (green = drivable)</h1>
<div class="wrap">
  <img id="overlay" src="/overlay.jpg" alt="waiting for first frame...">
  <table id="info"><tr><td>waiting for telemetry...</td></tr></table>
</div>
<script>
const ROWS = [
  ["state", "state"], ["reason", "reason"], ["linear", "linear"],
  ["angular", "angular"], ["distance_m", "distance to checkpoint (m)"],
  ["goal_offset_deg", "goal offset (deg)"], ["speed_ms", "speed (m/s)"],
  ["battery_pct", "battery (%)"], ["mask_age_s", "mask age (s)"],
  ["inference_s", "inference (s)"], ["checkpoint_seq", "checkpoint #"],
  ["checkpoints_reached", "reached"], ["tick", "tick"],
];
async function refresh() {
  try {
    const r = await fetch("/decision.json", {cache: "no-store"});
    if (r.ok) {
      const d = await r.json();
      const rows = ROWS.map(([k, label]) => {
        let v = d[k];
        if (v === null || v === undefined) v = "—";
        const cls = k === "state" ? (d.linear === 0 && d.angular === 0
                                     ? "state stopped" : "state") : "";
        return `<tr><td>${label}</td><td class="${cls}">${v}</td></tr>`;
      });
      document.getElementById("info").innerHTML = rows.join("");
    }
  } catch (e) { /* mission ended or not started yet */ }
  const img = document.getElementById("overlay");
  img.src = "/overlay.jpg?" + Date.now();
}
setInterval(refresh, 400);
refresh();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    shared: SharedState = None  # set by start_viewer

    def do_GET(self):  # noqa: N802 (stdlib naming)
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _PAGE.encode())
        elif path == "/overlay.jpg":
            snap = self.shared.latest_mask()
            if snap is None:
                self._send(404, "text/plain", b"no frame yet")
            else:
                self._send(200, "image/jpeg", snap.overlay_jpeg)
        elif path == "/decision.json":
            record = self.shared.latest_decision()
            if record is None:
                self._send(404, "text/plain", b"no decision yet")
            else:
                self._send(200, "application/json", json.dumps(record).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence per-request spam
        pass


def start_viewer(shared: SharedState, port: int = 8001,
                 host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Start the viewer in a daemon thread; returns the server (call
    ``shutdown()`` to stop it). Port 0 picks a free one (used by tests)."""
    handler = type("BoundHandler", (_Handler,), {"shared": shared})
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever,
                              name="viewer", daemon=True)
    thread.start()
    return server
