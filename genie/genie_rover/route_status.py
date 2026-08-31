"""Publicador del estado de la ruta indoor para el dashboard del SDK.

Escribe un unico JSON (por default dentro de `earth-rovers-sdk/static/`) que
`map.js` lee cada 1.5 s para dibujar, ENCIMA del mapa de localhost:8000:

  * la ruta de waypoints del yaml (`mission.waypoints_path`),
  * cual waypoint esta alcanzado / cual es el objetivo actual / cuales faltan,
  * la pose real del rover y su rastro,
  * los checkpoints (conos) ya completados y donde se completaron.

Diseno deliberado
-----------------
1. **Sin endpoints nuevos.** El SDK ya sirve `static/` tal cual, asi que
   `main.py` no se toca. El archivo ES el canal.
2. **El interruptor "solo indoor" es el archivo.** Si no hay mision indoor en
   `search_mode: "waypoints"`, el publicador queda `enabled=False`, nunca
   escribe, y `map.js` no agrega ni un nodo al DOM: el dashboard queda
   identico al original. `close()` marca `active: false` al salir para que no
   quede una ruta vieja dibujada despues de un Ctrl-C.
3. **Nunca voltea la mision.** Todo el I/O va envuelto en try/except con un
   solo aviso por consola. Si el disco falla, la mision sigue.
4. **Escritura atomica** (`tempfile` + `os.replace`) para que el navegador no
   lea nunca un JSON a medio escribir.

Marco de coordenadas
--------------------
Los waypoints y la pose estan en METROS en el marco de odometria/RTAB-Map
(+x adelante, +y a la izquierda; origen = donde arranco el bridge). Adentro
no hay GPS, asi que para poder dibujarlos sobre el mapa lat/lon del SDK hace
falta un ANCLA geografica. Hay tres formas de darla, en este orden:

  a. `anchor={"lat":..., "lon":..., "yaw_deg":...}` explicito en el config
     (`dashboard.anchor`), si sabes donde y con que rumbo arranca el rover.
  b. Nada: `map.js` toma como ancla la primera posicion valida que reporte el
     SDK cuando carga la pagina (es decir, "0,0 = donde esta el rover al
     cargar"), con `orientation` como rumbo.
  c. A mano desde el panel del dashboard ("Anclar aca" + rueda de rumbo), que
     queda guardado en el navegador.

`yaw_deg` del ancla es el rumbo del eje +x del robot, en grados horarios
desde el norte (0 = norte, 90 = este).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


class RouteStatus:
    """Publica el estado de la ruta a un JSON que el dashboard lee.

    Parametros
    ----------
    out_path : str | Path | None
        Destino, normalmente
        `<repo>/earth-rovers-sdk/static/genie_waypoints.json`.
        `None` equivale a `enabled=False`.
    waypoints : secuencia de (x_m, y_m)
        La ruta del yaml, en el mismo orden en que la sigue `WaypointRoute`.
    enabled : bool
        Interruptor maestro. Ponelo en
        `mission.search_mode == "waypoints" and bool(out_path)`.
    anchor : dict | None
        `{"lat":..., "lon":..., "yaw_deg":...}` opcional (caso (a) de arriba).
    reach_radius_m : float
        Solo informativo: el dashboard dibuja ese circulo alrededor del
        objetivo actual para que se vea cuando cuenta como alcanzado.
    min_period_s : float
        Throttle de escritura.
    trail_step_m : float
        Distancia minima entre puntos del rastro.
    trail_max_points : int
        Poda del rastro para que el JSON no crezca sin limite.
    """

    def __init__(
        self,
        out_path: str | Path | None,
        waypoints: Sequence[tuple[float, float]] | None = None,
        *,
        enabled: bool = True,
        anchor: dict | None = None,
        reach_radius_m: float = 0.6,
        min_period_s: float = 0.5,
        trail_step_m: float = 0.15,
        trail_max_points: int = 500,
        frame: str = "odom",
    ) -> None:
        self.enabled = bool(enabled) and out_path is not None
        self.path = Path(out_path) if out_path is not None else None
        self.waypoints = [(float(x), float(y)) for x, y in (waypoints or [])]
        self.anchor = self._clean_anchor(anchor)
        self.reach_radius_m = float(reach_radius_m)
        self.min_period_s = float(min_period_s)
        self.trail_step_m = float(trail_step_m)
        self.trail_max_points = int(trail_max_points)
        self.frame = str(frame)

        self._trail: list[list[float]] = []
        self._checkpoints: list[dict] = []
        self._max_index = 0
        self._last_write = 0.0
        self._warned = False
        self._started_at = time.time()

    # ------------------------------------------------------------------ api

    def publish(
        self,
        pose: Any = None,
        current_index: int | None = None,
        *,
        checkpoints: Iterable[Any] | None = None,
        checkpoints_done: int | None = None,
        state: str | None = None,
        mission_done: bool = False,
        pose_source: str | None = None,
        force: bool = False,
    ) -> None:
        """Escribe el estado actual. Llamalo una vez por frame; el throttle
        interno se encarga de no castigar el disco.

        `pose` puede ser cualquier objeto con `.x`, `.y`, `.theta` (radianes,
        antihorario) — es la `Pose` de `odometry.py` — o una tupla
        `(x, y, theta)`, o `None`.

        `checkpoints` es la lista de conos ya completados: objetos con
        `.x`/`.y` (o tuplas `(x, y)`, o dicts `{"x_m":..,"y_m":..}`).
        """
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_write) < self.min_period_s:
            return

        try:
            x, y, th = self._unpack_pose(pose)
            if x is not None:
                self._push_trail(x, y)
            if checkpoints is not None:
                self._checkpoints = self._clean_checkpoints(checkpoints)

            idx = self._max_index if current_index is None else int(current_index)
            # monotono: si la FSM se va al cono y vuelve, no "des-alcanzamos"
            # waypoints que ya estaban tachados.
            self._max_index = max(self._max_index, idx)
            idx = self._max_index

            n = len(self.waypoints)
            done = checkpoints_done
            if done is None:
                done = len(self._checkpoints)
            # Si la FSM cuenta mas conos completados de los que tenemos
            # anotados, damos por completado uno EN LA POSE ACTUAL. Asi el
            # dashboard marca el cono sin necesidad de enganchar nada donde el
            # bridge maneja request_photo: alcanza con pasar checkpoints_done.
            # (Si ademas llamas add_checkpoint() con la posicion real del cono,
            # esa gana, porque ya quedo anotada antes de llegar aca.)
            while int(done) > len(self._checkpoints) and x is not None:
                self._checkpoints.append({
                    "x_m": round(x, 3), "y_m": round(y, 3),
                    "photo": True,
                    "t": round(now - self._started_at, 1),
                    "label": f"cono {len(self._checkpoints) + 1}",
                    "approx": True,   # pose del rover, no del cono
                })

            payload = {
                "active": True,
                "ts": now,
                "elapsed_s": round(now - self._started_at, 2),
                "frame": self.frame,
                "pose_source": pose_source or "odometry",
                "anchor": self.anchor,
                "reach_radius_m": self.reach_radius_m,
                "waypoints": [{"x_m": round(px, 3), "y_m": round(py, 3)}
                              for px, py in self.waypoints],
                "current_index": min(idx, n),
                "total": n,
                "reached": [i < idx for i in range(n)],
                "route_done": idx >= n and n > 0,
                "pose": None if x is None else {
                    "x_m": round(x, 3),
                    "y_m": round(y, 3),
                    "yaw_deg": round(math.degrees(th), 1),
                },
                "trail": self._trail,
                "checkpoints": self._checkpoints,
                "checkpoints_done": int(done),
                "state": state,
                "mission_done": bool(mission_done),
            }
            self._write(payload)
            self._last_write = now
        except Exception as exc:  # nunca voltear la mision por el dashboard
            self._warn(exc)

    def add_checkpoint(self, x_m: float, y_m: float, *, photo: bool = True,
                       label: str | None = None) -> None:
        """Registra un cono completado (llamalo cuando la FSM pide la foto)."""
        if not self.enabled:
            return
        try:
            self._checkpoints.append({
                "x_m": round(float(x_m), 3),
                "y_m": round(float(y_m), 3),
                "photo": bool(photo),
                "t": round(time.time() - self._started_at, 1),
                "label": label or f"cono {len(self._checkpoints) + 1}",
            })
        except Exception as exc:
            self._warn(exc)

    def close(self) -> None:
        """Apaga el overlay: el dashboard vuelve a quedar exactamente como el
        original. Llamalo en el `finally` de `run()`."""
        if not self.enabled:
            return
        try:
            self._write({"active": False, "ts": time.time()})
        except Exception as exc:
            self._warn(exc)

    # -------------------------------------------------------------- interno

    def _write(self, payload: dict) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _push_trail(self, x: float, y: float) -> None:
        if self._trail:
            lx, ly = self._trail[-1]
            if math.hypot(x - lx, y - ly) < self.trail_step_m:
                return
        self._trail.append([round(x, 3), round(y, 3)])
        if len(self._trail) > self.trail_max_points:
            del self._trail[0:len(self._trail) - self.trail_max_points]

    @staticmethod
    def _unpack_pose(pose: Any) -> tuple[float | None, float, float]:
        if pose is None:
            return None, 0.0, 0.0
        if isinstance(pose, dict):
            return (float(pose.get("x", pose.get("x_m", 0.0))),
                    float(pose.get("y", pose.get("y_m", 0.0))),
                    float(pose.get("theta", pose.get("yaw", 0.0))))
        if isinstance(pose, (tuple, list)):
            vals = list(pose) + [0.0, 0.0, 0.0]
            return float(vals[0]), float(vals[1]), float(vals[2])
        return (float(getattr(pose, "x", 0.0)),
                float(getattr(pose, "y", 0.0)),
                float(getattr(pose, "theta", getattr(pose, "yaw", 0.0))))

    @staticmethod
    def _clean_anchor(anchor: dict | None) -> dict | None:
        if not anchor:
            return None
        try:
            lat = float(anchor.get("lat", anchor.get("latitude")))
            lon = float(anchor.get("lon", anchor.get("longitude")))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        yaw = anchor.get("yaw_deg", anchor.get("heading_deg", 0.0))
        try:
            yaw = float(yaw)
        except (TypeError, ValueError):
            yaw = 0.0
        return {"lat": lat, "lon": lon, "yaw_deg": yaw, "source": "config"}

    def _clean_checkpoints(self, items: Iterable[Any]) -> list[dict]:
        out: list[dict] = []
        for i, it in enumerate(items, start=1):
            try:
                if isinstance(it, dict):
                    x = float(it.get("x_m", it.get("x", 0.0)))
                    y = float(it.get("y_m", it.get("y", 0.0)))
                    photo = bool(it.get("photo", True))
                    label = it.get("label")
                elif isinstance(it, (tuple, list)):
                    x, y = float(it[0]), float(it[1])
                    photo, label = True, None
                else:
                    x = float(getattr(it, "x", 0.0))
                    y = float(getattr(it, "y", 0.0))
                    photo = bool(getattr(it, "photo", True))
                    label = getattr(it, "label", None)
            except (TypeError, ValueError, IndexError):
                continue
            out.append({
                "x_m": round(x, 3), "y_m": round(y, 3),
                "photo": photo, "label": label or f"cono {i}",
            })
        return out

    def _warn(self, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            print(f"[route_status] no se pudo publicar el estado al dashboard "
                  f"({type(exc).__name__}: {exc}). La mision sigue igual; no "
                  f"vuelvo a avisar.")


# --------------------------------------------------------------------- demo

def _demo(out: str) -> None:
    """Genera un archivo de ejemplo animado para probar el dashboard SIN el
    rover:  python3 route_status.py <ruta-a-static/genie_waypoints.json>
    """
    pts = [(2.0, 0.0), (4.0, 0.0), (6.0, -1.5), (6.0, -4.0), (3.5, -5.0)]
    rs = RouteStatus(out, pts, reach_radius_m=0.6, min_period_s=0.0)

    class P:  # pose falsa
        x = y = theta = 0.0

    p = P()
    print(f"[demo] escribiendo {out} — Ctrl-C para terminar (apaga el overlay)")
    try:
        i = 0
        t = 0.0
        while True:
            tx, ty = pts[min(i, len(pts) - 1)]
            dx, dy = tx - p.x, ty - p.y
            d = math.hypot(dx, dy)
            if d < 0.6:
                i += 1
                if i >= len(pts):
                    i = 0
                    p.x = p.y = 0.0
                    rs._max_index = 0
                    rs._trail.clear()
                    rs._checkpoints.clear()
                    continue
            p.theta = math.atan2(dy, dx)
            p.x += 0.25 * math.cos(p.theta)
            p.y += 0.25 * math.sin(p.theta)
            t += 0.4
            if i == 2 and not rs._checkpoints:
                rs.add_checkpoint(p.x, p.y)
            rs.publish(p, current_index=i, state="SEARCH", force=True)
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        rs.close()
        print("\n[demo] overlay apagado.")


if __name__ == "__main__":
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1 else "genie_waypoints.json")
