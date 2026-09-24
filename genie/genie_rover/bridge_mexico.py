"""Bridge para el circuito de Mexico: SIN regimen cercano ni frenado por obstaculos.
Va por el camino como pueda, guiado por checkpoints secundarios GPS (la ruta grabada en
genie/mexico_checkpoints_help/).

Diferencias con bridge.Bridge:

1. Meta. Entre checkpoints oficiales la meta local es un punto de la ruta
   grabada a lookahead_m por delante del progreso (ver route.py). El claim
   sigue siendo SOLO contra el checkpoint oficial del SDK: las migas de pan
   nunca disparan claim_checkpoint(). Cuando el oficial queda a menos de
   mexico.oficial_directo_m, se apunta directo a el.

2. Obstaculo al frente. IGNORADO COMPLETAMENTE: el robot sigue adelante sin
   frenar, retroceder, ni esquivar.

3. Planner sin camino. En vez de barrido/VLM, avanza despacio apuntando a la
   meta.

4. Progreso estancado. Si el robot no avanza sobre la ruta durante
   ruta.estancado_s, la meta salta unos metros adelante: el obstaculo estaba
   sobre la ruta y hay que rodearlo.

Requiere el hook _compute_goal y el flag brake_on_obstacle en bridge.Bridge.

Uso (simulacro):
    python -m genie_rover.bridge_mexico --config configs/frodobot_rover.yaml \
        --ruta genie/mexico_checkpoints_help/Mexico_mision1/mexico

De verdad (varias rutas se concatenan en orden):
    python -m genie_rover.bridge_mexico --config configs/frodobot_rover.yaml \
        --ruta genie/mexico_checkpoints_help/Mexico_mision2/mexico5 \
               genie/mexico_checkpoints_help/Mexico_mision2/mexico5part3.geojson \
        --go --start-mission --max-seconds 120 --debug-dir debug/mex2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from .bridge import Bridge, _check_placeholders
from .navigation import DriveCommand, goal_from_gps
from .route import RouteConfig, RouteFollower, cargar_ruta


@dataclass
class MexicoStats:
    avances_sin_camino: int = 0
    metas_ruta: int = 0
    metas_oficiales: int = 0


def _gps_valido(lat: float, lon: float) -> bool:
    """El SDK manda 0,0 cuando todavia no hay fix."""
    return abs(float(lat)) > 1e-6 or abs(float(lon)) > 1e-6


class _Recto:
    """Meta fija derecho adelante (mismo truco que el bridge base)."""
    def __init__(self, dist: float):
        self.x_right_m = 0.0
        self.y_forward_m = dist
        self.distance_m = dist
        self.relative_bearing_deg = 0.0


class BridgeMexico(Bridge):
    def __init__(self, cfg: dict, rutas: list[str], dry_run: bool = True,
                 debug_dir: str | None = None, dashboard_json: str | None = None):
        super().__init__(cfg, dry_run=dry_run, debug_dir=debug_dir)
        mx = cfg.get("mexico", {}) or {}

        # Overlay del dashboard del SDK (map.js / dashboard-map.js, modo GPS).
        # Es el mismo archivo que usa la ruta indoor: static/genie_waypoints.json.
        ruta_json = dashboard_json or mx.get("dashboard_json")
        self.dashboard_json = Path(ruta_json) if ruta_json else None
        self.dashboard_period_s = float(mx.get("dashboard_period_s", 1.0))
        self._dash_t = 0.0
        self._dash_state = ""
        self._dash_target: dict | None = None
        self._trail_gps: list[list[float]] = []

        puntos = []
        for r in rutas:
            pts = cargar_ruta(r)
            print(f"[mexico] ruta {Path(r).name}: {len(pts)} puntos")
            puntos += pts
        self.ruta: RouteFollower | None = None
        if len(puntos) >= 2:
            validos = {f.name for f in fields(RouteConfig)}
            rcfg = RouteConfig(**{k: v for k, v in (mx.get("ruta") or {}).items() if k in validos})
            self.ruta = RouteFollower(puntos, rcfg)
            print(f"[mexico] ruta total: {self.ruta.total_m:.1f} m, "
                  f"lookahead {rcfg.lookahead_m:.1f} m")
        else:
            print("[mexico] sin ruta: solo checkpoints oficiales")

        self.oficial_directo_m = float(mx.get("oficial_directo_m", 6.0))
        self.crawl_linear = float(mx.get("crawl_linear", 0.08))

        # Sin freno por obstaculo al frente: el frame sigue directo al planner.
        self.brake_on_obstacle = False
        self.recovery_after_empty = 1

        self._last_rel_deg = 0.0
        self.mx_stats = MexicoStats()

    # ----------------------------------------------------------------- meta

    def _compute_goal(self, telem, heading):
        oficial = self.current_target()
        g_of = None
        if oficial is not None and heading is not None:
            g_of = goal_from_gps(telem.latitude, telem.longitude, heading,
                                 oficial.latitude, oficial.longitude, self.goal_range_m)
            if g_of.distance_m < self.claim_radius_m:
                ok, msg = self.client.claim_checkpoint()
                if ok:
                    print(f"[bridge] ✓ checkpoint #{oficial.sequence} conseguido: {msg}")
                    self.refresh_checkpoints()
                else:
                    print(f"[bridge] cerca del checkpoint ({g_of.distance_m:.1f} m) "
                          f"pero rechazado: {msg}")

        # El progreso sobre la ruta solo necesita POSICION, no rumbo: se
        # actualiza aunque la brujula/track GPS todavia no den rumbo.
        gps_ok = _gps_valido(telem.latitude, telem.longitude)
        if self.ruta is not None and gps_ok:
            self.ruta.update(telem.latitude, telem.longitude, time.time())

        if not gps_ok or heading is None:
            self._last_rel_deg = 0.0
            self._dash_target = None
            motivo = "sin fix GPS (lat/lon en 0)" if not gps_ok else "sin rumbo todavia"
            self._dash_state = f"esperando: {motivo}"
            self._write_dashboard(telem)
            return _Recto(self.goal_range_m), f"derecho adelante ({motivo})"

        usar_oficial = g_of is not None and (
            self.ruta is None or self.ruta.terminada
            or g_of.distance_m <= self.oficial_directo_m)

        if usar_oficial:
            self.mx_stats.metas_oficiales += 1
            self._last_rel_deg = g_of.relative_bearing_deg
            self._dash_target = {"lat": oficial.latitude, "lon": oficial.longitude,
                                 "kind": "oficial"}
            self._dash_state = f"yendo al checkpoint oficial #{oficial.sequence}"
            self._write_dashboard(telem)
            return g_of, (f"cp#{oficial.sequence} a {g_of.distance_m:.0f} m, "
                          f"rel {g_of.relative_bearing_deg:+.0f} grados")

        if self.ruta is not None and not self.ruta.terminada:
            lat_t, lon_t = self.ruta.objetivo()
            g = goal_from_gps(telem.latitude, telem.longitude, heading,
                              lat_t, lon_t, self.goal_range_m)
            self.mx_stats.metas_ruta += 1
            self._last_rel_deg = g.relative_bearing_deg
            self._dash_target = {"lat": lat_t, "lon": lon_t, "kind": "ruta"}
            self._dash_state = "siguiendo la ruta"
            self._write_dashboard(telem)
            extra = f" | cp#{oficial.sequence} a {g_of.distance_m:.0f} m" if g_of else ""
            return g, (f"{self.ruta.descripcion()}, rel {g.relative_bearing_deg:+.0f} "
                       f"grados{extra}")

        self._last_rel_deg = 0.0
        self._dash_target = None
        self._dash_state = "ruta terminada, sin checkpoint pendiente"
        self._write_dashboard(telem)
        return _Recto(self.goal_range_m), "derecho adelante (ruta terminada, sin checkpoint)"

    # --------------------------------------------------------- sin camino

    def _recover(self) -> None:
        rel = self._last_rel_deg
        ang = self.follower.angular_sign * self.follower.kp * math.radians(rel)
        ang = float(np.clip(ang, -self.follower.max_angular, self.follower.max_angular))
        self.mx_stats.avances_sin_camino += 1
        self.send(DriveCommand(self.crawl_linear, ang,
                               f"sin camino: avanzo despacio hacia la meta ({rel:+.0f} grados)"))
        self._consecutive_empty = 0

    # ----------------------------------------------------------- dashboard

    def _write_dashboard(self, telem, force: bool = False) -> None:
        """Escribe static/genie_waypoints.json para el overlay del SDK.
        Atomico (tmp + rename): el navegador lo lee cada 1.5 s y no tiene que
        encontrarse nunca un JSON a medio escribir."""
        if self.dashboard_json is None:
            return
        lat = float(telem.latitude) if telem is not None else 0.0
        lon = float(telem.longitude) if telem is not None else 0.0
        gps_ok = _gps_valido(lat, lon)
        if gps_ok:
            if (not self._trail_gps
                    or abs(self._trail_gps[-1][0] - lat) > 2e-6
                    or abs(self._trail_gps[-1][1] - lon) > 2e-6):
                self._trail_gps.append([lat, lon])
                self._trail_gps = self._trail_gps[-600:]

        now = time.time()
        if not force and now - self._dash_t < self.dashboard_period_s:
            return
        self._dash_t = now

        payload: dict = {
            "active": True,
            "frame": "gps",
            "mode": "mexico",
            "ts": now,
            "state": self._dash_state,
            "checkpoints_done": int(self._latest_scanned),
            "pose_gps": {"lat": lat, "lon": lon} if gps_ok else None,
            "target": self._dash_target,
            "trail_gps": self._trail_gps,
        }
        if self.ruta is not None:
            payload.update({
                "waypoints": [{"lat": p.lat, "lon": p.lon, "motivo": p.motivo}
                              for p in self.ruta.puntos],
                "current_index": self.ruta.indice_actual(),
                "route_done": bool(self.ruta.terminada),
                "progress_m": round(self.ruta.progreso_m or 0.0, 2),
                "total_m": round(self.ruta.total_m, 2),
                "desvio_m": (round(self.ruta.desvio_m, 2)
                             if math.isfinite(self.ruta.desvio_m) else None),
                "saltos": self.ruta.saltos,
            })
        self._dump_dashboard(payload)

    def _dump_dashboard(self, payload: dict) -> None:
        try:
            self.dashboard_json.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.dashboard_json.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self.dashboard_json)
        except Exception as exc:
            print(f"[mexico] no pude escribir {self.dashboard_json}: {exc}")

    def run(self, max_seconds: float | None = None) -> None:
        # Publicar la ruta en el mapa YA, antes del primer frame: si la camara
        # o el GPS tardan, igual se ven los puntos intermedios.
        self._dash_state = "arrancando"
        self._write_dashboard(None, force=True)
        try:
            super().run(max_seconds=max_seconds)
        finally:
            # Apagar el overlay: el dashboard vuelve a quedar como el original.
            if self.dashboard_json is not None:
                self._dump_dashboard({"active": False, "ts": time.time()})

    # ------------------------------------------------------------- resumen

    def _print_summary(self) -> None:
        super()._print_summary()
        s = self.mx_stats
        print("\n  --- mexico ---")
        if self.ruta is not None:
            print(f"  {self.ruta.descripcion()}  reenganches={self.ruta.reenganches}"
                  f"  terminada={self.ruta.terminada}")
        print(f"  metas: ruta={s.metas_ruta} oficial={s.metas_oficiales}")
        print(f"  avances sin camino={s.avances_sin_camino}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ruta", nargs="*", default=[],
                    help="uno o mas archivos de ruta grabada, en orden")
    ap.add_argument("--go", action="store_true",
                    help="enviar comandos de verdad (sin esto es simulacro)")
    ap.add_argument("--start-mission", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--dashboard-json", default=None,
                    help="static/genie_waypoints.json del SDK, para ver los "
                         "puntos intermedios en el mapa")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    _check_placeholders(cfg)

    bridge = BridgeMexico(cfg, args.ruta, dry_run=not args.go, debug_dir=args.debug_dir,
                          dashboard_json=args.dashboard_json)

    if args.start_mission:
        print("[bridge] iniciando mision ...")
        print(f"   {bridge.client.start_mission()}")

    if args.go:
        print("\n" + "=" * 62)
        print("  MODO REAL (MEXICO, sin regimen cercano): el rover se va a mover.")
        print("  Ctrl-C frena. Tene el robot a la vista.")
        print("=" * 62)
        for i in (3, 2, 1):
            print(f"  {i} ...")
            time.sleep(1)

    bridge.run(max_seconds=args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())