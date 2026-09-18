#!/usr/bin/env python3
"""Graba un ride manual del rover y lo guarda como ruta de waypoints GPS.

Uso tipico (manejas el rover con teleop/joystick mientras esto corre al lado):

    python3 record_waypoints.py --out rutas/reja_norte.json

    # ruta larga donde no hace falta tanto detalle
    python3 record_waypoints.py --out rutas/campo.json --spacing-m 4.0 \
        --corner-min-m 1.5

Solo lee `GET /data`: no manda ni un comando al rover.

Salida:
  * `<out>.json`     ruta con la MISMA forma que /checkpoints-list
                     (id, sequence, latitude, longitude) -> la consume
                     cualquier cosa que ya entienda `Checkpoint`.
  * `<out>.geojson`  el trazo crudo + los waypoints, para tirar en
                     geojson.io y mirar si la ruta quedo razonable.

Criterio de muestreo (default: 1 m, ver la nota de abajo):
  - se graba un waypoint cada `--spacing-m` metros, y ademas
  - en cada curva: si el rumbo cambio mas de `--corner-deg` grados desde el
    ultimo waypoint (con al menos `--corner-min-m` metros de separacion),
    para que las esquinas no queden cortadas.
  - Enter en la terminal fuerza un waypoint en el lugar (util para marcar
    "aca hay que pasar si o si").

Que significa 1 m aca: el GPS del Mini+ entrega un fix nuevo del orden de 1 Hz
y con error absoluto de varios metros, asi que 1 m NO es precision de 1 m. Es
densidad: el resultado es una polilinea que describe la FORMA del ride. Se
consume como camino (lookahead alto en WaypointRoute, meta ~2 m adelante), no
como metas discretas: bajar `reach_radius_m` a 1 m para "respetarlo" deja al
rover orbitando puntos que nunca alcanza. Si andas despacio (<1 m/s) van a
salir waypoints separados por menos de 1 m, porque cada fix nuevo que supera
el umbral entra: eso es correcto, el filtro es un piso, no una grilla.

Filtros de GPS:
  - fixes con lat/lon 0 o fuera de rango se descartan;
  - saltos que implican mas de `--max-jump-speed-ms` m/s se descartan como
    outliers del GPS (el Mini+ los tira de a ratos);
  - `--min-gps-signal` descarta fixes con senal pobre (0 = filtro apagado).
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

EARTH_RADIUS_M = 6_371_000.0


def bearing_and_distance(lat1: float, lon1: float,
                         lat2: float, lon2: float) -> tuple[float, float]:
    """(rumbo_deg_desde_norte, distancia_m). Aproximacion equirectangular:
    de sobra a escala de decenas/cientos de metros."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat_media = math.radians((lat1 + lat2) / 2.0)
    este = dlon * math.cos(lat_media) * EARTH_RADIUS_M
    norte = dlat * EARTH_RADIUS_M
    return math.degrees(math.atan2(este, norte)), math.hypot(este, norte)


def wrap_deg(a: float) -> float:
    a = float(a) % 360.0
    return a - 360.0 if a > 180.0 else a


def fix_valido(lat, lon) -> bool:
    if lat is None or lon is None:
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if abs(lat) > 90.0 or abs(lon) > 180.0:
        return False
    # (0, 0) es "sin fix", no la isla Null.
    return not (abs(lat) < 1e-7 and abs(lon) < 1e-7)


class Grabador:
    def __init__(self, args):
        self.args = args
        self.track: list[dict] = []       # todos los fixes aceptados
        self.waypoints: list[dict] = []   # los que sobreviven al muestreo
        self._ultimo_fix: tuple[float, float, float] | None = None  # lat, lon, t
        self._rumbo_ultimo_wp: float | None = None
        self.descartes = {"invalido": 0, "senal": 0, "salto": 0, "repetido": 0}

    # ------------------------------------------------------------ muestreo

    def _agregar_waypoint(self, lat: float, lon: float, t: float,
                          motivo: str, extra: dict) -> None:
        seq = len(self.waypoints) + 1
        self.waypoints.append({
            "id": seq,
            "sequence": seq,
            "latitude": lat,
            "longitude": lon,
            "timestamp": t,
            "motivo": motivo,
            **extra,
        })
        print(f"  waypoint #{seq:>3}  {lat:.7f}, {lon:.7f}   ({motivo})")

    def considerar(self, lat: float, lon: float, t: float,
                   forzado: bool, extra: dict) -> None:
        if not self.waypoints:
            self._agregar_waypoint(lat, lon, t, "inicio", extra)
            return

        ult = self.waypoints[-1]
        rumbo, dist = bearing_and_distance(ult["latitude"], ult["longitude"], lat, lon)

        if forzado:
            self._agregar_waypoint(lat, lon, t, "manual",
                                   {**extra, "distance_from_prev_m": round(dist, 2)})
            self._rumbo_ultimo_wp = rumbo
            return

        if dist >= self.args.spacing_m:
            motivo = "espaciado"
        elif (self._rumbo_ultimo_wp is not None
              and dist >= self.args.corner_min_m
              and abs(wrap_deg(rumbo - self._rumbo_ultimo_wp)) >= self.args.corner_deg):
            motivo = "curva"
        else:
            return

        self._agregar_waypoint(lat, lon, t, motivo,
                               {**extra, "distance_from_prev_m": round(dist, 2)})
        self._rumbo_ultimo_wp = rumbo

    # --------------------------------------------------------------- fixes

    def ingerir(self, d: dict, forzado: bool) -> None:
        lat, lon = d.get("latitude"), d.get("longitude")
        if not fix_valido(lat, lon):
            self.descartes["invalido"] += 1
            return
        lat, lon = float(lat), float(lon)

        senal = d.get("gps_signal")
        if self.args.min_gps_signal > 0 and senal is not None:
            try:
                if float(senal) < self.args.min_gps_signal:
                    self.descartes["senal"] += 1
                    return
            except (TypeError, ValueError):
                pass

        t = float(d.get("timestamp") or time.time())

        # El GPS entrega un fix nuevo ~1 vez por segundo. Si leemos mas rapido
        # que eso, el SDK devuelve el MISMO fix otra vez: descartarlo por
        # timestamp evita inflar el track con puntos que no son mediciones
        # nuevas (y evita que un dt~0 dispare el filtro de salto).
        if self._ultimo_fix is not None and t <= self._ultimo_fix[2]:
            self.descartes["repetido"] += 1
            return

        if self._ultimo_fix is not None:
            lat0, lon0, t0 = self._ultimo_fix
            dt = max(1e-3, t - t0)
            _, salto = bearing_and_distance(lat0, lon0, lat, lon)
            if salto / dt > self.args.max_jump_speed_ms:
                self.descartes["salto"] += 1
                return

        self._ultimo_fix = (lat, lon, t)
        self.track.append({"latitude": lat, "longitude": lon, "timestamp": t})

        extra = {}
        if senal is not None:
            extra["gps_signal"] = senal
        self.considerar(lat, lon, t, forzado, extra)

    # -------------------------------------------------------------- salida

    def largo_track_m(self) -> float:
        total = 0.0
        for a, b in zip(self.track, self.track[1:]):
            total += bearing_and_distance(a["latitude"], a["longitude"],
                                          b["latitude"], b["longitude"])[1]
        return total

    def guardar(self, destino: Path) -> None:
        if not self.waypoints:
            print("\nNo se grabo ningun waypoint: no guardo nada.")
            return
        destino.parent.mkdir(parents=True, exist_ok=True)

        doc = {
            "version": 1,
            "name": self.args.name or destino.stem,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sdk_url": self.args.sdk_url,
            "sampling": {
                "spacing_m": self.args.spacing_m,
                "corner_deg": self.args.corner_deg,
                "corner_min_m": self.args.corner_min_m,
                "rate_hz": self.args.rate_hz,
            },
            "track_length_m": round(self.largo_track_m(), 1),
            "track_fixes": len(self.track),
            "count": len(self.waypoints),
            "waypoints": self.waypoints,
        }
        destino.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

        geo = destino.with_suffix(".geojson")
        geo.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature",
                 "properties": {"tipo": "track"},
                 "geometry": {"type": "LineString",
                              "coordinates": [[p["longitude"], p["latitude"]]
                                              for p in self.track]}},
                *[{"type": "Feature",
                   "properties": {"sequence": w["sequence"], "motivo": w["motivo"]},
                   "geometry": {"type": "Point",
                                "coordinates": [w["longitude"], w["latitude"]]}}
                  for w in self.waypoints],
            ],
        }, indent=2), encoding="utf-8")

        largo = doc["track_length_m"]
        print(f"\n{len(self.waypoints)} waypoints sobre {largo:.0f} m de recorrido "
              f"({largo / max(1, len(self.waypoints) - 1):.1f} m promedio entre puntos)")
        print(f"descartes: {self.descartes}")
        print(f"guardado:  {destino}")
        print(f"           {geo}   (arrastralo a geojson.io)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdk-url", default="http://localhost:8000")
    ap.add_argument("--out", required=True, help="ruta del .json de salida")
    ap.add_argument("--name", default=None)
    ap.add_argument("--rate-hz", type=float, default=5.0,
                    help="frecuencia de lectura de /data. Mas alta que la tasa "
                         "de fix del GPS no densifica nada, pero no molesta: "
                         "los fixes repetidos se descartan por timestamp")
    ap.add_argument("--spacing-m", type=float, default=1.0,
                    help="metros entre waypoints (default 1). A esta escala el "
                         "trazo es un CAMINO, no metas discretas: seguilo con "
                         "lookahead alto, no con reach_radius chico")
    ap.add_argument("--corner-deg", type=float, default=15.0)
    ap.add_argument("--corner-min-m", type=float, default=0.4,
                    help="tiene que ser menor que --spacing-m o la regla de "
                         "curva no dispara nunca")
    ap.add_argument("--min-gps-signal", type=float, default=0.0)
    ap.add_argument("--max-jump-speed-ms", type=float, default=5.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    if args.corner_min_m >= args.spacing_m:
        print(f"[rec] AVISO: --corner-min-m ({args.corner_min_m}) >= --spacing-m "
              f"({args.spacing_m}): la regla de curva no va a disparar nunca, "
              f"todo waypoint va a salir por espaciado.")

    destino = Path(args.out)
    grabador = Grabador(args)
    parar = threading.Event()
    marcar = threading.Event()

    def _pedir_parada(*_a):
        print("\n[rec] cortando ...")
        parar.set()

    signal.signal(signal.SIGINT, _pedir_parada)
    signal.signal(signal.SIGTERM, _pedir_parada)

    if sys.stdin is not None and sys.stdin.isatty():
        def _lector():
            try:
                for _ in sys.stdin:
                    marcar.set()
            except Exception:
                pass
        threading.Thread(target=_lector, daemon=True).start()
        print("[rec] Enter = marcar waypoint a mano | Ctrl-C = terminar y guardar")

    sesion = requests.Session()
    url = args.sdk_url.rstrip("/") + "/data"
    intervalo = 1.0 / max(0.5, args.rate_hz)
    t_inicio = time.monotonic()
    fallas = 0

    print(f"[rec] grabando desde {url} a {args.rate_hz:g} Hz ...")
    try:
        while not parar.is_set():
            ciclo = time.monotonic()
            try:
                r = sesion.get(url, timeout=3.0)
                r.raise_for_status()
                grabador.ingerir(r.json(), forzado=marcar.is_set())
                marcar.clear()
                fallas = 0
            except requests.RequestException as exc:
                fallas += 1
                if fallas in (1, 5) or fallas % 25 == 0:
                    print(f"[rec] /data falla ({fallas}): {exc}")
            except ValueError:
                print("[rec] /data devolvio algo que no es JSON")

            if args.max_seconds and time.monotonic() - t_inicio > args.max_seconds:
                print("\n[rec] alcanzado --max-seconds")
                break
            parar.wait(max(0.0, intervalo - (time.monotonic() - ciclo)))
    finally:
        sesion.close()
        grabador.guardar(destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
