"""Ruta grabada (breadcrumbs GPS) usada como guia secundaria entre checkpoints
oficiales. Lee los archivos de tools/record_waypoints.py (formato
{"waypoints": [...]}, guardados en rutas/ en la raiz del repo) y sus
.geojson equivalentes.

La idea es un pure pursuit sobre GPS: se proyecta la posicion sobre la
polilinea grabada, se lleva un "progreso" (metros recorridos sobre la ruta)
que NUNCA retrocede, y la meta es el punto de la ruta que esta lookahead_m
por delante de ese progreso. Con waypoints cada ~1 m y un GPS que salta
1-3 m, apuntar al waypoint mas cercano haria zigzag; apuntar varios metros
adelante sobre la ruta filtra ese ruido solo.

Los puntos de la ruta NO son obligatorios: nunca se reclaman en el SDK, y
si alguno queda tapado el progreso salta adelante (estancado_s). Quien
decide cuando seguir la ruta y cuando ir directo al checkpoint oficial es
el bridge.

No depende de nada del bridge (numpy puro), asi que se prueba offline.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

R_TIERRA_M = 6371000.0

# genie/genie_rover/route.py -> raiz del repo
REPO_ROOT = Path(__file__).resolve().parents[2]
RUTAS_DIRS = [
    REPO_ROOT / "genie" / "rutas",
    REPO_ROOT / "rutas",
    Path.cwd() / "genie" / "rutas",
    Path.cwd() / "rutas",
]
DASHBOARD_JSON_DEFAULT = REPO_ROOT / "earth-rovers-sdk" / "static" / "genie_waypoints.json"


def resolver_ruta(nombre: str | Path) -> Path:
    """Acepta una ruta tal cual, relativa al directorio actual o a la raiz
    del repo, o solo el nombre de un archivo de rutas/ (con o sin .json)."""
    p = Path(nombre).expanduser()
    candidatos = [p, REPO_ROOT / p]
    for d in RUTAS_DIRS:
        candidatos.append(d / p)
        if not p.suffix:
            candidatos.append(d / f"{p}.json")
            candidatos.append(d / f"{p}.geojson")
    if not p.suffix:
        candidatos += [p.with_suffix(".json"), p.with_suffix(".geojson")]
    for c in candidatos:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"No encuentro la ruta '{nombre}' (busque en {RUTAS_DIRS})")


@dataclass
class RoutePoint:
    lat: float
    lon: float
    motivo: str = ""


def cargar_ruta(path: str | Path) -> list[RoutePoint]:
    """Lee un archivo de ruta: el JSON del grabador ({"waypoints": [...]})
    o un GeoJSON FeatureCollection de Points con properties.sequence."""
    data = json.loads(Path(path).read_text())
    if data.get("type") == "FeatureCollection":
        feats = [f for f in data.get("features", [])
                 if f.get("geometry", {}).get("type") == "Point"]
        feats.sort(key=lambda f: f.get("properties", {}).get("sequence", 0))
        return [RoutePoint(float(f["geometry"]["coordinates"][1]),
                           float(f["geometry"]["coordinates"][0]),
                           str(f.get("properties", {}).get("motivo", "")))
                for f in feats]
    wps = sorted(data["waypoints"], key=lambda w: w.get("sequence", 0))
    return [RoutePoint(float(w["latitude"]), float(w["longitude"]),
                       str(w.get("motivo", ""))) for w in wps]


@dataclass
class RouteConfig:
    lookahead_m: float = 4.0            # meta = progreso + esto, sobre la ruta
    ventana_atras_m: float = 2.0        # donde buscar la proyeccion (hacia atras)
    ventana_adelante_m: float = 8.0     # ... y hacia adelante
    vel_max_mps: float = 0.6            # el progreso no avanza mas rapido que esto
    margen_avance_m: float = 0.2        # ... mas este margen por fix
    suavizado_s: float = 2.0            # promedio de posicion GPS (anti-trinquete)
    fuera_de_ruta_m: float = 6.0        # mas lejos que esto: buscar reenganche adelante
    fin_m: float = 1.5                  # a esto del final la ruta se da por terminada
    estancado_s: float = 25.0           # sin progreso este tiempo -> saltar adelante
    salto_estancado_m: float = 2.0      # cuanto saltar
    min_sep_m: float = 0.3              # puntos mas juntos que esto se descartan


class RouteFollower:
    def __init__(self, puntos: list[RoutePoint], cfg: RouteConfig | None = None):
        if len(puntos) < 2:
            raise ValueError("La ruta necesita al menos 2 puntos")
        self.cfg = cfg or RouteConfig()
        self.lat0, self.lon0 = puntos[0].lat, puntos[0].lon
        self._cos0 = math.cos(math.radians(self.lat0))

        xy = [self.to_local(puntos[0].lat, puntos[0].lon)]
        self.puntos = [puntos[0]]          # los que sobreviven al descarte
        for p in puntos[1:]:
            q = self.to_local(p.lat, p.lon)
            if math.hypot(q[0] - xy[-1][0], q[1] - xy[-1][1]) >= self.cfg.min_sep_m:
                xy.append(q)
                self.puntos.append(p)
        if len(xy) < 2:
            raise ValueError("La ruta colapsa a un solo punto (todos los fixes juntos)")
        self.xy = np.asarray(xy, dtype=np.float64)          # (n, 2): norte, este
        seg = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])     # arco acumulado
        self.total_m = float(self.s[-1])

        self.progreso_m: float | None = None
        self.desvio_m = float("inf")
        self.saltos = 0
        self.reenganches = 0
        self._ult_progreso = 0.0
        self._ult_avance_t: float | None = None
        self._ult_t: float | None = None
        self._buf: list[tuple[float, np.ndarray]] = []

    # ------------------------------------------------------------ geometria

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        n = math.radians(lat - self.lat0) * R_TIERRA_M
        e = math.radians(lon - self.lon0) * R_TIERRA_M * self._cos0
        return n, e

    def to_latlon(self, n: float, e: float) -> tuple[float, float]:
        lat = self.lat0 + math.degrees(n / R_TIERRA_M)
        lon = self.lon0 + math.degrees(e / (R_TIERRA_M * self._cos0))
        return lat, lon

    def _proyectar(self, p: np.ndarray, s_min: float, s_max: float) -> tuple[float, float]:
        """Punto de la ruta mas cercano a p con arco en [s_min, s_max].
        Devuelve (distancia_m, arco_m)."""
        a, b = self.xy[:-1], self.xy[1:]
        s0, s1 = self.s[:-1], self.s[1:]
        ok = (s1 >= s_min) & (s0 <= s_max)
        if not np.any(ok):
            s_c = float(np.clip(s_min, 0.0, self.total_m))
            q = self.punto_en(s_c)
            return float(np.linalg.norm(p - q)), s_c
        a, b, s0, s1 = a[ok], b[ok], s0[ok], s1[ok]
        ab = b - a
        L = np.maximum(s1 - s0, 1e-9)
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / (L * L), 0.0, 1.0)
        s_q = np.clip(s0 + t * L, s_min, s_max)
        # recalcular el punto por si el clip de s lo movio
        q = np.stack([self.punto_en(float(v)) for v in s_q])
        d = np.linalg.norm(q - p, axis=1)
        i = int(np.argmin(d))
        return float(d[i]), float(s_q[i])

    def punto_en(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.total_m))
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2))
        L = max(self.s[i + 1] - self.s[i], 1e-9)
        t = (s - self.s[i]) / L
        return self.xy[i] + t * (self.xy[i + 1] - self.xy[i])

    # --------------------------------------------------------------- estado

    def update(self, lat: float, lon: float, t: float) -> None:
        c = self.cfg
        crudo = np.asarray(self.to_local(lat, lon))
        # Promedio de los fixes recientes. Sin esto el ruido GPS funciona
        # como un trinquete: como el progreso nunca retrocede, cada fix que
        # cae adelante lo empuja y los que caen atras no lo devuelven.
        self._buf.append((t, crudo))
        self._buf = [(tt, q) for tt, q in self._buf if t - tt <= c.suavizado_s]
        p = np.mean([q for _, q in self._buf], axis=0)
        dt = 0.2 if self._ult_t is None else max(0.0, t - self._ult_t)
        self._ult_t = t

        if self.progreso_m is None:
            # Primer fix: el robot puede arrancar a mitad de ruta.
            d, s_q = self._proyectar(p, 0.0, self.total_m)
            self.progreso_m = s_q
        else:
            d, s_q = self._proyectar(p, self.progreso_m - c.ventana_atras_m,
                                     self.progreso_m + c.ventana_adelante_m)
            if s_q > self.progreso_m:
                self.progreso_m = min(s_q, self.progreso_m + c.vel_max_mps * dt + c.margen_avance_m)

        if d > c.fuera_de_ruta_m:
            # Muy afuera (rodeo largo o GPS malo): buscar el punto mas cercano
            # de TODO lo que queda por delante y engancharse ahi.
            d2, s2 = self._proyectar(p, self.progreso_m, self.total_m)
            if d2 < d:
                d = d2
                if d2 < c.fuera_de_ruta_m and s2 > self.progreso_m:
                    self.progreso_m = s2
                    self.reenganches += 1
        self.desvio_m = d

        if self._ult_avance_t is None or self.progreso_m > self._ult_progreso + 0.5:
            self._ult_progreso = self.progreso_m
            self._ult_avance_t = t
        elif t - self._ult_avance_t > c.estancado_s and not self.terminada:
            # Algo tapa la ruta y no se puede pasar por ahi: tomar la
            # siguiente miga de pan como meta.
            self.progreso_m = min(self.total_m, self.progreso_m + c.salto_estancado_m)
            self.saltos += 1
            self._ult_progreso = self.progreso_m
            self._ult_avance_t = t

    @property
    def terminada(self) -> bool:
        return self.progreso_m is not None and self.progreso_m >= self.total_m - self.cfg.fin_m

    def indice_actual(self) -> int:
        """Indice (en self.puntos) del primer punto que todavia esta adelante."""
        if self.progreso_m is None:
            return 0
        return int(min(len(self.s), np.searchsorted(self.s, self.progreso_m + 1e-6, side="right")))

    def objetivo(self) -> tuple[float, float]:
        """(lat, lon) del punto lookahead_m adelante del progreso."""
        s = (self.progreso_m or 0.0) + self.cfg.lookahead_m
        n, e = self.punto_en(s)
        return self.to_latlon(float(n), float(e))

    def descripcion(self) -> str:
        prog = self.progreso_m or 0.0
        return (f"ruta {prog:.1f}/{self.total_m:.1f} m desvio {self.desvio_m:.1f} m"
                + (f" saltos={self.saltos}" if self.saltos else ""))


def cargar_rutas(nombres: list[str], cfg: RouteConfig | None = None) -> RouteFollower | None:
    """Concatena en orden los archivos de ruta y arma el follower.
    Devuelve None si no se paso ninguna ruta."""
    puntos: list[RoutePoint] = []
    for n in nombres:
        path = resolver_ruta(n)
        pts = cargar_ruta(path)
        print(f"[ruta] {path.name}: {len(pts)} puntos")
        puntos += pts
    if not puntos:
        return None
    follower = RouteFollower(puntos, cfg)
    follower.nombre = "+".join(nombres)
    return follower


def gps_valido(lat: float | None, lon: float | None) -> bool:
    """El SDK manda 0,0 cuando todavia no hay fix."""
    if lat is None or lon is None:
        return False
    return abs(float(lat)) > 1e-6 or abs(float(lon)) > 1e-6


class DashboardOverlay:
    """Escribe static/genie_waypoints.json para el overlay del mapa del SDK
    (static/map.js y static/dashboard-map.js, modo "frame": "gps").

    Atomico (tmp + rename): el navegador lo lee cada ~1.5 s y no tiene que
    encontrarse nunca un JSON a medio escribir."""

    def __init__(self, path: str | Path, periodo_s: float = 1.0, mode: str = "ruta"):
        self.path = Path(path)
        self.periodo_s = float(periodo_s)
        self.mode = mode
        self._t = 0.0
        self._trail: list[list[float]] = []

    def escribir(self, ruta: RouteFollower | None, lat: float | None, lon: float | None,
                 estado: str, target: dict | None, checkpoints_done: int,
                 force: bool = False) -> None:
        ok = gps_valido(lat, lon)
        if ok and (not self._trail
                   or abs(self._trail[-1][0] - lat) > 2e-6
                   or abs(self._trail[-1][1] - lon) > 2e-6):
            self._trail.append([float(lat), float(lon)])
            self._trail = self._trail[-600:]

        now = time.time()
        if not force and now - self._t < self.periodo_s:
            return
        self._t = now

        payload: dict = {
            "active": True,
            "frame": "gps",
            "mode": self.mode,
            "ts": now,
            "state": estado,
            "checkpoints_done": int(checkpoints_done),
            "pose_gps": {"lat": float(lat), "lon": float(lon)} if ok else None,
            "target": target,
            "trail_gps": self._trail,
        }
        if ruta is not None:
            payload.update({
                "waypoints": [{"lat": p.lat, "lon": p.lon, "motivo": p.motivo}
                              for p in ruta.puntos],
                "current_index": ruta.indice_actual(),
                "route_done": bool(ruta.terminada),
                "progress_m": round(ruta.progreso_m or 0.0, 2),
                "total_m": round(ruta.total_m, 2),
                "desvio_m": round(ruta.desvio_m, 2) if math.isfinite(ruta.desvio_m) else None,
                "saltos": ruta.saltos,
            })
        self._dump(payload)

    def apagar(self) -> None:
        """Deja el dashboard como el original."""
        self._dump({"active": False, "ts": time.time()})

    def _dump(self, payload: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self.path)
        except Exception as exc:
            print(f"[ruta] no pude escribir {self.path}: {exc}")
