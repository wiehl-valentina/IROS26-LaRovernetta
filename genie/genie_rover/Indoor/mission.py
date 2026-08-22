"""Maquina de estados de la mision indoor: SEARCH -> NAVIGATE -> APPROACH ->
VERIFY -> STOP (el diagrama de Documentos/Modulos/Indoor/Navegacion_sin_gps.html).

Simplificacion consciente respecto del doc: SEARCH y NAVIGATE comparten el
mismo mecanismo de manejo (conducir evitando obstaculos via plan_on_bev,
igual que bridge.py sin GPS) y solo difieren en COMO se elige la meta local:

    SEARCH    no hay cono a la vista (ni memoria reciente de uno): la meta
              sale de la estrategia de busqueda (frontier / waypoints /
              derecho adelante).
    NAVIGATE  el cono esta a la vista (o se perdio hace poco) pero todavia
              lejos: la meta es la posicion del cono en el piso.
    APPROACH  igual que NAVIGATE pero mas cerca (approach_start_m): se limita
              la velocidad lineal (approach_max_linear_scale) para no llegar
              rapido a algo que todavia no esta confirmado.
    VERIFY    lo bastante cerca (verify_at_m): exige varias detecciones
              seguidas con confianza y posicion estables antes de confiar en
              que es el cono de verdad (evita frenar por un falso positivo de
              un solo frame). Opcionalmente confirma con un VLM
              (programs/client/genai_client, ya existe en el repo para
              vlm_recovery — mismo cliente, otro uso).
    STOP      confirmado y a stop_distance_m o menos: frena y no se mueve
              mas. Terminal (mission_done=True para siempre).

IMPORTANTE (misma logica de seguridad que bridge.py): esta maquina de
estados solo decide la META (x_right_m, y_forward_m) que se le pasa a
plan_on_bev. El camino en si sigue evitando obstaculos porque plan_on_bev y
front_is_blocked no cambian — a diferencia del pseudocodigo
"steering = Kp*error" del doc original (que dirige directo al cono sin mirar
el piso), aca la aproximacion al cono sigue pasando por el mismo planificador
de transitabilidad que el resto del recorrido. Es una desviacion deliberada
del doc, mas segura para un pasillo real con gente/mobiliario.

Autoprueba (no necesita robot, camara ni modelo):
    python -m genie_rover.mission
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .cone_detector import ConeDetection, GroundPoint
from ..odometry import Pose

if TYPE_CHECKING:  # solo para hints, sin crear un import circular en runtime
    from ..persistent_map import PersistentMap


# ------------------------------------------------------------------- config

@dataclass
class MissionConfig:
    search_mode: str = "wander"          # "wander" | "frontier" | "waypoints"
    waypoints_path: str | None = None
    waypoint_reach_radius_m: float = 0.6
    wander_forward_m: float = 3.0        # igual al fallback "derecho adelante" de bridge.py

    # --- exploracion por frontera (Opcion B del doc: mapa desconocido) -----
    frontier_free_thresh: float = 0.55
    frontier_min_m: float = 0.6
    frontier_max_m: float = 3.5
    frontier_preferred_m: float = 1.6
    frontier_fov_deg: float = 140.0
    frontier_goal_hold_s: float = 4.0

    # --- deteccion / aproximacion ------------------------------------------
    detect_confidence_min: float = 0.45
    approach_start_m: float = 2.0        # Fase 4 del doc
    verify_at_m: float = 0.8
    stop_distance_m: float = 0.5         # Fase 4 del doc
    approach_max_linear_scale: float = 0.6
    lost_cone_grace_s: float = 1.5

    # --- verificacion --------------------------------------------------------
    verify_hits_required: int = 3
    verify_window: int = 6
    verify_min_confidence: float = 0.55
    verify_max_jump_m: float = 0.6
    verify_with_vlm: bool = False
    vlm_timeout_s: float = 4.0
    vlm_min_confidence: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "MissionConfig":
        return cls(**(d or {}))


class MissionState(str, Enum):
    SEARCH = "SEARCH"
    NAVIGATE = "NAVIGATE"
    APPROACH = "APPROACH"
    VERIFY = "VERIFY"
    STOP = "STOP"


@dataclass
class MissionGoal:
    x_right_m: float
    y_forward_m: float
    state: str
    reason: str
    mission_done: bool = False
    linear_scale: float = 1.0


# --------------------------------------------------------- ruta de waypoints

class WaypointRoute:
    """Opcion A del doc: mapa conocido, secuencia de puntos a recorrer.

    Los puntos son (x_m, y_m) en el marco de ODOMETRIA (donde arranco el
    rover al prender el bridge), NO coordenadas GPS. Se arman midiendo a
    mano (o con tools/diag_odometry.py) cuanto avanza cada tramo del
    recorrido conocido: "derecho 5 m, doblar, derecho 3 m", etc.
    """

    def __init__(self, waypoints: list[tuple[float, float]], reach_radius_m: float = 0.6):
        self.points = list(waypoints)
        self.reach_radius_m = float(reach_radius_m)
        self.idx = 0

    @property
    def done(self) -> bool:
        return self.idx >= len(self.points)

    def current_goal(self, pose: Pose) -> GroundPoint | None:
        if self.done:
            return None
        tx, ty = self.points[self.idx]
        rel = Pose(tx, ty, 0.0).relative_to(pose)
        dist = math.hypot(rel.x, rel.y)
        if dist <= self.reach_radius_m:
            self.idx += 1
            return self.current_goal(pose)
        return GroundPoint(x_forward_m=rel.x, y_left_m=rel.y, distance_m=dist)

    @classmethod
    def from_file(cls, path: str, reach_radius_m: float = 0.6) -> "WaypointRoute":
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        pts = [(float(p["x_m"]), float(p["y_m"])) for p in data.get("waypoints", [])]
        return cls(pts, reach_radius_m=reach_radius_m)


# ------------------------------------------------------- exploracion frontera

def pick_frontier_goal(pmap: "PersistentMap", pose: Pose, cfg: MissionConfig
                       ) -> tuple[float, float] | None:
    """Elige un punto del MUNDO (x, y) en el borde entre lo conocido-libre y
    lo nunca-visto, para que el rover siga explorando sin repetir pasillo.

    Opera directamente sobre las grillas numpy de PersistentMap (value, conf)
    — reutiliza el mapa que bridge.py ya mantiene para memoria espacial, no
    agrega ningun estado propio. Devuelve None si no hay frontera candidata
    (mapa todavia vacio, o todo lo visible ya fue explorado), y quien llama
    debe caer a otra estrategia (wander).
    """
    conf, val = pmap.conf, pmap.value
    known = conf >= pmap.cfg.min_confidence
    free = known & (val > cfg.frontier_free_thresh)
    unknown = ~known

    if not np.any(free) or not np.any(unknown):
        return None

    padded = np.pad(unknown, 1, mode="constant", constant_values=True)
    neighbor_unknown = (
        padded[0:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, 0:-2] | padded[1:-1, 2:]
    )
    frontier_mask = free & neighbor_unknown
    fr, fc = np.nonzero(frontier_mask)
    if fr.size == 0:
        return None

    r = pmap.cfg.resolution_m_per_px
    n = pmap.n
    x_world = pmap.origin_x + (n / 2 - fr) * r
    y_world = pmap.origin_y + (n / 2 - fc) * r

    dx = x_world - pose.x
    dy = y_world - pose.y
    c, s = math.cos(-pose.theta), math.sin(-pose.theta)
    forward = c * dx - s * dy
    left = s * dx + c * dy
    dist = np.hypot(forward, left)

    fov = math.radians(cfg.frontier_fov_deg / 2.0)
    ang = np.arctan2(left, np.maximum(forward, 1e-6))
    in_cone = (np.abs(ang) <= fov) & (forward > 0)
    in_range = (dist >= cfg.frontier_min_m) & (dist <= cfg.frontier_max_m)

    candidatos = in_cone & in_range
    if not np.any(candidatos):
        # Nada adelante en rango: probablemente hay que girar/retroceder para
        # ver mas. Relajamos el cono de vision (cualquier direccion) antes de
        # rendirnos del todo — mejor intentar una frontera "rara" que quedar
        # sin meta y depender solo del barrido ciego de _recover().
        candidatos = in_range
        if not np.any(candidatos):
            candidatos = np.ones_like(dist, dtype=bool)

    idx_validos = np.nonzero(candidatos)[0]
    score = np.abs(dist[idx_validos] - cfg.frontier_preferred_m)
    mejor = idx_validos[int(np.argmin(score))]

    return float(x_world[mejor]), float(y_world[mejor])


# ---------------------------------------------------------- maquina de estados

class ConeMissionFSM:
    def __init__(self, cfg: MissionConfig):
        self.cfg = cfg
        self.state = MissionState.SEARCH
        self._route: WaypointRoute | None = None
        if cfg.search_mode == "waypoints" and cfg.waypoints_path:
            self._route = WaypointRoute.from_file(cfg.waypoints_path, cfg.waypoint_reach_radius_m)

        self._frontier_world: tuple[float, float] | None = None
        self._frontier_goal_t: float = 0.0

        self._hits: deque[tuple[float, float, float]] = deque(maxlen=cfg.verify_window)
        self._last_cone_ground: GroundPoint | None = None
        self._last_cone_t: float | None = None
        self._vlm_confirmed = False

    # ------------------------------------------------------------- publico

    def update(self, pose: Pose, pmap: "PersistentMap | None",
              cone: ConeDetection | None, cone_ground: GroundPoint | None,
              now: float) -> MissionGoal:
        if self.state == MissionState.STOP:
            return MissionGoal(0.0, 0.0, self.state.value,
                               "mision cumplida: cono alcanzado", mission_done=True)

        cone_visible = (cone is not None and cone_ground is not None
                        and cone.confidence >= self.cfg.detect_confidence_min)
        if cone_visible:
            self._last_cone_ground = cone_ground
            self._last_cone_t = now
            self._hits.append((now, cone.confidence, cone_ground.distance_m))

        grace_ok = (self._last_cone_t is not None
                   and (now - self._last_cone_t) <= self.cfg.lost_cone_grace_s)
        anchored = cone_visible or (grace_ok and self._last_cone_ground is not None)

        if not anchored:
            if self.state != MissionState.SEARCH:
                self.state = MissionState.SEARCH
                self._hits.clear()
            goal = self._search_goal(pose, pmap, now)
            x_right, y_fwd = goal.to_bev_goal()
            return MissionGoal(x_right, y_fwd, self.state.value,
                               f"explorando (modo={self.cfg.search_mode})")

        ground = self._last_cone_ground
        assert ground is not None
        dist = ground.distance_m

        if dist <= self.cfg.verify_at_m:
            self.state = MissionState.VERIFY
        elif dist <= self.cfg.approach_start_m:
            self.state = MissionState.APPROACH
        else:
            self.state = MissionState.NAVIGATE

        x_right, y_fwd = ground.to_bev_goal()

        if self.state == MissionState.VERIFY:
            stable = self._check_stable_hits()
            confirmado = stable and (not self.cfg.verify_with_vlm or self._vlm_confirmed)
            if confirmado and dist <= self.cfg.stop_distance_m:
                self.state = MissionState.STOP
                return MissionGoal(0.0, 0.0, self.state.value,
                                   f"cono verificado a {dist:.2f} m, freno",
                                   mission_done=True)
            razon = (f"verificando ({len(self._hits)} hits, estable={stable}, "
                    f"dist={dist:.2f} m)")
            return MissionGoal(x_right, y_fwd, self.state.value, razon,
                               linear_scale=self.cfg.approach_max_linear_scale)

        scale = self.cfg.approach_max_linear_scale if self.state == MissionState.APPROACH else 1.0
        return MissionGoal(x_right, y_fwd, self.state.value,
                           f"{self.state.value.lower()} hacia el cono (dist={dist:.2f} m)",
                           linear_scale=scale)

    def confirm_with_vlm(self, rgb_crop_bytes: bytes) -> bool:
        """Confirmacion opcional con Gemini (programs.client.genai_client, el
        mismo cliente que vlm_recovery usa en el resto del repo). Nunca
        lanza: si falla, la maquina de estados sigue esperando mas evidencia
        geometrica en vez de confiar ciegamente en una llamada que no llego.
        """
        try:
            from pydantic import BaseModel, Field

            from programs.client import genai_client

            class _EsCono(BaseModel):
                es_cono_de_trafico: bool = Field(
                    description="true si la imagen muestra un cono de trafico naranja")
                confianza: float = Field(description="0 a 1")

            genai_client.load_credentials()
            out = genai_client.ask_image_structured(
                rgb_crop_bytes,
                "Este es un recorte de la camara de un robot. Responde si "
                "muestra un cono de trafico/obra (tipicamente naranja con "
                "franjas blancas).",
                _EsCono, timeout_s=self.cfg.vlm_timeout_s,
            )
            ok = bool(out.es_cono_de_trafico) and out.confianza >= self.cfg.vlm_min_confidence
            self._vlm_confirmed = ok
            return ok
        except Exception as exc:
            print(f"[mission] verificacion VLM fallo ({exc}); sigo solo con la geometrica")
            return False

    # -------------------------------------------------------------- privado

    def _check_stable_hits(self) -> bool:
        if len(self._hits) < self.cfg.verify_hits_required:
            return False
        recent = list(self._hits)
        n_ok = sum(1 for _, c, _ in recent if c >= self.cfg.verify_min_confidence)
        if n_ok < self.cfg.verify_hits_required:
            return False
        dists = [d for _, _, d in recent]
        return (max(dists) - min(dists)) <= self.cfg.verify_max_jump_m

    def _search_goal(self, pose: Pose, pmap: "PersistentMap | None", now: float) -> GroundPoint:
        if self.cfg.search_mode == "waypoints" and self._route is not None and not self._route.done:
            g = self._route.current_goal(pose)
            if g is not None:
                return g

        if self.cfg.search_mode == "frontier" and pmap is not None:
            need_new = (self._frontier_world is None
                       or (now - self._frontier_goal_t) > self.cfg.frontier_goal_hold_s)
            if not need_new:
                tx, ty = self._frontier_world
                rel = Pose(tx, ty, 0.0).relative_to(pose)
                if math.hypot(rel.x, rel.y) < self.cfg.waypoint_reach_radius_m:
                    need_new = True
            if need_new:
                picked = pick_frontier_goal(pmap, pose, self.cfg)
                if picked is not None:
                    self._frontier_world = picked
                    self._frontier_goal_t = now
            if self._frontier_world is not None:
                tx, ty = self._frontier_world
                rel = Pose(tx, ty, 0.0).relative_to(pose)
                return GroundPoint(rel.x, rel.y, math.hypot(rel.x, rel.y))

        return GroundPoint(x_forward_m=self.cfg.wander_forward_m, y_left_m=0.0,
                           distance_m=self.cfg.wander_forward_m)


# --------------------------------------------------------------------- pruebas

def _self_test() -> None:
    print("=== WaypointRoute ===")
    route = WaypointRoute([(2.0, 0.0), (2.0, 2.0)], reach_radius_m=0.3)
    g = route.current_goal(Pose(0, 0, 0))
    print(f"  desde el origen: forward={g.x_forward_m:.2f} left={g.y_left_m:.2f}")
    assert abs(g.x_forward_m - 2.0) < 1e-6 and route.idx == 0

    g = route.current_goal(Pose(1.9, 0.0, 0.0))
    print(f"  a 10cm del wp0 (dentro del radio): avanza a wp1? idx={route.idx}")
    assert route.idx == 1, "deberia haber avanzado al siguiente waypoint"

    print("\n=== pick_frontier_goal ===")
    from ..persistent_map import MapConfig, PersistentMap
    pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.05))
    bev = np.ones((100, 100), dtype=np.float32)
    obs = np.ones((100, 100), dtype=np.uint8)
    pmap.integrate(bev, obs, Pose(0, 0, 0), 2.0, 1.5, t=0.0)
    cfg = MissionConfig(search_mode="frontier", frontier_min_m=0.3, frontier_max_m=3.0,
                        frontier_preferred_m=1.0)
    picked = pick_frontier_goal(pmap, Pose(0, 0, 0), cfg)
    print(f"  frontera elegida (mundo): {picked}")
    assert picked is not None
    # Deberia estar hacia adelante (x mundo > 0, con pose en el origen mirando +x)
    assert picked[0] > 0

    picked_vacio = pick_frontier_goal(PersistentMap(MapConfig()), Pose(0, 0, 0), cfg)
    print(f"  mapa vacio -> {picked_vacio}")
    assert picked_vacio is None

    print("\n=== ConeMissionFSM: acercamiento completo ===")
    cfg = MissionConfig(detect_confidence_min=0.4, approach_start_m=2.0, verify_at_m=0.8,
                        stop_distance_m=0.5, verify_hits_required=3, verify_window=5,
                        verify_min_confidence=0.5, verify_max_jump_m=0.5,
                        lost_cone_grace_s=1.0, search_mode="wander")
    fsm = ConeMissionFSM(cfg)
    pose = Pose(0, 0, 0)
    t = 0.0

    goal = fsm.update(pose, None, None, None, t)
    print(f"  sin cono: state={goal.state} meta=({goal.x_right_m:+.2f},{goal.y_forward_m:+.2f})")
    assert goal.state == "SEARCH" and abs(goal.y_forward_m - cfg.wander_forward_m) < 1e-6

    distancias = [3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.75, 0.7, 0.65, 0.6]
    estados = []
    for d in distancias:
        det = ConeDetection(bbox_xyxy=(100, 100, 150, 200), confidence=0.8)
        ground = GroundPoint(x_forward_m=d, y_left_m=0.0, distance_m=d)
        t += 0.3
        goal = fsm.update(pose, None, det, ground, t)
        estados.append(goal.state)
    print(f"  progresion de estados: {estados}")
    assert "NAVIGATE" in estados
    assert "APPROACH" in estados
    assert "VERIFY" in estados
    assert goal.state in ("VERIFY", "STOP"), "todavia no confirmo con suficientes hits"

    # Sigue acercandose una vez estable -> deberia parar cerca de stop_distance_m
    for d in [0.55, 0.5, 0.5]:
        det = ConeDetection(bbox_xyxy=(100, 100, 150, 200), confidence=0.8)
        ground = GroundPoint(x_forward_m=d, y_left_m=0.0, distance_m=d)
        t += 0.3
        goal = fsm.update(pose, None, det, ground, t)
    print(f"  estado final: {goal.state} mision_cumplida={goal.mission_done}")
    assert goal.state == "STOP" and goal.mission_done
    assert goal.x_right_m == 0.0 and goal.y_forward_m == 0.0

    # STOP es terminal: aunque vuelva a "ver" el cono lejos, se queda frenado.
    det = ConeDetection(bbox_xyxy=(100, 100, 150, 200), confidence=0.9)
    ground = GroundPoint(x_forward_m=3.0, y_left_m=0.0, distance_m=3.0)
    goal2 = fsm.update(pose, None, det, ground, t + 1.0)
    assert goal2.state == "STOP" and goal2.mission_done

    print("\n=== cono perdido momentaneamente (grace period) ===")
    cfg2 = MissionConfig(lost_cone_grace_s=1.0, approach_start_m=2.0, verify_at_m=0.8,
                         detect_confidence_min=0.4)
    fsm2 = ConeMissionFSM(cfg2)
    det = ConeDetection(bbox_xyxy=(0, 0, 10, 10), confidence=0.7)
    ground = GroundPoint(2.5, 0.0, 2.5)
    g1 = fsm2.update(pose, None, det, ground, 0.0)
    assert g1.state == "NAVIGATE"
    # se pierde el frame siguiente, pero dentro del periodo de gracia
    g2 = fsm2.update(pose, None, None, None, 0.3)
    print(f"  perdido dentro de la gracia: state={g2.state} (deberia seguir apuntando al cono)")
    assert g2.state == "NAVIGATE" and abs(g2.y_forward_m - 2.5) < 1e-6
    # se pierde por mucho mas que la gracia -> vuelve a SEARCH
    g3 = fsm2.update(pose, None, None, None, 5.0)
    print(f"  perdido mas alla de la gracia: state={g3.state}")
    assert g3.state == "SEARCH"

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
