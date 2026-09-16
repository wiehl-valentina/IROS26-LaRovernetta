"""Maquina de estados de la mision indoor: SEARCH -> NAVIGATE -> APPROACH ->
VERIFY -> PHOTO -> (SEARCH de nuevo, o STOP) — el diagrama de
Documentos/Modulos/Indoor/Navegacion_sin_gps.html, extendido a VARIOS conos:
cada cono es un checkpoint, no el final de la mision. Al confirmar y
fotografiar uno, la maquina vuelve a SEARCH a buscar el proximo en vez de
terminar ahi; STOP solo llega cuando se agota la ruta conocida (modo
"waypoints") o la exploracion por frontera (modo "frontier") SIN un cono
anchorado en ese momento — ver `_route_exhausted`.

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
              un solo frame). Existe un metodo `confirm_with_vlm()` opcional
              (mismo cliente Gemini que usa vlm_recovery en el resto del
              repo) para sumar una confirmacion visual, pero HOY esta
              desconectado del bridge a proposito: indoor_bridge.py NO lo
              llama, y `verify_with_vlm` viene en False por defecto. Queda
              el metodo listo en la clase para engancharlo el dia que se
              decida usarlo (ver mas abajo, "confirm_with_vlm"), pero no es
              parte de la solucion actual.
    PHOTO     el cono ya esta verificado y a stop_distance_m o menos: pide
              (via MissionGoal.request_photo=True) que quien llama guarde
              una foto del frame actual. Frena y espera — no vuelve a
              moverse — hasta que alguien confirma la foto con
              `mark_photo_taken()`. Ese cono queda registrado como
              "visitado" (posicion en el mapa) para no volver a engancharse
              con el (ver `revisit_radius_m`), y la maquina vuelve a SEARCH
              a buscar el proximo — SALVO que la ruta/exploracion ya se
              agoto, en cuyo caso pasa directo a STOP (ver `_route_exhausted`).
    STOP      terminal (mission_done=True para siempre): se llega aca desde
              PHOTO (o directo desde VERIFY si take_photo=False) cuando,
              ademas, ya no queda ruta fija por recorrer (modo "waypoints")
              o exploracion por frontera pendiente (modo "frontier"). En
              modo "wander" esta condicion nunca se cumple sola — ese modo
              no tiene fin propio, la corrida se corta por --max-seconds en
              indoor_bridge.py.

IMPORTANTE (misma logica de seguridad que bridge.py): esta maquina de
estados solo decide la META (x_right_m, y_forward_m) que se le pasa a
plan_on_bev. El camino en si sigue evitando obstaculos porque plan_on_bev y
front_is_blocked no cambian — a diferencia del pseudocodigo
"steering = Kp*error" del doc original (que dirige directo al cono sin mirar
el piso), aca la aproximacion al cono sigue pasando por el mismo planificador
de transitabilidad que el resto del recorrido. Es una desviacion deliberada
del doc, mas segura para un pasillo real con gente/mobiliario.

Autoprueba (no necesita robot, camara ni modelo):
    python -m genie_rover.Indoor.mission
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
from ..odometry import Pose, wrap_rad

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
    # Confirmacion opcional con VLM (Gemini). Metodo disponible en la clase
    # (confirm_with_vlm) pero NO conectado al bridge hoy: indoor_bridge.py
    # nunca lo invoca, e independientemente de este flag la mision solo lo
    # exigiria si alguien lo llama a mano. Se deja listo para el dia que se
    # decida incorporarlo, sin que forme parte de la solucion actual.
    verify_with_vlm: bool = False
    vlm_timeout_s: float = 4.0
    vlm_min_confidence: float = 0.5

    # --- foto del cono (checkpoint) ------------------------------------------
    take_photo: bool = True         # pedir foto al completar cada checkpoint
    photo_dir: str = "checkpoint_photos"

    # --- varios conos (tour de checkpoints) -----------------------------------
    # Radio (m), en el mapa/odometria, para considerar que una nueva deteccion
    # es el MISMO cono que uno ya visitado (y por lo tanto ignorarla y seguir
    # buscando otro) en vez de volver a enganchar NAVIGATE/APPROACH/VERIFY con
    # el que ya se fotografio.
    revisit_radius_m: float = 1.0
    # Modo "frontier": cuanto tiempo sin encontrar una frontera nueva antes de
    # dar la exploracion por agotada (y terminar la mision si en ese momento
    # no hay ningun cono anchorado). Evita cortar la mision por un solo frame
    # sin frontera candidata.
    frontier_exhausted_grace_s: float = 6.0

    @classmethod
    def from_dict(cls, d: dict) -> "MissionConfig":
        """Ignora las claves que no son campos de MissionConfig.

        Antes era `cls(**(d or {}))`, que explotaba con TypeError ante
        cualquier clave extra en la seccion `mission:` del yaml --
        incluida `console_color`, que indoor_bridge.py ya leia del dict
        crudo justamente para no tocar esta clase. Ahora ademas el mismo
        `mission:` puede traer las claves del modo semantico (`mode`,
        `segments`, `route_hint`, ...) sin romper el modo de conos.
        """
        d = d or {}
        conocidas = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**conocidas)


class MissionState(str, Enum):
    SEARCH = "SEARCH"
    NAVIGATE = "NAVIGATE"
    APPROACH = "APPROACH"
    VERIFY = "VERIFY"
    PHOTO = "PHOTO"
    STOP = "STOP"


@dataclass
class MissionGoal:
    x_right_m: float
    y_forward_m: float
    state: str
    reason: str
    mission_done: bool = False
    linear_scale: float = 1.0
    request_photo: bool = False


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
        self._photo_taken = False

        # --- tour de checkpoints -------------------------------------------
        self._visited_cones: list[tuple[float, float]] = []  # (x, y) mundo
        self._checkpoints_done = 0
        self._no_frontier_since: float | None = None

    @property
    def checkpoints_done(self) -> int:
        """Cuantos conos se dieron por completados (con o sin foto) hasta ahora."""
        return self._checkpoints_done

    # ------------------------------------------------------------- publico

    def update(self, pose: Pose, pmap: "PersistentMap | None",
              cone: ConeDetection | None, cone_ground: GroundPoint | None,
              now: float) -> MissionGoal:
        if self.state == MissionState.STOP:
            return MissionGoal(0.0, 0.0, self.state.value,
                               "mision cumplida: cono alcanzado", mission_done=True)

        if self.state == MissionState.PHOTO:
            # Sticky: no se mueve mas mientras espera que guarden la foto.
            # No depende de seguir viendo el cono (ya esta confirmado) para
            # no arrancar a moverse de nuevo por perderlo de vista un frame.
            if self._photo_taken:
                self._photo_taken = False
                return self._complete_checkpoint(pose, pmap, now)
            return MissionGoal(0.0, 0.0, self.state.value,
                               "cono confirmado: esperando que se guarde la foto antes de seguir",
                               request_photo=True)

        cone_visible = (cone is not None and cone_ground is not None
                        and cone.confidence >= self.cfg.detect_confidence_min)
        if cone_visible and self._visited_cones:
            wx, wy = self._cone_world_xy(pose, cone_ground)
            if any(math.hypot(wx - vx, wy - vy) <= self.cfg.revisit_radius_m
                  for vx, vy in self._visited_cones):
                # Mismo cono que ya fotografiamos (o completamos): ignorarlo
                # y seguir tratando este frame como "sin cono nuevo", asi la
                # busqueda no vuelve a enganchar con el de vuelta.
                cone_visible = False

        if cone_visible:
            self._last_cone_ground = cone_ground
            self._last_cone_t = now
            self._hits.append((now, cone.confidence, cone_ground.distance_m))

        grace_ok = (self._last_cone_t is not None
                   and (now - self._last_cone_t) <= self.cfg.lost_cone_grace_s)
        anchored = cone_visible or (grace_ok and self._last_cone_ground is not None)

        if not anchored:
            if self._route_exhausted(pose, pmap, now):
                self.state = MissionState.STOP
                return MissionGoal(0.0, 0.0, self.state.value,
                                   f"ruta/exploracion agotada ({self._checkpoints_done} "
                                   "checkpoint(s) completado(s)), freno",
                                   mission_done=True)
            if self.state != MissionState.SEARCH:
                self.state = MissionState.SEARCH
                self._hits.clear()
            goal = self._search_goal(pose, pmap, now)
            x_right, y_fwd = goal.to_bev_goal()
            return MissionGoal(x_right, y_fwd, self.state.value,
                               f"explorando (modo={self.cfg.search_mode}, "
                               f"{self._checkpoints_done} checkpoint(s) completado(s))")

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
                if self.cfg.take_photo:
                    self.state = MissionState.PHOTO
                    return MissionGoal(0.0, 0.0, self.state.value,
                                       f"cono verificado a {dist:.2f} m, tomando foto antes de seguir",
                                       mission_done=False, request_photo=True)
                return self._complete_checkpoint(pose, pmap, now)
            razon = (f"verificando ({len(self._hits)} hits, estable={stable}, "
                    f"dist={dist:.2f} m)")
            return MissionGoal(x_right, y_fwd, self.state.value, razon,
                               linear_scale=self.cfg.approach_max_linear_scale)

        scale = self.cfg.approach_max_linear_scale if self.state == MissionState.APPROACH else 1.0
        return MissionGoal(x_right, y_fwd, self.state.value,
                           f"{self.state.value.lower()} hacia el cono (dist={dist:.2f} m)",
                           linear_scale=scale)

    def mark_photo_taken(self) -> None:
        """Llamar despues de guardar la foto que pidio un MissionGoal con
        request_photo=True (estado PHOTO). El proximo update() completa ese
        checkpoint (ver `_complete_checkpoint`) y sigue con el proximo cono,
        salvo que la ruta/exploracion ya se haya agotado. Idempotente:
        llamarlo de mas no rompe nada, solo hace falta la primera vez."""
        self._photo_taken = True

    def confirm_with_vlm(self, rgb_crop_bytes: bytes) -> bool:
        """Confirmacion opcional con Gemini (programs.client.genai_client, el
        mismo cliente que vlm_recovery usa en el resto del repo). Nunca
        lanza: si falla, la maquina de estados sigue esperando mas evidencia
        geometrica en vez de confiar ciegamente en una llamada que no llego.

        IMPORTANTE: este metodo existe pero indoor_bridge.py NO lo llama.
        Hoy el VLM esta deliberadamente desconectado del bridge indoor (no
        forma parte de la solucion actual); esto queda como el punto de
        enganche para el dia que se decida sumarlo, sin tener que tocar la
        maquina de estados de nuevo.
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

    @staticmethod
    def _cone_world_xy(pose: Pose, ground: GroundPoint) -> tuple[float, float]:
        """Posicion (x, y) del cono en el marco MUNDO (mismo marco que
        PersistentMap/WaypointRoute), a partir de la pose del robot y la
        posicion del cono relativa a el (forward/left). Inversa de la
        rotacion que usa pick_frontier_goal para ir de mundo a robot."""
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        wx = pose.x + c * ground.x_forward_m - s * ground.y_left_m
        wy = pose.y + s * ground.x_forward_m + c * ground.y_left_m
        return wx, wy

    def _complete_checkpoint(self, pose: Pose, pmap: "PersistentMap | None",
                             now: float) -> MissionGoal:
        """Da un cono por completado (con o sin foto, segun take_photo):
        registra su posicion en el mundo para no re-engancharlo
        (revisit_radius_m), limpia el estado de verificacion, y decide si
        segir buscando el proximo (SEARCH) o terminar la mision (STOP,
        cuando ya no queda ruta/exploracion pendiente)."""
        if self._last_cone_ground is not None:
            self._visited_cones.append(self._cone_world_xy(pose, self._last_cone_ground))
        self._checkpoints_done += 1
        self._hits.clear()
        self._last_cone_ground = None
        self._last_cone_t = None
        self._vlm_confirmed = False

        if self._route_exhausted(pose, pmap, now):
            self.state = MissionState.STOP
            return MissionGoal(0.0, 0.0, self.state.value,
                               f"{self._checkpoints_done} checkpoint(s) completado(s), "
                               "ruta/exploracion agotada, freno",
                               mission_done=True)

        self.state = MissionState.SEARCH
        goal = self._search_goal(pose, pmap, now)
        x_right, y_fwd = goal.to_bev_goal()
        return MissionGoal(x_right, y_fwd, self.state.value,
                           f"checkpoint {self._checkpoints_done} completado, "
                           "sigo buscando el proximo cono",
                           mission_done=False)

    def _route_exhausted(self, pose: Pose, pmap: "PersistentMap | None", now: float) -> bool:
        """True si ya no queda ruta/exploracion pendiente:

        "waypoints"  la ruta fija (WaypointRoute) ya se recorrio entera.
        "frontier"   hace frontier_exhausted_grace_s que pick_frontier_goal
                     no encuentra una frontera nueva para explorar.
        "wander"     nunca (no tiene nocion de "recorrido completo"): en ese
                     modo la mision no termina sola por agotamiento de ruta,
                     la corta --max-seconds en indoor_bridge.py.
        """
        if self.cfg.search_mode == "waypoints":
            return self._route is not None and self._route.done

        if self.cfg.search_mode == "frontier" and pmap is not None:
            picked = pick_frontier_goal(pmap, pose, self.cfg)
            if picked is not None:
                self._no_frontier_since = None
                return False
            if self._no_frontier_since is None:
                self._no_frontier_since = now
                return False
            return (now - self._no_frontier_since) >= self.cfg.frontier_exhausted_grace_s

        return False

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


# =============================================================================
#  MODO SEMANTICO: tramos rectos + hitos visuales
# =============================================================================
#
# Convive con ConeMissionFSM, no la reemplaza: mission.mode elige cual usa
# indoor_bridge.py ("cone_tour" = la de siempre, "semantic_corridor" = esta).
# map_session.py y test_indoor_mission_offline.py siguen usando la vieja sin
# enterarse de que existe esta.
#
# Diferencia de fondo con ConeMissionFSM: alla la meta era un PUNTO (waypoint
# o cono) y llegar al punto era el evento. Aca la meta es un CARRIL (una
# semirrecta con rumbo fijo) que se desliza con el robot y no se agota nunca;
# lo que hace avanzar la mision es que el VLM confirme un hito visual. El cono
# deja de ser meta: solo dispara una foto, sin frenar ni desviar (ver
# ConePhotoLog, que lo maneja del lado del bridge).
#
# Igual que ConeMissionFSM, esta maquina SOLO decide la meta (x_right_m,
# y_forward_m) que se le pasa a plan_on_bev. El camino y la evitacion de
# obstaculos los sigue resolviendo SAM-TP + plan_on_bev + front_is_blocked,
# sin cambios.


# ------------------------------------------------- lo que aporta el VLM

@dataclass
class VlmQuery:
    """Que tiene que preguntarle el bridge al VLM en este frame.

    La FSM no llama al VLM (no conoce el cliente, no bloquea el lazo de
    control): expone que pregunta esta activa y consume la respuesta que le
    traigan, si llego. Si no llego ninguna, el tramo sigue recto -- que es
    el comportamiento seguro por defecto.
    """
    id: str
    prompt: str


@dataclass
class VlmObservation:
    """Una respuesta del VLM, ya parseada.

    `id` tiene que coincidir con el id de la VlmQuery activa: una respuesta
    vieja (del hito anterior, que llego tarde por latencia) se descarta sola
    en vez de disparar un cambio de fase equivocado.
    """
    id: str
    present: bool
    confidence: float = 0.0
    position: str = "centro"        # "izquierda" | "centro" | "derecha"
    distance_m: float | None = None
    reason: str = ""
    t: float = 0.0                  # timestamp de la respuesta


# ------------------------------------------- lo que aporta la geometria (BEV)

@dataclass
class CorridorHint:
    """Donde esta el centro del espacio libre delante del robot.

    lateral_offset_m: positivo = el centro libre esta a la IZQUIERDA (misma
                      convencion que GroundPoint.y_left_m).
    clearance_m:      cuanto se puede avanzar recto antes de topar con algo
                      no transitable, mirando la franja central.
    valid:            False si el BEV no observo lo suficiente como para que
                      estos numeros signifiquen algo.
    """
    lateral_offset_m: float = 0.0
    clearance_m: float = 0.0
    valid: bool = False


def corridor_hint_from_bev(bev_traversability: np.ndarray,
                           observed_mask: np.ndarray | None,
                           resolution_m: float,
                           lookahead_m: float,
                           band_m: float = 0.6,
                           free_thresh: float = 0.5,
                           center_band_m: float = 0.35,
                           min_free_cells: int = 6,
                           row0_is_far: bool = True) -> CorridorHint:
    """Centro del pasillo y despeje frontal, leidos del mismo BEV que ya usa
    plan_on_bev -- no agrega ninguna percepcion nueva.

    Convencion de ejes asumida (la misma que usa indoor_bridge.py al llamar
    plan_on_bev): las COLUMNAS son x_right, centradas en la columna del medio;
    las FILAS son distancia hacia adelante. `row0_is_far=True` significa que
    la fila 0 es la mas LEJANA (es como lo arma project_score_to_bev y como lo
    lee la barra de console_report.py). Si al mirar un volcado de --debug-dir
    resulta al reves, alcanza con pasar row0_is_far=False: no hay ningun otro
    lugar del modo semantico que dependa de esta convencion.
    """
    bev = np.asarray(bev_traversability, dtype=np.float32)
    if bev.ndim != 2 or bev.size == 0:
        return CorridorHint()
    filas, cols = bev.shape

    obs = (np.ones_like(bev, dtype=bool) if observed_mask is None
           else np.asarray(observed_mask).astype(bool))
    libre = (bev >= free_thresh) & obs

    # Distancia hacia adelante de cada fila.
    idx = np.arange(filas, dtype=np.float32)
    dist_fila = (filas - 1 - idx) * resolution_m if row0_is_far else idx * resolution_m

    # --- offset lateral: centroide de lo libre en la franja del lookahead ---
    franja = np.abs(dist_fila - lookahead_m) <= band_m
    if not np.any(franja):
        # lookahead mas lejos de lo que el BEV alcanza: usar la franja mas
        # lejana disponible en vez de devolver nada.
        franja = dist_fila >= (float(dist_fila.max()) - band_m)

    cols_libres = libre[franja].sum(axis=0)
    total = float(cols_libres.sum())
    if total < min_free_cells:
        return CorridorHint()

    centro_col = (cols - 1) / 2.0
    peso = cols_libres.astype(np.float32)
    centroide = float((peso * np.arange(cols, dtype=np.float32)).sum() / total)
    x_right = (centroide - centro_col) * resolution_m
    lateral_offset_m = -x_right          # a y_left (positivo = izquierda)

    # --- despeje frontal: hasta donde llega lo libre en la franja central ---
    media_col = max(1, int(round(center_band_m / resolution_m)))
    c0 = max(0, int(round(centro_col)) - media_col)
    c1 = min(cols, int(round(centro_col)) + media_col + 1)
    central = libre[:, c0:c1]
    fila_libre = central.mean(axis=1) >= 0.5

    orden = np.argsort(dist_fila)          # de cerca a lejos
    clearance = 0.0
    for r in orden:
        if not fila_libre[r]:
            break
        clearance = float(dist_fila[r])

    return CorridorHint(lateral_offset_m=lateral_offset_m,
                        clearance_m=clearance, valid=True)


# ------------------------------------------------------- el circuito, en datos

@dataclass
class MilestoneSpec:
    """El hito visual que termina un tramo."""
    id: str
    prompt: str
    confirm_hits: int = 3           # confirmaciones SEGUIDAS para creerle
    min_confidence: float = 0.6
    max_distance_m: float | None = None   # si el VLM estima distancia, exigir
                                          # que el hito este a menos de esto

    @classmethod
    def from_dict(cls, d: dict) -> "MilestoneSpec":
        return cls(**(d or {}))


@dataclass
class TurnSpec:
    """El giro que se hace DESPUES de confirmar el hito del tramo."""
    side: str                        # "left" | "right"
    look_for: str                    # prompt de la apertura/puerta/cono
    id: str | None = None            # se completa solo con el id del tramo
    confirm_hits: int = 2
    min_confidence: float = 0.55

    @classmethod
    def from_dict(cls, d: dict) -> "TurnSpec":
        t = cls(**(d or {}))
        if t.side not in ("left", "right"):
            raise ValueError(f"turn.side tiene que ser 'left' o 'right', no {t.side!r}")
        return t

    @property
    def sign(self) -> float:
        """+1 gira a la izquierda (theta crece, y_left positivo)."""
        return 1.0 if self.side == "left" else -1.0


@dataclass
class SegmentSpec:
    """Un tramo recto + el hito que lo cierra + que hacer despues."""
    id: str
    milestone: MilestoneSpec
    turn: TurnSpec | None = None
    max_distance_m: float = 30.0
    max_seconds: float | None = None
    on_timeout: str = "sweep"        # "sweep" | "advance" | "stop"
    final: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentSpec":
        d = dict(d or {})
        seg_id = str(d.pop("id"))
        milestone = MilestoneSpec.from_dict(d.pop("milestone"))
        turn_d = d.pop("on_milestone", None) or d.pop("turn", None)
        turn = None
        if turn_d:
            turn_d = dict(turn_d)
            turn_d.setdefault("id", f"{seg_id}__apertura")
            turn = TurnSpec.from_dict(turn_d)
        seg = cls(id=seg_id, milestone=milestone, turn=turn, **d)
        if seg.on_timeout not in ("sweep", "advance", "stop"):
            raise ValueError(f"{seg_id}: on_timeout invalido ({seg.on_timeout!r})")
        return seg


@dataclass
class SemanticMissionConfig:
    # --- carril virtual ------------------------------------------------------
    segment_lookahead_m: float = 2.5
    corridor_centering_weight: float = 0.6   # 0 = rumbo puro, 1 = centrado puro
    corridor_centering_max_m: float = 0.8    # tope de correccion lateral

    # --- giros ---------------------------------------------------------------
    turn_probe_m: float = 1.2        # cuan al costado se pone la meta al pivotear
    turn_max_deg: float = 150.0      # tope de giro buscando la apertura
    align_min_clearance_m: float = 1.5
    align_tolerance_deg: float = 8.0
    align_linear_scale: float = 0.25
    align_max_s: float = 12.0

    # --- barrido de rescate (on_timeout: "sweep") ----------------------------
    sweep_max_deg: float = 60.0

    # Convencion de filas del BEV que se le pasa a corridor_hint_from_bev
    # (True = la fila 0 es la mas lejana). Ver el docstring de esa funcion.
    bev_row0_is_far: bool = True

    # --- VLM (cadencia y anti-rebote) ---------------------------------------
    milestone_cooldown_s: float = 3.0   # tras cambiar de fase, ignorar hitos
    vlm_stale_s: float = 2.5            # respuesta mas vieja que esto = no hay

    # --- guia opcional del mapeo previo -------------------------------------
    route_hint_path: str | None = None
    route_hint_enabled: bool = False
    route_hint_max_deg: float = 25.0

    segments: list[SegmentSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticMissionConfig":
        d = dict(d or {})
        d.pop("mode", None)
        segs = [SegmentSpec.from_dict(s) for s in d.pop("segments", [])]
        hint = dict(d.pop("route_hint", {}) or {})
        cfg = cls(
            segments=segs,
            route_hint_path=hint.get("path"),
            route_hint_enabled=bool(hint.get("enabled", False)),
            route_hint_max_deg=float(hint.get("heading_bias_max_deg", 25.0)),
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )
        if not cfg.segments:
            raise ValueError("mission.segments vacio: el modo semantico necesita "
                             "al menos un tramo (ver indoor_semantic_tour.yaml)")
        return cfg


# --------------------------------------------------------------- carril virtual

class HeadingLock:
    """Semirrecta infinita desde donde se bloqueo el rumbo.

    NO es un punto fijo: la meta se calcula proyectando la pose actual sobre
    la semirrecta y adelantando `lookahead` metros sobre ella. Eso da dos
    cosas que un punto fijo no da: la meta nunca se agota (el tramo dura lo
    que tarde el hito en aparecer, no lo que dure la distancia), y el error
    lateral que deja un esquive se cancela solo al volver al carril, en vez
    de mandar al robot en diagonal contra la pared de enfrente.
    """

    def __init__(self, pose: Pose, theta: float | None = None):
        self.x0 = float(pose.x)
        self.y0 = float(pose.y)
        self.theta = float(pose.theta if theta is None else theta)

    def progress_m(self, pose: Pose) -> float:
        """Cuanto avanzo el robot SOBRE el carril (no en linea recta)."""
        dx, dy = pose.x - self.x0, pose.y - self.y0
        return dx * math.cos(self.theta) + dy * math.sin(self.theta)

    def lateral_error_m(self, pose: Pose) -> float:
        """Desvio perpendicular al carril; positivo = el robot esta a la
        izquierda del carril."""
        dx, dy = pose.x - self.x0, pose.y - self.y0
        return -dx * math.sin(self.theta) + dy * math.cos(self.theta)

    def goal_world(self, pose: Pose, lookahead_m: float) -> tuple[float, float]:
        s = max(0.0, self.progress_m(pose)) + float(lookahead_m)
        return (self.x0 + s * math.cos(self.theta),
                self.y0 + s * math.sin(self.theta))


class RouteHint:
    """La ruta del mapeo previo, degradada a SUGERENCIA.

    Nunca es meta del planner y su agotamiento nunca termina la mision: lo
    unico que puede hacer es corregir el rumbo en el momento de bloquearlo, y
    solo si la correccion es chica (heading_bias_max_deg). Si sugiere algo muy
    distinto se la ignora, porque en ese caso el mapeo previo esta viejo o el
    robot no esta donde la ruta cree.
    """

    def __init__(self, points: list[tuple[float, float]], max_bias_deg: float = 25.0):
        self.points = list(points)
        self.max_bias_deg = float(max_bias_deg)

    @classmethod
    def from_file(cls, path: str, max_bias_deg: float = 25.0) -> "RouteHint":
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        pts = [(float(p["x_m"]), float(p["y_m"])) for p in data.get("waypoints", [])]
        return cls(pts, max_bias_deg=max_bias_deg)

    def bias_heading(self, pose: Pose, theta: float) -> tuple[float, str]:
        """Devuelve (theta_corregido, motivo)."""
        if not self.points:
            return theta, "sin ruta de guia"
        # El punto util es el mas cercano que este ADELANTE del rumbo actual.
        mejor, mejor_d = None, float("inf")
        for (px, py) in self.points:
            rel = Pose(px, py, 0.0).relative_to(Pose(pose.x, pose.y, theta))
            if rel.x <= 0.3:
                continue
            d = math.hypot(rel.x, rel.y)
            if d < mejor_d:
                mejor, mejor_d = (px, py), d
        if mejor is None:
            return theta, "la ruta de guia no tiene puntos adelante"
        sugerido = math.atan2(mejor[1] - pose.y, mejor[0] - pose.x)
        delta = wrap_rad(sugerido - theta)
        if abs(math.degrees(delta)) > self.max_bias_deg:
            return theta, (f"guia descartada ({math.degrees(delta):+.0f} grados, "
                           f"mas de {self.max_bias_deg:.0f})")
        return wrap_rad(theta + delta), f"guia aplicada ({math.degrees(delta):+.0f} grados)"


# ------------------------------------------------------- foto del cono (evento)

class ConePhotoLog:
    """El cono ya no es meta ni checkpoint: es un disparador de foto.

    Esto vive aca y no en la FSM a proposito -- justamente para que sacar la
    foto no pueda alterar la trayectoria: indoor_bridge.py lo consulta aparte
    del mission_goal y, si da True, guarda el frame y sigue derecho en el
    mismo ciclo.
    """

    def __init__(self, revisit_radius_m: float = 1.0):
        self.revisit_radius_m = float(revisit_radius_m)
        self.seen: list[tuple[float, float]] = []

    def should_photograph(self, x_world: float, y_world: float) -> bool:
        for (px, py) in self.seen:
            if math.hypot(x_world - px, y_world - py) <= self.revisit_radius_m:
                return False
        return True

    def record(self, x_world: float, y_world: float) -> int:
        self.seen.append((float(x_world), float(y_world)))
        return len(self.seen)

    @property
    def photos(self) -> int:
        return len(self.seen)


# ----------------------------------------------------------- maquina de estados

class SemanticState(str, Enum):
    RUN_SEGMENT = "RUN_SEGMENT"
    TURN_SEARCH = "TURN_SEARCH"
    TURN_ALIGN = "TURN_ALIGN"
    SWEEP = "SWEEP"
    DONE = "DONE"


class SemanticMissionFSM:
    """Recorrido por tramos rectos con cambios de fase semanticos.

    Uso desde indoor_bridge.py:

        fsm = SemanticMissionFSM(sem_cfg)
        ...
        q = fsm.current_query()          # que preguntarle al VLM ahora
        obs = vlm.latest_for(q.id)       # respuesta cacheada, puede ser None
        hint = corridor_hint_from_bev(plan_bev, plan_obs, res, lookahead)
        goal = fsm.update(pose, now, vlm=obs, corridor=hint)
    """

    def __init__(self, cfg: SemanticMissionConfig):
        self.cfg = cfg
        self.state = SemanticState.RUN_SEGMENT
        self.segment_idx = 0
        self.segments_done = 0
        self.failed_reason: str | None = None

        self._lock: HeadingLock | None = None
        self._route_hint: RouteHint | None = None
        if cfg.route_hint_enabled and cfg.route_hint_path:
            self._route_hint = RouteHint.from_file(cfg.route_hint_path,
                                                   cfg.route_hint_max_deg)

        self._hits = 0
        self._last_obs_t: float = -1.0
        self._phase_t: float = 0.0          # cuando empezo la fase actual
        self._turn_theta0: float = 0.0
        self._sweep_leg = 0                  # 0 = a un lado, 1 = al otro
        self._started = False

    # ------------------------------------------------------------- publico

    @property
    def segment(self) -> SegmentSpec:
        idx = min(self.segment_idx, len(self.cfg.segments) - 1)
        return self.cfg.segments[idx]

    def current_query(self) -> VlmQuery | None:
        """Que hito hay que estar buscando en este momento. None = ninguno
        (la mision termino, o esta en alineacion fina, que es geometrica)."""
        if self.state == SemanticState.DONE:
            return None
        seg = self.segment
        if self.state in (SemanticState.RUN_SEGMENT, SemanticState.SWEEP):
            return VlmQuery(seg.milestone.id, seg.milestone.prompt)
        if self.state == SemanticState.TURN_SEARCH and seg.turn is not None:
            return VlmQuery(seg.turn.id or f"{seg.id}__apertura", seg.turn.look_for)
        return None

    def update(self, pose: Pose, now: float,
               vlm: VlmObservation | None = None,
               corridor: CorridorHint | None = None) -> MissionGoal:
        if not self._started:
            self._relock(pose, now, motivo="inicio de mision")
            self._started = True

        if self.state == SemanticState.DONE:
            razon = self.failed_reason or (
                f"recorrido completo: {self.segments_done} tramo(s)")
            return MissionGoal(0.0, 0.0, self.state.value, razon, mission_done=True)

        confirmado = self._count_hits(vlm, now)

        if self.state == SemanticState.RUN_SEGMENT:
            return self._run_segment(pose, now, confirmado, corridor)
        if self.state == SemanticState.TURN_SEARCH:
            return self._turn_search(pose, now, confirmado)
        if self.state == SemanticState.TURN_ALIGN:
            return self._turn_align(pose, now, corridor)
        return self._sweep(pose, now, confirmado)

    # -------------------------------------------------------------- estados

    def _run_segment(self, pose: Pose, now: float, confirmado: bool,
                     corridor: CorridorHint | None) -> MissionGoal:
        seg = self.segment
        if confirmado:
            return self._milestone_reached(pose, now, f"hito '{seg.milestone.id}' confirmado")

        assert self._lock is not None
        avance = self._lock.progress_m(pose)
        transcurrido = now - self._phase_t
        vencido = (avance >= seg.max_distance_m
                   or (seg.max_seconds is not None and transcurrido >= seg.max_seconds))
        if vencido:
            return self._on_timeout(pose, now, avance)

        x_fwd, y_left = self._lane_goal(pose, corridor)
        gp = GroundPoint(x_fwd, y_left, math.hypot(x_fwd, y_left))
        x_right, y_forward = gp.to_bev_goal()
        return MissionGoal(x_right, y_forward, self.state.value,
                           f"{seg.id}: recto {avance:.1f}/{seg.max_distance_m:.0f} m, "
                           f"buscando '{seg.milestone.id}'")

    def _turn_search(self, pose: Pose, now: float, confirmado: bool) -> MissionGoal:
        seg = self.segment
        turn = seg.turn
        assert turn is not None
        girado = math.degrees(wrap_rad(pose.theta - self._turn_theta0)) * turn.sign

        if confirmado:
            self._enter(SemanticState.TURN_ALIGN, now)
            return MissionGoal(0.0, 0.5, self.state.value,
                               f"{seg.id}: '{turn.look_for[:28]}' a la vista, alineando",
                               linear_scale=self.cfg.align_linear_scale)

        if girado >= self.cfg.turn_max_deg:
            if seg.on_timeout == "stop":
                return self._finish(f"{seg.id}: gire {girado:.0f} grados sin encontrar "
                                    f"la apertura")
            self._enter(SemanticState.TURN_ALIGN, now)
            return MissionGoal(0.0, 0.5, self.state.value,
                               f"{seg.id}: sin confirmacion tras {girado:.0f} grados, "
                               f"alineo por geometria igual",
                               linear_scale=self.cfg.align_linear_scale)

        # Pivote: meta al costado y linear_scale 0 -> el seguidor gira en el
        # lugar en vez de avanzar. La meta se mantiene un poco adelante (no
        # exactamente a 90 grados) para que plan_on_bev no reciba una meta
        # degenerada encima del robot.
        y_left = turn.sign * self.cfg.turn_probe_m
        gp = GroundPoint(0.3, y_left, math.hypot(0.3, y_left))
        x_right, y_forward = gp.to_bev_goal()
        return MissionGoal(x_right, y_forward, self.state.value,
                           f"{seg.id}: girando a la {turn.side} ({girado:.0f} grados), "
                           f"buscando apertura", linear_scale=0.0)

    def _turn_align(self, pose: Pose, now: float,
                    corridor: CorridorHint | None) -> MissionGoal:
        """Alineacion fina POR GEOMETRIA, no por VLM.

        El VLM sabe decir "hay una puerta a tu izquierda"; no sabe decir "te
        faltan 4 grados". El centro del hueco lo sabe el BEV, asi que la
        ultima parte del giro la cierra el BEV.
        """
        seg = self.segment
        if corridor is None or not corridor.valid:
            if (now - self._phase_t) >= self.cfg.align_max_s:
                return self._after_turn(pose, now,
                                        "alineacion sin BEV utilizable, bloqueo el rumbo actual")
            return MissionGoal(0.0, 0.5, self.state.value,
                               f"{seg.id}: esperando BEV para alinear",
                               linear_scale=0.0)

        rumbo = math.atan2(corridor.lateral_offset_m, self.cfg.segment_lookahead_m)
        alineado = abs(math.degrees(rumbo)) <= self.cfg.align_tolerance_deg
        despejado = corridor.clearance_m >= self.cfg.align_min_clearance_m
        if alineado and despejado:
            return self._after_turn(pose, now,
                                    f"alineado (desvio {math.degrees(rumbo):+.0f} grados, "
                                    f"despeje {corridor.clearance_m:.1f} m)",
                                    theta=wrap_rad(pose.theta + rumbo))

        if (now - self._phase_t) >= self.cfg.align_max_s:
            return self._after_turn(pose, now,
                                    f"alineacion agotada a {self.cfg.align_max_s:.0f} s",
                                    theta=wrap_rad(pose.theta + rumbo))

        gp = GroundPoint(self.cfg.segment_lookahead_m, corridor.lateral_offset_m,
                         self.cfg.segment_lookahead_m)
        x_right, y_forward = gp.to_bev_goal()
        return MissionGoal(x_right, y_forward, self.state.value,
                           f"{seg.id}: alineando (desvio {math.degrees(rumbo):+.0f} grados, "
                           f"despeje {corridor.clearance_m:.1f} m)",
                           linear_scale=self.cfg.align_linear_scale)

    def _sweep(self, pose: Pose, now: float, confirmado: bool) -> MissionGoal:
        """Rescate: el hito no aparecio en max_distance_m. Frena y barre a los
        dos lados preguntando lo mismo, antes de dar el tramo por perdido."""
        seg = self.segment
        if confirmado:
            return self._milestone_reached(pose, now,
                                           f"hito '{seg.milestone.id}' encontrado en el barrido")

        signo = 1.0 if self._sweep_leg == 0 else -1.0
        girado = math.degrees(wrap_rad(pose.theta - self._turn_theta0)) * signo
        tope = self.cfg.sweep_max_deg * (1 if self._sweep_leg == 0 else 2)
        if girado >= tope:
            if self._sweep_leg == 0:
                self._sweep_leg = 1
            else:
                # Barrido agotado: se aplica la politica como si fuera "advance"
                # (si era "stop" no habriamos entrado aca).
                return self._milestone_reached(
                    pose, now, f"{seg.id}: hito no encontrado ni en el barrido, sigo igual")

        y_left = signo * self.cfg.turn_probe_m
        gp = GroundPoint(0.3, y_left, math.hypot(0.3, y_left))
        x_right, y_forward = gp.to_bev_goal()
        return MissionGoal(x_right, y_forward, self.state.value,
                           f"{seg.id}: barrido de rescate ({girado:.0f} grados)",
                           linear_scale=0.0)

    # ------------------------------------------------------------- privado

    def _lane_goal(self, pose: Pose, corridor: CorridorHint | None) -> tuple[float, float]:
        """Meta del tramo: punto sobre el carril, corregido hacia el centro
        del pasillo. Devuelve (x_forward_m, y_left_m) en marco del robot."""
        assert self._lock is not None
        gx, gy = self._lock.goal_world(pose, self.cfg.segment_lookahead_m)
        rel = Pose(gx, gy, 0.0).relative_to(pose)
        x_fwd, y_left = rel.x, rel.y

        if corridor is not None and corridor.valid and self.cfg.corridor_centering_weight > 0:
            w = min(1.0, max(0.0, self.cfg.corridor_centering_weight))
            mezcla = (1.0 - w) * y_left + w * corridor.lateral_offset_m
            tope = self.cfg.corridor_centering_max_m
            y_left = y_left + max(-tope, min(tope, mezcla - y_left))
        return x_fwd, y_left

    def _count_hits(self, vlm: VlmObservation | None, now: float) -> bool:
        """Histeresis: N confirmaciones SEGUIDAS, no una.

        Con esta arquitectura un falso positivo no es un error de percepcion:
        es un giro a la izquierda contra una pared. Por eso una sola respuesta
        afirmativa nunca alcanza para cambiar de fase.
        """
        q = self.current_query()
        if q is None or vlm is None:
            return False
        if vlm.id != q.id:
            return False                      # respuesta de un hito viejo
        if vlm.t <= self._last_obs_t:
            return False                      # ya contada
        if (now - vlm.t) > self.cfg.vlm_stale_s:
            return False                      # llego demasiado tarde
        self._last_obs_t = vlm.t

        if (now - self._phase_t) < self.cfg.milestone_cooldown_s:
            return False                      # el hito anterior sigue en cuadro

        seg = self.segment
        if self.state == SemanticState.TURN_SEARCH and seg.turn is not None:
            requeridos, min_conf = seg.turn.confirm_hits, seg.turn.min_confidence
            max_d = None
        else:
            requeridos = seg.milestone.confirm_hits
            min_conf = seg.milestone.min_confidence
            max_d = seg.milestone.max_distance_m

        ok = vlm.present and vlm.confidence >= min_conf
        if ok and max_d is not None and vlm.distance_m is not None:
            ok = vlm.distance_m <= max_d
        self._hits = self._hits + 1 if ok else 0
        return self._hits >= requeridos

    def _milestone_reached(self, pose: Pose, now: float, razon: str) -> MissionGoal:
        seg = self.segment
        if seg.turn is not None:
            self._enter(SemanticState.TURN_SEARCH, now)
            self._turn_theta0 = pose.theta
            return MissionGoal(0.0, 0.0, self.state.value,
                               f"{razon}; giro a la {seg.turn.side}", linear_scale=0.0)
        return self._after_turn(pose, now, razon)

    def _after_turn(self, pose: Pose, now: float, razon: str,
                    theta: float | None = None) -> MissionGoal:
        """Cierra el tramo actual: o termina la mision, o bloquea rumbo nuevo."""
        seg = self.segment
        self.segments_done += 1
        if seg.final:
            return self._finish(f"{razon}; ultimo tramo completado", ok=True)

        self.segment_idx = min(self.segment_idx + 1, len(self.cfg.segments) - 1)
        self._relock(pose, now, motivo=razon, theta=theta)
        nuevo = self.segment
        return MissionGoal(0.0, 0.5, self.state.value,
                           f"{razon}; arranca '{nuevo.id}'",
                           linear_scale=self.cfg.align_linear_scale)

    def _on_timeout(self, pose: Pose, now: float, avance: float) -> MissionGoal:
        seg = self.segment
        razon = f"{seg.id}: {avance:.1f} m sin ver '{seg.milestone.id}'"
        if seg.on_timeout == "stop":
            return self._finish(razon + " (on_timeout: stop)")
        if seg.on_timeout == "advance":
            return self._milestone_reached(pose, now, razon + " (sigo igual)")
        self._enter(SemanticState.SWEEP, now)
        self._turn_theta0 = pose.theta
        self._sweep_leg = 0
        return MissionGoal(0.0, 0.0, self.state.value, razon + " (barrido de rescate)",
                           linear_scale=0.0)

    def _finish(self, razon: str, ok: bool = False) -> MissionGoal:
        self.state = SemanticState.DONE
        if not ok:
            self.failed_reason = razon
        return MissionGoal(0.0, 0.0, self.state.value, razon, mission_done=True)

    def _enter(self, state: SemanticState, now: float) -> None:
        self.state = state
        self._phase_t = now
        self._hits = 0

    def _relock(self, pose: Pose, now: float, motivo: str,
                theta: float | None = None) -> None:
        th = pose.theta if theta is None else theta
        if self._route_hint is not None:
            th, _ = self._route_hint.bias_heading(pose, th)
        self._lock = HeadingLock(pose, th)
        self._enter(SemanticState.RUN_SEGMENT, now)

    # ------------------------------------------------------------ inspeccion

    @property
    def checkpoints_done(self) -> int:
        """Alias de segments_done, con el nombre que ya usan
        MissionConsoleReporter, RouteStatus y el resumen final de
        indoor_bridge.py -- asi esos tres siguen andando sin ramificar por
        modo. En el modo semantico un "checkpoint" es un TRAMO cerrado, no
        un cono: los conos se cuentan aparte, en ConePhotoLog."""
        return self.segments_done

    @property
    def lane(self) -> HeadingLock | None:
        """El carril activo, para que route_status.py lo publique en el
        dashboard en lugar de los waypoints."""
        return self._lock


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
    assert goal.state in ("VERIFY", "PHOTO"), "todavia no confirmo con suficientes hits"

    # Sigue acercandose una vez estable -> deberia pedir foto cerca de stop_distance_m
    for d in [0.55, 0.5, 0.5]:
        det = ConeDetection(bbox_xyxy=(100, 100, 150, 200), confidence=0.8)
        ground = GroundPoint(x_forward_m=d, y_left_m=0.0, distance_m=d)
        t += 0.3
        goal = fsm.update(pose, None, det, ground, t)
    print(f"  estado tras confirmar: {goal.state} pide_foto={goal.request_photo}")
    assert goal.state == "PHOTO" and goal.request_photo and not goal.mission_done
    assert goal.x_right_m == 0.0 and goal.y_forward_m == 0.0

    # Sigue frenado hasta que se confirma la foto.
    goal_espera = fsm.update(pose, None, det, ground, t + 0.3)
    assert goal_espera.state == "PHOTO" and not goal_espera.mission_done

    # Con la foto confirmada: en modo "wander" no hay ruta que agotar, asi
    # que el cono es un CHECKPOINT, no el final -> vuelve a SEARCH, no a STOP.
    fsm.mark_photo_taken()
    goal = fsm.update(pose, None, det, ground, t + 0.6)
    print(f"  estado tras la foto: {goal.state} checkpoints_done={fsm.checkpoints_done}")
    assert goal.state == "SEARCH" and not goal.mission_done
    assert fsm.checkpoints_done == 1

    # El MISMO cono (misma posicion en el mundo, pose sin moverse) no debe
    # volver a enganchar NAVIGATE/APPROACH/VERIFY: esta dentro de
    # revisit_radius_m del que ya se fotografio.
    goal_mismo = fsm.update(pose, None, det, ground, t + 0.9)
    print(f"  mismo cono otra vez: state={goal_mismo.state} (deberia seguir en SEARCH)")
    assert goal_mismo.state == "SEARCH"

    # Un cono NUEVO, lejos del anterior, si debe enganchar de nuevo.
    det2 = ConeDetection(bbox_xyxy=(100, 100, 150, 200), confidence=0.9)
    ground2 = GroundPoint(x_forward_m=3.0, y_left_m=2.5, distance_m=3.9)
    goal2 = fsm.update(pose, None, det2, ground2, t + 1.2)
    print(f"  cono nuevo lejos del anterior: state={goal2.state}")
    assert goal2.state == "NAVIGATE"

    print("\n=== ConeMissionFSM: take_photo=False tambien sigue de largo (checkpoint sin foto) ===")
    cfg_sin_foto = MissionConfig(detect_confidence_min=0.4, approach_start_m=2.0,
                                 verify_at_m=0.8, stop_distance_m=0.5,
                                 verify_hits_required=2, verify_window=3,
                                 verify_min_confidence=0.5, verify_max_jump_m=0.5,
                                 take_photo=False)
    fsm_sf = ConeMissionFSM(cfg_sin_foto)
    tt = 0.0
    for d in [0.7, 0.6, 0.5, 0.48]:
        det = ConeDetection(bbox_xyxy=(100, 100, 150, 200), confidence=0.8)
        ground = GroundPoint(x_forward_m=d, y_left_m=0.0, distance_m=d)
        tt += 0.3
        goal_sf = fsm_sf.update(pose, None, det, ground, tt)
    assert goal_sf.state == "SEARCH" and not goal_sf.mission_done and fsm_sf.checkpoints_done == 1

    print("\n=== ConeMissionFSM: modo waypoints, STOP solo cuando se agota la ruta ===")
    cfg_wp = MissionConfig(detect_confidence_min=0.4, approach_start_m=2.0, verify_at_m=0.8,
                           stop_distance_m=0.5, verify_hits_required=2, verify_window=3,
                           verify_min_confidence=0.5, verify_max_jump_m=0.5,
                           search_mode="waypoints", take_photo=False)
    fsm_wp = ConeMissionFSM(cfg_wp)
    fsm_wp._route = WaypointRoute([(5.0, 0.0)], reach_radius_m=0.3)  # ruta larga, no se agota sola
    tw = 0.0
    for d in [0.7, 0.6, 0.5, 0.48]:
        det = ConeDetection(bbox_xyxy=(100, 100, 150, 200), confidence=0.8)
        ground = GroundPoint(x_forward_m=d, y_left_m=0.0, distance_m=d)
        tw += 0.3
        goal_wp = fsm_wp.update(pose, None, det, ground, tw)
    print(f"  tras completar el cono con ruta pendiente: state={goal_wp.state}")
    assert goal_wp.state == "SEARCH" and not goal_wp.mission_done, \
        "con ruta todavia pendiente no deberia terminar la mision"
    # Ahora se agota la ruta (sin cono a la vista) -> recien ahi STOP.
    fsm_wp._route.idx = len(fsm_wp._route.points)  # simula que ya se recorrio toda la ruta
    goal_wp_fin = fsm_wp.update(pose, None, None, None, tw + 5.0)
    print(f"  ruta agotada, sin cono: state={goal_wp_fin.state} mision_cumplida={goal_wp_fin.mission_done}")
    assert goal_wp_fin.state == "STOP" and goal_wp_fin.mission_done

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
