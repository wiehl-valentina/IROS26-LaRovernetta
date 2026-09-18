"""Mision indoor SEMANTICA: tramos rectos con el rumbo bloqueado + hitos
visuales confirmados por un VLM.

Esta version ya NO tiene nada de deteccion de conos: se eliminaron
MissionConfig, MissionState, WaypointRoute, ConeMissionFSM y ConePhotoLog
(y con ellas el modulo cone_detector.py entero). La unica mision que existe
es la semantica.

Como funciona, en corto:

    RUN_SEGMENT  el rover avanza sobre un CARRIL virtual (HeadingLock: una
                 semirrecta con rumbo fijo desde donde arranco el tramo).
                 La meta se desliza con el robot y no se agota nunca; lo que
                 termina el tramo es que el VLM confirme el hito visual del
                 tramo (un piano, tres sillas en hilera, el fin del pasillo),
                 o el fail-safe de distancia (`max_distance_m` + `on_timeout`).
    TURN_SEARCH  pivote hacia el lado que indica el tramo, preguntandole al
                 VLM por la apertura/puerta/pasillo que hay que tomar.
    TURN_ALIGN   alineacion fina POR GEOMETRIA (BEV), no por VLM.
    SWEEP        barrido de rescate cuando se agoto el tramo sin ver el hito
                 (`on_timeout: "sweep"`).
    DONE         terminal.

Ademas queda `FrontierExploreFSM`: exploracion por frontera sobre el
PersistentMap, SIN VLM ni tramos, que es lo que usa map_session.py para las
sesiones de mapeo puro. No tiene nada que ver con conos; es la estrategia de
"segui explorando donde el mapa todavia no vio nada".

IMPORTANTE (igual que antes): estas maquinas solo deciden la META
(x_right_m, y_forward_m) que se le pasa a plan_on_bev. El camino y la
evitacion de obstaculos los siguen resolviendo SAM-TP + plan_on_bev +
front_is_blocked, sin cambios.

Autoprueba (no necesita robot, camara, VLM ni modelo):
    python -m genie_rover.Indoor.mission
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..odometry import Pose, wrap_rad

if TYPE_CHECKING:  # solo para hints, sin crear un import circular en runtime
    from ..persistent_map import PersistentMap


# ------------------------------------------------------------------- basicos

@dataclass
class GroundPoint:
    """Un punto del plano del suelo en el marco del robot (odometry.Pose).

    Vivia en cone_detector.py, que ya no existe: es geometria generica (nada
    de conos), la usan las dos FSM para expresar su meta antes de pasarla a
    la convencion de plan_on_bev.
    """
    x_forward_m: float
    y_left_m: float
    distance_m: float

    def to_bev_goal(self) -> tuple[float, float]:
        """(x_right_m, y_forward_m): la convencion que espera plan_on_bev."""
        return -self.y_left_m, self.x_forward_m


@dataclass
class MissionGoal:
    x_right_m: float
    y_forward_m: float
    state: str
    reason: str
    mission_done: bool = False
    linear_scale: float = 1.0


# ------------------------------------------------------- exploracion frontera

@dataclass
class FrontierExploreConfig:
    """Parametros de la exploracion por frontera (sesion de mapeo)."""
    free_thresh: float = 0.55
    min_m: float = 0.6
    max_m: float = 3.5
    preferred_m: float = 1.8
    fov_deg: float = 140.0
    goal_hold_s: float = 4.0
    reach_radius_m: float = 0.6
    wander_forward_m: float = 3.0

    @classmethod
    def from_dict(cls, d: dict) -> "FrontierExploreConfig":
        """Acepta los nombres cortos y tambien los `frontier_*` que ya traen
        los yaml viejos (indoor_mapping.yaml)."""
        d = d or {}
        alias = {
            "free_thresh": ("free_thresh", "frontier_free_thresh"),
            "min_m": ("min_m", "frontier_min_m"),
            "max_m": ("max_m", "frontier_max_m"),
            "preferred_m": ("preferred_m", "frontier_preferred_m"),
            "fov_deg": ("fov_deg", "frontier_fov_deg"),
            "goal_hold_s": ("goal_hold_s", "frontier_goal_hold_s"),
            "reach_radius_m": ("reach_radius_m", "waypoint_reach_radius_m"),
            "wander_forward_m": ("wander_forward_m",),
        }
        vals = {}
        for campo, claves in alias.items():
            for clave in claves:
                if clave in d:
                    vals[campo] = float(d[clave])
                    break
        return cls(**vals)


def pick_frontier_goal(pmap: "PersistentMap", pose: Pose,
                       cfg: FrontierExploreConfig) -> tuple[float, float] | None:
    """Elige un punto del MUNDO (x, y) en el borde entre lo conocido-libre y
    lo nunca-visto, para que el rover siga explorando sin repetir pasillo.

    Opera directamente sobre las grillas numpy de PersistentMap (value, conf)
    — reutiliza el mapa que bridge.py ya mantiene para memoria espacial, no
    agrega ningun estado propio. Devuelve None si no hay frontera candidata
    (mapa todavia vacio, o todo lo visible ya fue explorado), y quien llama
    debe caer a otra estrategia (derecho adelante).
    """
    conf, val = pmap.conf, pmap.value
    known = conf >= pmap.cfg.min_confidence
    free = known & (val > cfg.free_thresh)
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

    fov = math.radians(cfg.fov_deg / 2.0)
    ang = np.arctan2(left, np.maximum(forward, 1e-6))
    in_cone = (np.abs(ang) <= fov) & (forward > 0)
    in_range = (dist >= cfg.min_m) & (dist <= cfg.max_m)

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
    score = np.abs(dist[idx_validos] - cfg.preferred_m)
    mejor = idx_validos[int(np.argmin(score))]

    return float(x_world[mejor]), float(y_world[mejor])


class _SegmentoFalso:
    """Lo minimo que el bridge le pide a `mission.segment` (solo el id)."""
    id = "explorar"
    milestone = None
    turn = None


class FrontierExploreFSM:
    """Exploracion por frontera pura, para las sesiones de mapeo.

    Expone la MISMA interfaz que SemanticMissionFSM (update / current_query /
    segment / checkpoints_done / state / failed_reason) para que
    indoor_bridge.py no tenga que ramificar por modo: map_session.py cambia
    `self.mission` y listo. No usa VLM ni tramos, y nunca termina sola (la
    corrida se corta con --max-seconds o Ctrl-C).
    """

    def __init__(self, cfg: FrontierExploreConfig | None = None):
        self.cfg = cfg or FrontierExploreConfig()
        self.state = "EXPLORE"
        self.segments_done = 0
        self.failed_reason: str | None = None
        self.segment = _SegmentoFalso()
        self._goal_world: tuple[float, float] | None = None
        self._goal_t = 0.0

    def current_query(self):
        return None

    @property
    def checkpoints_done(self) -> int:
        return 0

    @property
    def lane(self):
        return None

    def update(self, pose: Pose, now: float, vlm=None, corridor=None,
               pmap: "PersistentMap | None" = None) -> MissionGoal:
        gp = self._frontier_goal(pose, now, pmap)
        x_right, y_forward = gp.to_bev_goal()
        motivo = ("explorando frontera" if self._goal_world is not None
                  else "sin frontera candidata, derecho adelante")
        return MissionGoal(x_right, y_forward, self.state, motivo)

    def _frontier_goal(self, pose: Pose, now: float,
                       pmap: "PersistentMap | None") -> GroundPoint:
        if pmap is not None:
            nueva = (self._goal_world is None
                     or (now - self._goal_t) > self.cfg.goal_hold_s)
            if not nueva:
                tx, ty = self._goal_world
                rel = Pose(tx, ty, 0.0).relative_to(pose)
                if math.hypot(rel.x, rel.y) < self.cfg.reach_radius_m:
                    nueva = True
            if nueva:
                elegida = pick_frontier_goal(pmap, pose, self.cfg)
                if elegida is not None:
                    self._goal_world = elegida
                    self._goal_t = now
            if self._goal_world is not None:
                tx, ty = self._goal_world
                rel = Pose(tx, ty, 0.0).relative_to(pose)
                return GroundPoint(rel.x, rel.y, math.hypot(rel.x, rel.y))

        return GroundPoint(self.cfg.wander_forward_m, 0.0, self.cfg.wander_forward_m)


# =============================================================================
#  MISION SEMANTICA: tramos rectos + hitos visuales
# =============================================================================
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
               corridor: CorridorHint | None = None,
               pmap: "PersistentMap | None" = None) -> MissionGoal:
        # `pmap` no se usa aca: esta en la firma solo para que esta FSM y
        # FrontierExploreFSM se puedan intercambiar sin tocar el bridge.
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
        modo. Un "checkpoint" aca es un TRAMO cerrado."""
        return self.segments_done

    @property
    def lane(self) -> HeadingLock | None:
        """El carril activo, para que route_status.py lo publique en el
        dashboard."""
        return self._lock


# --------------------------------------------------------------------- pruebas

def _self_test() -> None:
    print("=== GroundPoint -> convencion de plan_on_bev ===")
    gp = GroundPoint(2.0, 1.0, math.hypot(2.0, 1.0))
    x_right, y_forward = gp.to_bev_goal()
    print(f"  2 m adelante, 1 m a la izquierda -> x_right={x_right:+.2f} "
          f"y_forward={y_forward:+.2f}")
    assert x_right == -1.0 and y_forward == 2.0

    print("\n=== HeadingLock: carril, no punto fijo ===")
    lock = HeadingLock(Pose(0, 0, 0))
    assert abs(lock.goal_world(Pose(0, 0, 0), 2.5)[0] - 2.5) < 1e-6
    assert abs(lock.goal_world(Pose(10, 0, 0), 2.5)[0] - 12.5) < 1e-6, "la meta no se desliza"
    desviado = Pose(4.0, 0.8, 0.0)
    gx, gy = lock.goal_world(desviado, 2.5)
    print(f"  robot desviado 0.8 m -> meta ({gx:.2f}, {gy:.2f})")
    assert abs(gy) < 1e-6, "la meta deberia estar SOBRE el carril"
    assert abs(lock.lateral_error_m(desviado) - 0.8) < 1e-6
    assert abs(lock.progress_m(desviado) - 4.0) < 1e-6

    print("\n=== corridor_hint_from_bev ===")
    filas, cols, res = 100, 80, 0.05
    centro_col = (cols - 1) / 2.0
    bev = np.zeros((filas, cols), dtype=np.float32)
    bev[:, int(centro_col - 0.5 / res - 10):int(centro_col - 0.5 / res + 10)] = 1.0
    hint = corridor_hint_from_bev(bev, None, res, lookahead_m=2.5)
    print(f"  pasillo corrido a la izq -> offset={hint.lateral_offset_m:+.2f} m")
    assert hint.valid and hint.lateral_offset_m > 0.3
    vacio = corridor_hint_from_bev(np.zeros((filas, cols), np.float32), None, res, 2.5)
    assert not vacio.valid, "sin espacio libre el hint no puede ser valido"

    print("\n=== RouteHint: guia, nunca obligacion ===")
    th, motivo = RouteHint([(5.0, 1.0)], max_bias_deg=25.0).bias_heading(Pose(0, 0, 0), 0.0)
    print(f"  correccion chica -> {math.degrees(th):+.1f} grados ({motivo})")
    assert 0 < math.degrees(th) < 25
    th2, _ = RouteHint([(0.5, 5.0)], max_bias_deg=25.0).bias_heading(Pose(0, 0, 0), 0.0)
    assert abs(th2) < 1e-9, "una sugerencia muy distinta tiene que ignorarse"

    print("\n=== SemanticMissionFSM: histeresis del hito ===")
    cfg = SemanticMissionConfig.from_dict({
        "milestone_cooldown_s": 0.0,
        "segments": [{
            "id": "f1", "final": True, "on_timeout": "stop", "max_distance_m": 50.0,
            "milestone": {"id": "piano", "prompt": "¿se ve un piano?",
                          "confirm_hits": 3, "min_confidence": 0.6},
            "on_milestone": {"side": "left", "look_for": "una puerta"},
        }],
    })
    fsm = SemanticMissionFSM(cfg)
    pose = Pose(0, 0, 0)
    fsm.update(pose, 0.0)
    q = fsm.current_query()
    assert q is not None and q.id == "piano"
    g = fsm.update(pose, 1.0, vlm=VlmObservation(q.id, True, 0.9, t=1.0))
    print(f"  tras 1 confirmacion: {g.state}")
    assert g.state == "RUN_SEGMENT", "un solo frame no puede disparar el giro"
    fsm.update(pose, 1.5, vlm=VlmObservation(q.id, True, 0.9, t=1.5))
    g = fsm.update(pose, 2.0, vlm=VlmObservation(q.id, True, 0.9, t=2.0))
    print(f"  tras 3 confirmaciones seguidas: {g.state}")
    assert g.state == "TURN_SEARCH"

    print("\n=== FrontierExploreFSM (sesion de mapeo) ===")
    from ..persistent_map import MapConfig, PersistentMap
    pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.05))
    pmap.integrate(np.ones((100, 100), np.float32), np.ones((100, 100), np.uint8),
                   Pose(0, 0, 0), 2.0, 1.5, t=0.0)
    explorador = FrontierExploreFSM(FrontierExploreConfig(min_m=0.3, max_m=3.0,
                                                          preferred_m=1.0))
    meta = explorador.update(Pose(0, 0, 0), 0.0, pmap=pmap)
    print(f"  meta: ({meta.x_right_m:+.2f}, {meta.y_forward_m:+.2f}) — {meta.reason}")
    assert meta.y_forward_m > 0, "la frontera deberia estar adelante"
    assert explorador.current_query() is None and not meta.mission_done

    vacio_pmap = PersistentMap(MapConfig())
    meta_vacia = FrontierExploreFSM().update(Pose(0, 0, 0), 0.0, pmap=vacio_pmap)
    print(f"  mapa vacio -> {meta_vacia.reason}")
    assert meta_vacia.y_forward_m > 0, "sin frontera tiene que seguir derecho"

    print("\nTodos los asserts pasaron. El circuito completo de tramos se prueba "
          "en test_semantic_mission.py.")


if __name__ == "__main__":
    _self_test()
