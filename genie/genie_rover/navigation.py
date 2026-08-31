"""Navegacion: GPS -> meta local, estimacion de rumbo, seguimiento de camino.

Convenciones (las mismas que genie_path_planner):
  * meta y camino en [x_right_m, y_forward_m] relativo al robot
  * rumbo en grados de brujula: 0 = norte, 90 = este, sentido horario

Prueba standalone (matematica pura, no necesita robot):
    python -m genie_rover.navigation
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .odometry import Pose

EARTH_R = 6378137.0


# ------------------------------------------------------------------------ geo

def latlon_to_local_ne(lat_ref: float, lon_ref: float,
                       lat: float, lon: float) -> tuple[float, float]:
    """Desplazamiento (norte_m, este_m) desde el punto de referencia.

    Aproximacion de plano tangente. A las distancias del ERC (cientos de metros)
    el error es de centimetros, muy por debajo del ruido del GPS.
    """
    dlat = math.radians(lat - lat_ref)
    dlon = math.radians(lon - lon_ref)
    north = dlat * EARTH_R
    east = dlon * EARTH_R * math.cos(math.radians(lat_ref))
    return north, east


def wrap_deg(angle: float) -> float:
    """Normaliza a [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


# -------------------------------------------------------------------- rumbo

class HeadingEstimator:
    """Estima el rumbo del rover.

    El magnetometro de un robot chico va montado al lado de los motores, asi que
    su lectura suele ser poco confiable. La fuente primaria aca es el rumbo
    derivado del propio desplazamiento GPS (course over ground), que es ruidoso
    pero no tiene sesgo sistematico. El campo 'orientation' del SDK se usa como
    respaldo cuando el robot esta quieto, con offset y signo configurables.
    """

    def __init__(self, min_displacement_m: float = 1.5, history: int = 12,
                 orientation_offset_deg: float = 0.0, orientation_sign: float = 1.0,
                 trust_orientation: bool = False):
        self.min_disp = float(min_displacement_m)
        self.buf: deque[tuple[float, float, float]] = deque(maxlen=int(history))
        self.offset = float(orientation_offset_deg)
        self.sign = float(orientation_sign)
        self.trust_orientation = bool(trust_orientation)
        self._heading: float | None = None
        self._source = "none"
        self.last_gps_heading: float | None = None
        self.last_orientation_heading: float | None = None

    def update(self, lat: float, lon: float, orientation: float, t: float) -> float | None:
        self.buf.append((lat, lon, t))

        compass = wrap_deg(self.sign * orientation + self.offset) % 360.0
        self.last_orientation_heading = compass

        gps_heading = self._heading_from_track()
        if gps_heading is not None:
            self.last_gps_heading = gps_heading

        if self.trust_orientation:
            self._heading, self._source = compass, "orientation"
        elif gps_heading is not None:
            self._heading, self._source = gps_heading, "gps_track"
        else:
            # Sin rumbo por GPS: seguir la brujula. Es ruidosa y con offset
            # variable, pero SIGUE las rotaciones, que es lo unico que hace
            # falta para que el lazo de navegacion cierre.
            #
            # Antes esta rama era "elif self._heading is None", asi que solo
            # corria en el PRIMER frame: despues el rumbo quedaba congelado en
            # ese valor inicial aunque el robot girara. goal_from_gps calcula
            # el rumbo relativo al checkpoint como (bearing - heading), asi que
            # con heading muerto la direccion de la meta giraba junto con el
            # robot y nunca se podia apuntar: giraba en el lugar o describia
            # arcos hacia una direccion equivocada.
            self._heading, self._source = compass, "orientation(fallback)"
        return self._heading

    def _heading_from_track(self) -> float | None:
        if len(self.buf) < 2:
            return None
        lat0, lon0, _ = self.buf[0]
        lat1, lon1, _ = self.buf[-1]
        north, east = latlon_to_local_ne(lat0, lon0, lat1, lon1)
        if math.hypot(north, east) < self.min_disp:
            return None
        return math.degrees(math.atan2(east, north)) % 360.0

    def reset_track(self) -> None:
        """Llamar despues de girar en el lugar: el track viejo ya no aplica."""
        self.buf.clear()

    @property
    def heading(self) -> float | None:
        return self._heading

    @property
    def source(self) -> str:
        return self._source

    def disagreement_deg(self) -> float | None:
        """Cuanto difieren GPS y magnetometro. Util para calibrar el offset."""
        if self.last_gps_heading is None or self.last_orientation_heading is None:
            return None
        return wrap_deg(self.last_gps_heading - self.last_orientation_heading)


# --------------------------------------------------------------------- meta

@dataclass
class LocalGoal:
    x_right_m: float
    y_forward_m: float
    distance_m: float
    relative_bearing_deg: float


def goal_from_gps(lat: float, lon: float, heading_deg: float,
                  target_lat: float, target_lon: float,
                  max_range_m: float) -> LocalGoal:
    """Convierte un checkpoint GPS en una meta local acotada al alcance del BEV.

    Cuando el checkpoint esta lejos (el caso normal en el ERC, cientos de
    metros) se lo recorta a max_range_m. Eso es la "meta intermedia" del paper:
    el planner solo necesita saber en que direccion tirar, no la distancia.
    """
    north, east = latlon_to_local_ne(lat, lon, target_lat, target_lon)
    distance = math.hypot(north, east)
    bearing = math.degrees(math.atan2(east, north)) % 360.0
    rel = wrap_deg(bearing - heading_deg)

    r = min(distance, float(max_range_m))
    rel_rad = math.radians(rel)
    return LocalGoal(
        x_right_m=r * math.sin(rel_rad),
        y_forward_m=r * math.cos(rel_rad),
        distance_m=distance,
        relative_bearing_deg=rel,
    )


# ------------------------------------------------------ camino <-> mundo

def path_to_world(path_xy_robot: np.ndarray, pose: "Pose") -> np.ndarray:
    """Camino [x_right_m, y_forward_m] relativo al robot -> puntos (x, y) del
    mundo, con la pose del robot al momento de planificar.

    Usa el mismo cambio de base que persistent_map.integrate(): el mundo va
    en la convencion x adelante / y izquierda (la de odometry.Pose), y el
    camino del planner va en x derecha / y adelante. 'adelante' del camino es
    x_robot en esa convencion, 'derecha' es -y_robot.
    """
    p = np.asarray(path_xy_robot, dtype=np.float64)
    x_fwd = p[:, 1]
    y_left = -p[:, 0]
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x_world = pose.x + c * x_fwd - s * y_left
    y_world = pose.y + s * x_fwd + c * y_left
    return np.stack([x_world, y_world], axis=1)


def path_to_robot(path_xy_world: np.ndarray, pose: "Pose") -> np.ndarray:
    """Inversa de path_to_world: puntos del mundo -> [x_right_m, y_forward_m]
    relativo a la pose actual del robot.

    Es lo que permite seguir un plan calculado hace unos frames sin volver a
    llamar al planner: el camino no se mueve, pero el punto de vista desde el
    que se lo describe (el robot) si, y hay que reproyectarlo en cada
    iteracion antes de pasarselo a PathFollower.
    """
    p = np.asarray(path_xy_world, dtype=np.float64)
    dx = p[:, 0] - pose.x
    dy = p[:, 1] - pose.y
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x_fwd = c * dx + s * dy
    y_left = -s * dx + c * dy
    return np.stack([-y_left, x_fwd], axis=1)


# -------------------------------------------------------------- controlador

@dataclass
class DriveCommand:
    linear: float
    angular: float
    reason: str


class PathFollower:
    """Convierte el camino de GeNIE en (linear, angular) para el SDK.

    Sigue el ciclo conservador del paper: si el error de rumbo es grande, gira
    en el lugar; si no, avanza reduciendo la velocidad segun el error.

    angular_sign existe porque la documentacion del SDK se contradice sobre si
    angular positivo es izquierda o derecha. Determinalo con
    tools/check_angular_sign.py antes de confiar en esto.
    """

    def __init__(self, lookahead_m: float = 1.0, align_threshold_deg: float = 25.0,
                 max_linear: float = 0.35, max_angular: float = 0.5,
                 turn_speed: float = 0.35, kp_angular: float = 0.9,
                 angular_sign: float = -1.0, min_linear_while_following: float = 0.0):
        self.lookahead_m = float(lookahead_m)
        self.align_threshold = float(align_threshold_deg)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.turn_speed = float(turn_speed)
        self.kp = float(kp_angular)
        self.angular_sign = float(angular_sign)
        # Velocidad lineal minima mientras se sigue un plan ya comprometido:
        # girar en el lugar no aporta nada si ya hay un camino elegido (la
        # unica razon real para pivotear es alinearse ANTES de tener un plan).
        # Curvar en vez de pivotear evita el patron arranca-frena-arranca.
        self.min_linear_while_following = float(min_linear_while_following)

    def lookahead_point(self, path_xy: np.ndarray) -> np.ndarray | None:
        """Punto a lookahead_m de arco desde el robot."""
        p = np.asarray(path_xy, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2:
            return None
        # El planner emite el camino desde el robot hacia adelante, pero si
        # algun dia cambia el orden esto lo detecta y lo corrige.
        if np.linalg.norm(p[0]) > np.linalg.norm(p[-1]):
            p = p[::-1]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= self.lookahead_m:
            return p[-1]
        idx = int(np.searchsorted(cum, self.lookahead_m))
        return p[min(idx, len(p) - 1)]

    def command(self, path_xy: np.ndarray | None, committed: bool = False) -> DriveCommand:
        """committed=True indica que path_xy no es una alineacion inicial
        sino un plan ya elegido que se viene siguiendo (recien calculado o
        reproyectado entre replanificaciones): ahi aplica min_linear_while_following
        en vez de pivotear en el lugar."""
        if path_xy is None or len(path_xy) < 2:
            return DriveCommand(0.0, 0.0, "sin camino")

        target = self.lookahead_point(path_xy)
        if target is None:
            return DriveCommand(0.0, 0.0, "camino degenerado")

        # x_right positivo = objetivo a la derecha
        error_deg = math.degrees(math.atan2(float(target[0]), float(target[1])))

        if abs(error_deg) > self.align_threshold:
            ang = self.angular_sign * math.copysign(self.turn_speed, error_deg)
            linear = self.min_linear_while_following if committed else 0.0
            reason = "curvando" if committed else "girando en el lugar"
            return DriveCommand(float(np.clip(linear, 0.0, self.max_linear)),
                                float(np.clip(ang, -self.max_angular, self.max_angular)),
                                f"{reason} ({error_deg:+.0f} grados)")

        ratio = abs(error_deg) / max(self.align_threshold, 1e-6)
        linear = self.max_linear * (1.0 - 0.5 * ratio)
        ang = self.angular_sign * self.kp * math.radians(error_deg)
        return DriveCommand(
            float(np.clip(linear, 0.0, self.max_linear)),
            float(np.clip(ang, -self.max_angular, self.max_angular)),
            f"siguiendo camino ({error_deg:+.0f} grados)",
        )


# ----------------------------------------------------------- chequeo frontal

def front_is_blocked(bev: np.ndarray, resolution_m: float,
                     near_m: float = 0.25, far_m: float = 0.9,
                     half_width_m: float = 0.30,
                     traversable_thresh: float = 0.4,
                     min_free_ratio: float = 0.5) -> bool:
    """Deteccion de colision estilo GeNIE: mira la franja justo delante.

    Corre a la frecuencia del bucle, independiente del planner, para poder
    frenar aunque el camino planificado siga pareciendo valido.
    """
    h, w = bev.shape
    r_far = h - 1 - int(far_m / resolution_m)
    r_near = h - 1 - int(near_m / resolution_m)
    c_half = int(half_width_m / resolution_m)
    c_mid = w // 2

    r0, r1 = max(0, r_far), min(h, r_near + 1)
    c0, c1 = max(0, c_mid - c_half), min(w, c_mid + c_half + 1)
    if r0 >= r1 or c0 >= c1:
        return False

    patch = bev[r0:r1, c0:c1]
    known = patch >= 0.0
    if not np.any(known):
        return False  # nada observado: que decida el planner, no frenamos a ciegas
    free_ratio = float(np.mean(patch[known] > traversable_thresh))
    return free_ratio < float(min_free_ratio)


def front_clearance_m(bev: np.ndarray, resolution_m: float,
                      near_m: float = 0.25, max_check_m: float = 1.5,
                      half_width_m: float = 0.30, traversable_thresh: float = 0.4,
                      min_free_ratio: float = 0.5, row_step_m: float = 0.05) -> float:
    """Version continua de front_is_blocked: distancia libre al frente, en vez
    de un bool. La usan tanto el perfil de velocidad como el disparo del
    regimen cercano.

    Barre filas desde near_m hacia max_check_m y devuelve la distancia de la
    primera que no llega a min_free_ratio de celdas transitables. Si una fila
    no tiene ninguna celda observada, tambien corta ahi: mas alla de lo
    observado no se puede afirmar que este libre.
    """
    h, w = bev.shape
    c_half = max(1, int(half_width_m / resolution_m))
    c_mid = w // 2
    c0, c1 = max(0, c_mid - c_half), min(w, c_mid + c_half + 1)
    if c0 >= c1:
        return float(max_check_m)

    step_px = max(1, int(round(row_step_m / resolution_m)))
    r_near = h - 1 - int(near_m / resolution_m)
    r_far = h - 1 - int(max_check_m / resolution_m)

    r = r_near
    dist = near_m
    while r > r_far and r >= 0:
        row = bev[r, c0:c1]
        known = row >= 0.0
        if not np.any(known):
            return float(dist)
        free_ratio = float(np.mean(row[known] > traversable_thresh))
        if free_ratio < min_free_ratio:
            return float(dist)
        r -= step_px
        dist += row_step_m
    return float(max_check_m)


# --------------------------------------------------------------------- tests

def _self_test() -> None:
    print("=== geo ===")
    # Un grado de latitud ~ 111 km
    n, e = latlon_to_local_ne(-34.9214, -57.9544, -34.9214 + 0.001, -57.9544)
    print(f"  +0.001 lat -> norte={n:.1f} m este={e:.1f} m  (esperado ~111 m, ~0 m)")
    assert 110 < n < 112 and abs(e) < 0.5

    n, e = latlon_to_local_ne(-34.9214, -57.9544, -34.9214, -57.9544 + 0.001)
    print(f"  +0.001 lon -> norte={n:.1f} m este={e:.1f} m  (esperado ~0 m, ~91 m a esta latitud)")
    assert abs(n) < 0.5 and 90 < e < 93

    print("\n=== meta desde GPS ===")
    lat, lon = -34.9214, -57.9544
    # Objetivo 100 m al norte, robot mirando al norte -> derecho adelante
    tgt = (lat + 100.0 / EARTH_R * 180.0 / math.pi, lon)
    g = goal_from_gps(lat, lon, 0.0, tgt[0], tgt[1], max_range_m=3.5)
    print(f"  mirando al norte, meta al norte: x_right={g.x_right_m:+.2f} y_forward={g.y_forward_m:+.2f} "
          f"dist={g.distance_m:.0f} m rel={g.relative_bearing_deg:+.0f} grados")
    assert abs(g.x_right_m) < 0.05 and abs(g.y_forward_m - 3.5) < 0.05

    # Mismo objetivo, robot mirando al este -> meta a la izquierda
    g = goal_from_gps(lat, lon, 90.0, tgt[0], tgt[1], max_range_m=3.5)
    print(f"  mirando al este,  meta al norte: x_right={g.x_right_m:+.2f} y_forward={g.y_forward_m:+.2f} "
          f"rel={g.relative_bearing_deg:+.0f} grados")
    assert g.x_right_m < -3.0 and abs(g.y_forward_m) < 0.05

    print("\n=== controlador ===")
    f = PathFollower(lookahead_m=1.0, angular_sign=-1.0)
    recto = np.stack([np.zeros(20), np.linspace(0, 2, 20)], axis=1)
    c = f.command(recto)
    print(f"  camino recto:            linear={c.linear:.2f} angular={c.angular:+.2f}  ({c.reason})")
    assert c.linear > 0.3 and abs(c.angular) < 0.02

    derecha = np.stack([np.linspace(0, 1.5, 20), np.linspace(0, 1.5, 20)], axis=1)
    c = f.command(derecha)
    print(f"  camino a la derecha:     linear={c.linear:.2f} angular={c.angular:+.2f}  ({c.reason})")
    assert c.linear == 0.0 and c.angular < 0  # con angular_sign=-1, derecha => angular negativo

    izquierda = np.stack([np.linspace(0, -1.5, 20), np.linspace(0, 1.5, 20)], axis=1)
    c = f.command(izquierda)
    print(f"  camino a la izquierda:   linear={c.linear:.2f} angular={c.angular:+.2f}  ({c.reason})")
    assert c.angular > 0

    c = f.command(None)
    print(f"  sin camino:              linear={c.linear:.2f} angular={c.angular:+.2f}  ({c.reason})")
    assert c.linear == 0.0 and c.angular == 0.0

    f2 = PathFollower(lookahead_m=1.0, angular_sign=-1.0, min_linear_while_following=0.08)
    c = f2.command(derecha, committed=True)
    print(f"  giro grande, plan comprometido: linear={c.linear:.2f} angular={c.angular:+.2f}  ({c.reason})")
    assert c.linear >= 0.08 - 1e-9, "deberia curvar, no pivotear, con un plan comprometido"

    print("\n=== chequeo frontal ===")
    libre = np.ones((134, 134), dtype=np.float32)
    bloqueado = np.ones((134, 134), dtype=np.float32)
    bloqueado[100:130, 50:84] = 0.05
    print(f"  BEV libre:      bloqueado={front_is_blocked(libre, 0.03)}")
    print(f"  BEV con muro:   bloqueado={front_is_blocked(bloqueado, 0.03)}")
    assert not front_is_blocked(libre, 0.03)
    assert front_is_blocked(bloqueado, 0.03)

    desconocido = np.full((134, 134), -1.0, dtype=np.float32)
    print(f"  BEV sin observar: bloqueado={front_is_blocked(desconocido, 0.03)} (no frena a ciegas)")

    print("\n=== clearance continuo ===")
    print(f"  BEV libre:      clearance={front_clearance_m(libre, 0.03, max_check_m=1.5):.2f} m (esperado ~1.5)")
    print(f"  BEV con muro:   clearance={front_clearance_m(bloqueado, 0.03, max_check_m=1.5):.2f} m (esperado bajo)")
    assert front_clearance_m(libre, 0.03, max_check_m=1.5) > 1.4
    assert front_clearance_m(bloqueado, 0.03, max_check_m=1.5) < 1.0

    print("\n=== camino <-> mundo (disparo espacial) ===")
    class _PoseStub:
        def __init__(self, x, y, theta):
            self.x, self.y, self.theta = x, y, theta

    # Robot en el origen mirando al "norte" del mundo (theta=0): un punto
    # derecho adelante en el camino (x_right=0, y_forward=1.5) tiene que caer
    # en (x=1.5, y=0) del mundo con esta convencion (x adelante, y izquierda).
    recto2 = np.array([[0.0, 1.5]])
    pose0 = _PoseStub(0.0, 0.0, 0.0)
    w = path_to_world(recto2, pose0)
    print(f"  derecho adelante desde el origen -> mundo {w[0]}")
    assert abs(w[0, 0] - 1.5) < 1e-6 and abs(w[0, 1]) < 1e-6

    # Ida y vuelta: planificado desde pose0, reproyectado sobre una pose que
    # avanzo 1 m y giro 90 grados -> el punto tiene que seguir siendo el mismo
    # lugar del mundo, descripto ahora desde el nuevo punto de vista.
    pose1 = _PoseStub(1.0, 0.0, math.pi / 2)
    back = path_to_robot(w, pose1)
    w2 = path_to_world(back, pose1)
    print(f"  ida y vuelta tras moverse: mundo original {w[0]}, mundo tras ida y vuelta {w2[0]}")
    assert np.allclose(w[0], w2[0], atol=1e-6)

    print("\n=== rumbo ===")
    he = HeadingEstimator(min_displacement_m=1.5)
    he.update(lat, lon, 0.0, 0.0)
    print(f"  1 muestra:  heading={he.heading} fuente={he.source}")
    for i in range(1, 8):
        he.update(lat + i * 3e-6, lon, 0.0, float(i))
    print(f"  moviendose al norte: heading={he.heading:.1f} grados fuente={he.source} (esperado ~0)")
    assert he.heading is not None and (he.heading < 5 or he.heading > 355)

    # Regresion: girando en el lugar el GPS no se mueve, asi que no hay
    # gps_track. El rumbo tiene que seguir a la brujula igual. Si vuelve a
    # quedar congelado en el primer valor, el robot gira sin poder apuntar
    # nunca al checkpoint.
    he2 = HeadingEstimator(min_displacement_m=1.5)
    vistos = [he2.update(lat, lon, o, float(i)) for i, o in enumerate([7, 90, 200, 300])]
    print(f"  girando en el lugar (GPS quieto): brujula 7->300, heading {vistos}")
    assert len(set(vistos)) > 1, "el rumbo quedo congelado: la brujula giro y el heading no"
    assert abs(wrap_deg(vistos[-1] - 300.0)) < 1e-6, "el rumbo no sigue a la brujula"

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
