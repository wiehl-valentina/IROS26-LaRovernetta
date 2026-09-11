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


# -------------------------------------------------------------------- calidad GPS

def gps_quality(hdop: float | None = None,
                fix_quality: int | None = None,
                gps_signal: float | None = None,
                hdop_nominal: float = 0.025,
                hdop_reject: float = 0.080) -> float:
    """Devuelve un factor de confianza del GPS en [0.0, 1.0].

    Arquitectura de validación (Fase 1):
    1. Gate primario: fix_quality (estándar NMEA).
       - 0 = sin fix / inválido -> 0.0 (descartar)
       - 1 = autónomo (SPS) -> 0.35 (confianza media, no RTK)
       - 2 = DGPS / SBAS -> 0.70 (alta)
       - 4 = RTK Fix -> 1.00 (máxima precisión centimétrica)
       - 5 = RTK Float -> 0.60 (precisión intermedia)
       Si fix_quality no está disponible (None o <= 0), se usa gps_signal como fallback:
       - gps_signal <= 0 -> 0.0
       - gps_signal > 0 -> min(1.0, gps_signal / 50.0) * 0.7
       Si no hay metadata de fix ni señal, asume 0.5 de base.

    2. Modulador secundario: hdop (dilución geométrica / métrica de dispersión).
       - hdop <= hdop_nominal -> multiplicador 1.0
       - hdop >= hdop_reject -> multiplicador 0.0 (rechazo por alta dispersión)
       - En (hdop_nominal, hdop_reject) -> decaimiento lineal a 0.0.

    NOTA DE CALIBRACIÓN: Los umbrales por defecto hdop_nominal=0.025 y hdop_reject=0.080
    son provisorios (no validados contra datos degradados reales en campo). Recalibrar
    en la primera corrida con GPS pobre.
    """
    if gps_signal is not None and gps_signal <= 0:
        return 0.0

    # 1. Gate primario: fix_quality
    if fix_quality is not None:
        if fix_quality <= 0:
            return 0.0
        elif fix_quality == 4:
            base_weight = 1.00
        elif fix_quality == 2:
            base_weight = 0.70
        elif fix_quality == 5:
            base_weight = 0.60
        elif fix_quality == 1:
            base_weight = 0.35
        else:
            base_weight = 0.30
    else:
        if gps_signal is not None and gps_signal > 0:
            base_weight = float(np.clip(gps_signal / 50.0 * 0.70, 0.1, 0.70))
        else:
            base_weight = 0.50

    # 2. Modulador secundario: hdop
    hdop_factor = 1.0
    if hdop is not None and hdop > 0.0:
        if hdop <= hdop_nominal:
            hdop_factor = 1.0
        elif hdop >= hdop_reject:
            hdop_factor = 0.0
        else:
            hdop_factor = (hdop_reject - hdop) / max(hdop_reject - hdop_nominal, 1e-6)
            hdop_factor = float(np.clip(hdop_factor, 0.0, 1.0))

    return float(np.clip(base_weight * hdop_factor, 0.0, 1.0))


# -------------------------------------------------------------------- rumbo

class HeadingEstimator:
    """Estima el rumbo del rover mediante fusión circular ponderada por incertidumbre.

    Fuentes fusionadas:
      * Compás (orientation del SDK): incertidumbre base fija configurable (~15°).
      * GPS track (course-over-ground): incertidumbre angular ~ atan(σ_pos / d) / calidad.
      * Giróscopo integrado: incertidumbre acumulativa σ(t) = sqrt(σ_base² + q²·Δt).

    Fusión circular ponderada por w_i = 1/σ_i²:
      S_x = sum(w_i * cos(rad(θ_i)))
      S_y = sum(w_i * sin(rad(θ_i)))
      θ_fused = deg(atan2(S_y, S_x)) % 360
      σ_fused = 1 / sqrt(sum(w_i))
    """

    def __init__(self, min_displacement_m: float = 1.5, history: int = 12,
                 orientation_offset_deg: float = 0.0, orientation_sign: float = 1.0,
                 trust_orientation: bool = False,
                 compass_sigma_deg: float = 15.0,
                 gyro_drift_rate_dps_per_sqrt_s: float = 0.5,
                 hdop_nominal: float = 0.025,
                 hdop_reject: float = 0.080,
                 use_ekf_udp: bool = True,
                 ekf_staleness_s: float = 1.5,
                 ekf_weight: float = 1.0):
        self.min_disp = float(min_displacement_m)
        self.buf: deque[tuple[float, float, float, float]] = deque(maxlen=int(history))
        self.offset = float(orientation_offset_deg)
        self.sign = float(orientation_sign)
        self.trust_orientation = bool(trust_orientation)
        self.compass_sigma = float(compass_sigma_deg)
        self.gyro_drift_rate = float(gyro_drift_rate_dps_per_sqrt_s)
        self.hdop_nominal = float(hdop_nominal)
        self.hdop_reject = float(hdop_reject)
        self.use_ekf_udp = bool(use_ekf_udp)
        self.ekf_staleness_s = float(ekf_staleness_s)
        self.ekf_weight = float(ekf_weight)

        self._heading: float | None = None
        self._uncertainty: float | None = None
        self._source = "none"
        self.last_gps_heading: float | None = None
        self.last_gps_sigma: float | None = None
        self.last_orientation_heading: float | None = None
        self.last_ekf_heading: float | None = None
        self.last_ekf_time: float | None = None

        # Estado del giróscopo integrado
        self._last_t: float | None = None
        self._gyro_heading: float | None = None
        self._gyro_sigma: float = float(compass_sigma_deg)

    def update(self, lat: float, lon: float, orientation: float, t: float,
               hdop: float | None = None, fix_quality: int | None = None,
               gps_signal: float | None = None,
               gyro_dps: float | None = None,
               ekf_heading: float | None = None,
               ekf_timestamp: float | None = None) -> float | None:
        """Actualiza el estimador con un nuevo conjunto de mediciones."""
        import time as _time

        # 0. Evaluar validez y frescura del heading UDP (EKF de ROS 2)
        ekf_valid = False
        if self.use_ekf_udp and ekf_heading is not None:
            self.last_ekf_heading = float(ekf_heading)
            if ekf_timestamp is not None:
                self.last_ekf_time = float(ekf_timestamp)
                age = _time.time() - self.last_ekf_time
            else:
                age = 0.0
            if age <= self.ekf_staleness_s:
                ekf_valid = True

        # 1. Evaluar calidad GPS
        q_gps = gps_quality(hdop=hdop, fix_quality=fix_quality, gps_signal=gps_signal,
                            hdop_nominal=self.hdop_nominal, hdop_reject=self.hdop_reject)
        if abs(lat) <= 90 and abs(lon) <= 180 and q_gps > 0.0:
            self.buf.append((lat, lon, t, q_gps))

        # 2. Medición del compás
        compass = wrap_deg(self.sign * orientation + self.offset) % 360.0
        self.last_orientation_heading = compass
        sigma_compass = self.compass_sigma

        # 3. Medición de GPS course-over-ground
        gps_track_res = self._heading_from_track()
        gps_heading: float | None = None
        sigma_gps: float | None = None
        if gps_track_res is not None:
            gps_heading, sigma_gps = gps_track_res
            self.last_gps_heading = gps_heading
            self.last_gps_sigma = sigma_gps
        else:
            self.last_gps_sigma = None

        # 4. Propagación del giróscopo integrado
        dt = 0.0
        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
        self._last_t = t

        if gyro_dps is not None and dt > 0 and dt < 1.0:
            if self._gyro_heading is not None:
                self._gyro_heading = (self._gyro_heading + gyro_dps * dt) % 360.0
                self._gyro_sigma = math.sqrt(self._gyro_sigma**2 + (self.gyro_drift_rate**2) * dt)
            elif self._heading is not None:
                self._gyro_heading = (self._heading + gyro_dps * dt) % 360.0
                self._gyro_sigma = math.sqrt((self._uncertainty or self.compass_sigma)**2 + (self.gyro_drift_rate**2) * dt)

        # 5. Modo EKF puro si está disponible y fresco
        if ekf_valid and self.last_ekf_heading is not None and self.ekf_weight >= 1.0:
            self._heading = wrap_deg(self.last_ekf_heading) % 360.0
            self._uncertainty = 2.0
            self._source = "ekf_udp"
            self._gyro_heading = self._heading
            self._gyro_sigma = 2.0
            return self._heading

        # 6. Override manual si trust_orientation está activo
        if self.trust_orientation:
            self._heading = compass
            self._uncertainty = sigma_compass
            self._source = "orientation"
            self._gyro_heading = compass
            self._gyro_sigma = sigma_compass
            return self._heading

        # 7. Fusión circular ponderada por 1 / sigma^2
        sources: list[tuple[float, float, str]] = []  # (angle_deg, sigma_deg, name)

        # Si hay EKF válido con blend parcial (< 1.0)
        if ekf_valid and self.last_ekf_heading is not None and self.ekf_weight > 0.0:
            sigma_ekf = 2.0 / max(self.ekf_weight, 0.05)
            sources.append((wrap_deg(self.last_ekf_heading) % 360.0, sigma_ekf, "ekf_udp"))

        # Compás siempre disponible
        sources.append((compass, sigma_compass, "compass"))

        # GPS track si superó el umbral y es válido
        if gps_heading is not None and sigma_gps is not None and sigma_gps < 60.0:
            sources.append((gps_heading, sigma_gps, "gps"))

        # Giróscopo si está activo y su sigma es razonable (< 45°)
        if self._gyro_heading is not None and self._gyro_sigma < 45.0 and gyro_dps is not None:
            sources.append((self._gyro_heading, self._gyro_sigma, "gyro"))

        # Promedio circular vectorial
        sx = 0.0
        sy = 0.0
        sum_w = 0.0
        names_used = []

        for angle_deg, sigma_deg, name in sources:
            w = 1.0 / max(sigma_deg**2, 1e-4)
            rad = math.radians(angle_deg)
            sx += w * math.cos(rad)
            sy += w * math.sin(rad)
            sum_w += w
            names_used.append(name)

        if sum_w > 0:
            fused_heading = math.degrees(math.atan2(sy, sx)) % 360.0
            fused_sigma = 1.0 / math.sqrt(sum_w)
        else:
            fused_heading = compass
            fused_sigma = sigma_compass
            names_used = ["compass"]

        self._heading = fused_heading
        self._uncertainty = fused_sigma
        if len(names_used) > 1:
            self._source = f"fused({'+'.join(names_used)})"
        elif "gps" in names_used:
            self._source = "gps_track"
        else:
            is_stale = (self.use_ekf_udp and self.last_ekf_heading is not None)
            self._source = "orientation(stale_fallback)" if is_stale else "orientation(fallback)"

        # Re-anclar giróscopo al rumbo fusionado
        self._gyro_heading = fused_heading
        self._gyro_sigma = fused_sigma

        return self._heading

    def _heading_from_track(self) -> tuple[float, float] | None:
        """Devuelve (heading_deg, sigma_deg) si el desplazamiento supera el umbral."""
        if len(self.buf) < 2:
            return None
        lat0, lon0, _, q0 = self.buf[0]
        lat1, lon1, _, q1 = self.buf[-1]
        q_eff = min(q0, q1)
        if q_eff <= 0.0:
            return None

        north, east = latlon_to_local_ne(lat0, lon0, lat1, lon1)
        disp = math.hypot(north, east)

        # Escalar min_displacement_m inversamente con la calidad GPS
        min_disp_eff = self.min_disp / max(q_eff, 0.25)
        if disp < min_disp_eff:
            return None

        heading_deg = math.degrees(math.atan2(east, north)) % 360.0
        # Incertidumbre angular: atan(sigma_pos / disp) en grados
        sigma_pos_m = 0.020 / max(q_eff, 0.1)
        sigma_rad = math.atan2(sigma_pos_m, disp)
        sigma_deg = float(np.clip(math.degrees(sigma_rad), 1.0, 90.0))
        return heading_deg, sigma_deg

    def reset_track(self) -> None:
        """Llamar despues de girar en el lugar: el track viejo ya no aplica."""
        self.buf.clear()
        if self._heading is not None:
            self._gyro_heading = self._heading
            self._gyro_sigma = self._uncertainty or self.compass_sigma

    @property
    def heading(self) -> float | None:
        return self._heading

    @property
    def uncertainty(self) -> float | None:
        return self._uncertainty

    @property
    def heading_uncertainty_deg(self) -> float | None:
        return self._uncertainty

    @property
    def source(self) -> str:
        return self._source

    def disagreement_deg(self) -> float | None:
        """Desacuerdo entre el heading activo y el curso GPS confiable (Ground Truth en rectas).

        Compara contra el curso GPS SOLO cuando este último tiene alta confianza
        (sigma_gps <= 15.0° derivado de suficiente avance rectilíneo y buen HDOP).
        Devuelve wrap_deg(heading_activo - last_gps_heading). Si el curso GPS no es confiable,
        devuelve None para evitar calibrar contra mediciones dudosas o estáticas.
        """
        if self.last_gps_heading is None or self.last_gps_sigma is None or self.last_gps_sigma > 15.0:
            return None

        # Heading activo: EKF si está disponible, o el rumbo fusionado actual
        ref_heading = self.last_ekf_heading if self.last_ekf_heading is not None else self._heading
        if ref_heading is None:
            return None

        return wrap_deg(ref_heading - self.last_gps_heading)

    def compass_distortion_deg(self) -> float | None:
        """Desacuerdo entre el compás crudo del SDK y la referencia confiable (EKF o GPS).

        Mide la interferencia magnética ambiental local (estructuras metálicas).
        Se reporta como diagnóstico informativo y NUNCA debe aplicarse como
        navigation.orientation_offset_deg en el config.
        """
        ref = self.last_ekf_heading if self.last_ekf_heading is not None else self.last_gps_heading
        if ref is None or self.last_orientation_heading is None:
            return None
        return wrap_deg(self.last_orientation_heading - ref)


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


def check_checkpoint_reached(rover_lat: float, rover_lon: float,
                             target_lat: float, target_lon: float,
                             radius_m: float = 13.0) -> tuple[bool, float]:
    """Verifica si el rover ha ingresado al radio del checkpoint objetivo.

    Portado de la lógica de gps_waypoint_controller (ROS 2) para La Rovernetta.
    Calcula la distancia geodésica euclidiana en plano tangente local y
    determina si está dentro del radio configurable checkpoint_reached_radius_m.

    Args:
        rover_lat: Latitud actual del rover (grados).
        rover_lon: Longitud actual del rover (grados).
        target_lat: Latitud del centro del checkpoint (grados).
        target_lon: Longitud del centro del checkpoint (grados).
        radius_m: Radio de arribo en metros (default: 13.0m).

    Returns:
        (reached: bool, distance_m: float)
    """
    north, east = latlon_to_local_ne(rover_lat, rover_lon, target_lat, target_lon)
    distance_m = math.hypot(north, east)
    return (distance_m <= float(radius_m)), distance_m


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
