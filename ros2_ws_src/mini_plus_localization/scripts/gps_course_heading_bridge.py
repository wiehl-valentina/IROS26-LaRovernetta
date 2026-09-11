#!/usr/bin/env python3
"""GPS Course Heading Bridge Node for Mini+ Rover (ROS 2 Jazzy).

Fase 2: Anclaje de rumbo absoluto por curso GNSS (course-over-ground) con
transición suave de covarianza e instrumentación de confianza en tiempo real.

==============================================================================
PASO A.0 — DOCUMENTACIÓN HONESTA DE LIMITACIONES Y ORIGEN DE UMBRALES
==============================================================================
1. ORIGEN DE LOS UMBRALES DE CALIDAD GNSS:
   - Los umbrales de activación:
     * HDOP < 2.5 (rechazo) / HDOP <= 1.0 (óptimo)
     * Velocidad lineal |v| > 0.15 m/s (corte) / |v| >= 0.35 m/s (nominal)
     * Desplazamiento acumulado Δd > 0.40 m (corte) / Δd >= 1.00 m (nominal)
   SON UMBRALES ASUMIDOS BASADOS EN HEURÍSTICAS DE MANUAL Y BUENAS PRÁCTICAS,
   NO EN MEDICIONES DE CAMPO CON GPS DEGRADADO REAL.
   Las pruebas de campo disponibles (Test D) se registraron en un sitio con
   recepción GNSS óptima (HDOP ~ 0.012). Por lo tanto, estos umbrales deben ser
   revisados y ajustados empíricamente cuando se disponga de telemetría de sitios
   con baja cobertura satelital o multicamino severo.

2. LIMITACIÓN FUNDAMENTAL CONOCIDA:
   - Si se presenta de forma simultánea saturación ferromagnética en el compás
     (Fase 1: compás rechazado) y degradación severa o pérdida de señal GNSS
     (Fase 2: sin avance recto con fix confiable), el sistema NO resuelve el
     heading de fondo; el EKF operará en modo inercial puro (integrando el
     giróscopo debiasado) y acumulará deriva angular con el tiempo.
   - Esta Fase 2 NO soluciona esa doble falla simultánea: provee detección,
     transición continua y la señal de diagnóstico en tiempo real
     ('/earth_rover/heading_confidence') con el contador de tiempo sin ancla
     ('time_without_anchor_s') para que las capas superiores de navegación
     puedan reaccionar adecuadamente.
==============================================================================
"""

import json
import math
import time
from collections import deque
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float32, String

# Radio medio terrestre (WGS-84 esférico)
EARTH_RADIUS_M = 6371000.0

# ==============================================================================
# PASO A.2 — TABLA DE CONVERGENCIA DE COVARIANZA DERIVADA DE DATOS (TEST D)
#
# 1. PUNTOS REALES EXACTOS MEDIDOS EN TEST D:
#    (>=1.0m -> 4.44°, >=2.0m -> 3.71°, >=3.0m -> 2.73°, >=5.0m -> 1.74°)
#    Convertidos directamente a varianza en rad^2: var = math.radians(sigma_deg)**2.
#
# 2. DISTANCIAS INTERMEDIAS (ej. 0.5m, 4.0m):
#    Interpoladas linealmente ENTRE las mediciones reales más cercanas (o entre
#    el piso asumido y la primera medición). Etiquetadas como
#    "interpolado linealmente entre mediciones", NUNCA "medido".
#
# 3. SUPUESTOS DE EXTRAPOLACIÓN FUERA DEL RANGO MEDIDO:
#    - Superior (> 5.0m): Se asume que la dispersión angular del GNSS no mejora
#      significativamente más allá de 5 metros de avance recto; se retiene la
#      asíntota medida en >=5m (1.74° -> 0.0009223 rad^2) constante para d > 5m.
#    - Inferior (< 1.0m): En el umbral de activación d_min = 0.40m, se asume un
#      piso conservador de sigma = 9.07° (0.02500 rad^2, incertidumbre nominal base
#      sin convergencia de avance). Para d < 0.40m se mantiene dicho piso constante
#      (además de que el score de confianza S_disp = 0.0 infla a 1e6 rad^2).
# ==============================================================================
CONVERGENCE_TABLE = [
    # (dist_m, sigma_deg, var_rad2, tipo)
    (0.40, 9.07, 0.0250000, "PISO ASUMIDO (umbral activacion d_min=0.4m, sin convergencia)"),
    (1.00, 4.44, math.radians(4.44) ** 2, "MEDIDO (Test D exacto: 4.44 deg)"),
    (2.00, 3.71, math.radians(3.71) ** 2, "MEDIDO (Test D exacto: 3.71 deg)"),
    (3.00, 2.73, math.radians(2.73) ** 2, "MEDIDO (Test D exacto: 2.73 deg)"),
    (5.00, 1.74, math.radians(1.74) ** 2, "MEDIDO (Test D exacto: 1.74 deg, asintota)"),
]

# Alias de compatibilidad (dist_m, var_rad2)
EMPIRICAL_CONVERGENCE_TABLE = [(d, c) for d, _, c, _ in CONVERGENCE_TABLE]


def interpolate_base_covariance(dist_m: float) -> float:
    """Interpola linealmente la covarianza base entre mediciones reales de Test D."""
    if dist_m <= CONVERGENCE_TABLE[0][0]:
        # Extrapolación inferior: piso conservador constante
        return CONVERGENCE_TABLE[0][2]
    if dist_m >= CONVERGENCE_TABLE[-1][0]:
        # Extrapolación superior: asíntota medida constante
        return CONVERGENCE_TABLE[-1][2]
    for i in range(len(CONVERGENCE_TABLE) - 1):
        d0, _, c0, _ = CONVERGENCE_TABLE[i]
        d1, _, c1, _ = CONVERGENCE_TABLE[i + 1]
        if d0 <= dist_m <= d1:
            frac = (dist_m - d0) / (d1 - d0)
            return c0 + frac * (c1 - c0)
    return CONVERGENCE_TABLE[-1][2]


def latlon_to_local_ne(lat_ref: float, lon_ref: float,
                       lat: float, lon: float) -> Tuple[float, float]:
    """Calcula desplazamiento (norte_m, este_m) en plano tangente local."""
    dlat = math.radians(lat - lat_ref)
    dlon = math.radians(lon - lon_ref)
    north = dlat * EARTH_RADIUS_M
    east = dlon * EARTH_RADIUS_M * math.cos(math.radians(lat_ref))
    return north, east


class GpsCourseHeadingBridge(Node):
    """Calcula el rumbo por vector de avance GNSS y lo inyecta como PoseWithCovarianceStamped al EKF."""

    def __init__(self):
        super().__init__("gps_course_heading_bridge")

        # ----------------------------------------------------------------------
        # Parámetros configurables
        # ----------------------------------------------------------------------
        self.declare_parameter("v_min", 0.15)          # m/s (corte de velocidad lineal)
        self.declare_parameter("v_nom", 0.35)          # m/s (velocidad de plena confianza)
        self.declare_parameter("disp_min", 0.40)       # m (corte de desplazamiento acumulado)
        self.declare_parameter("disp_nom", 1.00)       # m (desplazamiento de plena confianza)
        self.declare_parameter("hdop_max", 2.5)        # HDOP de corte de calidad
        self.declare_parameter("hdop_good", 1.0)       # HDOP de máxima calidad
        self.declare_parameter("turn_rate_thresh_dps", 8.0)  # °/s (umbral para considerar giro)
        self.declare_parameter("untrusted_cov", 1e6)   # rad^2 (covarianza ante no anclaje)
        self.declare_parameter("reverse_vel_thresh", -0.05)  # m/s (detección de reversa)

        self._v_min = float(self.get_parameter("v_min").value)
        self._v_nom = float(self.get_parameter("v_nom").value)
        self._disp_min = float(self.get_parameter("disp_min").value)
        self._disp_nom = float(self.get_parameter("disp_nom").value)
        self._hdop_max = float(self.get_parameter("hdop_max").value)
        self._hdop_good = float(self.get_parameter("hdop_good").value)
        self._turn_rate_thresh_dps = float(self.get_parameter("turn_rate_thresh_dps").value)
        self._untrusted_cov = float(self.get_parameter("untrusted_cov").value)
        self._reverse_vel_thresh = float(self.get_parameter("reverse_vel_thresh").value)

        # ----------------------------------------------------------------------
        # Perfiles QoS
        # ----------------------------------------------------------------------
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # ----------------------------------------------------------------------
        # Publicadores
        # ----------------------------------------------------------------------
        # Inyección directa a ekf_filter_node_map (pose0)
        self.heading_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/odometry/gps_heading", reliable_qos
        )
        # Diagnóstico de rumbo en grados de brújula (0°=Norte, horario)
        self.course_heading_pub = self.create_publisher(
            Float32, "/earth_rover/gps_course_heading", sensor_qos
        )
        # Paso A.5: Señal de confianza consolidada y tiempo sin ancla
        self.confidence_pub = self.create_publisher(
            String, "/earth_rover/heading_confidence", sensor_qos
        )

        # ----------------------------------------------------------------------
        # Suscripciones
        # ----------------------------------------------------------------------
        self.create_subscription(NavSatFix, "/gps/fix", self._on_gps_fix, reliable_qos)
        self.create_subscription(Odometry, "/wheel_odom", self._on_wheel_odom, reliable_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, sensor_qos)
        self.create_subscription(String, "/earth_rover/mag_gate_diag", self._on_mag_diag, sensor_qos)

        # ----------------------------------------------------------------------
        # Estado interno de cinemática y odometría
        # ----------------------------------------------------------------------
        self._linear_speed_m_s = 0.0
        self._angular_speed_dps = 0.0
        self._cmd_linear_x = 0.0
        self._last_odom_time = 0.0

        # Estado del buffer de posiciones para curso GPS
        # Almacena (lat, lon, timestamp, v_mps, hdop)
        self._track_buf: deque[Tuple[float, float, float, float, float]] = deque(maxlen=60)
        self._last_valid_yaw_enu: Optional[float] = None
        self._last_gps_course_deg: Optional[float] = None

        # Estado del compás (Fase 1)
        self._mag_confidence_score = 0.0
        self._mag_trusted = False
        self._last_mag_diag_time = 0.0

        # Estado del anclaje y temporizador (Paso A.5)
        self._last_anchored_monotonic = time.monotonic()
        self._gps_course_confidence_score = 0.0
        self._gps_course_trusted = False

        # Timer periódico para diagnóstico de confianza a 5 Hz
        self._diag_timer = self.create_timer(0.20, self._publish_confidence_diag)

        self.get_logger().info(
            f"gps_course_heading_bridge iniciado. v_min={self._v_min} m/s, "
            f"disp_min={self._disp_min} m, hdop_max={self._hdop_max}"
        )

    def _on_wheel_odom(self, msg: Odometry):
        self._linear_speed_m_s = msg.twist.twist.linear.x
        self._angular_speed_dps = math.degrees(msg.twist.twist.angular.z)
        self._last_odom_time = time.monotonic()

        # Si el robot está girando activamente en el lugar, invalidar el buffer de recta
        if abs(self._angular_speed_dps) > self._turn_rate_thresh_dps:
            self._reset_track_buffer()

    def _on_cmd_vel(self, msg: Twist):
        self._cmd_linear_x = msg.linear.x

    def _on_mag_diag(self, msg: String):
        try:
            payload = json.loads(msg.data)
            self._mag_confidence_score = float(payload.get("confidence_score", 0.0))
            self._mag_trusted = bool(payload.get("trusted", False))
            self._last_mag_diag_time = time.monotonic()
        except Exception:
            pass

    def _reset_track_buffer(self):
        """Reinicia el buffer de avance en recta (ej. tras rotación o parada)."""
        self._track_buf.clear()

    def _on_gps_fix(self, msg: NavSatFix):
        now_mono = time.monotonic()

        # 1. Validación de fix GNSS
        is_fix_ok = (msg.status.status >= NavSatStatus.STATUS_FIX and
                     abs(msg.latitude) <= 90.0 and abs(msg.longitude) <= 180.0 and
                     msg.latitude != 0.0)

        # 2. Extracción de HDOP
        # bridge_node escalas covarianza como base * hdop^2 con base=1.0
        hdop = 1.0
        if msg.position_covariance[0] > 0:
            hdop = math.sqrt(msg.position_covariance[0])

        t_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Determinar velocidad actual efectiva
        effective_speed = abs(self._linear_speed_m_s)
        if effective_speed < 0.01 and abs(self._cmd_linear_x) > 0.05:
            # Fallback a comando si la telemetría de odometría tarda
            effective_speed = abs(self._cmd_linear_x)

        # Si el rover está quieto o rotando, vaciar buffer de recta
        if effective_speed < self._v_min or abs(self._angular_speed_dps) > self._turn_rate_thresh_dps:
            self._reset_track_buffer()

        if is_fix_ok and effective_speed >= self._v_min:
            self._track_buf.append((msg.latitude, msg.longitude, t_stamp, effective_speed, hdop))

        # 3. Cálculo de desplazamiento y rumbo
        disp = 0.0
        yaw_enu = None
        compass_deg = None

        if len(self._track_buf) >= 2:
            lat0, lon0, _, _, _ = self._track_buf[0]
            lat1, lon1, _, _, _ = self._track_buf[-1]
            north, east = latlon_to_local_ne(lat0, lon0, lat1, lon1)
            disp = math.hypot(north, east)

            if disp >= self._disp_min:
                # Rumbo de movimiento en coordenadas ENU
                # atan2(north, east): 0 rad = Este, +pi/2 rad = Norte
                motion_yaw_enu = math.atan2(north, east)

                # Desambiguación avance vs retroceso (adelante / atrás)
                # Si velocidad de avance < -0.05 m/s o comando < -0.05, rover marcha atrás
                is_reversing = (self._linear_speed_m_s < self._reverse_vel_thresh or
                                self._cmd_linear_x < self._reverse_vel_thresh)

                if is_reversing:
                    # Si marcha atrás hacia el norte, la trompa mira al sur
                    vehicle_yaw_enu = math.atan2(
                        math.sin(motion_yaw_enu + math.pi),
                        math.cos(motion_yaw_enu + math.pi)
                    )
                else:
                    vehicle_yaw_enu = motion_yaw_enu

                yaw_enu = vehicle_yaw_enu
                compass_deg = (90.0 - math.degrees(yaw_enu)) % 360.0
                self._last_valid_yaw_enu = yaw_enu
                self._last_gps_course_deg = compass_deg

        # 4. PASO A.3 — Puntuación de confianza continua y covarianza suave
        s_v = max(0.0, min(1.0, (effective_speed - self._v_min) / max(1e-4, self._v_nom - self._v_min)))
        s_d = max(0.0, min(1.0, (disp - self._disp_min) / max(1e-4, self._disp_nom - self._disp_min)))
        s_hdop = max(0.0, min(1.0, (self._hdop_max - hdop) / max(1e-4, self._hdop_max - self._hdop_good)))

        if not is_fix_ok or yaw_enu is None:
            gps_course_score = 0.0
        else:
            gps_course_score = s_v * s_d * s_hdop

        self._gps_course_confidence_score = gps_course_score
        self._gps_course_trusted = (gps_course_score >= 0.5)

        # PASO A.2: Covarianza base interpolada de Test D
        base_cov = interpolate_base_covariance(disp)

        # Transición continua a covarianza untrusted (1e6) cuando la confianza cae
        if gps_course_score >= 0.999:
            cov_yaw = base_cov
        else:
            cov_yaw = base_cov + ((1.0 - gps_course_score) ** 2) * self._untrusted_cov

        # 5. Publicación en /odometry/gps_heading (PoseWithCovarianceStamped)
        target_yaw = yaw_enu if yaw_enu is not None else (self._last_valid_yaw_enu or 0.0)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = "map"  # Inyección en marco global map

        pose_msg.pose.pose.position.x = 0.0
        pose_msg.pose.pose.position.y = 0.0
        pose_msg.pose.pose.position.z = 0.0

        pose_msg.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(target_yaw / 2.0),
            w=math.cos(target_yaw / 2.0),
        )

        # Inicializar covarianza 6x6 (36 elementos) con varianza alta para estados no medidos
        cov_matrix = [0.0] * 36
        for diag_idx in [0, 7, 14, 21, 28]:
            cov_matrix[diag_idx] = self._untrusted_cov
        cov_matrix[35] = float(cov_yaw)  # [5, 5] = yaw variance
        pose_msg.pose.covariance = cov_matrix

        self.heading_pose_pub.publish(pose_msg)

        # Publicar rumbo en grados para monitoreo
        if compass_deg is not None:
            course_msg = Float32()
            course_msg.data = float(compass_deg)
            self.course_heading_pub.publish(course_msg)

    def _publish_confidence_diag(self):
        """Paso A.5: Publica el estado de confianza y el tiempo transcurrido sin ancla."""
        now_mono = time.monotonic()

        # Comprobar frescura del diagnóstico de compás (< 3s)
        mag_fresh = (now_mono - self._last_mag_diag_time) < 3.0
        effective_mag_trusted = self._mag_trusted if mag_fresh else False

        # Evaluación de ancla activa
        any_anchor_active = effective_mag_trusted or self._gps_course_trusted

        if any_anchor_active:
            self._last_anchored_monotonic = now_mono
            time_without_anchor_s = 0.0
        else:
            time_without_anchor_s = now_mono - self._last_anchored_monotonic

        # Determinar fuente ancla descriptiva
        if effective_mag_trusted and self._gps_course_trusted:
            active_anchor = "both"
        elif self._gps_course_trusted:
            active_anchor = "gps_course"
        elif effective_mag_trusted:
            active_anchor = "mag_compass"
        else:
            active_anchor = "none"

        diag_data = {
            "time_without_anchor_s": round(time_without_anchor_s, 2),
            "mag_confidence_score": round(self._mag_confidence_score, 3),
            "mag_trusted": effective_mag_trusted,
            "gps_course_confidence_score": round(self._gps_course_confidence_score, 3),
            "gps_course_trusted": self._gps_course_trusted,
            "active_anchor": active_anchor,
            "anchor_available": any_anchor_active,
        }

        diag_msg = String()
        diag_msg.data = json.dumps(diag_data)
        self.confidence_pub.publish(diag_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GpsCourseHeadingBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
