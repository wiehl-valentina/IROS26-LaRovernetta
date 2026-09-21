#!/usr/bin/env python3
"""GPS Course Heading Bridge Node for Mini+ Rover (ROS 2 Jazzy).

# Fase 2: Anclaje de rumbo absoluto por curso GNSS (course-over-ground) con
# ventana adaptativa por desplazamiento neto, referencia por promedio en ventana (Opción b),
# desambiguación por signo de RPMs, mitigación de sobreconfianza en EKF e instrumentación en tiempo real.
#
# ==============================================================================
# PASO A.0 / FASE 2 — DOCUMENTACIÓN HONESTA DE LIMITACIONES Y DISEÑO
# ==============================================================================
# 1. ORIGEN DE LAS 0 MUESTRAS ÚTILES EN CORRIDA REAL DE 83M (Paso 2.1):
#    En la implementación anterior, el buffer de puntos se vaciaba en dos condiciones:
#      (a) `if abs(self._angular_speed_dps) > self._turn_rate_thresh_dps: self._reset_track_buffer()`
#      (b) `if effective_speed < self._v_min or abs(self._angular_speed_dps) > self._turn_rate_thresh_dps: ...`
#    con `turn_rate_thresh_dps = 8.0` °/s (apenas 0.14 rad/s).
#    En una navegación en exteriores esquivando obstáculos o zigzagueando en pasto,
#    las correcciones motrices superan continuamente 8°/s. Como consecuencia, el buffer
#    se reseteaba a cero repetidamente antes de poder acumular el desplazamiento mínimo
#    necesario, produciendo exactamente 0 muestras útiles de curso GPS en toda la corrida.
#    SOLUCIÓN DE FASE 2: Se eliminó el reseteo por velocidad angular. La ventana no se vacía
#    por girar; la dispersión angular durante el avance se mide mediante la desviación
#    estándar circular y modula continuamente la confianza (inflando la covarianza sin escalones).
#
# 2. MITIGACIÓN DE SOBRECONFIANZA EN EKF POR VENTANAS SOLAPADAS (Paso 2.2 / Revisión):
#    Al evaluar la ventana deslizante a 1 Hz, dos evaluaciones consecutivas comparten la mayor
#    parte de sus fixes (solapamiento temporal). Si se inyectan como observaciones independientes,
#    el EKF contrae espuriamente su covarianza posterior (sobreconfianza).
#    SOLUCIÓN:
#      - Por default (`disjoint_windows=False`), se infla la covarianza base por el factor de
#        solapamiento N_window: `base_cov_eff = base_cov * N_window`. De esta forma, la ganancia de
#        información acumulada por el filtro a lo largo de los N segundos de la ventana equivale
#        exactamente a 1 sola observación estadísticamente independiente, manteniendo la publicación
#        continua a 1 Hz requerida para no violar `sensor_timeout: 2.0` de `ekf.yaml`.
#      - Alternativamente, `disjoint_windows=True` emite únicamente ventanas no solapadas (disjuntas).
#
# 3. UNIFICACIÓN DEL MODELO DE RUIDO GPS (Fase 1 vs Fase 2):
#    - En Fase 1 (guarda de salto), el margen de ruido se fijó en 1.5 m (`gps_jump_noise_margin_m = 1.5`),
#      representativo del error horizontal 2-sigma en GNSS autónomo/DGPS estándar en cielo abierto.
#    - En Fase 2, la tabla empírica de convergencia (`CONVERGENCE_TABLE`) proviene del Test D, grabado
#      en un sitio con recepción GNSS EXCELENTE / RTK (HDOP ~ 0.012, dispersión horizontal ~0.08 - 0.10 m).
#    - ADVERTENCIA CRÍTICA: En sitios con mala cobertura satelital o sin corrección diferencial,
#      el ruido real es de ~1.5 m; en tales condiciones, un avance de d_min = 2.0 m tiene una incertidumbre
#      angular geométrica de arctan(1.5 / 2.0) ≈ 36° (no los 3.71° de Test D). En esos escenarios
#      degradados, LA ÚNICA DEFENSA del sistema es el factor reductor por HDOP (`s_hdop`), cuya escala
#      (`hdop_good=1.0`, `hdop_max=2.5`) es ASUMIDA por heurísticas de manual y NO está validada en campo
#      con GPS degradado real.
#
# 4. REFERENCIA DE RUMBO EN VENTANA (Paso 2.3 — Opción b):
#    La cuerda neta representa la dirección promedio de traslación durante la ventana [t_start, t_end],
#    no necesariamente la actitud instantánea de la trompa del rover al cerrar la ventana (especialmente
#    en un zigzag).
#    Se evaluaron dos opciones:
#      (a) Publicar la cuerda con el timestamp del punto medio de la ventana: Descartada porque en
#          `ekf.yaml` el parámetro `delay` es de 100 ms (0.1s) y `smooth_lagged_data` / `history_length`
#          no están activados. Publicar con 2-5 segundos de retraso provocaría el rechazo de la
#          medición por el EKF o requeriría un rewind/replay computacionalmente costoso en el microcontrolador.
#      (b) Opción adoptada: Se calcula la corrección angular respecto a la media estimada en la ventana:
#              correccion = wrap_angle(cuerda_yaw - mean_yaw_ventana)
#              yaw_publicado = wrap_angle(yaw_actual + correccion)
#          Esta opción es una aproximación: asume que el giróscopo integró razonablemente la variación
#          relativa dentro de la ventana de unos pocos segundos, transfiriendo la calibración absoluta
#          de la cuerda directamente a la orientación instantánea actual del vehículo, publicada
#          con el timestamp actual para consumo inmediato por el EKF.
#
# 5. UMBRALES DE DISTANCIA (Paso 2.2 / Revisión):
#    `disp_min = 1.50` m (corte unificado) y `disp_nom = 3.00` m son valores ELEGIDOS POR DISEÑO (ASUMIDOS).
#    - Reducción de disp_min a 1.50 m (justificación de costo en ruido):
#      A d = 2.0 m, sigma_1sigma = 3.71° (Test D).
#      A d = 1.50 m, sigma_1sigma ≈ 4.07° (interpolación Test D; cota conservadora ~5.4°).
#      A 3-sigma (16.2°) + derrape ASUMIDO (8.0°) = 24.2° < 30.0° (umbral de gating).
#      Costo en ruido: aumento marginal de ~0.36° a 1.7° en 1-sigma, compensado ampliamente por una
#      latencia de anclaje 25% menor (4.3s vs 5.7s a 0.35 m/s), crucial en maniobras urbanas/jardín.
#    - Aclaración: la mención previa a cuerdas de 1.3 m fue un ejemplo ilustrativo de tramos
#      cortos en logs de corrida manual pasada; el piso operacional unificado es estrictamente 1.50 m.
#
# 6. DERIVA INERCIAL PURA EN ZIGZAG PROLONGADO:
#    - Bias residual de giróscopo medido en reposo: bg ≈ 0.083°/s (≈ 5.0°/min).
#    - Durante un zigzag prolongado sin ancla (coherence_zigzag_timeout_s = 60.0 s), la deriva
#      acumulada en inercial puro es de ~5.0° (error transversal ~1.3 m sobre 15 m de avance),
#      perfectamente tolerable por el campo de visión del BEV costmap y evitación de obstáculos.
#    - Al superar los 60.0 s (is_prolonged_zigzag = True), el sistema alerta al supervisor de misión
#      para reducir velocidad (degraded_linear_scale = 0.6) o solicitar escaneo, PREVINIENDO la
#      reactivación ciega del compás si este fue previamente invalidado por discrepancia.
# ==============================================================================
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
from std_msgs.msg import Bool, Float32, String

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
#    Interpoladas linealmente ENTRE las mediciones reales más cercanas.
#
# 3. SUPUESTOS DE EXTRAPOLACIÓN FUERA DEL RANGO MEDIDO:
#    - Superior (> 5.0m): Se retiene la asíntota medida en >=5m (1.74° -> 0.0009223 rad^2) constante.
#    - Inferior (< 1.0m): En d_min = 0.40m piso conservador de sigma = 9.07° (0.02500 rad^2).
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
        return CONVERGENCE_TABLE[0][2]
    if dist_m >= CONVERGENCE_TABLE[-1][0]:
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
        # Parámetros configurables (Fase 2)
        # ----------------------------------------------------------------------
        self.declare_parameter("v_min", 0.15)          # m/s (corte de velocidad lineal)
        self.declare_parameter("v_nom", 0.35)          # m/s (velocidad de plena confianza)
        self.declare_parameter("disp_min", 1.50)       # m (desplazamiento neto mínimo unificado de alta confianza)
        self.declare_parameter("disp_nom", 3.00)       # m (desplazamiento de plena confianza de tabla empírica)
        self.declare_parameter("t_max", 15.0)          # s (tiempo máximo de ventana antes de descartar fixes si no alcanza disp_min)
        self.declare_parameter("hdop_max", 2.5)        # HDOP de corte de calidad
        self.declare_parameter("hdop_good", 1.0)       # HDOP de máxima calidad
        self.declare_parameter("turn_rate_thresh_dps", 8.0)  # °/s (conservado por compatibilidad de parámetros; ya NO resetea buffer)
        self.declare_parameter("untrusted_cov", 1e6)   # rad^2 (covarianza ante no anclaje)
        self.declare_parameter("reverse_vel_thresh", -0.05)  # m/s (detección de reversa)
        # ASUMIDO: Valida la geometría de la cuerda dentro de la ventana (rectitud del avance).
        # NO detecta ni penaliza deriva del heading estimado (debe anclar GPS con plena confianza
        # en tramos con deriva siempre que el rover avance en una trayectoria sin virajes bruscos).
        self.declare_parameter("heading_spread_nominal_deg", 20.0)
        self.declare_parameter("heading_spread_max_deg", 45.0)      # ° (límite superior de dispersión donde la cuerda deja de representar traslación)
        self.declare_parameter("disjoint_windows", False)           # Si True, emite sólo ventanas disjuntas; si False, infla covarianza por solapamiento
        # Gating de Coherencia Compás vs Curso GNSS (a) y Giróscopo en Viraje (b)
        self.declare_parameter("coherence_heading_diff_thresh_deg", 30.0)   # ° (cota 3-sigma 16.2° + 8° derrape ASUMIDO = 24.2° < 30.0°)
        # ASUMIDO: Umbral de discrepancia compás vs integral de giróscopo en viraje de 8s.
        # Presupuesto de error:
        #   1. Error de factor de escala del giróscopo (~5% medido): en viraje de 90° aporta ~4.5°.
        #   2. Error de integración dinámica: el transitorio de aceleración angular (w_dot ≈ 30-100°/s²)
        #      dura solo t_ramp ≈ 0.2-0.3s (no los 2.0s de cadencia de paquetes del SDK; en crucero w_dot ≈ 0).
        #      Con 1 a 5 muestras internas intra-paquete (dt ≈ 0.4s), el error de integración Euler en rampas
        #      es de ~5-7° (la fórmula 1/2 * w_dot * dt^2 con dt=2s daría 60° espuriamente si se asumiera
        #      aceleración constante sostenida durante todo el paquete).
        #   3. Deriva de bias en 8s: ~0.083°/s * 8s ≈ 0.7°.
        #   4. Retardo / amortiguamiento dinámico de la brújula: ~5.0°.
        #   Total presupuesto: 4.5° + 6.0° + 0.7° + 5.0° = 16.2° -> margen de seguridad a 25.0°.
        # Pendiente de calibrar en campo con un giro en sitio sin interferencia magnética (en tests
        # anteriores el compás estuvo congelado).
        self.declare_parameter("coherence_turn_rate_error_thresh_deg", 25.0) # ° ASUMIDO
        self.declare_parameter("coherence_min_conflict_windows", 3)         # N=3 ventanas independientes consecutivas para declarar conflicto en (a)
        self.declare_parameter("coherence_recovery_windows", 2)             # N=2 ventanas independientes consecutivas para reactivación suave
        self.declare_parameter("coherence_zigzag_timeout_s", 60.0)          # s (tiempo límite sin cuerdas de alta confianza)

        self._v_min = float(self.get_parameter("v_min").value)
        self._v_nom = float(self.get_parameter("v_nom").value)
        self._disp_min = float(self.get_parameter("disp_min").value)
        self._disp_nom = float(self.get_parameter("disp_nom").value)
        self._t_max = float(self.get_parameter("t_max").value)
        self._hdop_max = float(self.get_parameter("hdop_max").value)
        self._hdop_good = float(self.get_parameter("hdop_good").value)
        self._turn_rate_thresh_dps = float(self.get_parameter("turn_rate_thresh_dps").value)
        self._untrusted_cov = float(self.get_parameter("untrusted_cov").value)
        self._reverse_vel_thresh = float(self.get_parameter("reverse_vel_thresh").value)
        self._heading_spread_nominal_deg = float(self.get_parameter("heading_spread_nominal_deg").value)
        self._heading_spread_max_deg = float(self.get_parameter("heading_spread_max_deg").value)
        self._disjoint_windows = bool(self.get_parameter("disjoint_windows").value)
        self._coherence_heading_diff_thresh_deg = float(self.get_parameter("coherence_heading_diff_thresh_deg").value)
        self._coherence_turn_rate_error_thresh_deg = float(self.get_parameter("coherence_turn_rate_error_thresh_deg").value)
        self._coherence_min_conflict_windows = int(self.get_parameter("coherence_min_conflict_windows").value)
        self._coherence_recovery_windows = int(self.get_parameter("coherence_recovery_windows").value)
        self._coherence_zigzag_timeout_s = float(self.get_parameter("coherence_zigzag_timeout_s").value)

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
        # Publicador de conflicto GPS vs compás (anula compás en EKF)
        self.gps_mag_conflict_pub = self.create_publisher(
            Bool, "/earth_rover/gps_mag_conflict", reliable_qos
        )
        self.mag_coherence_diag_pub = self.create_publisher(
            String, "/earth_rover/mag_coherence_diag", sensor_qos
        )

        # Estado del Gating de Coherencia
        self._consecutive_gps_mag_conflicts = 0
        self._consecutive_gps_mag_agreements = 0
        self._gps_mag_conflict = False
        self._gyro_mag_turn_conflict = False
        self._last_high_conf_chord_time = time.monotonic()
        self._last_discrepancy_deg = 0.0
        self._last_rot_err_deg = 0.0
        self._last_gating_lat: Optional[float] = None
        self._last_gating_lon: Optional[float] = None

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

        # Ventana adaptativa de fixes GNSS: almacena (lat, lon, timestamp, v_mps, hdop)
        self._gps_window: deque[Tuple[float, float, float, float, float]] = deque(maxlen=120)
        self._track_buf = self._gps_window  # Alias para compatibilidad hacia atrás

        # Ventana temporal de odometría/orientación para evaluar rumbo medio y variabilidad angular
        # Almacena (timestamp, yaw_enu, omega_z_rad_s, v_linear_x)
        self._odom_window: deque[Tuple[float, Optional[float], float, float]] = deque(maxlen=300)

        self._current_yaw_enu: Optional[float] = None
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
            f"gps_course_heading_bridge iniciado (Fase 2). v_min={self._v_min} m/s, "
            f"disp_min={self._disp_min} m, t_max={self._t_max} s, hdop_max={self._hdop_max}"
        )

    def _on_wheel_odom(self, msg: Odometry):
        self._linear_speed_m_s = msg.twist.twist.linear.x
        self._angular_speed_dps = math.degrees(msg.twist.twist.angular.z)
        self._last_odom_time = time.monotonic()

        t_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t_stamp <= 0.0:
            t_stamp = self._last_odom_time

        q = msg.pose.pose.orientation
        norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if norm_sq > 0.5:
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._current_yaw_enu = math.atan2(siny_cosp, cosy_cosp)

        # Buffer temporal de odometría y orientación para la ventana adaptativa
        self._odom_window.append(
            (t_stamp, self._current_yaw_enu, msg.twist.twist.angular.z, self._linear_speed_m_s)
        )
        while self._odom_window and (t_stamp - self._odom_window[0][0]) > (self._t_max + 2.0):
            self._odom_window.popleft()

        # FASE 2: SE ELIMINA EL VACIADO DEL BUFFER POR VELOCIDAD ANGULAR (|w_z| > 8°/s).
        # En la versión anterior, cualquier corrección de rumbo o zigzag en pasto (> 8°/s = 0.14 rad/s)
        # vaciaba el buffer instantáneamente, resultando en 0 muestras útiles en la corrida de 83m.
        # En el nuevo diseño, la ventana adaptativa conserva los fixes y mide la dispersión angular
        # inflando continuamente la covarianza.

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
        """Reinicia la ventana adaptativa (ej. ante desanclaje o reinicio manual)."""
        self._gps_window.clear()
        self._odom_window.clear()
        self._last_gating_lat = None
        self._last_gating_lon = None

    def _on_gps_fix(self, msg: NavSatFix):
        now_mono = time.monotonic()

        # 1. Validación de fix GNSS
        is_fix_ok = (msg.status.status >= NavSatStatus.STATUS_FIX and
                     abs(msg.latitude) <= 90.0 and abs(msg.longitude) <= 180.0 and
                     msg.latitude != 0.0)

        # 2. Extracción de HDOP
        hdop = 1.0
        if msg.position_covariance[0] > 0:
            hdop = math.sqrt(msg.position_covariance[0])

        t_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t_stamp <= 0.0:
            t_stamp = now_mono

        # Determinar velocidad actual efectiva en m/s
        effective_speed = abs(self._linear_speed_m_s)
        if effective_speed < 0.01 and abs(self._cmd_linear_x) > 0.05:
            # ASUMIDO: Fallback a comando si la telemetría de odometría tarda.
            # cmd_linear_x es acelerador normalizado [0, 1] donde 1.0 ~ 4.0 km/h (1.111 m/s).
            # Relación lineal provisional asumida hasta medir curva comando->velocidad en campo (Paso 0.2):
            effective_speed = abs(self._cmd_linear_x) * (4.0 / 3.6)

        # Paso 2.2: Descartar de la ventana fixes expirados (t_stamp - t > t_max)
        while self._gps_window and (t_stamp - self._gps_window[0][2]) > self._t_max:
            self._gps_window.popleft()

        if is_fix_ok and effective_speed >= self._v_min:
            self._gps_window.append((msg.latitude, msg.longitude, t_stamp, effective_speed, hdop))

        # 3. Evaluación de ventana adaptativa por desplazamiento neto
        disp = 0.0
        yaw_enu = None
        compass_deg = None
        heading_spread_deg = 0.0
        mixed_directions = False

        if len(self._gps_window) >= 2:
            lat_new, lon_new, t_new, _, hdop_new = self._gps_window[-1]

            # Buscar desde el fix más viejo hasta encontrar el primero que cumpla desplazamiento neto >= disp_min
            idx_found = None
            north_found = 0.0
            east_found = 0.0
            disp_found = 0.0

            for i in range(len(self._gps_window) - 1):
                lat_old, lon_old, t_old, _, _ = self._gps_window[i]
                n, e = latlon_to_local_ne(lat_old, lon_old, lat_new, lon_new)
                d = math.hypot(n, e)
                if d >= self._disp_min:
                    idx_found = i
                    north_found = n
                    east_found = e
                    disp_found = d
                    break

            if idx_found is not None:
                disp = disp_found
                t_start = self._gps_window[idx_found][2]
                t_end = t_new

                # Ventana adaptativa:
                if self._disjoint_windows:
                    last_fix = self._gps_window[-1]
                    self._gps_window.clear()
                    self._gps_window.append(last_fix)
                else:
                    # Descartar fixes anteriores a idx_found para mantener ventana deslizante
                    for _ in range(idx_found):
                        self._gps_window.popleft()

                # Extraer muestras de odometría/orientación en el intervalo [t_start, t_end]
                odom_samples = [
                    s for s in self._odom_window
                    if (t_start - 0.25) <= s[0] <= (t_end + 0.25)
                ]
                if not odom_samples and self._odom_window:
                    odom_samples = list(self._odom_window)

                # Desambiguación adelante / atrás con el signo de las RPM / velocidad lineal (Paso 2.4)
                speeds = [s[3] for s in odom_samples] if odom_samples else [self._linear_speed_m_s]
                has_forward = any(s > 0.05 for s in speeds) or (self._cmd_linear_x > 0.05)
                has_reverse = any(s < self._reverse_vel_thresh for s in speeds) or (self._cmd_linear_x < self._reverse_vel_thresh)

                if has_forward and has_reverse:
                    # Mezcla de avance y retroceso en la ventana -> descartar ventana
                    mixed_directions = True
                    self._gps_window.clear()
                else:
                    is_reversing = has_reverse or (self._linear_speed_m_s < self._reverse_vel_thresh)
                    motion_yaw_enu = math.atan2(north_found, east_found)

                    if is_reversing:
                        chord_yaw_enu = math.atan2(
                            math.sin(motion_yaw_enu + math.pi),
                            math.cos(motion_yaw_enu + math.pi)
                        )
                    else:
                        chord_yaw_enu = motion_yaw_enu

                    # Paso 2.3: Opción (b) — Referencia de rumbo por promedio en ventana
                    yaws = [s[1] for s in odom_samples if s[1] is not None]
                    if len(yaws) >= 2:
                        sin_m = sum(math.sin(y) for y in yaws) / len(yaws)
                        cos_m = sum(math.cos(y) for y in yaws) / len(yaws)
                        R = math.hypot(sin_m, cos_m)
                        mean_yaw = math.atan2(sin_m, cos_m)

                        # Dispersión circular en grados
                        R_clamped = max(1e-6, min(1.0, R))
                        sigma_rad = math.sqrt(-2.0 * math.log(R_clamped))
                        sigma_deg = math.degrees(sigma_rad)

                        max_dev_rad = max(
                            abs(math.atan2(math.sin(y - mean_yaw), math.cos(y - mean_yaw)))
                            for y in yaws
                        )
                        max_dev_deg = math.degrees(max_dev_rad)
                        heading_spread_deg = max(sigma_deg, max_dev_deg / math.sqrt(2.0))

                        # Opción (b): corrección = cuerda - promedio_heading_ventana
                        # Publicar rumbo actual corregido por el desvío medido de la cuerda
                        correction = math.atan2(
                            math.sin(chord_yaw_enu - mean_yaw),
                            math.cos(chord_yaw_enu - mean_yaw)
                        )
                        curr_yaw = self._current_yaw_enu if self._current_yaw_enu is not None else yaws[-1]
                        vehicle_yaw_enu = math.atan2(
                            math.sin(curr_yaw + correction),
                            math.cos(curr_yaw + correction)
                        )
                    else:
                        heading_spread_deg = 0.0
                        vehicle_yaw_enu = chord_yaw_enu

                    yaw_enu = vehicle_yaw_enu
                    compass_deg = (90.0 - math.degrees(yaw_enu)) % 360.0
                    self._last_valid_yaw_enu = yaw_enu
                    self._last_gps_course_deg = compass_deg

                    # Gating de Coherencia (a): Curso GPS vs Promedio Circular de Compás
                    # Evaluado únicamente sobre ventanas INDEPENDIENTES (sin solapamiento espacial)
                    d_since_last_gating = 0.0
                    if self._last_gating_lat is not None:
                        gn, ge = latlon_to_local_ne(self._last_gating_lat, self._last_gating_lon, lat_new, lon_new)
                        d_since_last_gating = math.hypot(gn, ge)
                    else:
                        d_since_last_gating = disp

                    is_independent_gating = self._disjoint_windows or (d_since_last_gating >= self._disp_min)

                    if is_independent_gating:
                        self._last_gating_lat = lat_new
                        self._last_gating_lon = lon_new

                        is_high_conf_chord = (
                            is_fix_ok
                            and not mixed_directions
                            and disp >= self._disp_min  # >= 1.50 m
                            and hdop <= self._hdop_good # <= 1.0
                            and heading_spread_deg <= self._heading_spread_nominal_deg # <= 20.0°
                            and effective_speed >= self._v_min
                            and len(yaws) >= 2
                        )
                        now_mono = time.monotonic()
                        if is_high_conf_chord:
                            self._last_high_conf_chord_time = now_mono
                            discrepancy_rad = abs(math.atan2(math.sin(chord_yaw_enu - mean_yaw), math.cos(chord_yaw_enu - mean_yaw)))
                            discrepancy_deg = math.degrees(discrepancy_rad)
                            self._last_discrepancy_deg = discrepancy_deg

                            # (a) Cota 3-sigma (16.2°) + derrape ASUMIDO (8.0°) = 24.2° < 30.0°
                            # Exigir N=3 ventanas INDEPENDIENTES consecutivas para confirmar error sistemático del compás
                            if discrepancy_deg > self._coherence_heading_diff_thresh_deg:
                                self._consecutive_gps_mag_conflicts += 1
                                self._consecutive_gps_mag_agreements = 0
                                if self._consecutive_gps_mag_conflicts >= self._coherence_min_conflict_windows:
                                    self._gps_mag_conflict = True
                                    self.get_logger().warn(
                                        f"[GATING-AB] Conflicto sistemático GPS vs Compás ({self._consecutive_gps_mag_conflicts} ventanas): "
                                        f"Δθ={discrepancy_deg:.1f}° > {self._coherence_heading_diff_thresh_deg:.1f}°. Compás invalidado en EKF."
                                    )
                            elif discrepancy_deg <= (self._coherence_heading_diff_thresh_deg - 10.0):  # <= 20.0°
                                self._consecutive_gps_mag_agreements += 1
                                self._consecutive_gps_mag_conflicts = 0
                                if self._consecutive_gps_mag_agreements >= self._coherence_recovery_windows:
                                    if self._gps_mag_conflict:
                                        self.get_logger().info(
                                            f"[GATING-AB] Compás rehabilitado: concordancia GPS-Compás confirmada en "
                                            f"{self._consecutive_gps_mag_agreements} ventanas consecutivas (Δθ={discrepancy_deg:.1f}°)."
                                        )
                                    self._gps_mag_conflict = False
                        # Ventanas que NO califican (dispersión > 20°, HDOP fuera de rango, etc.) NO suman ni resetean el contador.

        # 4. Confianza continua y covarianza suave (Paso 2.4)
        s_v = max(0.0, min(1.0, (effective_speed - self._v_min) / max(1e-4, self._v_nom - self._v_min)))
        if disp >= self._disp_min:
            s_d = 0.5 + 0.5 * min(1.0, (disp - self._disp_min) / max(1e-4, self._disp_nom - self._disp_min))
        else:
            s_d = 0.0

        s_hdop = max(0.0, min(1.0, (self._hdop_max - hdop) / max(1e-4, self._hdop_max - self._hdop_good)))

        if heading_spread_deg <= self._heading_spread_nominal_deg:
            s_heading = 1.0
        elif heading_spread_deg >= self._heading_spread_max_deg:
            s_heading = 0.0
        else:
            s_heading = (self._heading_spread_max_deg - heading_spread_deg) / (
                self._heading_spread_max_deg - self._heading_spread_nominal_deg
            )

        if not is_fix_ok or yaw_enu is None or mixed_directions:
            gps_course_score = 0.0
        else:
            gps_course_score = s_v * s_d * s_hdop * s_heading

        self._gps_course_confidence_score = gps_course_score
        self._gps_course_trusted = (gps_course_score >= 0.5)

        base_cov = interpolate_base_covariance(disp)
        if not self._disjoint_windows:
            n_overlap = max(1.0, float(len(self._gps_window)))
            base_cov = base_cov * n_overlap

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

    def _check_gyro_mag_turn_coherence(self):
        """Chequeo (b): Cambio de compás vs integral de giróscopo en ventana de 8.0 s."""
        if len(self._odom_window) < 4:
            return
        t_now = self._odom_window[-1][0]
        # Filtrar muestras en ventana de 8.0 s (cubre ~4 reportes del SDK con tasa de ~2s)
        win = [s for s in self._odom_window if (t_now - s[0]) <= 8.0]
        if len(win) < 4:
            return

        # Integral del giróscopo debiasado
        delta_gyro_rad = 0.0
        for i in range(1, len(win)):
            dt = win[i][0] - win[i - 1][0]
            if 0.0 < dt < 3.0:
                delta_gyro_rad += win[i][2] * dt
        delta_gyro_deg = math.degrees(delta_gyro_rad)

        first_yaw = win[0][1]
        last_yaw = win[-1][1]
        if first_yaw is not None and last_yaw is not None:
            delta_mag_rad = math.atan2(math.sin(last_yaw - first_yaw), math.cos(last_yaw - first_yaw))
            delta_mag_deg = math.degrees(delta_mag_rad)

            rot_err_deg = abs(delta_mag_deg - delta_gyro_deg)
            self._last_rot_err_deg = rot_err_deg

            # Presupuesto de error en viraje de 8s (5% de escala + integración dispersa):
            # Solo evaluar si hubo giro significativo (|Δgyro| >= 15.0°). Umbral 25.0°.
            if abs(delta_gyro_deg) >= 15.0:
                if rot_err_deg > self._coherence_turn_rate_error_thresh_deg:
                    if not self._gyro_mag_turn_conflict:
                        self.get_logger().warn(
                            f"[GATING-AB] Incoherencia en viraje compás vs giróscopo: "
                            f"rot_err={rot_err_deg:.1f}° > {self._coherence_turn_rate_error_thresh_deg:.1f}° "
                            f"(Δgyro={delta_gyro_deg:.1f}°, Δmag={delta_mag_deg:.1f}°)"
                        )
                    self._gyro_mag_turn_conflict = True
                else:
                    self._gyro_mag_turn_conflict = False

    def _publish_confidence_diag(self):
        """Paso A.5: Publica el estado de confianza y el tiempo transcurrido sin ancla."""
        now_mono = time.monotonic()

        # Evaluación del chequeo de coherencia en viraje (b)
        self._check_gyro_mag_turn_coherence()

        # Estado consolidado de conflicto (a) o (b)
        # Nota: Un conflicto originado por (a) NO puede ser limpiado por (b)
        conflict_active = self._gps_mag_conflict or self._gyro_mag_turn_conflict
        conflict_msg = Bool()
        conflict_msg.data = bool(conflict_active)
        self.gps_mag_conflict_pub.publish(conflict_msg)

        # Condición de zigzag prolongado sin cuerdas de alta confianza (Punto 4)
        time_since_high_conf_s = now_mono - self._last_high_conf_chord_time
        is_prolonged_zigzag = time_since_high_conf_s > self._coherence_zigzag_timeout_s

        # Diagnóstico de coherencia
        coherence_data = {
            "conflict_active": conflict_active,
            "gps_mag_conflict": self._gps_mag_conflict,
            "gyro_mag_turn_conflict": self._gyro_mag_turn_conflict,
            "consecutive_conflicts": self._consecutive_gps_mag_conflicts,
            "consecutive_agreements": self._consecutive_gps_mag_agreements,
            "last_discrepancy_deg": round(self._last_discrepancy_deg, 2),
            "last_rot_err_deg": round(self._last_rot_err_deg, 2),
            "time_since_high_conf_s": round(time_since_high_conf_s, 1),
            "is_prolonged_zigzag": is_prolonged_zigzag,
        }
        coherence_msg = String()
        coherence_msg.data = json.dumps(coherence_data)
        self.mag_coherence_diag_pub.publish(coherence_msg)

        # Comprobar frescura del diagnóstico de compás (< 3s)
        mag_fresh = (now_mono - self._last_mag_diag_time) < 3.0
        effective_mag_trusted = (self._mag_trusted and not conflict_active) if mag_fresh else False

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
            "mag_confidence_score": round(self._mag_confidence_score if not conflict_active else 0.0, 3),
            "mag_trusted": effective_mag_trusted,
            "gps_course_confidence_score": round(self._gps_course_confidence_score, 3),
            "gps_course_trusted": self._gps_course_trusted,
            "active_anchor": active_anchor,
            "anchor_available": any_anchor_active,
            "gps_mag_conflict": conflict_active,
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
