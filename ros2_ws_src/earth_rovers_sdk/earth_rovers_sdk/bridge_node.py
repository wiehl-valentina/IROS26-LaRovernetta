"""ROS2 bridge node for the Earth Rovers SDK.

This module is adapted from `mini_plus_localization/scripts/earth_rover_bridge.py`
and exposed as a `console_scripts` entrypoint so it can be run with
`ros2 run earth_rovers_sdk earth_rover_bridge` after `colcon build`.

Changes vs. the original version (see chat for details):
  - Compass/magnetometer heading is converted from the "0=North, clockwise"
    convention to the ROS/REP-103 ENU yaw convention (0=East, counter-clockwise)
    before it's used anywhere (IMU orientation, odom orientation, dead-reckoning).
  - IMU / Odometry / GPS publishers use RELIABLE QoS to match robot_localization's
    default subscriber QoS (BEST_EFFORT publisher + RELIABLE subscriber = no data
    delivered at all, which silently starves the EKF).
  - GPS topic renamed to /gps/fix, the default navsat_transform_node expects.
"""

import json
import math
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import rclpy
import requests
import pygeomag
import websocket
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, Image, Imu, NavSatFix, NavSatStatus
from std_msgs.msg import Bool, Float32, String

CONTROL_RATE_HZ = 10.0
CMD_VEL_TIMEOUT_S = 0.5
CONTROL_HTTP_TIMEOUT_S = 1.0
GRAVITY_M_S2 = 9.80665

# ==============================================================================
# TENSOR DE COVARIANZAS (Recalibración IROS 2026)
# ==============================================================================
# ODOMETRÍA DE RUEDAS (Cinemática)
# Penalizamos la velocidad lineal Y (vy) porque el rover no debería patinar 
# lateralmente. Confiamos moderadamente en vx (avance) pero asumimos deslizamiento.
ODOM_POSE_COVARIANCE = [
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.1,
]
ODOM_TWIST_COVARIANCE = [
    0.5,  0.0,  0.0,  0.0,  0.0,  0.0, # vx: deslizamiento longitudinal moderado
    0.0,  2.0,  0.0,  0.0,  0.0,  0.0, # vy: penalización por resbalón lateral
    0.0,  0.0, 10.0,  0.0,  0.0,  0.0, # vz: ignorado (rover terrestre)
    0.0,  0.0,  0.0, 10.0,  0.0,  0.0, # roll rate: ruidoso, se ignora
    0.0,  0.0,  0.0,  0.0, 10.0,  0.0, # pitch rate: ruidoso, se ignora
    0.0,  0.0,  0.0,  0.0,  0.0,  0.2, # yaw rate: confiable pero sujeto a derrapes
]

# IMU MPU-6050 (Fusión Inercial)
# Covarianza de orientación calibrada (Brief 3 / Tramo 3.1 & Brief 13): 0.025 rad^2 (sigma = 9.07 deg = 0.158 rad).
# NOTA TÉCNICA: 0.025 rad^2 es adecuado para terreno nominal/plano pero sigue siendo optimista
# en pendientes mientras no haya tilt compensation activa en el cálculo del compás del SDK,
# ya que en pendientes de 10°-18° el error de proyección magnética puede alcanzar hasta 28° de desvío.
IMU_ORIENTATION_COVARIANCE = [
    0.025, 0.0,   0.0,
    0.0,   0.025, 0.0,
    0.0,   0.0,   0.025, # Magnetómetro absoluto + brújula SDK
]
IMU_ANGULAR_VELOCITY_COVARIANCE = [
    0.01, 0.0,  0.0,
    0.0,  0.01, 0.0,
    0.0,  0.0,  0.05, # Yaw rate: penalizado para mitigar el retardo de fase de red
]
IMU_LINEAR_ACCELERATION_COVARIANCE = [
    0.05, 0.0,  0.0,
    0.0,  0.05, 0.0,
    0.0,  0.0,  0.1,  # Z-accel: absorbe los impactos mecánicos contra el terreno
]

# SISTEMA GNSS (Posicionamiento Global)
# Ajustado a 1 metro de precisión autónoma (Varianza = 1.0^2 = 1.0)
GPS_POSITION_COVARIANCE = [
    1.0,  0.0,  0.0,
    0.0,  1.0,  0.0,
    0.0,  0.0, 100.0, # Altitud sigue penalizada masivamente
]
# ==============================================================================

class EarthRoverBridge(Node):
    def __init__(self):
        super().__init__("earth_rover_bridge")
        self.declare_parameter("sdk_url", "http://localhost:8000")
        self.declare_parameter("feed_fps", 15)
        self.sdk_url = self.get_parameter("sdk_url").value.rstrip("/")
        self.feed_fps = int(self.get_parameter("feed_fps").value)

        self.bridge = CvBridge()

        # Camera: best-effort is fine and desirable here, we don't want the
        # feed thread blocked waiting for slow consumers.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # IMU / Odom / GPS feed robot_localization, whose subscriptions default
        # to RELIABLE. A BEST_EFFORT publisher + RELIABLE subscriber pair is an
        # incompatible QoS combination in ROS2 -- messages get silently dropped
        # at the DDS layer and the EKF never receives anything, even though the
        # topics look "connected". Use RELIABLE here to match.
        filter_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        command_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)

        self.image_pub = self.create_publisher(
            Image, "earth_rover/front/image_raw", image_qos
        )
        # navsat_transform_node's default input topic is /gps/fix.
        # If your launch file remaps this differently, adjust the topic name
        # here (or add a remap in the launch file) so they match.
        self.gps_pub = self.create_publisher(NavSatFix, "/gps/fix", filter_qos)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", filter_qos)
        self.odom_pub = self.create_publisher(Odometry, "/wheel_odom", filter_qos)
        self.battery_pub = self.create_publisher(
            BatteryState, "earth_rover/battery", image_qos
        )
        # Heading crudo de la brújula (sin filtrar). Tópico exclusivo de diagnóstico/debug.
        # Los nodos de navegación (gps_waypoint_controller, bev_planner_node) consumen
        # exclusivamente el heading fusionado por el EKF en 'earth_rover/heading'.
        self.heading_pub = self.create_publisher(
            Float32, "earth_rover/heading_raw", image_qos
        )
        self.heading_tilt_comp_pub = self.create_publisher(
            Float32, "earth_rover/heading_tilt_comp", image_qos
        )
        self.tilt_gate_diag_pub = self.create_publisher(
            String, "earth_rover/tilt_gate_diag", image_qos
        )
        self.mag_gate_diag_pub = self.create_publisher(
            String, "earth_rover/mag_gate_diag", image_qos
        )
        self.vibration_pub = self.create_publisher(
            Float32, "earth_rover/vibration", image_qos
        )

        self.declare_parameter("publish_bridge_debug", True)
        self.publish_bridge_debug = bool(self.get_parameter("publish_bridge_debug").value)
        self.bridge_debug_pub = self.create_publisher(
            String, "earth_rover/bridge_debug", image_qos
        )

        self.declare_parameter("odom_pose_covariance", ODOM_POSE_COVARIANCE)
        self.declare_parameter("odom_twist_covariance", ODOM_TWIST_COVARIANCE)
        self.declare_parameter(
            "imu_orientation_covariance", IMU_ORIENTATION_COVARIANCE
        )
        self.declare_parameter(
            "imu_angular_velocity_covariance", IMU_ANGULAR_VELOCITY_COVARIANCE
        )
        self.declare_parameter(
            "imu_linear_acceleration_covariance", IMU_LINEAR_ACCELERATION_COVARIANCE
        )
        self.declare_parameter("gps_position_covariance", GPS_POSITION_COVARIANCE)
        # Set this to your local magnetic declination (radians, positive = East)
        # ONLY if you'd rather correct it here instead of in navsat_transform's
        # `magnetic_declination_radians` param (that's the more idiomatic place,
        # this param is provided as a convenience / fallback).
        self.declare_parameter("magnetic_declination_radians", 0.0)

        # Parámetros del Filtro Complementario Roll/Pitch e Inercial (Brief 5 / E.1 & Brief 15 / O.4)
        self.declare_parameter("tilt_filter_alpha", 0.20)
        # Factor de escala de acelerómetro: 1.0 (actualizado 2026-09-08 al confirmarse que el
        # SDK entrega telemetría ya normalizada en ~1.0g; el factor previo de 0.5098
        # correspondía a un comportamiento obsoleto del SDK donde la norma medía ~1.96g).
        self.declare_parameter("accel_scale_factor", 1.0)
        self.declare_parameter("gyro_bias_x", 0.0)
        self.declare_parameter("gyro_bias_y", 0.0)
        # Bias de giróscopo Z: +1.027725 deg/s = +0.0179372 rad/s (medido en Test A, reposo 60s, 2026-09-10)
        self.declare_parameter("gyro_bias_z", 0.0179372)
        self.declare_parameter("accel_bias_x", 0.0)
        self.declare_parameter("accel_bias_y", 0.0)
        self.declare_parameter("accel_bias_z", 0.0)
        self.declare_parameter("gyro_bias_file", "")
        self.declare_parameter("accel_bias_file", "")
        self.declare_parameter("gyro_drift_noise_density", 0.005)

        # Parámetros del Detector de Saturación Magnética y Gating de Brújula (Fase 1)
        # Origen de valores:
        # - mag_norm_reference: 3330.0 raw counts (medido en Test A en reposo, 2026-09-10; media = 3331.3 counts)
        # - mag_gross_factor_min / mag_gross_factor_max: [0.3, 2.5] factores relativos a mag_norm_reference
        #   para saturación bruta (física: el campo terrestre + hard-iron jamás excede ~1.5 - 2.0x ref;
        #   en sitio saturado con interferencia ferromagnética se midió norma = 12221.7 counts = 3.67x ref).
        #   Expresar los límites relativos a la referencia garantiza portabilidad universal entre sitios y sensores.
        # - mag_tol_inner_factor / mag_tol_outer_factor: [0.20, 0.50] banda de tolerancia relativa suave (20% y 50%)
        # - mag_tilt_level_deg / slope_deg: [6.0, 18.0] umbrales de inclinación (roll/pitch).
        #   Cerca de nivel (<6°), la norma es confiable. En pendiente (>18°), el sesgo hard-iron altera la norma
        #   y se reduce su peso.
        # - mag_tilt_weight_min: 0.15 (peso residual de la norma en pendientes pronunciadas)
        # - mag_stuck_std_thresh: 5.0 raw counts (en giro motorizado saturado, std fue exactamente 0.00 counts)
        # - mag_turn_rate_thresh: 0.052359877 rad/s (3.0 deg/s; umbral para exigir variación magnética en giro)
        # - mag_calib_samples_needed: 10 muestras consecutivas a nivel y reposo para calibrar mag_norm_reference
        # - mag_untrusted_yaw_covariance: 1.0e6 rad^2 (penalización para descarte de facto en robot_localization)
        self.declare_parameter("mag_norm_reference", 3330.0)
        self.declare_parameter("mag_gross_factor_min", 0.3)
        self.declare_parameter("mag_gross_factor_max", 2.5)
        self.declare_parameter("mag_tol_inner_factor", 0.20)
        self.declare_parameter("mag_tol_outer_factor", 0.50)
        self.declare_parameter("mag_tilt_level_deg", 6.0)
        self.declare_parameter("mag_tilt_slope_deg", 18.0)
        self.declare_parameter("mag_tilt_weight_min", 0.15)
        self.declare_parameter("mag_stuck_std_thresh", 5.0)
        self.declare_parameter("mag_turn_rate_thresh", 0.052359877)
        self.declare_parameter("mag_calib_samples_needed", 10)
        self.declare_parameter("mag_calibration_file", "")
        self.declare_parameter("mag_untrusted_yaw_covariance", 1.0e6)

        # Parámetros de Guarda contra GPS malo (Fase 1)
        self.declare_parameter("gps_guard_enabled", True)
        self.declare_parameter("gps_v_max_phys_m_s", 1.111)
        self.declare_parameter("gps_jump_noise_margin_m", 1.5)
        self.declare_parameter("gps_bad_fix_consecutive_thresh", 3)
        self.declare_parameter("gps_min_fix_interval_s", 0.8)

        self._odom_pose_covariance = self.get_parameter(
            "odom_pose_covariance"
        ).value
        self._odom_twist_covariance = self.get_parameter(
            "odom_twist_covariance"
        ).value
        self._imu_orientation_covariance = self.get_parameter(
            "imu_orientation_covariance"
        ).value
        self._imu_angular_velocity_covariance = self.get_parameter(
            "imu_angular_velocity_covariance"
        ).value
        self._imu_linear_acceleration_covariance = self.get_parameter(
            "imu_linear_acceleration_covariance"
        ).value
        self._gps_position_covariance = self.get_parameter(
            "gps_position_covariance"
        ).value
        self._magnetic_declination_radians = float(
            self.get_parameter("magnetic_declination_radians").value
        )
        self._tilt_filter_alpha = float(
            self.get_parameter("tilt_filter_alpha").value
        )
        self._accel_scale_factor = float(
            self.get_parameter("accel_scale_factor").value
        )
        self._gyro_bias_x = float(self.get_parameter("gyro_bias_x").value)
        self._gyro_bias_y = float(self.get_parameter("gyro_bias_y").value)
        self._gyro_bias_z = float(self.get_parameter("gyro_bias_z").value)
        self._accel_bias_x = float(self.get_parameter("accel_bias_x").value)
        self._accel_bias_y = float(self.get_parameter("accel_bias_y").value)
        self._accel_bias_z = float(self.get_parameter("accel_bias_z").value)
        self._gyro_drift_noise_density = float(
            self.get_parameter("gyro_drift_noise_density").value
        )
        self._mag_norm_reference = float(
            self.get_parameter("mag_norm_reference").value
        )
        self._mag_gross_factor_min = float(
            self.get_parameter("mag_gross_factor_min").value
        )
        self._mag_gross_factor_max = float(
            self.get_parameter("mag_gross_factor_max").value
        )
        self._mag_tol_inner_factor = float(
            self.get_parameter("mag_tol_inner_factor").value
        )
        self._mag_tol_outer_factor = float(
            self.get_parameter("mag_tol_outer_factor").value
        )
        self._mag_tilt_level_deg = float(
            self.get_parameter("mag_tilt_level_deg").value
        )
        self._mag_tilt_slope_deg = float(
            self.get_parameter("mag_tilt_slope_deg").value
        )
        self._mag_tilt_weight_min = float(
            self.get_parameter("mag_tilt_weight_min").value
        )
        self._mag_stuck_std_thresh = float(
            self.get_parameter("mag_stuck_std_thresh").value
        )
        self._mag_turn_rate_thresh = float(
            self.get_parameter("mag_turn_rate_thresh").value
        )
        self._mag_calib_samples_needed = int(
            self.get_parameter("mag_calib_samples_needed").value
        )
        self._mag_untrusted_yaw_covariance = float(
            self.get_parameter("mag_untrusted_yaw_covariance").value
        )

        self._gps_guard_enabled = bool(self.get_parameter("gps_guard_enabled").value)
        self._gps_v_max_phys_m_s = float(self.get_parameter("gps_v_max_phys_m_s").value)
        self._gps_jump_noise_margin_m = float(self.get_parameter("gps_jump_noise_margin_m").value)
        self._gps_bad_fix_consecutive_thresh = int(self.get_parameter("gps_bad_fix_consecutive_thresh").value)
        self._gps_min_fix_interval_s = float(self.get_parameter("gps_min_fix_interval_s").value)

        # Estado de guarda contra GPS malo (Fase 1)
        self._last_published_gps_stamp = None
        self._last_published_gps_lat = None
        self._last_published_gps_lon = None
        self._consecutive_bad_gps = 0
        self._gps_guard_level = 1

        self._latest_cmd = None
        self._last_cmd_at = 0.0
        self._last_cmd_rx_ros_sec = None
        self._stopped = True
        self._cmd_lock = threading.Lock()

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._last_odom_time = None
        self._last_yaw = None

        # Estado del Filtro Complementario (E.1)
        self._filtered_roll = 0.0
        self._filtered_pitch = 0.0
        self._tilt_uncertainty_rad = math.radians(1.0)
        self._last_accel_valid_time = None
        self._last_tilt_update_time = None
        self._tilt_gate_history = deque(maxlen=30)

        # Estado del Detector de Anomalía Magnética y Compuerta Continua (Fase 1)
        self._mag_history = deque(maxlen=20)
        self._mag_calibrated = True # Forzamos True para no calibrar offline con el sensor
        self._mag_calib_buffer = []
        self._mag_confidence_score = 1.0
        self._last_mag_diag = None

        # Cargar archivos de calibración JSON si existen (Brief 15 / O.4 y Fase 1)
        self._load_inertial_calibration_files()

        # Integración WMM/IGRF dinámica
        self._geo_model = pygeomag.GeoMag()
        self.declare_parameter("mag_dynamic_update_dist_m", 1000.0)
        self._mag_dynamic_update_dist_m = float(self.get_parameter("mag_dynamic_update_dist_m").value)
        self._last_calc_lat = None
        self._last_calc_lon = None
        # TODO: Refinar offset de Hard-Iron.
        # El valor de +2278.1 fue calibrado con UN SOLO punto (Gaborone) y es una 
        # aproximación aditiva en norma (1D). No captura que la relación vectorial 
        # entre el sesgo del chasis y el campo terrestre cambia con la inclinación 
        # geomagnética del sitio (que varía por latitud, independientemente del tilt 
        # del rover). Es candidato a refinar con más puntos de calibración en sitios 
        # con campo geomagnético distinto, cuando haya rover disponible.
        self._hard_iron_offset_counts = 2278.1

        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, command_qos)
        self._gps_mag_conflict = False
        self.create_subscription(
            Bool, "/earth_rover/gps_mag_conflict", self._on_gps_mag_conflict, filter_qos
        )

        self._session = requests.Session()
        self._running = True
        self._stop_event = threading.Event()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        threading.Thread(target=self._feed_loop, daemon=True).start()
        threading.Thread(target=self._telemetry_loop, daemon=True).start()

        self.get_logger().info(f"Bridging Earth Rovers SDK at {self.sdk_url}")

    def _find_calibration_file(self, filename: str) -> str | None:
        """Busca un archivo de calibración sin depender de nombres de usuario ni rutas fijas."""
        import os
        # 1. Intentar mediante el share directory del paquete instalado
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory("earth_rovers_sdk")
            cand = os.path.join(share_dir, "config", filename)
            if os.path.isfile(cand):
                return cand
        except Exception:
            pass

        # 2. Búsqueda en árbol fuente relativo al script y directorio de trabajo
        this_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(this_dir, "..", "config", filename),
            os.path.join(this_dir, "..", "..", "config", filename),
            os.path.join(this_dir, "..", "..", "..", "config", filename),
            os.path.join(os.getcwd(), "config", filename),
            os.path.join(os.getcwd(), "ros2_ws_src", "earth_rovers_sdk", "config", filename),
            os.path.join(os.getcwd(), "earth_rovers_sdk", "config", filename),
            f"/root/ros2_ws/config/{filename}",
            f"/root/ros2_ws/src/earth_rovers_sdk/config/{filename}",
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
        return None

    def _load_inertial_calibration_files(self):
        """Carga archivos de calibración de sesgo gyro_bias.json, accel_bias.json y mag_calibration.json."""
        import os
        # 1. Calibración de Giróscopo
        gyro_file = str(self.get_parameter("gyro_bias_file").value).strip()
        if not gyro_file or not os.path.isfile(gyro_file):
            gyro_file = self._find_calibration_file("gyro_bias.json")

        if gyro_file and os.path.isfile(gyro_file):
            try:
                with open(gyro_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._gyro_bias_x = float(data.get("gyro_bias_x", self._gyro_bias_x))
                self._gyro_bias_y = float(data.get("gyro_bias_y", self._gyro_bias_y))
                self._gyro_bias_z = float(data.get("gyro_bias_z", self._gyro_bias_z))
                self.get_logger().info(
                    f"Calibración de Giróscopo CARGADA desde '{gyro_file}': "
                    f"bias=({self._gyro_bias_x:.6f}, {self._gyro_bias_y:.6f}, {self._gyro_bias_z:.6f}) rad/s"
                )
            except Exception as e:
                self.get_logger().error(f"Error leyendo '{gyro_file}': {e}. Usando parámetros ROS.")
        else:
            self.get_logger().info(
                f"Sin archivo gyro_bias.json; usando bias inercial: "
                f"({self._gyro_bias_x:.6f}, {self._gyro_bias_y:.6f}, {self._gyro_bias_z:.6f}) rad/s"
            )

        # 2. Calibración de Acelerómetro
        accel_file = str(self.get_parameter("accel_bias_file").value).strip()
        if not accel_file or not os.path.isfile(accel_file):
            accel_file = self._find_calibration_file("accel_bias.json")

        if accel_file and os.path.isfile(accel_file):
            try:
                with open(accel_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._accel_bias_x = float(data.get("accel_bias_x", self._accel_bias_x))
                self._accel_bias_y = float(data.get("accel_bias_y", self._accel_bias_y))
                self._accel_bias_z = float(data.get("accel_bias_z", self._accel_bias_z))
                self.get_logger().info(
                    f"Calibración de Acelerómetro CARGADA desde '{accel_file}': "
                    f"bias=({self._accel_bias_x:.6f}, {self._accel_bias_y:.6f}, {self._accel_bias_z:.6f}) g"
                )
            except Exception as e:
                self.get_logger().error(f"Error leyendo '{accel_file}': {e}. Usando parámetros ROS.")
        else:
            self.get_logger().info(
                f"Sin archivo accel_bias.json; usando bias inercial: "
                f"({self._accel_bias_x:.6f}, {self._accel_bias_y:.6f}, {self._accel_bias_z:.6f}) g"
            )

        # 3. Calibración de Magnetómetro (Fase 1)
        mag_file = str(self.get_parameter("mag_calibration_file").value).strip()
        if not mag_file or not os.path.isfile(mag_file):
            mag_file = self._find_calibration_file("mag_calibration.json")

        if mag_file and os.path.isfile(mag_file):
            try:
                with open(mag_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._mag_norm_reference = float(data.get("mag_norm_reference", self._mag_norm_reference))
                self._mag_gross_factor_min = float(data.get("mag_gross_factor_min", self._mag_gross_factor_min))
                self._mag_gross_factor_max = float(data.get("mag_gross_factor_max", self._mag_gross_factor_max))
                self._mag_tol_inner_factor = float(data.get("mag_tol_inner_factor", self._mag_tol_inner_factor))
                self._mag_tol_outer_factor = float(data.get("mag_tol_outer_factor", self._mag_tol_outer_factor))
                self._mag_tilt_level_deg = float(data.get("mag_tilt_level_deg", self._mag_tilt_level_deg))
                self._mag_tilt_slope_deg = float(data.get("mag_tilt_slope_deg", self._mag_tilt_slope_deg))
                self._mag_tilt_weight_min = float(data.get("mag_tilt_weight_min", self._mag_tilt_weight_min))
                self._mag_stuck_std_thresh = float(data.get("mag_stuck_std_thresh", self._mag_stuck_std_thresh))
                self._mag_calibrated = bool(data.get("calibrated", self._mag_calibrated))
                self.get_logger().info(
                    f"Calibración de Magnetómetro CARGADA desde '{mag_file}': "
                    f"norm_ref={self._mag_norm_reference:.1f}, "
                    f"gross_factors=[{self._mag_gross_factor_min}, {self._mag_gross_factor_max}], "
                    f"calibrated={self._mag_calibrated}"
                )
            except Exception as e:
                self.get_logger().error(f"Error leyendo '{mag_file}': {e}. Usando parámetros ROS.")
        else:
            self.get_logger().info(
                f"Sin archivo mag_calibration.json; usando referencia inicial: {self._mag_norm_reference:.1f} counts"
            )

    def _on_cmd_vel(self, msg: Twist):
        with self._cmd_lock:
            self._latest_cmd = {
                "linear": max(-1.0, min(1.0, msg.linear.x)),
                "angular": max(-1.0, min(1.0, msg.angular.z)),
            }
            self._last_cmd_at = time.monotonic()
            self._last_cmd_rx_ros_sec = self.get_clock().now().nanoseconds / 1e9
            self._stopped = False

    def _on_gps_mag_conflict(self, msg: Bool):
        self._gps_mag_conflict = bool(msg.data)

    def _control_tick(self):
        with self._cmd_lock:
            quiet = time.monotonic() - self._last_cmd_at > CMD_VEL_TIMEOUT_S
            if self._latest_cmd is None or (quiet and self._stopped):
                return
            command = {"linear": 0, "angular": 0} if quiet else dict(self._latest_cmd)
            last_cmd_at = self._last_cmd_at
            last_rx_ros_sec = self._last_cmd_rx_ros_sec

        send_mono = time.monotonic()
        send_ros_sec = self.get_clock().now().nanoseconds / 1e9
        status_code = None
        error_str = None

        try:
            response = self._session.post(
                f"{self.sdk_url}/control",
                json={"command": command},
                timeout=CONTROL_HTTP_TIMEOUT_S,
            )
            status_code = response.status_code
            response.raise_for_status()

            if quiet:
                with self._cmd_lock:
                    if (
                        self._last_cmd_at == last_cmd_at
                        and time.monotonic() - self._last_cmd_at > CMD_VEL_TIMEOUT_S
                    ):
                        self._stopped = True
        except requests.RequestException as e:
            error_str = str(e)
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", status_code)
            self.get_logger().warning(f"/control failed: {e}", throttle_duration_sec=5)
        finally:
            resp_mono = time.monotonic()
            resp_ros_sec = self.get_clock().now().nanoseconds / 1e9
            roundtrip_ms = (resp_mono - send_mono) * 1000.0

            if self.publish_bridge_debug:
                dbg_payload = {
                    "type": "control_actuation",
                    "cmd_rx_ros_sec": last_rx_ros_sec,
                    "http_send_ros_sec": send_ros_sec,
                    "http_resp_ros_sec": resp_ros_sec,
                    "roundtrip_ms": roundtrip_ms,
                    "status_code": status_code,
                    "command": command,
                    "error": error_str,
                }
                dbg_msg = String()
                dbg_msg.data = json.dumps(dbg_payload)
                self.bridge_debug_pub.publish(dbg_msg)

    def _control_loop(self):
        interval = 1.0 / CONTROL_RATE_HZ
        deadline = time.monotonic()
        while self._running and rclpy.ok():
            self._control_tick()
            deadline += interval
            wait = max(0.0, deadline - time.monotonic())
            if self._stop_event.wait(wait):
                break
            if time.monotonic() - deadline > interval:
                deadline = time.monotonic()

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _compass_heading_to_enu_yaw(self, heading_deg: float) -> float:
        """Convert a compass heading (degrees, 0=North, clockwise-positive) into
        a ROS/REP-103 ENU yaw (radians, 0=East, counter-clockwise-positive).

        This is the conversion robot_localization / navsat_transform assume for
        the IMU orientation they fuse. Getting this wrong flips or offsets the
        heading used to rotate GPS readings into the odom/map frame.
        """
        heading_rad = math.radians(heading_deg)
        yaw = (math.pi / 2.0) - heading_rad
        yaw += self._magnetic_declination_radians
        return self._normalize_angle(yaw)

    def _evaluate_magnetic_gate(
        self,
        mx: float,
        my: float,
        mz: float,
        omega_z: float,
        accel_gate_open: bool,
    ) -> tuple[float, float, dict]:
        """Evalúa la confiabilidad del magnetómetro aplicando gating continuo.

        NOTA FÍSICA Y LIMITACIÓN DE DISEÑO:
        El campo magnético medido por el rover NO es invariante a la orientación
        cuando existe sesgo magnético propio de la estructura (hard-iron onboard).
        La norma escalar medible ||M|| varía con el ángulo entre el vector de campo
        terrestre y dicho sesgo conforme el rover cambia de inclinación (roll/pitch).
        Por ende:
          1. La calibración de `mag_norm_reference` se ejecuta EXCLUSIVAMENTE cuando
             el rover se encuentra cerca de nivel (|roll|, |pitch| < mag_tilt_level_deg)
             y en reposo/aceleración gravitatoria limpia (accel_gate_open). Si arranca
             inclinado, no se calibra la referencia para evitar sesgos sistemáticos.
          2. La detección de anomalía por norma absoluta es confiable cerca de nivel
             y SE DEGRADA EN PENDIENTE POR DISEÑO (no por bug). En pendientes
             significativas, se reduce el peso de la discrepancia de norma (w_norm)
             y el detector se apoya primordialmente en la varianza temporal dinámica
             (std(M) ≈ 0 ante giro angular medido por giróscopo), la cual es estrictamente
             invariante a la inclinación del terreno.

        Retorna:
          (confidence_score, effective_yaw_covariance, diag_dict)
        """
        now_mono = time.monotonic()
        norm_m = math.sqrt(mx**2 + my**2 + mz**2)
        self._mag_history.append((now_mono, mx, my, mz, norm_m, omega_z))

        roll_deg = abs(math.degrees(self._filtered_roll))
        pitch_deg = abs(math.degrees(self._filtered_pitch))
        max_tilt_deg = max(roll_deg, pitch_deg)
        is_level = max_tilt_deg < self._mag_tilt_level_deg
        is_stationary = abs(omega_z) < math.radians(2.0)

        # 1. Calibración online de mag_norm_reference a nivel
        if not self._mag_calibrated:
            if is_level and accel_gate_open and is_stationary:
                self._mag_calib_buffer.append(norm_m)
                if len(self._mag_calib_buffer) >= self._mag_calib_samples_needed:
                    mean_calib = sum(self._mag_calib_buffer) / len(self._mag_calib_buffer)
                    gross_min_calib = self._mag_gross_factor_min * self._mag_norm_reference
                    gross_max_calib = self._mag_gross_factor_max * self._mag_norm_reference
                    if gross_min_calib <= mean_calib <= gross_max_calib:
                        self._mag_norm_reference = mean_calib
                        self.get_logger().info(
                            f"[MAG_GATE] Calibración de mag_norm_reference completada a nivel: "
                            f"{self._mag_norm_reference:.1f} counts (tilt: {max_tilt_deg:.2f}°)"
                        )
                    else:
                        self.get_logger().warn(
                            f"[MAG_GATE] Norma a nivel ({mean_calib:.1f} counts) fuera de rango relativo admisible "
                            f"[{gross_min_calib:.1f}, {gross_max_calib:.1f}]. Posible arranque en sitio saturado. "
                            f"Manteniendo referencia nominal {self._mag_norm_reference:.1f} counts."
                        )
                    self._mag_calibrated = True
            else:
                if not is_level:
                    self.get_logger().info(
                        f"[MAG_GATE] Rover inclinado (roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}° > {self._mag_tilt_level_deg}°). "
                        "Esperando condición nivelada para calibrar mag_norm_reference.",
                        throttle_duration_sec=10.0,
                    )

        # 2. Umbrales relativos a mag_norm_reference (garantizan portabilidad universal)
        gross_min = self._mag_gross_factor_min * self._mag_norm_reference
        gross_max = self._mag_gross_factor_max * self._mag_norm_reference
        tol_inner = self._mag_tol_inner_factor * self._mag_norm_reference
        tol_outer = self._mag_tol_outer_factor * self._mag_norm_reference

        # Si la norma excede gross_max (ej. 2.5x ref) o cae bajo gross_min (ej. 0.3x ref),
        # estamos ante SATURACIÓN FERROMAGNÉTICA BRUTA o fallo de sensor.
        # Esto NUNCA puede ser atribuido al tilt (el campo terrestre más hard-iron jamás supera ~1.5 - 2.0x ref).
        is_gross_saturation = (norm_m < gross_min or norm_m > gross_max)

        delta_norm = abs(norm_m - self._mag_norm_reference)
        if is_gross_saturation:
            s_norm = 0.0
        elif delta_norm <= tol_inner:
            s_norm = 1.0
        elif delta_norm >= tol_outer:
            s_norm = 0.0
        else:
            s_norm = 1.0 - (delta_norm - tol_inner) / (tol_outer - tol_inner)

        # 3. Ponderación por inclinación actual (w_norm)
        # En pendientes (>18°), el sesgo hard-iron altera la norma medida dentro del rango admisible;
        # bajamos su peso a mag_tilt_weight_min (0.15) para no descartar el compás por variaciones de pendiente.
        # PERO si hay saturación bruta (norm_m fuera de [gross_min, gross_max]), w_norm es 1.0 y s_norm_weighted es 0.0 forzoso.
        if is_gross_saturation:
            w_norm = 1.0
            s_norm_weighted = 0.0
        else:
            if max_tilt_deg <= self._mag_tilt_level_deg:
                w_norm = 1.0
            elif max_tilt_deg >= self._mag_tilt_slope_deg:
                w_norm = self._mag_tilt_weight_min
            else:
                ratio = (max_tilt_deg - self._mag_tilt_level_deg) / (self._mag_tilt_slope_deg - self._mag_tilt_level_deg)
                w_norm = 1.0 - ratio * (1.0 - self._mag_tilt_weight_min)

            s_norm_weighted = w_norm * s_norm + (1.0 - w_norm) * 1.0

        # 4. Componente de varianza temporal / sensor congelado (S_dynamic)
        s_dynamic = 1.0
        mag_std = 0.0
        avg_omega_z = 0.0
        window = list(self._mag_history)[-10:]
        if len(window) >= 5:
            xs = [h[1] for h in window]
            ys = [h[2] for h in window]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
            std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / len(ys))
            mag_std = math.sqrt(std_x ** 2 + std_y ** 2)

            avg_omega_z = sum(abs(h[5]) for h in window) / len(window)
            if avg_omega_z > self._mag_turn_rate_thresh:
                if mag_std < self._mag_stuck_std_thresh:
                    s_dynamic = max(0.0, min(1.0, mag_std / self._mag_stuck_std_thresh))
                else:
                    s_dynamic = 1.0

        # 5. Fusión de score continuo y filtro IIR asimétrico
        s_raw = s_dynamic * s_norm_weighted
        alpha_filter = 0.8 if s_raw < self._mag_confidence_score else 0.3
        self._mag_confidence_score = (1.0 - alpha_filter) * self._mag_confidence_score + alpha_filter * s_raw
        self._mag_confidence_score = max(0.0, min(1.0, self._mag_confidence_score))

        # 6. Mapeo continuo de covarianza para EKF
        nominal_cov = self._imu_orientation_covariance[8]
        cov_yaw = nominal_cov + ((1.0 - self._mag_confidence_score) ** 2) * self._mag_untrusted_yaw_covariance

        is_trusted = (self._mag_confidence_score > 0.6) and (not is_gross_saturation)

        diag_payload = {
            "mag_norm": round(norm_m, 2),
            "mag_norm_ref": round(self._mag_norm_reference, 2),
            "gross_min": round(gross_min, 2),
            "gross_max": round(gross_max, 2),
            "delta_norm": round(delta_norm, 2),
            "score_norm": round(s_norm, 4),
            "tilt_deg": round(max_tilt_deg, 2),
            "tilt_weight": round(w_norm, 4),
            "score_norm_weighted": round(s_norm_weighted, 4),
            "mag_std": round(mag_std, 4),
            "avg_omega_z_dps": round(math.degrees(avg_omega_z), 2),
            "score_dynamic": round(s_dynamic, 4),
            "confidence_score": round(self._mag_confidence_score, 4),
            "cov_yaw": round(cov_yaw, 2),
            "trusted": is_trusted,
            "calibrated": self._mag_calibrated,
        }
        self._last_mag_diag = diag_payload

        d_msg = String()
        d_msg.data = json.dumps(diag_payload)
        self.mag_gate_diag_pub.publish(d_msg)

        return self._mag_confidence_score, cov_yaw, diag_payload

    def _feed_loop(self):
        url = f"{self.sdk_url}/feed?view=front&fps={self.feed_fps}"
        while self._running and rclpy.ok():
            capture = cv2.VideoCapture(url)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                self.get_logger().warning(
                    "/feed not available, retrying in 3s", throttle_duration_sec=10
                )
                time.sleep(3)
                continue
            self.get_logger().info("Connected to /feed")
            while self._running and rclpy.ok():
                ok, frame = capture.read()
                if not ok:
                    break
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "earth_rover_front_camera"
                self.image_pub.publish(msg)
            capture.release()
            time.sleep(1)

    def _telemetry_loop(self):
        ws_url = self.sdk_url.replace("http", "ws", 1) + "/ws/data"
        dump_count = 0
        while self._running and rclpy.ok():
            ws = None
            try:
                ws = websocket.create_connection(ws_url, timeout=10)
                self.get_logger().info("Connected to /ws/data")
                while self._running and rclpy.ok():
                    msg = json.loads(ws.recv())
                    if msg.get("type") in ("snapshot", "telemetry") and msg.get("data"):
                        if dump_count < 5:
                            dump_count += 1
                            self.get_logger().info(
                                f"[TELEMETRY_DUMP #{dump_count}]\n{json.dumps(msg['data'], indent=2)}"
                            )
                        self._publish_telemetry(msg["data"])
            except Exception as e:
                self.get_logger().warning(
                    f"/ws/data reconnecting: {e}", throttle_duration_sec=10
                )
                time.sleep(2)
            finally:
                if ws is not None:
                    ws.close()

    def _haversine_dist(self, lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def _update_dynamic_mag_reference(self, lat: float, lon: float):
        if self._last_calc_lat is not None and self._last_calc_lon is not None:
            dist = self._haversine_dist(self._last_calc_lat, self._last_calc_lon, lat, lon)
            if dist < self._mag_dynamic_update_dist_m:
                return
        
        # Calcular fecha decimal actual para el modelo WMM
        try:
            from datetime import timezone
            now_dt = datetime.now(timezone.utc)
        except Exception:
            now_dt = datetime.utcnow()
        year = now_dt.year
        day_of_year = now_dt.timetuple().tm_yday
        is_leap = 1 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 0
        days_in_year = 366 if is_leap else 365
        dec_year = year + (day_of_year / days_in_year)

        try:
            # altitud 0, modelo offline incrustado
            result = self._geo_model.calculate(lat, lon, 0.0, dec_year)
            F_nT = result.f
            F_gauss = F_nT / 100000.0
            # Sensibilidad: 3750 LSB/Gauss (RNG=\u00b18G en QMC5883P)
            dynamic_counts = F_gauss * 3750.0
            
            new_ref = dynamic_counts + self._hard_iron_offset_counts
            
            self.get_logger().info(
                f"[MAG_DYNAMIC] WMM Ref. actualizada: lat={lat:.4f}, lon={lon:.4f} -> "
                f"F_geomag={dynamic_counts:.1f} + HI={self._hard_iron_offset_counts:.1f} "
                f"=> {new_ref:.1f} counts"
            )
            self._mag_norm_reference = new_ref
            self._last_calc_lat = lat
            self._last_calc_lon = lon
        except Exception as e:
            self.get_logger().error(f"Error calculando modelo geomagnético dinámico: {e}")

    def _publish_telemetry(self, data: dict):
        now = self.get_clock().now().to_msg()

        # ---------------------------------------------------------
        # 1. GNSS (GPS) CON COVARIANZA DINÁMICA AVANZADA Y GUARDA (Fase 1)
        # ---------------------------------------------------------
        lat, lng = data.get("latitude"), data.get("longitude")
        if lat is not None and lng is not None:
            lat_f = float(lat)
            lng_f = float(lng)
            gps_ts = float(data.get("gps_timestamp") or data.get("timestamp") or 0.0)

            # Paso 1.1: Deduplicación por timestamp y coordenadas (tasa efectiva 1 Hz)
            is_duplicate = False
            if self._gps_guard_enabled and self._last_published_gps_stamp is not None and gps_ts > 0.0:
                dt_ts = gps_ts - self._last_published_gps_stamp
                if dt_ts <= 0.0 or (lat_f == self._last_published_gps_lat and lng_f == self._last_published_gps_lon and dt_ts < self._gps_min_fix_interval_s):
                    is_duplicate = True

            if not is_duplicate:
                gps = NavSatFix()
                if gps_ts > 0.0:
                    sec = int(gps_ts)
                    nanosec = int((gps_ts - sec) * 1e9)
                    gps.header.stamp.sec = sec
                    gps.header.stamp.nanosec = nanosec
                else:
                    gps.header.stamp = now
                gps.header.frame_id = "earth_rover_gps"

                # 1.1 Extracción de Metadatos del Hardware
                gps_signal = data.get("gps_signal")   # Calidad de señal en porcentaje (%)
                fix_quality = data.get("fix_quality") # Flag NMEA oficial de la placa (0=sin fix, 1=autónomo, 2=DGPS, 4=RTK fijo, 5=RTK float)
                
                try:
                    hdop = float(data.get("hdop", 1.0))
                except (TypeError, ValueError):
                    hdop = 1.0

                # 1.2 Máquina de Estados de Validación de Fix
                is_fix_valid = False
                
                if fix_quality is not None:
                    # Prioridad Absoluta: El hardware reporta si logró resolver la ecuación
                    if int(fix_quality) > 0:
                        is_fix_valid = True
                elif gps_signal is not None:
                    # Respaldo si el hardware no expone fix_quality: señal positiva
                    try:
                        sig = float(gps_signal)
                        if sig > 0.0: 
                            is_fix_valid = True
                    except (TypeError, ValueError):
                        pass

                # Paso 1.2: Detección física de saltos
                is_jump = False
                if self._gps_guard_enabled and self._last_published_gps_lat is not None and self._last_published_gps_lon is not None:
                    dt_jump = gps_ts - self._last_published_gps_stamp if (gps_ts > 0 and self._last_published_gps_stamp) else 1.0
                    if dt_jump <= 0.0:
                        dt_jump = 1.0
                    d_max_allowed = self._gps_v_max_phys_m_s * dt_jump + self._gps_jump_noise_margin_m
                    d_meas = self._haversine_dist(self._last_published_gps_lat, self._last_published_gps_lon, lat_f, lng_f)
                    if d_meas > d_max_allowed:
                        is_jump = True
                        self.get_logger().warning(
                            f"[GPS_GUARD_ROS2] Salto GNSS descartado: dist={d_meas:.2f}m > d_max={d_max_allowed:.2f}m (dt={dt_jump:.2f}s).",
                            throttle_duration_sec=1.0
                        )

                # Paso 1.4: Comportamiento escalonado
                is_bad = is_jump or not is_fix_valid or hdop >= 20.0
                if is_bad:
                    self._consecutive_bad_gps += 1
                    if self._consecutive_bad_gps >= self._gps_bad_fix_consecutive_thresh:
                        if self._gps_guard_level < 2:
                            self._gps_guard_level = 2
                            self.get_logger().warning(
                                f"[GPS_GUARD_ROS2] NIVEL 1 -> NIVEL 2: GPS malo sostenido ({self._consecutive_bad_gps} fixes malos seguidos). "
                                "Forzando EKF a Dead-Reckoning y bloqueando /gps/fix.",
                                throttle_duration_sec=2.0
                            )
                    else:
                        self.get_logger().warning(
                            f"[GPS_GUARD_ROS2] Fix malo/salto aislado descartado ({self._consecutive_bad_gps}/{self._gps_bad_fix_consecutive_thresh}).",
                            throttle_duration_sec=1.0
                        )
                else:
                    if self._gps_guard_level > 1:
                        self.get_logger().info(
                            f"[GPS_GUARD_ROS2] NIVEL {self._gps_guard_level} -> NIVEL 1: Fix GPS válido recuperado ({lat_f:.7f}, {lng_f:.7f})."
                        )
                    self._gps_guard_level = 1
                    self._consecutive_bad_gps = 0
                    self._last_published_gps_lat = lat_f
                    self._last_published_gps_lon = lng_f
                    self._last_published_gps_stamp = gps_ts if gps_ts > 0.0 else time.time()

                    gps.status.status = NavSatStatus.STATUS_FIX
                    self._update_dynamic_mag_reference(lat_f, lng_f)
                    gps.status.service = NavSatStatus.SERVICE_GPS
                    gps.latitude = lat_f
                    gps.longitude = lng_f

                    # 1.3 Escalado Tensorial de Covarianza
                    hdop_factor = hdop ** 2
                    gps.position_covariance = [
                        cov * hdop_factor for cov in self._gps_position_covariance
                    ]
                    gps.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

                    self.gps_pub.publish(gps)

        # ---------------------------------------------------------
        # 2. ORIENTACIÓN MAGNÉTICA (Compass)
        # ---------------------------------------------------------
        orientation = data.get("orientation")
        yaw = None
        if orientation is not None:
            try:
                heading_deg = float(orientation)
            except (TypeError, ValueError):
                heading_deg = None
            if heading_deg is not None:
                heading = Float32()
                heading.data = heading_deg
                self.heading_pub.publish(heading)
                yaw = self._compass_heading_to_enu_yaw(heading_deg)

        # ---------------------------------------------------------
        # 3. BATERÍA Y VIBRACIÓN
        # ---------------------------------------------------------
        battery = data.get("battery")
        if battery is not None:
            batt = BatteryState()
            batt.header.stamp = now
            batt.percentage = float(battery) / 100.0
            batt.present = True
            self.battery_pub.publish(batt)

        vibration = data.get("vibration")
        if vibration is not None:
            try:
                vib_msg = Float32()
                vib_msg.data = float(vibration)
                self.vibration_pub.publish(vib_msg)
            except (TypeError, ValueError):
                pass

        # ---------------------------------------------------------
        # 3b. ESTIMACIÓN ROLL/PITCH: FILTRO COMPLEMENTARIO (Brief 5 / E.1)
        # ---------------------------------------------------------
        accels = data.get("accels") or []
        gyros = data.get("gyros") or []
        mags = data.get("mags") or []
        gate_open = False

        if mags and accels:
            try:
                mx = float(mags[-1][0])
                my = float(mags[-1][1])
                mz = float(mags[-1][2])

                # Velocidad angular promediada y corregida por bias
                num_gyros = len(gyros) if gyros else 1
                avg_gx = sum(float(g[0]) for g in gyros) / num_gyros if gyros else 0.0
                avg_gy = sum(float(g[1]) for g in gyros) / num_gyros if gyros else 0.0
                omega_x = math.radians(avg_gx) - self._gyro_bias_x
                omega_y = math.radians(avg_gy) - self._gyro_bias_y

                now_mono = time.monotonic()
                dt_tilt = (
                    (now_mono - self._last_tilt_update_time)
                    if self._last_tilt_update_time is not None
                    else 2.0
                )
                if dt_tilt <= 0.0 or dt_tilt > 5.0:
                    dt_tilt = 2.0
                self._last_tilt_update_time = now_mono

                # 1. Cálculo individual por muestra de aceleración corregida por bias
                mags_a = []
                rolls_a = []
                pitches_a = []
                for s in accels:
                    ax_s = float(s[0]) * self._accel_scale_factor - self._accel_bias_x
                    ay_s = float(s[1]) * self._accel_scale_factor - self._accel_bias_y
                    az_s = float(s[2]) * self._accel_scale_factor - self._accel_bias_z
                    norm_s = math.sqrt(ax_s**2 + ay_s**2 + az_s**2)
                    mags_a.append(norm_s)
                    rolls_a.append(math.atan2(ay_s, az_s))
                    pitches_a.append(math.atan2(-ax_s, math.sqrt(ay_s**2 + az_s**2)))

                num_samples = len(mags_a)
                mean_norm = sum(mags_a) / num_samples
                var_norm = sum((m - mean_norm) ** 2 for m in mags_a) / num_samples
                std_norm = math.sqrt(var_norm)

                gate_open = abs(mean_norm - 1.0) < 0.08 and std_norm < 0.06
                self._tilt_gate_history.append(1 if gate_open else 0)

                duty_cycle_pct = (
                    sum(self._tilt_gate_history) / len(self._tilt_gate_history) * 100.0
                    if self._tilt_gate_history
                    else 0.0
                )

                # Ruido de proceso del gyro y ruido de medición del acelerómetro
                q_gyro = self._gyro_drift_noise_density
                sigma_acc = math.radians(0.85)  # ~0.015 rad de ruido base del acelerómetro en reposo
                alpha = self._tilt_filter_alpha

                if gate_open:
                    rolls_a.sort()
                    pitches_a.sort()
                    roll_acc = rolls_a[num_samples // 2]
                    pitch_acc = pitches_a[num_samples // 2]

                    # Propagación por giróscopo + Corrección por acelerómetro
                    roll_pred = self._filtered_roll + omega_x * dt_tilt
                    pitch_pred = self._filtered_pitch + omega_y * dt_tilt
                    self._filtered_roll = (1.0 - alpha) * roll_pred + alpha * roll_acc
                    self._filtered_pitch = (1.0 - alpha) * pitch_pred + alpha * pitch_acc

                    # Actualización de incertidumbre
                    sigma_pred_sq = self._tilt_uncertainty_rad**2 + (q_gyro**2) * dt_tilt
                    self._tilt_uncertainty_rad = math.sqrt(
                        (1.0 - alpha)**2 * sigma_pred_sq + (alpha**2) * (sigma_acc**2)
                    )
                    self._last_accel_valid_time = now_mono
                else:
                    # Propagación pura por giróscopo (sin salto discontinuo)
                    self._filtered_roll += omega_x * dt_tilt
                    self._filtered_pitch += omega_y * dt_tilt

                    # Incertidumbre acumulada en régimen de dead-reckoning angular
                    self._tilt_uncertainty_rad = math.sqrt(
                        self._tilt_uncertainty_rad**2 + (q_gyro**2) * dt_tilt
                    )

                time_since_accel_s = (
                    (now_mono - self._last_accel_valid_time)
                    if self._last_accel_valid_time is not None
                    else 999.0
                )

                roll_to_use = self._filtered_roll
                pitch_to_use = self._filtered_pitch

                bx = mx * math.cos(pitch_to_use) + mz * math.sin(pitch_to_use)
                by = (
                    mx * math.sin(roll_to_use) * math.sin(pitch_to_use)
                    + my * math.cos(roll_to_use)
                    - mz * math.sin(roll_to_use) * math.cos(pitch_to_use)
                )

                heading_tilt_comp_rad = math.atan2(-by, bx)
                heading_tilt_comp_deg = (
                    math.degrees(heading_tilt_comp_rad) + 360.0
                ) % 360.0

                tilt_msg = Float32()
                tilt_msg.data = heading_tilt_comp_deg
                self.heading_tilt_comp_pub.publish(tilt_msg)

                # Publicar diagnóstico enriquecido de inclinación (E.1.5)
                diag_data = {
                    "mean_norm_g": round(mean_norm, 4),
                    "std_norm_g": round(std_norm, 4),
                    "gate_open": gate_open,
                    "duty_cycle_pct": round(duty_cycle_pct, 1),
                    "time_since_accel_s": round(time_since_accel_s, 1),
                    "filtered_roll_deg": round(math.degrees(self._filtered_roll), 2),
                    "filtered_pitch_deg": round(math.degrees(self._filtered_pitch), 2),
                    "tilt_uncertainty_deg": round(math.degrees(self._tilt_uncertainty_rad), 2),
                    "heading_tilt_comp_deg": round(heading_tilt_comp_deg, 2),
                }
                diag_msg = String()
                diag_msg.data = json.dumps(diag_data)
                self.tilt_gate_diag_pub.publish(diag_msg)

                if duty_cycle_pct < 30.0 and len(self._tilt_gate_history) >= 15:
                    self.get_logger().warn(
                        f"Tilt gate duty cycle bajo ({duty_cycle_pct:.1f}% < 30%). "
                        f"Vibración continua (std_norm: {std_norm:.3f}g). Umbral requiere revisión.",
                        throttle_duration_sec=15.0,
                    )
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                pass

        # ---------------------------------------------------------
        # 3c. DETECCIÓN DE SATURACIÓN MAGNÉTICA Y GATING CONTINUO (Fase 1)
        # ---------------------------------------------------------
        mag_conf = 1.0
        cov_yaw = float(self._imu_orientation_covariance[8])
        if mags:
            try:
                mx_last = float(mags[-1][0])
                my_last = float(mags[-1][1])
                mz_last = float(mags[-1][2])
                num_gyros = len(gyros) if gyros else 1
                avg_gz_val = sum(float(g[2]) for g in gyros) / num_gyros if gyros else 0.0
                omega_z_val = math.radians(avg_gz_val) - self._gyro_bias_z
                mag_conf, cov_yaw, _ = self._evaluate_magnetic_gate(
                    mx_last, my_last, mz_last, omega_z_val, accel_gate_open=gate_open
                )
                if getattr(self, "_gps_mag_conflict", False):
                    cov_yaw = float(self._mag_untrusted_yaw_covariance)
                    mag_conf = 0.0
            except Exception as e:
                self.get_logger().error(f"Error evaluando compuerta magnética: {e}")

        # ---------------------------------------------------------
        # 4. IMU Y ODOMETRÍA DE RUEDAS
        # ---------------------------------------------------------
        # Publicamos un único mensaje IMU por reporte de telemetría (frecuencia ~0.5 Hz)
        if yaw is not None:
            imu = Imu()
            imu.header.stamp = now
            imu.header.frame_id = "base_link"

            if accels:
                num_samples = len(accels)
                avg_ax = (sum(float(sample[0]) for sample in accels) / num_samples * self._accel_scale_factor - self._accel_bias_x) * GRAVITY_M_S2
                avg_ay = (sum(float(sample[1]) for sample in accels) / num_samples * self._accel_scale_factor - self._accel_bias_y) * GRAVITY_M_S2
                avg_az = (sum(float(sample[2]) for sample in accels) / num_samples * self._accel_scale_factor - self._accel_bias_z) * GRAVITY_M_S2
                imu.linear_acceleration.x = avg_ax
                imu.linear_acceleration.y = avg_ay
                imu.linear_acceleration.z = avg_az

            if gyros:
                num_gyros = len(gyros)
                avg_gx = sum(float(g[0]) for g in gyros) / num_gyros
                avg_gy = sum(float(g[1]) for g in gyros) / num_gyros
                avg_gz = sum(float(g[2]) for g in gyros) / num_gyros
                imu.angular_velocity.x = math.radians(avg_gx) - self._gyro_bias_x
                imu.angular_velocity.y = math.radians(avg_gy) - self._gyro_bias_y
                imu.angular_velocity.z = math.radians(avg_gz) - self._gyro_bias_z

            imu.orientation.x = 0.0
            imu.orientation.y = 0.0
            imu.orientation.z = math.sin(yaw / 2.0)
            imu.orientation.w = math.cos(yaw / 2.0)

            imu_ori_cov = list(self._imu_orientation_covariance)
            imu_ori_cov[8] = cov_yaw
            if mag_conf < 0.4:
                # Si la confianza es muy baja o nula (saturación/congelamiento),
                # penalizar masivamente roll y pitch de orientación para que robot_localization
                # ignore la actitud magnética corrupta por completo.
                imu_ori_cov[0] = self._mag_untrusted_yaw_covariance
                imu_ori_cov[4] = self._mag_untrusted_yaw_covariance

            imu.orientation_covariance = imu_ori_cov
            imu.angular_velocity_covariance = self._imu_angular_velocity_covariance
            imu.linear_acceleration_covariance = self._imu_linear_acceleration_covariance
            
            self.imu_pub.publish(imu)

        speed = data.get("speed")
        speed_m_s = None
        if speed is not None:
            try:
                # El campo 'speed' de la telemetría del SDK se reporta en km/h (confirmado empíricamente en Test D:
                # a ~112 RPM da speed=1.91 km/h, equivalente a 0.53 m/s).
                # Conversión estándar a m/s para odometría de ROS 2 (REP-103):
                speed_m_s = float(speed) / 3.6
            except (TypeError, ValueError):
                speed_m_s = None

        yaw_rate = None
        if gyros:
            try:
                num_gyros = len(gyros)
                avg_gz = sum(float(g[2]) for g in gyros) / num_gyros
                yaw_rate = math.radians(avg_gz)
            except (TypeError, ValueError, IndexError):
                yaw_rate = None

        current_time = time.monotonic()
        dt = None
        if self._last_odom_time is not None:
            dt = current_time - self._last_odom_time
            
        if speed_m_s is not None and yaw is not None and dt is not None and dt > 0:
            self._odom_x += speed_m_s * math.cos(yaw) * dt
            self._odom_y += speed_m_s * math.sin(yaw) * dt

        if speed_m_s is not None and yaw is not None:
            if yaw_rate is None and self._last_yaw is not None and dt is not None and dt > 0:
                yaw_rate = self._normalize_angle(yaw - self._last_yaw) / dt
            self._last_yaw = yaw
            self._last_odom_time = current_time

            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"
            odom.pose.pose.position.x = self._odom_x
            odom.pose.pose.position.y = self._odom_y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = Quaternion(
                x=0.0,
                y=0.0,
                z=math.sin(yaw / 2.0),
                w=math.cos(yaw / 2.0),
            )
            odom.pose.covariance = self._odom_pose_covariance
            odom.twist.twist.linear.x = speed_m_s
            odom.twist.twist.linear.y = 0.0
            odom.twist.twist.linear.z = 0.0
            odom.twist.twist.angular.z = yaw_rate if yaw_rate is not None else 0.0
            odom.twist.covariance = self._odom_twist_covariance
            self.odom_pub.publish(odom)

    def destroy_node(self):
        self._running = False
        self._stop_event.set()
        self._control_thread.join(timeout=1.0)
        for _ in range(3):
            try:
                response = self._session.post(
                    f"{self.sdk_url}/control",
                    json={"command": {"linear": 0, "angular": 0}},
                    timeout=CONTROL_HTTP_TIMEOUT_S,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                continue
        self._session.close()
        super().destroy_node()


def main(argv=None):
    rclpy.init(args=argv)
    node = EarthRoverBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()