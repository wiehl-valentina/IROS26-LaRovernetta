#!/usr/bin/env python3
"""
GPS Waypoint Navigation Controller for Earth Rover (IROS 2026).
Arquitectura Híbrida: Máquina de estados reactiva con mitigación de latencia de red (Burst & Wait)
y filtrado pasa-bajos para brújula ruidosa.
"""

import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float32, String, Bool, Int32

class GPSWaypointController(Node):
    def __init__(self):
        super().__init__("gps_waypoint_controller")

        # 1. DECLARACIÓN ESTRICTA DE PARÁMETROS (Tipado Fuerte)
        self.declare_parameter("goal_tolerance_m", 13.0) # Radio de llegada al waypoint/checkpoint (metros)
        self.declare_parameter("goal_dwell_s", 1.0)
        self.declare_parameter("align_threshold_deg", 18.0)
        self.declare_parameter("coarse_align_threshold_deg", 25.0)
        self.declare_parameter("approach_align_distance_m", 8.0)
        
        # Parámetros de acelerador normalizado [0.0, 1.0] (Brief 18 / R.1) con aliases de migración
        self.declare_parameter("forward_throttle", 0.40)
        self.declare_parameter("forward_speed", -1.0)        # alias de migración
        self.declare_parameter("turn_throttle", 0.70)
        self.declare_parameter("turn_speed", -1.0)           # alias de migración
        self.declare_parameter("recovery_turn_throttle", 0.30)
        self.declare_parameter("recovery_turn_speed", -1.0)  # alias de migración
        self.declare_parameter("geodesic_fallback_throttle", 0.20)
        self.declare_parameter("geodesic_fallback_speed", -1.0) # alias de migración
        self.declare_parameter("max_linear_speed_mps", 1.111)

        self.declare_parameter("drive_correction_gain", 0.01)
        self.declare_parameter("max_drive_angular", 0.45)
        self.declare_parameter("invert_angular", True)
        self.declare_parameter("control_loop_hz", 3.0)
        self.declare_parameter("turn_burst_s", 0.25)
        self.declare_parameter("pause_after_turn_s", 0.5)
        self.declare_parameter("max_heading_jump_deg", 150.0)
        self.declare_parameter("heading_filter_alpha", 0.35)
        self.declare_parameter("reached_publish_period_s", 1.0)
        self.declare_parameter("gps_max_stale_s", 2.0)
        self.declare_parameter("path_topic", "earth_rover/planned_path")
        self.declare_parameter("path_valid_topic", "earth_rover/planner_valid")
        self.declare_parameter("path_max_stale_s", 8.0)
        self.declare_parameter("lookahead_distance_m", 1.0)
        self.declare_parameter("path_following_enabled", True)
        self.declare_parameter("max_total_drive_angular", 0.8)
        self.declare_parameter("publish_control_debug", True)
        self.declare_parameter("safe_velocity_limit_topic", "earth_rover/safe_velocity_limit")
        self.declare_parameter("heading_max_stale_s", 3.5)
        self.declare_parameter("heading_fresh_wait_timeout_s", 2.5)
        self.declare_parameter("require_velocity_governor", True)

        # Parámetros FIX 1a / FIX 1b (Histéresis y clamp BEV)
        self.declare_parameter("drive_abort_threshold_deg", 65.0)
        self.declare_parameter("drive_abort_dwell_s", 1.5)
        self.declare_parameter("max_bev_deviation_deg", 45.0)
        self.declare_parameter("drive_pivot_threshold_deg", 30.0)

        # Parámetros FIX 2a / FIX 2b (Burst proporcional e incertidumbre de rumbo)
        self.declare_parameter("turn_burst_min_s", 0.15)
        self.declare_parameter("turn_burst_max_s", 1.20)
        self.declare_parameter("yaw_rate_deg_s", 17.0)
        self.declare_parameter("turn_burst_damping", 0.6)
        self.declare_parameter("heading_trust_threshold_deg", 10.0)

        # Parámetros FIX 3 (Sticky RECOVERY y timeout)
        self.declare_parameter("recovery_max_duration_s", 20.0)

        # Parámetros FIX 1 (Acoplamiento lineal/angular y reducción de avance en giro)
        self.declare_parameter("turn_slowdown_factor", 0.7)
        self.declare_parameter("min_drive_throttle_ratio", 0.3)
        self.declare_parameter("motor_deadband_throttle", 0.15)

        # Parámetros FIX 2 (Reducción por congestión de obstáculos en BEV alive_paths)
        self.declare_parameter("alive_paths_topic", "earth_rover/alive_paths")
        self.declare_parameter("alive_paths_nominal", 40)
        self.declare_parameter("min_congestion_ratio", 0.25)
        self.declare_parameter("alive_paths_max_stale_s", 5.0)

        # 2. EXTRACCIÓN DIRECTA DE PARÁMETROS (sin clamps, valores tal cual el yaml)

        # --- Tolerancias y Distancias ---
        self.goal_tolerance = float(self.get_parameter("goal_tolerance_m").value)
        self.base_goal_tolerance = self.goal_tolerance  # Respaldo de la tolerancia original
        self.goal_dwell_s = float(self.get_parameter("goal_dwell_s").value)
        self.approach_align_distance = float(self.get_parameter("approach_align_distance_m").value)

        # --- Umbrales de Alineación ---
        self.align_threshold = float(self.get_parameter("align_threshold_deg").value)
        self.coarse_align_threshold = float(self.get_parameter("coarse_align_threshold_deg").value)

        # --- Dinámica de Conducción y Giro (Acelerador normalizado [0.0, 1.0]) ---
        f_spd = float(self.get_parameter("forward_speed").value)
        f_thr = float(self.get_parameter("forward_throttle").value)
        self.forward_throttle = f_spd if f_spd >= 0.0 else f_thr
        self.forward_speed = self.forward_throttle  # alias retrocompatibilidad interna

        t_spd = float(self.get_parameter("turn_speed").value)
        t_thr = float(self.get_parameter("turn_throttle").value)
        self.turn_throttle = t_spd if t_spd >= 0.0 else t_thr
        self.turn_speed = self.turn_throttle

        r_spd = float(self.get_parameter("recovery_turn_speed").value)
        r_thr = float(self.get_parameter("recovery_turn_throttle").value)
        self.recovery_turn_throttle = r_spd if r_spd >= 0.0 else r_thr
        self.recovery_turn_speed = self.recovery_turn_throttle

        g_spd = float(self.get_parameter("geodesic_fallback_speed").value)
        g_thr = float(self.get_parameter("geodesic_fallback_throttle").value)
        self.geodesic_fallback_throttle = g_spd if g_spd >= 0.0 else g_thr
        self.geodesic_fallback_speed = self.geodesic_fallback_throttle

        self.max_linear_speed_mps = float(self.get_parameter("max_linear_speed_mps").value)

        # --- Dinámica de Conducción en Curva ---
        self.drive_correction_gain = float(self.get_parameter("drive_correction_gain").value)
        self.max_drive_angular = float(self.get_parameter("max_drive_angular").value)
        self.invert_angular = bool(self.get_parameter("invert_angular").value)
        self.max_total_drive_angular = float(self.get_parameter("max_total_drive_angular").value)

        # --- Guard de GPS y Seguimiento de Trayectorias BEV ---
        self.gps_max_stale_s = float(self.get_parameter("gps_max_stale_s").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.path_valid_topic = str(self.get_parameter("path_valid_topic").value)
        self.path_max_stale_s = float(self.get_parameter("path_max_stale_s").value)
        self.lookahead_distance_m = float(self.get_parameter("lookahead_distance_m").value)
        self.path_following_enabled = bool(self.get_parameter("path_following_enabled").value)

        # --- Tiempos de Ráfaga y Filtros ---
        self.turn_burst_s = float(self.get_parameter("turn_burst_s").value)
        self.pause_after_turn_s = float(self.get_parameter("pause_after_turn_s").value)
        self.max_heading_jump = float(self.get_parameter("max_heading_jump_deg").value)

        self.heading_filter_alpha = float(self.get_parameter("heading_filter_alpha").value)
        self.reached_publish_period_s = float(self.get_parameter("reached_publish_period_s").value)
        self.loop_hz = float(self.get_parameter("control_loop_hz").value)
        self.publish_control_debug = bool(self.get_parameter("publish_control_debug").value)
        self.heading_max_stale_s = float(self.get_parameter("heading_max_stale_s").value)
        self.heading_fresh_wait_timeout_s = float(
            self.get_parameter("heading_fresh_wait_timeout_s").value
        )
        self.require_velocity_governor = bool(self.get_parameter("require_velocity_governor").value)

        # --- Extracción FIX 1, FIX 2, FIX 3 ---
        self.drive_abort_threshold_deg = float(self.get_parameter("drive_abort_threshold_deg").value)
        self.drive_abort_dwell_s = float(self.get_parameter("drive_abort_dwell_s").value)
        self.max_bev_deviation_deg = float(self.get_parameter("max_bev_deviation_deg").value)
        self.drive_pivot_threshold_deg = float(self.get_parameter("drive_pivot_threshold_deg").value)
        self.turn_burst_min_s = float(self.get_parameter("turn_burst_min_s").value)
        self.turn_burst_max_s = float(self.get_parameter("turn_burst_max_s").value)
        self.yaw_rate_deg_s = float(self.get_parameter("yaw_rate_deg_s").value)
        self.turn_burst_damping = float(self.get_parameter("turn_burst_damping").value)
        self.heading_trust_threshold_deg = float(self.get_parameter("heading_trust_threshold_deg").value)
        self.recovery_max_duration_s = float(self.get_parameter("recovery_max_duration_s").value)
        self.turn_slowdown_factor = float(self.get_parameter("turn_slowdown_factor").value)
        self.min_drive_throttle_ratio = float(self.get_parameter("min_drive_throttle_ratio").value)
        self.motor_deadband_throttle = float(self.get_parameter("motor_deadband_throttle").value)
        self.alive_paths_nominal = int(self.get_parameter("alive_paths_nominal").value)
        self.min_congestion_ratio = float(self.get_parameter("min_congestion_ratio").value)
        self.alive_paths_max_stale_s = float(self.get_parameter("alive_paths_max_stale_s").value)

        # 3. Perfiles QoS Diferenciados (Crítico para Jazzy)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # 4. Suscriptores y Publicadores
        self.create_subscription(NavSatFix, "gps/filtered", self._on_gps, sensor_qos)
        self.create_subscription(
            Float32,
            "earth_rover/heading",   # <--- Salida directa de ekf_heading_bridge
            self._on_heading,
            sensor_qos,              # BEST_EFFORT
        )
        self.create_subscription(
            Float32,
            "earth_rover/heading_raw",  # Salida directa de telemetría SDK
            self._on_heading_compass,
            sensor_qos,
        )
        self.create_subscription(
            Float32,
            "earth_rover/heading_compass",  # Salida de compás de ekf_heading_bridge
            self._on_heading_compass,
            sensor_qos,
        )
        self.create_subscription(
            Imu,
            "/imu/data",
            self._on_imu_data,
            sensor_qos,
        )
        self.create_subscription(
            Path,
            self.path_topic,
            self._on_planned_path,
            sensor_qos,              # BEST_EFFORT
        )
        self.create_subscription(
            Bool,
            self.path_valid_topic,
            self._on_path_valid,
            sensor_qos,              # BEST_EFFORT
        )
        self.create_subscription(NavSatFix, "earth_rover/target_waypoint", self._on_target, reliable_qos)
        self.create_subscription(Bool, "earth_rover/navigation_pause", self._on_navigation_pause, reliable_qos)
        self.create_subscription(String, "earth_rover/waypoint_status", self._on_mission_status, reliable_qos)
        safe_vel_topic = str(self.get_parameter("safe_velocity_limit_topic").value)
        self.create_subscription(Float32, safe_vel_topic, self._on_safe_vel_limit, sensor_qos)
        self.create_subscription(
            Float32,
            "earth_rover/heading_uncertainty",
            self._on_heading_uncertainty,
            sensor_qos,
        )
        alive_paths_topic = str(self.get_parameter("alive_paths_topic").value)
        self.create_subscription(
            Int32,
            alive_paths_topic,
            self._on_alive_paths,
            sensor_qos,
        )

        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", reliable_qos)
        self.status_pub = self.create_publisher(String, "earth_rover/waypoint_status", reliable_qos)
        self.control_debug_pub = self.create_publisher(String, "earth_rover/control_debug", sensor_qos)

        # 5. Inicialización de Vectores de Estado
        self.current_lat = None
        self.current_lon = None
        self._gps_last_update = None
        self.current_heading = None
        self._raw_heading = None
        self._heading_last_rx = None
        self._heading_seq: int = 0
        self._heading_compass_last: float | None = None
        self._compass_seq: int = 0
        self._gyro_z_raw: float | None = None
        self._last_turn_heading_seq: int | None = None
        self._last_turn_heading_rx = None
        self.target_lat = None
        self.target_lon = None

        # Gobernador de Velocidad (H.2 / R.1)
        self._safe_velocity_limit: float = self.max_linear_speed_mps
        self._safe_velocity_limit_last_rx = None

        # Seguimiento de Trayectorias Planificadas (BEV)
        self._path_poses: list[tuple[float, float]] = []
        self._path_valid: bool = False
        self._path_last_update = None
        self._alive_paths: int | None = None
        self._alive_paths_last_rx = None
        
        # Flags de Máquina de Estados
        self.active_goal = False
        self._reached_since = None
        self._awaiting_next_target = False
        self._last_reached_publish_at = None
        self._align_phase = "PAUSE"
        self._align_phase_started_at = None
        self._burst_turn_sign = 0
        self._navigation_paused = False
        self._control_mode: str = "ALIGN"
        self._drive_abort_started_at = None
        self._recovery_started_at = None
        self._recovery_turn_sign: int = 1
        self._heading_uncertainty_deg: float | None = None
        self._heading_uncertainty_last_rx = None
        self._current_burst_duration: float = self.turn_burst_s

        # Instrumentación de Ciclo de Trabajo (Brief 18 / R.3.1)
        self._duty_drive_s: float = 0.0
        self._duty_pivot_s: float = 0.0
        self._duty_turn_s: float = 0.0
        self._duty_pause_s: float = 0.0
        self._duty_recovery_s: float = 0.0
        self._duty_total_s: float = 0.0
        self._last_duty_tick_at = None
        self._last_duty_log_at = None

        # 6. Bucle de Control Principal
        self.timer = self.create_timer(1.0 / self.loop_hz, self._control_loop)
        self.get_logger().info(
            f"Controlador Híbrido Iniciado | Loop: {self.loop_hz} Hz | Burst: {self.turn_burst_s}s | "
            f"Require Governor: {self.require_velocity_governor} | Path Following: {self.path_following_enabled} | "
            f"Forward Throttle: {self.forward_throttle:.2f} (max {self.max_linear_speed_mps:.2f} m/s) | "
            f"Fallback Throttle: {self.geodesic_fallback_throttle:.2f}"
        )

    # --- CALLBACKS DE SENSORES Y PERCEPCIÓN ---
    def _on_gps(self, msg: NavSatFix):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self._gps_last_update = self.get_clock().now()

    def _on_heading(self, msg: Float32):
        self._heading_last_rx = self.get_clock().now()
        raw = float(msg.data) % 360.0
        self._raw_heading = raw

        if self.current_heading is None:
            self.current_heading = raw
            self._heading_seq += 1
            return

        jump = abs(self.angle_error_deg(raw, self.current_heading))
        if jump > self.max_heading_jump:
            # AHORA SABREMOS SI LA BRÚJULA SE ESTÁ RECHAZANDO
            self.get_logger().warn(f"Salto magnético gigante rechazado: {jump:.1f}°")
            return

        delta = self.angle_error_deg(raw, self.current_heading)
        self.current_heading = (self.current_heading + self.heading_filter_alpha * delta) % 360.0
        self._heading_seq += 1

    def _on_heading_compass(self, msg: Float32):
        self._heading_compass_last = float(msg.data) % 360.0
        self._compass_seq += 1

    def _on_imu_data(self, msg: Imu):
        self._gyro_z_raw = float(msg.angular_velocity.z)

    def _on_planned_path(self, msg: Path):
        poses = []
        for pose_stamped in msg.poses:
            poses.append((float(pose_stamped.pose.position.x), float(pose_stamped.pose.position.y)))
        self._path_poses = poses
        self._path_last_update = self.get_clock().now()

    def _on_path_valid(self, msg: Bool):
        self._path_valid = bool(msg.data)
        self._path_last_update = self.get_clock().now()

    def _on_safe_vel_limit(self, msg: Float32):
        self._safe_velocity_limit = float(msg.data)
        self._safe_velocity_limit_last_rx = self.get_clock().now()

    def _on_heading_uncertainty(self, msg: Float32):
        self._heading_uncertainty_deg = float(msg.data)
        self._heading_uncertainty_last_rx = self.get_clock().now()

    def _on_alive_paths(self, msg: Int32):
        self._alive_paths = int(msg.data)
        self._alive_paths_last_rx = self.get_clock().now()

    # --- CALLBACKS DE MÁQUINA DE ESTADOS ---
    def _on_target(self, msg: NavSatFix):
        self.target_lat = msg.latitude
        self.target_lon = msg.longitude
        self.active_goal = True
        self._reached_since = None
        self._awaiting_next_target = False
        self._last_reached_publish_at = None
        self._align_phase = "PAUSE"
        self._align_phase_started_at = None
        self._burst_turn_sign = 0
        self._navigation_paused = False
        self._control_mode = "ALIGN"
        self._drive_abort_started_at = None
        self._recovery_started_at = None
        self._recovery_turn_sign = 1
        self._last_turn_heading_seq = None
        self._last_turn_heading_rx = None
        
        # CRÍTICO: Restaurar tolerancia original al cambiar a una nueva meta
        self.goal_tolerance = self.base_goal_tolerance
        
        self._stop_robot()
        self.get_logger().info(f"Target Fijado: ({self.target_lat:.6f}, {self.target_lon:.6f}) | Tolerancia: {self.goal_tolerance}m")

    def _on_navigation_pause(self, msg: Bool):
        was_paused = self._navigation_paused
        self._navigation_paused = bool(msg.data)
        
        # --- DETECTOR DE RECHAZO DEL SDK ---
        # Si estábamos pausados, nos despausan, y seguimos esperando una meta nueva...
        if was_paused and not self._navigation_paused and self._awaiting_next_target:
            # ¡Significa que el SDK dijo que NO! Estrangulamos la tolerancia a la mitad.
            self.goal_tolerance = max(0.5, self.goal_tolerance * 0.5)
            self.get_logger().warn(
                f"¡Rechazo del SDK detectado! Estrangulando tolerancia geodésica a {self.goal_tolerance:.1f}m"
            )
            self._awaiting_next_target = False
            self._reached_since = None

        if self._navigation_paused:
            self._stop_robot()
            self.get_logger().info("Pausa comandada por mission_manager.")

    def _on_mission_status(self, msg: String):
        if msg.data == "MISSION_FINISHED":
            self._awaiting_next_target = False
            self._navigation_paused = True
            self.active_goal = False
            self._stop_robot()
            self.get_logger().info("Misión finalizada. Controlador inactivo.")

    # --- MOTOR MATEMÁTICO GEODÉSICO ---
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        r = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2.0) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2)
        return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2):
        dlon = math.radians(lon2 - lon1)
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        y = math.sin(dlon) * math.cos(lat2_r)
        x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    @staticmethod
    def angle_error_deg(target_deg, current_deg):
        return (target_deg - current_deg + 540.0) % 360.0 - 180.0

    def _apply_angular_sign(self, angular):
        return -angular if self.invert_angular else angular

    # --- HELPERS DE TIEMPO, FRESCURA Y PATH FOLLOWING ---
    def _phase_elapsed(self, now):
        if self._align_phase_started_at is None:
            return 0.0
        return (now - self._align_phase_started_at).nanoseconds / 1e9

    def _begin_align_phase(self, phase, now):
        self._align_phase = phase
        self._align_phase_started_at = now

    def _publish_reached(self, now):
        self._stop_robot()
        self._last_turn_heading_seq = None
        self._last_turn_heading_rx = None
        msg = String()
        msg.data = "REACHED"
        self.status_pub.publish(msg)
        self._last_reached_publish_at = now
        self.get_logger().info("Señal REACHED publicada -> Esperando al manager.")

    def _stop_robot(self):
        self._align_phase = "PAUSE"
        self._align_phase_started_at = None
        self._burst_turn_sign = 0
        self.cmd_pub.publish(Twist())

    def _gps_is_fresh(self) -> bool:
        if self._gps_last_update is None:
            return False
        age_s = (self.get_clock().now() - self._gps_last_update).nanoseconds / 1e9
        return age_s <= self.gps_max_stale_s

    def _heading_is_fresh(self) -> bool:
        if self._heading_last_rx is None:
            return False
        age_s = (self.get_clock().now() - self._heading_last_rx).nanoseconds / 1e9
        return age_s <= self.heading_max_stale_s

    def _path_is_fresh(self) -> bool:
        if not self.path_following_enabled or self._path_last_update is None:
            return False
        age_s = (self.get_clock().now() - self._path_last_update).nanoseconds / 1e9
        return age_s <= self.path_max_stale_s

    def _compute_path_heading_error_deg(self) -> float | None:
        """
        Recorre self._path_poses (metros, base_link: x=adelante, y=izquierda)
        acumulando distancia hasta encontrar el primer punto a >= 
        lookahead_distance_m del origen (0,0, la posición actual del robot).
        Si el path es más corto que el lookahead, usa el último punto.
        Devuelve el heading_error en grados, MISMA convención que
        angle_error_deg ya usada en el resto del archivo (positivo = el
        objetivo está a la derecha, coherente con cómo se usa heading_error
        en el resto de _control_loop).

        Convención de signos y derivación:
        - Marco ROS REP-103 (base_link): +X = Adelante, +Y = Izquierda, -Y = Derecha.
        - math.atan2(y, x): da ángulo positivo hacia la izquierda (+Y) y negativo hacia la derecha (-Y).
        - Convención de heading_error en gps_waypoint_controller:
            heading_error = angle_error_deg(target_bearing, current_heading)
            -> Si la meta está a la derecha del rumbo actual: heading_error > 0 (positivo).
            -> Si la meta está a la izquierda del rumbo actual: heading_error < 0 (negativo).
        - Por lo tanto, para convertir el ángulo de base_link a heading_error:
            heading_error = -math.degrees(math.atan2(y, x))

        Ejemplos numéricos de verificación:
          1. Punto lookahead a la derecha: (x=+2.0m adelante, y=-1.0m derecha)
             -> atan2(-1.0, 2.0) = -26.57° (ángulo base_link)
             -> heading_error = -(-26.57°) = +26.57° (positivo -> el controlador gira a la DERECHA)
          2. Punto lookahead a la izquierda: (x=+2.0m adelante, y=+1.0m izquierda)
             -> atan2(+1.0, 2.0) = +26.57° (ángulo base_link)
             -> heading_error = -(+26.57°) = -26.57° (negativo -> el controlador gira a la IZQUIERDA)
          3. Punto lookahead recto adelante: (x=+2.0m adelante, y=0.0m centro)
             -> atan2(0.0, 2.0) = 0.0°
             -> heading_error = 0.0°

        Devuelve None si self._path_poses tiene menos de 2 puntos.
        """
        if len(self._path_poses) < 2:
            return None

        accumulated_dist = 0.0
        target_x, target_y = self._path_poses[-1]

        prev_x, prev_y = 0.0, 0.0
        for x, y in self._path_poses:
            seg_dist = math.hypot(x - prev_x, y - prev_y)
            accumulated_dist += seg_dist
            prev_x, prev_y = x, y
            if accumulated_dist >= self.lookahead_distance_m:
                target_x, target_y = x, y
                break

        angle_base_link_rad = math.atan2(target_y, target_x)
        heading_error = -math.degrees(angle_base_link_rad)
        return float(heading_error)

    def _speed_to_throttle(self, v_mps: float) -> float:
        """Convierte una velocidad física (m/s) a fracción de acelerador normalizado [0.0, 1.0].
        
        Provisionalmente utiliza la relación lineal (throttle = v_mps / max_linear_speed_mps).
        A calibrar en hardware con la curva real del Mini+ (Brief 18 / R.1.2 y R.1.3).
        """
        if v_mps <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, v_mps / max(self.max_linear_speed_mps, 0.01))))

    def _control_loop(self):
        now = self.get_clock().now()

        # Guarda de seguridad 1: Pausa externa de navegación
        if self._navigation_paused:
            self._stop_robot()
            return

        # Guarda de seguridad 2: Esperando confirmación de waypoint por el SDK
        if self._awaiting_next_target:
            self._stop_robot()
            if self._last_reached_publish_at is None:
                elapsed = self.reached_publish_period_s
            else:
                elapsed = (now - self._last_reached_publish_at).nanoseconds / 1e9
            # Re-publicar REACHED si el manager tardó en procesar
            if elapsed >= self.reached_publish_period_s:
                self._publish_reached(now)
            return

        # Guarda de seguridad 3: Datos insuficientes
        if not self.active_goal or self.current_lat is None or self.current_lon is None:
            return

        # Guarda de seguridad 4: GPS Stale Guard
        if not self._gps_is_fresh():
            self._stop_robot()
            self.get_logger().warn("GPS stale: frenando y esperando.", throttle_duration_sec=2.0)
            return

        # Guarda de seguridad 5: Heading Stale Guard (Brief 14 / N.2)
        if not self._heading_is_fresh():
            self._stop_robot()
            self.get_logger().warn(
                "Heading stale o no recibido: frenando y esperando por seguridad.",
                throttle_duration_sec=2.0,
            )
            return

        distance = self.haversine_distance(self.current_lat, self.current_lon, self.target_lat, self.target_lon)

        # 1. EVALUACIÓN DE META ALCANZADA
        if distance <= self.goal_tolerance:
            self._stop_robot()
            if self._reached_since is None:
                self._reached_since = now
            
            # Filtro anti-rebote espacial (Dwell Time)
            if (now - self._reached_since).nanoseconds / 1e9 >= self.goal_dwell_s:
                self.get_logger().info(f"¡Meta Alcanzada! ({distance:.1f}m error geodésico)")
                self.active_goal = False
                self._awaiting_next_target = True
                self._navigation_paused = True
                self._publish_reached(now)
            return

        self._reached_since = None

        if self.current_heading is None:
            self._stop_robot()
            return

        # 2. CÁLCULO DE RUMBO GEODÉSICO ABSOLUTO (Brief 24 / X.2)
        bearing = self.calculate_bearing(self.current_lat, self.current_lon, self.target_lat, self.target_lon)
        geodesic_heading_error = self.angle_error_deg(bearing, self.current_heading)

        # Rumbo deseado en marco mundo. Es la MISMA referencia para ALIGN y DRIVE.
        # Por qué esto NO reintroduce el bucle infinito del brief 24: target_heading_world
        # está anclado a bearing ± max_bev_deviation_deg en marco mundo. No rota con el
        # chasis. Si el rover gira hacia el objetivo, el error decrece monótonamente.
        # Verificación del caso límite (incluir como comentario en el código): si el rover
        # ya está en bearing + 45° y el BEV pide otros +45° desde esa vista,
        # bev_desired_world = bearing + 90°, que se recorta a bearing + 45° — el rumbo
        # actual. Error cero. Punto fijo estable.
        if self.path_following_enabled and self._path_is_fresh() and self._path_valid:
            bev_delta = self._compute_path_heading_error_deg()      # relativo a base_link
        else:
            bev_delta = None

        if bev_delta is not None:
            bev_desired_world = (self.current_heading + bev_delta) % 360.0
            deviation = self.angle_error_deg(bev_desired_world, bearing)
            clamped_deviation = max(-self.max_bev_deviation_deg,
                                    min(self.max_bev_deviation_deg, deviation))
            target_heading_world = (bearing + clamped_deviation) % 360.0
            heading_source = "bev_clamped"
        else:
            target_heading_world = bearing
            heading_source = "geodesic"

        heading_error = self.angle_error_deg(target_heading_world, self.current_heading)

        # Umbral dinámico de alineación (más estricto al acercarse)
        align_threshold = (
            self.align_threshold
            if distance <= self.approach_align_distance
            else self.coarse_align_threshold
        )

        # 3. MÁQUINA DE ESTADOS PRINCIPAL: ALIGN vs DRIVE vs RECOVERY (Brief 24 / FIX 1 / FIX 3)
        # Nota de diseño (FIX 3): RECOVERY tiene prioridad sobre la alineación geodésica porque el
        # objetivo geodésico puede ser físicamente inalcanzable en línea recta; sin esta prioridad,
        # el controlador insiste en apuntar hacia un obstáculo indefinidamente.
        if self._control_mode == "RECOVERY":
            if self._recovery_started_at is None:
                self._recovery_started_at = now
                self._recovery_turn_sign = 1 if geodesic_heading_error >= 0.0 else -1
            # Condiciones de salida de RECOVERY (evaluadas en este orden):
            # 1. _path_valid == True con path fresco -> salir a DRIVE
            if self.path_following_enabled and self._path_is_fresh() and self._path_valid:
                self._control_mode = "DRIVE"
                self._recovery_started_at = None
                self._drive_abort_started_at = None
                self.get_logger().info("RECOVERY: Camino válido encontrado -> saliendo a DRIVE")
            # 2. Timeout recovery_max_duration_s sin encontrar path válido -> salir a ALIGN
            elif (
                self._recovery_started_at is not None
                and (now - self._recovery_started_at).nanoseconds / 1e9 >= self.recovery_max_duration_s
            ):
                self._control_mode = "ALIGN"
                self._recovery_started_at = None
                self._drive_abort_started_at = None
                self.get_logger().warn(
                    f"RECOVERY: Timeout ({self.recovery_max_duration_s:.1f}s) sin encontrar salida -> saliendo a ALIGN"
                )

        if self._control_mode == "ALIGN":
            # ALIGN -> DRIVE: cuando error respecto al rumbo deseado cae por debajo del umbral
            if abs(heading_error) <= align_threshold:
                self._control_mode = "DRIVE"
                self._drive_abort_started_at = None
        elif self._control_mode == "DRIVE":
            # Si el planner indica camino inválido, entra a RECOVERY
            if self.path_following_enabled and self._path_is_fresh() and not self._path_valid:
                self._control_mode = "RECOVERY"
                self._recovery_started_at = now
                self._drive_abort_started_at = None
                # FIX 3: Girar hacia el lado donde está el objetivo geodésico
                self._recovery_turn_sign = 1 if geodesic_heading_error >= 0.0 else -1
                self.get_logger().warn(
                    f"DRIVE: Planner sin camino válido -> entrando a RECOVERY (giro sign={self._recovery_turn_sign:+} hacia err={geodesic_heading_error:+.1f}°)"
                )
            # FIX 1a: Transición asimétrica DRIVE -> ALIGN con dwell time
            elif abs(geodesic_heading_error) > self.drive_abort_threshold_deg:
                if self._drive_abort_started_at is None:
                    self._drive_abort_started_at = now
                elif (now - self._drive_abort_started_at).nanoseconds / 1e9 >= self.drive_abort_dwell_s:
                    self._control_mode = "ALIGN"
                    self._drive_abort_started_at = None
                    self.get_logger().warn(
                        f"DRIVE: Abortado por desvío geodésico sostenido ({abs(geodesic_heading_error):.1f}° > "
                        f"{self.drive_abort_threshold_deg:.1f}° por {self.drive_abort_dwell_s:.1f}s) -> ALIGN"
                    )
            else:
                self._drive_abort_started_at = None

        twist = Twist()

        if self._control_mode == "ALIGN":
            mode = "ALIGN"
            twist.linear.x = 0.0
            elapsed = self._phase_elapsed(now)

            if self._align_phase == "PAUSE":
                twist.angular.z = 0.0
                if self._align_phase_started_at is None or elapsed >= self.pause_after_turn_s:
                    # FIX 2a: No esperar al compás si la propagación es confiable
                    has_new_heading = (
                        self._last_turn_heading_seq is None
                        or (
                            self._compass_seq > self._last_turn_heading_seq
                            if self._compass_seq > 0
                            else self._heading_seq > self._last_turn_heading_seq
                        )
                        or (
                            self._heading_uncertainty_deg is not None
                            and self._heading_uncertainty_deg < self.heading_trust_threshold_deg
                        )
                    )
                    pause_timeout_reached = elapsed >= self.heading_fresh_wait_timeout_s

                    if not has_new_heading and not pause_timeout_reached:
                        # Extender PAUSE a la espera de un heading fresco
                        curr_seq = self._compass_seq if self._compass_seq > 0 else self._heading_seq
                        unc_str = f", unc={self._heading_uncertainty_deg:.1f}°" if self._heading_uncertainty_deg is not None else ""
                        self.get_logger().info(
                            f"[ALIGN] Esperando confirmación de rumbo antes de nueva ráfaga "
                            f"(seq={curr_seq}, last={self._last_turn_heading_seq}{unc_str}, pausa_elapsed={elapsed:.2f}s < {self.heading_fresh_wait_timeout_s:.1f}s)",
                            throttle_duration_sec=1.0,
                        )
                    else:
                        if not has_new_heading and pause_timeout_reached:
                            self.get_logger().warn(
                                f"[ALIGN] Timeout esperando nuevo heading ({elapsed:.2f}s >= {self.heading_fresh_wait_timeout_s:.1f}s). "
                                f"Permitiendo ráfaga de reintento."
                            )
                        self._burst_turn_sign = 1 if heading_error > 0.0 else -1
                        self._last_turn_heading_seq = (
                            self._compass_seq if self._compass_seq > 0 else self._heading_seq
                        )
                        self._last_turn_heading_rx = self._heading_last_rx

                        # FIX 2b: Ráfaga proporcional al error angular
                        burst_s = (abs(heading_error) / max(self.yaw_rate_deg_s, 0.01)) * self.turn_burst_damping
                        self._current_burst_duration = max(self.turn_burst_min_s, min(self.turn_burst_max_s, burst_s))

                        self._begin_align_phase("TURN", now)
                        twist.angular.z = self._apply_angular_sign(self._burst_turn_sign * self.turn_throttle)
            else: # Fase TURN
                twist.angular.z = self._apply_angular_sign(self._burst_turn_sign * self.turn_throttle)
                
                # FIX 2b: Duración calculada de la ráfaga
                burst_s = (abs(heading_error) / max(self.yaw_rate_deg_s, 0.01)) * self.turn_burst_damping
                dynamic_burst = getattr(
                    self,
                    "_current_burst_duration",
                    max(self.turn_burst_min_s, min(self.turn_burst_max_s, burst_s)),
                )
                
                if elapsed >= dynamic_burst:
                    self._begin_align_phase("PAUSE", now)
                    twist.angular.z = 0.0

            # Duty cycle tracking
            if self._last_duty_tick_at is not None:
                dt_duty = (now - self._last_duty_tick_at).nanoseconds / 1e9
                if 0.0 < dt_duty < 2.0:
                    if self._align_phase == "TURN":
                        self._duty_turn_s += dt_duty
                    else:
                        self._duty_pause_s += dt_duty
                    self._duty_total_s += dt_duty
            self._last_duty_tick_at = now

            self.get_logger().info(
                f"[ALIGN] dist={distance:.1f}m, head_err={heading_error:+.1f}°, "
                f"cmd_v={twist.linear.x:.2f}, cmd_w={twist.angular.z:+.2f}, align_phase={self._align_phase}",
                throttle_duration_sec=1.0,
            )
            safe_throttle_limit = self._speed_to_throttle(self._safe_velocity_limit)

        elif self._control_mode == "RECOVERY":
            mode = "RECOVERY"
            self._align_phase = "PAUSE"
            self._align_phase_started_at = None
            self._burst_turn_sign = 0
            self._last_turn_heading_seq = None
            self._last_turn_heading_rx = None

            heading_error = None
            heading_source = "none"
            twist.linear.x = 0.0
            turn_sign = getattr(self, "_recovery_turn_sign", 1)
            twist.angular.z = self._apply_angular_sign(turn_sign * self.recovery_turn_throttle)
            safe_throttle_limit = 0.0

            # Duty cycle tracking
            if self._last_duty_tick_at is not None:
                dt_duty = (now - self._last_duty_tick_at).nanoseconds / 1e9
                if 0.0 < dt_duty < 2.0:
                    self._duty_recovery_s += dt_duty
                    self._duty_total_s += dt_duty
            self._last_duty_tick_at = now

            self.get_logger().warn(
                "Planner: sin camino válido (recovery turn activo).",
                throttle_duration_sec=2.0,
            )
            self.get_logger().info(
                f"[RECOVERY] dist={distance:.1f}m, head_err=None, "
                f"cmd_v=0.00, cmd_w={twist.angular.z:+.2f}, align_phase={self._align_phase}",
                throttle_duration_sec=1.0,
            )

        else: # self._control_mode == "DRIVE"
            mode = "DRIVE"
            self._align_phase_started_at = None
            self._burst_turn_sign = 0
            self._last_turn_heading_seq = None
            self._last_turn_heading_rx = None

            # Gobernador de velocidad dinámico fail-safe (Brief 14 / N.1, Brief 15 / O.2, Brief 18 / R.1)
            safe_throttle_limit = self._speed_to_throttle(self._safe_velocity_limit)

            if self._safe_velocity_limit_last_rx is None:
                if self.require_velocity_governor:
                    # Caso 1a: Esperado pero nunca recibido (arranque, planner no listo aún) -> acelerador 0
                    effective_throttle = 0.0
                    self.get_logger().warn(
                        "Gobernador: Sin límite de velocidad recibido aún (require_velocity_governor=true). Deteniendo rover por seguridad (thr=0.0).",
                        throttle_duration_sec=2.0,
                    )
                else:
                    # Caso 1b: Modo geodésico puro deliberado (sin gobernador esperado) -> acelerador reducido conservador
                    effective_throttle = max(0.0, min(self.geodesic_fallback_throttle, self.forward_throttle))
                    self.get_logger().info(
                        f"Gobernador no requerido (require_velocity_governor=false). Navegando a acelerador geodésico conservador (thr={effective_throttle:.2f}).",
                        throttle_duration_sec=5.0,
                    )
            else:
                age_safe_vel = (now - self._safe_velocity_limit_last_rx).nanoseconds / 1e9
                if age_safe_vel <= 3.0:
                    # Caso 2: Recibido y vigente -> usar valor dinámico convertido a acelerador
                    effective_throttle = max(0.0, min(self.forward_throttle, safe_throttle_limit))
                else:
                    # Caso 3: Recibido pero expirado (>3.0s)
                    if self.path_following_enabled:
                        # Si depende del planner BEV y éste dejó de publicar -> fail-safe parada (thr=0.0)
                        effective_throttle = 0.0
                        self.get_logger().warn(
                            f"Gobernador EXPIRADO (edad={age_safe_vel:.1f}s > 3.0s) con path following activo. "
                            "Deteniendo rover por seguridad (thr=0.0).",
                            throttle_duration_sec=2.0,
                        )
                    else:
                        # Navegación geodésica pura (sin planner BEV) -> acelerador reducido conservador
                        effective_throttle = max(0.0, min(self.geodesic_fallback_throttle, self.forward_throttle))
                        self.get_logger().warn(
                            f"Gobernador EXPIRADO (edad={age_safe_vel:.1f}s > 3.0s) en modo geodésico puro. "
                            f"Limitando a acelerador conservador (thr={effective_throttle:.2f}).",
                            throttle_duration_sec=2.0,
                        )

            if effective_throttle <= 0.0:
                # Si el gobernador expira o el fail-safe detiene el avance en DRIVE por pérdida de percepción,
                # cancelar también el giro para no rotar en el lugar a ciegas.
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self._align_phase = "PAUSE"
            elif abs(heading_error) > self.drive_pivot_threshold_deg:
                # FIX 2 (CRÍTICO): Pivote en el lugar dentro de DRIVE
                # Pivote en el lugar: girar rápido hacia el rumbo deseado sin avanzar.
                # Girar despacio mientras se avanza no alcanza para esquivar: a w=0.15 el
                # giro es ~3.6 grados/s y corregir 45 grados toma 12 s, durante los cuales
                # el rover recorre varios metros hacia el obstáculo.
                twist.linear.x = 0.0
                twist.angular.z = self._apply_angular_sign(
                    math.copysign(self.turn_throttle, heading_error)
                )
                self._align_phase = "PIVOT"
            else:
                self._align_phase = "DRIVE"
                # 1. Reducción de avance por congestión de obstáculos en BEV (alive_paths, FIX 2)
                congestion_factor = 1.0
                if self._alive_paths is not None and self._alive_paths_last_rx is not None:
                    age_alive = (now - self._alive_paths_last_rx).nanoseconds / 1e9
                    if age_alive <= self.alive_paths_max_stale_s:
                        path_freedom = min(1.0, float(self._alive_paths) / max(1.0, float(self.alive_paths_nominal)))
                        congestion_factor = max(self.min_congestion_ratio, path_freedom)

                # 2. Corrección angular sobre la marcha y reducción por giro (FIX 1b)
                correction = self.drive_correction_gain * heading_error
                angular_demand = min(1.0, abs(correction) / max(self.max_drive_angular, 1e-6))
                turn_slowdown = max(self.min_drive_throttle_ratio, 1.0 - self.turn_slowdown_factor * angular_demand)

                # FIX A.1 (CRÍTICO): Las reducciones NO se multiplican: se toma la más restrictiva (el mínimo).
                # Multiplicarlas lleva el comando por debajo de la zona muerta de los motores y produce deadlock.
                reduction_factor = min(congestion_factor, turn_slowdown)
                effective_throttle = effective_throttle * reduction_factor

                # FIX A.2 (CRÍTICO): Piso de zona muerta. El comando es o cero explícito o por encima
                # de la zona muerta estimada de arranque de motores, nunca un valor intermedio inmóvil.
                if effective_throttle > 0.0 and effective_throttle < self.motor_deadband_throttle:
                    effective_throttle = self.motor_deadband_throttle

                # FIX 1a (CRÍTICO): Acotar el giro para que nunca supere al avance lineal
                # El comando angular nunca puede superar al lineal: con mezcla diferencial,
                # |w| > thr invierte la rueda interna produciendo un pivote violento en vez de un arco.
                w_limit = min(self.max_drive_angular, self.max_total_drive_angular, effective_throttle)
                clamped_angular = max(-w_limit, min(w_limit, correction))

                twist.linear.x = effective_throttle
                twist.angular.z = self._apply_angular_sign(clamped_angular)

            # Duty cycle tracking
            if self._last_duty_tick_at is not None:
                dt_duty = (now - self._last_duty_tick_at).nanoseconds / 1e9
                if 0.0 < dt_duty < 2.0:
                    if self._align_phase == "PIVOT":
                        self._duty_pivot_s += dt_duty
                    else:
                        self._duty_drive_s += dt_duty
                    self._duty_total_s += dt_duty
            self._last_duty_tick_at = now

            self.get_logger().info(
                f"[DRIVE] dist={distance:.1f}m, head_err={heading_error:+.1f}°, "
                f"cmd_thr={twist.linear.x:.2f} (safe_lim_mps={self._safe_velocity_limit:.2f}, safe_thr={safe_throttle_limit:.2f}), "
                f"cmd_w={twist.angular.z:+.2f}, align_phase={self._align_phase}",
                throttle_duration_sec=1.0,
            )

        now_stamp = (now.nanoseconds) / 1e9
        compass_str = f"{self._heading_compass_last:.1f}°" if self._heading_compass_last is not None else "None"
        prop_str = f"{self.current_heading:.1f}°" if self.current_heading is not None else "None"
        gyro_str = f"{self._gyro_z_raw:+.3f}" if self._gyro_z_raw is not None else "None"
        err_str = f"{heading_error:+.1f}°" if heading_error is not None else "None"
        self.get_logger().info(
            f"[TRACE][CTRL] stamp={now_stamp:.3f}s | mode={mode} | thr={twist.linear.x:.2f} | w={twist.angular.z:+.2f} | "
            f"dist={distance:.1f}m | err={err_str} | src={heading_source} | compass={compass_str} | prop={prop_str} | gyro_z={gyro_str}"
        )

        self.cmd_pub.publish(twist)

        # Telemetría interna para Depuración
        status = (
            f"[{mode}] dist={distance:.1f}m, "
            f"head_err={err_str}, "
            f"cmd_thr={twist.linear.x:.2f}, cmd_w={twist.angular.z:+.2f}"
        )
        out = String()
        out.data = status
        self.status_pub.publish(out)

        # Publicación de Telemetría para Identificación de Sistema (Fase 5.B / Brief 18 / R.3.1)
        if self.publish_control_debug:
            now_sec = now.nanoseconds / 1e9
            heading_rx_sec = (
                (self._heading_last_rx.nanoseconds / 1e9)
                if self._heading_last_rx is not None
                else None
            )
            gps_age = (
                ((now - self._gps_last_update).nanoseconds / 1e9)
                if self._gps_last_update is not None
                else None
            )
            path_age = (
                ((now - self._path_last_update).nanoseconds / 1e9)
                if self._path_last_update is not None
                else None
            )
            pct_drive = (self._duty_drive_s / self._duty_total_s * 100.0) if self._duty_total_s > 0 else 0.0
            pct_pivot = (self._duty_pivot_s / self._duty_total_s * 100.0) if self._duty_total_s > 0 else 0.0
            pct_turn = (self._duty_turn_s / self._duty_total_s * 100.0) if self._duty_total_s > 0 else 0.0
            pct_pause = (self._duty_pause_s / self._duty_total_s * 100.0) if self._duty_total_s > 0 else 0.0
            pct_rec = (self._duty_recovery_s / self._duty_total_s * 100.0) if self._duty_total_s > 0 else 0.0
            
            debug_payload = {
                "timestamp_sec": now_sec,
                "mode": mode,
                "heading_error": float(heading_error) if heading_error is not None else None,
                "heading_source": heading_source,
                "geodesic_heading_error": float(geodesic_heading_error),
                "current_heading": float(self.current_heading) if self.current_heading is not None else None,
                "heading_compass_last": (
                    float(self._heading_compass_last)
                    if self._heading_compass_last is not None
                    else None
                ),
                "heading_propagated": (
                    float(self.current_heading)
                    if self.current_heading is not None
                    else None
                ),
                "gyro_z_raw": (
                    float(self._gyro_z_raw)
                    if self._gyro_z_raw is not None
                    else None
                ),
                "heading_rx_sec": heading_rx_sec,
                "cmd_linear_x": float(twist.linear.x),
                "safe_velocity_limit_mps": float(self._safe_velocity_limit),
                "safe_throttle_limit": float(safe_throttle_limit),
                "cmd_angular_z": float(twist.angular.z),
                "align_phase": self._align_phase,
                "heading_uncertainty_deg": (
                    float(self._heading_uncertainty_deg)
                    if self._heading_uncertainty_deg is not None
                    else None
                ),
                "heading_seq": int(self._heading_seq),
                "compass_seq": int(self._compass_seq),
                "last_turn_heading_seq": self._last_turn_heading_seq,
                "waiting_fresh_heading": bool(
                    self._align_phase == "PAUSE"
                    and self._last_turn_heading_seq is not None
                    and (
                        (
                            self._compass_seq <= self._last_turn_heading_seq
                            if self._compass_seq > 0
                            else self._heading_seq <= self._last_turn_heading_seq
                        )
                        and not (
                            self._heading_uncertainty_deg is not None
                            and self._heading_uncertainty_deg < self.heading_trust_threshold_deg
                        )
                    )
                ),
                "gps_age_s": gps_age,
                "path_age_s": path_age,
                "distance_m": float(distance),
                "duty_cycle": {
                    "drive_pct": round(pct_drive, 1),
                    "pivot_pct": round(pct_pivot, 1),
                    "turn_pct": round(pct_turn, 1),
                    "pause_pct": round(pct_pause, 1),
                    "recovery_pct": round(pct_rec, 1),
                    "total_active_s": round(self._duty_total_s, 2),
                },
            }
            dbg_msg = String()
            dbg_msg.data = json.dumps(debug_payload)
            self.control_debug_pub.publish(dbg_msg)

        # Log periódico del ciclo de trabajo cada 10 segundos
        if self._last_duty_log_at is None or (now - self._last_duty_log_at).nanoseconds / 1e9 >= 10.0:
            self._last_duty_log_at = now
            if self._duty_total_s > 0.0:
                pct_drive = self._duty_drive_s / self._duty_total_s * 100.0
                pct_pivot = self._duty_pivot_s / self._duty_total_s * 100.0
                pct_turn = self._duty_turn_s / self._duty_total_s * 100.0
                pct_pause = self._duty_pause_s / self._duty_total_s * 100.0
                pct_rec = self._duty_recovery_s / self._duty_total_s * 100.0
                self.get_logger().info(
                    f"[DUTY_CYCLE] DRIVE={pct_drive:.1f}% | PIVOT={pct_pivot:.1f}% | TURN={pct_turn:.1f}% | PAUSE={pct_pause:.1f}% | RECOVERY={pct_rec:.1f}% (total={self._duty_total_s:.1f}s)"
                )


def main(args=None):
    rclpy.init(args=args)
    node = GPSWaypointController()
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