#!/usr/bin/env python3
"""
Nodo ROS 2 de Planificación de Rutas BEV (GeNIE / SAM-TP) para Earth Rover.

Arquitectura:
- Se suscribe a la cámara frontal ('earth_rover/front/image_raw'), odometría/GPS
  ('gps/filtered'), heading ('earth_rover/heading') y meta ('earth_rover/target_waypoint').
- Corre la predicción de transitabilidad (SAM-TP) y la proyección a vista aérea (BEV)
  junto con el planificador de trayectorias polinomiales (GeNIE) en un hilo desacoplado
  en segundo plano para no bloquear el executor de ROS.
- Publica el camino planificado ('earth_rover/planned_path') en el marco estándar 'base_link'
  (REP-103: +X adelante, +Y izquierda), la visualización de depuración y el estado de validez.
"""

import collections
import json
import math
import threading
import time
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Bool, Float32, Int32, String
import tf2_ros


def genie_xy_to_ros_base_link(x_right_m: float, y_forward_m: float) -> tuple[float, float]:
    """
    Convierte un punto del marco local de GeNIE al marco estándar ROS REP-103 (base_link).

    Convenciones de ejes:
    - GeNIE BEV: +X = Derecha (lateral right), +Y = Adelante (forward).
    - ROS (REP-103 base_link): +X = Adelante (forward), +Y = Izquierda (lateral left).

    Transformación matemática:
      x_ros (adelante)   = y_forward_m
      y_ros (izquierda)  = -x_right_m

    Ejemplo numérico de verificación:
      Punto GeNIE: (x_right=+2.0m, y_forward=+5.0m) [2m a la derecha, 5m adelante]
      -> x_ros = +5.0m (adelante)
      -> y_ros = -2.0m (derecha en ROS, ya que +Y es izquierda)

      Punto GeNIE: (x_right=-1.5m, y_forward=+3.0m) [1.5m a la izquierda, 3m adelante]
      -> x_ros = +3.0m (adelante)
      -> y_ros = +1.5m (izquierda en ROS)
    """
    x_ros = float(y_forward_m)
    y_ros = -float(x_right_m)
    return x_ros, y_ros


def compute_footprint_px(
    robot_l: float,
    robot_w: float,
    bev_resolution: float,
    grid_n: int,
    bev_h: int,
    margin: float = 1.05,
) -> tuple[int, float]:
    """Deriva analíticamente el radio de huella en píxeles (footprint_px) para la grilla GeNIE.

    Aplica D_circ = sqrt(L^2 + W^2) y escala con la relación grid_n / bev_h y margen de seguridad,
    aplicando ceil estrictamente al producto final para evitar sobreinflación por redondeos intermedios.

    Retorna (footprint_px, d_circ).
    """
    d_circ = math.sqrt(robot_l ** 2 + robot_w ** 2)
    footprint_px = int(
        math.ceil((d_circ / float(bev_resolution)) * (float(grid_n) / float(bev_h)) * margin)
    )
    return footprint_px, d_circ


class BEVPlannerNode(Node):
    def __init__(self):
        super().__init__("bev_planner_node")

        # ----------------------------------------------------------------------
        # 1. Declaración y Extracción de Parámetros
        # ----------------------------------------------------------------------
        self.declare_parameter("image_topic", "earth_rover/front/image_raw")
        self.declare_parameter("gps_topic", "gps/filtered")
        self.declare_parameter("heading_topic", "earth_rover/heading")
        self.declare_parameter("target_topic", "earth_rover/target_waypoint")
        self.declare_parameter("planned_path_topic", "earth_rover/planned_path")
        self.declare_parameter("visualization_topic", "earth_rover/planner_visualization")
        self.declare_parameter("valid_topic", "earth_rover/planner_valid")
        self.declare_parameter("local_bev_grid_topic", "earth_rover/local_bev_grid")
        self.declare_parameter("alive_paths_topic", "earth_rover/alive_paths")
        self.declare_parameter("publish_visualization", True)

        # Integración con Planificador Global (Fase 4)
        self.declare_parameter("global_path_topic", "earth_rover/global_path")
        self.declare_parameter("global_planner_valid_topic", "earth_rover/global_planner_valid")
        self.declare_parameter("use_global_path_guidance", True)
        self.declare_parameter("global_path_max_stale_s", 3.0)
        self.declare_parameter("global_lookahead_distance_m", 3.5)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_lookup_timeout_s", 0.2)

        self.declare_parameter("planning_min_period_s", 0.1)
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("hf_repo", "")
        self.declare_parameter("device", "")
        self.declare_parameter("contrast_refine", True)

        # Parámetros de proyección BEV
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("resolution_m_per_px", 0.03)
        self.declare_parameter("forward_range_m", 4.0)
        self.declare_parameter("side_range_m", 2.0)
        self.declare_parameter("max_ray_distance_m", 6.0)
        self.declare_parameter("camera_k_path", "")
        self.declare_parameter("camera_t_base_camera_path", "")

        # Parámetros del planificador GeNIE
        self.declare_parameter("grid_size", 240)
        self.declare_parameter("unknown_cost", 0.2)
        self.declare_parameter("smooth_kernel", 3)
        self.declare_parameter("num_goals", 30)
        self.declare_parameter("num_mid_points_per_goal", 20)
        self.declare_parameter("path_num_samples", 100)
        # Parámetros físicos del vehículo para derivación geométrica de footprint
        self.declare_parameter("robot_length_m", 0.250)
        self.declare_parameter("robot_width_m", 0.190)
        self.declare_parameter("footprint_safety_margin", 1.05)
        # Si footprint_px <= 0, se deriva dinámicamente en código; si > 0, actúa como override manual
        self.declare_parameter("footprint_px", 0)
        self.declare_parameter("threshold_cost", 0.50)
        self.declare_parameter("threshold_points_ratio", 0.05)
        self.declare_parameter("number_of_points_to_filter", 60)
        self.declare_parameter("alpha", 1.0)
        self.declare_parameter("best_k", 12)
        self.declare_parameter("use_clustering", True)
        self.declare_parameter("max_clusters", 4)
        self.declare_parameter("cluster_angle_threshold_deg", 40.0)
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("include_goal_in_path_bank", False)
        self.declare_parameter("include_random_goals", True)

        # Parámetros del Gobernador Dinámico de Velocidad (Brief 8 / H.2)
        self.declare_parameter("safe_velocity_limit_topic", "earth_rover/safe_velocity_limit")
        self.declare_parameter("planner_diagnostics_topic", "earth_rover/planner_diagnostics")
        self.declare_parameter("brake_accel_mps2", 1.5)
        self.declare_parameter("horizon_safety_margin", 1.5)
        self.declare_parameter("rtt_delay_s", 0.061)
        self.declare_parameter("transport_delay_s", 0.080)

        # Extracción
        image_topic = str(self.get_parameter("image_topic").value)
        gps_topic = str(self.get_parameter("gps_topic").value)
        heading_topic = str(self.get_parameter("heading_topic").value)
        target_topic = str(self.get_parameter("target_topic").value)
        planned_path_topic = str(self.get_parameter("planned_path_topic").value)
        visualization_topic = str(self.get_parameter("visualization_topic").value)
        valid_topic = str(self.get_parameter("valid_topic").value)
        local_bev_grid_topic = str(self.get_parameter("local_bev_grid_topic").value)
        alive_paths_topic = str(self.get_parameter("alive_paths_topic").value)
        self.publish_visualization = bool(self.get_parameter("publish_visualization").value)

        safe_velocity_limit_topic = str(self.get_parameter("safe_velocity_limit_topic").value)
        planner_diagnostics_topic = str(self.get_parameter("planner_diagnostics_topic").value)
        self.brake_accel = float(self.get_parameter("brake_accel_mps2").value)
        self.horizon_safety_margin = float(self.get_parameter("horizon_safety_margin").value)
        self.rtt_delay = float(self.get_parameter("rtt_delay_s").value)
        self.transport_delay = float(self.get_parameter("transport_delay_s").value)

        global_path_topic = str(self.get_parameter("global_path_topic").value)
        global_planner_valid_topic = str(self.get_parameter("global_planner_valid_topic").value)
        self.use_global_path_guidance = bool(self.get_parameter("use_global_path_guidance").value)
        self.global_path_max_stale_s = float(self.get_parameter("global_path_max_stale_s").value)
        self.global_lookahead_distance_m = float(self.get_parameter("global_lookahead_distance_m").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)

        self.planning_min_period_s = float(self.get_parameter("planning_min_period_s").value)
        checkpoint_path = str(self.get_parameter("checkpoint_path").value) or None
        hf_repo = str(self.get_parameter("hf_repo").value) or None
        device = str(self.get_parameter("device").value) or None
        contrast_refine = bool(self.get_parameter("contrast_refine").value)

        self.ground_z = float(self.get_parameter("ground_z").value)
        self.bev_resolution = float(self.get_parameter("resolution_m_per_px").value)
        self.forward_range = float(self.get_parameter("forward_range_m").value)
        self.side_range = float(self.get_parameter("side_range_m").value)
        self.max_ray_distance = float(self.get_parameter("max_ray_distance_m").value)

        camera_k_path = str(self.get_parameter("camera_k_path").value)
        camera_t_path = str(self.get_parameter("camera_t_base_camera_path").value)

        # ----------------------------------------------------------------------
        # 2. Carga del Modelo de Percepción, Calibración y Módulos de Planificación
        # ----------------------------------------------------------------------
        try:
            from rover_traversability.calibration import load_camera_K, load_T_base_camera
            from rover_traversability.predictor import TraversabilityPredictor
            from rover_traversability.weights import SamNotInstalledError
            from genie_path_planner.planner import PlannerConfig, plan_on_bev, _resize_pixel
            from genie_path_planner.projection import project_score_to_bev
            from genie_path_planner.path_sampling import sample_paths_polynomial
        except ImportError as exc:
            self.get_logger().error(
                "No se pudo importar rover_traversability / genie_path_planner. Instalá las "
                "dependencias en el entorno:\n"
                "    pip install torch torchvision\n"
                "    pip install --no-build-isolation -e ./genie\n"
                "    pip install -e './traversability[hf]'\n"
                f"Error original: {exc}"
            )
            raise

        self._plan_on_bev = plan_on_bev
        self._project_score_to_bev = project_score_to_bev

        # Cargar calibración de cámara una sola vez
        if camera_k_path:
            self._camera_k = np.load(camera_k_path).astype(np.float64)
        else:
            self._camera_k = load_camera_K().astype(np.float64)

        if camera_t_path:
            self._camera_t = np.load(camera_t_path).astype(np.float64)
        else:
            self._camera_t = load_T_base_camera().astype(np.float64)

        # Configuración del algoritmo de caminos
        seed_val = self.get_parameter("random_seed").value
        random_seed = int(seed_val) if seed_val is not None else None

        bev_h = max(1, int(np.ceil(float(self.forward_range) / float(self.bev_resolution))))
        bev_w = max(1, int(np.ceil((2.0 * float(self.side_range)) / float(self.bev_resolution))))

        # F.1.3: Validación estricta de isotropía geométrica de GeNIE
        if bev_h != bev_w:
            raise ValueError(
                f"Incoherencia de isotropía en BEVPlannerNode: bev_h ({bev_h}) != bev_w ({bev_w}). "
                f"GeNIE asume una grilla euclidiana uniforme e isótropa (grid_size escalar). "
                f"Ajuste forward_range ({self.forward_range}m) y 2*side_range ({2*self.side_range}m) "
                "para que sean exactamente iguales."
            )

        # F.3: Derivación dinámica del footprint_px
        user_footprint = int(self.get_parameter("footprint_px").value)
        grid_n = int(self.get_parameter("grid_size").value)
        if user_footprint > 0:
            footprint_derived = user_footprint
            self.get_logger().info(f"Footprint manual configurado por parámetro: {footprint_derived} px.")
        else:
            robot_l = float(self.get_parameter("robot_length_m").value)
            robot_w = float(self.get_parameter("robot_width_m").value)
            margin = float(self.get_parameter("footprint_safety_margin").value)
            # Nota de derivación (G.4): Aplicamos ceil estrictamente al producto final
            # para evitar inflación de área por doble redondeo intermedio:
            # (D_circ / res) * (grid_n / bev_h) * margin
            footprint_derived, d_circ = compute_footprint_px(
                robot_l=robot_l,
                robot_w=robot_w,
                bev_resolution=float(self.bev_resolution),
                grid_n=grid_n,
                bev_h=bev_h,
                margin=margin,
            )
            self.get_logger().info(
                f"Footprint derivado dinámicamente: {footprint_derived} px "
                f"(Mini+: {robot_l:.3f}m x {robot_w:.3f}m, D_circ={d_circ:.3f}m, "
                f"BEV {bev_h}x{bev_w} @ {self.bev_resolution}m/px -> Grid {grid_n}x{grid_n}, Margen={margin:.2f})."
            )

        self._planner_cfg = PlannerConfig(
            grid_size=grid_n,
            unknown_cost=float(self.get_parameter("unknown_cost").value),
            smooth_kernel=int(self.get_parameter("smooth_kernel").value),
            num_goals=int(self.get_parameter("num_goals").value),
            num_mid_points_per_goal=int(self.get_parameter("num_mid_points_per_goal").value),
            path_num_samples=int(self.get_parameter("path_num_samples").value),
            footprint_px=footprint_derived,
            threshold_cost=float(self.get_parameter("threshold_cost").value),
            threshold_points_ratio=float(self.get_parameter("threshold_points_ratio").value),
            number_of_points_to_filter=int(self.get_parameter("number_of_points_to_filter").value),
            alpha=float(self.get_parameter("alpha").value),
            best_k=int(self.get_parameter("best_k").value),
            use_clustering=bool(self.get_parameter("use_clustering").value),
            max_clusters=int(self.get_parameter("max_clusters").value),
            cluster_angle_threshold_deg=float(self.get_parameter("cluster_angle_threshold_deg").value),
            random_seed=random_seed,
            include_goal_in_path_bank=bool(self.get_parameter("include_goal_in_path_bank").value),
            include_random_goals=bool(self.get_parameter("include_random_goals").value),
        )

        # Precomputar el banco de caminos fijos (GeNIE) si no depende de la meta dinámica (G.5)
        if not self._planner_cfg.include_goal_in_path_bank:
            start0 = (bev_h - 1, bev_w // 2)
            planner_start = _resize_pixel(start0, (bev_h, bev_w), int(self._planner_cfg.grid_size))
            self.get_logger().info(
                f"Precomputando banco de trayectorias GeNIE en robot_start={planner_start}..."
            )
            self._candidate_path_bank = sample_paths_polynomial(
                robot=planner_start,
                num_goals=int(self._planner_cfg.num_goals),
                num_mid_points_per_goal=int(self._planner_cfg.num_mid_points_per_goal),
                num_samples=int(self._planner_cfg.path_num_samples),
                grid_size=int(self._planner_cfg.grid_size),
                goal=None,
                include_random_goals=bool(self._planner_cfg.include_random_goals),
                random_seed=self._planner_cfg.random_seed,
            )
            self.get_logger().info(
                f"Banco GeNIE precomputado: {len(self._candidate_path_bank)} caminos válidos cargados en memoria "
                f"(BEV: {bev_h}x{bev_w} @ {self.bev_resolution}m/px | Grid: {grid_n}x{grid_n} | "
                f"Robot Start: {planner_start} | Footprint: {footprint_derived}px)."
            )
        else:
            self._candidate_path_bank = None
            self.get_logger().warn(
                "ADVERTENCIA CRÍTICA DE LATENCIA: candidate_path_bank es None porque include_goal_in_path_bank=True. "
                "Las trayectorias polinomiales se recomputarán en cada frame (+4000ms de penalización en CPU). "
                "Para operación en tiempo real, configure include_goal_in_path_bank=False."
            )

        try:
            self._predictor = TraversabilityPredictor(
                checkpoint=checkpoint_path,
                device=device,
                hf_repo=hf_repo,
                contrast_refine=contrast_refine,
            )
        except SamNotInstalledError as exc:
            self.get_logger().error(str(exc))
            raise

        self.get_logger().info(
            f"BEV Planner cargado con SAM-TP en device={self._predictor.device} | "
            f"BEV: {self.forward_range}m x {2*self.side_range}m @ {self.bev_resolution}m/px"
        )

        # ----------------------------------------------------------------------
        # 3. Perfiles QoS, Suscriptores y Publicadores
        # ----------------------------------------------------------------------
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

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, image_topic, self._on_image, sensor_qos)
        self.create_subscription(NavSatFix, gps_topic, self._on_gps, sensor_qos)
        self.create_subscription(Float32, heading_topic, self._on_heading, sensor_qos)
        self.create_subscription(NavSatFix, target_topic, self._on_target, reliable_qos)

        # Suscripción al camino global D* Lite (marco 'map')
        self.create_subscription(Path, global_path_topic, self._on_global_path, sensor_qos)
        self.create_subscription(Bool, global_planner_valid_topic, self._on_global_valid, sensor_qos)

        self.path_pub = self.create_publisher(Path, planned_path_topic, reliable_qos)
        self.valid_pub = self.create_publisher(Bool, valid_topic, sensor_qos)
        self.local_grid_pub = self.create_publisher(OccupancyGrid, local_bev_grid_topic, sensor_qos)
        self.alive_paths_pub = self.create_publisher(Int32, alive_paths_topic, sensor_qos)
        self.safe_vel_pub = self.create_publisher(Float32, safe_velocity_limit_topic, sensor_qos)
        self.planner_diag_pub = self.create_publisher(String, planner_diagnostics_topic, sensor_qos)
        self.vis_pub = (
            self.create_publisher(Image, visualization_topic, sensor_qos)
            if self.publish_visualization
            else None
        )

        # Historial de tiempos de ciclo para el gobernador dinámico de velocidad (H.2)
        self._cycle_times_ms: collections.deque[float] = collections.deque(maxlen=10)

        # ----------------------------------------------------------------------
        # 4. Estado de Navegación y Sincronización del Hilo de Planificación
        # ----------------------------------------------------------------------
        self._current_lat: float | None = None
        self._current_lon: float | None = None
        self._current_heading: float | None = None
        self._target_lat: float | None = None
        self._target_lon: float | None = None

        # Estado del Plan Global (D* Lite)
        self._global_lock = threading.Lock()
        self._global_path_poses: list[tuple[float, float]] = []
        self._global_path_valid: bool = False
        self._global_path_last_update: Time | None = None
        self._global_valid_last_update: Time | None = None

        self._frame_lock = threading.Lock()
        self._latest_rgb: np.ndarray | None = None
        self._latest_stamp = None
        self._stop_event = threading.Event()
        self._infer_thread = threading.Thread(target=self._planning_loop, daemon=True)
        self._infer_thread.start()

        self.get_logger().info(
            f"BEV Planner Node inicializado | image={image_topic} | path={planned_path_topic}"
        )

    # --------------------------------------------------------------------------
    # Callbacks de Sensores y Guía Global
    # --------------------------------------------------------------------------
    def _on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"No se pudo convertir el frame de cámara: {exc}", throttle_duration_sec=5.0)
            return

        rgb = bgr[:, :, ::-1]
        with self._frame_lock:
            self._latest_rgb = rgb
            self._latest_stamp = msg.header.stamp

    def _on_gps(self, msg: NavSatFix):
        self._current_lat = float(msg.latitude)
        self._current_lon = float(msg.longitude)

    def _on_heading(self, msg: Float32):
        self._current_heading = float(msg.data) % 360.0

    def _on_target(self, msg: NavSatFix):
        self._target_lat = float(msg.latitude)
        self._target_lon = float(msg.longitude)

    def _on_global_path(self, msg: Path):
        poses = []
        for p in msg.poses:
            poses.append((float(p.pose.position.x), float(p.pose.position.y)))
        with self._global_lock:
            self._global_path_poses = poses
            self._global_path_last_update = self.get_clock().now()

    def _on_global_valid(self, msg: Bool):
        with self._global_lock:
            self._global_path_valid = bool(msg.data)
            self._global_valid_last_update = self.get_clock().now()

    def _global_path_is_fresh(self) -> bool:
        with self._global_lock:
            if not self.use_global_path_guidance or self._global_path_last_update is None:
                return False
            age_s = (self.get_clock().now() - self._global_path_last_update).nanoseconds / 1e9
            return age_s <= self.global_path_max_stale_s

    # --------------------------------------------------------------------------
    # Motor Matemático Geodésico (Idéntico a gps_waypoint_controller)
    # --------------------------------------------------------------------------
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2.0) ** 2
        )
        return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        dlon = math.radians(lon2 - lon1)
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        y = math.sin(dlon) * math.cos(lat2_r)
        x = (
            math.cos(lat1_r) * math.sin(lat2_r)
            - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        )
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    @staticmethod
    def angle_error_deg(target_deg: float, current_deg: float) -> float:
        return (target_deg - current_deg + 540.0) % 360.0 - 180.0

    # --------------------------------------------------------------------------
    # Bucle Principal de Planificación (Hilo desacoplado)
    # --------------------------------------------------------------------------
    def _planning_loop(self):
        last_start = 0.0
        while not self._stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_rgb
                frame_stamp = self._latest_stamp
                self._latest_rgb = None  # Consumir frame para evitar re-procesamiento

            if frame is None:
                time.sleep(0.02)
                continue

            elapsed_since_last = time.monotonic() - last_start
            if elapsed_since_last < self.planning_min_period_s:
                time.sleep(self.planning_min_period_s - elapsed_since_last)

            last_start = time.monotonic()
            try:
                self._run_planning(frame, frame_stamp)
            except Exception as exc:
                self.get_logger().error(f"Fallo en iteración de planificación BEV: {exc}", throttle_duration_sec=5.0)

    def _compute_relative_goal(self) -> tuple[float, float]:
        """
        Calcula la posición de la meta en coordenadas relativas de GeNIE (x_right, y_forward) en metros.

        Fórmula y Derivación:
        1. Distancia geodésica 'd' vía Haversine.
        2. Bearing absoluto hacia la meta (0=Norte, 90=Este, en sentido horario).
        3. Error angular: theta = bearing - current_heading (en [-180, 180] deg).
           - theta > 0: meta a la derecha del rover.
           - theta < 0: meta a la izquierda del rover.
        4. Descomposición polar en marco GeNIE (+X derecha, +Y adelante):
           - goal_x_m (derecha)   = d * sin(theta)
           - goal_y_m (adelante)  = d * cos(theta)
        """
        if (
            self._current_lat is None
            or self._current_lon is None
            or self._current_heading is None
            or self._target_lat is None
            or self._target_lon is None
        ):
            # Fallback seguro: si aún no hay GPS o meta, apuntar recto hacia adelante dentro del horizonte local
            return 0.0, float(self.forward_range)

        dist = self.haversine_distance(
            self._current_lat, self._current_lon, self._target_lat, self._target_lon
        )
        bearing = self.calculate_bearing(
            self._current_lat, self._current_lon, self._target_lat, self._target_lon
        )
        heading_error = self.angle_error_deg(bearing, self._current_heading)
        theta = math.radians(heading_error)

        goal_x_m = dist * math.sin(theta)
        goal_y_m = dist * math.cos(theta)
        return float(goal_x_m), float(goal_y_m)

    def _compute_global_subgoal_base_link(self) -> tuple[float, float] | None:
        """
        Calcula la sub-meta proyectada desde el path global en coordenadas relativas de GeNIE (x_right, y_forward) en metros.

        Transformación geométrica y convención de signos:
        1. Lookup TF map -> base_link: pose actual del robot (tx, ty) con yaw theta (ENU).
        2. Búsqueda del punto del path más cercano a (tx, ty).
        3. Acumulación de distancia métrica hacia adelante a lo largo del path hasta global_lookahead_distance_m.
        4. Transformación inversa map -> base_link (REP-103: +X adelante, +Y izquierda):
             dx = p_x - tx
             dy = p_y - ty
             x_base_link (adelante)   =  dx * cos(theta) + dy * sin(theta)
             y_base_link (izquierda)  = -dx * sin(theta) + dy * cos(theta)
        5. Conversión a marco GeNIE (+X derecha, +Y adelante):
             goal_x_genie = -y_base_link =  dx * sin(theta) - dy * cos(theta)
             goal_y_genie =  x_base_link =  dx * cos(theta) + dy * sin(theta)

        Ejemplo numérico de verificación con theta = 45 deg (cos=sin=sqrt(2)/2 ≈ 0.7071):
          Robot en (tx=0, ty=0) orientado a 45 deg en map.
          - Punto 2.0m adelante (en map a 45 deg: dx=+1.4142, dy=+1.4142):
              x_base_link = 1.4142*0.7071 + 1.4142*0.7071 = +2.0m
              y_base_link = -1.4142*0.7071 + 1.4142*0.7071 = 0.0m
              -> goal_x_genie = 0.0m, goal_y_genie = +2.0m (adelante)
          - Punto 2.0m a la izquierda (en map a 135 deg: dx=-1.4142, dy=+1.4142):
              x_base_link = -1.4142*0.7071 + 1.4142*0.7071 = 0.0m
              y_base_link = -(-1.4142)*0.7071 + 1.4142*0.7071 = +2.0m
              -> goal_x_genie = -2.0m (izquierda), goal_y_genie = 0.0m
        """
        with self._global_lock:
            poses = list(self._global_path_poses)

        if len(poses) < 2:
            return None

        try:
            # Nota de threading: tf2_ros.Buffer es thread-safe y soporta lookup_transform concurrente desde este hilo
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"No se pudo obtener TF {self.map_frame} -> {self.base_frame} para sub-meta global: {exc}",
                throttle_duration_sec=3.0,
            )
            return None

        tx = float(tf_msg.transform.translation.x)
        ty = float(tf_msg.transform.translation.y)
        qx = float(tf_msg.transform.rotation.x)
        qy = float(tf_msg.transform.rotation.y)
        qz = float(tf_msg.transform.rotation.z)
        qw = float(tf_msg.transform.rotation.w)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 1. Encontrar el punto más cercano en el path global
        dists_sq = [(px - tx) ** 2 + (py - ty) ** 2 for px, py in poses]
        closest_idx = int(np.argmin(dists_sq))

        # 2. Acumular distancia métrica hacia adelante a lo largo del path
        accum_dist = 0.0
        target_px, target_py = poses[-1]
        prev_x, prev_y = poses[closest_idx]

        for i in range(closest_idx, len(poses)):
            px, py = poses[i]
            accum_dist += math.hypot(px - prev_x, py - prev_y)
            prev_x, prev_y = px, py
            if accum_dist >= self.global_lookahead_distance_m:
                target_px, target_py = px, py
                break

        # 3. Transformación inversa map -> base_link
        dx = target_px - tx
        dy = target_py - ty

        x_base_link = dx * cos_yaw + dy * sin_yaw
        y_base_link = -dx * sin_yaw + dy * cos_yaw

        # 4. Mapeo a convención GeNIE (+X derecha, +Y adelante)
        goal_x_genie = -y_base_link
        goal_y_genie = x_base_link

        # 5. Monitoreo de coherencia entre fuentes de localización (Fase 4.B)
        if (
            self._current_lat is not None
            and self._current_lon is not None
            and self._current_heading is not None
            and self._target_lat is not None
            and self._target_lon is not None
        ):
            bearing_gps = self.calculate_bearing(
                self._current_lat, self._current_lon, self._target_lat, self._target_lon
            )
            err_gps = self.angle_error_deg(bearing_gps, self._current_heading)
            angle_subgoal = math.degrees(math.atan2(goal_x_genie, goal_y_genie))
            diff_sources = abs(self.angle_error_deg(err_gps, angle_subgoal))
            self.get_logger().debug(
                f"Comparación de rumbo a meta: GPS crudo={err_gps:+.1f}° vs Sub-meta Global={angle_subgoal:+.1f}° | Diff={diff_sources:.1f}°",
                throttle_duration_sec=3.0,
            )

        return float(goal_x_genie), float(goal_y_genie)

    def _run_planning(self, rgb: np.ndarray, frame_stamp=None):
        t_start = time.perf_counter()

        # 1. Inferencia neuronal de transitabilidad sobre la imagen frontal
        predict_res = self._predictor.predict(rgb)
        t_after_infer = time.perf_counter()
        score_mask = predict_res.mask.astype(np.float32)

        # 2. Proyección de la máscara de transitabilidad a vista aérea (BEV)
        bev_flat, observed, _stats = self._project_score_to_bev(
            score_map=score_mask,
            camera_k=self._camera_k,
            camera_pose=self._camera_t,
            ground_z=self.ground_z,
            bev_resolution_m_per_px=self.bev_resolution,
            bev_forward_range_m=self.forward_range,
            bev_side_range_m=self.side_range,
            max_ray_distance_m=self.max_ray_distance,
        )
        t_after_bev = time.perf_counter()

        # 2b. Publicación de la grilla BEV local cruda en nav_msgs/OccupancyGrid (marco 'base_link')
        bev_h, bev_w = bev_flat.shape
        cost_bev = np.where(
            observed > 0,
            np.clip(np.round((1.0 - bev_flat) * 100.0), 0, 100).astype(np.int8),
            -1,
        )
        grid_2d = cost_bev[::-1, ::-1].T

        local_grid_msg = OccupancyGrid()
        if frame_stamp is not None:
            local_grid_msg.header.stamp = frame_stamp
            stamp_sec = float(frame_stamp.sec) + float(frame_stamp.nanosec) * 1e-9
        else:
            local_grid_msg.header.stamp = self.get_clock().now().to_msg()
            stamp_sec = float(local_grid_msg.header.stamp.sec) + float(local_grid_msg.header.stamp.nanosec) * 1e-9

        local_grid_msg.header.frame_id = "base_link"
        local_grid_msg.info.resolution = float(self.bev_resolution)
        local_grid_msg.info.width = int(bev_h)
        local_grid_msg.info.height = int(bev_w)
        local_grid_msg.info.origin.position.x = 0.0
        local_grid_msg.info.origin.position.y = -float(bev_w // 2) * float(self.bev_resolution)
        local_grid_msg.info.origin.position.z = 0.0
        local_grid_msg.info.origin.orientation.w = 1.0
        local_grid_msg.data = grid_2d.flatten().tolist()
        self.local_grid_pub.publish(local_grid_msg)

        # 3. Cálculo de la meta relativa (x_right, y_forward) con lectura atómica bajo _global_lock
        use_subgoal = False
        with self._global_lock:
            if (
                self.use_global_path_guidance
                and self._global_path_last_update is not None
                and (self.get_clock().now() - self._global_path_last_update).nanoseconds / 1e9 <= self.global_path_max_stale_s
                and self._global_path_valid
                and len(self._global_path_poses) >= 2
            ):
                use_subgoal = True

        if use_subgoal:
            sub_goal = self._compute_global_subgoal_base_link()
            goal_x_m, goal_y_m = (
                sub_goal if sub_goal is not None else self._compute_relative_goal()
            )
        else:
            goal_x_m, goal_y_m = self._compute_relative_goal()

        # 4. Planificación del camino sobre la grilla de costos BEV
        planned = self._plan_on_bev(
            bev_traversability=bev_flat,
            observed_mask=observed,
            goal_x_m=goal_x_m,
            goal_y_m=goal_y_m,
            bev_resolution_m=self.bev_resolution,
            config=self._planner_cfg,
            candidate_path_bank=self._candidate_path_bank,
        )
        t_after_plan = time.perf_counter()

        t_infer_ms = (t_after_infer - t_start) * 1000.0
        t_bev_ms = (t_after_bev - t_after_infer) * 1000.0
        t_plan_ms = (t_after_plan - t_after_bev) * 1000.0
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        now = self.get_clock().now()
        is_valid = bool(
            planned.final_path_xy_m is not None
            and planned.final_path_xy_m.shape[0] > 0
            and planned.metadata.get("status") == "ok"
        )

        self.get_logger().info(
            f"[TRACE][BEV] img_stamp={stamp_sec:.3f}s | infer={t_infer_ms:.1f}ms | bev={t_bev_ms:.1f}ms | plan={t_plan_ms:.1f}ms | total={t_total_ms:.1f}ms | valid={is_valid}"
        )

        # 5. Publicación del estado de validez
        valid_msg = Bool()
        valid_msg.data = is_valid
        self.valid_pub.publish(valid_msg)

        # 6. Publicación del camino planificado en nav_msgs/Path (marco 'base_link')
        path_msg = Path()
        path_msg.header.stamp = now.to_msg()
        path_msg.header.frame_id = "base_link"

        if is_valid:
            for pt in planned.final_path_xy_m:
                x_genie_right = float(pt[0])
                y_genie_forward = float(pt[1])
                x_ros, y_ros = genie_xy_to_ros_base_link(x_genie_right, y_genie_forward)

                pose_stamped = PoseStamped()
                pose_stamped.header = path_msg.header
                pose_stamped.pose.position.x = x_ros
                pose_stamped.pose.position.y = y_ros
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.w = 1.0
                path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

        # 7. Publicación de imagen de depuración si está habilitada
        if self.vis_pub is not None and isinstance(planned.visualization, np.ndarray):
            vis_bgr = planned.visualization[:, :, ::-1]  # RGB a BGR para OpenCV / cv_bridge
            vis_msg = self.bridge.cv2_to_imgmsg(vis_bgr, encoding="bgr8")
            vis_msg.header.stamp = now.to_msg()
            vis_msg.header.frame_id = "base_link"
            self.vis_pub.publish(vis_msg)

        t_end = time.perf_counter()
        infer_ms = predict_res.inference_s * 1000.0
        bev_ms = (t_after_bev - t_after_infer) * 1000.0
        plan_ms = (t_after_plan - t_after_bev) * 1000.0
        total_latency_ms = (t_end - t_start) * 1000.0

        # 8. Gobernador Dinámico de Velocidad basado en Latencia Real P95 (Brief 9 / I.2)
        self._cycle_times_ms.append(total_latency_ms)
        # Usamos el percentil 95 de la ventana móvil para reaccionar inmediatamente a picos de latencia
        t_plan_p95_s = float(np.percentile(list(self._cycle_times_ms), 95)) / 1000.0

        b_term = t_plan_p95_s + self.rtt_delay + self.transport_delay
        discrim = (b_term ** 2) + (2.0 * float(self.forward_range)) / (self.horizon_safety_margin * self.brake_accel)
        if discrim > 0.0:
            v_safe = self.brake_accel * (math.sqrt(discrim) - b_term)
        else:
            v_safe = 0.0

        # Piso de velocidad efectiva de arranque de tracción (0.15 m/s): si v_safe cae por debajo,
        # detenerse por seguridad (Stop & Wait) en vez de comandar velocidades inoperantes
        min_effective_speed = 0.15
        if v_safe < min_effective_speed:
            v_safe = 0.0
        v_safe = float(v_safe)

        # Guarda de evasión dinámica de peatones/tráfico (I.2.2)
        dynamic_traffic_safe = bool(t_plan_p95_s <= 1.0)
        if not dynamic_traffic_safe:
            self.get_logger().warn(
                f"Latencia de ciclo alta ({t_plan_p95_s:.2f}s > 1.0s). "
                "Evasión de obstáculos dinámicos/peatones no garantizada a esta tasa.",
                throttle_duration_sec=5.0,
            )

        v_limit_msg = Float32()
        v_limit_msg.data = v_safe
        self.safe_vel_pub.publish(v_limit_msg)

        alive_paths = int(planned.metadata.get("filtered_paths", len(planned.filtered_paths) if planned.filtered_paths is not None else 0))
        alive_msg = Int32()
        alive_msg.data = alive_paths
        self.alive_paths_pub.publish(alive_msg)
        diag_data = {
            "t_plan_p95_ms": round(t_plan_p95_s * 1000.0, 1),
            "infer_nn_ms": round(infer_ms, 1),
            "bev_proj_ms": round(bev_ms, 1),
            "plan_genie_ms": round(plan_ms, 1),
            "total_ms": round(total_latency_ms, 1),
            "alive_paths": alive_paths,
            "v_safe_limit_mps": round(v_safe, 3),
            "forward_range_m": round(float(self.forward_range), 2),
            "dynamic_traffic_safe": dynamic_traffic_safe,
            "valid": is_valid,
        }
        diag_msg = String()
        diag_msg.data = json.dumps(diag_data)
        self.planner_diag_pub.publish(diag_msg)

        self.get_logger().info(
            f"[LATENCY] frame_total={total_latency_ms:.1f}ms (P95={t_plan_p95_s*1000.0:.1f}ms) | "
            f"infer_nn={infer_ms:.1f}ms | bev_proj={bev_ms:.1f}ms | plan_genie={plan_ms:.1f}ms | "
            f"alive_paths={alive_paths} | v_safe={v_safe:.2f}m/s | valid={is_valid} points={len(path_msg.poses)}"
        )

    def destroy_node(self):
        self._stop_event.set()
        self._infer_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BEVPlannerNode()
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
