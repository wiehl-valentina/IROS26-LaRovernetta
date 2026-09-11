#!/usr/bin/env python3
"""
Persistent Map Node for Earth Rover (IROS 2026 / FrodoBots).

Acumula las grillas locales BEV ('earth_rover/local_bev_grid') en un mapa global
persistente ('earth_rover/persistent_map') en el marco 'map' (REP-105) con
lógica de confianza Bayesiana acumulativa y decaimiento exponencial periódico.
"""

from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
import tf2_ros


class PersistentMapNode(Node):
    def __init__(self):
        super().__init__("persistent_map_node")

        self.declare_parameter("local_grid_topic", "earth_rover/local_bev_grid")
        self.declare_parameter("map_topic", "earth_rover/persistent_map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("map_width_m", 400.0)
        self.declare_parameter("map_height_m", 400.0)
        self.declare_parameter("map_resolution_m_per_px", 0.20)
        self.declare_parameter("map_origin_x_m", -200.0)
        self.declare_parameter("map_origin_y_m", -200.0)
        self.declare_parameter("hit_gain", 15.0)
        self.declare_parameter("miss_gain", 10.0)
        self.declare_parameter("confidence_max", 100.0)
        self.declare_parameter("hit_vote_threshold", 0.15)
        self.declare_parameter("unknown_evidence_band", 5.0)
        self.declare_parameter("publish_quantization_step", 5)
        self.declare_parameter("occupied_threshold", 55.0)
        self.declare_parameter("free_threshold", -20.0)
        self.declare_parameter("occupied_cost_cutoff", 50.0)
        self.declare_parameter("decay_period_s", 5.0)
        self.declare_parameter("decay_factor", 0.7738)
        self.declare_parameter("map_publish_period_s", 1.0)
        self.declare_parameter("tf_lookup_timeout_s", 0.2)
        self.declare_parameter("seed_map_path", "")
        self.declare_parameter("seed_confidence_scale", 0.3)
        self.declare_parameter("semantic_layer_enabled", True)
        self.declare_parameter("semantic_override_threshold", 30.0)

        local_grid_topic = str(self.get_parameter("local_grid_topic").value)
        map_topic = str(self.get_parameter("map_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)

        self.map_width_m = float(self.get_parameter("map_width_m").value)
        self.map_height_m = float(self.get_parameter("map_height_m").value)
        self.map_resolution = float(self.get_parameter("map_resolution_m_per_px").value)
        self.map_origin_x = float(self.get_parameter("map_origin_x_m").value)
        self.map_origin_y = float(self.get_parameter("map_origin_y_m").value)

        self.hit_gain = float(self.get_parameter("hit_gain").value)
        self.miss_gain = float(self.get_parameter("miss_gain").value)
        self.confidence_max = float(self.get_parameter("confidence_max").value)
        self.hit_vote_threshold = float(self.get_parameter("hit_vote_threshold").value)
        self.unknown_evidence_band = float(self.get_parameter("unknown_evidence_band").value)
        self.publish_quantization_step = int(self.get_parameter("publish_quantization_step").value)
        self.occupied_threshold = float(self.get_parameter("occupied_threshold").value)
        self.free_threshold = float(self.get_parameter("free_threshold").value)
        self.occupied_cost_cutoff = float(self.get_parameter("occupied_cost_cutoff").value)

        self.decay_period_s = float(self.get_parameter("decay_period_s").value)
        self.decay_factor = float(self.get_parameter("decay_factor").value)
        self.map_publish_period_s = float(self.get_parameter("map_publish_period_s").value)
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)
        self.seed_map_path = str(self.get_parameter("seed_map_path").value).strip()
        self.seed_confidence_scale = float(self.get_parameter("seed_confidence_scale").value)
        self.semantic_layer_enabled = bool(self.get_parameter("semantic_layer_enabled").value)
        self.semantic_override_threshold = float(self.get_parameter("semantic_override_threshold").value)

        # ----------------------------------------------------------------------
        # 2. Inicialización de Canales de Grilla (Dinámico y Semántico)
        # ----------------------------------------------------------------------
        self.grid_w = max(1, int(round(self.map_width_m / self.map_resolution)))
        self.grid_h = max(1, int(round(self.map_height_m / self.map_resolution)))

        self._lock = threading.Lock()
        # Canal 1: Evidencia dinámica (>0 = obstáculo, <0 = libre, 0 = desconocido).
        # Afectado por observaciones de cámara y decaimiento exponencial.
        self._confidence = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Canal 2: Prior semántico estático (vereda=-24.0, calle=-6.0, neutral=0.0).
        # Inmune al decaimiento temporal y desacoplado de las observaciones locales.
        self._semantic = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Precarga opcional de mapa semilla (OSM prior) en canal semántico
        if self.seed_map_path:
            self._load_seed_map(self.seed_map_path)

        # ----------------------------------------------------------------------
        # 3. Transform Buffer y QoS
        # ----------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(OccupancyGrid, local_grid_topic, self._on_local_grid, sensor_qos)
        self.map_pub = self.create_publisher(OccupancyGrid, map_topic, map_qos)

        # ----------------------------------------------------------------------
        # 4. Temporizadores de Decaimiento y Publicación
        # ----------------------------------------------------------------------
        self.decay_timer = self.create_timer(self.decay_period_s, self._decay_timer_cb)
        self.publish_timer = self.create_timer(self.map_publish_period_s, self._publish_timer_cb)

        self.get_logger().info(
            f"PersistentMapNode inicializado | Mapa: {self.map_width_m}x{self.map_height_m}m "
            f"({self.grid_w}x{self.grid_h} px @ {self.map_resolution}m/px) | Frame: {self.map_frame}"
        )

    def _load_seed_map(self, path: str) -> None:
        """
        Carga un mapa semilla (.npy) como prior semántico estático en self._semantic.

        Lógica de Carga y Validación:
          1. Fail Open: Si el archivo no existe o falla la lectura, se loggea WARNING y se continúa con grilla semántica en 0.
          2. Validación de Shape: Si las dimensiones no coinciden exactamente con (grid_h, grid_w),
             se loggea ERROR y se descarta (se inicia en 0) para evitar desalineación espacial.
          3. Escalado de Confianza: self._semantic = np.clip(seed_array * self.seed_confidence_scale, -self.confidence_max, self.confidence_max).
             self._confidence permanece en 0.0 (canal de evidencia dinámica desacoplado).

        Ejemplo Numérico:
          - Vereda generada por OSM: raw_seed = -80.0.
          - Con seed_confidence_scale = 0.3:
              prior_semantico = -80.0 * 0.3 = -24.0.
          - Calle generada por OSM: raw_seed = -20.0 -> prior_semantico = -20.0 * 0.3 = -6.0.
          - Neutral / Desconocido: raw_seed = 0.0 -> prior_semantico = 0.0.
        """
        if not os.path.isfile(path):
            self.get_logger().warn(
                f"Mapa semilla no encontrado en '{path}'. "
                "Iniciando con capa semántica en cero (fail open)."
            )
            return

        try:
            seed_array = np.load(path)
        except Exception as exc:
            self.get_logger().warn(
                f"Fallo al cargar archivo de mapa semilla '{path}': {exc}. "
                "Iniciando con capa semántica en cero (fail open)."
            )
            return

        expected_shape = (self.grid_h, self.grid_w)
        if seed_array.shape != expected_shape:
            self.get_logger().error(
                f"Dimensiones del mapa semilla {seed_array.shape} no coinciden con la grilla "
                f"esperada {expected_shape}. Mapa semilla descartado (iniciando en cero)."
            )
            return

        scaled_seed = (seed_array * self.seed_confidence_scale).astype(np.float32)
        np.clip(scaled_seed, -self.confidence_max, self.confidence_max, out=scaled_seed)
        with self._lock:
            self._semantic = scaled_seed

        n_loaded_free = int(np.count_nonzero(self._semantic <= self.free_threshold))
        n_loaded_occ = int(np.count_nonzero(self._semantic >= self.occupied_threshold))
        self.get_logger().info(
            f"Mapa semilla precargado exitosamente en capa semántica desde '{path}' "
            f"(escala={self.seed_confidence_scale:.2f}, celdas_libres={n_loaded_free}, celdas_ocupadas={n_loaded_occ})."
        )

    # --------------------------------------------------------------------------
    # Callback de Grilla BEV Local
    # --------------------------------------------------------------------------
    def _on_local_grid(self, msg: OccupancyGrid):
        """
        Integra la grilla BEV local recibida en el mapa global usando TF map -> base_link.

        Transformación de Coordenadas y Convención de Signos:
          Sea p_local = [x_local, y_local] la posición métrica de una celda en base_link.
          La pose del robot en el frame 'map' obtenida vía TF es (tx, ty) con rotación yaw theta:
            x_map = tx + x_local * cos(theta) - y_local * sin(theta)
            y_map = ty + x_local * sin(theta) + y_local * cos(theta)

          Ejemplo numérico de verificación:
            Robot en (tx=10.0m, ty=20.0m), orientado al Norte (theta = +90 deg, cos=0, sin=1):
            - Celda 2m adelante (x_local=+2.0, y_local=0.0):
                x_map = 10.0 + 2.0*0 - 0*1 = 10.0m
                y_map = 20.0 + 2.0*1 + 0*0 = 22.0m (2m al Norte en 'map')
            - Celda 2m a la izquierda (x_local=0.0, y_local=+2.0):
                x_map = 10.0 + 0*0 - 2.0*1 = 8.0m (2m al Oeste en 'map')
                y_map = 20.0 + 0*1 + 2.0*0 = 20.0m

        Lógica Bayesiana de Confianza y Ejemplos Numéricos:
          hit_gain = 15.0, miss_gain = 10.0, occupied_threshold = 55.0, free_threshold = -20.0
          - Acumulación de Obstáculo:
              Celda arranca en confidence = 0.0 (desconocido).
              Frame 1: +15 -> 15.0 (desconocido)
              Frame 2: +15 -> 30.0 (desconocido)
              Frame 3: +15 -> 45.0 (desconocido)
              Frame 4: +15 -> 60.0 >= 55.0 -> Supera occupied_threshold -> Pasa a OCUPADA (100).
              -> Requiere ceil(55/15) = 4 observaciones consecutivas para marcar obstáculo.
          - Acumulación de Terreno Libre:
              Celda arranca en confidence = 0.0 (desconocido).
              Frame 1: -10 -> -10.0 (desconocido)
              Frame 2: -10 -> -20.0 <= -20.0 -> Pasa a LIBRE (0).
              -> Requiere ceil(20/10) = 2 observaciones consecutivas para confirmar libre.
        """
        t_start = time.perf_counter()
        # 1. Lookup de la transformación map -> base_link
        target_frame = self.map_frame
        source_frame = msg.header.frame_id if msg.header.frame_id else "base_link"

        try:
            lookup_stamp = Time.from_msg(msg.header.stamp)
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                lookup_stamp,
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except Exception as exc:
            # Fallback seguro con tiempo cero si la interpolación por timestamp exacto falla
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=self.tf_lookup_timeout_s),
                )
            except Exception as exc2:
                self.get_logger().warn(
                    f"No se pudo obtener TF {target_frame} -> {source_frame}: {exc2}",
                    throttle_duration_sec=3.0,
                )
                return

        # 2. Extracción de traslación y ángulo de rumbo (yaw) desde el cuaternión TF
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

        # 3. Vectorización de celdas observadas
        local_w = int(msg.info.width)
        local_h = int(msg.info.height)
        local_res = float(msg.info.resolution)
        orig_x = float(msg.info.origin.position.x)
        orig_y = float(msg.info.origin.position.y)

        raw_data = np.asarray(msg.data, dtype=np.int16).reshape((local_h, local_w))
        obs_rows, obs_cols = np.where(raw_data != -1)
        if obs_rows.size == 0:
            return

        # Coordenadas locales en metros (centro del píxel local)
        x_local = orig_x + (obs_cols.astype(np.float64) + 0.5) * local_res
        y_local = orig_y + (obs_rows.astype(np.float64) + 0.5) * local_res

        # 4. Proyección rígida 2D al marco global 'map'
        x_map = tx + x_local * cos_yaw - y_local * sin_yaw
        y_map = ty + x_local * sin_yaw + y_local * cos_yaw

        # 5. Mapeo a índices discretos de la grilla persistente
        map_cols = np.floor((x_map - self.map_origin_x) / self.map_resolution).astype(np.int32)
        map_rows = np.floor((y_map - self.map_origin_y) / self.map_resolution).astype(np.int32)

        # 6. Filtrado de límites del mapa
        in_bounds = (
            (map_cols >= 0)
            & (map_cols < self.grid_w)
            & (map_rows >= 0)
            & (map_rows < self.grid_h)
        )
        dropped_cells = int(np.count_nonzero(~in_bounds))
        if dropped_cells > 0:
            self.get_logger().warn(
                f"{dropped_cells}/{obs_rows.size} celdas BEV cayeron fuera de los límites del mapa persistente.",
                throttle_duration_sec=5.0,
            )

        valid_cols = map_cols[in_bounds]
        valid_rows = map_rows[in_bounds]
        valid_costs = raw_data[obs_rows[in_bounds], obs_cols[in_bounds]]

        if valid_rows.size == 0:
            return

        # 7. Agrupamiento y Voto Agregado por Celda Persistente (Mitigación de Saturación por Resolución)
        # Ratio de resolución (0.20m persistente / 0.03m local)^2 ≈ 44 celdas locales por píxel persistente.
        # Agrupamos las celdas locales que caen sobre la misma celda persistente y emitimos UN SOLO voto por frame.
        stacked = np.stack((valid_rows, valid_cols), axis=1)
        unique_cells, inverse_indices = np.unique(stacked, axis=0, return_inverse=True)
        uniq_rows = unique_cells[:, 0]
        uniq_cols = unique_cells[:, 1]
        num_unique = len(unique_cells)

        is_hit = valid_costs >= self.occupied_cost_cutoff
        count_total = np.bincount(inverse_indices, minlength=num_unique)
        count_hit = np.bincount(inverse_indices, weights=is_hit.astype(np.int32), minlength=num_unique)
        hit_fraction = count_hit / np.maximum(1, count_total)

        # Regla de decisión de voto:
        # hit_vote_threshold = 0.15 (15% de evidencia de obstáculo en las observaciones locales).
        # En el borde exacto hit_fraction == hit_vote_threshold se decide a favor de HIT (conservador por seguridad).
        # Justificación: con un rover de 250mm de ancho, un obstáculo fino (ej. poste) que ocupe 3-4 de 44 celdas
        # locales (~7-9%) o ligeramente más (>=15%) debe registrarse como obstáculo para evitar colisiones,
        # en concordancia con el threshold_points_ratio: 0.05 del planificador local.
        is_hit_unique = hit_fraction >= self.hit_vote_threshold
        deltas_unique = np.where(is_hit_unique, self.hit_gain, -self.miss_gain).astype(np.float32)

        with self._lock:
            # Los índices son únicos por construcción mediante np.unique, fancy indexing directo es seguro
            self._confidence[uniq_rows, uniq_cols] += deltas_unique
            np.clip(self._confidence, -self.confidence_max, self.confidence_max, out=self._confidence)

        t_int_ms = (time.perf_counter() - t_start) * 1000.0
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.get_logger().info(
            f"[TRACE][MAP_INTEGRATE] grid_stamp={stamp_sec:.3f}s | duration={t_int_ms:.1f}ms | unique_cells={num_unique}"
        )

    # --------------------------------------------------------------------------
    # Temporizador de Decaimiento Exponencial
    # --------------------------------------------------------------------------
    def _decay_timer_cb(self):
        """
        Aplica decaimiento exponencial periódico EXCLUSIVAMENTE a la grilla de evidencia dinámica (self._confidence).
        La grilla semántica estática (self._semantic) NO sufre decaimiento.

        Fórmula Matemática:
          confidence[t] = confidence[t - 1] * decay_factor

        Ejemplo Numérico y Cálculo de Tiempo de Olvido:
          Con decay_factor = 0.7738 y decay_period_s = 5.0s (equivalente a 0.95^5 cada 5s):
          - Caso 1: Celda en umbral ocupado (confidence = 55.0).
              t=5s: 55.0 * 0.7738 = 42.56 < 55.0 -> Deja de considerarse obstáculo en 1 período (5s).
          - Caso 2: Obstáculo persistente saturado al máximo (confidence = 100.0).
              Buscamos k períodos para que 100.0 * (0.7738)^k < 55.0:
                k > ln(0.55) / ln(0.7738) = (-0.597837) / (-0.256441) = 2.331 períodos.
                Tiempo efectivo = 2.331 * 5.0s = 11.66 segundos (~12s).
        """
        with self._lock:
            self._confidence *= np.float32(self.decay_factor)

    # --------------------------------------------------------------------------
    # Temporizador de Publicación del Mapa Persistente
    # --------------------------------------------------------------------------
    def _publish_timer_cb(self):
        """
        Publica la grilla persistente como nav_msgs/OccupancyGrid graduado en el marco 'map'.

        Fusión de Canales (Evidencia Dinámica + Prior Semántico Estático):
        ------------------------------------------------------------------
        Se combinan dos fuentes de información ortogonales:
          1. self._confidence: Evidencia dinámica observada por sensores locales (SAM-TP).
             Sujeta a decaimiento exponencial para olvidar obstáculos móviles.
          2. self._semantic: Prior topológico / semántico de OpenStreetMap (vereda=-24.0, calle=-6.0).
             Inmune al decaimiento temporal y desacoplado de las observaciones locales.

        Fórmula de Combinación (Parte B):
        ---------------------------------
        Sean C = conf_copy (evidencia dinámica) y S = sem_copy (prior semántico estático).
        Si semantic_layer_enabled es True:
          - Condición de Override de Obstáculo: has_obstacle = (C >= semantic_override_threshold)
          - Condición de Prior Semántico:       has_semantic = (|S| >= unknown_evidence_band)

          grid_effective = np.where(
              has_obstacle,
              np.maximum(C, S),               # Evidencia suficiente de obstáculo domina incondicionalmente
              np.where(has_semantic, S, C)    # Sin obstáculo suficiente: prior semántico rige; si neutral, usa C
          )
        Si semantic_layer_enabled es False:
          grid_effective = conf_copy

        Derivación Numérica y Validación de Casos (global_planner_node._on_map):
        -------------------------------------------------------------------------
        Parámetros del contrato:
          - confidence_max = 100.0, unknown_evidence_band = 5.0, semantic_override_threshold = 30.0, quantization_step = 5
          - free_ref_value = 40, occupied_ref_value = 78, unknown_cell_cost = 2.5, max_finite_cost = 15.0
          - scaled = ((grid_effective + 100) / 200) * 100.0
          - occ_grid = round(scaled / 5) * 5 (con occ_grid[|grid_effective| < 5.0] = -1)
          - En global_planner:
              norm = clip((raw_data - 40) / 38, 0.0, 1.0)
              cost = 1.0 + norm * 14.0 (o inf si raw_data >= 78, o 2.5 si raw_data == -1)

        1. Vereda de OSM, sin evidencia de cámara (C=0.0, S=-24.0):
           - has_obstacle = False (0.0 < 30.0), has_semantic = True (|-24.0| >= 5.0).
           - grid_effective = -24.0.
           - scaled = ((-24.0 + 100) / 200) * 100 = 38.0 -> occ_grid = round(38/5)*5 = 40.
           - global_planner: raw_data = 40 <= 40 -> norm = 0.0 -> cost = 1.0 + 0.0 * 14.0 = 1.00.

        2. Calle de OSM, sin evidencia de cámara (C=0.0, S=-6.0):
           - has_obstacle = False (0.0 < 30.0), has_semantic = True (|-6.0| >= 5.0).
           - grid_effective = -6.0.
           - scaled = ((-6.0 + 100) / 200) * 100 = 47.0 -> occ_grid = round(47/5)*5 = 45.
           - global_planner: raw_data = 45 -> norm = (45 - 40) / 38 = 5/38 = 0.13158 -> cost = 1.0 + 0.13158 * 14.0 = 2.8421 ≈ 2.84.

        3. Sin prior ni evidencia (C=0.0, S=0.0):
           - has_obstacle = False, has_semantic = False.
           - grid_effective = 0.0 -> |0.0| < 5.0 (banda desconocida) -> occ_grid = -1.
           - global_planner: raw_data = -1 -> cost = unknown_cell_cost = 2.50.

        4. Obstáculo confirmado por cámara (C=55.0, S=0.0):
           - has_obstacle = True (55.0 >= 30.0).
           - grid_effective = max(55.0, 0.0) = 55.0.
           - scaled = ((55.0 + 100) / 200) * 100 = 77.5 -> occ_grid = 80.
           - global_planner: raw_data = 80 >= 78 -> cost = inf.

        5. Vereda con obstáculo confirmado (C=55.0, S=-24.0):
           - has_obstacle = True (55.0 >= 30.0).
           - grid_effective = max(55.0, -24.0) = 55.0.
           - scaled = 77.5 -> occ_grid = 80.
           - global_planner: raw_data = 80 >= 78 -> cost = inf (NO se diluye por ser vereda).
        """
        t_pub_start = time.perf_counter()
        with self._lock:
            conf_copy = self._confidence.copy()
            sem_copy = self._semantic.copy()

        if self.semantic_layer_enabled:
            has_obstacle = conf_copy >= self.semantic_override_threshold
            has_semantic = np.abs(sem_copy) >= self.unknown_evidence_band
            grid_effective = np.where(
                has_obstacle,
                np.maximum(conf_copy, sem_copy),
                np.where(has_semantic, sem_copy, conf_copy),
            )
        else:
            grid_effective = conf_copy

        abs_conf = np.abs(grid_effective)
        unknown_mask = abs_conf < self.unknown_evidence_band

        scaled = np.clip(
            ((grid_effective + self.confidence_max) / (2.0 * self.confidence_max)) * 100.0,
            0.0,
            100.0,
        )
        step = max(1, int(self.publish_quantization_step))
        occ_grid = (np.round(np.round(scaled, 6) / step) * step).astype(np.int8)
        occ_grid[unknown_mask] = -1

        # Diagnóstico estadístico informativo throttled
        n_occupied = int(np.count_nonzero(grid_effective >= self.occupied_threshold))
        n_free = int(np.count_nonzero(grid_effective <= self.free_threshold))
        n_partial = int(
            np.count_nonzero(
                (grid_effective > self.free_threshold)
                & (grid_effective < self.occupied_threshold)
                & ~unknown_mask
            )
        )
        self.get_logger().debug(
            f"PersistentMap stats: ocupadas={n_occupied}, libres={n_free}, evidencia_parcial={n_partial}",
            throttle_duration_sec=self.map_publish_period_s * 5.0,
        )

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        msg.info.resolution = float(self.map_resolution)
        msg.info.width = int(self.grid_w)
        msg.info.height = int(self.grid_h)

        msg.info.origin.position.x = float(self.map_origin_x)
        msg.info.origin.position.y = float(self.map_origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = occ_grid.flatten().tolist()
        self.map_pub.publish(msg)

        t_pub_ms = (time.perf_counter() - t_pub_start) * 1000.0
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.get_logger().info(
            f"[TRACE][MAP_PUB] stamp={stamp_sec:.3f}s | duration={t_pub_ms:.1f}ms | occ_cells={n_occupied}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PersistentMapNode()
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
