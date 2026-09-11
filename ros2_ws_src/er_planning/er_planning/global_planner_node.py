#!/usr/bin/env python3
"""
Global Path Planner Node (D* Lite) for Earth Rover (IROS 2026 / FrodoBots).

Implementa el algoritmo de búsqueda incremental D* Lite (Koenig & Likhachev, 2002/2005)
sobre el mapa persistente global ('earth_rover/persistent_map'), reparando eficientemente
el camino ante cambios dinámicos del entorno y movimiento del rover hacia el checkpoint objetivo.
"""

from __future__ import annotations

import heapq
import math
import threading
import time

import numpy as np
import rclpy
import scipy.ndimage
from geographic_msgs.msg import GeoPoint
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from robot_localization.srv import FromLL
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
import tf2_ros


class GlobalPlannerNode(Node):
    def __init__(self):
        super().__init__("global_planner_node")

        # ----------------------------------------------------------------------
        # 1. Declaración y Extracción de Parámetros
        # ----------------------------------------------------------------------
        self.declare_parameter("map_topic", "earth_rover/persistent_map")
        self.declare_parameter("target_topic", "earth_rover/target_waypoint")
        self.declare_parameter("global_path_topic", "earth_rover/global_path")
        self.declare_parameter("global_planner_valid_topic", "earth_rover/global_planner_valid")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("from_ll_service", "/fromLL")
        self.declare_parameter("replan_min_period_s", 2.0)
        self.declare_parameter("occupied_ref_value", 78)
        self.declare_parameter("free_ref_value", 40)
        self.declare_parameter("unknown_cell_cost", 2.5)
        self.declare_parameter("max_finite_cost", 15.0)
        self.declare_parameter("cost_change_epsilon", 0.75)
        self.declare_parameter("max_vertex_updates_per_cycle", 20000)
        self.declare_parameter("footprint_inflation_radius_m", 0.15)
        self.declare_parameter("goal_search_radius_m", 13.0)
        self.declare_parameter("connectivity", 8)
        self.declare_parameter("tf_lookup_timeout_s", 0.2)

        map_topic = str(self.get_parameter("map_topic").value)
        target_topic = str(self.get_parameter("target_topic").value)
        global_path_topic = str(self.get_parameter("global_path_topic").value)
        global_planner_valid_topic = str(self.get_parameter("global_planner_valid_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        from_ll_service = str(self.get_parameter("from_ll_service").value)

        self.replan_min_period_s = float(self.get_parameter("replan_min_period_s").value)
        self.occupied_ref_value = int(self.get_parameter("occupied_ref_value").value)
        self.free_ref_value = int(self.get_parameter("free_ref_value").value)
        self.unknown_cell_cost = float(self.get_parameter("unknown_cell_cost").value)
        self.max_finite_cost = float(self.get_parameter("max_finite_cost").value)
        self.cost_change_epsilon = float(self.get_parameter("cost_change_epsilon").value)
        self.max_vertex_updates_per_cycle = int(self.get_parameter("max_vertex_updates_per_cycle").value)
        self.footprint_inflation_radius_m = float(self.get_parameter("footprint_inflation_radius_m").value)
        self.goal_search_radius_m = float(self.get_parameter("goal_search_radius_m").value)
        self.connectivity = int(self.get_parameter("connectivity").value)
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)

        # ----------------------------------------------------------------------
        # 2. Clientes de Servicio, TF y Perfiles QoS
        # ----------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.from_ll_client = self.create_client(FromLL, from_ll_service)

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(OccupancyGrid, map_topic, self._on_map, reliable_qos)
        self.create_subscription(NavSatFix, target_topic, self._on_target, reliable_qos)

        self.path_pub = self.create_publisher(Path, global_path_topic, reliable_qos)
        self.valid_pub = self.create_publisher(Bool, global_planner_valid_topic, reliable_qos)

        # ----------------------------------------------------------------------
        # 3. Estado del Mapa Persistente y Metadatos Geométricos
        # ----------------------------------------------------------------------
        self._lock = threading.Lock()
        self._map_data: np.ndarray | None = None
        self._map_res: float = 0.20
        self._map_w: int = 0
        self._map_h: int = 0
        self._map_orig_x: float = 0.0
        self._map_orig_y: float = 0.0
        self._cell_costs: np.ndarray | None = None  # Matriz 2D de costos discretos

        # Meta geodésica y cartesiana en marco 'map'
        self._current_target_lat: float | None = None
        self._current_target_lon: float | None = None
        self._goal_map_xy: tuple[float, float] | None = None
        self._from_ll_pending: bool = False

        # ----------------------------------------------------------------------
        # 4. Estructuras de Datos de D* Lite (Koenig & Likhachev)
        # ----------------------------------------------------------------------
        # Estados: u = (row, col)
        self._s_start: tuple[int, int] | None = None
        self._s_goal: tuple[int, int] | None = None
        self._s_last: tuple[int, int] | None = None
        self._km: float = 0.0

        # Costos g(u) y rhs(u) almacenados en matrices densas 2D float64 (default = inf)
        self._g: np.ndarray | None = None
        self._rhs: np.ndarray | None = None

        # Banco de direcciones y distancias precomputado para vecinos adyacentes
        self._neighbors_8: tuple[tuple[int, int, float], ...] = ()
        self._precompute_neighbor_offsets()

        # Cola de prioridad U con clave [k1, k2] y versioning para borrado O(1) perezoso
        self._pq_heap: list[tuple[float, float, tuple[int, int], int]] = []
        self._pq_dict: dict[tuple[int, int], tuple[float, float, int]] = {}
        self._pq_entry_id: int = 0

        # Control de Re-planificación periódica (Heartbeat a 0.5s para movimiento del rover)
        self._last_plan_time: float = 0.0
        heartbeat_period_s = min(0.5, self.replan_min_period_s)
        self._replan_timer = self.create_timer(heartbeat_period_s, self._replan_timer_cb)

        self.get_logger().info(
            f"GlobalPlannerNode (D* Lite) inicializado | map={map_topic} | "
            f"target={target_topic} | path={global_path_topic} | replan_period={self.replan_min_period_s}s"
        )

    def _precompute_neighbor_offsets(self):
        """Precomputa las tuplas fijas de (dr, dc, base_dist) para conectividad 4 u 8."""
        diag_dist = math.sqrt(2.0) * self._map_res
        if self.connectivity == 8:
            self._neighbors_8 = (
                (-1, 0, self._map_res),
                (1, 0, self._map_res),
                (0, -1, self._map_res),
                (0, 1, self._map_res),
                (-1, -1, diag_dist),
                (-1, 1, diag_dist),
                (1, -1, diag_dist),
                (1, 1, diag_dist),
            )
        else:
            self._neighbors_8 = (
                (-1, 0, self._map_res),
                (1, 0, self._map_res),
                (0, -1, self._map_res),
                (0, 1, self._map_res),
            )

    # --------------------------------------------------------------------------
    # Callbacks de Sensores y Servicios
    # --------------------------------------------------------------------------
    def _on_target(self, msg: NavSatFix):
        """
        Gestiona la meta geodésica. Si cambia el waypoint, solicita la proyección
        a coordenadas cartesianas 'map' vía el servicio /fromLL de robot_localization.
        """
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if (
            self._goal_map_xy is not None
            and self._current_target_lat is not None
            and self._current_target_lon is not None
            and math.isclose(lat, self._current_target_lat, abs_tol=1e-7)
            and math.isclose(lon, self._current_target_lon, abs_tol=1e-7)
        ):
            return

        if not self.from_ll_client.service_is_ready():
            self.get_logger().warn(
                f"Servicio {self.from_ll_client.srv_name} no disponible aún. Esperando...",
                throttle_duration_sec=3.0,
            )
            return

        self._current_target_lat = lat
        self._current_target_lon = lon

        req = FromLL.Request()
        req.ll_point = GeoPoint(latitude=lat, longitude=lon, altitude=0.0)

        self._from_ll_pending = True
        future = self.from_ll_client.call_async(req)
        future.add_done_callback(self._on_from_ll_response)

    def _on_from_ll_response(self, future):
        self._from_ll_pending = False
        try:
            res = future.result()
            target_x = float(res.map_point.x)
            target_y = float(res.map_point.y)
            with self._lock:
                self._goal_map_xy = (target_x, target_y)
                self.get_logger().info(
                    f"Nueva meta global recibida en coordenadas map: ({target_x:.2f}m, {target_y:.2f}m)"
                )
                # Reinicio completo de D* Lite al cambiar la meta
                self._reset_dstar_lite()
        except Exception as exc:
            self.get_logger().error(f"Fallo en llamada al servicio /fromLL: {exc}")

    def _on_map(self, msg: OccupancyGrid):
        """
        Recibe el mapa persistente acumulado. Detecta celdas cuyo costo cambió
        (por nuevas observaciones o por decaimiento de confianza) y ejecuta
        UpdateVertex sobre ellas para reparar incrementalmente el grafo.

        Mapeo de costos graduado (Fase 2):
          - Desconocido (-1): cost = unknown_cell_cost (2.5)
          - Confirmado libre (<= free_ref_value = 40): cost = 1.0
          - Evidencia parcial (40 < raw < 78): interpolación lineal de 1.0 a max_finite_cost (15.0)
          - Confirmado ocupado (>= occupied_ref_value = 78): cost = inf
          - Inflación de huella (footprint): dilata celdas inf por radio footprint_inflation_radius_m (0.15m).
        """
        t_map_start = time.perf_counter()
        w = int(msg.info.width)
        h = int(msg.info.height)
        res = float(msg.info.resolution)
        orig_x = float(msg.info.origin.position.x)
        orig_y = float(msg.info.origin.position.y)

        raw_data = np.asarray(msg.data, dtype=np.int16).reshape((h, w))

        # 1. Asignación de costos referenciada a la escala graduada (C2)
        new_costs = np.full((h, w), float("inf"), dtype=np.float64)
        known = raw_data != -1
        occupied = known & (raw_data >= self.occupied_ref_value)
        free_ish = known & ~occupied

        new_costs[~known] = float(self.unknown_cell_cost)

        span = max(1.0, float(self.occupied_ref_value - self.free_ref_value))
        norm = np.clip(
            (raw_data[free_ish].astype(np.float64) - self.free_ref_value) / span,
            0.0,
            1.0,
        )
        new_costs[free_ish] = 1.0 + norm * (self.max_finite_cost - 1.0)
        # 'occupied' queda en float("inf") por inicialización

        # 2. Inflación por footprint del rover (Fase 2.C)
        # Radio de inflación del Mini+ (ancho 250mm -> semiancho 0.125m ~ 0.15m con margen)
        inflation_cells = max(1, int(math.ceil(self.footprint_inflation_radius_m / res)))
        occupied_binary = np.isinf(new_costs)
        if np.any(occupied_binary):
            y_grid, x_grid = np.ogrid[
                -inflation_cells : inflation_cells + 1,
                -inflation_cells : inflation_cells + 1,
            ]
            structure = (x_grid * x_grid + y_grid * y_grid) <= (inflation_cells * inflation_cells)
            inflated_binary = scipy.ndimage.binary_dilation(occupied_binary, structure=structure)
            new_costs[inflated_binary] = float("inf")

        num_diff = 0
        with self._lock:
            old_costs = self._cell_costs
            self._map_data = raw_data
            self._map_w = w
            self._map_h = h
            self._map_res = res
            self._map_orig_x = orig_x
            self._map_orig_y = orig_y
            self._cell_costs = new_costs
            self._precompute_neighbor_offsets()

            if self._s_goal is None and self._goal_map_xy is not None:
                self._reset_dstar_lite()

            # Si D* Lite está activo y hubo un mapa previo, reparar los vértices modificados
            # NOTA CRÍTICA: El diff se compara DESPUÉS de la inflación (Fase 2.C).
            if (
                self._s_goal is not None
                and old_costs is not None
                and old_costs.shape == new_costs.shape
            ):
                inf_changed = np.isinf(old_costs) != np.isinf(new_costs)
                finite_changed = (
                    ~np.isinf(old_costs)
                    & ~np.isinf(new_costs)
                    & (np.abs(old_costs - new_costs) > self.cost_change_epsilon)
                )
                diff_mask = inf_changed | finite_changed
                diff_rows, diff_cols = np.where(diff_mask)
                num_diff = int(diff_rows.size)

                if num_diff > 0:
                    if num_diff > self.max_vertex_updates_per_cycle:
                        self.get_logger().warn(
                            f"Diff de mapa superó límite ({num_diff} > {self.max_vertex_updates_per_cycle}). "
                            "Truncando actualizaciones; grafo temporalmente sub-reparado (degradación controlada).",
                            throttle_duration_sec=5.0,
                        )
                        diff_rows = diff_rows[: self.max_vertex_updates_per_cycle]
                        diff_cols = diff_cols[: self.max_vertex_updates_per_cycle]

                    for r, c in zip(diff_rows, diff_cols):
                        u = (int(r), int(c))
                        self._update_vertex(u)
                        for dr, dc, _ in self._neighbors_8:
                            nr, nc = u[0] + dr, u[1] + dc
                            if 0 <= nr < self._map_h and 0 <= nc < self._map_w:
                                self._update_vertex((nr, nc))

        elapsed_map_ms = (time.perf_counter() - t_map_start) * 1000.0
        self.get_logger().debug(
            f"_on_map procesado en {elapsed_map_ms:.1f}ms | celdas modificadas: {num_diff}",
            throttle_duration_sec=2.0,
        )

        # 3. Throttle real de replanificación (2.H)
        now_mono = time.monotonic()
        if (now_mono - self._last_plan_time) >= self.replan_min_period_s:
            self._plan_and_publish()

    # --------------------------------------------------------------------------
    # Motor Algorítmico D* Lite (Koenig & Likhachev)
    # --------------------------------------------------------------------------
    def _reset_dstar_lite(self):
        """Reinicializa el estado completo de D* Lite hacia la meta actual."""
        if self._goal_map_xy is None or self._map_data is None or self._cell_costs is None:
            return

        gx, gy = self._goal_map_xy
        c_goal = int(math.floor((gx - self._map_orig_x) / self._map_res))
        r_goal = int(math.floor((gy - self._map_orig_y) / self._map_res))

        if not (0 <= c_goal < self._map_w and 0 <= r_goal < self._map_h):
            self.get_logger().error(
                f"La meta ({gx:.1f}m, {gy:.1f}m) cae fuera de la grilla del mapa ({c_goal}, {r_goal})"
            )
            return

        # 2.E Meta bloqueada: buscar celda libre más cercana dentro de goal_search_radius_m
        if math.isinf(float(self._cell_costs[r_goal, c_goal])):
            r_search = int(math.ceil(self.goal_search_radius_m / self._map_res))
            r_min = max(0, r_goal - r_search)
            r_max = min(self._map_h, r_goal + r_search + 1)
            c_min = max(0, c_goal - r_search)
            c_max = min(self._map_w, c_goal + r_search + 1)

            sub_costs = self._cell_costs[r_min:r_max, c_min:c_max]
            sub_r, sub_c = np.ogrid[r_min:r_max, c_min:c_max]
            dist_sq = ((sub_r - r_goal) ** 2 + (sub_c - c_goal) ** 2) * (self._map_res ** 2)
            valid_mask = (~np.isinf(sub_costs)) & (dist_sq <= self.goal_search_radius_m ** 2)

            if np.any(valid_mask):
                dist_masked = np.where(valid_mask, dist_sq, float("inf"))
                min_idx = np.argmin(dist_masked)
                local_r, local_c = np.unravel_index(min_idx, sub_costs.shape)
                best_r = r_min + int(local_r)
                best_c = c_min + int(local_c)
                shift_dist = math.sqrt(float(dist_sq[local_r, local_c]))
                self.get_logger().info(
                    f"Meta original bloqueada, usando celda libre más cercana a {shift_dist:.1f}m"
                )
                r_goal, c_goal = best_r, best_c
            else:
                self.get_logger().warn("Meta inalcanzable: sin celda libre cerca de la meta")
                self._s_goal = None
                self._publish_path_msg(None, is_valid=False)
                return

        self._s_goal = (r_goal, c_goal)
        self._km = 0.0

        if self._g is None or self._g.shape != (self._map_h, self._map_w):
            self._g = np.full((self._map_h, self._map_w), float("inf"), dtype=np.float64)
            self._rhs = np.full((self._map_h, self._map_w), float("inf"), dtype=np.float64)
        else:
            self._g.fill(float("inf"))
            self._rhs.fill(float("inf"))

        self._pq_heap.clear()
        self._pq_dict.clear()
        self._pq_entry_id = 0

        # Inicialización fundamental: rhs(s_goal) = 0, resto = inf
        self._rhs[r_goal, c_goal] = 0.0

        rover_cell = self._get_rover_cell()
        if rover_cell is not None:
            self._s_start = rover_cell
            self._s_last = rover_cell
            self._pq_insert(self._s_goal, self._calculate_key(self._s_goal))
        else:
            self._s_start = None
            self._s_last = None

    def _get_rover_cell(self) -> tuple[int, int] | None:
        """Obtiene la posición actual del rover en coordenadas discretas de la grilla vía TF."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
            rx = float(tf_msg.transform.translation.x)
            ry = float(tf_msg.transform.translation.y)
            c_rover = int(math.floor((rx - self._map_orig_x) / self._map_res))
            r_rover = int(math.floor((ry - self._map_orig_y) / self._map_res))
            if 0 <= c_rover < self._map_w and 0 <= r_rover < self._map_h:
                return (r_rover, c_rover)
            return None
        except Exception:
            return None

    def _get_g(self, u: tuple[int, int]) -> float:
        if self._g is None:
            return float("inf")
        return float(self._g[u[0], u[1]])

    def _set_g(self, u: tuple[int, int], val: float):
        if self._g is not None:
            self._g[u[0], u[1]] = float(val)

    def _get_rhs(self, u: tuple[int, int]) -> float:
        if self._rhs is None:
            return float("inf")
        return float(self._rhs[u[0], u[1]])

    def _set_rhs(self, u: tuple[int, int], val: float):
        if self._rhs is not None:
            self._rhs[u[0], u[1]] = float(val)

    def _heuristic(self, u: tuple[int, int], v: tuple[int, int]) -> float:
        """Distancia euclídea admisible y consistente en el espacio de la grilla métrica."""
        dr = (u[0] - v[0]) * self._map_res
        dc = (u[1] - v[1]) * self._map_res
        return math.hypot(dr, dc)

    def _calculate_key(self, u: tuple[int, int]) -> tuple[float, float]:
        """Calcula la clave de prioridad k(u) = [k1, k2]."""
        r, c = u
        g_val = float(self._g[r, c]) if self._g is not None else float("inf")
        rhs_val = float(self._rhs[r, c]) if self._rhs is not None else float("inf")
        min_val = g_val if g_val < rhs_val else rhs_val
        if self._s_start is not None:
            dr = (self._s_start[0] - r) * self._map_res
            dc = (self._s_start[1] - c) * self._map_res
            h_val = math.hypot(dr, dc)
        else:
            h_val = 0.0
        k1 = min_val + h_val + self._km
        k2 = min_val
        return (k1, k2)

    def _transition_cost(self, u: tuple[int, int], v: tuple[int, int]) -> float:
        """
        Costo de transición c(u, v) entre celdas vecinas adyacentes.
        Diagonal = sqrt(2) * res, Ortogonal = res.
        """
        if self._cell_costs is None:
            return float("inf")

        cost_u = float(self._cell_costs[u[0], u[1]])
        cost_v = float(self._cell_costs[v[0], v[1]])

        if math.isinf(cost_u) or math.isinf(cost_v):
            return float("inf")

        dr = abs(u[0] - v[0])
        dc = abs(u[1] - v[1])
        base_dist = math.sqrt(2.0) * self._map_res if (dr == 1 and dc == 1) else self._map_res

        # Costo ponderado promedio por transitabilidad de ambas celdas
        return base_dist * ((cost_u + cost_v) * 0.5)

    def _get_neighbors(self, u: tuple[int, int]) -> list[tuple[int, int]]:
        """Retorna los vecinos válidos según la conectividad precomputada."""
        r, c = u
        h, w = self._map_h, self._map_w
        neighbors = []
        for dr, dc, _ in self._neighbors_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append((nr, nc))
        return neighbors

    # --------------------------------------------------------------------------
    # Operaciones de Cola de Prioridad U (Min-Heap con Lazy Deletion)
    # --------------------------------------------------------------------------
    def _pq_insert(self, u: tuple[int, int], key: tuple[float, float]):
        self._pq_entry_id += 1
        self._pq_dict[u] = (key[0], key[1], self._pq_entry_id)
        heapq.heappush(self._pq_heap, (key[0], key[1], u, self._pq_entry_id))

    def _pq_remove(self, u: tuple[int, int]):
        self._pq_dict.pop(u, None)

    def _pq_contains(self, u: tuple[int, int]) -> bool:
        return u in self._pq_dict

    def _pq_top_key(self) -> tuple[float, float]:
        while self._pq_heap:
            k1, k2, u, entry_id = self._pq_heap[0]
            val = self._pq_dict.get(u)
            if val is not None and val[2] == entry_id:
                return (k1, k2)
            heapq.heappop(self._pq_heap)
        return (float("inf"), float("inf"))

    def _pq_pop(self) -> tuple[int, int] | None:
        while self._pq_heap:
            k1, k2, u, entry_id = heapq.heappop(self._pq_heap)
            val = self._pq_dict.get(u)
            if val is not None and val[2] == entry_id:
                del self._pq_dict[u]
                return u
        return None

    # --------------------------------------------------------------------------
    # Reparación Incremental de Vértices y Búsqueda del Camino Más Corto
    # --------------------------------------------------------------------------
    def _update_vertex(self, u: tuple[int, int]):
        r, c = u
        if u != self._s_goal:
            cost_u = float(self._cell_costs[r, c])
            if math.isinf(cost_u):
                min_rhs = float("inf")
            else:
                min_rhs = float("inf")
                h, w = self._map_h, self._map_w
                costs = self._cell_costs
                g = self._g
                for dr, dc, base_dist in self._neighbors_8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        cost_v = float(costs[nr, nc])
                        if not math.isinf(cost_v):
                            g_v = float(g[nr, nc])
                            if not math.isinf(g_v):
                                candidate = base_dist * ((cost_u + cost_v) * 0.5) + g_v
                                if candidate < min_rhs:
                                    min_rhs = candidate
            self._rhs[r, c] = min_rhs

        g_val = float(self._g[r, c])
        rhs_val = float(self._rhs[r, c])

        if u in self._pq_dict:
            del self._pq_dict[u]

        if not math.isclose(g_val, rhs_val, abs_tol=1e-5):
            self._pq_insert(u, self._calculate_key(u))

    def _compute_shortest_path(self, max_expansions: int = 40000) -> bool:
        """
        Bucle central de D* Lite optimizado con arrays directos y offsets precomputados.
        """
        if self._s_start is None or self._s_goal is None or self._g is None or self._rhs is None:
            return False

        expansions = 0
        h, w = self._map_h, self._map_w
        neighbors_8 = self._neighbors_8

        while True:
            top_k = self._pq_top_key()
            start_k = self._calculate_key(self._s_start)
            g_start = float(self._g[self._s_start[0], self._s_start[1]])
            rhs_start = float(self._rhs[self._s_start[0], self._s_start[1]])

            # Condición de parada de D* Lite: la cima de la cola es peor que la clave del inicio
            # y el vértice de inicio es consistente
            if not (top_k < start_k or not math.isclose(rhs_start, g_start, abs_tol=1e-5)):
                break

            expansions += 1
            if expansions > max_expansions:
                self.get_logger().warn(
                    f"D* Lite alcanzó el límite de expansiones ({max_expansions}). Grafo complejo o no conexo."
                )
                return False

            u = self._pq_pop()
            if u is None:
                break

            k_old = top_k
            k_new = self._calculate_key(u)

            if k_old < k_new:
                self._pq_insert(u, k_new)
            else:
                ur, uc = u
                g_u = float(self._g[ur, uc])
                rhs_u = float(self._rhs[ur, uc])

                if g_u > rhs_u:
                    self._g[ur, uc] = rhs_u
                    for dr, dc, _ in neighbors_8:
                        nr = ur + dr
                        nc = uc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            self._update_vertex((nr, nc))
                else:
                    self._g[ur, uc] = float("inf")
                    self._update_vertex(u)
                    for dr, dc, _ in neighbors_8:
                        nr = ur + dr
                        nc = uc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            self._update_vertex((nr, nc))

        return not math.isinf(float(self._g[self._s_start[0], self._s_start[1]]))

    def _extract_path(self) -> list[tuple[float, float]] | None:
        """
        Extrae el camino óptimo descendiendo por el gradiente de g desde la pose
        actual del rover (s_start) hasta la meta (s_goal).
        """
        if self._s_start is None or self._s_goal is None or self._g is None or self._cell_costs is None:
            return None

        if math.isinf(float(self._g[self._s_start[0], self._s_start[1]])):
            return None

        path_cells = [self._s_start]
        curr = self._s_start
        visited = {curr}
        max_steps = 10000
        h, w = self._map_h, self._map_w
        neighbors_8 = self._neighbors_8
        costs = self._cell_costs
        g = self._g

        while curr != self._s_goal and len(path_cells) < max_steps:
            cr, cc = curr
            cost_u = float(costs[cr, cc])
            best_next = None
            min_cost = float("inf")

            for dr, dc, base_dist in neighbors_8:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    cost_v = float(costs[nr, nc])
                    if not math.isinf(cost_v):
                        g_v = float(g[nr, nc])
                        if not math.isinf(g_v):
                            c_val = base_dist * ((cost_u + cost_v) * 0.5)
                            candidate_g = c_val + g_v
                            if candidate_g < min_cost:
                                min_cost = candidate_g
                                best_next = (nr, nc)

            if best_next is None or math.isinf(min_cost) or best_next in visited:
                break

            curr = best_next
            visited.add(curr)
            path_cells.append(curr)

        if curr != self._s_goal:
            # Fase 2.G: Distinguir fallo de extracción interno de ausencia de camino
            if not math.isinf(float(self._g[self._s_start[0], self._s_start[1]])):
                self.get_logger().error(
                    "g(s_start) finito pero extracción de camino falló — posible inconsistencia del grafo"
                )
            return None

        # Conversión a coordenadas continuas métricas en el marco 'map'
        path_xy = []
        for r, c in path_cells:
            mx = self._map_orig_x + (float(c) + 0.5) * self._map_res
            my = self._map_orig_y + (float(r) + 0.5) * self._map_res
            path_xy.append((mx, my))

        return path_xy

    # --------------------------------------------------------------------------
    # Ejecución y Publicación del Plan Global
    # --------------------------------------------------------------------------
    def _plan_and_publish(self):
        """
        Ejecuta la búsqueda D* Lite y publica el camino global si transcurrió el período mínimo.

        Comportamiento del Throttle (Fase 2.H):
        - El timer de heartbeat corre cada 0.5s como mecanismo de sondeo periódico, pero sólo ejecuta
          la búsqueda D* Lite si transcurrieron al menos `replan_min_period_s` (ej. 2.0s) desde el último plan.
          En estado estacionario (mapa estable, rover quieto), si el último plan ocurrió en t=0.0s, los ticks
          de t=0.5s, 1.0s y 1.5s retornan sin computar, y el plan se ejecuta en t=2.0s (tasa máxima: 0.5 Hz).
        - Disparo reactivo por mapa: si el último plan fue en t=0.0s y a t=2.1s (una vez cumplidos los 2.0s mínimos)
          llega una actualización relevante en `_on_map`, se dispara el replan de forma inmediata en t=2.1s sin
          tener que esperar al siguiente tick del heartbeat en t=2.5s (ahorro de hasta 0.5s en tiempo de reacción).
        """
        now_mono = time.monotonic()
        if (now_mono - self._last_plan_time) < self.replan_min_period_s:
            return

        with self._lock:
            if self._map_data is None or self._goal_map_xy is None or self._s_goal is None:
                return

            rover_cell = self._get_rover_cell()
            if rover_cell is None:
                self.get_logger().warn(
                    "No se pudo determinar la pose actual del rover en frame 'map' vía TF.",
                    throttle_duration_sec=3.0,
                )
                return

            # Fase 2.F Rover bloqueado:
            if self._cell_costs is not None and math.isinf(float(self._cell_costs[rover_cell[0], rover_cell[1]])):
                self.get_logger().warn(
                    "Rover sobre celda ocupada/inflada — posible error de localización",
                    throttle_duration_sec=2.0,
                )
                self._publish_path_msg(None, is_valid=False)
                return

            # Si el rover se movió desde la última iteración, actualizar km
            if self._s_start is None:
                self._s_start = rover_cell
                self._s_last = rover_cell
                if self._s_goal is not None and not self._pq_contains(self._s_goal):
                    self._pq_insert(self._s_goal, self._calculate_key(self._s_goal))
            elif rover_cell != self._s_start:
                if self._s_last is not None:
                    self._km += self._heuristic(self._s_last, rover_cell)
                self._s_start = rover_cell
                self._s_last = rover_cell

            # Ejecutar búsqueda incremental D* Lite
            t_dstar_start = time.perf_counter()
            success = self._compute_shortest_path()
            path_points = self._extract_path() if success else None
            t_dstar_ms = (time.perf_counter() - t_dstar_start) * 1000.0
            self._last_plan_time = time.monotonic()

        is_valid = path_points is not None and len(path_points) > 0
        self.get_logger().debug(
            f"D* Lite Plan: plan={t_dstar_ms:.1f}ms | valid={is_valid} | points={len(path_points) if path_points else 0}"
        )
        self._publish_path_msg(path_points, is_valid=is_valid)

    def _publish_path_msg(self, path_points: list[tuple[float, float]] | None, is_valid: bool):
        now = self.get_clock().now()

        valid_msg = Bool()
        valid_msg.data = bool(is_valid)
        self.valid_pub.publish(valid_msg)

        path_msg = Path()
        path_msg.header.stamp = now.to_msg()
        path_msg.header.frame_id = self.map_frame

        if is_valid and path_points is not None:
            for mx, my in path_points:
                pose_stamped = PoseStamped()
                pose_stamped.header = path_msg.header
                pose_stamped.pose.position.x = float(mx)
                pose_stamped.pose.position.y = float(my)
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.w = 1.0
                path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)
        self.get_logger().debug(
            f"D* Lite Plan: valid={is_valid} points={len(path_msg.poses)} km={self._km:.2f}"
        )

    def _replan_timer_cb(self):
        self._plan_and_publish()


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
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
