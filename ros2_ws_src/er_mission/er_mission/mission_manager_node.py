#!/usr/bin/env python3
"""Mission Manager Node for Earth Rover Mission 1."""

import json
import math
import threading
import requests
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, Bool


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__("mission_manager_node")

        self.declare_parameter("sdk_url", "http://localhost:8000")
        self.declare_parameter("checkpoint_post_retries", 15)
        self.declare_parameter("checkpoint_post_min_interval_s", 2.0)
        self.declare_parameter("checkpoint_max_distance_m", 14.5)  # Dispara a 50cm dentro del perímetro    
        self.declare_parameter("proximity_dwell_s", 15.0)
        self.declare_parameter("min_navigation_time_s", 8.0)
        self.declare_parameter("pre_post_stop_s", 2.5)
        self.declare_parameter("gps_retention_max_age_s", 10.0)

        self.sdk_url = self.get_parameter("sdk_url").value.rstrip("/")
        self.checkpoint_post_retries = int(self.get_parameter("checkpoint_post_retries").value)
        self.checkpoint_post_min_interval_s = float(
            self.get_parameter("checkpoint_post_min_interval_s").value
        )
        self.checkpoint_max_distance_m = float(
            self.get_parameter("checkpoint_max_distance_m").value
        )
        self.proximity_dwell_s = float(self.get_parameter("proximity_dwell_s").value)
        self.min_navigation_time_s = float(self.get_parameter("min_navigation_time_s").value)
        self.pre_post_stop_s = float(self.get_parameter("pre_post_stop_s").value)
        self.gps_retention_max_age_s = float(
            self.get_parameter("gps_retention_max_age_s").value
        )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.target_pub = self.create_publisher(NavSatFix, "earth_rover/target_waypoint", 10)
        self.pause_pub = self.create_publisher(Bool, "earth_rover/navigation_pause", 10)
        self.create_subscription(
            NavSatFix, 
            "gps/filtered", 
            self._on_gps, 
            sensor_qos
        )
        self.create_subscription(String, "earth_rover/waypoint_status", self._on_waypoint_status, status_qos)

        self.current_lat = None
        self.current_lon = None
        self._last_valid_gps = None
        self._last_valid_gps_time = None

        self.checkpoints = []
        self.current_checkpoint_idx = 0
        self.latest_scanned_checkpoint = 0
        self.state = "STARTING_MISSION"
        self._start_retry_period_s = 10.0
        self._next_start_attempt_at = 0.0
        self._checkpoint_post_attempts = 0
        self._pending_confirmation_sequence = None
        self._checkpoint_post_ok = False
        self._last_post_response = None
        self._last_post_attempt_at = None
        self._handling_checkpoint = False
        self._near_checkpoint_since = None
        self._checkpoint_target_since = None
        self._controller_reached = False
        self._pre_post_stop_until = None
        self._mission_completed_by_sdk = False
        self.status_pub = self.create_publisher(String, "earth_rover/waypoint_status", 10)

        # Sincronización concurrente para peticiones HTTP asíncronas (Brief 14 / N.3)
        self._http_lock = threading.Lock()
        self._http_request_in_flight = False
        self._http_response_data = None
        self._http_response_status = None

        self.timer = self.create_timer(1.0, self._state_machine_loop)
        self.get_logger().info("Mission Manager Node Initialized")


    def _abort_and_resume_navigation(self):
        """Libera el candado de los motores si el SDK rechaza la llegada y garantiza publicación del objetivo."""
        self.state = "NAVIGATING_CHECKPOINT"
        self._checkpoint_post_attempts = 0
        self._checkpoint_post_ok = False
        self._last_post_response = None
        self._pending_confirmation_sequence = None
        self._near_checkpoint_since = None
        
        # Garantizar que el objetivo actual siempre quede publicado en earth_rover/target_waypoint (Brief 14 / N.4)
        self._publish_current_checkpoint_goal()
        self.get_logger().info("Navegación reanudada: Objetivo republicado y control cedido al controlador.")

    def _on_gps(self, msg: NavSatFix):
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            self.current_lat = None
            self.current_lon = None
            return
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self._last_valid_gps = (msg.latitude, msg.longitude)
        self._last_valid_gps_time = self.get_clock().now()

    def _get_effective_gps(self):
        """Retorna (lat, lon, age_s) o (None, None, None) aplicando guarda de retención.
        
        Si current_lat/lon están disponibles de una muestra reciente, los usa directamente (age_s = 0.0).
        Si son None (corte transitorio), utiliza la última coordenada válida (_last_valid_gps)
        siempre que tenga menos de gps_retention_max_age_s (10.0 s).
        
        Justificación del umbral de 10 s (Brief 19 / S.2): A la velocidad de avance crucero actual
        (forward_throttle=0.40, aproximadamente 0.44 m/s si la relación es lineal), 10 s corresponden
        a ~4.4 m de desplazamiento. Con una tolerancia de checkpoint de 13.0 m (disparo de proximidad a 14.5 m),
        el error de posición acumulado queda holgadamente dentro del margen.
        Si forward_throttle se incrementa en el futuro, este umbral debe recalibrarse.
        """
        if self.current_lat is not None and self.current_lon is not None:
            return self.current_lat, self.current_lon, 0.0

        if self._last_valid_gps is not None and self._last_valid_gps_time is not None:
            now = self.get_clock().now()
            age_s = (now - self._last_valid_gps_time).nanoseconds / 1e9
            if age_s <= self.gps_retention_max_age_s:
                lat, lon = self._last_valid_gps
                return lat, lon, age_s

        return None, None, None

    def _on_waypoint_status(self, msg: String):
        if msg.data == "REACHED":
            self._controller_reached = True
            if self.state in ("NAVIGATING_CHECKPOINT", "AWAITING_SDK_CONFIRMATION", "CONFIRMING_CHECKPOINT", "PRE_POST_STOP"):
                self.get_logger().info(f"Controller REACHED in state {self.state}")
                self._begin_checkpoint_post()
            elif self.state != "FINISHED":
                self.get_logger().warning(
                    f"Ignoring REACHED signal in state {self.state}",
                    throttle_duration_sec=5,
                )
            return

        if msg.data.startswith("ALIGN/") or msg.data.startswith("DRIVE/"):
            return

    def _begin_checkpoint_post(self):
        if self.state in ("FINISHED", "PRE_POST_STOP"):
            return
        pause = Bool()
        pause.data = True
        self.pause_pub.publish(pause)
        self._pre_post_stop_until = self.get_clock().now() + Duration(seconds=self.pre_post_stop_s)
        self.state = "PRE_POST_STOP"
        self.get_logger().info(
            f"Stopping robot for {self.pre_post_stop_s:.1f}s before POST /checkpoint-reached"
        )

    def _pre_post_stop_elapsed(self):
        if self._pre_post_stop_until is None:
            return True
        return self.get_clock().now() >= self._pre_post_stop_until

    def _state_machine_loop(self):
        if self.state == "PRE_POST_STOP":
            if not self._pre_post_stop_elapsed():
                return
            self._pre_post_stop_until = None
            self._handle_checkpoint_reached()
            return

        if self.state == "AWAITING_SDK_CONFIRMATION":
            if self._pre_post_stop_until is None:
                self._begin_checkpoint_post()
                return
            if not self._pre_post_stop_elapsed():
                return
            self._pre_post_stop_until = None
            self._handle_checkpoint_reached()
            return

        if self.state == "CONFIRMING_CHECKPOINT":
            return

        if self.state == "NAVIGATING_CHECKPOINT":
            self._check_proximity_to_checkpoint()

        if self.state == "AWAITING_HTTP_RESPONSE":
            with self._http_lock:
                if self._http_request_in_flight:
                    return  # Seguimos esperando al hilo de red
                
                # El hilo terminó de forma segura, extraemos los datos atómicamente
                data = self._http_response_data
                status = self._http_response_status
                # Limpiar variables de estado para evitar reprocesamiento
                self._http_response_data = None
                self._http_response_status = None

            if status == 200:
                msg = data.get("message", "Exito") if isinstance(data, dict) else "Exito"
                seq = data.get("next_checkpoint_sequence", "N/A") if isinstance(data, dict) else "N/A"
                comp = data.get("mission_completed", False) if isinstance(data, dict) else False
                self.get_logger().info(f"¡POST Exitoso! {msg}. Siguiente secuencia: {seq}. Misión completada: {comp}")
                
                self._checkpoint_post_ok = True
                self._last_post_response = data
                self.state = "CONFIRMING_CHECKPOINT"
                self._handle_checkpoint_reached()
            else:
                # Parseo de la estructura de error anidada del SDK
                if isinstance(data, dict) and "detail" in data:
                    detail = data["detail"]
                    dist = detail.get("proximate_distance_to_checkpoint", "Desconocida")
                    err = detail.get("error", "Error del SDK")
                    self.get_logger().warn(f"Rechazo del SDK: {err} | Distancia del servidor: {dist}m")
                else:
                    self.get_logger().error(f"Fallo en POST asíncrono. HTTP {status}: {data}")
                    
                self.state = "AWAITING_SDK_CONFIRMATION"
            return
        
        if self.state == "STARTING_MISSION":
            now = self.get_clock().now().nanoseconds / 1e9
            if now < self._next_start_attempt_at:
                return
            self._next_start_attempt_at = now + self._start_retry_period_s

            self.get_logger().info("Solicitando start-mission al SDK...")
            try:
                # Equivalente exacto a: curl --location --request POST 'http://localhost:8000/start-mission'
                res = requests.post(f"{self.sdk_url}/start-mission", timeout=5.0)
                
                if res.status_code == 200:
                    self.get_logger().info(f"Misión iniciada: {res.json()}")
                    self.state = "FETCHING_CHECKPOINTS"
                else:
                    # Parseo inteligente de errores tipo FastAPI {"detail": "..."}
                    error_msg = res.text
                    try:
                        data = res.json()
                        if "detail" in data:
                            error_msg = data["detail"]
                    except ValueError:
                        pass
                    
                    self.get_logger().warning(
                        f"Rechazo del SDK al iniciar (HTTP {res.status_code}): {error_msg}. "
                        f"Reintentando en {self._start_retry_period_s:.0f}s."
                    )
            except Exception as e:
                self.get_logger().error(f"Error de red iniciando misión: {e}")

        elif self.state == "FETCHING_CHECKPOINTS":
            if self._refresh_checkpoints_from_sdk():
                self.state = "WAITING_FOR_GPS"

        elif self.state == "WAITING_FOR_GPS":
            if self.current_lat is None or self.current_lon is None:
                self.get_logger().info(
                    "Waiting for first GPS fix before navigating checkpoints...",
                    throttle_duration_sec=5,
                )
                return

            if not self.checkpoints:
                self.get_logger().warning("No checkpoints found. Mission finished without navigation.")
                self._finish_mission()
                return

            # Asignamos el índice del objetivo inicial (Forzado a 0 por nuestra amnesia previa)
            self.current_checkpoint_idx = self._first_pending_checkpoint_index()
            
            if self.current_checkpoint_idx < len(self.checkpoints):
                # --- INYECCIÓN DE LÓGICA WARM START (SPAWN-POINT VERIFICATION) ---
                dist = self._distance_to_current_checkpoint()
                
                # Validamos si ya "nacimos" dentro de la meta
                if dist is not None and dist <= self.checkpoint_max_distance_m:
                    self.get_logger().info(
                        f"¡Warm Start Detectado! Rover a {dist:.1f}m del Checkpoint {self.current_checkpoint_idx + 1}. "
                        "Disparando POST HTTP..."
                    )
                    
                    pause_msg = Bool()
                    pause_msg.data = True
                    self.pause_pub.publish(pause_msg)

                    # Reseteamos ANTES de llamar, no después (por las dudas de que
                    # queden pisadas banderas de un checkpoint anterior)
                    self._checkpoint_post_attempts = 0
                    self._checkpoint_post_ok = False 
                    self._pending_confirmation_sequence = self.checkpoints[self.current_checkpoint_idx]["sequence"]

                    # Llamamos directamente a la función que maneja el POST; ella misma
                    # deja self.state en el valor correcto (AWAITING_HTTP_RESPONSE, etc.)
                    self._handle_checkpoint_reached_impl()
                    
                else:
                    # Comportamiento normal: Estamos fuera de la meta, toca conducir.
                    if dist is not None:
                        self.get_logger().info(f"Distancia inicial al Checkpoint: {dist:.1f}m. Iniciando aproximación física.")
                    
                    self._publish_current_checkpoint_goal()
                    self.state = "NAVIGATING_CHECKPOINT"
                # -----------------------------------------------------------------
            else:
                self.get_logger().info("All checkpoints already completed.")
                self._finish_mission()

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2):
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

    def _distance_to_current_checkpoint(self):
        if self.current_checkpoint_idx >= len(self.checkpoints):
            return None
        lat, lon, age_s = self._get_effective_gps()
        if lat is None or lon is None:
            return None
        if age_s > 0.0:
            self.get_logger().warning(
                f"Calculando distancia con coordenada GPS retenida ({lat:.6f}, {lon:.6f}, antigüedad={age_s:.1f}s <= {self.gps_retention_max_age_s:.1f}s)",
                throttle_duration_sec=2.0,
            )
        cp = self.checkpoints[self.current_checkpoint_idx]
        cp_lat = float(cp.get("latitude", cp.get("lat", 0)))
        cp_lon = float(cp.get("longitude", cp.get("lon", 0)))
        return self._haversine_m(lat, lon, cp_lat, cp_lon)

    def _refresh_checkpoints_from_sdk(self):
        self.get_logger().info("Descargando mission checkpoints list...")
        try:
            # Equivalente exacto a: curl --location 'http://localhost:8000/checkpoints-list'
            res = requests.get(f"{self.sdk_url}/checkpoints-list", timeout=5.0)
            
            if res.status_code != 200:
                self.get_logger().error(f"Fallo al obtener checkpoints: HTTP {res.status_code}")
                return False

            data = res.json()
            
            # Extracción segura de la lista anidada
            if isinstance(data, dict):
                if isinstance(data.get("checkpoints_list"), dict):
                    nested = data["checkpoints_list"]
                    self.checkpoints = nested.get("checkpoints_list", [])
                    sdk_latest = int(nested.get("latest_scanned_checkpoint") or 0)
                else:
                    self.checkpoints = data.get("checkpoints_list", [])
                    sdk_latest = int(data.get("latest_scanned_checkpoint") or 0)
            elif isinstance(data, list):
                self.checkpoints = data
                sdk_latest = 0
            else:
                self.checkpoints = []
                sdk_latest = 0

            # -----------------------------------------------------------------
            # PARCHE DE AMNESIA: Forzar siempre Checkpoint 1 al inicio
            # -----------------------------------------------------------------
            # Si estamos en la fase inicial de la misión, ignoramos lo que diga el SDK
            if self.state in ("FETCHING_CHECKPOINTS", "WAITING_FOR_GPS") and self.current_checkpoint_idx == 0:
                self.latest_scanned_checkpoint = 0
                self.get_logger().info("Amnesia activada: Forzando inicio desde el Checkpoint 1.")
            else:
                # Si ya estamos navegando, actualizamos normalmente para no perder progreso
                self.latest_scanned_checkpoint = max(self.latest_scanned_checkpoint, sdk_latest)

            self.get_logger().info(
                f"Fetched {len(self.checkpoints)} checkpoints. "
                f"Latest scanned internal memory: {self.latest_scanned_checkpoint}"
            )
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error fetching checkpoints: {e}")
            return False

    def _check_proximity_to_checkpoint(self):
        # Antes esto solo corria DESPUES de que gps_waypoint_controller ya
        # habia declarado "Target reached" (frenado incluido). Ahora chequea
        # la distancia real en todo momento mientras navega -- si entra en
        # rango del SDK ANTES de que el controller termine de alinearse/
        # frenar, reporta altiro y avanza al siguiente checkpoint, sin
        # esperar el ciclo completo de parada.
        if self._checkpoint_target_since is None:
            return

        elapsed_nav = (
            self.get_clock().now() - self._checkpoint_target_since
        ).nanoseconds / 1e9
        if elapsed_nav < self.min_navigation_time_s:
            return

        dist = self._distance_to_current_checkpoint()
        if dist is None:
            self._near_checkpoint_since = None
            return

        if dist > self.checkpoint_max_distance_m:
            self._near_checkpoint_since = None
            return

        now = self.get_clock().now()
        if self._near_checkpoint_since is None:
            self._near_checkpoint_since = now
            self.get_logger().info(
                f"Within {dist:.1f}m of checkpoint (<= {self.checkpoint_max_distance_m:.0f}m), "
                f"waiting {self.proximity_dwell_s:.0f}s before POST",
                throttle_duration_sec=5,
            )
            return

        elapsed = (now - self._near_checkpoint_since).nanoseconds / 1e9
        if elapsed < self.proximity_dwell_s:
            return

        self._near_checkpoint_since = None
        self.get_logger().info(
            f"Proximity backup: {dist:.1f}m for {elapsed:.1f}s — triggering POST"
        )
        self._begin_checkpoint_post()

    def _publish_current_checkpoint_goal(self):
        if self.current_checkpoint_idx >= len(self.checkpoints):
            self.get_logger().info("No pending checkpoints found.")
            self._finish_mission()
            return

        cp = self.checkpoints[self.current_checkpoint_idx]
        target_msg = NavSatFix()
        target_msg.latitude = float(cp.get("latitude", cp.get("lat", 0)))
        target_msg.longitude = float(cp.get("longitude", cp.get("lon", 0)))
        self.target_pub.publish(target_msg)
        resume = Bool()
        resume.data = False
        self.pause_pub.publish(resume)
        self._near_checkpoint_since = None
        self._checkpoint_target_since = self.get_clock().now()
        self._controller_reached = False
        self._pre_post_stop_until = None
        self._checkpoint_post_attempts = 0
        self._pending_confirmation_sequence = None
        self._checkpoint_post_ok = False
        self._last_post_response = None
        self.get_logger().info(
            f"Navigating to Checkpoint sequence {cp.get('sequence', self.current_checkpoint_idx + 1)} "
            f"({self.current_checkpoint_idx + 1}/{len(self.checkpoints)}): "
            f"({target_msg.latitude}, {target_msg.longitude})"
        )

    def _first_pending_checkpoint_index(self):
        for idx, checkpoint in enumerate(self.checkpoints):
            try:
                sequence = int(checkpoint.get("sequence", idx + 1))
            except (TypeError, ValueError):
                sequence = idx + 1
            if sequence > self.latest_scanned_checkpoint:
                return idx
        return len(self.checkpoints)

    def _current_checkpoint_sequence(self):
        if self.current_checkpoint_idx >= len(self.checkpoints):
            return None
        cp = self.checkpoints[self.current_checkpoint_idx]
        try:
            return int(cp.get("sequence", self.current_checkpoint_idx + 1))
        except (TypeError, ValueError):
            return self.current_checkpoint_idx + 1

    def _notify_checkpoint_reached(self, sequence):
        lat, lon, age_s = self._get_effective_gps()
        if lat is None or lon is None:
            self.get_logger().error(
                f"Imposible publicar checkpoint {sequence} sin GPS (sin fix reciente ni retenido <= {self.gps_retention_max_age_s:.1f}s)"
            )
            return False, None

        if age_s > 0.0:
            self.get_logger().warning(
                f"Notificando checkpoint {sequence} usando coordenada GPS retenida ({lat:.6f}, {lon:.6f}, antigüedad={age_s:.1f}s <= {self.gps_retention_max_age_s:.1f}s)"
            )

        now = self.get_clock().now()
        if self._last_post_attempt_at is not None:
            elapsed = (now - self._last_post_attempt_at).nanoseconds / 1e9
            if elapsed < self.checkpoint_post_min_interval_s:
                return False, None

        self._checkpoint_post_attempts += 1
        self._last_post_attempt_at = now
        
        self.get_logger().info(f"Iniciando POST asíncrono para checkpoint {sequence}...")

        with self._http_lock:
            self._http_request_in_flight = True
            self._http_response_data = None
            self._http_response_status = None

        def http_worker():
            status = 500
            data = None
            try:
                # Implementación exacta de: curl -X POST ... --header 'Content-Type: application/json' --data '{}'
                url = f"{self.sdk_url}/checkpoint-reached"
                headers = {'Content-Type': 'application/json'}
                
                res = requests.post(url, headers=headers, json={}, timeout=10.0)
                status = res.status_code
                
                try:
                    data = res.json()
                except ValueError:
                    data = {"detail": {"error": res.text}}
                    
            except Exception as e:
                status = 500
                data = {"detail": {"error": f"Error de red: {str(e)}" }}
            finally:
                with self._http_lock:
                    self._http_response_status = status
                    self._http_response_data = data
                    self._http_request_in_flight = False

        # Lanzar hilo en background para no asfixiar el middleware DDS de ROS 2
        threading.Thread(target=http_worker, daemon=True).start()
        
        # Inmediatamente cambiamos a un nuevo estado de espera
        self.state = "AWAITING_HTTP_RESPONSE"
        return "in_flight", None

    def _confirm_checkpoint(self, sequence, post_data):
        if post_data:
            if post_data.get("mission_completed"):
                self.latest_scanned_checkpoint = max(self.latest_scanned_checkpoint, sequence)
                return True

            if post_data.get("message") == "Checkpoint reached successfully":
                self.latest_scanned_checkpoint = max(self.latest_scanned_checkpoint, sequence)
                next_seq = post_data.get("next_checkpoint_sequence")
                try:
                    if next_seq not in (None, "") and int(next_seq) > sequence:
                        return True
                except (TypeError, ValueError):
                    pass
                return True

        if self._refresh_checkpoints_from_sdk():
            if self.latest_scanned_checkpoint >= sequence:
                self.get_logger().info(
                    f"SDK list confirms checkpoint {sequence} "
                    f"(latest_scanned_checkpoint={self.latest_scanned_checkpoint})"
                )
                return True

        return False

    def _handle_checkpoint_reached(self):
        if self._handling_checkpoint:
            return
        self._handling_checkpoint = True
        try:
            self._handle_checkpoint_reached_impl()
        finally:
            self._handling_checkpoint = False

    def _handle_checkpoint_reached_impl(self):
        sequence = self._pending_confirmation_sequence
        if sequence is None:
            sequence = self._current_checkpoint_sequence()
            self._pending_confirmation_sequence = sequence

        dist = self._distance_to_current_checkpoint()
        if dist is not None and dist > self.checkpoint_max_distance_m:
            self.get_logger().warning(
                f"REACHED ignorado: {dist:.1f}m away (> {self.checkpoint_max_distance_m:.1f}m)"
            )
            self._abort_and_resume_navigation()
            return

        self.state = "CONFIRMING_CHECKPOINT"

        if not self._checkpoint_post_ok:
            ok, data = self._notify_checkpoint_reached(sequence)
            if ok == "mission_ended":
                self._mission_completed_by_sdk = True
                self._finish_mission()
                return
            # --- INYECCIÓN CRÍTICA ---
            elif ok == "in_flight":
                # El hilo se lanzó con éxito. Cortamos la ejecución aquí
                # para que el estado AWAITING_HTTP_RESPONSE haga su magia.
                return
            # -------------------------
            if not ok:
                # CRÍTICO: Si fallan los reintentos de red, no paralizamos el rover
                if self._checkpoint_post_attempts >= self.checkpoint_post_retries:
                    self.get_logger().error("SDK inalcanzable tras múltiples intentos. Abortando POST.")
                    self._abort_and_resume_navigation()
                else:
                    self.state = "AWAITING_SDK_CONFIRMATION"
                return
            
            self._checkpoint_post_ok = True
            self._last_post_response = data
        else:
            data = self._last_post_response

        # CRÍTICO: Si el SDK rechaza confirmar el Checkpoint (ej. por estar muy lejos físicamente)
        if not self._confirm_checkpoint(sequence, data):
            self.get_logger().warning(f"SDK rechazó Checkpoint {sequence}. Reanudando aproximación.")
            self._abort_and_resume_navigation()
            return

        # Si llegamos aquí, el SDK confirmó el éxito.
        self.get_logger().info(f"Checkpoint {sequence} fully confirmed.")
        self._pending_confirmation_sequence = None
        self._checkpoint_post_attempts = 0
        self._checkpoint_post_ok = False
        self._last_post_response = None
        self._last_post_attempt_at = None

        if data and data.get("mission_completed"):
            self._mission_completed_by_sdk = True
            self._finish_mission()
            return

        self.current_checkpoint_idx += 1

        if self.current_checkpoint_idx < len(self.checkpoints):
            self.get_logger().info("Advancing to next checkpoint.")
            self._publish_current_checkpoint_goal()
            self.state = "NAVIGATING_CHECKPOINT"
        else:
            self.get_logger().info("All checkpoints completed.")
            self._finish_mission()

    def _finish_mission(self):
        if self.state == "FINISHED":
            return
        self.state = "FINISHED"
        pause = Bool()
        pause.data = True
        self.pause_pub.publish(pause)
        done = String()
        done.data = "MISSION_FINISHED"
        self.status_pub.publish(done)

        if self._mission_completed_by_sdk:
            self.get_logger().info("Mission already completed by SDK on last checkpoint POST.")
            return

        self.get_logger().info("Ending mission via SDK...")
        try:
            res = requests.post(f"{self.sdk_url}/end-mission", timeout=3.0)
            self.get_logger().info(f"End mission response ({res.status_code}): {res.text}")
        except Exception as e:
            self.get_logger().error(f"Error ending mission: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
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