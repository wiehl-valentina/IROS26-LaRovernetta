#!/usr/bin/env python3
"""
Nodo de percepción de transitabilidad (SAM-TP / GeNIE) para Earth Rover.

Se suscribe a la cámara frontal ya publicada por earth_rover_bridge
("earth_rover/front/image_raw") y corre el modelo SAM-TP (vendorizado en el
paquete "genie" / "rover_traversability" de sana-earth-rover-policy) en un
hilo propio, a su propio ritmo (0.15-2s/frame según hardware), SIN bloquear
el executor de ROS ni el loop de control de gps_waypoint_controller.

Publica dos señales simples, listas para que gps_waypoint_controller las
combine con su propia corrección de rumbo por GPS:

  - earth_rover/traversability_blocked  (std_msgs/Bool)
        True si el corredor central está bloqueado y ningún corredor lateral
        supera el score mínimo (equivalente al "stop" de policy.suggest_command).

  - earth_rover/traversability_angular_bias (std_msgs/Float32)
        Corrección angular pura de visión (sin sesgo hacia el GPS goal:
        goal_offset_deg=None, "obstacle avoidance" puro), en la MISMA
        convención que un Twist.angular.z ya resuelto (positivo = IZQUIERDA,
        REP-103 estándar). Ver nota de convenciones en gps_waypoint_controller:
        el twist.angular.z que ese nodo publica, luego de _apply_angular_sign(),
        ya queda en esta misma convención estándar -- por eso este bias se puede
        sumar directo a twist.angular.z sin flips adicionales.

  - earth_rover/traversability_overlay (sensor_msgs/Image, opcional)
        Overlay verde/rojo (drivable/bloqueado) para debug en rqt_image_view.

Requiere las dependencias de sana-earth-rover-policy instaladas en el mismo
entorno donde corren los nodos ROS:

    pip install torch torchvision
    pip install --no-build-isolation -e ./genie
    pip install -e './traversability[hf]'

Si no están instaladas, el nodo loguea un error claro con las instrucciones
de instalación y no arranca (falla rápido, en vez de quedar "vivo" sin
publicar nada).
"""

import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


class TraversabilityNode(Node):
    def __init__(self):
        super().__init__("traversability_node")

        # 1. Parámetros
        self.declare_parameter("image_topic", "earth_rover/front/image_raw")
        self.declare_parameter("blocked_topic", "earth_rover/traversability_blocked")
        self.declare_parameter("bias_topic", "earth_rover/traversability_angular_bias")
        self.declare_parameter("overlay_topic", "earth_rover/traversability_overlay")
        self.declare_parameter("publish_overlay", True)

        # Resolución de checkpoint / device (ver rover_traversability.weights
        # para la cadena de resolución si se dejan vacíos)
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("hf_repo", "")
        self.declare_parameter("device", "")
        self.declare_parameter("contrast_refine", True)

        # Throttle opcional: piso de tiempo entre inferencias, además del que
        # ya impone la duración real de predict(). 0.0 = sin piso adicional.
        self.declare_parameter("inference_min_period_s", 0.0)

        # PolicyConfig (traversability/rover_traversability/policy.py) --
        # mismos nombres y defaults que el dataclass, expuestos como params.
        self.declare_parameter("roi_top", 0.55)
        self.declare_parameter("drivable_thresh", 0.5)
        self.declare_parameter("num_corridors", 9)
        self.declare_parameter("stop_center_fraction", 0.40)
        self.declare_parameter("min_corridor_score", 0.35)
        self.declare_parameter("bottom_weight", 2.0)
        self.declare_parameter("min_valid_pixels", 200)
        self.declare_parameter("max_linear", 0.5)
        self.declare_parameter("min_linear", 0.15)
        self.declare_parameter("max_angular", 0.5)
        self.declare_parameter("k_angular", 1.2)
        self.declare_parameter("hfov_deg", 92.7)
        self.declare_parameter("goal_sigma", 0.5)
        self.declare_parameter("goal_bias_floor", 0.2)

        image_topic = str(self.get_parameter("image_topic").value)
        blocked_topic = str(self.get_parameter("blocked_topic").value)
        bias_topic = str(self.get_parameter("bias_topic").value)
        overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.inference_min_period_s = float(self.get_parameter("inference_min_period_s").value)

        checkpoint_path = str(self.get_parameter("checkpoint_path").value) or None
        hf_repo = str(self.get_parameter("hf_repo").value) or None
        device = str(self.get_parameter("device").value) or None
        contrast_refine = bool(self.get_parameter("contrast_refine").value)

        # 2. Carga perezosa del modelo (pesado: torch + sam2 + checkpoint).
        #    Falla rápido y con un mensaje claro si las deps no están.
        try:
            from rover_traversability.policy import PolicyConfig, suggest_command
            from rover_traversability.predictor import TraversabilityPredictor
            from rover_traversability.weights import SamNotInstalledError
        except ImportError as exc:
            self.get_logger().error(
                "No se pudo importar rover_traversability / sam2. Instalá las "
                "dependencias en este mismo entorno:\n"
                "    pip install torch torchvision\n"
                "    pip install --no-build-isolation -e ./genie\n"
                "    pip install -e './traversability[hf]'\n"
                f"Error original: {exc}"
            )
            raise

        self._suggest_command = suggest_command
        self._policy_cfg = PolicyConfig(
            roi_top=float(self.get_parameter("roi_top").value),
            drivable_thresh=float(self.get_parameter("drivable_thresh").value),
            num_corridors=int(self.get_parameter("num_corridors").value),
            stop_center_fraction=float(self.get_parameter("stop_center_fraction").value),
            min_corridor_score=float(self.get_parameter("min_corridor_score").value),
            bottom_weight=float(self.get_parameter("bottom_weight").value),
            min_valid_pixels=int(self.get_parameter("min_valid_pixels").value),
            max_linear=float(self.get_parameter("max_linear").value),
            min_linear=float(self.get_parameter("min_linear").value),
            max_angular=float(self.get_parameter("max_angular").value),
            k_angular=float(self.get_parameter("k_angular").value),
            hfov_deg=float(self.get_parameter("hfov_deg").value),
            goal_sigma=float(self.get_parameter("goal_sigma").value),
            goal_bias_floor=float(self.get_parameter("goal_bias_floor").value),
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
            f"SAM-TP cargado en device={self._predictor.device} "
            f"(contrast_refine={contrast_refine})"
        )

        # 3. QoS: cámara/salidas de percepción son sensor-like -> BEST_EFFORT,
        #    profundidad 1 (siempre el dato más fresco, nunca cola vieja).
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.bridge = CvBridge()
        self.create_subscription(Image, image_topic, self._on_image, sensor_qos)

        self.blocked_pub = self.create_publisher(Bool, blocked_topic, sensor_qos)
        self.bias_pub = self.create_publisher(Float32, bias_topic, sensor_qos)
        self.overlay_pub = (
            self.create_publisher(Image, overlay_topic, sensor_qos)
            if self.publish_overlay
            else None
        )

        # 4. Estado compartido con el hilo de inferencia
        self._frame_lock = threading.Lock()
        self._latest_rgb = None  # np.ndarray HxWx3 uint8, RGB
        self._stop_event = threading.Event()
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

        self.get_logger().info(
            f"Traversability Node iniciado | image_topic={image_topic} | "
            f"blocked_topic={blocked_topic} | bias_topic={bias_topic}"
        )

    # --- CALLBACK DE CÁMARA ---
    def _on_image(self, msg: Image):
        # earth_rover_bridge publica en bgr8 (cv2_to_imgmsg(frame, "bgr8")).
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - conversión de imagen, no crítico
            self.get_logger().warn(f"No se pudo convertir el frame: {exc}", throttle_duration_sec=5)
            return

        rgb = bgr[:, :, ::-1]  # BGR -> RGB, sin depender de cv2 acá
        with self._frame_lock:
            self._latest_rgb = rgb

    # --- HILO DE INFERENCIA (libre, no atado al executor de ROS) ---
    def _inference_loop(self):
        last_start = 0.0
        while not self._stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_rgb
                self._latest_rgb = None  # consumimos: no reprocesar el mismo frame

            if frame is None:
                time.sleep(0.02)
                continue

            elapsed_since_last = time.monotonic() - last_start
            if elapsed_since_last < self.inference_min_period_s:
                time.sleep(self.inference_min_period_s - elapsed_since_last)

            last_start = time.monotonic()
            try:
                self._run_inference(frame)
            except Exception as exc:  # noqa: BLE001 - nunca tirar abajo el hilo
                self.get_logger().error(f"Fallo de inferencia SAM-TP: {exc}", throttle_duration_sec=5)

    def _run_inference(self, rgb: np.ndarray):
        result = self._predictor.predict(rgb)

        # Bias de visión puro (sin sesgo hacia el GPS goal): goal_offset_deg=None.
        # gps_waypoint_controller ya sabe cuál es el heading_error hacia el
        # checkpoint; ese blending queda de su lado, este nodo solo aporta
        # la lectura de la cámara.
        decision = self._suggest_command(result.mask, self._policy_cfg, goal_offset_deg=None)

        blocked_msg = Bool()
        blocked_msg.data = bool(decision.stop)
        self.blocked_pub.publish(blocked_msg)

        bias_msg = Float32()
        bias_msg.data = float(decision.angular)
        self.bias_pub.publish(bias_msg)

        if self.overlay_pub is not None:
            overlay_bgr = result.overlay[:, :, ::-1]  # RGB -> BGR para cv_bridge
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
            overlay_msg.header.stamp = self.get_clock().now().to_msg()
            overlay_msg.header.frame_id = "earth_rover_front_camera"
            self.overlay_pub.publish(overlay_msg)

        self.get_logger().debug(
            f"[{decision.reason}] blocked={decision.stop} angular={decision.angular:+.3f} "
            f"inference={result.inference_s * 1000:.0f}ms"
        )

    def destroy_node(self):
        self._stop_event.set()
        self._infer_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
