#!/usr/bin/env python3
"""
Nodo de publicación de imágenes de prueba para offline benchmarking y tests de percepción.

Publica imágenes periódicamente al tópico de cámara frontal ('earth_rover/front/image_raw')
utilizando la misma especificación (BEST_EFFORT, frame_id, resolución 1024x576 BGR8)
que el bridge real del Earth Rover.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__("test_image_publisher")

        # Declaración de Parámetros
        self.declare_parameter("image_path", "")
        self.declare_parameter("image_topic", "earth_rover/front/image_raw")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("frame_id", "earth_rover_front_camera")
        self.declare_parameter("loop_count", 0)  # 0 = infinito

        self.image_path = str(self.get_parameter("image_path").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.max_loops = int(self.get_parameter("loop_count").value)

        self.bridge = CvBridge()
        self._published_frames = 0

        # Cargar o generar imagen de prueba (1024x576 BGR8)
        self._frame = self._load_or_generate_image()

        # Configuración QoS idéntica al bridge de hardware
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.publisher_ = self.create_publisher(Image, self.image_topic, sensor_qos)
        period_s = 1.0 / max(self.publish_rate_hz, 0.1)
        self.timer = self.create_timer(period_s, self._timer_callback)

        self.get_logger().info(
            f"TestImagePublisher inicializado | topic={self.image_topic} | "
            f"rate={self.publish_rate_hz}Hz | frame_id={self.frame_id} | "
            f"resolution={self._frame.shape[1]}x{self._frame.shape[0]}"
        )

    def _load_or_generate_image(self) -> np.ndarray:
        if self.image_path and os.path.isfile(self.image_path):
            img = cv2.imread(self.image_path)
            if img is not None:
                self.get_logger().info(f"Imagen cargada desde {self.image_path} ({img.shape})")
                return img
            self.get_logger().warn(f"No se pudo decodificar {self.image_path}, generando imagen sintética.")

        # Generar imagen sintética estándar de rover (1024x576 BGR8 con cielo, horizonte y suelo transitable)
        h, w = 576, 1024
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Cielo (azul claro)
        horizon = int(h * 0.45)
        for y in range(horizon):
            ratio = y / horizon
            b = int(220 * (1 - 0.3 * ratio))
            g = int(180 * (1 - 0.2 * ratio))
            r = int(140 * (1 - 0.1 * ratio))
            img[y, :] = (b, g, r)

        # Suelo / Camino (gris/tierra)
        for y in range(horizon, h):
            ratio = (y - horizon) / (h - horizon)
            b = int(70 + 40 * ratio)
            g = int(90 + 50 * ratio)
            r = int(110 + 60 * ratio)
            img[y, :] = (b, g, r)

        # Añadir textura sutil de terreno
        noise = np.random.RandomState(42).randint(-15, 15, (h - horizon, w, 3))
        ground_patch = img[horizon:, :].astype(np.int16) + noise
        img[horizon:, :] = np.clip(ground_patch, 0, 255).astype(np.uint8)

        self.get_logger().info(f"Imagen sintética de terreno generada ({w}x{h} BGR8)")
        return img

    def _timer_callback(self):
        msg = self.bridge.cv2_to_imgmsg(self._frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher_.publish(msg)
        self._published_frames += 1

        if self.max_loops > 0 and self._published_frames >= self.max_loops:
            self.get_logger().info(f"Se completó la publicación de {self._published_frames} frames.")
            self.timer.cancel()
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
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
