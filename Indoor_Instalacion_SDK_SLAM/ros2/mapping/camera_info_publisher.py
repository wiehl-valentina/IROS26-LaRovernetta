#!/usr/bin/env python3
"""Publica sensor_msgs/CameraInfo para la camara frontal del rover.

RTAB-Map necesita camera_info para registrar imagenes (aunque sea RGB sin
profundidad). El SDK no expone una calibracion propia, asi que este nodo la
toma de PARAMETROS -- usa a proposito los mismos numeros que ya tengas
calibrados y en uso en tu config de genie_rover (`camera.intrinsics` en
`genie/configs/*.yaml`), para que la proyeccion de RTAB-Map y la de
SAM-TP/BEV coincidan. NO trae valores por defecto inventados: si no pasas
fx/fy/cx/cy falla explicitamente en vez de calibrar mal en silencio.

Si nunca calibraste la camara, correr `ros2 run camera_calibration
cameracalibrator` con un tablero de ajedrez es el camino recomendado; hasta
entonces podes arrancar con una aproximacion (fx=fy=ancho_px, cx=ancho/2,
cy=alto/2) sabiendo que el registro de RTAB-Map va a ser menos preciso.

Uso:
    python3 camera_info_publisher.py --ros-args \
        -p width:=1280 -p height:=720 \
        -p fx:=900.0 -p fy:=900.0 -p cx:=640.0 -p cy:=360.0
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo


class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__("earth_rover_camera_info_publisher")
        self.declare_parameter("width", 0)
        self.declare_parameter("height", 0)
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)
        self.declare_parameter("frame_id", "earth_rover_front_camera")
        self.declare_parameter("rate_hz", 10.0)

        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)

        if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
            self.get_logger().error(
                "Faltan width/height/fx/fy (o son <= 0). Pasalos por "
                "--ros-args -p width:=... -p height:=... -p fx:=... -p fy:=... "
                "-p cx:=... -p cy:=... -- usa los mismos valores que "
                "camera.intrinsics en tu config de genie_rover."
            )
            sys.exit(1)

        self.msg = CameraInfo()
        self.msg.width = width
        self.msg.height = height
        self.msg.distortion_model = "plumb_bob"
        self.msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.msg.header.frame_id = self.get_parameter("frame_id").value

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(CameraInfo, "earth_rover/front/camera_info", qos)
        rate = float(self.get_parameter("rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"Publicando camera_info {width}x{height} fx={fx:.1f} fy={fy:.1f} "
            f"cx={cx:.1f} cy={cy:.1f}"
        )

    def _tick(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoPublisher()
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
