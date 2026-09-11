#!/usr/bin/env python3
"""Republishes the EKF's fused absolute yaw (ENU, from ekf_filter_node_map's
/odometry/global) as a compass heading in degrees (0=North, clockwise), on
the same topic/type gps_waypoint_controller already expects
(earth_rover/heading, std_msgs/Float32) — heading publisher in BEST_EFFORT.

Brief 22 / V.3:
Propagación continua de rumbo entre actualizaciones discretas del compás (~2s)
integrando la velocidad angular del giróscopo (/imu/data, gyro_z):
    yaw_estimado(t) = yaw_ultimo_compass + ∫ gyro_z dt
Publica rumbo propagado en earth_rover/heading e incertidumbre en
earth_rover/heading_uncertainty.
"""
import json
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String


class EkfHeadingBridge(Node):
    def __init__(self):
        super().__init__("ekf_heading_bridge")

        # 1. Parámetros (Brief 22 / V.3)
        self.declare_parameter("gyro_bias_z", 0.0)
        self.declare_parameter("q_gyro_deg_s", 0.5)  # Crecimiento de incertidumbre: 0.5° / sqrt(s)
        self.declare_parameter("base_compass_uncertainty_deg", 3.0)  # Incertidumbre base del compás
        self.declare_parameter("realign_alpha", 1.0)  # 1.0 = referencia absoluta directa al compás

        self.gyro_bias_z = float(self.get_parameter("gyro_bias_z").value)
        self.q_gyro_deg_s = float(self.get_parameter("q_gyro_deg_s").value)
        self.base_compass_uncertainty_deg = float(self.get_parameter("base_compass_uncertainty_deg").value)
        self.realign_alpha = float(self.get_parameter("realign_alpha").value)

        # 2. Perfiles QoS
        in_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        out_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # 3. Publicadores
        self.heading_pub = self.create_publisher(Float32, "earth_rover/heading", out_qos)
        self.compass_pub = self.create_publisher(
            Float32, "earth_rover/heading_compass", out_qos
        )
        self.uncertainty_pub = self.create_publisher(
            Float32, "earth_rover/heading_uncertainty", out_qos
        )
        self.diag_pub = self.create_publisher(
            String, "earth_rover/heading_diag", out_qos
        )

        # 4. Estado de estimación continua
        self.current_heading_deg: float | None = None
        self._last_compass_heading: float | None = None
        self._last_raw_gyro_z: float = 0.0
        self.heading_uncertainty_deg: float = self.base_compass_uncertainty_deg
        self._last_imu_stamp = None
        self._last_compass_stamp = None
        self._has_imu_updates = False

        # 5. Suscripciones
        self.create_subscription(Odometry, "odometry/global", self._on_odom, in_qos)
        self.create_subscription(Imu, "/imu/data", self._on_imu, sensor_qos)

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_enu = math.atan2(siny_cosp, cosy_cosp)

        # Conversión a rumbo de brújula (0=North, clockwise)
        compass_heading_deg = (90.0 - math.degrees(yaw_enu)) % 360.0
        now = (
            msg.header.stamp
            if (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0)
            else self.get_clock().now().to_msg()
        )

        if self.current_heading_deg is None:
            self.current_heading_deg = compass_heading_deg
            self.heading_uncertainty_deg = self.base_compass_uncertainty_deg
        else:
            # Re-alineación al recibir actualización absoluta del compás/EKF
            if self.realign_alpha >= 1.0:
                self.current_heading_deg = compass_heading_deg
            else:
                diff = (compass_heading_deg - self.current_heading_deg + 540.0) % 360.0 - 180.0
                self.current_heading_deg = (self.current_heading_deg + self.realign_alpha * diff) % 360.0

            # Reseteo de incertidumbre a la base del compás
            self.heading_uncertainty_deg = self.base_compass_uncertainty_deg

        self._last_compass_stamp = now
        self._last_compass_heading = float(compass_heading_deg)

        # Publicar lectura absoluta del compás
        c_msg = Float32()
        c_msg.data = float(compass_heading_deg)
        self.compass_pub.publish(c_msg)

        # Si no hay flujo de IMU disponible, publicamos directamente el compás (compatibilidad total)
        if not self._has_imu_updates:
            self._publish_heading_and_uncertainty()

    def _on_imu(self, msg: Imu):
        has_stamp = (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0)
        stamp_s = (
            (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            if has_stamp
            else (self.get_clock().now().nanoseconds * 1e-9)
        )

        self._last_raw_gyro_z = float(msg.angular_velocity.z)

        if self._last_imu_stamp is None:
            self._last_imu_stamp = stamp_s
            self._has_imu_updates = True
            return

        dt = stamp_s - self._last_imu_stamp
        self._last_imu_stamp = stamp_s

        if dt <= 0.0 or dt > 2.5:
            # Salto temporal anómalo o reinicio
            return

        self._has_imu_updates = True

        if self.current_heading_deg is None:
            return

        # Velocidad angular en Z (base_link, CCW positivo en REP-103)
        omega_z = msg.angular_velocity.z - self.gyro_bias_z

        # Giro CCW (omega_z > 0) -> heading decrece
        # Giro CW  (omega_z < 0) -> heading crece
        delta_heading_deg = - math.degrees(omega_z * dt)
        self.current_heading_deg = (self.current_heading_deg + delta_heading_deg) % 360.0

        # Propagación de incertidumbre acumulada en dead reckoning
        self.heading_uncertainty_deg = math.sqrt(
            self.heading_uncertainty_deg**2 + (self.q_gyro_deg_s**2) * dt
        )

        self._publish_heading_and_uncertainty()

    def _publish_heading_and_uncertainty(self):
        if self.current_heading_deg is None:
            return

        h_msg = Float32()
        h_msg.data = float(self.current_heading_deg)
        self.heading_pub.publish(h_msg)

        u_msg = Float32()
        u_msg.data = float(self.heading_uncertainty_deg)
        self.uncertainty_pub.publish(u_msg)

        # Brief 23 / W.1.1: Diagnóstico de estimación continua
        diag_payload = {
            "heading_compass_last": (
                float(self._last_compass_heading)
                if self._last_compass_heading is not None
                else None
            ),
            "heading_propagated": float(self.current_heading_deg),
            "gyro_z_raw": float(self._last_raw_gyro_z),
            "uncertainty_deg": float(self.heading_uncertainty_deg),
        }
        d_msg = String()
        d_msg.data = json.dumps(diag_payload)
        self.diag_pub.publish(d_msg)


def main(argv=None):
    rclpy.init(args=argv)
    node = EkfHeadingBridge()
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