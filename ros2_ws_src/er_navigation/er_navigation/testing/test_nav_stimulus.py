#!/usr/bin/env python3
"""
Synthetic Navigation Stimulus Node for Earth Rover offline testing.

Publishes simulated GPS, heading, target waypoints, and optional BEV paths
to exercise gps_waypoint_controller states (ALIGN, DRIVE, RECOVERY) without hardware.
Supports fault injection: heading jumps and periodic target republishing.
"""

from __future__ import annotations

import math
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robot_localization.srv import FromLL
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32
import tf2_ros


class TestNavStimulus(Node):
    def __init__(self):
        super().__init__("test_nav_stimulus")

        # 1. Parámetros del estímulo base
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("initial_lat", -34.603700)
        self.declare_parameter("initial_lon", -58.381600)
        self.declare_parameter("target_distance_m", 20.0)
        self.declare_parameter("target_bearing_deg", 0.0)      # 0 = Norte
        self.declare_parameter("initial_heading_deg", 90.0)    # 90 = Este (desalineado 90°)
        self.declare_parameter("simulate_kinematics", True)    # Integra cmd_vel en rumbo y posición
        self.declare_parameter("sim_turn_speed_scale", 1.0)
        self.declare_parameter("publish_bev_path", False)
        self.declare_parameter("bev_path_valid", True)
        self.declare_parameter("bev_path_curvature", 0.0)
        self.declare_parameter("bev_path_length_m", 3.5)
        self.declare_parameter("broadcast_tf", True)

        # 2. Parámetros de inyección de fallas / diagnósticos
        self.declare_parameter("republish_target_period_s", 0.0) # >0 = republica periódicamente
        self.declare_parameter("inject_heading_jump_deg", 0.0)   # Salto brusco a inyectar (ej. 170.0°)
        self.declare_parameter("inject_heading_jump_at_s", 0.0)  # Segundo de la corrida en que se dispara
        self.declare_parameter("jump_is_permanent", False)       # True = el offset se mantiene en el sensor

        self.rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.initial_lat = float(self.get_parameter("initial_lat").value)
        self.initial_lon = float(self.get_parameter("initial_lon").value)
        self.current_lat = self.initial_lat
        self.current_lon = self.initial_lon
        self.target_dist_m = float(self.get_parameter("target_distance_m").value)
        self.target_bearing_deg = float(self.get_parameter("target_bearing_deg").value)
        self.current_heading_deg = float(self.get_parameter("initial_heading_deg").value)
        self.simulate_kinematics = bool(self.get_parameter("simulate_kinematics").value)
        self.sim_turn_scale = float(self.get_parameter("sim_turn_speed_scale").value)
        self.publish_bev_path = bool(self.get_parameter("publish_bev_path").value)
        self.bev_path_valid = bool(self.get_parameter("bev_path_valid").value)
        self.bev_path_curvature = float(self.get_parameter("bev_path_curvature").value)
        self.bev_path_length_m = float(self.get_parameter("bev_path_length_m").value)
        self.broadcast_tf = bool(self.get_parameter("broadcast_tf").value)

        self.republish_target_period = float(self.get_parameter("republish_target_period_s").value)
        self.jump_deg = float(self.get_parameter("inject_heading_jump_deg").value)
        self.jump_at_s = float(self.get_parameter("inject_heading_jump_at_s").value)
        self.jump_is_permanent = bool(self.get_parameter("jump_is_permanent").value)

        # Factores de proyección geodésica plana centrada en el origen
        lat_rad = math.radians(self.initial_lat)
        self._m_per_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        self._m_per_lon = 111412.84 * math.cos(lat_rad)

        # Calcular coordenadas del target
        self.target_lat, self.target_lon = self._calculate_destination(
            self.current_lat, self.current_lon, self.target_dist_m, self.target_bearing_deg
        )

        # Estado cinemático y temporizadores
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self._start_time = self.get_clock().now()
        self._last_tick_time = self._start_time
        self._target_published_count = 0
        self._last_target_pub_time = None
        self._jump_injected = False
        self._permanent_offset_deg = 0.0

        # Broadcaster de Transformaciones TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Servidor de proyección /fromLL para compatibilidad nativa con GlobalPlannerNode
        self.from_ll_srv = self.create_service(FromLL, "/fromLL", self._on_from_ll)

        # Perfiles QoS
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

        # Publicadores
        self.gps_pub = self.create_publisher(NavSatFix, "gps/filtered", sensor_qos)
        self.heading_pub = self.create_publisher(Float32, "earth_rover/heading", sensor_qos)
        self.uncertainty_pub = self.create_publisher(Float32, "earth_rover/heading_uncertainty", sensor_qos)
        self.target_pub = self.create_publisher(NavSatFix, "earth_rover/target_waypoint", reliable_qos)
        self.path_pub = self.create_publisher(Path, "earth_rover/planned_path", sensor_qos)
        self.valid_pub = self.create_publisher(Bool, "earth_rover/planner_valid", sensor_qos)

        # Suscriptor a cmd_vel para cinemática en bucle cerrado
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, reliable_qos)

        # Temporizador principal
        period_s = 1.0 / max(0.1, self.rate_hz)
        self.timer = self.create_timer(period_s, self._timer_callback)

        self.get_logger().info(
            f"TestNavStimulus inicializado | Heading inicial: {self.current_heading_deg:.1f}° | "
            f"Target a {self.target_dist_m:.1f}m rumbo {self.target_bearing_deg:.1f}° | "
            f"Republish target: {self.republish_target_period}s | "
            f"Jump: {self.jump_deg}° a los {self.jump_at_s}s (permanente={self.jump_is_permanent}) | "
            f"TF broadcast={self.broadcast_tf}"
        )

    def _on_from_ll(self, req: FromLL.Request, res: FromLL.Response) -> FromLL.Response:
        dlat = float(req.ll_point.latitude) - self.initial_lat
        dlon = float(req.ll_point.longitude) - self.initial_lon
        res.map_point = Point(
            x=dlon * self._m_per_lon,
            y=dlat * self._m_per_lat,
            z=0.0,
        )
        return res

    def _on_cmd_vel(self, msg: Twist):
        self.last_cmd_v = float(msg.linear.x)
        self.last_cmd_w = float(msg.angular.z)

    @staticmethod
    def _calculate_destination(lat, lon, distance_m, bearing_deg):
        """Calcula destino geodésico dadas distancia y azimut."""
        r = 6371000.0
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        b_rad = math.radians(bearing_deg)
        d_div_r = distance_m / r

        dest_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(d_div_r)
            + math.cos(lat_rad) * math.sin(d_div_r) * math.cos(b_rad)
        )
        dest_lon_rad = lon_rad + math.atan2(
            math.sin(b_rad) * math.sin(d_div_r) * math.cos(lat_rad),
            math.cos(d_div_r) - math.sin(lat_rad) * math.sin(dest_lat_rad),
        )
        return math.degrees(dest_lat_rad), math.degrees(dest_lon_rad)

    def _timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self._last_tick_time).nanoseconds / 1e9
        self._last_tick_time = now
        elapsed_total = (now - self._start_time).nanoseconds / 1e9

        if self.simulate_kinematics and dt > 0.0:
            rad2deg = 180.0 / math.pi
            delta_heading = - self.last_cmd_w * rad2deg * dt * self.sim_turn_scale
            self.current_heading_deg = (self.current_heading_deg + delta_heading) % 360.0

            if self.last_cmd_v > 0.0:
                dist_step = self.last_cmd_v * dt
                self.current_lat, self.current_lon = self._calculate_destination(
                    self.current_lat, self.current_lon, dist_step, self.current_heading_deg
                )

        # Inyección de salto magnético
        published_heading = self.current_heading_deg + self._permanent_offset_deg
        if (
            self.jump_deg != 0.0
            and self.jump_at_s > 0.0
            and elapsed_total >= self.jump_at_s
            and not self._jump_injected
        ):
            self.get_logger().warn(
                f"[STIMULUS] ¡Inyectando salto de heading de {self.jump_deg:+.1f}° a los {elapsed_total:.2f}s! "
                f"Heading previo={self.current_heading_deg:.1f}° -> Salto={self.current_heading_deg + self.jump_deg:.1f}°"
            )
            published_heading = (self.current_heading_deg + self.jump_deg) % 360.0
            if self.jump_is_permanent:
                self._permanent_offset_deg = self.jump_deg
            self._jump_injected = True

        # 1. Publicar GPS
        gps_msg = NavSatFix()
        gps_msg.header.stamp = now.to_msg()
        gps_msg.header.frame_id = "earth_rover_gps"
        gps_msg.latitude = self.current_lat
        gps_msg.longitude = self.current_lon
        gps_msg.status.status = 0
        self.gps_pub.publish(gps_msg)

        # 2. Publicar Heading
        head_msg = Float32()
        head_msg.data = float(published_heading % 360.0)
        self.heading_pub.publish(head_msg)

        u_msg = Float32()
        u_msg.data = 3.0
        self.uncertainty_pub.publish(u_msg)

        # 2b. Broadcast TF (map -> base_link) para localización sintética
        if self.broadcast_tf:
            dlat = self.current_lat - self.initial_lat
            dlon = self.current_lon - self.initial_lon
            tx = dlon * self._m_per_lon
            ty = dlat * self._m_per_lat
            yaw_rad = math.radians(90.0 - (published_heading % 360.0))

            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            t.transform.translation.x = float(tx)
            t.transform.translation.y = float(ty)
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = math.sin(yaw_rad * 0.5)
            t.transform.rotation.w = math.cos(yaw_rad * 0.5)
            self.tf_broadcaster.sendTransform(t)

            # TF estático/dinámico base_link -> earth_rover_gps
            t_gps = TransformStamped()
            t_gps.header.stamp = now.to_msg()
            t_gps.header.frame_id = "base_link"
            t_gps.child_frame_id = "earth_rover_gps"
            t_gps.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_gps)

            # TF estático/dinámico base_link -> earth_rover_front_camera
            t_cam = TransformStamped()
            t_cam.header.stamp = now.to_msg()
            t_cam.header.frame_id = "base_link"
            t_cam.child_frame_id = "earth_rover_front_camera"
            t_cam.transform.translation.x = 0.15
            t_cam.transform.translation.z = 0.20
            t_cam.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t_cam)

        # 3. Publicar Target (inicial o periódico según parámetro)
        should_publish_target = False
        if self.republish_target_period > 0.0:
            if (
                self._last_target_pub_time is None
                or (now - self._last_target_pub_time).nanoseconds / 1e9 >= self.republish_target_period
            ):
                should_publish_target = True
        else:
            if self._target_published_count < 3:
                should_publish_target = True

        if should_publish_target:
            target_msg = NavSatFix()
            target_msg.header.stamp = now.to_msg()
            target_msg.header.frame_id = "earth_rover_gps"
            target_msg.latitude = self.target_lat
            target_msg.longitude = self.target_lon
            self.target_pub.publish(target_msg)
            self._target_published_count += 1
            self._last_target_pub_time = now

        # 4. Publicar Path BEV opcional
        if self.publish_bev_path:
            path_msg = Path()
            path_msg.header.stamp = now.to_msg()
            path_msg.header.frame_id = "base_link"

            n_points = 20
            for i in range(n_points):
                s = (i / (n_points - 1)) * self.bev_path_length_m
                px = s
                py = self.bev_path_curvature * (s**2)

                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(px)
                pose.pose.position.y = float(py)
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)

            valid_msg = Bool()
            valid_msg.data = self.bev_path_valid
            self.valid_pub.publish(valid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestNavStimulus()
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
