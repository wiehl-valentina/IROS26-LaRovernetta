#!/usr/bin/env python3
"""ROS2 bridge for the Earth Rovers SDK.

Bridges the SDK's HTTP/WebSocket API into standard ROS2 topics:

  Subscribes
    /cmd_vel (geometry_msgs/Twist)   -> POST /control (latest-wins at 10 Hz,
                                        automatic stop when cmd_vel goes quiet)

  Publishes
    /earth_rover/front/image_raw (sensor_msgs/Image)  <- GET /feed (MJPEG)
    /earth_rover/gps (sensor_msgs/NavSatFix)          <- WS /ws/data
    /earth_rover/imu (sensor_msgs/Imu)                <- WS /ws/data
    /earth_rover/battery (sensor_msgs/BatteryState)   <- WS /ws/data
    /earth_rover/heading (std_msgs/Float32)           <- WS /ws/data
    /earth_rover/odom (nav_msgs/Odometry)             <- WS /ws/data (rpms+gyros)
    TF odom -> base_link                              <- lo mismo que /earth_rover/odom
    TF base_link -> earth_rover_front_camera (estatico, una vez al arrancar)

Usage:
    # SDK running on localhost:8000 (mission started if required)
    ros2 run <your_pkg> earth_rover_bridge.py
    # or directly:
    python3 earth_rover_bridge.py --ros-args -p sdk_url:=http://localhost:8000

Dependencies (besides a ROS2 distro with rclpy + cv_bridge + tf2_ros):
    pip install requests websocket-client opencv-python

--------------------------------------------------------------------------
NOTA sobre /earth_rover/odom (agregado para mapeo con RTAB-Map):

Se agrego odometria por rueda+giroscopo, con la MISMA logica y valores por
defecto que usa `genie_rover/odometry.py` del lado no-ROS del proyecto (ver
`differential_odometry.py`, en esta misma carpeta) -- asi el marco de
referencia (x adelante, y izquierda, theta antihorario) coincide con el que
ya usan PersistentMap/planner/mission de genie_rover, y el mapa que arme
RTAB-Map/PersistentMap con estos datos es directamente compatible.

Los parametros `wheel_radius_m`, `track_width_m`, `left_rpm_indices`,
`right_rpm_indices`, `rotation_sign` deben coincidir con los que ya tengas
calibrados en tu config de genie_rover (ver `genie/configs/*.yaml`,
seccion `odometry:`) -- si difieren, la odometria de este bridge y la que
arma tu programa de mapeo van a divergir.
"""

import json
import math
import threading
import time

import cv2
import rclpy
import requests
import websocket
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry as OdometryMsg
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, Image, Imu, NavSatFix
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from differential_odometry import Odometry, OdometryConfig

CONTROL_RATE_HZ = 10.0
CMD_VEL_TIMEOUT_S = 0.5  # no cmd_vel for this long -> send stop
# /control dispatches without waiting for the rover's acknowledgement, so
# responses are fast; 1s leaves margin without stalling the 10 Hz loop.
CONTROL_HTTP_TIMEOUT_S = 1.0


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) para una rotacion pura en yaw (robot plano)."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


class EarthRoverBridge(Node):
    def __init__(self):
        super().__init__("earth_rover_bridge")
        self.declare_parameter("sdk_url", "http://localhost:8000")
        self.declare_parameter("feed_fps", 15)
        self.sdk_url = self.get_parameter("sdk_url").value.rstrip("/")
        self.feed_fps = int(self.get_parameter("feed_fps").value)

        # ---- odometria (rueda+giro) ---------------------------------------
        # Mismos defaults que genie/configs/frodobot_rover.yaml -> odometry:
        # si tu rover esta calibrado distinto, pasalos por --ros-args -p ...
        self.declare_parameter("wheel_radius_m", 0.045)
        self.declare_parameter("track_width_m", 0.15)
        self.declare_parameter("left_rpm_indices", [0, 2])
        self.declare_parameter("right_rpm_indices", [1, 3])
        self.declare_parameter("rotation_sign", -1.0)
        self.declare_parameter("use_gyro_for_rotation", True)
        self.declare_parameter("gps_correction", True)
        self.declare_parameter("publish_odom_tf", True)
        self.odometry = Odometry(OdometryConfig(
            wheel_radius_m=float(self.get_parameter("wheel_radius_m").value),
            track_width_m=float(self.get_parameter("track_width_m").value),
            left_rpm_indices=tuple(self.get_parameter("left_rpm_indices").value),
            right_rpm_indices=tuple(self.get_parameter("right_rpm_indices").value),
            rotation_sign=float(self.get_parameter("rotation_sign").value),
            use_gyro_for_rotation=bool(self.get_parameter("use_gyro_for_rotation").value),
            gps_correction=bool(self.get_parameter("gps_correction").value),
        ))
        self.publish_odom_tf = bool(self.get_parameter("publish_odom_tf").value)

        # ---- extrinsecos camara (TF estatico base_link -> camara) ---------
        # Mismo significado que camera.height_m / camera.pitch_down_deg en
        # los configs de genie_rover (perception.py). Ajustalos a tu montaje.
        self.declare_parameter("camera_height_m", 0.20)
        self.declare_parameter("camera_forward_offset_m", 0.05)
        self.declare_parameter("camera_pitch_down_deg", 15.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("camera_frame", "earth_rover_front_camera")
        self.base_frame = self.get_parameter("base_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.bridge = CvBridge()
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        command_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        self.image_pub = self.create_publisher(
            Image, "earth_rover/front/image_raw", sensor_qos
        )
        self.gps_pub = self.create_publisher(NavSatFix, "earth_rover/gps", sensor_qos)
        self.imu_pub = self.create_publisher(Imu, "earth_rover/imu", sensor_qos)
        self.battery_pub = self.create_publisher(
            BatteryState, "earth_rover/battery", sensor_qos
        )
        self.heading_pub = self.create_publisher(
            Float32, "earth_rover/heading", sensor_qos
        )
        self.odom_pub = self.create_publisher(OdometryMsg, "earth_rover/odom", sensor_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_sent = False
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._send_static_camera_tf()

        # cmd_vel -> /control: keep only the latest command, send at a fixed
        # rate, and stop the rover if commands stop arriving.
        self._latest_cmd = None
        self._last_cmd_at = 0.0
        self._stopped = True
        self._cmd_lock = threading.Lock()
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, command_qos)

        self._session = requests.Session()
        self._running = True
        self._stop_event = threading.Event()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        threading.Thread(target=self._feed_loop, daemon=True).start()
        threading.Thread(target=self._telemetry_loop, daemon=True).start()

        self.get_logger().info(f"Bridging Earth Rovers SDK at {self.sdk_url}")

    # ------------------------------------------------------------- control

    def _on_cmd_vel(self, msg: Twist):
        with self._cmd_lock:
            self._latest_cmd = {
                # SDK expects -1..1; Twist for this rover is already normalized.
                "linear": max(-1.0, min(1.0, msg.linear.x)),
                "angular": max(-1.0, min(1.0, msg.angular.z)),
            }
            self._last_cmd_at = time.monotonic()
            self._stopped = False

    def _control_tick(self):
        with self._cmd_lock:
            quiet = time.monotonic() - self._last_cmd_at > CMD_VEL_TIMEOUT_S
            if self._latest_cmd is None or (quiet and self._stopped):
                return
            command = {"linear": 0, "angular": 0} if quiet else dict(self._latest_cmd)
            last_cmd_at = self._last_cmd_at
        try:
            response = self._session.post(
                f"{self.sdk_url}/control",
                json={"command": command},
                timeout=CONTROL_HTTP_TIMEOUT_S,
            )
            response.raise_for_status()
            if quiet:
                # A successful HTTP response means the stop was dispatched.
                # The SDK watchdog independently tracks peer confirmation and
                # retries zero if this asynchronous delivery later fails.
                with self._cmd_lock:
                    if (
                        self._last_cmd_at == last_cmd_at
                        and time.monotonic() - self._last_cmd_at > CMD_VEL_TIMEOUT_S
                    ):
                        self._stopped = True
        except requests.RequestException as e:
            self.get_logger().warning(f"/control failed: {e}", throttle_duration_sec=5)

    def _control_loop(self):
        interval = 1.0 / CONTROL_RATE_HZ
        deadline = time.monotonic()
        while self._running and rclpy.ok():
            self._control_tick()
            deadline += interval
            wait = max(0.0, deadline - time.monotonic())
            if self._stop_event.wait(wait):
                break
            if time.monotonic() - deadline > interval:
                deadline = time.monotonic()

    # ---------------------------------------------------------------- feed

    def _feed_loop(self):
        url = f"{self.sdk_url}/feed?view=front&fps={self.feed_fps}"
        while self._running and rclpy.ok():
            capture = cv2.VideoCapture(url)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                self.get_logger().warning(
                    "/feed not available, retrying in 3s", throttle_duration_sec=10
                )
                time.sleep(3)
                continue
            self.get_logger().info("Connected to /feed")
            while self._running and rclpy.ok():
                ok, frame = capture.read()
                if not ok:
                    break
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.camera_frame
                self.image_pub.publish(msg)
            capture.release()
            time.sleep(1)

    # ----------------------------------------------------------- telemetry

    def _telemetry_loop(self):
        ws_url = self.sdk_url.replace("http", "ws", 1) + "/ws/data"
        while self._running and rclpy.ok():
            ws = None
            try:
                ws = websocket.create_connection(ws_url, timeout=10)
                self.get_logger().info("Connected to /ws/data")
                while self._running and rclpy.ok():
                    msg = json.loads(ws.recv())
                    if msg.get("type") in ("snapshot", "telemetry") and msg.get("data"):
                        self._publish_telemetry(msg["data"])
            except Exception as e:
                self.get_logger().warning(
                    f"/ws/data reconnecting: {e}", throttle_duration_sec=10
                )
                time.sleep(2)
            finally:
                if ws is not None:
                    ws.close()

    def _publish_telemetry(self, data: dict):
        now = self.get_clock().now().to_msg()

        lat, lng = data.get("latitude"), data.get("longitude")
        if lat is not None and lng is not None:
            gps = NavSatFix()
            gps.header.stamp = now
            gps.header.frame_id = "earth_rover_gps"
            gps.latitude = float(lat)
            gps.longitude = float(lng)
            self.gps_pub.publish(gps)

        orientation = data.get("orientation")
        if orientation is not None:
            heading = Float32()
            heading.data = float(orientation)
            self.heading_pub.publish(heading)

        battery = data.get("battery")
        if battery is not None:
            batt = BatteryState()
            batt.header.stamp = now
            batt.percentage = float(battery) / 100.0
            batt.present = True
            self.battery_pub.publish(batt)

        # accels/gyros/mags/rpms are arrays of samples: [.., ..., unix_ts]
        accels, gyros = data.get("accels") or [], data.get("gyros") or []
        if accels or gyros:
            imu = Imu()
            imu.header.stamp = now
            imu.header.frame_id = "earth_rover_imu"
            if accels:
                sample = accels[-1]
                imu.linear_acceleration.x = float(sample[0])
                imu.linear_acceleration.y = float(sample[1])
                imu.linear_acceleration.z = float(sample[2])
            if gyros:
                sample = gyros[-1]
                imu.angular_velocity.x = math.radians(float(sample[0]))
                imu.angular_velocity.y = math.radians(float(sample[1]))
                imu.angular_velocity.z = math.radians(float(sample[2]))
            self.imu_pub.publish(imu)

        # ---- odometria: alimenta con el mismo lote crudo que usa el resto
        # de la telemetria (necesita "rpms" y "gyros", que el SDK ya manda) --
        if data.get("rpms") or data.get("gyros"):
            self._publish_odometry(data, now)

    def _publish_odometry(self, data: dict, stamp) -> None:
        pose = self.odometry.update(data)

        qx, qy, qz, qw = yaw_to_quaternion(pose.theta)

        odom = OdometryMsg()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)

        if self.publish_odom_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = pose.x
            tf.transform.translation.y = pose.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

    def _send_static_camera_tf(self) -> None:
        """TF fijo base_link -> camara, a partir de height/offset/pitch.

        Misma convencion que camera_pose_from_height_pitch() en
        genie_rover/perception.py: camara mirando hacia adelante, inclinada
        pitch_down_deg hacia el piso.
        """
        height = float(self.get_parameter("camera_height_m").value)
        forward = float(self.get_parameter("camera_forward_offset_m").value)
        pitch = math.radians(float(self.get_parameter("camera_pitch_down_deg").value))

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.base_frame
        tf.child_frame_id = self.camera_frame
        tf.transform.translation.x = forward
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = height
        # Rotacion: eje optico de la camara apuntando "adelante y abajo".
        # Pitch puro alrededor del eje Y de base_link (mirando hacia abajo
        # es pitch NEGATIVO en la convencion REP-103 estandar de ROS).
        half = -pitch / 2.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = math.sin(half)
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = math.cos(half)
        self._static_tf_broadcaster.sendTransform(tf)
        self.get_logger().info(
            f"TF estatico {self.base_frame} -> {self.camera_frame}: "
            f"height={height:.3f}m forward={forward:.3f}m pitch={math.degrees(pitch):.1f}deg"
        )

    def destroy_node(self):
        self._running = False
        self._stop_event.set()
        self._control_thread.join(timeout=1.0)
        # Do not leave the last motion command active when the bridge exits.
        for _ in range(3):
            try:
                response = self._session.post(
                    f"{self.sdk_url}/control",
                    json={"command": {"linear": 0, "angular": 0}},
                    timeout=CONTROL_HTTP_TIMEOUT_S,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                continue
        self._session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EarthRoverBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # ROS signal handling may already have shut the context down.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
