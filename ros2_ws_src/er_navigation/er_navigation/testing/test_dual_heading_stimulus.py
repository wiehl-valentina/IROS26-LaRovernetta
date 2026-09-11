#!/usr/bin/env python3
"""
test_dual_heading_stimulus.py

Simulates sensor stimulus to reproduce and benchmark the dual publisher conflict
on the topic 'earth_rover/heading'.

Modes:
  1. dual:      Simulates both EKF (10Hz, smooth) and Raw Compass (0.5Hz-1Hz bursty, noisy).
  2. ekf_only:  Simulates only the EKF fused heading (10Hz, smooth).
  3. raw_only:  Simulates only the raw compass heading (0.67Hz, noisy).
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist


def angle_diff_deg(a: float, b: float) -> float:
    """Calculates shortest signed difference a - b in [-180, 180]."""
    return (a - b + 540.0) % 360.0 - 180.0


@dataclass
class SimulationMetrics:
    mode_name: str
    duration_s: float = 0.0
    total_heading_msgs: int = 0
    raw_heading_msgs: int = 0
    ekf_heading_msgs: int = 0
    control_ticks: int = 0
    align_ticks: int = 0
    drive_ticks: int = 0
    recovery_ticks: int = 0
    mode_switches: int = 0
    heading_errors: list[float] = field(default_factory=list)
    current_headings: list[float] = field(default_factory=list)
    heading_deltas: list[float] = field(default_factory=list)
    cmd_linear_x: list[float] = field(default_factory=list)
    cmd_angular_z: list[float] = field(default_factory=list)
    status_logs: list[str] = field(default_factory=list)


class DualHeadingStimulusNode(Node):
    def __init__(self, mode: str = "dual", duration_s: float = 30.0, raw_noise_deg: float = 8.0, raw_glitch_prob: float = 0.15):
        super().__init__("dual_heading_stimulus")
        self.mode = mode
        self.duration_s = duration_s
        self.raw_noise_deg = raw_noise_deg
        self.raw_glitch_prob = raw_glitch_prob

        # QoS
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

        # Publishers
        self.ekf_heading_pub = self.create_publisher(Float32, "earth_rover/heading", sensor_qos)
        self.raw_heading_pub = self.create_publisher(Float32, "earth_rover/heading_raw", sensor_qos)
        # Legacy/Buggy publisher for prefix reproduction
        self.legacy_conflict_pub = self.create_publisher(Float32, "earth_rover/heading", sensor_qos)

        self.gps_pub = self.create_publisher(NavSatFix, "gps/filtered", sensor_qos)
        self.target_pub = self.create_publisher(NavSatFix, "earth_rover/target_waypoint", reliable_qos)

        # Subscribers to evaluate controller response
        self.create_subscription(String, "earth_rover/control_debug", self._on_control_debug, sensor_qos)
        self.create_subscription(String, "earth_rover/waypoint_status", self._on_status, reliable_qos)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, reliable_qos)
        self.create_subscription(Float32, "earth_rover/heading", self._on_heading_echo, sensor_qos)

        # Simulation Ground Truth State
        # Rover at (lat=37.77490, lon=-122.41940)
        # Target at (lat=37.77530, lon=-122.41890) -> ~50m away at bearing ~43°
        self.sim_lat = 37.77490
        self.sim_lon = -122.41940
        self.target_lat = 37.77530
        self.target_lon = -122.41890

        # True heading starts at 25.0° (target bearing is ~43.4°, so initial error is ~+18.4°, right at align threshold!)
        self.true_heading = 25.0
        self.last_heading_val = None

        # Rover Kinematics
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_kinematics_update = time.time()

        # Metrics collection
        self.metrics = SimulationMetrics(mode_name=mode)
        self.last_mode = None

        # Timers
        # GPS at 5 Hz
        self.create_timer(0.2, self._tick_gps)
        # Target latch at 1 Hz
        self.create_timer(1.0, self._tick_target)
        # Kinematics integration at 50 Hz
        self.create_timer(0.02, self._tick_kinematics)

        # Heading publishers based on mode
        if self.mode in ("dual", "postfix_dual", "prefix_dual", "ekf_only"):
            # EKF 10 Hz (every 100ms)
            self.create_timer(0.10, self._tick_ekf_heading)

        if self.mode in ("dual", "postfix_dual", "prefix_dual", "raw_only"):
            # Raw compass bursty (every 1.5s, matching telemetry packet interval)
            self.create_timer(1.5, self._tick_raw_heading)

        self.start_time = time.time()
        self.get_logger().info(
            f"=== Stimulus Node Iniciado | Modo: {self.mode.upper()} | Duración: {self.duration_s}s | Ruido Brújula: ±{self.raw_noise_deg}° ==="
        )

    def _tick_kinematics(self):
        now = time.time()
        dt = now - self.last_kinematics_update
        self.last_kinematics_update = now

        # Simple rover dynamic simulation
        # In gps_waypoint_controller, invert_angular is true, cmd_w is sent to bridge
        # Forward turn speed is ~0.7 rad/s ~= 40 deg/s
        turn_rate_deg_s = 40.0 * (-self.last_cmd_w)
        self.true_heading = (self.true_heading + turn_rate_deg_s * dt) % 360.0

        # Linear motion
        forward_m_s = self.last_cmd_v * 0.4
        if forward_m_s > 0:
            heading_rad = math.radians(self.true_heading)
            # Rough lat/lon displacement (1 deg lat ~ 111139 m, 1 deg lon ~ 88000 m)
            dlat = (forward_m_s * dt * math.cos(heading_rad)) / 111139.0
            dlon = (forward_m_s * dt * math.sin(heading_rad)) / 88000.0
            self.sim_lat += dlat
            self.sim_lon += dlon

    def _tick_gps(self):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "earth_rover_gps"
        msg.latitude = self.sim_lat
        msg.longitude = self.sim_lon
        self.gps_pub.publish(msg)

    def _tick_target(self):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.latitude = self.target_lat
        msg.longitude = self.target_lon
        self.target_pub.publish(msg)

    def _tick_ekf_heading(self):
        # EKF filtered output: very smooth, low noise (sigma = 0.4°) -> earth_rover/heading
        noise = random.gauss(0.0, 0.4)
        ekf_val = (self.true_heading + noise) % 360.0

        msg = Float32()
        msg.data = float(ekf_val)
        self.ekf_heading_pub.publish(msg)
        self.metrics.ekf_heading_msgs += 1

    def _tick_raw_heading(self):
        # Raw compass output: noisy (sigma = raw_noise_deg) + occasional magnetic glitch/jump
        noise = random.gauss(0.0, self.raw_noise_deg)
        glitch = 0.0
        if random.random() < self.raw_glitch_prob:
            # Magnetic spike near chassis/motors (e.g. ±20° to ±40°)
            glitch = random.choice([-1.0, 1.0]) * random.uniform(15.0, 35.0)

        raw_val = (self.true_heading + noise + glitch) % 360.0
        msg = Float32()
        msg.data = float(raw_val)

        if self.mode == "prefix_dual":
            # Buggy pre-fix behavior: raw published directly to earth_rover/heading
            self.legacy_conflict_pub.publish(msg)
        else:
            # Fixed post-fix behavior: raw published to separate debug topic earth_rover/heading_raw
            self.raw_heading_pub.publish(msg)

        self.metrics.raw_heading_msgs += 1

    def _on_heading_echo(self, msg: Float32):
        self.metrics.total_heading_msgs += 1
        val = msg.data
        if self.last_heading_val is not None:
            delta = abs(angle_diff_deg(val, self.last_heading_val))
            self.metrics.heading_deltas.append(delta)
        self.last_heading_val = val

    def _on_cmd_vel(self, msg: Twist):
        self.last_cmd_v = msg.linear.x
        self.last_cmd_w = msg.angular.z
        self.metrics.cmd_linear_x.append(msg.linear.x)
        self.metrics.cmd_angular_z.append(msg.angular.z)

    def _on_status(self, msg: String):
        self.metrics.status_logs.append(msg.data)

    def _on_control_debug(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.metrics.control_ticks += 1
            mode = data.get("mode")
            if mode == "ALIGN":
                self.metrics.align_ticks += 1
            elif mode == "DRIVE":
                self.metrics.drive_ticks += 1
            elif mode == "RECOVERY":
                self.metrics.recovery_ticks += 1

            if self.last_mode is not None and mode != self.last_mode:
                self.metrics.mode_switches += 1
            self.last_mode = mode

            h_err = data.get("heading_error")
            if h_err is not None:
                self.metrics.heading_errors.append(h_err)

            c_h = data.get("current_heading")
            if c_h is not None:
                self.metrics.current_headings.append(c_h)
        except Exception:
            pass


def run_benchmark(mode: str, duration_s: float = 30.0) -> SimulationMetrics:
    rclpy.init()
    node = DualHeadingStimulusNode(mode=mode, duration_s=duration_s)
    start_t = time.time()
    try:
        while rclpy.ok() and (time.time() - start_t) < duration_s:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.metrics.duration_s = time.time() - start_t
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
    return node.metrics


def main(args=None):
    parser = argparse.ArgumentParser(description="Test dual heading stimulus node")
    parser.add_argument("--mode", choices=["dual", "postfix_dual", "prefix_dual", "ekf_only", "raw_only"], default="postfix_dual")
    parser.add_argument("--duration", type=float, default=30.0)
    parsed_args, _ = parser.parse_known_args(args)

    metrics = run_benchmark(parsed_args.mode, parsed_args.duration)
    print(f"\n--- Resumen Simulación [{metrics.mode_name}] ---")
    print(f"Mensajes Heading Totales: {metrics.total_heading_msgs} (EKF: {metrics.ekf_heading_msgs}, Raw: {metrics.raw_heading_msgs})")
    print(f"Ticks de Control: {metrics.control_ticks} (ALIGN: {metrics.align_ticks}, DRIVE: {metrics.drive_ticks})")
    print(f"Cambios de Modo ALIGN/DRIVE (Jitter/Thrashing): {metrics.mode_switches}")
    if metrics.heading_errors:
        mean_err = sum(metrics.heading_errors) / len(metrics.heading_errors)
        std_err = math.sqrt(sum((x - mean_err) ** 2 for x in metrics.heading_errors) / len(metrics.heading_errors))
        print(f"Heading Error: media = {mean_err:.2f}°, std = {std_err:.2f}°, min = {min(metrics.heading_errors):.2f}°, max = {max(metrics.heading_errors):.2f}°")
    if metrics.heading_deltas:
        mean_delta = sum(metrics.heading_deltas) / len(metrics.heading_deltas)
        max_delta = max(metrics.heading_deltas)
        print(f"Delta entre mensajes sucesivos en topic: media = {mean_delta:.2f}°, max = {max_delta:.2f}°")


if __name__ == "__main__":
    main()
