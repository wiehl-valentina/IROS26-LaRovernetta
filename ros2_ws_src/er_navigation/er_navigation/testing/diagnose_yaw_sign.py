#!/usr/bin/env python3
"""
diagnose_yaw_sign.py — Herramienta de diagnóstico de signo de yaw y propagación inercial (Brief 23 / W.1)

Permite:
1. Comandar una ráfaga de giro controlada (ej: w=+0.70 durante 1.0s con rover quieto).
2. Capturar con precisión temporal:
   - heading_compass_last (del compás/EKF)
   - heading_propagated (de ekf_heading_bridge)
   - gyro_z_raw (de /imu/data)
3. Evaluar la correlación física y determinar si el signo en ekf_heading_bridge.py está invertido.

Uso:
  ros2 run er_navigation diagnose_yaw_sign --burst --w 0.70 --duration 1.0
  ros2 run er_navigation diagnose_yaw_sign --monitor --duration 10.0
"""

import argparse
import json
import math
import sys
import time
from typing import List, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String


class YawSignDiagnostic(Node):
    def __init__(self, target_w: float = 0.70, burst_duration_s: float = 3.0):
        super().__init__("yaw_sign_diagnostic")
        self.target_w = target_w
        self.burst_duration_s = burst_duration_s

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Publicador de comandos de velocidad
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", reliable_qos)

        # Suscripciones
        self.create_subscription(Float32, "earth_rover/heading", self._on_prop_heading, sensor_qos)
        self.create_subscription(Float32, "earth_rover/heading_raw", self._on_raw_heading, sensor_qos)
        self.create_subscription(Float32, "earth_rover/heading_compass", self._on_compass_heading, sensor_qos)
        self.create_subscription(Imu, "/imu/data", self._on_imu, sensor_qos)
        self.create_subscription(String, "earth_rover/control_debug", self._on_control_debug, sensor_qos)

        # Variables de captura
        self.last_prop_heading: float | None = None
        self.last_compass_heading: float | None = None
        self.last_gyro_z_raw: float | None = None

        self.samples: List[Dict[str, Any]] = []
        self.is_recording = False
        self.start_record_t = 0.0

    def _on_prop_heading(self, msg: Float32):
        self.last_prop_heading = float(msg.data)
        self._record_sample(event="prop_heading")

    def _on_raw_heading(self, msg: Float32):
        self.last_compass_heading = float(msg.data)
        self._record_sample(event="compass_raw")

    def _on_compass_heading(self, msg: Float32):
        self.last_compass_heading = float(msg.data)
        self._record_sample(event="compass_bridge")

    def _on_imu(self, msg: Imu):
        self.last_gyro_z_raw = float(msg.angular_velocity.z)
        self._record_sample(event="imu")

    def _on_control_debug(self, msg: String):
        try:
            data = json.loads(msg.data)
            if data.get("heading_compass_last") is not None:
                self.last_compass_heading = float(data["heading_compass_last"])
            if data.get("heading_propagated") is not None:
                self.last_prop_heading = float(data["heading_propagated"])
            if data.get("gyro_z_raw") is not None:
                self.last_gyro_z_raw = float(data["gyro_z_raw"])
            self._record_sample(event="control_debug")
        except Exception:
            pass

    def _record_sample(self, event: str):
        if not self.is_recording:
            return
        t_now = time.time() - self.start_record_t
        self.samples.append({
            "t": t_now,
            "event": event,
            "compass": self.last_compass_heading,
            "prop": self.last_prop_heading,
            "gyro_z": self.last_gyro_z_raw,
        })

    def stop_robot(self):
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        for _ in range(3):
            self.cmd_pub.publish(stop_twist)
            time.sleep(0.05)


def angle_diff_deg(a: float, b: float) -> float:
    """Retorna diferencia angular a - b en el rango [-180, 180]."""
    return (a - b + 540.0) % 360.0 - 180.0


def run_diagnostic(args):
    rclpy.init()
    diag = YawSignDiagnostic(target_w=args.w, burst_duration_s=args.duration)

    print("=" * 70)
    print(" Earth Rover - Diagnóstico de Signo de Yaw (Brief 23 / W.1)")
    print("=" * 70)
    print(f"Comando configurado: w = {args.w:+.2f} | Duración: {args.duration:.2f} s")
    print("Esperando telemetría inicial (compás / heading / IMU)...")

    # Esperar hasta 8 segundos a que haya lecturas tanto de compás como de propagado
    t_wait_start = time.time()
    while time.time() - t_wait_start < 8.0:
        rclpy.spin_once(diag, timeout_sec=0.1)
        if diag.last_compass_heading is not None and diag.last_prop_heading is not None:
            break

    print(f"Estado inicial capturado:")
    print(f"  - Compás inicial:    {diag.last_compass_heading}°")
    print(f"  - Propagado inicial: {diag.last_prop_heading}°")
    print(f"  - Gyro Z inicial:    {diag.last_gyro_z_raw} rad/s")

    diag.is_recording = True
    diag.start_record_t = time.time()

    initial_compass = diag.last_compass_heading
    initial_prop = diag.last_prop_heading

    if args.burst:
        print("\n>>> INICIANDO RÁFAGA DE GIRO CONTROLADA <<<")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = float(args.w)

        t_burst_start = time.time()
        while time.time() - t_burst_start < args.duration:
            diag.cmd_pub.publish(twist)
            rclpy.spin_once(diag, timeout_sec=0.05)

        print(">>> RÁFAGA COMPLETADA — FRENANDO Y OBSERVANDO RESPUESTA (5.0s) <<<")
        diag.stop_robot()

        # Monitorear 5 segundos tras la ráfaga para capturar el nuevo reporte del compás (~2s)
        t_post_start = time.time()
        while time.time() - t_post_start < 5.0:
            rclpy.spin_once(diag, timeout_sec=0.1)
    else:
        print(f"\n>>> MODO MONITOREO PASIVO ({args.duration}s) <<<")
        t_mon_start = time.time()
        while time.time() - t_mon_start < args.duration:
            rclpy.spin_once(diag, timeout_sec=0.1)

    diag.is_recording = False
    diag.stop_robot()

    final_compass = diag.last_compass_heading
    final_prop = diag.last_prop_heading

    print("\n" + "=" * 70)
    print(" ANÁLISIS DE CORRELACIÓN Y SIGNO (Brief 23 / W.1.3)")
    print("=" * 70)

    # 1. Delta observado en compás y propagado
    d_compass = (
        angle_diff_deg(final_compass, initial_compass)
        if (final_compass is not None and initial_compass is not None)
        else None
    )
    d_prop = (
        angle_diff_deg(final_prop, initial_prop)
        if (final_prop is not None and initial_prop is not None)
        else None
    )

    print(f"Compás:    Inicial = {initial_compass}° -> Final = {final_compass}° | Δ = {d_compass:+.1f}°" if d_compass is not None else "Compás: datos insuficientes")
    print(f"Propagado: Inicial = {initial_prop}° -> Final = {final_prop}° | Δ = {d_prop:+.1f}°" if d_prop is not None else "Propagado: datos insuficientes")

    # 2. Análisis de muestras de giróscopo durante la ráfaga
    gyro_samples = [s["gyro_z"] for s in diag.samples if s["gyro_z"] is not None]
    if gyro_samples:
        avg_gyro_z = sum(gyro_samples) / len(gyro_samples)
        max_gyro_z = max(gyro_samples)
        min_gyro_z = min(gyro_samples)
        print(f"Giróscopo Z: promedio = {avg_gyro_z:+.4f} rad/s | min = {min_gyro_z:+.4f} | max = {max_gyro_z:+.4f}")
    else:
        avg_gyro_z = None
        print("Giróscopo Z: No se recibieron muestras de IMU en /imu/data")

    # 3. Diagnóstico formal de convención
    print("\nEvaluación formal de convención:")
    # Convención de compás: 0=N, 90=E, 180=S, 270=W.
    # Girar a la IZQUIERDA (CCW) -> rumbo DECRECE (ej: 90° -> 80° -> 70°).
    # Girar a la DERECHA   (CW)  -> rumbo CRECE   (ej: 0° -> 10° -> 20°).
    print("  - Giro a la IZQUIERDA (CCW): El rumbo de brújula debe DECRECER (Δ < 0).")
    print("  - Giro a la DERECHA   (CW):  El rumbo de brújula debe CRECER   (Δ > 0).")

    if d_compass is not None and d_prop is not None:
        sign_compass = 1 if d_compass > 0 else (-1 if d_compass < 0 else 0)
        sign_prop = 1 if d_prop > 0 else (-1 if d_prop < 0 else 0)

        if sign_compass != 0 and sign_prop != 0:
            if sign_compass == sign_prop:
                print(f"\n[RESULTADO: SIGNO CONSISTENTE] Δcompass ({d_compass:+.1f}°) y Δprop ({d_prop:+.1f}°) tienen el MISMO SIGNO.")
                print("  -> La fórmula actual en ekf_heading_bridge.py propaga en la dirección correcta.")
            else:
                print(f"\n[RESULTADO: SIGNO INVERTIDO (BUG CONFIRMADO)]")
                print(f"  -> Δcompass = {d_compass:+.1f}°")
                print(f"  -> Δprop    = {d_prop:+.1f}°")
                print("  -> ¡El compás físico rotó en un sentido y la propagación inercial rotó en el sentido contrario!")
                print("  -> Proceder con W.2.1: Invertir el signo en ekf_heading_bridge.py:")
                print("     heading(t) = (heading_compass + ∫ω_z dt) mod 360°")
    else:
        print("\n[RESULTADO: INDETERMINADO - Sin datos suficientes de compás]")
        print("Asegurate de que el rover y el bridge de telemetría estén activos.")

    diag.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Herramienta de Diagnóstico de Signo de Yaw")
    parser.add_argument("--burst", action="store_true", help="Comandar ráfaga activa de giro")
    parser.add_argument("--monitor", action="store_true", help="Monitorear pasivamente")
    parser.add_argument("--w", type=float, default=0.70, help="Velocidad angular de prueba (default: 0.70)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duración de prueba en segundos (default: 3.0s)")

    args = parser.parse_args()
    if not args.burst and not args.monitor:
        args.burst = True  # Por defecto ráfaga activa
    run_diagnostic(args)


if __name__ == "__main__":
    main()
