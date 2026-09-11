#!/usr/bin/env python3
"""
heading_diagnostic_monitor.py

Herramienta de diagnóstico e instrumentación no invasiva para detectar el conflicto
de doble publicador en el tópico 'earth_rover/heading' en tiempo real (o grabado).

Registra por cada mensaje:
  - Timestamp (ROS y reloj de pared)
  - Valor de heading (° brújula)
  - Delta temporal (dt) respecto al mensaje anterior (ms)
  - Delta angular (d_theta) respecto al mensaje anterior (grados con signo más corto)
  - Firma de conflicto / Zig-Zag: Detección de alternancia rápida entre dos secuencias
    discrepantes (ej: EKF suave @ 10Hz vs Brújula cruda ruidosa @ ~0.5-1Hz).
"""

import argparse
import csv
import math
import sys
import time
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32


def angle_diff_deg(a: float, b: float) -> float:
    """Calcula la diferencia angular con signo más corta (a - b) en [-180, 180]."""
    return (a - b + 540.0) % 360.0 - 180.0


class HeadingDiagnosticMonitor(Node):
    def __init__(self, output_csv: str = "", alert_threshold_deg: float = 6.0):
        super().__init__("heading_diagnostic_monitor")
        self.output_csv = output_csv
        self.alert_threshold_deg = alert_threshold_deg

        # QoS idéntico al que usan el controlador y los bridges (BEST_EFFORT)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.sub = self.create_subscription(
            Float32,
            "earth_rover/heading",
            self._on_heading,
            sensor_qos,
        )

        # Historial para análisis de patrones de alternancia
        self.history = deque(maxlen=20)
        self.prev_msg_wall_time = None
        self.prev_heading = None
        self.prev_delta = None

        # Estadísticas
        self.total_msgs = 0
        self.total_jumps = 0
        self.total_zigzags = 0
        self.delta_list = []
        self.dt_list = []

        # CSV Writer
        self.csv_file = None
        self.csv_writer = None
        if self.output_csv:
            self.csv_file = open(self.output_csv, mode="w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "msg_idx",
                "wall_time_iso",
                "ros_time_sec",
                "heading_deg",
                "dt_ms",
                "delta_deg",
                "is_jump",
                "is_zigzag_signature",
            ])

        self.get_logger().info(
            f"=== Heading Diagnostic Monitor Activo | Umbral Alerta: {self.alert_threshold_deg}° ==="
        )
        self.get_logger().info(
            "Monitoreando 'earth_rover/heading' para detectar firmas de doble publicador (intercalación EKF/Raw)..."
        )

    def _on_heading(self, msg: Float32):
        now_wall = time.time()
        now_ros = self.get_clock().now()
        now_ros_sec = now_ros.nanoseconds / 1e9

        val = float(msg.data) % 360.0
        self.total_msgs += 1

        dt_ms = 0.0
        delta = 0.0
        is_jump = False
        is_zigzag = False

        if self.prev_msg_wall_time is not None:
            dt_ms = (now_wall - self.prev_msg_wall_time) * 1000.0
            delta = angle_diff_deg(val, self.prev_heading)
            self.dt_list.append(dt_ms)
            self.delta_list.append(abs(delta))

            if abs(delta) >= self.alert_threshold_deg:
                is_jump = True
                self.total_jumps += 1

            # Detección de Firma de Doble Publicador: Zig-Zag / Rebote Inmediato
            # Ocurre cuando un mensaje pega un salto abrupto en una dirección (+D)
            # y el siguiente mensaje inmediatamente vuelve en dirección opuesta (-D),
            # porque se intercalan dos fuentes con estimaciones distintas.
            if self.prev_delta is not None and len(self.history) >= 2:
                # Si delta actual y delta anterior tienen signos opuestos y magnitudes similares
                sum_deltas = abs(delta + self.prev_delta)
                sum_abs = abs(delta) + abs(self.prev_delta)
                if sum_abs > (2.0 * self.alert_threshold_deg) and sum_deltas < 0.5 * sum_abs:
                    is_zigzag = True
                    self.total_zigzags += 1

        # Logging en pantalla con formato claro
        flag = ""
        if is_zigzag:
            flag = " [FIRMA: DOBLE PUBLICADOR (ZIG-ZAG DETECTADO)]"
        elif is_jump:
            flag = f" [ALERTA: SALTO BRUSCO > {self.alert_threshold_deg}°]"

        log_str = (
            f"#{self.total_msgs:04d} | t={now_ros_sec:10.3f}s | "
            f"heading={val:6.2f}° | dt={dt_ms:6.1f}ms | delta={delta:+6.2f}°{flag}"
        )

        if is_zigzag:
            self.get_logger().error(log_str)
        elif is_jump:
            self.get_logger().warn(log_str)
        else:
            self.get_logger().info(log_str)

        # Guardar en CSV si está habilitado
        if self.csv_writer:
            wall_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now_wall)) + f".{int((now_wall%1)*1000):03d}"
            self.csv_writer.writerow([
                self.total_msgs,
                wall_iso,
                f"{now_ros_sec:.4f}",
                f"{val:.3f}",
                f"{dt_ms:.2f}",
                f"{delta:.3f}",
                int(is_jump),
                int(is_zigzag),
            ])
            self.csv_file.flush()

        self.history.append((now_wall, val, delta))
        self.prev_msg_wall_time = now_wall
        self.prev_heading = val
        self.prev_delta = delta

    def print_summary(self):
        print("\n" + "=" * 70)
        print(" RESUMEN DE DIAGNÓSTICO DE 'earth_rover/heading'")
        print("=" * 70)
        print(f"Total de mensajes analizados: {self.total_msgs}")
        print(f"Saltos bruscos (> {self.alert_threshold_deg}°): {self.total_jumps}")
        print(f"Patrones de Zig-Zag (Firma doble publicador): {self.total_zigzags}")
        if self.dt_list:
            avg_dt = sum(self.dt_list) / len(self.dt_list)
            print(f"Intervalo medio entre mensajes (dt): {avg_dt:.1f} ms (~{1000.0/avg_dt:.1f} Hz)")
        if self.delta_list:
            avg_delta = sum(self.delta_list) / len(self.delta_list)
            max_delta = max(self.delta_list)
            print(f"Delta angular medio entre mensajes: {avg_delta:.2f}° (Máximo: {max_delta:.2f}°)")

        if self.total_zigzags > 0:
            print("\n>> CONCLUSIÓN DEL DIAGNÓSTICO:")
            print("   [!] SE DETECTÓ ACTIVIDAD DE DOBLE PUBLICADOR EN 'earth_rover/heading'.")
            print("   Se observa intercalación de valores que genera saltos y rebotes inmediatos.")
        elif self.total_jumps > 0:
            print("\n>> CONCLUSIÓN DEL DIAGNÓSTICO:")
            print("   [?] Se detectaron saltos de valor, pero sin patrón claro de alternancia rápida.")
        else:
            print("\n>> CONCLUSIÓN DEL DIAGNÓSTICO:")
            print("   [OK] Flujo de heading suave y continuo. No se detecta conflicto.")
        print("=" * 70 + "\n")
        if self.csv_file:
            self.csv_file.close()


def main(args=None):
    parser = argparse.ArgumentParser(description="Monitor de diagnóstico de conflicto en earth_rover/heading")
    parser.add_argument("--csv", type=str, default="", help="Ruta de archivo CSV de salida")
    parser.add_argument("--threshold", type=float, default=6.0, help="Umbral de salto brusco en grados (default: 6.0°)")
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = HeadingDiagnosticMonitor(output_csv=parsed_args.csv, alert_threshold_deg=parsed_args.threshold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
