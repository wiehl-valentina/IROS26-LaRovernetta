#!/usr/bin/env python3
"""
run_heading_conflict_experiment.py

Ejecuta el experimento comparativo completo de la Parte B:
  1. Escenario DUAL: EKF (10Hz) + Brújula Cruda (~0.67Hz, ±8° ruido + glitches).
  2. Escenario CONTROL (SINGLE): Solo EKF (10Hz, filtrado y suave).
  3. Compara estabilidad de heading_error, jitter de modo ALIGN/DRIVE, y logs de advertencias.
"""

import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict


def run_experiment_scenario(mode: str, duration_s: float = 30.0) -> dict:
    print(f"\n" + "=" * 70)
    print(f" INICIANDO ESCENARIO: {mode.upper()} ({duration_s} segundos)")
    print("=" * 70)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    this_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(this_dir)

    # Lanzar gps_waypoint_controller
    controller_cmd = [
        sys.executable,
        os.path.join(pkg_dir, "gps_waypoint_controller.py"),
        "--ros-args",
        "-p", "control_loop_hz:=5.0",
        "-p", "publish_control_debug:=true",
    ]

    # Lanzar test_dual_heading_stimulus
    stimulus_cmd = [
        sys.executable,
        os.path.join(this_dir, "test_dual_heading_stimulus.py"),
        "--mode", mode,
        "--duration", str(duration_s),
    ]

    # Lanzar monitor de diagnóstico
    csv_path = f"/tmp/heading_diag_{mode}.csv"
    monitor_cmd = [
        sys.executable,
        os.path.join(this_dir, "heading_diagnostic_monitor.py"),
        "--csv", csv_path,
        "--threshold", "6.0",
    ]

    controller_proc = subprocess.Popen(
        controller_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    monitor_proc = subprocess.Popen(
        monitor_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    time.sleep(1.5)  # Esperar que los nodos inicialicen suscripciones

    stimulus_proc = subprocess.Popen(
        stimulus_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    stim_out, _ = stimulus_proc.communicate()

    time.sleep(0.5)

    # Terminar controlador y monitor
    controller_proc.send_signal(signal.SIGINT)
    monitor_proc.send_signal(signal.SIGINT)

    try:
        ctrl_out, _ = controller_proc.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        controller_proc.kill()
        ctrl_out, _ = controller_proc.communicate()

    try:
        mon_out, _ = monitor_proc.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        monitor_proc.kill()
        mon_out, _ = monitor_proc.communicate()

    # Parsear logs del controlador
    ctrl_lines = ctrl_out.splitlines() if ctrl_out else []
    jump_warnings = [l for l in ctrl_lines if "Salto magnético gigante rechazado" in l]

    return {
        "mode": mode,
        "stimulus_output": stim_out,
        "controller_output": ctrl_out,
        "monitor_output": mon_out,
        "jump_warnings": jump_warnings,
        "csv_path": csv_path,
    }


def analyze_csv(csv_path):
    import csv
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    
    dts = [float(r['dt_ms']) for r in rows[1:]]
    deltas = [abs(float(r['delta_deg'])) for r in rows[1:]]
    collisions = sum(1 for dt in dts if dt < 10.0)
    jumps = sum(int(r['is_jump']) for r in rows)
    zigzags = sum(int(r['is_zigzag_signature']) for r in rows)
    
    mean_delta = sum(deltas)/len(deltas) if deltas else 0
    std_delta = math.sqrt(sum((x - mean_delta)**2 for x in deltas)/len(deltas)) if deltas else 0
    
    return {
        'total': len(rows),
        'mean_dt': sum(dts)/len(dts) if dts else 0,
        'collisions': collisions,
        'mean_delta': mean_delta,
        'std_delta': std_delta,
        'max_delta': max(deltas) if deltas else 0,
        'jumps': jumps,
        'zigzags': zigzags,
    }


def main():
    duration = 30.0
    print("\n=======================================================")
    print(" EJECUCIÓN DE ARNES DE PRUEBAS: VALIDACIÓN POST-FIX")
    print("=======================================================")

    # 1. Escenario PRE-FIX (Conflicto original: ambos a earth_rover/heading)
    prefix_res = run_experiment_scenario("prefix_dual", duration_s=duration)

    time.sleep(2.0)

    # 2. Escenario SINGLE (Control EKF: solo EKF a earth_rover/heading)
    single_res = run_experiment_scenario("ekf_only", duration_s=duration)

    time.sleep(2.0)

    # 3. Escenario POST-FIX (Ambos activos: EKF a heading, Raw a heading_raw)
    postfix_res = run_experiment_scenario("postfix_dual", duration_s=duration)

    # Comparación
    print("\n\n" + "#" * 88)
    print(" RESULTADOS COMPARATIVOS: VALIDACIÓN POST-FIX vs PRE-FIX vs CONTROL")
    print("#" * 88)

    pre_stats = analyze_csv(prefix_res["csv_path"])
    s_stats = analyze_csv(single_res["csv_path"])
    post_stats = analyze_csv(postfix_res["csv_path"])

    print("\n" + "=" * 92)
    print(" TABLA COMPARATIVA: PRE-FIX (BUG) vs CONTROL (SINGLE) vs POST-FIX (RESUELTO)")
    print("=" * 92)
    print(f"{'Métrica':<40} | {'PRE-FIX (Bug)':<14} | {'CONTROL (Single)':<16} | {'POST-FIX (Fix)':<14}")
    print("-" * 92)
    print(f"{'Mensajes totales en earth_rover/heading':<40} | {pre_stats.get('total', 0):<14} | {s_stats.get('total', 0):<16} | {post_stats.get('total', 0):<14}")
    print(f"{'Intervalo medio entre mensajes (dt)':<40} | {pre_stats.get('mean_dt', 0):<11.2f} ms | {s_stats.get('mean_dt', 0):<13.2f} ms | {post_stats.get('mean_dt', 0):<11.2f} ms")
    print(f"{'Colisiones temporales (dt < 10ms)':<40} | {pre_stats.get('collisions', 0):<14} | {s_stats.get('collisions', 0):<16} | {post_stats.get('collisions', 0):<14}")
    print(f"{'Delta angular medio entre msgs sucesivos':<40} | {pre_stats.get('mean_delta', 0):<13.2f}° | {s_stats.get('mean_delta', 0):<15.2f}° | {post_stats.get('mean_delta', 0):<13.2f}°")
    print(f"{'Desviación estándar de deltas':<40} | {pre_stats.get('std_delta', 0):<13.2f}° | {s_stats.get('std_delta', 0):<15.2f}° | {post_stats.get('std_delta', 0):<13.2f}°")
    print(f"{'Delta angular máximo puntual':<40} | {pre_stats.get('max_delta', 0):<13.2f}° | {s_stats.get('max_delta', 0):<15.2f}° | {post_stats.get('max_delta', 0):<13.2f}°")
    print(f"{'Saltos bruscos (> 6.0°)':<40} | {pre_stats.get('jumps', 0):<14} | {s_stats.get('jumps', 0):<16} | {post_stats.get('jumps', 0):<14}")
    print(f"{'Firmas Zig-Zag (Doble publicador)':<40} | {pre_stats.get('zigzags', 0):<14} | {s_stats.get('zigzags', 0):<16} | {post_stats.get('zigzags', 0):<14}")
    print("=" * 92)

    print("\n--- SALIDA ESTIMULO PRE-FIX (DUAL CON CONFLICTO) ---")
    print(prefix_res["stimulus_output"])

    print("--- SALIDA ESTIMULO POST-FIX (DUAL AISLADO: RAW -> HEADING_RAW) ---")
    print(postfix_res["stimulus_output"])


if __name__ == "__main__":
    main()

