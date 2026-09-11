#!/usr/bin/env python3
"""Herramienta de Diagnóstico de Campo del Heading y Fuentes Inmunes al Magnetismo.

Soporta:
1. Re-análisis con debiasado exacto del giróscopo sobre CSVs previos:
   python -m genie_rover.diag_heading --reanalyze diagnostico_heading/diag_test_a_quieto_60s_20260910_121617.csv

2. Captura fija por tiempo (ej. Test A):
   python -m genie_rover.diag_heading --duration 60 --label test_a_quieto

3. Test B': Giros conocidos con giróscopo medido ÚNICAMENTE durante la rotación activa:
   python -m genie_rover.diag_heading --turns-isolated --label test_b_prime

4. Test D: Avance en recta para evaluar gps_track (adelante o reversa):
   python -m genie_rover.diag_heading --straight-track --label test_d_recta_adelante
   python -m genie_rover.diag_heading --straight-track --reverse --label test_d_recta_reversa
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from genie_rover.sdk_client import RoverClient, Telemetry

EARTH_R = 6371000.0


def latlon_to_local_ne(lat_ref: float, lon_ref: float,
                       lat: float, lon: float) -> tuple[float, float]:
    """Desplazamiento (norte_m, este_m) desde el punto de referencia."""
    dlat = math.radians(lat - lat_ref)
    dlon = math.radians(lon - lon_ref)
    north = dlat * EARTH_R
    east = dlon * EARTH_R * math.cos(math.radians(lat_ref))
    return north, east


def wrap_deg(deg: float) -> float:
    """Envuelve ángulo a [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def wrap_360(deg: float) -> float:
    """Envuelve ángulo a [0, 360)."""
    return deg % 360.0


class HeadingDiagnosticLogger:
    def __init__(self, base_url: str = "http://localhost:8000", output_dir: str = "diagnostico_heading",
                 bias_z: float = 1.027725):
        self.client = RoverClient(base_url=base_url, timeout=3.0, stop_on_exit=False)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bias_z = float(bias_z)

        # Estado de integración de giróscopo
        self.last_gyro_t: Optional[float] = None
        self.last_poll_local_t: Optional[float] = None
        self.integrated_yaw_raw: float = 0.0
        self.integrated_yaw_debiased: float = 0.0
        self.initial_orientation: Optional[float] = None
        self.initial_ekf: Optional[float] = None

        # Estado de integración cinemática por ruedas (RPM)
        self.track_m: float = 0.16
        self.wheel_radius_m: float = 0.0527  # Radio de rueda nominal Mini+
        self.last_rpm_t: Optional[float] = None
        self.integrated_yaw_rpm: float = 0.0
        self.integrated_yaw_rpm_raw: float = 0.0

        # Estado de GPS
        self.initial_lat: Optional[float] = None
        self.initial_lon: Optional[float] = None

        # Almacenamiento en memoria
        self.records: List[dict[str, Any]] = []

    def poll_sample(self, marker: str = "") -> Optional[dict[str, Any]]:
        """Consulta el SDK y el socket UDP y genera un registro consolidado."""
        try:
            telem: Telemetry = self.client.telemetry()
        except Exception as exc:
            print(f"[WARN] Error al consultar telemetría: {exc}", file=sys.stderr)
            return None

        raw = telem.raw
        t_poll = telem.timestamp
        t_local = time.time()

        if self.initial_orientation is None and telem.orientation is not None:
            self.initial_orientation = float(telem.orientation)
            self.integrated_yaw_raw = self.initial_orientation
            self.integrated_yaw_debiased = self.initial_orientation
            self.integrated_yaw_rpm = self.initial_orientation

        if self.initial_ekf is None and telem.ekf_heading is not None:
            self.initial_ekf = float(telem.ekf_heading)

        # 1. Procesar gyros
        gyros = raw.get("gyros") or []
        latest_gx = 0.0
        latest_gy = 0.0
        latest_gz = 0.0
        gyro_samples_in_batch = len(gyros)

        if gyros:
            for g in gyros:
                if len(g) >= 4:
                    gx, gy, gz, t_sub = float(g[0]), float(g[1]), float(g[2]), float(g[3])
                    latest_gx, latest_gy, latest_gz = gx, gy, gz
                    if self.last_gyro_t is not None:
                        dt = t_sub - self.last_gyro_t
                        if 0.0 < dt < 0.5:
                            # Girar a la derecha (CW) -> gz < 0 -> heading crece (+dt * |gz|)
                            # Girar a la izquierda (CCW) -> gz > 0 -> heading decrece (-dt * gz)
                            self.integrated_yaw_raw = wrap_360(self.integrated_yaw_raw - gz * dt)
                            self.integrated_yaw_debiased = wrap_360(self.integrated_yaw_debiased - (gz - self.bias_z) * dt)
                    self.last_gyro_t = t_sub
        else:
            # Fallback a dt local si gyros no viniera con timestamps
            if self.last_poll_local_t is not None:
                dt_l = t_local - self.last_poll_local_t
                if 0.0 < dt_l < 0.5:
                    self.integrated_yaw_raw = wrap_360(self.integrated_yaw_raw - latest_gz * dt_l)
                    self.integrated_yaw_debiased = wrap_360(self.integrated_yaw_debiased - (latest_gz - self.bias_z) * dt_l)
        self.last_poll_local_t = t_local

        # 2. Mags crudos
        mags = raw.get("mags") or []
        latest_mx = 0.0
        latest_my = 0.0
        latest_mz = 0.0
        if mags and len(mags[-1]) >= 3:
            latest_mx = float(mags[-1][0])
            latest_my = float(mags[-1][1])
            latest_mz = float(mags[-1][2])

        # 3. RPMs por rueda y odometría diferencial
        rpms = raw.get("rpms") or []
        rpm0, rpm1, rpm2, rpm3 = 0.0, 0.0, 0.0, 0.0
        latest_rpm_diff = 0.0
        latest_omega_rpm_dps = 0.0

        if rpms:
            for r_row in rpms:
                if len(r_row) >= 4:
                    r0, r1, r2, r3 = float(r_row[0]), float(r_row[1]), float(r_row[2]), float(r_row[3])
                    rpm0, rpm1, rpm2, rpm3 = r0, r1, r2, r3
                    rpm_l = (r0 + r2) / 2.0
                    rpm_r = (r1 + r3) / 2.0
                    latest_rpm_diff = rpm_r - rpm_l
                    # omega = (v_r - v_l) / track
                    # v = rpm * (2*pi/60) * r
                    # omega_dps = (6.0 * r / track) * (rpm_r - rpm_l)
                    omega_dps = (6.0 * self.wheel_radius_m / self.track_m) * latest_rpm_diff
                    latest_omega_rpm_dps = omega_dps

                    if len(r_row) >= 5:
                        t_rpm = float(r_row[4])
                        if self.last_rpm_t is not None:
                            dt_r = t_rpm - self.last_rpm_t
                            if 0.0 < dt_r < 0.5:
                                # Giro a la izquierda (CCW, rpm_r > rpm_l) -> omega > 0 -> heading decrece
                                self.integrated_yaw_rpm = wrap_360(self.integrated_yaw_rpm - omega_dps * dt_r)
                                self.integrated_yaw_rpm_raw += (latest_rpm_diff / self.track_m) * dt_r
                        self.last_rpm_t = t_rpm
        else:
            if self.last_poll_local_t is not None:
                dt_l = t_local - self.last_poll_local_t
                if 0.0 < dt_l < 0.5:
                    self.integrated_yaw_rpm = wrap_360(self.integrated_yaw_rpm - latest_omega_rpm_dps * dt_l)
                    self.integrated_yaw_rpm_raw += (latest_rpm_diff / self.track_m) * dt_l

        # 4. GPS y Track
        disp_from_start_m = 0.0
        gps_track_deg: Optional[float] = None
        if abs(telem.latitude) <= 90 and abs(telem.longitude) <= 180 and telem.latitude != 0:
            if self.initial_lat is None:
                self.initial_lat = telem.latitude
                self.initial_lon = telem.longitude
            else:
                n, e = latlon_to_local_ne(self.initial_lat, self.initial_lon, telem.latitude, telem.longitude)
                disp_from_start_m = math.hypot(n, e)
                if disp_from_start_m >= 0.5:
                    gps_track_deg = math.degrees(math.atan2(e, n)) % 360.0

        record = {
            "timestamp_local": t_local,
            "timestamp_sdk": t_poll,
            "marker": marker,
            "orientation": telem.orientation,
            "ekf_heading": telem.ekf_heading,
            "integrated_yaw_raw": round(self.integrated_yaw_raw, 2),
            "integrated_yaw_debiased": round(self.integrated_yaw_debiased, 2),
            "integrated_yaw": round(self.integrated_yaw_debiased, 2),  # alias compat
            "integrated_yaw_rpm": round(self.integrated_yaw_rpm, 2),
            "integrated_yaw_rpm_raw": round(self.integrated_yaw_rpm_raw, 3),
            "gyro_x": latest_gx,
            "gyro_y": latest_gy,
            "gyro_z": latest_gz,
            "omega_gyro_deb_dps": round(latest_gz - self.bias_z, 3),
            "omega_rpm_dps": round(latest_omega_rpm_dps, 3),
            "rpm_diff": round(latest_rpm_diff, 2),
            "mag_x": latest_mx,
            "mag_y": latest_my,
            "mag_z": latest_mz,
            "mag_norm": round(math.sqrt(latest_mx**2 + latest_my**2 + latest_mz**2), 1),
            "rpm0": rpm0,
            "rpm1": rpm1,
            "rpm2": rpm2,
            "rpm3": rpm3,
            "speed": telem.speed,
            "latitude": telem.latitude,
            "longitude": telem.longitude,
            "disp_from_start_m": round(disp_from_start_m, 2),
            "gps_track_deg": round(gps_track_deg, 1) if gps_track_deg is not None else None,
            "battery": telem.battery,
            "gps_signal": telem.gps_signal,
            "gyro_batch_len": gyro_samples_in_batch,
        }
        self.records.append(record)
        return record

    def save(self, label: str) -> tuple[Path, Path]:
        """Guarda los datos en CSV y JSONL."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"diag_{label}_{ts_str}.csv"
        jsonl_path = self.output_dir / f"diag_{label}_{ts_str}.jsonl"

        if not self.records:
            print("[WARN] No hay registros para guardar.")
            return csv_path, jsonl_path

        fieldnames = list(self.records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")

        return csv_path, jsonl_path

    def print_summary_statistics(self, label: str):
        """Calcula y muestra estadísticas resumidas para análisis inmediato."""
        if not self.records:
            print("Sin datos suficientes para estadísticas.")
            return

        n = len(self.records)
        t_start = self.records[0]["timestamp_local"]
        t_end = self.records[-1]["timestamp_local"]
        duration = max(0.001, t_end - t_start)

        gz_vals = [r["gyro_z"] for r in self.records]
        gx_vals = [r["gyro_x"] for r in self.records]
        gy_vals = [r["gyro_y"] for r in self.records]
        mx_vals = [r["mag_x"] for r in self.records]
        my_vals = [r["mag_y"] for r in self.records]
        mz_vals = [r["mag_z"] for r in self.records]
        mnorm_vals = [r["mag_norm"] for r in self.records]
        ori_vals = [r["orientation"] for r in self.records if r["orientation"] is not None]
        ekf_vals = [r["ekf_heading"] for r in self.records if r["ekf_heading"] is not None]
        raw_vals = [r["integrated_yaw_raw"] for r in self.records]
        deb_vals = [r["integrated_yaw_debiased"] for r in self.records]

        def stats(vals):
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = math.sqrt(var)
            return mean, std, min(vals), max(vals)

        gz_mean, gz_std, gz_min, gz_max = stats(gz_vals)
        gx_mean, gx_std, gx_min, gx_max = stats(gx_vals)
        gy_mean, gy_std, gy_min, gy_max = stats(gy_vals)
        mx_mean, mx_std, mx_min, mx_max = stats(mx_vals)
        my_mean, my_std, my_min, my_max = stats(my_vals)
        mz_mean, mz_std, mz_min, mz_max = stats(mz_vals)
        mnorm_mean, mnorm_std, mnorm_min, mnorm_max = stats(mnorm_vals)

        print("\n" + "=" * 75)
        print(f" RESUMEN ESTADÍSTICO — {label.upper()} ({n} muestras, {duration:.1f} s)")
        print("=" * 75)
        print(f"• Gyro X (Roll/Pitch): mean={gx_mean:+.4f} dps | std={gx_std:.4f} | [{gx_min:+.2f}, {gx_max:+.2f}]")
        print(f"• Gyro Y (Pitch/Roll): mean={gy_mean:+.4f} dps | std={gy_std:.4f} | [{gy_min:+.2f}, {gy_max:+.2f}]")
        print(f"• Gyro Z (Yaw rate):   mean={gz_mean:+.4f} dps | std={gz_std:.4f} | [{gz_min:+.2f}, {gz_max:+.2f}]")
        print("-" * 75)
        print(f"• Mag X: mean={mx_mean:+.1f} | std={mx_std:.2f} | [{mx_min:+.0f}, {mx_max:+.0f}]")
        print(f"• Mag Y: mean={my_mean:+.1f} | std={my_std:.2f} | [{my_min:+.0f}, {my_max:+.0f}]")
        print(f"• Mag Z: mean={mz_mean:+.1f} | std={mz_std:.2f} | [{mz_min:+.0f}, {mz_max:+.0f}]")
        print(f"• Mag Norma: mean={mnorm_mean:+.1f} | std={mnorm_std:.2f} | [{mnorm_min:+.0f}, {mnorm_max:+.0f}]")
        print("-" * 75)

        if ori_vals:
            ori_ini, ori_fin = ori_vals[0], ori_vals[-1]
            ori_diff = wrap_deg(ori_fin - ori_ini)
            print(f"• Compás crudo (SDK):  inicio={ori_ini:.1f}° | final={ori_fin:.1f}° | deriva={ori_diff:+.2f}°")
        else:
            print("• Compás crudo (SDK):  Sin lecturas")

        if ekf_vals:
            ekf_ini, ekf_fin = ekf_vals[0], ekf_vals[-1]
            ekf_diff = wrap_deg(ekf_fin - ekf_ini)
            print(f"• Heading EKF (UDP):   inicio={ekf_ini:.1f}° | final={ekf_fin:.1f}° | deriva={ekf_diff:+.2f}° ({len(ekf_vals)} lecturas)")
        else:
            print("• Heading EKF (UDP):   No se recibieron paquetes EKF en puerto 9876")

        raw_diff = wrap_deg(raw_vals[-1] - raw_vals[0])
        deb_diff = wrap_deg(deb_vals[-1] - deb_vals[0])
        print(f"• Giróscopo CRUDO:     inicio={raw_vals[0]:.1f}° | final={raw_vals[-1]:.1f}° | deriva={raw_diff:+.2f}°")
        print(f"• Giróscopo DEBIASADO: inicio={deb_vals[0]:.1f}° | final={deb_vals[-1]:.1f}° | deriva={deb_diff:+.2f}° (bias_z={self.bias_z:.4f}°/s)")
        rpm_vals = [r.get("integrated_yaw_rpm", 0.0) for r in self.records if "integrated_yaw_rpm" in r]
        if rpm_vals:
            rpm_diff_val = wrap_deg(rpm_vals[-1] - rpm_vals[0])
            print(f"• Yaw por RPM (Diff):  inicio={rpm_vals[0]:.1f}° | final={rpm_vals[-1]:.1f}° | deriva={rpm_diff_val:+.2f}° (track={self.track_m}m)")
        print("=" * 75)


def run_reanalyze_csv(csv_path: str, bias_z: float = 1.027725):
    """Paso 1: Re-analiza un CSV previo restando el bias exacto medido."""
    print(f"\n[RE-ANÁLISIS] Evaluando archivo: {csv_path}")
    print(f"Bias Z configurado: {bias_z:.6f} deg/s")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV vacío.")
        return

    n = len(rows)
    # Detectar columna de tiempo
    time_col = "timestamp_local" if "timestamp_local" in rows[0] else "timestamp"

    t0 = float(rows[0][time_col])
    t_end = float(rows[-1][time_col])
    duration = t_end - t0

    gz_vals = [float(r["gyro_z"]) for r in rows]
    mean_gz = sum(gz_vals) / len(gz_vals)

    ini_ori = float(rows[0]["orientation"])
    fin_ori = float(rows[-1]["orientation"])

    # Integración paso a paso
    yaw_raw = ini_ori
    yaw_deb = ini_ori

    for i in range(1, n):
        dt = float(rows[i][time_col]) - float(rows[i - 1][time_col])
        gz = float(rows[i]["gyro_z"])
        yaw_raw = wrap_360(yaw_raw - gz * dt)
        yaw_deb = wrap_360(yaw_deb - (gz - bias_z) * dt)

    raw_drift = wrap_deg(yaw_raw - ini_ori)
    deb_drift = wrap_deg(yaw_deb - ini_ori)

    print("=" * 75)
    print(f" RESULTADO RE-ANÁLISIS DE TEST A ({n} muestras, {duration:.1f} s)")
    print("=" * 75)
    print(f"• Bias Z medido real en este CSV: {mean_gz:+.6f} deg/s")
    print(f"• Bias Z aplicado en debiasado:   {bias_z:+.6f} deg/s")
    print("-" * 75)
    print(f"• Compás crudo (SDK):        {ini_ori:.1f}° -> {fin_ori:.1f}° | Deriva: {wrap_deg(fin_ori - ini_ori):+.2f}°")
    print(f"• Yaw integrado CRUDO:       {ini_ori:.1f}° -> {yaw_raw:.2f}° | Deriva: {raw_drift:+.2f}°")
    print(f"• Yaw integrado DEBIASADO:   {ini_ori:.1f}° -> {yaw_deb:.2f}° | Deriva: {deb_drift:+.2f}°")
    print("=" * 75)
    print(f"\n>>> CRITERIO PASO 1: Deriva del debiasado en 60 s = {deb_drift:+.2f}° (prácticamente nula vs {raw_drift:+.1f}° del crudo). <<<\n")


def run_turns_isolated(logger: HeadingDiagnosticLogger, label: str, rate_hz: float = 10.0):
    """Paso 2 (Test B'): Giros conocidos midiendo el giróscopo debiasado SOLO durante la rotación activa."""
    turns_def = [
        ("Giro 1", 0, 90, +90),
        ("Giro 2", 90, 180, +90),
        ("Giro 3", 180, 270, +90),
        ("Giro 4", 270, 360, +90),
    ]
    results = []
    period = 1.0 / rate_hz

    print("\n" + "=" * 85)
    print(f" PROTOCOLO TEST B': GIROS CONOCIDOS CON ROTACIÓN AISLADA — {label.upper()}")
    print("=" * 85)
    print("Este protocolo elimina la contaminación del reposo:")
    print("  1. Alinea el rover en la marca inicial.")
    print("  2. Presiona ENTER justo antes de empezar a girar el robot.")
    print("  3. Gira el rover físicamente hasta la siguiente marca de 90°.")
    print("  4. En cuanto termines de rotar, presiona ENTER inmediatamente.")
    print("=" * 85 + "\n")

    input("Coloca el rover en la marca inicial (0°) y presiona ENTER para calibrar referencia...")
    # Tomar referencia inicial en reposo
    ref_samples = []
    t_ref0 = time.time()
    while time.time() - t_ref0 < 1.0:
        rec = logger.poll_sample(marker="calib_inicial")
        if rec:
            ref_samples.append(rec)
        time.sleep(period)

    for name, start_deg, target_deg, nominal_turn in turns_def:
        print(f"\n---------------------------------------------------------------------------")
        print(f" >>> {name.upper()}: Rotación física nominal de +90° ({start_deg}° -> {target_deg}°)")
        print(f"---------------------------------------------------------------------------")
        input(f"Asegúrate de estar en {start_deg}°. Presiona ENTER para INICIAR la rotación...")

        # Capturar línea base justo al comenzar a rotar
        pre_rec = logger.poll_sample(marker=f"{name}_start")
        start_t = time.time()
        start_deb_yaw = logger.integrated_yaw_debiased
        start_ori = pre_rec["orientation"] if pre_rec else None
        start_ekf = pre_rec["ekf_heading"] if pre_rec else None

        print(f"  [ROTANDO...] Gira el rover hacia {target_deg}°. Presiona ENTER al completar el giro.")

        turn_samples = []
        # Loop interactivo mientras gira
        import select
        while True:
            # Check si el usuario presionó ENTER
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break

            rec = logger.poll_sample(marker=f"{name}_rotating")
            if rec:
                turn_samples.append(rec)
                d_curr = wrap_deg(logger.integrated_yaw_debiased - start_deb_yaw)
                print(f"\r  Giro en progreso -> ΔYawDebiasado: {d_curr:+6.1f}° | GyroZ: {rec['gyro_z']:+6.2f} dps", end="", flush=True)
            time.sleep(period)

        post_rec = logger.poll_sample(marker=f"{name}_finish")
        end_deb_yaw = logger.integrated_yaw_debiased
        end_ori = post_rec["orientation"] if post_rec else None
        end_ekf = post_rec["ekf_heading"] if post_rec else None

        delta_deb_yaw = wrap_deg(end_deb_yaw - start_deb_yaw)
        delta_ori = wrap_deg(end_ori - start_ori) if (end_ori is not None and start_ori is not None) else None
        delta_ekf = wrap_deg(end_ekf - start_ekf) if (end_ekf is not None and start_ekf is not None) else None

        # RPMs durante la rotación activa (Paso 4 opcional)
        avg_rpm_l = 0.0
        avg_rpm_r = 0.0
        if turn_samples:
            avg_rpm_l = sum((s["rpm0"] + s["rpm2"]) / 2.0 for s in turn_samples) / len(turn_samples)
            avg_rpm_r = sum((s["rpm1"] + s["rpm3"]) / 2.0 for s in turn_samples) / len(turn_samples)

        results.append({
            "name": name,
            "target": f"{start_deg}° -> {target_deg}°",
            "nominal": nominal_turn,
            "delta_deb_yaw": delta_deb_yaw,
            "delta_ori": delta_ori,
            "delta_ekf": delta_ekf,
            "rpm_l": avg_rpm_l,
            "rpm_r": avg_rpm_r,
            "duration": time.time() - start_t,
        })

        print(f"\n  [COMPLETADO] ΔYawDeb: {delta_deb_yaw:+5.1f}° | ΔCompás: {f'{delta_ori:+5.1f}°' if delta_ori is not None else 'N/A'} | ΔEKF: {f'{delta_ekf:+5.1f}°' if delta_ekf is not None else 'N/A'}")

    csv_p, jsonl_p = logger.save(label)
    logger.print_summary_statistics(label)

    # Imprimir Tabla Formal Test B'
    print("\n" + "=" * 95)
    print(f" TABLA RESULTADOS TEST B': GIROS MEDIDOS SOLO DURANTE ROTACIÓN ({label.upper()})")
    print("=" * 95)
    print(f"{'Giro':<8} | {'Tramo':<13} | {'Físico':<8} | {'ΔYaw Debiasado':<16} | {'ΔCompás SDK':<14} | {'ΔEKF UDP':<12} | {'RPM L/R':<12}")
    print("-" * 95)

    for r in results:
        g_deb = f"{r['delta_deb_yaw']:+6.1f}°"
        g_ori = f"{r['delta_ori']:+6.1f}°" if r['delta_ori'] is not None else "N/A"
        g_ekf = f"{r['delta_ekf']:+6.1f}°" if r['delta_ekf'] is not None else "N/A"
        rpm_s = f"{r['rpm_l']:+.1f} / {r['rpm_r']:+.1f}"
        print(f"{r['name']:<8} | {r['target']:<13} | {r['nominal']:+4d}°   | {g_deb:<16} | {g_ori:<14} | {g_ekf:<12} | {rpm_s:<12}")
    print("=" * 95)
    print(f"\nArchivos guardados en: {csv_p}")


def run_straight_track(logger: HeadingDiagnosticLogger, label: str, is_reverse: bool = False, rate_hz: float = 10.0):
    """Paso 3 (Test D): Evalúa gps_track en avance en línea recta."""
    direction_desc = "REVERSA" if is_reverse else "ADELANTE"
    print("\n" + "=" * 85)
    print(f" PROTOCOLO TEST D: AVANCE EN LÍNEA RECTA ({direction_desc}) — {label.upper()}")
    print("=" * 85)
    print(f"Condición: Rover avanzando en recta {'hacia atrás' if is_reverse else 'hacia adelante'} (mínimo 10-15 m).")
    print("Instrucciones:")
    print("  1. Alinea el rover en el punto de partida apuntando a lo largo del tramo.")
    print("  2. Presiona ENTER para iniciar el registro y comienza a avanzar.")
    print("  3. Al llegar al final del tramo recto, presiona ENTER para detener y analizar.")
    print("=" * 85 + "\n")

    input("Presiona ENTER para iniciar la captura de recta...")
    period = 1.0 / rate_hz
    t_start = time.time()
    last_print = 0.0

    import select
    print("\n>>> AVANZANDO EN LÍNEA RECTA... (Presiona ENTER al llegar al final) <<<\n")

    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break

        rec = logger.poll_sample(marker=f"straight_{direction_desc.lower()}")
        t_now = time.time()
        if rec and (t_now - last_print >= 0.5):
            last_print = t_now
            track_s = f"{rec['gps_track_deg']:5.1f}°" if rec['gps_track_deg'] is not None else "  N/A"
            ekf_s = f"{rec['ekf_heading']:5.1f}°" if rec['ekf_heading'] is not None else "  N/A"
            rpm_l = (rec['rpm0'] + rec['rpm2']) / 2.0
            rpm_r = (rec['rpm1'] + rec['rpm3']) / 2.0
            print(f"[{t_now - t_start:5.1f}s] Dist: {rec['disp_from_start_m']:4.1f}m | "
                  f"gps_track: {track_s} | EKF: {ekf_s} | Compás: {rec['orientation']:5.1f}° | "
                  f"YawDeb: {rec['integrated_yaw_debiased']:5.1f}° | Speed: {rec['speed']:.2f} | RPM: {rpm_l:.0f}/{rpm_r:.0f}")
        time.sleep(period)

    csv_p, jsonl_p = logger.save(label)
    logger.print_summary_statistics(label)

    # Análisis específico de gps_track a diferentes distancias
    print("\n" + "=" * 85)
    print(f" ANÁLISIS DE CONVERGENCIA DE GPS_TRACK ({label.upper()})")
    print("=" * 85)
    thresholds = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
    print(f"{'Distancia Mín':<15} | {'Muestras GPS':<14} | {'gps_track Medio':<18} | {'Desvío Estándar':<16}")
    print("-" * 85)

    for th in thresholds:
        matching = [r for r in logger.records if r["disp_from_start_m"] >= th and r["gps_track_deg"] is not None]
        if matching:
            tracks = [m["gps_track_deg"] for m in matching]
            # promedio circular
            sin_s = sum(math.sin(math.radians(x)) for x in tracks) / len(tracks)
            cos_s = sum(math.cos(math.radians(x)) for x in tracks) / len(tracks)
            mean_deg = math.degrees(math.atan2(sin_s, cos_s)) % 360.0
            std_deg = math.sqrt(sum(wrap_deg(x - mean_deg) ** 2 for x in tracks) / len(tracks))
            print(f">= {th:4.1f} m        | {len(matching):<14} | {mean_deg:6.1f}°            | {std_deg:5.2f}°")
        else:
            print(f">= {th:4.1f} m        | {'0 (no alcanzado)':<14} | {'N/A':<18} | {'N/A':<16}")
    print("=" * 85)
    print(f"\nArchivos guardados en: {csv_p}")


def run_fixed_duration(logger: HeadingDiagnosticLogger, duration_s: float, label: str, rate_hz: float = 10.0):
    """Ejecuta una captura continua durante un tiempo fijo (ej. Test A)."""
    print(f"\n[INICIO] Captura por tiempo fijo: {duration_s:.1f} segundos (Label: '{label}') a ~{rate_hz:.0f} Hz...")
    period = 1.0 / rate_hz
    t_start = time.time()
    next_time = t_start
    last_print = 0.0

    try:
        while True:
            t_now = time.time()
            elapsed = t_now - t_start
            if elapsed >= duration_s:
                break

            record = logger.poll_sample()
            if record and (t_now - last_print >= 0.5):
                last_print = t_now
                ekf_s = f"{record['ekf_heading']:6.1f}°" if record['ekf_heading'] is not None else "  None"
                print(f"[{elapsed:5.1f}s] Compás: {record['orientation']:5.1f}° | EKF: {ekf_s} | "
                      f"GyroZ: {record['gyro_z']:+6.2f} dps | IntYawDeb: {record['integrated_yaw_debiased']:5.1f}° | "
                      f"Speed: {record['speed']:.2f}")

            next_time += period
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()
    except KeyboardInterrupt:
        print("\n[INFO] Captura interrumpida manualmente por usuario.")

    csv_p, jsonl_p = logger.save(label)
    logger.print_summary_statistics(label)
    print(f"\n[ARCHIVOS GUARDADOS]")
    print(f"  CSV:   {csv_p}")
    print(f"  JSONL: {jsonl_p}")


def run_motor_sweep(logger: HeadingDiagnosticLogger, label: str,
                    angulars: Optional[List[float]] = None,
                    step_duration: float = 5.0,
                    pause_duration: float = 3.0,
                    rate_hz: float = 10.0,
                    no_input: bool = False):
    """Paso 1: Confirmar que el rover mantiene giro en el lugar por comando.
    Prueba varios comandos de giro puro en el lugar y registra la velocidad angular resultante."""
    if angulars is None:
        angulars = [0.20, 0.30, 0.45]

    period = 1.0 / rate_hz
    print("\n" + "=" * 95)
    print(f" PROTOCOLO PASO 1: BARRIDO DE VELOCIDAD DE GIRO MOTORIZADO — {label.upper()}")
    print("=" * 95)
    print(f"Valores angulares a probar: {angulars}")
    print(f"Duración por escalón: {step_duration:.1f} s | Pausa entre giros: {pause_duration:.1f} s | Tasa: {rate_hz:.0f} Hz")
    print("SEGURIDAD: Asegurarse de tener 1 metro de espacio despejado alrededor del rover.")
    print("=" * 95 + "\n")

    if not no_input and sys.stdin.isatty():
        input("Presiona ENTER para iniciar la secuencia de giro...")
    else:
        print("Iniciando en 3 segundos...")
        for i in (3, 2, 1):
            print(f"  {i}...")
            time.sleep(1.0)

    sweep_results = []

    try:
        # 1. Breve reposo inicial
        t_init = time.time()
        while time.time() - t_init < 1.5:
            logger.poll_sample(marker="sweep_baseline")
            time.sleep(period)

        for ang in angulars:
            print(f"\n>>> Probando comando angular = {ang:+.2f} durante {step_duration:.1f} s...")
            t_step_start = time.time()
            step_samples = []

            # Loop de giro activo
            while time.time() - t_step_start < step_duration:
                # Mantener activo el comando contra el watchdog del SDK
                logger.client.control(linear=0.0, angular=ang)
                rec = logger.poll_sample(marker=f"sweep_ang_{ang:.2f}_active")
                if rec:
                    step_samples.append(rec)
                    print(f"\r  [GIRANDO {ang:+.2f}] Gyro Z: {rec['gyro_z']:+6.2f} dps | "
                          f"Omega RPM: {rec.get('omega_rpm_dps', 0.0):+6.2f} dps | "
                          f"RPM diff: {rec.get('rpm_diff', 0.0):+5.1f}", end="", flush=True)
                time.sleep(period)

            # Freno inmediato
            logger.client.stop()
            logger.client.stop()
            print(f"\n  [FRENO] Esperando reposo ({pause_duration:.1f} s)...")

            # Descartar primeros 0.6 s de aceleración para evaluar régimen estacionario
            steady_samples = [s for s in step_samples if (s["timestamp_local"] - t_step_start) >= 0.6]
            if not steady_samples:
                steady_samples = step_samples

            gz_vals = [s["gyro_z"] for s in steady_samples]
            gz_deb_vals = [s["omega_gyro_deb_dps"] for s in steady_samples]
            rpm_omega_vals = [s["omega_rpm_dps"] for s in steady_samples]
            rpm_diff_vals = [s["rpm_diff"] for s in steady_samples]

            def calc_mean_std(vals):
                if not vals:
                    return 0.0, 0.0
                m = sum(vals) / len(vals)
                v = sum((x - m) ** 2 for x in vals) / len(vals)
                return m, math.sqrt(v)

            gz_mean, gz_std = calc_mean_std(gz_vals)
            gz_deb_mean, gz_deb_std = calc_mean_std(gz_deb_vals)
            rpm_om_mean, rpm_om_std = calc_mean_std(rpm_omega_vals)
            rpm_diff_mean, _ = calc_mean_std(rpm_diff_vals)

            ratio = gz_mean / ang if abs(ang) > 1e-4 else 0.0

            sweep_results.append({
                "cmd": ang,
                "gz_mean": gz_mean,
                "gz_std": gz_std,
                "gz_deb_mean": gz_deb_mean,
                "gz_deb_std": gz_deb_std,
                "rpm_diff_mean": rpm_diff_mean,
                "omega_rpm_mean": rpm_om_mean,
                "omega_rpm_std": rpm_om_std,
                "ratio": ratio,
                "samples_count": len(steady_samples),
            })

            # Pausa de reposo entre escalones
            t_pause_start = time.time()
            while time.time() - t_pause_start < pause_duration:
                logger.poll_sample(marker=f"sweep_ang_{ang:.2f}_pause")
                time.sleep(period)

    finally:
        logger.client.stop()
        logger.client.stop()

    csv_p, jsonl_p = logger.save(label)
    logger.print_summary_statistics(label)

    # Imprimir Tabla Formal de Paso 1
    print("\n" + "=" * 105)
    print(f" TABLA RESULTADOS PASO 1: COMANDO ANGULAR vs VELOCIDAD ANGULAR REAL ({label.upper()})")
    print("=" * 105)
    print(f"{'Comando':<9} | {'Gyro Z Crudo':<18} | {'Gyro Z Debias':<18} | {'Omega RPM':<18} | {'RPM diff (R-L)':<16} | {'Ratio (°/s/cmd)':<15}")
    print(f"{'[-1..1]':<9} | {'Mean ± Std (°/s)':<18} | {'Mean ± Std (°/s)':<18} | {'Mean ± Std (°/s)':<18} | {'Media (RPM)':<16} | {'':<15}")
    print("-" * 105)

    for r in sweep_results:
        raw_s = f"{r['gz_mean']:+6.2f} ± {r['gz_std']:.2f}"
        deb_s = f"{r['gz_deb_mean']:+6.2f} ± {r['gz_deb_std']:.2f}"
        rpm_s = f"{r['omega_rpm_mean']:+6.2f} ± {r['omega_rpm_std']:.2f}"
        diff_s = f"{r['rpm_diff_mean']:+6.1f}"
        ratio_s = f"{r['ratio']:+6.1f}"
        print(f"{r['cmd']:+6.2f}    | {raw_s:<18} | {deb_s:<18} | {rpm_s:<18} | {diff_s:<16} | {ratio_s:<15}")
    print("=" * 105)
    print(f"\nArchivos guardados en:\n  CSV:   {csv_p}\n  JSONL: {jsonl_p}\n")
    return sweep_results


def run_motor_turn_angle(logger: HeadingDiagnosticLogger, label: str,
                         turn_speeds: Optional[List[float]] = None,
                         target_angle_deg: float = 360.0,
                         duration_per_turn: float = 0.0,
                         rate_hz: float = 10.0,
                         repetitions: int = 1,
                         no_input: bool = False):
    """Paso 2 & Paso 3: Giros motorizados de ángulo conocido (ej. 360°) y comparación Giróscopo vs RPM."""
    if turn_speeds is None:
        turn_speeds = [0.20, 0.45]

    period = 1.0 / rate_hz
    print("\n" + "=" * 95)
    print(f" PROTOCOLO PASO 2: GIROS MOTORIZADOS DE ÁNGULO CONOCIDO ({target_angle_deg:.0f}°) — {label.upper()}")
    print("=" * 95)
    print(f"Ángulo físico objetivo: {target_angle_deg:.1f}°")
    print(f"Velocidades angulares a evaluar: {turn_speeds}")
    print(f"Repeticiones por velocidad: {repetitions}")
    if duration_per_turn > 0:
        print(f"Modo temporizado automático: {duration_per_turn:.1f} s por giro")
    else:
        print("Modo interactivo: Presiona ENTER para iniciar y ENTER para detener al completar la marca.")
    print("=" * 95 + "\n")

    results_turns = []

    try:
        turn_idx = 0
        for speed in turn_speeds:
            for rep in range(1, repetitions + 1):
                turn_idx += 1
                turn_name = f"V{speed:+.2f}_R{rep}"
                print(f"\n---------------------------------------------------------------------------------")
                print(f" >>> GIRO #{turn_idx}: Velocidad = {speed:+.2f} | Repetición {rep}/{repetitions} | Meta: {target_angle_deg:.0f}°")
                print(f"---------------------------------------------------------------------------------")

                if not no_input and sys.stdin.isatty():
                    input(f"Alinea el rover en la marca inicial de 0° y presiona ENTER para iniciar giro...")
                else:
                    print(f"Iniciando giro en 3 segundos...")
                    for c in (3, 2, 1):
                        print(f"  {c}...")
                        time.sleep(1.0)

                # Tomar snapshot inicial en reposo
                pre_sample = logger.poll_sample(marker=f"{turn_name}_pre")
                start_t = time.time()
                start_ori = pre_sample["orientation"] if pre_sample else None
                start_ekf = pre_sample["ekf_heading"] if pre_sample else None

                # Acumuladores continuos (sin envolver a 360°)
                cum_gyro_raw = 0.0
                cum_gyro_deb = 0.0
                cum_rpm_yaw = 0.0
                cum_rpm_raw = 0.0

                turn_samples = []
                last_sample_t = start_t
                import select

                print(f"  [ROTANDO...] Girando a comando {speed:+.2f}. "
                      f"{'Esperando '+str(duration_per_turn)+'s...' if duration_per_turn > 0 else 'Presiona ENTER al alcanzar la marca final.'}")

                while True:
                    t_now = time.time()
                    elapsed = t_now - start_t

                    stop_requested = False
                    if duration_per_turn > 0 and elapsed >= duration_per_turn:
                        stop_requested = True
                    elif sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        stop_requested = True

                    if stop_requested:
                        break

                    # Enviar comando de giro activo
                    logger.client.control(linear=0.0, angular=speed)
                    rec = logger.poll_sample(marker=f"{turn_name}_rotating")
                    if rec:
                        turn_samples.append(rec)
                        dt = rec["timestamp_local"] - last_sample_t
                        if 0.0 < dt < 0.5:
                            cum_gyro_raw += (-rec["gyro_z"]) * dt
                            cum_gyro_deb += (-rec["omega_gyro_deb_dps"]) * dt
                            cum_rpm_yaw += (-rec["omega_rpm_dps"]) * dt
                            cum_rpm_raw += (-rec["rpm_diff"] / logger.track_m) * dt
                        last_sample_t = rec["timestamp_local"]

                        ekf_str = f"{rec['ekf_heading']:5.1f}°" if rec['ekf_heading'] is not None else " N/A "
                        comp_str = f"{rec['orientation']:5.1f}°" if rec['orientation'] is not None else " N/A "
                        print(f"\r  [{elapsed:4.1f}s] ΔDeb: {cum_gyro_deb:+6.1f}° | "
                              f"ΔRPM: {cum_rpm_yaw:+6.1f}° | GyroZ: {rec['gyro_z']:+6.1f} dps | "
                              f"EKF: {ekf_str} | Compás: {comp_str}", end="", flush=True)

                    time.sleep(period)

                # Frenado inmediato doble
                logger.client.stop()
                logger.client.stop()
                turn_duration = time.time() - start_t
                print(f"\n  [FRENO EJECUTADO] Giro completado en {turn_duration:.2f} s.")

                # Reposo post-giro
                time.sleep(0.5)
                post_sample = logger.poll_sample(marker=f"{turn_name}_post")
                end_ori = post_sample["orientation"] if post_sample else None
                end_ekf = post_sample["ekf_heading"] if post_sample else None

                delta_ori = wrap_deg(end_ori - start_ori) if (end_ori is not None and start_ori is not None) else None
                delta_ekf = wrap_deg(end_ekf - start_ekf) if (end_ekf is not None and start_ekf is not None) else None

                mag_deb = abs(cum_gyro_deb)
                mag_rpm = abs(cum_rpm_yaw)
                mag_raw_rpm = abs(cum_rpm_raw)

                scale_gyro = mag_deb / target_angle_deg if target_angle_deg > 0 else 0.0
                scale_rpm = mag_rpm / target_angle_deg if target_angle_deg > 0 else 0.0
                error_gyro = mag_deb - target_angle_deg
                error_rpm = mag_rpm - target_angle_deg
                empirical_k_rpm = target_angle_deg / mag_raw_rpm if mag_raw_rpm > 1e-3 else 0.0

                # Paso 3: Análisis instante a instante (descartando transitorios de arranque y frenado)
                steady_samples = [s for s in turn_samples if 0.8 <= (s["timestamp_local"] - start_t) <= (turn_duration - 0.5)]
                if len(steady_samples) < 5:
                    steady_samples = turn_samples

                gz_series = [s["omega_gyro_deb_dps"] for s in steady_samples]
                rpm_series = [s["omega_rpm_dps"] for s in steady_samples]

                # Imprimir traza muestra a muestra en tramo sostenido (primeras 10 muestras del régimen permanente)
                print(f"\n  [TRAZA MUESTRA A MUESTRA EN CRUCERO ({turn_name})]")
                print(f"  {'#':<4} | {'dt (s)':<7} | {'Gyro Z Crudo':<14} | {'Gyro Z Debias':<14} | {'RPM R-L':<10} | {'Omega RPM (Asum)':<16}")
                print(f"  {'-'*75}")
                trace_subset = steady_samples[:12]
                for idx, s in enumerate(trace_subset):
                    t_rel = s["timestamp_local"] - start_t
                    print(f"  {idx+1:<4} | {t_rel:5.2f} s | {s['gyro_z']:+6.2f} dps    | {s['omega_gyro_deb_dps']:+6.2f} dps    | {s['rpm_diff']:+6.1f}     | {s['omega_rpm_dps']:+6.2f} dps")
                print(f"  {'-'*75}\n")

                corr = 0.0
                mean_rel_err = 0.0
                if len(gz_series) > 3:
                    m_gz = sum(gz_series) / len(gz_series)
                    m_rpm = sum(rpm_series) / len(rpm_series)
                    cov = sum((g - m_gz) * (r - m_rpm) for g, r in zip(gz_series, rpm_series))
                    var_g = sum((g - m_gz) ** 2 for g in gz_series)
                    var_r = sum((r - m_rpm) ** 2 for r in rpm_series)
                    denom = math.sqrt(var_g * var_r)
                    if denom > 1e-6:
                        corr = cov / denom

                    rel_errs = [abs(g - r) / max(1.0, abs(g)) for g, r in zip(gz_series, rpm_series)]
                    mean_rel_err = (sum(rel_errs) / len(rel_errs)) * 100.0

                turn_data = {
                    "name": turn_name,
                    "speed_cmd": speed,
                    "target_deg": target_angle_deg,
                    "duration_s": turn_duration,
                    "cum_gyro_deb": cum_gyro_deb,
                    "cum_rpm_yaw": cum_rpm_yaw,
                    "cum_rpm_raw": cum_rpm_raw,
                    "delta_ori": delta_ori,
                    "delta_ekf": delta_ekf,
                    "scale_gyro": scale_gyro,
                    "scale_rpm": scale_rpm,
                    "error_gyro": error_gyro,
                    "error_rpm": error_rpm,
                    "empirical_k_rpm": empirical_k_rpm,
                    "correlation": corr,
                    "mean_rel_err_pct": mean_rel_err,
                }
                results_turns.append(turn_data)

                time.sleep(2.0)

    finally:
        logger.client.stop()
        logger.client.stop()

    csv_p, jsonl_p = logger.save(label)
    logger.print_summary_statistics(label)

    # Imprimir Tabla Formal de Paso 2
    print("\n" + "=" * 130)
    print(f" TABLA RESULTADOS PASO 2: GIROS DE ÁNGULO CONOCIDO ({target_angle_deg:.0f}°) ({label.upper()})")
    print("=" * 130)
    print(f"{'Giro':<10} | {'Cmd':<5} | {'Real':<6} | {'ΔYaw Gyro':<13} | {'Err Gyro':<10} | {'Factor Gyro':<11} | {'RPM Crudo Int':<14} | {'k_rpm Directo':<14} | {'ΔCompás':<9} | {'ΔEKF':<9}")
    print(f"{'':<10} | {'':<5} | {'':<6} | {'(debias)':<13} | {'(° vs real)':<10} | {'(med/real)':<11} | {'(ΔRPM·s/track)':<14} | {'(°/unid_cruda)':<14} | {'':<9} | {'':<9}")
    print("-" * 130)
    for r in results_turns:
        g_s = f"{r['cum_gyro_deb']:+6.1f}°"
        err_g_s = f"{r['error_gyro']:+5.1f}°"
        fact_g = f"{r['scale_gyro']:.3f}"
        raw_rpm_s = f"{r['cum_rpm_raw']:+8.2f}"
        k_rpm_s = f"{r['empirical_k_rpm']:8.4f}"
        ori_s = f"{r['delta_ori']:+5.1f}°" if r['delta_ori'] is not None else " N/A "
        ekf_s = f"{r['delta_ekf']:+5.1f}°" if r['delta_ekf'] is not None else " N/A "
        print(f"{r['name']:<10} | {r['speed_cmd']:+4.2f} | {r['target_deg']:4.0f}° | {g_s:<13} | {err_g_s:<10} | {fact_g:<11} | {raw_rpm_s:<14} | {k_rpm_s:<14} | {ori_s:<9} | {ekf_s:<9}")
    print("=" * 130)
    print("  * Nota: 'k_rpm Directo' es el factor de calibración empírico independiente del radio de rueda:")
    print("    Multiplicar la integral de (ΔRPM / track)*dt por k_rpm reproduce exactamente el giro en grados.\n")

    # Imprimir Tabla Formal de Paso 3
    print("=" * 95)
    print(f" TABLA RESULTADOS PASO 3: CONCORDANCIA GIRÓSCOPO vs RPM INSTANTE A INSTANTE")
    print("=" * 95)
    print(f"{'Giro':<12} | {'Velocidad Cmd':<14} | {'Correlación (r)':<16} | {'Error Relativo Medio':<22} | {'k_rpm Empírico':<15}")
    print("-" * 95)
    for r in results_turns:
        print(f"{r['name']:<12} | {r['speed_cmd']:+6.2f}         | {r['correlation']:+6.3f}           | {r['mean_rel_err_pct']:5.1f} %                | {r['empirical_k_rpm']:.4f}")
    print("=" * 95)
    print(f"\nArchivos guardados en:\n  CSV:   {csv_p}\n  JSONL: {jsonl_p}\n")
    return results_turns


def main():
    parser = argparse.ArgumentParser(description="Logger de Diagnóstico de Heading y Sensores Inerciales")
    parser.add_argument("--base-url", default="http://localhost:8000", help="URL del servidor HTTP del SDK")
    parser.add_argument("--duration", type=float, default=0.0, help="Duración de captura en segundos (0 = interactivo)")
    parser.add_argument("--turns-isolated", action="store_true", help="Protocolo Test B': giros conocidos con rotación aislada")
    parser.add_argument("--straight-track", action="store_true", help="Protocolo Test D: avance en recta para gps_track")
    parser.add_argument("--reverse", action="store_true", help="Para Test D: tramo marcha atrás")
    parser.add_argument("--reanalyze", type=str, default="", help="Ruta a CSV de Test A para re-analizar con debiasado")
    parser.add_argument("--bias-z", type=float, default=1.027725, help="Bias del giróscopo Z en deg/s (default: 1.027725)")
    parser.add_argument("--label", default="diagnostico", help="Etiqueta para identificar el test")
    parser.add_argument("--output-dir", default="diagnostico_heading", help="Directorio donde guardar los logs")
    parser.add_argument("--rate", type=float, default=10.0, help="Frecuencia de muestreo en Hz (default: 10)")

    # Flags nuevos para Test de Giro Motorizado (Pasos 1, 2 y 3)
    parser.add_argument("--motor-sweep", action="store_true", help="Paso 1: Barrido de comandos angulares en el lugar")
    parser.add_argument("--angulars", type=str, default="0.20,0.30,0.45", help="Comandos angulares separados por coma (default: 0.20,0.30,0.45)")
    parser.add_argument("--step-duration", type=float, default=5.0, help="Duración en segundos por escalón en sweep (default: 5.0)")
    parser.add_argument("--pause-duration", type=float, default=3.0, help="Pausa en segundos entre giros en sweep (default: 3.0)")

    parser.add_argument("--motor-turn-angle", action="store_true", help="Paso 2 y 3: Giros motorizados de ángulo conocido (ej. 360°)")
    parser.add_argument("--turn-speeds", type=str, default="0.20,0.45", help="Velocidades angulares separadas por coma para --motor-turn-angle (default: 0.20,0.45)")
    parser.add_argument("--target-angle", type=float, default=360.0, help="Ángulo objetivo físico en grados (default: 360.0)")
    parser.add_argument("--duration-per-turn", type=float, default=0.0, help="Duración fija opcional por giro en segundos (0 = presionar ENTER para frenar)")
    parser.add_argument("--repetitions", type=int, default=1, help="Repeticiones por velocidad en --motor-turn-angle (default: 1)")
    parser.add_argument("--no-input", action="store_true", help="Omitir prompts interactivos e iniciar con cuenta regresiva")

    args = parser.parse_args()

    if args.reanalyze:
        run_reanalyze_csv(args.reanalyze, bias_z=args.bias_z)
        return

    logger = HeadingDiagnosticLogger(base_url=args.base_url, output_dir=args.output_dir, bias_z=args.bias_z)

    if args.motor_sweep:
        ang_list = [float(x.strip()) for x in args.angulars.split(",") if x.strip()]
        run_motor_sweep(logger, args.label, angulars=ang_list, step_duration=args.step_duration,
                        pause_duration=args.pause_duration, rate_hz=args.rate, no_input=args.no_input)
    elif args.motor_turn_angle:
        spd_list = [float(x.strip()) for x in args.turn_speeds.split(",") if x.strip()]
        run_motor_turn_angle(logger, args.label, turn_speeds=spd_list, target_angle_deg=args.target_angle,
                             duration_per_turn=args.duration_per_turn, rate_hz=args.rate,
                             repetitions=args.repetitions, no_input=args.no_input)
    elif args.turns_isolated:
        run_turns_isolated(logger, args.label, rate_hz=args.rate)
    elif args.straight_track:
        run_straight_track(logger, args.label, is_reverse=args.reverse, rate_hz=args.rate)
    elif args.duration > 0:
        run_fixed_duration(logger, args.duration, args.label, rate_hz=args.rate)
    else:
        print("[INFO] Sin flags específicos. Ejecutando prueba de 5 segundos...")
        run_fixed_duration(logger, 5.0, args.label, rate_hz=args.rate)


if __name__ == "__main__":
    main()
