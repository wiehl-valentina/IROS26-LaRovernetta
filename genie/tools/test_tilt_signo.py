"""Test de signo del tilt con el rover (NO mueve motores).

Paso 0:
  - Conecta al SDK igual que el bridge (reusar RoverClient y carga de yaml).
  - Toma 1 frame + 2 s de telemetria, calcula roll/pitch con Odometry.update + current_roll_pitch().
  - Procesa el frame con PerceptionPipeline dos veces: con (roll, pitch) y con (None, None).
  - Guarda en genie/debug/tilt_signo/<timestamp>/:
      * frame RGB (frame_rgb.png)
      * BEV_con_correccion.png
      * BEV_sin_correccion.png
    con líneas horizontales dibujadas a 1.0 m y 1.5 m.
  - Imprime roll y pitch en grados.

Uso:
    python tools/test_tilt_signo.py --config configs/frodobot_rover.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import yaml

# Asegurar importacion de modulos locales
_THIS_DIR = Path(__file__).resolve().parent
_GENIE_DIR = _THIS_DIR.parent
if str(_GENIE_DIR) not in sys.path:
    sys.path.insert(0, str(_GENIE_DIR))

from genie_rover.odometry import Odometry, OdometryConfig, estimate_roll_pitch
from genie_rover.perception import PerceptionPipeline
from genie_rover.sdk_client import RoverClient


def _bev_to_bgr_vis(trav: np.ndarray, observed: np.ndarray | None,
                     resolution_m_per_px: float,
                     scale: int = 6,
                     title: str = "") -> np.ndarray:
    """Convierte matriz de transitabilidad [-1, 1] a imagen BGR escalada con lineas a 1.0m y 1.5m."""
    h, w = trav.shape
    img_bgr = np.full((h, w, 3), 40, dtype=np.uint8)  # Gris oscuro para no observado

    if observed is not None:
        valid = (observed > 0) & (trav >= 0.0)
    else:
        valid = (trav >= 0.0)

    # 0 = obstaculo (rojo), 1 = transitable (verde)
    # BGR: rojo = (0, 0, 255), verde = (0, 255, 0)
    val = np.clip(trav, 0.0, 1.0)
    img_bgr[valid, 0] = 0                                       # B
    img_bgr[valid, 1] = (val[valid] * 255.0).astype(np.uint8)   # G
    img_bgr[valid, 2] = ((1.0 - val[valid]) * 255.0).astype(np.uint8)  # R

    # Escalar imagen para visualización nítida
    w_scaled = w * scale
    h_scaled = h * scale
    out_bgr = cv2.resize(img_bgr, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)

    # Coordenadas de filas segun projection.py:
    # row = bev_h - 1 - np.floor(forward_m / bev_resolution_m_per_px)
    # En la imagen escalada: y = int((row + 0.5) * scale)
    for dist_m, color, label in [
        (1.0, (0, 255, 255), "1.0 m"),   # Amarillo (BGR)
        (1.5, (255, 200, 0), "1.5 m"),   # Celeste/Cyan (BGR)
    ]:
        row = h - 1 - int(np.floor(dist_m / resolution_m_per_px))
        if 0 <= row < h:
            y = int((row + 0.5) * scale)
            cv2.line(out_bgr, (0, y), (w_scaled - 1, y), color, thickness=2)
            cv2.putText(
                out_bgr,
                label,
                (10, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    if title:
        cv2.putText(
            out_bgr,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return out_bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Test de signo de tilt con rover")
    parser.add_argument("--config", default=str(_GENIE_DIR / "configs" / "frodobot_rover.yaml"),
                        help="Ruta al archivo yaml de configuracion")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duracion de adquisicion de telemetria en segundos (def: 2.0)")
    parser.add_argument("--tag", type=str, default="",
                        help="Etiqueta opcional para identificar la corrida (ej. plano, morro_arriba, morro_abajo)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _GENIE_DIR / args.config
    if not cfg_path.is_file():
        print(f"Error: no se encontro el archivo de configuracion {args.config}")
        return 1

    cfg = yaml.safe_load(cfg_path.read_text())

    # Cambiar al directorio de genie para que rutas relativas (sam2, pesos) funcionen
    os.chdir(_GENIE_DIR)

    print(f"[test_tilt_signo] Conectando a {cfg['rover']['base_url']} ...")
    client = RoverClient(cfg["rover"]["base_url"], timeout=cfg["rover"].get("timeout_s", 5.0))

    # Inicializar Odometria
    odo_cfg = cfg.get("odometry", {})
    odometry = Odometry(OdometryConfig(
        wheel_radius_m=float(odo_cfg.get("wheel_radius_m", 0.0475)),
        track_width_m=float(odo_cfg.get("track_width_m", 0.15)),
        left_rpm_indices=tuple(odo_cfg.get("left_rpm_indices", (0, 2))),
        right_rpm_indices=tuple(odo_cfg.get("right_rpm_indices", (1, 3))),
        rotation_sign=float(odo_cfg.get("rotation_sign", 1.0)),
        use_gyro_for_rotation=bool(odo_cfg.get("use_gyro_for_rotation", True)),
        gps_correction=bool(odo_cfg.get("gps_correction", True)),
        min_gps_displacement_m=float(odo_cfg.get("min_gps_displacement_m", 1.0)),
        gyro_yaw_index=int(odo_cfg.get("gyro_yaw_index", 2)),
        gyro_sign=float(odo_cfg.get("gyro_sign", 1.0)),
        gyro_yaw_bias_dps=float(odo_cfg.get("gyro_yaw_bias_dps", 1.2784)),
        gyro_deadband_dps=float(odo_cfg.get("gyro_deadband_dps", 0.5)),
        ekf_heading_correction=bool(odo_cfg.get("ekf_heading_correction", True)),
        ekf_heading_max_age_s=float(odo_cfg.get("ekf_heading_max_age_s", 1.5)),
        heading_blend=float(odo_cfg.get("heading_blend", 0.3)),
        heading_blend_tau_s=float(odo_cfg.get("heading_blend_tau_s", 4.0)),
        accel_gate_norm_tol=float(odo_cfg.get("accel_gate_norm_tol", 0.08)),
        accel_gate_std_tol=float(odo_cfg.get("accel_gate_std_tol", 0.06)),
        gyro_bias_x_dps=float(odo_cfg.get("gyro_bias_x_dps", 0.0831)),
        gyro_bias_y_dps=float(odo_cfg.get("gyro_bias_y_dps", -0.0098)),
        use_tilt_projection=bool(odo_cfg.get("use_tilt_projection", True)),
        tilt_blend_start_deg=float(odo_cfg.get("tilt_blend_start_deg", 5.0)),
        tilt_blend_max_deg=float(odo_cfg.get("tilt_blend_max_deg", 20.0)),
        tilt_max_staleness_s=float(odo_cfg.get("tilt_max_staleness_s", 5.0)),
    ))

    # Inicializar PerceptionPipeline
    print("[test_tilt_signo] Inicializando PerceptionPipeline...")
    pipeline = PerceptionPipeline(cfg)

    # 1. Capturar frame RGB frontal
    print("[test_tilt_signo] Capturando frame frontal de la camara...")
    rgb, frame_ts = client.front_frame()
    print(f"  ✓ Frame obtenido: {rgb.shape[1]}x{rgb.shape[0]} px (ts={frame_ts:.3f})")

    # 2. Adquirir telemetria durante N segundos
    print(f"[test_tilt_signo] Muestreando telemetria durante {args.duration:.1f} s ...")
    t_start = time.time()
    last_raw = {}
    sample_count = 0
    while (time.time() - t_start) < args.duration:
        now = time.time()
        telem = client.telemetry()
        last_raw = telem.raw
        odometry.update(
            telem.raw,
            ekf_heading=getattr(telem, "ekf_heading", None),
            ekf_timestamp=getattr(telem, "ekf_heading_time", None),
            now=now,
        )
        sample_count += 1
        time.sleep(0.05)

    print(f"  ✓ {sample_count} muestras de telemetria integradas.")

    # 3. Calcular roll / pitch
    roll_pitch = odometry.current_roll_pitch(now=time.time())
    if roll_pitch is None:
        print("[test_tilt_signo] AVISO: Odometry.current_roll_pitch devolvio None (gate cerrado).")
        tilt_direct = estimate_roll_pitch(
            last_raw.get("accels", []),
            gate_norm_tol=0.20,
            gate_std_tol=0.15,
        )
        if tilt_direct is not None:
            roll_pitch = (tilt_direct[0], tilt_direct[1])
            print("  -> Usando fallback directo con tolerancia relajada.")
        else:
            print("  -> No se pudo obtener estimacion de tilt. Usando (0.0, 0.0).")
            roll_pitch = (0.0, 0.0)

    roll_rad, pitch_rad = roll_pitch
    roll_deg = math.degrees(roll_rad)
    pitch_deg = math.degrees(pitch_rad)

    print("=" * 60)
    print(f"  ROLL:   {roll_deg:+.2f}° ({roll_rad:+.4f} rad)")
    print(f"  PITCH:  {pitch_deg:+.2f}° ({pitch_rad:+.4f} rad)")
    if odometry.last_accel_norm is not None:
        print(f"  |a|:    {odometry.last_accel_norm:.3f} g (gate open: {odometry.tilt_gate_open})")
    print("=" * 60)

    # 4. Procesar el frame dos veces
    print("\n[test_tilt_signo] Procesando con correccion de tilt...")
    res_corr = pipeline.process(rgb, roll_rad=roll_rad, pitch_rad=pitch_rad)

    print("[test_tilt_signo] Procesando sin correccion de tilt...")
    res_nocorr = pipeline.process(rgb, roll_rad=None, pitch_rad=None)

    # 5. Guardar en genie/debug/tilt_signo/<timestamp>/
    tag_str = f"_{args.tag}" if args.tag else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + tag_str
    out_dir = _GENIE_DIR / "debug" / "tilt_signo" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frame RGB
    rgb_path = out_dir / "frame_rgb.png"
    Image.fromarray(rgb).save(rgb_path)

    # BEV con correccion
    res_m = float(cfg["projection"]["resolution_m_per_px"])
    bev_corr_bgr = _bev_to_bgr_vis(
        res_corr.traversability,
        res_corr.observed,
        resolution_m_per_px=res_m,
        title=f"CON corr: roll={roll_deg:+.1f} deg, pitch={pitch_deg:+.1f} deg",
    )
    corr_path = out_dir / "BEV_con_correccion.png"
    cv2.imwrite(str(corr_path), bev_corr_bgr)

    # BEV sin correccion
    bev_nocorr_bgr = _bev_to_bgr_vis(
        res_nocorr.traversability,
        res_nocorr.observed,
        resolution_m_per_px=res_m,
        title="SIN corr: pose base nivelada",
    )
    nocorr_path = out_dir / "BEV_sin_correccion.png"
    cv2.imwrite(str(nocorr_path), bev_nocorr_bgr)

    # Guardar tambien info en texto para facil referencia
    info_path = out_dir / "info.txt"
    info_path.write_text(
        f"timestamp: {timestamp}\n"
        f"roll_deg: {roll_deg:+.4f}\n"
        f"pitch_deg: {pitch_deg:+.4f}\n"
        f"roll_rad: {roll_rad:+.6f}\n"
        f"pitch_rad: {pitch_rad:+.6f}\n"
        f"accel_norm_g: {odometry.last_accel_norm}\n"
        f"tilt_gate_open: {odometry.tilt_gate_open}\n"
    )

    print(f"\n[test_tilt_signo] Resultados guardados en: {out_dir}")
    print(f"  - {rgb_path.name}")
    print(f"  - {corr_path.name}")
    print(f"  - {nocorr_path.name}")
    print(f"  - {info_path.name}")
    print("\nListo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
