"""Test remoto de signo de tilt (100% autónomo, NO requiere operador presencial ni mover el robot).

Metodología:
1. Muestrea telemetría estática del MPU6050 y captura un frame RGB en la posición actual del rover.
2. Calcula la inclinación actual (roll y pitch) vía Odometry / estimate_roll_pitch.
3. Proyecta el BEV bajo tres hipótesis:
   - Hipótesis 0: Pose base sin corrección (pitch = 0).
   - Hipótesis (+): Pose con pitch = +p (convención anterior: x -> [cos p, 0, -sin p]).
   - Hipótesis (-): Pose con pitch = -p (signo invertido: x -> [cos p, 0, +sin p]).
4. Analiza la transitabilidad en el corredor frontal cercano (0.4 m a 1.2 m):
   - Con el signo correcto, un terreno transitable en pendiente mantiene su clearance frontal.
   - Con el signo invertido, los rayos lejanos impactan falsamente en el suelo a 0.5 m, generando
     un muro de falsos obstáculos y bloqueando el frente.
5. Guarda las 3 imágenes BEV y el frame RGB en genie/debug/tilt_remoto/<timestamp>/.

Uso:
    python tools/test_tilt_remoto.py --config configs/frodobot_rover.yaml
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

_THIS_DIR = Path(__file__).resolve().parent
_GENIE_DIR = _THIS_DIR.parent
if str(_GENIE_DIR) not in sys.path:
    sys.path.insert(0, str(_GENIE_DIR))

from genie_rover.odometry import Odometry, OdometryConfig, estimate_roll_pitch
from genie_rover.perception import PerceptionPipeline
from genie_rover.sdk_client import RoverClient


def _bev_to_bgr(trav: np.ndarray, observed: np.ndarray | None,
                resolution_m_per_px: float, scale: int = 6,
                title: str = "") -> np.ndarray:
    h, w = trav.shape
    img_bgr = np.full((h, w, 3), 40, dtype=np.uint8)

    if observed is not None:
        valid = (observed > 0) & (trav >= 0.0)
    else:
        valid = (trav >= 0.0)

    val = np.clip(trav, 0.0, 1.0)
    img_bgr[valid, 0] = 0
    img_bgr[valid, 1] = (val[valid] * 255.0).astype(np.uint8)   # G
    img_bgr[valid, 2] = ((1.0 - val[valid]) * 255.0).astype(np.uint8)  # R

    w_scaled = w * scale
    h_scaled = h * scale
    out_bgr = cv2.resize(img_bgr, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)

    for dist_m, color, label in [(1.0, (0, 255, 255), "1.0 m"), (1.5, (255, 200, 0), "1.5 m")]:
        row = h - 1 - int(np.floor(dist_m / resolution_m_per_px))
        if 0 <= row < h:
            y = int((row + 0.5) * scale)
            cv2.line(out_bgr, (0, y), (w_scaled - 1, y), color, thickness=2)
            cv2.putText(out_bgr, label, (10, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness=2, lineType=cv2.LINE_AA)

    if title:
        cv2.putText(out_bgr, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return out_bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Test remoto de signo de tilt")
    parser.add_argument("--config", default=str(_GENIE_DIR / "configs" / "frodobot_rover.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _GENIE_DIR / args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    os.chdir(_GENIE_DIR)

    client = RoverClient(cfg["rover"]["base_url"], timeout=cfg["rover"].get("timeout_s", 5.0))
    odo_cfg = cfg.get("odometry", {})
    odometry = Odometry(OdometryConfig(
        wheel_radius_m=float(odo_cfg.get("wheel_radius_m", 0.0475)),
        track_width_m=float(odo_cfg.get("track_width_m", 0.15)),
        left_rpm_indices=tuple(odo_cfg.get("left_rpm_indices", (0, 2))),
        right_rpm_indices=tuple(odo_cfg.get("right_rpm_indices", (1, 3))),
        rotation_sign=float(odo_cfg.get("rotation_sign", 1.0)),
        use_gyro_for_rotation=bool(odo_cfg.get("use_gyro_for_rotation", True)),
        gyro_yaw_index=int(odo_cfg.get("gyro_yaw_index", 2)),
        gyro_sign=float(odo_cfg.get("gyro_sign", 1.0)),
        gyro_yaw_bias_dps=float(odo_cfg.get("gyro_yaw_bias_dps", 1.2784)),
        gyro_deadband_dps=float(odo_cfg.get("gyro_deadband_dps", 0.5)),
        accel_gate_norm_tol=float(odo_cfg.get("accel_gate_norm_tol", 0.08)),
        accel_gate_std_tol=float(odo_cfg.get("accel_gate_std_tol", 0.06)),
    ))

    pipeline = PerceptionPipeline(cfg)

    print("[test_tilt_remoto] Adquiriendo frame y telemetría...")
    rgb, _ = client.front_frame()

    for _ in range(15):
        t = client.telemetry()
        odometry.update(t.raw, now=time.time())
        time.sleep(0.05)

    roll_pitch = odometry.current_roll_pitch(now=time.time())
    if roll_pitch is None:
        t_raw = client.telemetry().raw
        est = estimate_roll_pitch(t_raw.get("accels", []), gate_norm_tol=0.20, gate_std_tol=0.15)
        roll_pitch = (est[0], est[1]) if est else (0.0, 0.0)

    r_rad, p_rad = roll_pitch
    r_deg, p_deg = math.degrees(r_rad), math.degrees(p_rad)

    print("=" * 65)
    print(f"  TELEMETRÍA EN REPOSO:")
    print(f"  Roll medido:  {r_deg:+.2f}° ({r_rad:+.4f} rad)")
    print(f"  Pitch medido: {p_deg:+.2f}° ({p_rad:+.4f} rad)")
    print("=" * 65)

    # 1. Sin corrección (pose nominal nivelada)
    pipeline._cached_roll = None
    pipeline._cached_pitch = None
    res_zero = pipeline.process(rgb, roll_rad=None, pitch_rad=None)

    # 2. Con corrección actual (+p)
    pipeline._cached_roll = None
    pipeline._cached_pitch = None
    res_plus = pipeline.process(rgb, roll_rad=r_rad, pitch_rad=p_rad)

    # 3. Con corrección invertida (-p)
    pipeline._cached_roll = None
    pipeline._cached_pitch = None
    res_minus = pipeline.process(rgb, roll_rad=r_rad, pitch_rad=-p_rad)

    # Evaluación cuantitativa en corredor frontal [0.4 m, 1.2 m]
    res_m = float(cfg["projection"]["resolution_m_per_px"])
    bev_h, bev_w = res_zero.traversability.shape
    row_04 = bev_h - 1 - int(np.floor(0.40 / res_m))
    row_12 = bev_h - 1 - int(np.floor(1.20 / res_m))
    col_c = bev_w // 2
    col_half_w = int(np.ceil(0.25 / res_m))  # ancho corredor ±25 cm
    c_slice = slice(col_c - col_half_w, col_c + col_half_w + 1)
    r_slice = slice(min(row_12, row_04), max(row_12, row_04) + 1)

    def eval_corridor(trav, obs):
        sub_t = trav[r_slice, c_slice]
        sub_o = obs[r_slice, c_slice]
        valid = (sub_o > 0) & (sub_t >= 0.0)
        if not np.any(valid):
            return 0.0, 0
        free_ratio = float(np.mean(sub_t[valid] >= 0.5))
        obs_count = int(np.sum(sub_t[valid] < 0.5))
        return free_ratio, obs_count

    free_0, obs_0 = eval_corridor(res_zero.traversability, res_zero.observed)
    free_p, obs_p = eval_corridor(res_plus.traversability, res_plus.observed)
    free_m, obs_m = eval_corridor(res_minus.traversability, res_minus.observed)

    print("\nANÁLISIS DEL CORREDOR FRONTAL (0.4 m a 1.2 m):")
    print(f"  [0] Sin corrección (p=0°):     libre = {free_0*100:5.1f}%, celdas obstáculo = {obs_0:3d}")
    print(f"  [+] Con pitch actual (+{p_deg:.1f}°):   libre = {free_p*100:5.1f}%, celdas obstáculo = {obs_p:3d}")
    print(f"  [-] Con pitch invertido (-{p_deg:.1f}°): libre = {free_m*100:5.1f}%, celdas obstáculo = {obs_m:3d}")

    out_dir = _GENIE_DIR / "debug" / "tilt_remoto" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(out_dir / "frame_rgb.png")
    cv2.imwrite(str(out_dir / "BEV_sin_correccion.png"),
                _bev_to_bgr(res_zero.traversability, res_zero.observed, res_m, title="Base (p=0)"))
    cv2.imwrite(str(out_dir / "BEV_con_pitch_actual_plus.png"),
                _bev_to_bgr(res_plus.traversability, res_plus.observed, res_m, title=f"Actual (+{p_deg:.1f} deg)"))
    cv2.imwrite(str(out_dir / "BEV_con_pitch_invertido_minus.png"),
                _bev_to_bgr(res_minus.traversability, res_minus.observed, res_m, title=f"Invertido (-{p_deg:.1f} deg)"))

    print(f"\nImágenes guardadas en: {out_dir}")

    if obs_p > obs_m and free_m > free_p:
        print("\n>>> DIAGNÓSTICO: SIGNO INVERTIDO CONFIRMADO REMOTAMENTE.")
        print("    '+p' colapsa el BEV frontal con falsos obstáculos.")
        print("    '-p' preserva la transitabilidad real del terreno en pendiente.")
        print("    ACCIÓN: En perception.py::process pasar -p a camera_pose_with_tilt.")
    else:
        print("\n>>> DIAGNÓSTICO: Signo actual parece consistente.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
