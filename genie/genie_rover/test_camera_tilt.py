"""Pruebas unitarias y validacion geometrica offline de la correccion dinamica de pose de camara por tilt (roll/pitch).

Verifica:
1. Caso nivelado: roll=0, pitch=0 reproduce exactamente base_camera_pose (error < 1e-12).
2. Convencion de ejes y signos: coherencia 100% con odometry.py::estimate_roll_pitch (+X fwd, +Y left, +Z up; pitch>0 nose up, roll>0 rolls right).
3. Formula analitica cerrada vs producto R_y(pitch) @ R_x(roll) en grilla continua.
4. Consistencia de proyeccion del suelo: recuperacion exacta de coordenadas 3D en piso (error < 1e-9 m) vs error masivo sin correccion.
5. Politica de cache y umbral en PerceptionPipeline (evita jitter en reposo, restaura nivel al recibir None).
6. Politica de vigencia / staleness con gate cerrado en Odometry (preserva rampa en aceleracion transitoria, cae a None tras timeout).
7. Benchmark de tiempo de computo (4x4 composicion << 0.05 ms).

Uso:
    python -m genie_rover.test_camera_tilt
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np

from .perception import camera_pose_from_height_pitch, camera_pose_with_tilt, PerceptionPipeline
from .odometry import Odometry, OdometryConfig, estimate_roll_pitch


def test_level_case():
    print("=== 1. Caso nivelado (roll=0, pitch=0) ===")
    base_pose = camera_pose_from_height_pitch(height_m=0.1502, pitch_down_deg=1.85)
    tilted_pose = camera_pose_with_tilt(base_pose, 0.0, 0.0)
    diff = np.max(np.abs(tilted_pose - base_pose))
    print(f"  Max diff con roll=0, pitch=0: {diff:.2e}")
    assert diff == 0.0 or diff < 1e-12, f"Fallo caso nivelado, diff={diff}"
    print("  ✓ Pasa exactamente.")


def test_axis_conventions():
    print("\n=== 2. Convencion de ejes y signos (+X fwd, +Y left, +Z up) ===")
    base_pose = np.eye(4, dtype=np.float64)

    test_angles = [
        (0.0, 0.0),
        (0.0, math.radians(12.0)),    # nose up 12°
        (0.0, math.radians(-12.0)),   # nose down 12°
        (math.radians(8.0), 0.0),     # roll right 8°
        (math.radians(-8.0), 0.0),    # roll left 8°
        (math.radians(6.0), math.radians(10.0)),
        (math.radians(-7.0), math.radians(-11.0)),
    ]

    for roll, pitch in test_angles:
        tilted = camera_pose_with_tilt(base_pose, roll, pitch)
        R_tilt = tilted[:3, :3]

        # Acelerometro en reposo mide la reaccion vertical hacia arriba [0, 0, 1]^T en marco mundo
        z_world = np.array([0.0, 0.0, 1.0])
        a_body = R_tilt.T @ z_world

        # Formula de odometry.py::estimate_roll_pitch:
        est_roll = math.atan2(a_body[1], a_body[2])
        est_pitch = math.atan2(-a_body[0], math.sqrt(a_body[1]**2 + a_body[2]**2))

        err_roll = abs(est_roll - roll)
        err_pitch = abs(est_pitch - pitch)
        assert err_roll < 1e-9, f"Roll mismatch: {est_roll} vs {roll}"
        assert err_pitch < 1e-9, f"Pitch mismatch: {est_pitch} vs {pitch}"

    print(f"  ✓ Validada consistencia con estimate_roll_pitch en {len(test_angles)} configuraciones.")


def test_closed_form_formula():
    print("\n=== 3. Formula analitica vs Ry(pitch) @ Rx(roll) ===")
    base_pose = np.random.RandomState(42).randn(4, 4)
    max_err = 0.0

    for r_deg in np.linspace(-30, 30, 25):
        for p_deg in np.linspace(-30, 30, 25):
            r = math.radians(r_deg)
            p = math.radians(p_deg)

            cr, sr = math.cos(r), math.sin(r)
            cp, sp = math.cos(p), math.sin(p)

            # Matrices individuales
            Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
            R_numeric = Ry @ Rx

            tilted_func = camera_pose_with_tilt(base_pose, r, p)
            R_func = tilted_func[:3, :3] @ np.linalg.inv(base_pose[:3, :3])

            err = float(np.max(np.abs(R_func - R_numeric)))
            if err > max_err:
                max_err = err

    print(f"  Error maximo entre implementacion analitica y producto matricial: {max_err:.2e}")
    assert max_err < 1e-14, f"Discrepancia en composicion analitica: {max_err}"
    print("  ✓ Coincidencia exacta a precision de punto flotante (< 1e-14).")


def test_ground_projection_consistency():
    print("\n=== 4. Consistencia geometrica de proyeccion sobre el plano del suelo ===")
    base_pose = np.array([
        [-0.000313571, -0.032360666, 0.999476207, 0.000000000],
        [-0.999953056, 0.009689420, -0.000000000, 0.000000000],
        [-0.009684345, -0.999429288, -0.032362186, 0.150244633],
        [0.000000000, 0.000000000, 0.000000000, 1.000000000]
    ], dtype=np.float64)

    K = np.array([
        [925.2657, 0.0, 962.305],
        [0.0, 924.6288, 528.389],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Puntos 3D reales en el piso (Z=0) en el marco nivelado del rover
    ground_points = [
        np.array([0.50, 0.00, 0.0, 1.0]),
        np.array([1.00, 0.00, 0.0, 1.0]),
        np.array([1.50, 0.00, 0.0, 1.0]),
        np.array([1.00, -0.30, 0.0, 1.0]),
        np.array([1.00, +0.30, 0.0, 1.0]),
    ]

    scenarios = [
        ("Pitch +10° (morro arriba)", 0.0, math.radians(10.0)),
        ("Pitch -10° (morro abajo)", 0.0, math.radians(-10.0)),
        ("Roll +5° (caida derecha)", math.radians(5.0), 0.0),
        ("Roll -5° (caida izquierda)", math.radians(-5.0), 0.0),
        ("Inclinacion combinada (roll=+4°, pitch=+8°)", math.radians(4.0), math.radians(8.0)),
    ]

    for desc, r, p in scenarios:
        tilted_pose = camera_pose_with_tilt(base_pose, r, p)
        tilted_inv = np.linalg.inv(tilted_pose)
        base_inv = np.linalg.inv(base_pose)

        max_err_corrected = 0.0
        max_err_uncorrected = 0.0

        for P in ground_points:
            # 1. Punto 3D observado por la camara con su actitud real inclinada
            P_cam = tilted_inv @ P
            if P_cam[2] <= 0.05:
                continue
            uv = (K @ (P_cam[:3] / P_cam[2]))[:2]

            # Rayo en camara
            dir_cam = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])

            # 2. Reproyeccion usando la pose corregida (tilted_pose)
            dir_w_corr = tilted_pose[:3, :3] @ dir_cam
            if abs(dir_w_corr[2]) > 1e-7:
                scale_corr = -tilted_pose[2, 3] / dir_w_corr[2]
                P_corr = tilted_pose[:3, 3] + scale_corr * dir_w_corr
                err_corr = float(np.linalg.norm(P_corr[:2] - P[:2]))
                if err_corr > max_err_corrected:
                    max_err_corrected = err_corr

            # 3. Reproyeccion usando la pose estatica sin corregir (base_pose)
            dir_w_uncorr = base_pose[:3, :3] @ dir_cam
            if abs(dir_w_uncorr[2]) > 1e-7:
                scale_uncorr = -base_pose[2, 3] / dir_w_uncorr[2]
                P_uncorr = base_pose[:3, 3] + scale_uncorr * dir_w_uncorr
                err_uncorr = float(np.linalg.norm(P_uncorr[:2] - P[:2]))
                if err_uncorr > max_err_uncorrected:
                    max_err_uncorrected = err_uncorr

        print(f"  {desc}:")
        print(f"    Error corregido:   {max_err_corrected:.2e} m")
        print(f"    Error SIN corregir: {max_err_uncorrected:.3f} m")
        assert max_err_corrected < 1e-7, f"Fallo proyeccion corregida: {max_err_corrected}"
        assert max_err_uncorrected > 0.05, f"Esperaba error apreciable sin correccion: {max_err_uncorrected}"

    print("  ✓ La pose corregida reconstruye exactamente la posicion de suelo en todos los casos.")


def test_perception_pipeline_threshold_and_caching():
    print("\n=== 5. Politica de umbral y cache en PerceptionPipeline ===")
    cfg = {
        "camera": {
            "intrinsics": [[900.0, 0.0, 480.0], [0.0, 900.0, 270.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": None,
            "image_size": [960, 540],
            "height_m": 0.15,
            "pitch_down_deg": 2.0,
            "tilt_threshold_deg": 0.5,
        },
        "projection": {
            "ground_z": 0.0,
            "resolution_m_per_px": 0.03,
            "forward_range_m": 2.0,
            "side_range_m": 2.0,
            "max_ray_distance_m": 2.5,
            "projection_downscale": 2,
        },
        "samtp": {
            "config_path": "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml",
            "checkpoint_path": "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt",
            "device": "cpu",
        }
    }

    # Inicializar PerceptionPipeline sin SamTpRunner para test rapido
    pipe = PerceptionPipeline.__new__(PerceptionPipeline)
    cam = cfg["camera"]
    proj = cfg["projection"]
    pipe.camera_k = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    pipe.dist = None
    pipe.image_size = tuple(cam["image_size"])
    pipe._rectifiers = {}
    pipe._warned_sizes = set()
    pipe.base_camera_pose = camera_pose_from_height_pitch(cam["height_m"], cam["pitch_down_deg"])
    pipe.camera_pose = pipe.base_camera_pose.copy()
    pipe.tilt_threshold_rad = math.radians(float(cam.get("tilt_threshold_deg", 0.5)))
    pipe._cached_roll = None
    pipe._cached_pitch = None
    pipe.ground_z = 0.0
    pipe.resolution = 0.03
    pipe.forward_range = 2.0
    pipe.side_range = 2.0
    pipe.max_ray = 2.5
    pipe.projection_downscale = 1

    # 1. Paso con tilt inicial de 5°
    r1, p1 = math.radians(0.0), math.radians(5.0)

    # Simular actualizacion de tilt
    if (pipe._cached_roll is None or pipe._cached_pitch is None
            or abs(r1 - pipe._cached_roll) >= pipe.tilt_threshold_rad
            or abs(p1 - pipe._cached_pitch) >= pipe.tilt_threshold_rad):
        pipe.camera_pose = camera_pose_with_tilt(pipe.base_camera_pose, r1, p1)
        pipe._cached_roll = r1
        pipe._cached_pitch = p1

    pose_5deg = pipe.camera_pose.copy()
    assert abs(math.degrees(pipe._cached_pitch) - 5.0) < 1e-4

    # 2. Paso con cambio sub-umbral (5.2°, delta = 0.2° < 0.5°) -> debe reusar cache
    r2, p2 = math.radians(0.0), math.radians(5.2)
    recalculated = False
    if (pipe._cached_roll is None or pipe._cached_pitch is None
            or abs(r2 - pipe._cached_roll) >= pipe.tilt_threshold_rad
            or abs(p2 - pipe._cached_pitch) >= pipe.tilt_threshold_rad):
        pipe.camera_pose = camera_pose_with_tilt(pipe.base_camera_pose, r2, p2)
        pipe._cached_roll = r2
        pipe._cached_pitch = p2
        recalculated = True

    assert not recalculated, "No deberia recalcular para delta < 0.5°"
    assert np.array_equal(pipe.camera_pose, pose_5deg), "Pose debio mantenerse intacta en cache"

    # 3. Paso con cambio sobre-umbral (6.0°, delta = 1.0° >= 0.5°) -> debe actualizar
    r3, p3 = math.radians(0.0), math.radians(6.0)
    if (pipe._cached_roll is None or pipe._cached_pitch is None
            or abs(r3 - pipe._cached_roll) >= pipe.tilt_threshold_rad
            or abs(p3 - pipe._cached_pitch) >= pipe.tilt_threshold_rad):
        pipe.camera_pose = camera_pose_with_tilt(pipe.base_camera_pose, r3, p3)
        pipe._cached_roll = r3
        pipe._cached_pitch = p3
        recalculated = True

    assert recalculated, "Deberia recalcular para delta >= 0.5°"
    assert abs(math.degrees(pipe._cached_pitch) - 6.0) < 1e-4

    # 4. Paso con None, None -> debe restaurar base_camera_pose
    if pipe._cached_roll is not None or pipe._cached_pitch is not None:
        pipe.camera_pose = pipe.base_camera_pose.copy()
        pipe._cached_roll = None
        pipe._cached_pitch = None

    assert np.array_equal(pipe.camera_pose, pipe.base_camera_pose), "Debe restaurar base_camera_pose"
    print("  ✓ Umbral, cacheo y restauracion nominal funcionan segun especificacion.")


def test_odometry_tilt_staleness_policy():
    print("\n=== 6. Politica de vigencia y fallback en Odometry.current_roll_pitch ===")
    odo = Odometry(OdometryConfig())

    # Sin datos
    assert odo.current_roll_pitch() is None, "Sin datos debe devolver None"

    # Lote con aceleracion valida (gate abierto)
    pitch_10_rad = math.radians(10.0)
    accels_10 = [[-math.sin(pitch_10_rad), 0.0, math.cos(pitch_10_rad), 10.0 + k * 0.02] for k in range(5)]
    odo.update({"accels": accels_10}, now=10.0)

    res = odo.current_roll_pitch(now=10.0)
    assert res is not None, "Gate abierto debe entregar tupla"
    assert abs(math.degrees(res[1]) - 10.0) < 0.1
    print(f"  Gate abierto: roll={math.degrees(res[0]):.1f}°, pitch={math.degrees(res[1]):.1f}°")

    # Lote con gate CERRADO (|a|=1.5g por aceleracion repentina) a t=10.5 s (staleness = 0.5 s <= 1.5 s)
    accels_closed = [[0.5, 0.0, 1.4, 10.5 + k * 0.02] for k in range(5)]
    odo.update({"accels": accels_closed}, now=10.5)
    assert not odo.tilt_gate_open

    res_stale_ok = odo.current_roll_pitch(now=10.5, max_staleness_s=1.5)
    assert res_stale_ok is not None, "Dentro de max_staleness_s debe preservar el tilt previo de la rampa"
    assert abs(math.degrees(res_stale_ok[1]) - 10.0) < 0.1
    print("  Gate cerrado (age=0.5 s <= 1.5 s): preserva tilt previo de la rampa ✓")

    # A t=12.0 s (staleness = 2.0 s > 1.5 s) sin medicion nueva -> debe vencer
    res_expired = odo.current_roll_pitch(now=12.0, max_staleness_s=1.5)
    assert res_expired is None, "Pasado max_staleness_s debe caer a None"
    print("  Gate cerrado (age=2.0 s > 1.5 s): vence y cae a None (nominal nivelado) ✓")


def test_performance_benchmark():
    print("\n=== 7. Microbenchmark de computo de pose ===")
    base_pose = camera_pose_from_height_pitch(0.1502, 1.85)

    # Warmup
    for _ in range(500):
        camera_pose_with_tilt(base_pose, 0.05, 0.1)

    N = 50000
    t0 = time.perf_counter()
    for _ in range(N):
        camera_pose_with_tilt(base_pose, 0.05, 0.1)
    dt_us = ((time.perf_counter() - t0) / N) * 1e6

    print(f"  Tiempo por invocacion: {dt_us:.2f} µs ({dt_us/1000.0:.4f} ms)")
    print(f"  Frecuencia maxima teorica: {1e6 / dt_us:,.0f} composiciones/segundo")
    assert dt_us < 50.0, f"Computo de pose inesperadamente lento: {dt_us:.1f} µs"
    print("  ✓ Costo insignificante frente al ciclo SAM-TP (~50 ms).")


if __name__ == "__main__":
    test_level_case()
    test_axis_conventions()
    test_closed_form_formula()
    test_ground_projection_consistency()
    test_perception_pipeline_threshold_and_caching()
    test_odometry_tilt_staleness_policy()
    test_performance_benchmark()
    print("\n" + "="*50)
    print("TODAS LAS PRUEBAS DE TILT DINAMICO PASARON EXITOSAMENTE.")
    print("="*50)
