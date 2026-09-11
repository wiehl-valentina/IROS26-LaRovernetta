#!/usr/bin/env python3
"""test_mag_gating.py — Validación de Detección de Saturación Magnética y Gating Ponderado por Tilt (Fase 1).

Verifica:
1. Campo nominal a nivel (Test A, norma ~3331 counts): confianza 1.0, covarianza 0.025 rad^2.
2. Campo saturado en sitio con metal (norma ~12222 counts, std=0): confianza 0.0, covarianza 1e6 rad^2.
3. Calibración de mag_norm_reference: bloqueada en pendiente (|tilt| > 6°), activada solo a nivel (|tilt| < 6°).
4. Ponderación en pendiente: en rampa de 15° con rotación física real, se reduce el peso de la norma (w_norm)
   y se mantiene la confianza alta (>0.8).
5. Detección de congelamiento (std=0 con giro de giróscopo): veto dinámico inmediato (s_dynamic=0).
"""
import math
import pytest
import rclpy
from earth_rovers_sdk.bridge_node import EarthRoverBridge


@pytest.fixture
def ros_context():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_mag_nominal_field_trusted(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge._mag_calibrated = True
    bridge._filtered_roll = 0.0
    bridge._filtered_pitch = 0.0

    # Lectura nominal de Test A: norma ~3331.3 counts
    # mx=-95, my=126, mz=3308 -> sqrt(95^2 + 126^2 + 3308^2) = 3331.7 counts
    conf, cov_yaw, diag = bridge._evaluate_magnetic_gate(
        mx=-95.0, my=126.0, mz=3308.0, omega_z=0.0, accel_gate_open=True
    )

    assert conf > 0.95, f"Confianza nominal esperada ~1.0, obtenido {conf}"
    assert math.isclose(cov_yaw, 0.025, abs_tol=0.01), f"Covarianza esperada ~0.025, obtenido {cov_yaw}"
    assert diag["trusted"] is True
    assert diag["score_norm"] == 1.0

    bridge.destroy_node()


def test_mag_saturated_field_rejected(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge._mag_calibrated = True
    bridge._filtered_roll = 0.0
    bridge._filtered_pitch = 0.0

    # Lectura saturada en sitio con metal: norma ~12221.7 counts (std=0)
    for _ in range(10):
        conf, cov_yaw, diag = bridge._evaluate_magnetic_gate(
            mx=-128.0, my=-6270.0, mz=10490.0, omega_z=0.0, accel_gate_open=True
        )

    assert conf < 0.05, f"Confianza saturada esperada ~0.0, obtenido {conf}"
    assert cov_yaw >= 9.0e5, f"Covarianza saturada debe ser >= 9e5 rad^2, obtenido {cov_yaw}"
    assert diag["trusted"] is False
    assert diag["score_norm"] == 0.0

    bridge.destroy_node()


def test_mag_calibration_gated_by_tilt(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge._mag_calibrated = False
    bridge._mag_calib_buffer = []
    bridge._mag_norm_reference = 3330.0

    # 1. Rover con inclinación significativa (roll=15.0°, pitch=5.0° > 6.0°)
    bridge._filtered_roll = math.radians(15.0)
    bridge._filtered_pitch = math.radians(5.0)

    for _ in range(15):
        bridge._evaluate_magnetic_gate(
            mx=0.0, my=0.0, mz=4000.0, omega_z=0.0, accel_gate_open=True
        )

    # NO debe haberse calibrado la referencia en pendiente
    assert bridge._mag_calibrated is False, "No debe calibrar con tilt significativo"
    assert bridge._mag_norm_reference == 3330.0

    # 2. Rover pasa a nivel (|roll| < 6°, |pitch| < 6°)
    bridge._filtered_roll = math.radians(1.0)
    bridge._filtered_pitch = math.radians(0.5)

    for _ in range(12):
        bridge._evaluate_magnetic_gate(
            mx=0.0, my=0.0, mz=3400.0, omega_z=0.0, accel_gate_open=True
        )

    # Ahora sí debe haberse calibrado
    assert bridge._mag_calibrated is True, "Debe calibrar una vez nivelado"
    assert math.isclose(bridge._mag_norm_reference, 3400.0, abs_tol=10.0)

    bridge.destroy_node()


def test_mag_slope_operation_weighting(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge._mag_calibrated = True
    bridge._mag_norm_reference = 3330.0

    # Rover en rampa de 15°, donde el sesgo hard-iron desplaza la norma a 4150 counts (+820)
    bridge._filtered_roll = math.radians(15.0)
    bridge._filtered_pitch = math.radians(2.0)

    # Rover girando físicamente a 10 deg/s en la rampa (las componentes varían normalmente)
    for i in range(15):
        angle = math.radians(i * 10.0)
        mx = 2000.0 * math.cos(angle)
        my = 2000.0 * math.sin(angle)
        mz = 3600.0
        conf, cov_yaw, diag = bridge._evaluate_magnetic_gate(
            mx=mx, my=my, mz=mz, omega_z=math.radians(10.0), accel_gate_open=True
        )

    # La ponderación w_norm debe haber bajado significativamente (de 1.0 hacia ~0.36)
    assert diag["tilt_weight"] < 0.5, f"tilt_weight debe ser < 0.5 en pendiente, obtenido {diag['tilt_weight']}"
    # El sensor no está clavado, por lo que la confianza global se preserva alta
    assert conf > 0.80, f"Confianza en pendiente con giro real debe ser > 0.80, obtenido {conf}"
    assert diag["trusted"] is True

    bridge.destroy_node()


def test_mag_stuck_detector_during_rotation(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge._mag_calibrated = True

    # Rover girando rápido a 20 deg/s (omega_z = 0.35 rad/s)
    # pero el magnetómetro está congelado (valores idénticos cuadro a cuadro)
    for _ in range(12):
        conf, cov_yaw, diag = bridge._evaluate_magnetic_gate(
            mx=2000.0, my=1000.0, mz=2400.0, omega_z=math.radians(20.0), accel_gate_open=True
        )

    assert diag["score_dynamic"] == 0.0, "Debe detectar sensor congelado durante rotación (score_dynamic=0)"
    assert conf < 0.10, "Confianza debe ser prácticamente 0 ante sensor congelado"
    assert diag["trusted"] is False
    assert cov_yaw > 8.0e5, "Covarianza debe estar fuertemente inflada"

    bridge.destroy_node()
