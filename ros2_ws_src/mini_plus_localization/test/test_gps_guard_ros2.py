#!/usr/bin/env python3
"""test_gps_guard_ros2.py — Validación de Guarda de GPS en EarthRoverBridge (Fase 1: ROS 2).

Verifica:
1. Deduplicación por timestamp y frecuencia 1 Hz (Paso 1.1).
2. Salto aislado descartado sin alimentar /gps/fix (Paso 1.2).
3. Modo degradado ante fixes malos sostenidos (>=3 seguidos) bloqueando /gps/fix (Paso 1.4).
4. Recuperación limpia a Nivel 1 tras fix válido.
"""

from unittest.mock import MagicMock
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


def test_ros2_gps_deduplication(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge.gps_pub.publish = MagicMock()

    # 1. Primer fix nominal
    data0 = {
        "latitude": 9.791693,
        "longitude": -84.105941,
        "timestamp": 100.0,
        "gps_timestamp": 100.0,
        "fix_quality": 4,
        "gps_signal": 45.0,
        "hdop": 0.012,
    }
    bridge._publish_telemetry(data0)
    assert bridge.gps_pub.publish.call_count == 1

    # 2. Mismo fix por polling a 10 Hz (mismo timestamp)
    bridge._publish_telemetry(data0)
    assert bridge.gps_pub.publish.call_count == 1, "Fix con mismo timestamp debe ser descartado como duplicado"

    # 3. Telemetría a 200ms (<0.8s) con mismas coordenadas
    data_fast = dict(data0, timestamp=100.2, gps_timestamp=100.2)
    bridge._publish_telemetry(data_fast)
    assert bridge.gps_pub.publish.call_count == 1, "Fix con coordenadas idénticas y dt < 0.8s debe ser descartado"

    # 4. Nuevo fix a 1.0s (>0.8s) con coordenadas avanzadas
    data_new = dict(data0, latitude=9.791698, timestamp=101.0, gps_timestamp=101.0)
    bridge._publish_telemetry(data_new)
    assert bridge.gps_pub.publish.call_count == 2, "Fix nuevo a 1.0s debe ser publicado"

    bridge.destroy_node()


def test_ros2_gps_isolated_jump_rejected(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge.gps_pub.publish = MagicMock()

    # 1. Primer fix nominal (origen)
    data0 = {
        "latitude": 9.791693,
        "longitude": -84.105941,
        "timestamp": 100.0,
        "gps_timestamp": 100.0,
        "fix_quality": 4,
        "gps_signal": 45.0,
        "hdop": 0.012,
    }
    bridge._publish_telemetry(data0)
    assert bridge.gps_pub.publish.call_count == 1
    assert bridge._consecutive_bad_gps == 0

    # 2. Salto espurio de +15m en 1 segundo (d_max permitido ~ 1.11*1 + 1.5 = 2.61m)
    # 15m al norte en latitud es ~ 15 / 111111 = 0.000135 grados
    data_jump = dict(data0, latitude=9.791693 + 0.000135, timestamp=101.0, gps_timestamp=101.0)
    bridge._publish_telemetry(data_jump)

    assert bridge.gps_pub.publish.call_count == 1, "Salto GPS no debe publicarse en /gps/fix"
    assert bridge._consecutive_bad_gps == 1
    assert bridge._gps_guard_level == 1, "Un salto aislado mantiene Nivel 1"

    # 3. Fix válido subsiguiente a 0.5m del origen (dt=2.0s respecto a data0)
    data_valid = dict(data0, latitude=9.791693 + 0.0000045, timestamp=102.0, gps_timestamp=102.0)
    bridge._publish_telemetry(data_valid)

    assert bridge.gps_pub.publish.call_count == 2, "Fix válido subsiguiente debe publicarse"
    assert bridge._consecutive_bad_gps == 0

    bridge.destroy_node()


def test_ros2_gps_sustained_bad_fixes_degraded_mode(ros_context):
    bridge = EarthRoverBridge()
    bridge._running = False
    bridge.gps_pub.publish = MagicMock()

    # 1. Primer fix nominal
    data0 = {
        "latitude": 9.791693,
        "longitude": -84.105941,
        "timestamp": 100.0,
        "gps_timestamp": 100.0,
        "fix_quality": 4,
        "gps_signal": 45.0,
        "hdop": 0.012,
    }
    bridge._publish_telemetry(data0)
    assert bridge.gps_pub.publish.call_count == 1

    # 2. Enviar 3 saltos consecutivos (+15m, +20m, +25m)
    t = 100.0
    for i in range(1, 4):
        t += 1.0
        data_bad = dict(
            data0,
            latitude=9.791693 + (0.000135 * i),
            timestamp=t,
            gps_timestamp=t,
        )
        bridge._publish_telemetry(data_bad)

    assert bridge._consecutive_bad_gps == 3
    assert bridge._gps_guard_level == 2, "Con 3 fixes malos consecutivos debe entrar en Nivel 2 (Modo Degradado)"
    assert bridge.gps_pub.publish.call_count == 1, "Ningún fix malo debe haber sido publicado a /gps/fix"

    # 3. Recuperación tras recibir un fix nominal confiable
    t += 1.0
    data_recovered = dict(
        data0,
        latitude=9.791693 + 0.000008,
        timestamp=t,
        gps_timestamp=t,
    )
    bridge._publish_telemetry(data_recovered)
    assert bridge._gps_guard_level == 1
    assert bridge._consecutive_bad_gps == 0
    assert bridge.gps_pub.publish.call_count == 2, "Fix recuperado debe ser publicado"

    bridge.destroy_node()
