#!/usr/bin/env python3
"""
test_diagnose_yaw_sign.py — Test unitario para la herramienta de diagnóstico de signo de yaw (Brief 23 / W.1)
"""

import math
import pytest
from er_navigation.testing.diagnose_yaw_sign import angle_diff_deg


def test_angle_diff_deg():
    # Giros normales
    assert math.isclose(angle_diff_deg(90.0, 0.0), 90.0)
    assert math.isclose(angle_diff_deg(0.0, 90.0), -90.0)
    
    # Cruces por 0° / 360° (Norte)
    # De 350° a 10° es un giro horario (+20°)
    assert math.isclose(angle_diff_deg(10.0, 350.0), 20.0)
    # De 10° a 350° es un giro antihorario (-20°)
    assert math.isclose(angle_diff_deg(350.0, 10.0), -20.0)
    
    # Cruce por 180° (Sur)
    assert math.isclose(angle_diff_deg(179.0, -179.0), -2.0)
    assert math.isclose(angle_diff_deg(-179.0, 179.0), 2.0)


def test_sign_correlation_logic():
    # Caso 1: Consistente (ambos aumentan tras giro horario a la derecha)
    d_compass = +25.0
    d_prop = +23.5
    sign_compass = 1 if d_compass > 0 else -1
    sign_prop = 1 if d_prop > 0 else -1
    assert sign_compass == sign_prop

    # Caso 2: Invertido (compás aumenta pero propagación disminuye)
    d_compass = +25.0
    d_prop_inv = -23.5
    sign_compass = 1 if d_compass > 0 else -1
    sign_prop_inv = 1 if d_prop_inv > 0 else -1
    assert sign_compass != sign_prop_inv
