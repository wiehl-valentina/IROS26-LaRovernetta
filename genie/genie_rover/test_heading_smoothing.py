"""Tests para el filtrado exponencial circular de rumbo EKF (Fase 1 - FIX 3).

Verifica:
  (a) Cruce por 0° (ej. 359° -> 1° o viceversa) no produce valores espurios cercanos a 180°.
  (b) Respuesta ante escalón de 90° alcanza ~63.2% tras dt = tau (propiedad de filtro exponencial de 1er orden).
  (c) tau <= 0 reproduce el valor crudo instantáneo (comportamiento sin filtro).
  (d) Reset de filtro cuando dt > 2.0 s (reconexión / telemetría discontinua).
"""

from __future__ import annotations

import math
import pytest

from .navigation import HeadingEstimator, wrap_deg


def test_circular_wrap_around():
    """(a) cruce 359°->1° no produce valor cercano a 180°."""
    tau = 0.6
    he = HeadingEstimator(use_ekf_udp=True, ekf_weight=1.0, ekf_smooth_tau_s=tau)

    # 1. Muestra inicial en 359.0° a t=0.0
    h0 = he.update(lat=0.0, lon=0.0, orientation=0.0, t=0.0, ekf_heading=359.0)
    assert h0 is not None
    assert math.isclose(h0, 359.0, abs_tol=1e-5)

    # 2. Paso a 1.0° con dt = 0.1 s
    # La interpolación circular más corta va de 359° a 1° a través de 0° (delta = +2°).
    # Un promedio aritmético erróneo daría (359 + 1)/2 = 180°.
    h1 = he.update(lat=0.0, lon=0.0, orientation=0.0, t=0.1, ekf_heading=1.0)
    assert h1 is not None

    # El rumbo resultante debe estar muy cerca de 0° / 360° y JAMÁS cerca de 180°
    assert not (100.0 < h1 < 260.0), f"Filtro promedió incorrectamente en grados lineales: dio {h1}°"
    # delta = wrap_deg(1 - 359) = +2.0°
    # alpha = 1 - exp(-0.1 / 0.6) = 1 - exp(-0.1667) ≈ 0.1535
    # h1 esperado = (359 + 0.1535 * 2) % 360 = 359.307°
    expected = (359.0 + (1.0 - math.exp(-0.1 / tau)) * 2.0) % 360.0
    assert math.isclose(h1, expected, abs_tol=0.05)

    # 3. Paso inverso: 1.0° a 359.0°
    he2 = HeadingEstimator(use_ekf_udp=True, ekf_weight=1.0, ekf_smooth_tau_s=tau)
    he2.update(lat=0.0, lon=0.0, orientation=0.0, t=0.0, ekf_heading=1.0)
    h_rev = he2.update(lat=0.0, lon=0.0, orientation=0.0, t=0.1, ekf_heading=359.0)
    assert h_rev is not None
    assert not (100.0 < h_rev < 260.0), f"Filtro inverso promedió incorrectamente: dio {h_rev}°"
    expected_rev = (1.0 + (1.0 - math.exp(-0.1 / tau)) * (-2.0)) % 360.0
    assert math.isclose(h_rev, expected_rev, abs_tol=0.05)


def test_step_response_tau():
    """(b) un escalón de 90° se alcanza al ~63% tras dt=tau."""
    tau = 0.6
    he = HeadingEstimator(use_ekf_udp=True, ekf_weight=1.0, ekf_smooth_tau_s=tau)

    # 1. Establecer rumbo inicial en 0.0° a t=0.0
    h0 = he.update(lat=0.0, lon=0.0, orientation=0.0, t=0.0, ekf_heading=0.0)
    assert h0 is not None and math.isclose(h0, 0.0, abs_tol=1e-5)

    # 2. Escalón a 90.0° evaluado exactamente tras dt = tau = 0.6 s
    # alpha = 1 - exp(-0.6 / 0.6) = 1 - exp(-1) ≈ 0.6321205588
    # h(tau) = 0 + alpha * (90 - 0) = 56.89085° (63.21% de 90°)
    h_tau = he.update(lat=0.0, lon=0.0, orientation=0.0, t=tau, ekf_heading=90.0)
    assert h_tau is not None

    fraction = h_tau / 90.0
    expected_fraction = 1.0 - math.exp(-1.0)
    assert math.isclose(fraction, expected_fraction, abs_tol=1e-3), (
        f"Esperaba ~63.2% de respuesta tras dt=tau, dio {fraction * 100:.2f}% ({h_tau:.2f}°)"
    )


def test_tau_zero_raw_behavior():
    """(c) tau=0 reproduce el valor crudo."""
    he = HeadingEstimator(use_ekf_udp=True, ekf_weight=1.0, ekf_smooth_tau_s=0.0)

    # Primera muestra a t=0.0
    h0 = he.update(lat=0.0, lon=0.0, orientation=0.0, t=0.0, ekf_heading=10.0)
    assert h0 is not None and math.isclose(h0, 10.0, abs_tol=1e-5)

    # Muestra siguiente tras dt=0.05 s con tau=0: debe tomar inmediatamente 90.0° sin retardo
    h1 = he.update(lat=0.0, lon=0.0, orientation=0.0, t=0.05, ekf_heading=90.0)
    assert h1 is not None
    assert math.isclose(h1, 90.0, abs_tol=1e-5), f"Con tau=0 debería ser exactamente 90.0°, dio {h1}"


def test_dt_large_resets_raw():
    """(d) Si dt > 2.0 s, toma el valor crudo sin filtrar."""
    tau = 0.6
    he = HeadingEstimator(use_ekf_udp=True, ekf_weight=1.0, ekf_smooth_tau_s=tau)

    he.update(lat=0.0, lon=0.0, orientation=0.0, t=0.0, ekf_heading=0.0)
    # Hueco de 3.0 s en telemetría
    h_jump = he.update(lat=0.0, lon=0.0, orientation=0.0, t=3.0, ekf_heading=120.0)
    assert h_jump is not None
    assert math.isclose(h_jump, 120.0, abs_tol=1e-5), f"dt > 2.0s debe tomar valor crudo, dio {h_jump}"
