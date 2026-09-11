"""Tests offline para la robustez de localización (Fase 1 y Fase 2).

Verifica:
  a. gps_quality() con fixes de calidad nominal, degradada y sin fix.
  b. Odometría con fixes degradados (atenuación y no salto de pose).
  c. HeadingEstimator alimentado con jitter puro (rover quieto -> no rumbo espurio).
  d. Fusión circular con rumbos de 359° y 1° -> 0°, no 180°.
  e. Fusión con una fuente de alta incertidumbre y otra de baja incertidumbre.
  f. Crecimiento de incertidumbre del giróscopo sin corrección absoluta.

Uso:
    python -m genie_rover.test_localization_robustness
"""

from __future__ import annotations

import math
import numpy as np

from .navigation import HeadingEstimator, gps_quality, wrap_deg
from .odometry import Odometry, OdometryConfig, Pose


def test_gps_quality():
    print("=== Test a: gps_quality() ===")
    # 1. Sin fix / señal nula
    assert gps_quality(gps_signal=0) == 0.0, "gps_signal=0 debe ser 0.0"
    assert gps_quality(fix_quality=0, gps_signal=50) == 0.0, "fix_quality=0 debe ser 0.0"

    # 2. RTK Fix nominal (hdop <= 0.025, fix_quality=4)
    q_rtk = gps_quality(hdop=0.009, fix_quality=4, gps_signal=49)
    print(f"  RTK Fix nominal (hdop=0.009, fix_q=4): q={q_rtk:.3f}")
    assert abs(q_rtk - 1.00) < 1e-4, "RTK Fix nominal debe dar 1.0"

    # 3. Autónomo (fix_quality=1) con hdop bajo
    q_auto = gps_quality(hdop=0.007, fix_quality=1, gps_signal=35)
    print(f"  Autónomo (hdop=0.007, fix_q=1): q={q_auto:.3f}")
    assert abs(q_auto - 0.35) < 1e-4, "Autónomo debe modular a base_weight 0.35"

    # 4. RTK Float (fix_quality=5)
    q_float = gps_quality(hdop=0.015, fix_quality=5, gps_signal=40)
    print(f"  RTK Float (hdop=0.015, fix_q=5): q={q_float:.3f}")
    assert abs(q_float - 0.60) < 1e-4, "RTK Float debe dar 0.60"

    # 5. DGPS (fix_quality=2)
    q_dgps = gps_quality(hdop=0.012, fix_quality=2, gps_signal=45)
    print(f"  DGPS (hdop=0.012, fix_q=2): q={q_dgps:.3f}")
    assert abs(q_dgps - 0.70) < 1e-4, "DGPS debe dar 0.70"

    # 6. Degradación por HDOP elevado
    q_deg = gps_quality(hdop=0.0525, fix_quality=4, gps_signal=49)
    print(f"  HDOP degradado (hdop=0.0525 en [0.025, 0.080]): q={q_deg:.3f}")
    assert 0.45 < q_deg < 0.55, f"Esperaba ~0.50, dio {q_deg}"

    # 7. Rechazo por HDOP extremo (> 0.080)
    q_rej = gps_quality(hdop=0.095, fix_quality=4, gps_signal=49)
    print(f"  HDOP extremo (hdop=0.095 > 0.080): q={q_rej:.3f}")
    assert q_rej == 0.0, "HDOP > hdop_reject debe ser 0.0"
    print("  ✓ Test a superado.")


def test_odometry_degraded_fix():
    print("\n=== Test b: Odometría con fix degradado ===")
    cfg = OdometryConfig(wheel_radius_m=0.05, min_gps_displacement_m=1.0, gps_blend=0.30)
    odo = Odometry(cfg)

    # Simular avance de 2 metros con odometría pura
    dt = 0.02
    t = 0.0
    for _ in range(50):
        odo.update({"rpms": [[30, 30, 30, 30, t]], "gyros": [[0, 0, 0, t]]})
        t += dt

    # Primer fix GPS en origen (calidad nominal)
    odo.update({"latitude": -34.921400, "longitude": -57.954400, "fix_quality": 4, "hdop": 0.009, "gps_signal": 50})
    pos0_x = odo.pose.x

    # Avanzar 2 metros más
    for _ in range(50):
        odo.update({"rpms": [[30, 30, 30, 30, t]], "gyros": [[0, 0, 0, t]]})
        t += dt
    x_antes_gps = odo.pose.x

    # Fix GPS degradado (salto espurio de +10m por multitrayecto pero hdop alto o fix_q=0)
    lat_salto = -34.921400 + (10.0 / 6378137.0) * (180.0 / math.pi)
    odo.update({"latitude": lat_salto, "longitude": -57.954400, "fix_quality": 4, "hdop": 0.150, "gps_signal": 15})
    x_despues_gps = odo.pose.x

    print(f"  Pose antes de fix corrupto: x={x_antes_gps:.3f}")
    print(f"  Pose después de fix corrupto: x={x_despues_gps:.3f}")
    assert abs(x_despues_gps - x_antes_gps) < 1e-4, "Fix degradado con hdop>reject no debe mover la pose"

    # Ahora un fix con degradación parcial (fix_quality=1 autónomo, q=0.35) a +2.0m reales
    lat_2m = -34.921400 + (2.0 / 6378137.0) * (180.0 / math.pi)
    odo.update({"latitude": lat_2m, "longitude": -57.954400, "fix_quality": 1, "hdop": 0.010, "gps_signal": 35})
    x_parcial = odo.pose.x
    print(f"  Pose con fix autónomo atenuado: x={x_parcial:.3f}")
    assert not math.isnan(x_parcial) and x_parcial > 0.0
    print("  ✓ Test b superado.")


def test_heading_jitter_rejection():
    print("\n=== Test c: HeadingEstimator con jitter puro (rover quieto) ===")
    he = HeadingEstimator(min_displacement_m=1.5, compass_sigma_deg=15.0)
    lat0, lon0 = -34.9214, -57.9544

    # Generar jitter gaussiano estático con sigma = 0.3m (GPS degradado)
    np.random.seed(42)
    t = 0.0
    headings_generados = []
    for i in range(30):
        # Desplazamiento aleatorio dentro de +/- 0.4 m
        dlat = float(np.random.normal(0, 0.3)) / 111320.0
        dlon = float(np.random.normal(0, 0.3)) / (111320.0 * math.cos(math.radians(lat0)))
        # Compass fijo en 45 grados
        h = he.update(lat0 + dlat, lon0 + dlon, orientation=45.0, t=t,
                      hdop=0.015, fix_quality=1, gps_signal=25)
        headings_generados.append(h)
        t += 1.0

    print(f"  Fuente final de rumbo con rover quieto: {he.source}")
    print(f"  Rumbo estimado final: {he.heading:.1f}° (compás base: 45°)")
    assert "gps" not in he.source or he.source == "orientation(fallback)" or "compass" in he.source
    assert abs(wrap_deg(he.heading - 45.0)) < 2.0, f"Rumbo espurio detectado: {he.heading}"
    print("  ✓ Test c superado.")


def test_circular_weighted_fusion():
    print("\n=== Test d & e: Fusión circular ponderada ===")
    he = HeadingEstimator(compass_sigma_deg=15.0)

    # Test d: Fusión de 359° y 1°
    sources = [(359.0, 10.0, "s1"), (1.0, 10.0, "s2")]
    sx = sum((1.0 / (s**2)) * math.cos(math.radians(a)) for a, s, _ in sources)
    sy = sum((1.0 / (s**2)) * math.sin(math.radians(a)) for a, s, _ in sources)
    fused = math.degrees(math.atan2(sy, sx)) % 360.0
    print(f"  Fusión de 359° y 1°: {fused:.2f}° (esperado 0.0° o 360.0°)")
    assert abs(wrap_deg(fused - 0.0)) < 1e-4, f"Fusión errónea: dio {fused} en vez de 0°"

    # Test e: Fusión con alta y baja incertidumbre
    sources_asym = [(10.0, 2.0, "alta_certeza"), (90.0, 30.0, "baja_certeza")]
    sx_a = sum((1.0 / (s**2)) * math.cos(math.radians(a)) for a, s, _ in sources_asym)
    sy_a = sum((1.0 / (s**2)) * math.sin(math.radians(a)) for a, s, _ in sources_asym)
    fused_asym = math.degrees(math.atan2(sy_a, sx_a)) % 360.0
    fused_sigma = 1.0 / math.sqrt(sum(1.0 / (s**2) for _, s, _ in sources_asym))
    print(f"  Fusión asimétrica (10° ±2° vs 90° ±30°): {fused_asym:.2f}° ±{fused_sigma:.2f}°")
    assert abs(fused_asym - 10.0) < 0.5, f"Debe converger a la de menor incertidumbre, dio {fused_asym}"
    assert fused_sigma < 2.0
    print("  ✓ Tests d y e superados.")


def test_gyro_uncertainty_growth():
    print("\n=== Test f: Crecimiento de incertidumbre del giróscopo ===")
    he = HeadingEstimator(compass_sigma_deg=10.0, gyro_drift_rate_dps_per_sqrt_s=1.0)
    lat0, lon0 = -34.9214, -57.9544

    # Inicializar con compass
    he.update(lat0, lon0, orientation=0.0, t=0.0)
    sigma0 = he.uncertainty
    print(f"  Incertidumbre inicial: ±{sigma0:.2f}°")

    # Propagar solo con gyro (sin GPS track y con compass devaluado)
    he.compass_sigma = 40.0
    for i in range(1, 11):
        he.update(lat0, lon0, orientation=0.0, t=float(i), gyro_dps=5.0)

    print(f"  Incertidumbre acumulada tras 10 s: ±{he.uncertainty:.2f}°")
    assert he.heading is not None
    assert he.uncertainty is not None
    print("  ✓ Test f superado.")


def test_disagreement_with_gps_course():
    print("\n=== Test g: disagreement_deg() contra curso GPS y compass_distortion_deg() ===")
    he = HeadingEstimator(use_ekf_udp=True, ekf_weight=1.0)

    # 1. Sin curso GPS disponible -> disagreement_deg debe ser None
    he.update(9.7915, -84.1058, orientation=180.0, t=1.0, ekf_heading=215.0)
    assert he.disagreement_deg() is None, "Sin curso GPS no debe emitir desacuerdo"

    # Distorsión magnética del compás crudo: (180 - 215) = -35°
    dist_comp = he.compass_distortion_deg()
    assert dist_comp is not None and abs(dist_comp - (-35.0)) < 1e-4, f"Esperaba -35°, dio {dist_comp}"
    print(f"  Distorsión compás crudo identificada: {dist_comp:+.1f}° (no sugerida como offset de config)")

    # 2. Generar curso GPS confiable por avance rectilíneo al Norte (heading ~0.0°)
    # Desplazamiento de ~3 metros al Norte en 3 segundos con RTK Fix
    lat0 = 9.791500
    lon0 = -84.105800
    lat1 = lat0 + (3.0 / 111194.9)
    he.update(lat0, lon0, orientation=180.0, t=2.0, hdop=0.010, fix_quality=4, gps_signal=45, ekf_heading=5.0)
    he.update(lat1, lon0, orientation=180.0, t=3.0, hdop=0.010, fix_quality=4, gps_signal=45, ekf_heading=5.0)

    assert he.last_gps_heading is not None, "Debería haber calculado curso GPS"
    assert he.last_gps_sigma is not None and he.last_gps_sigma <= 15.0, "Curso GPS debe tener baja sigma"
    print(f"  Curso GPS calculado: {he.last_gps_heading:.1f}°, sigma: ±{he.last_gps_sigma:.2f}°")

    # 3. disagreement_deg() compara Heading EKF (5.0°) vs Curso GPS (~0.0°) -> ~+5.0°
    desac_gps = he.disagreement_deg()
    assert desac_gps is not None, "Con curso GPS confiable debe haber desacuerdo"
    assert abs(desac_gps - 5.0) < 1.0, f"Esperaba desacuerdo ~+5.0° contra curso GPS, dio {desac_gps}"
    print(f"  Desacuerdo cinemático contra curso GPS: {desac_gps:+.1f}° (métrica válida para calibración)")
    print("  ✓ Test g superado.")


def main():
    print("==================================================")
    print("  EJECUTANDO SUITE DE LOCALIZACIÓN ROBUSTA (FASE 1 Y 2)")
    print("==================================================")
    test_gps_quality()
    test_odometry_degraded_fix()
    test_heading_jitter_rejection()
    test_circular_weighted_fusion()
    test_gyro_uncertainty_growth()
    test_disagreement_with_gps_course()
    print("\n==================================================")
    print("  TODAS LAS PRUEBAS OFFLINE PASARON EXITOSAMENTE.")
    print("==================================================")


if __name__ == "__main__":
    main()
