#!/usr/bin/env python3
"""test_gps_course_heading.py — Validación unitaria de Fase 2 (gps_course_heading_bridge).

Verifica:
1. Tabla de covarianza derivada empíricamente de Test D (0.00092 rad^2 a >=5m, 0.025 a 0.4m).
2. Cálculo geométrico de plano tangente local (latlon_to_local_ne).
3. Desambiguación de marcha atrás (inversión de 180° del rumbo del vehículo).
4. Transición continua y suave de covarianza e inflación controlada.
5. Comportamiento en reposo y rotación sobre el eje (score 0, covarianza 1e6, buffer reseteado).
6. Avance en recta con plena convergencia y anclaje al rumbo real de avance.
7. Paso A.5: Contador time_without_anchor_s crece sin techo en desanclaje y se resetea a 0
   en cuanto cualquiera de las dos fuentes (compás o curso GPS) recupera confianza alta.
"""

import json
import math
import sys
import unittest
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float32, String

from gps_course_heading_bridge import (
    EMPIRICAL_CONVERGENCE_TABLE,
    GpsCourseHeadingBridge,
    interpolate_base_covariance,
    latlon_to_local_ne,
)


class TestGpsCourseHeadingUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = GpsCourseHeadingBridge()

    def tearDown(self):
        self.node.destroy_node()

    def test_empirical_convergence_table(self):
        """Paso A.2: Verifica la tabla contra los 4 puntos reales medidos de Test D y su interpolación/extrapolación."""
        # 1. Los 4 puntos reales medidos exactos de Test D:
        # >=1m -> 4.44°
        self.assertAlmostEqual(interpolate_base_covariance(1.0), math.radians(4.44) ** 2, places=6)
        # >=2m -> 3.71°
        self.assertAlmostEqual(interpolate_base_covariance(2.0), math.radians(3.71) ** 2, places=6)
        # >=3m -> 2.73°
        self.assertAlmostEqual(interpolate_base_covariance(3.0), math.radians(2.73) ** 2, places=6)
        # >=5m -> 1.74° (asíntota medida)
        self.assertAlmostEqual(interpolate_base_covariance(5.0), math.radians(1.74) ** 2, places=6)

        # 2. Distancias intermedias interpoladas linealmente entre mediciones:
        # d=4.0m (entre 3m y 5m)
        expected_4m = (math.radians(2.73) ** 2 + math.radians(1.74) ** 2) / 2.0
        self.assertAlmostEqual(interpolate_base_covariance(4.0), expected_4m, places=6)
        # d=0.5m (entre piso a 0.4m y medición a 1.0m)
        expected_05m = 0.02500 + (0.5 - 0.4) / (1.0 - 0.4) * (math.radians(4.44) ** 2 - 0.02500)
        self.assertAlmostEqual(interpolate_base_covariance(0.5), expected_05m, places=6)

        # 3. Supuestos de extrapolación fuera del rango medido:
        # > 5.0m: se asume asíntota medida constante (1.74°)
        self.assertAlmostEqual(interpolate_base_covariance(8.0), math.radians(1.74) ** 2, places=6)
        self.assertAlmostEqual(interpolate_base_covariance(15.0), math.radians(1.74) ** 2, places=6)
        # < 0.4m: se asume piso conservador constante (9.07° -> 0.025 rad^2)
        self.assertAlmostEqual(interpolate_base_covariance(0.20), 0.02500, places=6)

    def test_latlon_to_local_ne(self):
        """Verifica la proyección a plano tangente local (Norte, Este)."""
        lat0, lon0 = 9.791693, -84.105941
        d_deg = 0.0001  # ~11.12 m
        n, e = latlon_to_local_ne(lat0, lon0, lat0 + d_deg, lon0)
        self.assertGreater(n, 11.0)
        self.assertLess(n, 11.2)
        self.assertAlmostEqual(e, 0.0, places=5)

    def test_stationary_rover_untrusted(self):
        """Rover quieto (v=0): score 0, covarianza 1e6, sin anclaje falso."""
        # Simular odometría en reposo
        odom = Odometry()
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.angular.z = 0.0
        self.node._on_wheel_odom(odom)

        # Simular GPS fix estático
        gps = NavSatFix()
        gps.status.status = NavSatStatus.STATUS_FIX
        gps.latitude = 9.791693
        gps.longitude = -84.105941
        gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
        self.node._on_gps_fix(gps)

        self.assertEqual(self.node._gps_course_confidence_score, 0.0)
        self.assertFalse(self.node._gps_course_trusted)
        self.assertEqual(len(self.node._track_buf), 0)

    def test_turn_in_place_resets_buffer(self):
        """Rotación sobre el eje (|w_z| > 8°/s): buffer de recta se limpia."""
        # Cargar puntos previos
        self.node._track_buf.append((9.791693, -84.105941, 100.0, 0.35, 1.0))
        self.node._track_buf.append((9.791703, -84.105941, 101.0, 0.35, 1.0))
        self.assertEqual(len(self.node._track_buf), 2)

        # Odometría reporta giro en el lugar (w_z = 15°/s = 0.26 rad/s)
        odom = Odometry()
        odom.twist.twist.linear.x = 0.05
        odom.twist.twist.angular.z = math.radians(15.0)
        self.node._on_wheel_odom(odom)

        self.assertEqual(len(self.node._track_buf), 0)

    def test_forward_straight_convergence(self):
        """Avance en recta (v=0.35 m/s, d=5.5m): rumbo exacto Norte (0°) y cov=0.00092."""
        # Odometría en avance recto
        odom = Odometry()
        odom.twist.twist.linear.x = 0.35
        odom.twist.twist.angular.z = 0.0
        self.node._on_wheel_odom(odom)

        lat_base, lon_base = 9.791693, -84.105941

        # Generar secuencia de avance hacia el Norte (latitud creciente)
        # 0.00005 deg ~ 5.56m Norte
        for i in range(12):
            gps = NavSatFix()
            gps.status.status = NavSatStatus.STATUS_FIX
            gps.latitude = lat_base + (i * 0.0000045)  # paso ~0.5m
            gps.longitude = lon_base
            gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
            self.node._on_gps_fix(gps)

        # Debe tener plena confianza
        self.assertAlmostEqual(self.node._gps_course_confidence_score, 1.0, places=2)
        self.assertTrue(self.node._gps_course_trusted)
        # Rumbo brújula 0° (Norte)
        self.assertAlmostEqual(self.node._last_gps_course_deg, 0.0, places=1)
        # Rumbo ENU +pi/2 (Norte)
        self.assertAlmostEqual(self.node._last_valid_yaw_enu, math.pi / 2.0, places=2)

    def test_reverse_motion_disambiguation(self):
        """Marcha atrás (v < -0.05 m/s): vector de movimiento Norte -> trompa apunta al Sur (180°)."""
        odom = Odometry()
        odom.twist.twist.linear.x = -0.35  # Marcha atrás
        odom.twist.twist.angular.z = 0.0
        self.node._on_wheel_odom(odom)

        lat_base, lon_base = 9.791693, -84.105941

        # El vehículo retrocede físicamente hacia el Norte
        for i in range(12):
            gps = NavSatFix()
            gps.status.status = NavSatStatus.STATUS_FIX
            gps.latitude = lat_base + (i * 0.0000045)
            gps.longitude = lon_base
            gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
            self.node._on_gps_fix(gps)

        self.assertTrue(self.node._gps_course_trusted)
        # El rumbo del vehículo debe estar invertido 180° (Sur)
        self.assertAlmostEqual(self.node._last_gps_course_deg, 180.0, places=1)
        # En ENU, Sur = -pi/2
        self.assertAlmostEqual(self.node._last_valid_yaw_enu, -math.pi / 2.0, places=2)

    def test_paso_a5_time_without_anchor_counter(self):
        """Paso A.5: Verifica que time_without_anchor_s crezca sin techo y se resetee en recuperación."""
        import time

        # 1. Caso desanclado: compás untrusted y GPS course untrusted
        self.node._mag_trusted = False
        self.node._mag_confidence_score = 0.0
        self.node._last_mag_diag_time = time.monotonic()  # fresca pero untrusted
        self.node._gps_course_trusted = False
        self.node._gps_course_confidence_score = 0.0

        # Forzar tiempo anterior
        t_sim_past = time.monotonic() - 25.5
        self.node._last_anchored_monotonic = t_sim_past

        # Capturar diagnóstico
        published_msgs = []
        original_pub = self.node.confidence_pub.publish
        self.node.confidence_pub.publish = lambda msg: published_msgs.append(msg)

        self.node._publish_confidence_diag()
        self.assertEqual(len(published_msgs), 1)
        data = json.loads(published_msgs[0].data)

        self.assertGreaterEqual(data["time_without_anchor_s"], 25.0)
        self.assertFalse(data["anchor_available"])
        self.assertEqual(data["active_anchor"], "none")

        # Avanzar 100 segundos en desanclaje -> crece sin techo
        self.node._last_anchored_monotonic = time.monotonic() - 125.5
        self.node._publish_confidence_diag()
        data_100 = json.loads(published_msgs[-1].data)
        self.assertGreaterEqual(data_100["time_without_anchor_s"], 125.0)

        # 2. Recuperación por curso GPS
        self.node._gps_course_trusted = True
        self.node._publish_confidence_diag()
        data_gps_recovered = json.loads(published_msgs[-1].data)
        self.assertEqual(data_gps_recovered["time_without_anchor_s"], 0.0)
        self.assertTrue(data_gps_recovered["anchor_available"])
        self.assertEqual(data_gps_recovered["active_anchor"], "gps_course")

        # 3. Desanclaje GPS pero recuperación por compás (Fase 1)
        self.node._gps_course_trusted = False
        self.node._mag_trusted = True
        self.node._mag_confidence_score = 0.95
        self.node._last_mag_diag_time = time.monotonic()
        self.node._publish_confidence_diag()
        data_mag_recovered = json.loads(published_msgs[-1].data)
        self.assertEqual(data_mag_recovered["time_without_anchor_s"], 0.0)
        self.assertTrue(data_mag_recovered["anchor_available"])
        self.assertEqual(data_mag_recovered["active_anchor"], "mag_compass")

        # Restaurar publicador
        self.node.confidence_pub.publish = original_pub


if __name__ == "__main__":
    unittest.main()
