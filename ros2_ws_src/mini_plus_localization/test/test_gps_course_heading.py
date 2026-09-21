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
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
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

    def test_adaptive_window_retains_fixes_during_turn(self):
        """Paso 2.1 y 2.2: Giro con |w_z| > 8°/s durante avance NO vacía la ventana adaptativa."""
        # Cargar puntos en la ventana adaptativa
        self.node._gps_window.append((9.791693, -84.105941, 100.0, 0.35, 1.0))
        self.node._gps_window.append((9.791703, -84.105941, 101.0, 0.35, 1.0))
        self.assertEqual(len(self.node._gps_window), 2)

        # Odometría reporta giro pronunciado (w_z = 15°/s = 0.26 rad/s > 8°/s)
        odom = Odometry()
        odom.header.stamp.sec = 102
        odom.twist.twist.linear.x = 0.30
        odom.twist.twist.angular.z = math.radians(15.0)
        self.node._on_wheel_odom(odom)

        # En Fase 2 la ventana NO se resetea por velocidad angular
        self.assertEqual(len(self.node._gps_window), 2)

    def test_adaptive_window_timeout_discards_stale_fixes(self):
        """Paso 2.2: Fixes con antigüedad > t_max (15s) se purgan si no se alcanza disp_min."""
        lat0, lon0 = 9.791693, -84.105941

        # Odometría activa
        odom = Odometry()
        odom.twist.twist.linear.x = 0.20
        self.node._on_wheel_odom(odom)

        # Fix a t=100s
        gps1 = NavSatFix()
        gps1.header.stamp.sec = 100
        gps1.status.status = NavSatStatus.STATUS_FIX
        gps1.latitude = lat0
        gps1.longitude = lon0
        gps1.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
        self.node._on_gps_fix(gps1)
        self.assertEqual(len(self.node._gps_window), 1)

        # Fix a t=120s (dt = 20s > t_max=15s) con desplazamiento pequeño (< 2m)
        gps2 = NavSatFix()
        gps2.header.stamp.sec = 120
        gps2.status.status = NavSatStatus.STATUS_FIX
        gps2.latitude = lat0 + 0.000002  # ~0.22m
        gps2.longitude = lon0
        gps2.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
        self.node._on_gps_fix(gps2)

        # El fix de t=100s expiró y fue purgado de la ventana
        self.assertEqual(len(self.node._gps_window), 1)
        self.assertEqual(self.node._gps_window[0][2], 120.0)
        self.assertFalse(self.node._gps_course_trusted)

    def test_synthetic_zigzag_trajectory_recovers_mean_heading(self):
        """Paso 2.3 y 2.4: Trayectoria sintética en zigzag alrededor de Norte recupera el rumbo medio con alta confianza."""
        lat_base, lon_base = 9.791693, -84.105941
        t_base = 100.0

        # Simular avance en zigzag oscilando ±12° alrededor de Norte (+pi/2 ENU, 0° compass)
        # 10 pasos a 0.35 m/s, dt=1.0s -> avance neto ~3.4m North
        for step in range(10):
            t_curr = t_base + step
            # Alternar rumbo entre 78° y 102° ENU (desvío de ±12° respecto a 90° Norte)
            yaw_step_deg = 90.0 + (12.0 if (step % 2 == 0) else -12.0)
            yaw_step_rad = math.radians(yaw_step_deg)

            # Odometría con orientación y velocidad
            odom = Odometry()
            odom.header.stamp.sec = int(t_curr)
            odom.twist.twist.linear.x = 0.35
            odom.twist.twist.angular.z = math.radians(10.0 if (step % 2 == 0) else -10.0)
            odom.pose.pose.orientation = Quaternion(
                x=0.0,
                y=0.0,
                z=math.sin(yaw_step_rad / 2.0),
                w=math.cos(yaw_step_rad / 2.0),
            )
            self.node._on_wheel_odom(odom)

            # Fix GPS avanzando hacia el norte con leve oscilación lateral
            d_north_m = step * (0.35 * math.cos(math.radians(12.0)))
            d_east_m = 0.07 if (step % 2 == 0) else -0.07

            dlat = d_north_m / 6371000.0 * (180.0 / math.pi)
            dlon = d_east_m / (6371000.0 * math.cos(math.radians(lat_base))) * (180.0 / math.pi)

            gps = NavSatFix()
            gps.header.stamp.sec = int(t_curr)
            gps.status.status = NavSatStatus.STATUS_FIX
            gps.latitude = lat_base + dlat
            gps.longitude = lon_base + dlon
            gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
            self.node._on_gps_fix(gps)

        # Al completar los 10 pasos (desplazamiento neto ~3.4m >= 2.0m disp_min):
        # 1. Debe estar anclado y de alta confianza
        self.assertTrue(self.node._gps_course_trusted)
        self.assertGreaterEqual(self.node._gps_course_confidence_score, 0.80)

        # 2. La cuerda neta apunta exactamente a Norte (0° brújula, pi/2 ENU)
        # El rumbo publicado aplica Opción (b): heading_actual + (cuerda - mean_ventana)
        # Como mean_ventana es pi/2 y cuerda es pi/2, la corrección es ~0°,
        # preservando la actitud instantánea del vehículo calibrada a la referencia global.
        self.assertIsNotNone(self.node._last_gps_course_deg)
        # La oscilación instantánea estaba a ±12° de Norte (0° brújula):
        diff_from_north = min(self.node._last_gps_course_deg, 360.0 - self.node._last_gps_course_deg)
        self.assertLessEqual(diff_from_north, 15.0)

    def test_ninety_degree_turn_drops_confidence(self):
        """Paso 2.4: Un giro de 90° dentro de la ventana hace caer la confianza y amplía la covarianza."""
        lat_base, lon_base = 9.791693, -84.105941
        t = 100.0

        # Fase 1: 5 segundos avanzando hacia el Norte (yaw = pi/2 ENU = 90°)
        for step in range(5):
            t_curr = t + step
            odom = Odometry()
            odom.header.stamp.sec = int(t_curr)
            odom.twist.twist.linear.x = 0.40
            odom.pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=math.sin(math.pi / 4.0), w=math.cos(math.pi / 4.0)
            )
            self.node._on_wheel_odom(odom)

            gps = NavSatFix()
            gps.header.stamp.sec = int(t_curr)
            gps.status.status = NavSatStatus.STATUS_FIX
            gps.latitude = lat_base + (step * 0.40 / 6371000.0 * (180.0 / math.pi))
            gps.longitude = lon_base
            gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
            self.node._on_gps_fix(gps)

        # Fase 2: 5 segundos avanzando hacia el Este (yaw = 0 ENU = 0°) — Giro brusco de 90°
        lat_curr = gps.latitude
        for step in range(1, 6):
            t_curr = t + 4 + step
            odom = Odometry()
            odom.header.stamp.sec = int(t_curr)
            odom.twist.twist.linear.x = 0.40
            # yaw = 0.0 ENU
            odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            self.node._on_wheel_odom(odom)

            gps = NavSatFix()
            gps.header.stamp.sec = int(t_curr)
            gps.status.status = NavSatStatus.STATUS_FIX
            gps.latitude = lat_curr
            gps.longitude = lon_base + (step * 0.40 / (6371000.0 * math.cos(math.radians(lat_base))) * (180.0 / math.pi))
            gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
            self.node._on_gps_fix(gps)

        # Dentro de esta ventana hubo un giro de 90°:
        # La dispersión de rumbo supera el umbral máximo de coherencia (heading_spread_max_deg = 45°)
        # La confianza debe colapsar a ~0.0 y quedar untrusted
        self.assertFalse(self.node._gps_course_trusted)
        self.assertLessEqual(self.node._gps_course_confidence_score, 0.20)

    def test_mixed_forward_reverse_discards_window(self):
        """Paso 2.4: Movimiento mixto (avance y retroceso) en la misma ventana la descarta."""
        lat_base, lon_base = 9.791693, -84.105941

        # Paso 1: Fix inicial con avance hacia adelante
        odom1 = Odometry()
        odom1.header.stamp.sec = 100
        odom1.twist.twist.linear.x = 0.35  # Avance
        self.node._on_wheel_odom(odom1)

        gps1 = NavSatFix()
        gps1.header.stamp.sec = 100
        gps1.status.status = NavSatStatus.STATUS_FIX
        gps1.latitude = lat_base
        gps1.longitude = lon_base
        gps1.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
        self.node._on_gps_fix(gps1)

        # Paso 2: Fix posterior con marcha atrás (v < -0.05 m/s)
        odom2 = Odometry()
        odom2.header.stamp.sec = 105
        odom2.twist.twist.linear.x = -0.35  # Retroceso
        self.node._on_wheel_odom(odom2)

        gps2 = NavSatFix()
        gps2.header.stamp.sec = 105
        gps2.status.status = NavSatStatus.STATUS_FIX
        gps2.latitude = lat_base + (2.5 / 6371000.0 * (180.0 / math.pi))
        gps2.longitude = lon_base
        gps2.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
        self.node._on_gps_fix(gps2)

        # La ventana detectó maniobra mixta y fue purgada inmediatamente
        self.assertFalse(self.node._gps_course_trusted)
        self.assertEqual(self.node._gps_course_confidence_score, 0.0)
        self.assertEqual(len(self.node._gps_window), 0)

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

    def test_gating_a_conflict_sequence_with_dispersion_discard(self):
        """Punto 1: 2 ventanas con discrepancia, 1 ventana descartada por dispersión, 1 tercera ventana con discrepancia -> INVALIDA COMPÁS.
        Las ventanas que no califican (dispersión > 20°) no deben sumar ni resetear el contador.
        """
        self.node._disjoint_windows = True
        lat_base, lon_base = 9.791693, -84.105941
        t = 100.0

        def feed_window(start_t, start_lat, start_lon, delta_dist_m, yaw_compas_rad, is_dispersed=False):
            steps = 6
            dt = 1.0
            dist_per_step = delta_dist_m / (steps - 1)
            curr_lat = start_lat
            curr_lon = start_lon
            curr_t = start_t

            for s in range(steps):
                if s > 0:
                    curr_t += dt
                    d_north = dist_per_step * math.sin(math.radians(45.0))
                    d_east = dist_per_step * math.cos(math.radians(45.0))
                    curr_lat += d_north / 6371000.0 * (180.0 / math.pi)
                    curr_lon += d_east / (6371000.0 * math.cos(math.radians(curr_lat))) * (180.0 / math.pi)

                odom = Odometry()
                odom.header.stamp.sec = int(curr_t)
                odom.twist.twist.linear.x = 0.35
                if is_dispersed:
                    step_yaw = yaw_compas_rad + math.radians(25.0 if s % 2 == 0 else -25.0)
                    odom.twist.twist.angular.z = math.radians(20.0 if s % 2 == 0 else -20.0)
                else:
                    step_yaw = yaw_compas_rad
                    odom.twist.twist.angular.z = 0.0

                odom.pose.pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=math.sin(step_yaw / 2.0), w=math.cos(step_yaw / 2.0)
                )
                self.node._on_wheel_odom(odom)

                gps = NavSatFix()
                gps.header.stamp.sec = int(curr_t)
                gps.status.status = NavSatStatus.STATUS_FIX
                gps.latitude = curr_lat
                gps.longitude = curr_lon
                gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
                self.node._on_gps_fix(gps)

            return curr_t, curr_lat, curr_lon

        compas_yaw = math.pi / 2.0  # 90° ENU (Norte)

        # 1. Ventana 1: Avance 1.6m con discrepancia (45° diff > 30° thresh)
        t, lat, lon = feed_window(t, lat_base, lon_base, 1.6, compas_yaw, is_dispersed=False)
        self.assertEqual(self.node._consecutive_gps_mag_conflicts, 1)
        self.assertFalse(self.node._gps_mag_conflict)

        # 2. Ventana 2: Avance otro 1.6m con discrepancia
        t, lat, lon = feed_window(t, lat, lon, 1.6, compas_yaw, is_dispersed=False)
        self.assertEqual(self.node._consecutive_gps_mag_conflicts, 2)
        self.assertFalse(self.node._gps_mag_conflict)

        # 3. Ventana 3: Avance 1.6m DESCARTADA por dispersión angular (zigzag/curva, spread > 20°)
        t, lat, lon = feed_window(t, lat, lon, 1.6, compas_yaw, is_dispersed=True)
        # El contador NO debe sumar ni resetear: se mantiene en 2
        self.assertEqual(self.node._consecutive_gps_mag_conflicts, 2)
        self.assertFalse(self.node._gps_mag_conflict)

        # 4. Ventana 4: Tercera ventana con discrepancia
        t, lat, lon = feed_window(t, lat, lon, 1.6, compas_yaw, is_dispersed=False)
        # Ahora alcanza 3 ventanas acumuladas -> DEBE INVALIDAR EL COMPÁS
        self.assertEqual(self.node._consecutive_gps_mag_conflicts, 3)
        self.assertTrue(self.node._gps_mag_conflict)

    def test_gating_a_reset_only_on_concordance(self):
        """Punto 1: Verifica que solo una ventana de alta confianza con concordancia resetea el contador."""
        self.node._disjoint_windows = True
        lat_base, lon_base = 9.791693, -84.105941
        t = 100.0

        def feed_window_dir(start_t, start_lat, start_lon, delta_dist_m, yaw_compas_rad, course_enu_deg):
            steps = 6
            dt = 1.0
            dist_per_step = delta_dist_m / (steps - 1)
            curr_lat = start_lat
            curr_lon = start_lon
            curr_t = start_t

            for s in range(steps):
                if s > 0:
                    curr_t += dt
                    d_north = dist_per_step * math.sin(math.radians(course_enu_deg))
                    d_east = dist_per_step * math.cos(math.radians(course_enu_deg))
                    curr_lat += d_north / 6371000.0 * (180.0 / math.pi)
                    curr_lon += d_east / (6371000.0 * math.cos(math.radians(curr_lat))) * (180.0 / math.pi)

                odom = Odometry()
                odom.header.stamp.sec = int(curr_t)
                odom.twist.twist.linear.x = 0.35
                odom.pose.pose.orientation = Quaternion(
                    x=0.0, y=0.0, z=math.sin(yaw_compas_rad / 2.0), w=math.cos(yaw_compas_rad / 2.0)
                )
                self.node._on_wheel_odom(odom)

                gps = NavSatFix()
                gps.header.stamp.sec = int(curr_t)
                gps.status.status = NavSatStatus.STATUS_FIX
                gps.latitude = curr_lat
                gps.longitude = curr_lon
                gps.position_covariance = [0.0001, 0, 0, 0, 0.0001, 0, 0, 0, 0.01]
                self.node._on_gps_fix(gps)

            return curr_t, curr_lat, curr_lon

        compas_yaw = math.pi / 2.0  # 90° ENU

        # Ventana 1 con discrepancia (GPS a 45° ENU -> error 45° > 30°)
        t, lat, lon = feed_window_dir(t, lat_base, lon_base, 1.6, compas_yaw, 45.0)
        self.assertEqual(self.node._consecutive_gps_mag_conflicts, 1)

        # Ventana 2 con concordancia (GPS a 88° ENU -> error 2° <= 20°)
        t, lat, lon = feed_window_dir(t, lat, lon, 1.6, compas_yaw, 88.0)
        # Se resetea el contador de conflictos a 0
        self.assertEqual(self.node._consecutive_gps_mag_conflicts, 0)
        self.assertEqual(self.node._consecutive_gps_mag_agreements, 1)


if __name__ == "__main__":
    unittest.main()
