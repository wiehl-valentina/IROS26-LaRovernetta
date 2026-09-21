"""Pruebas unitarias para la maniobra de escaneo de 360° continuo en recovery,
sincronización frame/pose, corte anticipado, reintento de mapa y guardas anti-bucle.

Casos evaluados:
1. Sentido de arranque: meta a la izquierda (bearing < 0) -> antihorario (CCW);
   meta a la derecha/frente (bearing >= 0) -> horario (CW).
2. Corte anticipado: libre >= 70% al frente tras superar startup_latency corta inmediatamente
   y arranca con _unstick(); libre=45% (viable pero < 70%) NO corta y sigue barriendo.
3. Sincronización crítica frame/pose durante rotación: con rotación continua y latencia
   de pipeline, la proyección BEV en el mapa persistente usa la pose en frame_ts
   (no la pose desplazada tras el procesamiento).
4. Reintento de nivel 1 con datos frescos tras escaneo completo:
   mapa inicial vacío -> escaneo 360° repuebla -> reintento de mapa elige salida sin llamar a VLM.
5. Guardas anti-bucle: recovery_scan_max_per_stuck y cooldown temporal impiden escaneos
   repetidos en el mismo atascamiento hasta que el rover se desplace una distancia mínima.
6. Veto por inclinación: pendiente peligrosa (>= 8.0°) bloquea el escaneo de 360°.

Uso:
    pytest genie/genie_rover/test_recovery_scan_360.py
"""

from __future__ import annotations

import math
import time
import types
import unittest
import numpy as np

from .bridge import Bridge, LoopStats
from .navigation import DriveCommand, PathFollower
from .odometry import Odometry, OdometryConfig, Pose, wrap_rad
from .persistent_map import MapConfig, PersistentMap


def _build_scan_stub(tilt_pitch_deg: float = 0.0, tilt_roll_deg: float = 0.0):
    """Construye un stub desacoplado de Bridge con odometría y mapa persistente."""
    stub = types.SimpleNamespace()
    stub.resolution = 0.03
    stub.follower = PathFollower(angular_sign=-1.0, turn_speed=0.55, max_linear=0.40, max_angular=0.65)
    stub.forward_range = 2.0
    stub.side_range = 1.0
    stub.plan_forward_m = 2.0
    stub.plan_side_m = 1.0
    stub.use_map = True

    # Parámetros de recovery estándar
    stub.retroceso_max_m = 0.6
    stub.retroceso_paso_m = 0.2
    stub.retroceso_linear = -0.18
    stub.retroceso_min_libre_pct = 55.0
    stub.retroceso_min_cobertura_pct = 25.0
    stub.recovery_headings_deg = [0.0, 45.0, -45.0, 90.0, -90.0, 180.0]
    stub.recovery_min_cobertura_pct = 25.0
    stub.recovery_min_libre_pct = 30.0
    stub.recovery_goal_weight = 1.2
    stub.recovery_clearance_weight = 1.0
    stub.heading_search_radius_m = 1.5
    stub.recovery_turn_speed = 0.75
    stub.recovery_deg_per_s = 20.0
    stub.recovery_step_deg = 30.0
    stub.recovery_startup_latency_s = 0.1  # Acotado para tests rápidos
    stub.recovery_turn_tolerance_deg = 15.0
    stub.recovery_tilt_veto_deg = 8.0
    stub.unstick_forward_s = 0.2
    stub._stop_requested = False

    # Parámetros nuevos de escaneo 360°
    stub.use_recovery_scan = True
    stub.recovery_scan_deg_per_s = 20.0
    stub.recovery_scan_turn_speed = 0.75
    stub.recovery_scan_early_exit_libre_pct = 70.0
    stub.recovery_scan_max_per_stuck = 1
    stub.recovery_scan_min_disp_m = 1.0
    stub.recovery_scan_cooldown_s = 30.0
    stub.recovery_scan_timeout_s = 5.0
    stub._scan_count_at_stuck = 0
    stub._last_scan_pose = None
    stub._last_scan_time = 0.0

    # VLM config
    stub.use_vlm_recovery = True
    stub._vlm_called = 0

    stub.stats = LoopStats()

    sent: list[DriveCommand] = []
    stub.send = lambda cmd: sent.append(cmd)
    stub._sent = sent

    # Odometría simulada
    odo_cfg = OdometryConfig()
    odo = Odometry(odo_cfg)
    stub.odometry = odo
    if tilt_pitch_deg != 0.0 or tilt_roll_deg != 0.0:
        odo.last_pitch = math.radians(tilt_pitch_deg)
        odo.last_roll = math.radians(tilt_roll_deg)
        odo.tilt_gate_open = True

    # Mapa persistente inicial vacío (cobertura 0)
    pmap = PersistentMap(MapConfig(size_m=6.0, resolution_m_per_px=0.03))
    stub.pmap = pmap

    # Mock client
    stub._sim_frame_ts = 1000.0
    stub._sim_rgb = np.zeros((16, 16, 3), dtype=np.uint8)

    class _MockClient:
        def front_frame(self):
            return stub._sim_rgb, stub._sim_frame_ts

        def telemetry(self):
            return types.SimpleNamespace(raw={}, ekf_heading=None, ekf_heading_time=None)

    stub.client = _MockClient()

    # Mock perception
    class _MockPerception:
        def __init__(self):
            self.bev = np.ones((16, 16), dtype=np.float32)
            self.obs = np.ones((16, 16), dtype=np.float32)

        def process(self, rgb, roll_rad=None, pitch_rad=None):
            return types.SimpleNamespace(traversability=self.bev, observed=self.obs, stats={})

    stub.perception = _MockPerception()

    # Bind methods
    for name in ("_recover_informado", "_map_free_and_coverage", "_girar_hacia",
                 "_preguntar_vlm", "_barrido_ciego", "_retroceder",
                 "_retroceso_y_recover", "_recover", "_unstick",
                 "_reset_recovery_state", "_get_goal_relative_bearing_deg",
                 "_is_tilt_too_steep_for_recovery", "_get_estimated_tilt_deg",
                 "_is_front_blocked", "_evaluar_candidatos_recovery_mapa",
                 "_escanear_360"):
        if hasattr(Bridge, name):
            setattr(stub, name, types.MethodType(getattr(Bridge, name), stub))

    return stub


class TestRecoveryScan360(unittest.TestCase):

    def test_scan_direction_towards_checkpoint_left(self):
        """Paso 1: Si el checkpoint está a la izquierda (bearing < 0), debe iniciar girando antihorario (CCW)."""
        stub = _build_scan_stub()
        stub._get_goal_relative_bearing_deg = lambda: -60.0  # Meta a la izquierda (-60°)

        # Simulamos que tras el primer comando corta por stop_requested
        original_send = stub.send

        def _intercept_send(cmd: DriveCommand):
            original_send(cmd)
            stub._stop_requested = True

        stub.send = _intercept_send
        stub._escanear_360()

        # Con angular_sign = -1.0 y turn_dir = -1.0 (izquierda / antihorario):
        # cmd_ang = (-1.0) * (-0.75) = +0.75 > 0 (en frodobot, angular positivo gira a la izquierda)
        first_scan_cmd = [c for c in stub._sent if "escaneo 360°" in c.reason][0]
        self.assertGreater(first_scan_cmd.angular, 0.0,
                           "Con meta a la izquierda (bearing < 0), debe enviar comando angular positivo (antihorario/izq)")

    def test_scan_direction_towards_checkpoint_right(self):
        """Paso 1: Si el checkpoint está a la derecha (bearing > 0), debe iniciar girando horario (CW)."""
        stub = _build_scan_stub()
        stub._get_goal_relative_bearing_deg = lambda: +45.0  # Meta a la derecha (+45°)

        original_send = stub.send

        def _intercept_send(cmd: DriveCommand):
            original_send(cmd)
            stub._stop_requested = True

        stub.send = _intercept_send
        stub._escanear_360()

        # Con angular_sign = -1.0 y turn_dir = +1.0 (derecha / horario):
        # cmd_ang = (-1.0) * (+0.75) = -0.75 < 0 (en frodobot, angular negativo gira a la derecha)
        first_scan_cmd = [c for c in stub._sent if "escaneo 360°" in c.reason][0]
        self.assertLess(first_scan_cmd.angular, 0.0,
                        "Con meta a la derecha (bearing > 0), debe enviar comando angular negativo (horario/der)")

    def test_early_exit_triggers_at_75_percent_free(self):
        """Paso 3: Si durante el giro encuentra un pasaje al 75% libre (>= 70%), debe cortar anticipadamente y arrancar."""
        stub = _build_scan_stub()
        stub.recovery_startup_latency_s = 0.0
        stub._get_goal_relative_bearing_deg = lambda: 0.0

        # Odometría que simula rotación física
        iter_count = 0

        def _mock_update(*a, **kw):
            nonlocal iter_count
            iter_count += 1
            # Cada iteración rota 20°
            stub.odometry.pose.theta = wrap_rad(math.radians(iter_count * 20.0))
            return stub.odometry.pose

        stub.odometry.update = _mock_update

        # Mock de mapa: en la segunda iteración (girado >= 15°), el frente tiene 75% libre y 100% cobertura
        def _mock_map_free(pose, heading_rel_deg, radius):
            if iter_count >= 2:
                return 75.0, 100.0  # Claro libre >= 70%
            return 10.0, 100.0

        stub._map_free_and_coverage = _mock_map_free

        corte = stub._escanear_360()
        self.assertTrue(corte, "Debe retornar True indicando corte anticipado exitoso")

        # Verificar que se envió comando de parada y luego arranque forzado (_unstick)
        reasons = [c.reason for c in stub._sent]
        self.assertTrue(any("corte anticipado de escaneo 360" in r for r in reasons),
                        "Debe enviar comando de parada al detectar corte anticipado")
        self.assertTrue(any("avance forzado" in r for r in reasons),
                        "Debe arrancar hacia la salida clara (_unstick)")

    def test_no_early_exit_at_45_percent_free(self):
        """Paso 3: Un pasaje al 45% libre (< 70%) es viable pero no claramente despejado: NO debe cortar."""
        stub = _build_scan_stub()
        stub.recovery_startup_latency_s = 0.0
        stub.recovery_scan_timeout_s = 0.2  # Cortar rápido por timeout para verificar que no cortó anticipado
        stub._get_goal_relative_bearing_deg = lambda: 0.0

        iter_count = 0

        def _mock_update(*a, **kw):
            nonlocal iter_count
            iter_count += 1
            stub.odometry.pose.theta = wrap_rad(math.radians(iter_count * 20.0))
            return stub.odometry.pose

        stub.odometry.update = _mock_update

        # 45% libre (viable pero por debajo del umbral de corte anticipado 70%)
        stub._map_free_and_coverage = lambda pose, heading, radius: (45.0, 100.0)

        corte = stub._escanear_360()
        self.assertFalse(corte, "Con 45% libre NO debe activar corte anticipado")
        reasons = [c.reason for c in stub._sent]
        self.assertFalse(any("corte anticipado" in r for r in reasons),
                         "No debe registrar corte anticipado con 45% libre")

    def test_frame_pose_synchronization_with_pipeline_latency(self):
        """Paso 2: Verifica que el frame capturado se proyecte con el theta de frame_ts, no el de post-procesamiento."""
        stub = _build_scan_stub()

        integrated_poses: list[Pose] = []
        stub.pmap.integrate = lambda bev, obs, pose, *a, **kw: integrated_poses.append(
            Pose(pose.x, pose.y, pose.theta)
        )

        # Configuramos odometría con historial temporal
        t_capture = 100.0
        t_post_inference = 100.283  # 283ms después (a 20°/s el robot rotó ~5.66°)

        pose_at_capture = Pose(0.0, 0.0, math.radians(45.0))
        pose_at_post = Pose(0.0, 0.0, math.radians(50.66))

        stub.odometry._pose_history.append((t_capture, pose_at_capture))
        stub.odometry._pose_history.append((t_post_inference, pose_at_post))
        stub.odometry.pose = Pose(0.0, 0.0, math.radians(50.66))

        stub._sim_frame_ts = t_capture

        # Interrumpir tras la primera integración
        def _mock_update(*a, **kw):
            stub._stop_requested = True
            return stub.odometry.pose

        stub.odometry.update = _mock_update

        stub._escanear_360()

        self.assertGreater(len(integrated_poses), 0, "Debe haber integrado al menos un frame")
        theta_integrated_deg = math.degrees(integrated_poses[0].theta)
        self.assertAlmostEqual(theta_integrated_deg, 45.0, places=1,
                               msg="La observación debe integrarse con el theta de captura (45°), NO con el posterior a la latencia (50.66°)")

    def test_full_cascade_retry_map_after_scan_succeeds(self):
        """Paso 4: Cascada completa: Nivel 1 falla -> escaneo 360° repuebla mapa -> reintento de mapa encuentra salida y NO llama VLM."""
        stub = _build_scan_stub()
        stub._preguntar_vlm = lambda: (_ for _ in ()).throw(AssertionError("VLM no debió llamarse si el reintento de mapa encontró salida"))

        # 1. En el primer chequeo (Nivel 1), mapa vacío -> sin salida
        # 2. Durante el escaneo, no hay corte anticipado
        # 3. En el reintento tras escaneo, 90° tiene 85% libre con 100% cobertura
        first_call = True

        def _mock_eval_map(veto_tilt, razon_tilt):
            nonlocal first_call
            if first_call:
                first_call = False
                return None  # Nivel 1 falla
            # Reintento tras escaneo: encuentra rumbo lateral +90°
            return {
                "heading": 90.0,
                "is_180": False,
                "libre_pct": 85.0,
                "cobertura_pct": 100.0,
                "score": 1.7,
            }

        stub._evaluar_candidatos_recovery_mapa = _mock_eval_map
        stub._escanear_360 = lambda: False  # Completa vuelta completa sin corte

        stub._recover_informado()

        self.assertEqual(stub.stats.recoveries_por_mapa, 1, "Debe contar como recuperación por mapa exitosa")
        reasons = [c.reason for c in stub._sent]
        self.assertTrue(any("girando hacia +90" in r for r in reasons),
                        "Debe haber girado hacia el rumbo +90° encontrado en el reintento tras escaneo")

    def test_anti_loop_limits_scan_repetition(self):
        """Paso 5: Evitar bucles de escaneo si el rover sigue encajonado en la misma posición."""
        stub = _build_scan_stub()
        stub.recovery_scan_max_per_stuck = 1
        stub.recovery_scan_cooldown_s = 60.0
        stub.recovery_scan_min_disp_m = 1.0

        vlm_called = [0]
        stub._preguntar_vlm = lambda: vlm_called.__setitem__(0, vlm_called[0] + 1) or None
        stub._barrido_ciego = lambda: None

        scan_called = [0]

        def _mock_scan():
            scan_called[0] += 1
            return False

        stub._escanear_360 = _mock_scan
        stub._evaluar_candidatos_recovery_mapa = lambda *a, **kw: None  # Mapa nunca encuentra salida

        # Intento 1 en stuck: el escaneo DEBE dispararse
        stub._recover_informado()
        self.assertEqual(scan_called[0], 1, "Primer intento debe ejecutar el escaneo 360°")

        # Intento 2 en el mismo sitio (sin desplazamiento): el escaneo NO debe repetirse, salta directo a VLM
        stub._recover_informado()
        self.assertEqual(scan_called[0], 1, "Segundo intento atascado en el mismo sitio NO debe ejecutar escaneo")
        self.assertGreater(vlm_called[0], 0, "Debe haber saltado directamente a VLM")

        # Ahora el robot avanza 1.5 metros (> min_disp_m = 1.0m)
        stub.odometry.pose.x += 1.5
        # Forzar que transcurra el cooldown para evaluar la guarda espacial
        stub._last_scan_time = time.time() - 70.0

        # Intento 3 tras desplazamiento: el escaneo vuelve a estar disponible
        stub._recover_informado()
        self.assertEqual(scan_called[0], 2, "Tras desplazarse > 1.0m, el escaneo 360° debe volver a estar disponible")

    def test_steep_slope_vetoes_360_scan(self):
        """Restricción: Si la inclinación supera recovery_tilt_veto_deg (8.0°), el escaneo 360° se veta."""
        stub = _build_scan_stub(tilt_pitch_deg=10.0, tilt_roll_deg=0.0)  # Pendiente de 10°

        scan_called = False

        def _mock_scan():
            nonlocal scan_called
            scan_called = True
            return False

        stub._escanear_360 = _mock_scan
        stub._evaluar_candidatos_recovery_mapa = lambda *a, **kw: None
        vlm_called = False

        def _mock_vlm():
            nonlocal vlm_called
            vlm_called = True
            return None

        stub._preguntar_vlm = _mock_vlm
        stub._barrido_ciego = lambda: None

        stub._recover_informado()

        self.assertFalse(scan_called, "En pendiente >= 8.0°, el escaneo 360° debe estar VETADO")
        self.assertTrue(vlm_called, "Debe saltear directamente a VLM sin intentar escaneo 360°")


if __name__ == "__main__":
    unittest.main()
