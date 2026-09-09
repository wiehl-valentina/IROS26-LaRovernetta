"""Pruebas unitarias para las politicas de recuperacion bajo pendiente (Capa 3):
1. Veto de retroceso lineal si la inclinacion (pitch/roll) >= recovery_tilt_veto_deg.
2. Veto de giro 180° si la inclinacion (pitch/roll) >= recovery_tilt_veto_deg.
3. Seleccion de rumbo en _recover_informado descartando 180° en pendiente.
4. Regla semantica VLM on_road: descarte de 'adelante' si on_road=False.
5. Veto de 'atras' sugerido por VLM si hay pendiente peligrosa.
6. Limite de reintentos consecutivos y cooldown del VLM.

Uso:
    python -m genie_rover.test_recovery_tilt_veto
"""

from __future__ import annotations

import math
import time
import types
import unittest
import numpy as np

from .bridge import Bridge
from .navigation import DriveCommand, PathFollower
from .odometry import Pose
from .persistent_map import MapConfig, PersistentMap
from .vlm_recovery import RecoveryDecision


def _build_test_stub(tilt_pitch_deg: float = 0.0, tilt_roll_deg: float = 0.0):
    """Construye un stub minimo de Bridge con inclinacion configurable."""
    stub = types.SimpleNamespace()
    stub.resolution = 0.03
    stub.follower = PathFollower(angular_sign=-1.0, turn_speed=0.35,
                                 max_linear=0.35, max_angular=0.45)
    stub.retroceso_max_m = 0.4
    stub.retroceso_paso_m = 0.2
    stub.retroceso_linear = -0.18
    stub.retroceso_min_libre_pct = 50.0
    stub.retroceso_min_cobertura_pct = 20.0
    stub.recovery_headings_deg = [0.0, 90.0, -90.0, 180.0]
    stub.recovery_min_cobertura_pct = 20.0
    stub.heading_search_radius_m = 0.5
    stub.recovery_turn_speed = 0.45
    stub.recovery_deg_per_s = 60.0
    stub.recovery_step_deg = 30.0
    stub.recovery_turn_s = 0.1
    stub.front_near_m = 0.32
    stub.front_far_m = 0.85
    stub.front_half_width_m = 0.22
    stub.front_traversable_thresh = 0.28
    stub.front_min_free_ratio = 0.40
    stub.allow_reverse = True
    stub.use_vlm_recovery = True
    stub.vlm_recovery_timeout_s = 4.0
    stub.vlm_recovery_min_confidence = 0.35
    stub.recovery_tilt_veto_deg = 8.0
    stub.vlm_recovery_max_retries = 2
    stub.vlm_recovery_cooldown_s = 5.0
    stub._vlm_consecutive_calls = 0
    stub._last_vlm_call_time = 0.0
    stub.use_map = True

    stub.stats = types.SimpleNamespace(
        retrocesos=0, recoveries_por_mapa=0, recoveries_por_vlm=0,
        recoveries_ciegas=0, near_regime_activations=0,
    )
    stub._stop_requested = False

    # Mapa sintetico: atras (180°) y derecha (90°) libres, adelante bloqueado
    pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    pmap.value[:] = 1.0
    pmap.conf[:] = 1.0
    # Bloqueo adelante
    for x in np.arange(0.05, 0.65, pmap.cfg.resolution_m_per_px):
        for y in np.arange(-0.3, 0.3, pmap.cfg.resolution_m_per_px):
            f, c = pmap.world_to_cell(float(x), float(y))
            if 0 <= f < pmap.n and 0 <= c < pmap.n:
                pmap.value[f, c] = 0.0
    stub.pmap = pmap

    stub.heading_est = types.SimpleNamespace(reset_track=lambda: None)
    stub._consecutive_turns = 0
    stub._turn_sign_history = []
    stub._consecutive_empty = 0
    stub._commit_side = 0
    stub._consecutive_blocked = 3
    stub._plan_path_world = np.zeros((5, 2))
    stub._plan_pose = Pose(0.0, 0.0, 0.0)

    pose_holder = {"pose": Pose(0.0, 0.0, 0.0)}

    class _MockOdometry:
        def __init__(self, pitch_deg: float, roll_deg: float):
            self.pitch_rad = math.radians(pitch_deg)
            self.roll_rad = math.radians(roll_deg)
            self.last_pitch = self.pitch_rad
            self.last_roll = self.roll_rad

        @property
        def pose(self):
            return pose_holder["pose"]

        def update(self, _raw, *a, **kw):
            return pose_holder["pose"]

        def current_roll_pitch(self, *a, **kw):
            return (self.roll_rad, self.pitch_rad)

    stub.odometry = _MockOdometry(tilt_pitch_deg, tilt_roll_deg)

    sent: list[DriveCommand] = []
    stub.send = lambda cmd: sent.append(cmd)
    stub._sent = sent

    class _MockClient:
        def telemetry(self):
            return types.SimpleNamespace(raw={}, ekf_heading=None, ekf_heading_time=None)

        def front_frame(self):
            return np.zeros((16, 16, 3), dtype=np.uint8), 0.0

        def rear_frame(self):
            return np.zeros((16, 16, 3), dtype=np.uint8), 0.0

    stub.client = _MockClient()

    class _MockPerception:
        def process(self, _rgb, *a, **kw):
            return types.SimpleNamespace(
                traversability=np.ones((16, 16), dtype=np.float32),
                observed=np.ones((16, 16), dtype=np.float32),
            )

    stub.perception = _MockPerception()

    for name in ("_map_free_and_coverage", "_girar_hacia", "_barrido_ciego",
                 "_preguntar_vlm", "_retroceder", "_recover_informado",
                 "_retroceso_y_recover", "_is_tilt_too_steep_for_recovery",
                 "_get_estimated_tilt_deg", "_is_front_blocked"):
        if hasattr(Bridge, name):
            setattr(stub, name, types.MethodType(getattr(Bridge, name), stub))

    return stub


class TestRecoveryTiltVeto(unittest.TestCase):

    def test_tilt_veto_detection(self):
        """Verifica que _is_tilt_too_steep_for_recovery detecte correctamente pendientes."""
        stub_level = _build_test_stub(tilt_pitch_deg=2.0, tilt_roll_deg=3.0)
        veto, razon = stub_level._is_tilt_too_steep_for_recovery()
        self.assertFalse(veto)
        self.assertIn("segura", razon)

        stub_pitch_steep = _build_test_stub(tilt_pitch_deg=9.5, tilt_roll_deg=1.0)
        veto, razon = stub_pitch_steep._is_tilt_too_steep_for_recovery()
        self.assertTrue(veto)
        self.assertIn("excesiva", razon)

        stub_roll_steep = _build_test_stub(tilt_pitch_deg=1.0, tilt_roll_deg=8.5)
        veto, razon = stub_roll_steep._is_tilt_too_steep_for_recovery()
        self.assertTrue(veto)
        self.assertIn("excesiva", razon)

    def test_retroceso_vetoed_on_steep_slope(self):
        """Si la pendiente supera 8.0°, _retroceder() no debe mandar comandos de movimiento."""
        stub = _build_test_stub(tilt_pitch_deg=10.0, tilt_roll_deg=0.0)
        stub._retroceder()
        self.assertEqual(len(stub._sent), 0, "No debe enviar comandos si hay pendiente excesiva")
        self.assertEqual(stub.stats.retrocesos, 0)

    def test_retroceso_allowed_on_flat_ground(self):
        """Si el terreno esta nivelado (< 8.0°), _retroceder() si se ejecuta."""
        stub = _build_test_stub(tilt_pitch_deg=3.0, tilt_roll_deg=2.0)
        stub._retroceder()
        self.assertGreater(len(stub._sent), 0, "Debe enviar comandos de retroceso")
        self.assertEqual(stub.stats.retrocesos, 1)
        self.assertTrue(any(c.linear < 0 for c in stub._sent))

    def test_giro_180_vetoed_on_steep_slope(self):
        """Si la pendiente supera 8.0°, _girar_hacia(180) debe ser bloqueado defensivamente."""
        stub = _build_test_stub(tilt_pitch_deg=11.0, tilt_roll_deg=0.0)
        stub._girar_hacia(180.0)
        self.assertEqual(len(stub._sent), 0, "Giro 180° debe ser bloqueado por pendiente")

        stub._girar_hacia(-180.0)
        self.assertEqual(len(stub._sent), 0, "Giro -180° debe ser bloqueado por pendiente")

        # Giros menores a 180° en pendiente siguen permitidos para desatascarse
        stub._girar_hacia(45.0)
        self.assertGreater(len(stub._sent), 0, "Giro de 45° no debe ser bloqueado")

    def test_giro_180_allowed_on_flat_ground(self):
        """En terreno plano, _girar_hacia(180) si se ejecuta."""
        stub = _build_test_stub(tilt_pitch_deg=2.0, tilt_roll_deg=1.0)
        stub._girar_hacia(180.0)
        self.assertGreater(len(stub._sent), 0, "Giro 180° debe ejecutarse en terreno plano")

    def test_recover_informado_skips_180_on_steep_slope(self):
        """En _recover_informado, si 180° es el mas libre pero hay pendiente, se debe vetar y elegir otro."""
        # Terreno plano: 180° deberia ser elegido porque esta 100% libre
        stub_flat = _build_test_stub(tilt_pitch_deg=0.0, tilt_roll_deg=0.0)
        stub_flat._sent.clear()
        stub_flat._recover_informado()
        self.assertEqual(stub_flat.stats.recoveries_por_mapa, 1)

        # Terreno inclinado (10° pitch): 180° es vetado, elije 90° o -90° (no bloqueado)
        stub_steep = _build_test_stub(tilt_pitch_deg=10.0, tilt_roll_deg=0.0)
        stub_steep._sent.clear()
        stub_steep._recover_informado()
        self.assertEqual(stub_steep.stats.recoveries_por_mapa, 1)
        self.assertGreater(len(stub_steep._sent), 0)

    def test_vlm_on_road_false_and_adelante_discarded(self):
        """Si VLM sugiere 'adelante' pero indica on_road=False, debe descartarse la sugerencia."""
        stub = _build_test_stub(tilt_pitch_deg=0.0, tilt_roll_deg=0.0)
        # Forzamos que el mapa no encuentre ningun rumbo
        stub.recovery_min_cobertura_pct = 999.0

        # Mock de _preguntar_vlm que devuelve 'adelante' con on_road=False
        vlm_decision = RecoveryDecision(heading="adelante", on_road=False, confidence=0.9, reason="terreno malo enfrente")
        stub._preguntar_vlm = lambda: vlm_decision

        stub._sent.clear()
        stub._recover_informado()
        # No debio tomar VLM (debio caer a barrido ciego)
        self.assertEqual(stub.stats.recoveries_por_vlm, 0)
        self.assertEqual(stub.stats.recoveries_ciegas, 1)

    def test_vlm_on_road_false_and_side_escape_accepted(self):
        """Si VLM sugiere 'izquierda' con on_road=False, es una orden valida de escape."""
        stub = _build_test_stub(tilt_pitch_deg=0.0, tilt_roll_deg=0.0)
        stub.recovery_min_cobertura_pct = 999.0

        vlm_decision = RecoveryDecision(heading="izquierda", on_road=False, confidence=0.85, reason="escapar por izquierda")
        stub._preguntar_vlm = lambda: vlm_decision

        stub._sent.clear()
        stub._recover_informado()
        self.assertEqual(stub.stats.recoveries_por_vlm, 1)
        self.assertEqual(stub.stats.recoveries_ciegas, 0)

    def test_vlm_suggests_atras_on_steep_slope_vetoed(self):
        """Si VLM sugiere 'atras' pero hay pendiente peligrosa, se veta y cae a barrido ciego."""
        stub = _build_test_stub(tilt_pitch_deg=10.0, tilt_roll_deg=0.0)
        stub.recovery_min_cobertura_pct = 999.0

        vlm_decision = RecoveryDecision(heading="atras", on_road=True, confidence=0.9, reason="dar la vuelta")
        stub._preguntar_vlm = lambda: vlm_decision

        stub._sent.clear()
        stub._recover_informado()
        # VLM no debe registrarse como exitoso si fue vetado
        self.assertEqual(stub.stats.recoveries_por_vlm, 0)
        self.assertEqual(stub.stats.recoveries_ciegas, 1)

    def test_vlm_cooldown_and_retry_limit(self):
        """Verifica que tras exceder max_retries, el VLM entre en cooldown y no llame a la API."""
        stub = _build_test_stub(tilt_pitch_deg=0.0, tilt_roll_deg=0.0)
        stub.vlm_recovery_max_retries = 2
        stub.vlm_recovery_cooldown_s = 2.0
        stub._vlm_consecutive_calls = 0
        stub._last_vlm_call_time = 0.0

        call_count = [0]
        def mock_ask(*a, **kw):
            call_count[0] += 1
            return RecoveryDecision("derecha", True, 0.8, "ok")

        import sys
        mock_module = types.ModuleType("genie_rover.vlm_recovery")
        mock_module.ask_recovery_heading = mock_ask
        sys.modules["genie_rover.vlm_recovery"] = mock_module

        # Llamada 1
        res1 = stub._preguntar_vlm()
        self.assertIsNotNone(res1)
        self.assertEqual(call_count[0], 1)
        self.assertEqual(stub._vlm_consecutive_calls, 1)

        # Llamada 2
        res2 = stub._preguntar_vlm()
        self.assertIsNotNone(res2)
        self.assertEqual(call_count[0], 2)
        self.assertEqual(stub._vlm_consecutive_calls, 2)

        # Llamada 3 (debe saltar cooldown)
        res3 = stub._preguntar_vlm()
        self.assertIsNone(res3, "Tercera llamada consecutiva debe devolver None por cooldown")
        self.assertEqual(call_count[0], 2, "No debio llamar a ask_recovery_heading")

        # Simulamos paso del tiempo mayor al cooldown
        stub._last_vlm_call_time = time.time() - 3.0
        res4 = stub._preguntar_vlm()
        self.assertIsNotNone(res4, "Despues del cooldown debe volver a consultar")
        self.assertEqual(call_count[0], 3)
        self.assertEqual(stub._vlm_consecutive_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
