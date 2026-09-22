#!/usr/bin/env python3
"""test_pre_claim_dwell.py — Pruebas unitarias para la parada de estabilización pre-reclamo (Paso 3).

Verifica:
1. Al detectar llegada al checkpoint (dist <= radio), inicia dwell y envía (0.0, 0.0) sin reclamar de inmediato.
2. Durante el dwell (t < pre_claim_dwell_s), mantiene la parada (0.0, 0.0) y no llama a claim_checkpoint().
3. Cumplido el dwell (t >= pre_claim_dwell_s) con frente despejado, ejecuta claim_checkpoint().
4. Si el SDK rechaza el reclamo, resetea el dwell y aplica el estrangulamiento de tolerancia geodésica (halving).
5. Si aparece un obstáculo frontal durante el dwell, el dwell se aborta inmediatamente y se atiende la seguridad.
6. Reintento automático tras despeje de obstáculo:
   dwell abortado por obstáculo -> obstáculo se despeja (rover sigue en radio) -> se reinicia un dwell fresco ->
   se completa el tiempo requerido -> claim_checkpoint() se dispara exitosamente (no queda huérfano).
7. Si GpsGuard bloquea el reclamo (modo degradado), el dwell no se inicia ni se ejecuta el reclamo.
"""

import math
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from genie_rover.bridge import Bridge, LoopStats
from genie_rover.navigation import DriveCommand
from genie_rover.sdk_client import Checkpoint


def _build_pre_claim_bridge_stub(target_seq: int = 1, pre_claim_dwell_s: float = 2.0):
    """Construye un stub desacoplado de Bridge para evaluar _step() y la lógica de dwell pre-reclamo."""
    stub = types.SimpleNamespace()

    # Configuración de radio y dwell
    stub.checkpoint_reached_radius_m = 13.0
    stub.claim_radius_m = 13.0
    stub._base_checkpoint_radius_m = 13.0
    stub._current_checkpoint_radius_m = 13.0
    stub._last_target_sequence = None
    stub.pre_claim_dwell_s = pre_claim_dwell_s
    stub._dwell_start_time = None
    stub._dwell_target_seq = None

    # Estado general
    stub.stats = LoopStats()
    stub._last_frame_ts = 0.0
    stub._last_frame_change = 100.0
    stub.stale_frame_s = 2.0
    stub.goal_range_m = 3.5
    stub.resolution = 0.03
    stub.forward_range = 2.0
    stub.side_range = 2.0
    stub.plan_forward_m = 3.0
    stub.plan_side_m = 2.0
    stub.use_map = False
    stub.odometry = None
    stub.pmap = None
    stub.obstacle_persist_frames = 4
    stub._consecutive_blocked = 0
    stub._consecutive_empty = 0
    stub._consecutive_empty_recoveries = 0
    stub._stop_requested = False
    stub.dry_run = False

    stub._disagreements_gps = []
    stub._disagreements_compass = []

    # Target inicial
    stub._target = Checkpoint(
        id=target_seq,
        sequence=target_seq,
        latitude=9.791535,
        longitude=-84.105873,
    )
    stub.current_target = lambda: stub._target

    # Tracking de comandos enviados
    stub.sent_commands: list[DriveCommand] = []

    def mock_send(cmd: DriveCommand):
        stub.sent_commands.append(cmd)

    stub.send = mock_send

    # Mock del SDK client
    stub.client = MagicMock()
    stub.client.front_frame = lambda: (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)
    stub.stale_frame_s = 999.0

    # Telemetría ubicada dentro del radio del checkpoint (~5m al norte)
    cp_lat = stub._target.latitude
    cp_lon = stub._target.longitude
    rover_lat_inside = cp_lat + (5.0 / 111194.9)
    rover_lon_inside = cp_lon

    stub._rover_lat = rover_lat_inside
    stub._rover_lon = rover_lon_inside

    telem_mock = types.SimpleNamespace(
        latitude=rover_lat_inside,
        longitude=rover_lon_inside,
        orientation=0.0,
        timestamp=100.0,
        raw={},
    )
    stub.client.telemetry.return_value = telem_mock
    stub.client.claim_checkpoint.return_value = (True, "OK Claimed")

    def mock_refresh():
        stub.refreshed = True

    stub.refresh_checkpoints = mock_refresh
    stub.refreshed = False

    # Mock heading estimator
    stub.heading_est = types.SimpleNamespace(
        source="compass",
        update=lambda *a, **kw: 0.0,
        disagreement_deg=lambda: None,
        compass_distortion_deg=lambda: None,
        reset_track=lambda: None,
    )

    # Mock GPS guard (Nivel 1 nominal)
    stub._can_claim_checkpoints = True
    stub._guard_level = 1

    class MockGpsGuard:
        @property
        def level(self):
            return stub._guard_level

        def update(self, *a, **kw):
            return types.SimpleNamespace(
                level=stub._guard_level,
                effective_lat=stub._rover_lat,
                effective_lon=stub._rover_lon,
                can_claim_checkpoints=stub._can_claim_checkpoints,
                reason="OK",
            )

        def apply_throttle(self, lin):
            return lin

    stub.gps_guard = MockGpsGuard()

    # Mock percepción
    stub._front_blocked = False

    class MockPerception:
        def process(self, *a, **kw):
            return types.SimpleNamespace(
                traversability=np.ones((20, 20)),
                observed=np.ones((20, 20)),
                stats={"bev_observed_cells": 400},
            )

    stub.perception = MockPerception()
    stub._is_front_blocked = lambda trav: stub._front_blocked

    # Mock gobernador
    stub.governor = MagicMock()
    stub.governor.cfg.enabled = True
    stub.governor.apply_throttle_limit = lambda lin: lin

    # Planificador fallback (por si cae al ciclo de navegación normal)
    stub.planner_cfg = types.SimpleNamespace(grid_size=240, include_goal_in_path_bank=False)
    stub._plan_path_world = None
    stub._plan_pose = None
    stub._plan_t = 0.0
    stub.replan_every_m = 0.1
    stub.replan_min_remaining_m = 0.4
    stub.replan_max_s = 2.0
    stub._path_bank = lambda *a, **kw: None
    stub._send_path_command = lambda path: True
    stub._maybe_dump_debug = lambda *a, **kw: None

    # Vincular métodos reales de Bridge
    for name in ["_step"]:
        setattr(stub, name, types.MethodType(getattr(Bridge, name), stub))

    return stub


class TestPreClaimDwell(unittest.TestCase):
    def test_dwell_starts_and_halts_rover_without_immediate_claim(self):
        """Al entrar en radio de checkpoint, inicia dwell de 2.0s y envía (0, 0) sin invocar claim."""
        bridge = _build_pre_claim_bridge_stub(pre_claim_dwell_s=2.0)

        with patch("genie_rover.bridge.time.time", return_value=100.0):
            bridge._step()

        # Verifica que el dwell inició en t=100.0s
        self.assertEqual(bridge._dwell_start_time, 100.0)
        self.assertEqual(bridge._dwell_target_seq, 1)

        # Rover debe haber enviado comando de freno (0.0, 0.0)
        self.assertEqual(len(bridge.sent_commands), 1)
        self.assertEqual(bridge.sent_commands[0].linear, 0.0)
        self.assertEqual(bridge.sent_commands[0].angular, 0.0)
        self.assertIn("Pre-Claim", bridge.sent_commands[0].reason)

        # NO debe haber llamado a claim_checkpoint aún
        bridge.client.claim_checkpoint.assert_not_called()

    def test_dwell_holds_stop_until_elapsed(self):
        """A los 1.0s de dwell (menor a 2.0s), continúa enviando parada y no reclama."""
        bridge = _build_pre_claim_bridge_stub(pre_claim_dwell_s=2.0)

        # Paso 1 a t=100.0s
        with patch("genie_rover.bridge.time.time", return_value=100.0):
            bridge._step()

        self.assertEqual(bridge._dwell_start_time, 100.0)
        bridge.client.claim_checkpoint.assert_not_called()

        # Paso 2 a t=101.0s (transcurrió 1.0s < 2.0s)
        with patch("genie_rover.bridge.time.time", return_value=101.0):
            bridge._step()

        self.assertEqual(len(bridge.sent_commands), 2)
        self.assertEqual(bridge.sent_commands[-1].linear, 0.0)
        self.assertEqual(bridge.sent_commands[-1].angular, 0.0)
        self.assertIn("Estabilizando", bridge.sent_commands[-1].reason)
        bridge.client.claim_checkpoint.assert_not_called()

    def test_dwell_claims_when_completed(self):
        """Al transcurrir >= 2.0s con frente despejado, ejecuta claim_checkpoint() y resetea dwell."""
        bridge = _build_pre_claim_bridge_stub(pre_claim_dwell_s=2.0)

        # Paso 1: t=100.0s (inicio)
        with patch("genie_rover.bridge.time.time", return_value=100.0):
            bridge._step()

        # Paso 2: t=102.1s (transcurrió 2.1s >= 2.0s)
        with patch("genie_rover.bridge.time.time", return_value=102.1):
            bridge._step()

        bridge.client.claim_checkpoint.assert_called_once()
        self.assertIsNone(bridge._dwell_start_time)
        self.assertIsNone(bridge._dwell_target_seq)
        self.assertTrue(bridge.refreshed)
        self.assertEqual(bridge._current_checkpoint_radius_m, 13.0)

    def test_dwell_rejection_halves_tolerance(self):
        """Si el SDK rechaza el reclamo post-dwell, se aplica halving de tolerancia (13.0 -> 6.5m)."""
        bridge = _build_pre_claim_bridge_stub(pre_claim_dwell_s=2.0)
        bridge.client.claim_checkpoint.return_value = (False, "SDK 422 Rejection")

        with patch("genie_rover.bridge.time.time", return_value=100.0):
            bridge._step()

        with patch("genie_rover.bridge.time.time", return_value=102.5), \
             patch("genie_rover.bridge.plan_on_bev", return_value=types.SimpleNamespace(final_path_xy_m=np.zeros((5, 2)))):
            bridge._step()

        bridge.client.claim_checkpoint.assert_called_once()
        self.assertIsNone(bridge._dwell_start_time)
        # Tolerancia estrangulada a la mitad
        self.assertAlmostEqual(bridge._current_checkpoint_radius_m, 6.5)

    def test_obstacle_aborts_dwell(self):
        """Si aparece un obstáculo frontal durante el dwell, se aborta y se resetea _dwell_start_time."""
        bridge = _build_pre_claim_bridge_stub(pre_claim_dwell_s=2.0)

        # Paso 1: t=100.0s (inicia dwell sin obstáculo)
        with patch("genie_rover.bridge.time.time", return_value=100.0):
            bridge._step()
        self.assertEqual(bridge._dwell_start_time, 100.0)

        # Paso 2: t=101.0s, aparece obstáculo frontal
        bridge._front_blocked = True
        with patch("genie_rover.bridge.time.time", return_value=101.0):
            bridge._step()

        # Dwell debe haber sido abortado
        self.assertIsNone(bridge._dwell_start_time)
        self.assertIsNone(bridge._dwell_target_seq)
        # Comando debe ser de obstáculo al frente
        self.assertIn("OBSTACULO", bridge.sent_commands[-1].reason)
        bridge.client.claim_checkpoint.assert_not_called()

    def test_obstacle_aborts_dwell_and_restarts_cleanly_when_cleared(self):
        """Dwell abortado por obstáculo -> obstáculo se despeja -> confirma reinicio limpio de dwell y reclamo."""
        bridge = _build_pre_claim_bridge_stub(pre_claim_dwell_s=2.0)

        # 1. t=100.0s: inicia dwell limpio
        with patch("genie_rover.bridge.time.time", return_value=100.0):
            bridge._step()
        self.assertEqual(bridge._dwell_start_time, 100.0)

        # 2. t=101.0s: se cruza un obstáculo -> aborta dwell
        bridge._front_blocked = True
        with patch("genie_rover.bridge.time.time", return_value=101.0):
            bridge._step()
        self.assertIsNone(bridge._dwell_start_time)
        bridge.client.claim_checkpoint.assert_not_called()

        # 3. t=103.0s: el obstáculo se despeja. El rover sigue dentro del radio del checkpoint.
        bridge._front_blocked = False
        with patch("genie_rover.bridge.time.time", return_value=103.0):
            bridge._step()

        # El dwell se debe haber REINICIADO limpiamente a t=103.0s
        self.assertEqual(bridge._dwell_start_time, 103.0)
        self.assertEqual(bridge._dwell_target_seq, 1)
        bridge.client.claim_checkpoint.assert_not_called()

        # 4. t=104.0s (1.0s transcurrido < 2.0s): sigue estabilizando
        with patch("genie_rover.bridge.time.time", return_value=104.0):
            bridge._step()
        self.assertEqual(bridge._dwell_start_time, 103.0)
        bridge.client.claim_checkpoint.assert_not_called()

        # 5. t=105.2s (2.2s transcurrido >= 2.0s): completa el reposo y dispara el reclamo
        with patch("genie_rover.bridge.time.time", return_value=105.2):
            bridge._step()

        bridge.client.claim_checkpoint.assert_called_once()
        self.assertIsNone(bridge._dwell_start_time)
        self.assertTrue(bridge.refreshed)

    def test_gps_guard_blocks_dwell_and_claim(self):
        """Si GpsGuard bloquea el reclamo (ej. sin anclaje de rumbo o degradado), no inicia dwell."""
        bridge = _build_pre_claim_bridge_stub(pre_claim_dwell_s=2.0)
        bridge._can_claim_checkpoints = False

        with patch("genie_rover.bridge.time.time", return_value=100.0), \
             patch("genie_rover.bridge.plan_on_bev", return_value=types.SimpleNamespace(final_path_xy_m=np.zeros((5, 2)))):
            bridge._step()

        self.assertIsNone(bridge._dwell_start_time)
        bridge.client.claim_checkpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
