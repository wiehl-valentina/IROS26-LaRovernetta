"""Pruebas unitarias para la recuperacion bilateral, ponderacion hacia la meta,
neutralidad de reversa y limpieza del endpoint de recovery.

Casos verificados:
(a) El lado izquierdo tiene clearance y esta mas cerca del checkpoint que 180°
    -> debe elegir izquierda (-90°), no 180°.
(b) Ningun lado lateral tiene clearance suficiente
    -> debe caer a 180° como ultimo recurso (escape trasero).
(c) Durante el retroceso, el comando angular es estrictamente neutro (0.0),
    garantizando que el path follower no desvie la trayectoria.
(d) Endpoint de recovery: confirma que al terminar la recuperacion no queda
    ningun estado residual (_commit_side, _commit_until, _plan_path_world,
    _plan_pose, _consecutive_turns, etc.) que contamine el siguiente ciclo.
(e) Preservacion de veto por inclinacion: pendiente excesiva (>= 8.0°) veta 180°
    incluso si los laterales estan bloqueados, cayendo limpiamente a la cascada.

Uso:
    python -m genie_rover.test_bilateral_recovery
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


def _build_recovery_stub(tilt_pitch_deg: float = 0.0, tilt_roll_deg: float = 0.0):
    """Construye un stub minimo de Bridge para evaluar _recover_informado y el ciclo de recovery."""
    stub = types.SimpleNamespace()
    stub.resolution = 0.03
    stub.follower = PathFollower(angular_sign=-1.0, turn_speed=0.35, max_linear=0.35, max_angular=0.45)
    stub.retroceso_max_m = 0.4
    stub.retroceso_paso_m = 0.2
    stub.retroceso_linear = -0.18
    stub.retroceso_min_libre_pct = 50.0
    stub.retroceso_min_cobertura_pct = 20.0
    stub.recovery_headings_deg = [0.0, 90.0, -90.0, 180.0]
    stub.recovery_min_cobertura_pct = 20.0
    stub.recovery_min_libre_pct = 30.0
    stub.recovery_goal_weight = 1.2
    stub.recovery_clearance_weight = 1.0
    stub.heading_search_radius_m = 0.5
    stub.recovery_turn_speed = 0.45
    stub.recovery_deg_per_s = 60.0
    stub.recovery_step_deg = 30.0
    stub.recovery_turn_s = 0.1
    stub.recovery_tilt_veto_deg = 8.0
    stub.front_near_m = 0.32
    stub.front_far_m = 0.85
    stub.front_half_width_m = 0.22
    stub.front_traversable_thresh = 0.28
    stub.front_min_free_ratio = 0.40
    stub.allow_reverse = True
    stub.use_vlm_recovery = False
    stub.use_recovery_scan = False
    stub.stats = types.SimpleNamespace(
        retrocesos=0, recoveries_por_mapa=0, recoveries_por_vlm=0,
        recoveries_ciegas=0, near_regime_activations=0, unstucks=0,
        escaneos_360=0,
    )
    stub._stop_requested = False

    stub.heading_est = types.SimpleNamespace(
        reset_track=lambda: None,
        heading=lambda: 0.0,
    )
    stub._consecutive_turns = 0
    stub._turn_sign_history = []
    stub._consecutive_empty = 0
    stub._commit_side = 0
    stub._commit_until = 0.0
    stub._consecutive_blocked = 3
    stub._plan_path_world = np.zeros((5, 2))
    stub._plan_pose = Pose(0.0, 0.0, 0.0)

    stub.forward_range = 2.0
    stub.side_range = 1.0
    stub.use_map = True

    class _MockOdometry:
        def __init__(self, pitch_deg: float, roll_deg: float):
            self.last_pitch = math.radians(pitch_deg)
            self.last_roll = math.radians(roll_deg)
            self.pose_val = Pose(0.0, 0.0, 0.0)

        @property
        def pose(self):
            return self.pose_val

        def update(self, _raw, *a, **kw):
            return self.pose_val

        def current_roll_pitch(self, *a, **kw):
            return (self.last_roll, self.last_pitch)

    stub.odometry = _MockOdometry(tilt_pitch_deg, tilt_roll_deg)

    sent: list[DriveCommand] = []
    stub.send = lambda cmd: sent.append(cmd)
    stub._sent = sent

    stub.client = types.SimpleNamespace(
        telemetry=lambda: types.SimpleNamespace(raw={}, latitude=0.0, longitude=0.0),
        front_frame=lambda: (np.zeros((8, 8, 3), dtype=np.uint8), 0.0),
    )

    stub.perception = types.SimpleNamespace(
        process=lambda _rgb, *a, **kw: types.SimpleNamespace(traversability=np.ones((16, 16), dtype=np.float32)),
    )

    # Vincular métodos reales de Bridge
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


class TestBilateralRecovery(unittest.TestCase):
    def test_left_clearance_prefers_goal_over_180(self):
        """(a) Si el lado izquierdo tiene clearance y la meta esta a la izquierda,
        debe elegir izquierda (-90°), NO 180° (aunque 180° tenga 100% de libre)."""
        stub = _build_recovery_stub()

        # Mapa:
        # Frente (x>0, |y|<0.3): bloqueado
        # Derecha (x=0, y>0): bloqueado parcialmente (libre < 50%)
        # Izquierda (x=0, y<0): libre (libre = 85%)
        # Atras (x<0): totalmente libre (libre = 100%)
        pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
        pmap.value[:] = 1.0
        pmap.conf[:] = 1.0

        # Bloqueo frontal
        for x in np.arange(0.05, 0.70, pmap.cfg.resolution_m_per_px):
            for y in np.arange(-0.35, 0.35, pmap.cfg.resolution_m_per_px):
                f, c = pmap.world_to_cell(float(x), float(y))
                if 0 <= f < pmap.n and 0 <= c < pmap.n:
                    pmap.value[f, c] = 0.0

        # Bloqueo derecha (y > 0.1)
        for x in np.arange(-0.5, 0.5, pmap.cfg.resolution_m_per_px):
            for y in np.arange(0.15, 0.70, pmap.cfg.resolution_m_per_px):
                f, c = pmap.world_to_cell(float(x), float(y))
                if 0 <= f < pmap.n and 0 <= c < pmap.n:
                    pmap.value[f, c] = 0.0

        stub.pmap = pmap

        # Meta ubicada a la izquierda (bearing relativo -45°)
        stub._goal_relative_bearing_deg = -45.0

        stub._sent.clear()
        stub._recover_informado()

        self.assertEqual(stub.stats.recoveries_por_mapa, 1)
        # Comprobar que los comandos de giro fueron hacia la izquierda (angular_sign=-1 => angular > 0)
        angulares = [c for c in stub._sent if c.angular != 0.0]
        self.assertGreater(len(angulares), 0)
        # Para heading_rel_deg = -90.0, copysign(speed, -90) es negativo, y * (-1.0) da positivo (> 0)
        self.assertTrue(all(c.angular > 0 for c in angulares),
                        f"Se esperaba giro a la izquierda (angular > 0), comandos: {angulares}")

    def test_fallback_to_180_when_no_lateral_clearance(self):
        """(b) Si ningun lado lateral ofrece clearance suficiente (ambos bloqueados),
        debe recurrir a 180° como ultimo recurso antes de caer a VLM/ciego."""
        stub = _build_recovery_stub()

        # Mapa: Frente, Izquierda y Derecha bloqueados. Solo atras esta libre.
        pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
        pmap.value[:] = 1.0
        pmap.conf[:] = 1.0

        # Bloquear frente y ambos laterales (forma de herradura / callejón)
        for x in np.arange(-0.35, 0.70, pmap.cfg.resolution_m_per_px):
            for y in np.arange(-0.70, 0.70, pmap.cfg.resolution_m_per_px):
                if x > 0.0 or abs(y) > 0.10:
                    f, c = pmap.world_to_cell(float(x), float(y))
                    if 0 <= f < pmap.n and 0 <= c < pmap.n:
                        pmap.value[f, c] = 0.0

        stub.pmap = pmap
        stub._goal_relative_bearing_deg = 0.0

        stub._sent.clear()
        stub._recover_informado()

        self.assertEqual(stub.stats.recoveries_por_mapa, 1)
        # Debe haber ejecutado giro de 180°
        turn_cmds = [c for c in stub._sent if "girando hacia +180" in c.reason]
        self.assertGreater(len(turn_cmds), 0, "Debió elegir 180° como último recurso")

    def test_lateral_partial_clearance_40pct_wins_over_180(self):
        """Valida que un lateral con clearance parcial típico de corrida real (~40%),
        pero alineado a la meta, sea seleccionado en vez de recurrir a 180° (88%).
        Con el umbral anterior de 50%, 40% era descartado y caía erróneamente a 180°."""
        stub = _build_recovery_stub()
        stub.recovery_min_libre_pct = 30.0

        pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
        pmap.value[:] = 1.0
        pmap.conf[:] = 1.0

        # Bloqueo frontal
        for x in np.arange(0.05, 0.70, pmap.cfg.resolution_m_per_px):
            for y in np.arange(-0.35, 0.35, pmap.cfg.resolution_m_per_px):
                f, c = pmap.world_to_cell(float(x), float(y))
                if 0 <= f < pmap.n and 0 <= c < pmap.n:
                    pmap.value[f, c] = 0.0

        # Bloqueo en lateral izquierdo parcial para dejar ~35-45% libre
        for x in np.arange(0.0, 0.50, pmap.cfg.resolution_m_per_px):
            for y in np.arange(-0.60, -0.15, pmap.cfg.resolution_m_per_px):
                f, c = pmap.world_to_cell(float(x), float(y))
                if 0 <= f < pmap.n and 0 <= c < pmap.n:
                    pmap.value[f, c] = 0.0

        # Bloqueo severo en lateral derecho (<20% libre)
        for x in np.arange(-0.4, 0.50, pmap.cfg.resolution_m_per_px):
            for y in np.arange(0.08, 0.60, pmap.cfg.resolution_m_per_px):
                f, c = pmap.world_to_cell(float(x), float(y))
                if 0 <= f < pmap.n and 0 <= c < pmap.n:
                    pmap.value[f, c] = 0.0

        stub.pmap = pmap
        stub._goal_relative_bearing_deg = -90.0

        stub._sent.clear()
        stub._recover_informado()

        self.assertEqual(stub.stats.recoveries_por_mapa, 1)
        angulares = [c for c in stub._sent if c.angular != 0.0]
        self.assertGreater(len(angulares), 0)
        self.assertTrue(all(c.angular > 0 for c in angulares),
                        f"Se esperaba giro a la izquierda (angular > 0), comandos: {angulares}")

    def test_reverse_angular_is_strictly_neutral(self):
        """(c) Durante la reversa en _retroceder(), el comando angular debe ser estrictamente
        0.0 (neutro/recto), sin que ningun controlador de seguimiento activo induzca virajes."""
        stub = _build_recovery_stub()
        # Inicializar con compromiso de giro hacia la izquierda
        stub._commit_side = -1
        stub._commit_until = time.time() + 10.0
        stub._plan_path_world = np.array([[0.0, 0.0], [-0.5, 0.5]])

        stub._sent.clear()
        stub._retroceder()

        # Todos los comandos de movimiento en retroceso deben tener angular == 0.0
        motion_cmds = [c for c in stub._sent if c.linear != 0.0]
        self.assertGreater(len(motion_cmds), 0, "Debió enviar comandos de retroceso")
        for cmd in motion_cmds:
            self.assertEqual(cmd.angular, 0.0, f"Comando con desvío angular durante reversa: {cmd}")
            self.assertLess(cmd.linear, 0.0, "La velocidad lineal debe ser negativa durante retroceso")

        self.assertEqual(stub._sent[-1].reason, "fin del retroceso")
        self.assertEqual(stub._sent[-1].linear, 0.0)
        self.assertEqual(stub._sent[-1].angular, 0.0)

    def test_recovery_endpoint_clears_all_residual_state(self):
        """(d) Confirmar que al finalizar el recovery no queda ningun setpoint o estado
        residual que pueda arrastrarse al siguiente ciclo de navegacion normal."""
        stub = _build_recovery_stub()

        # Ensuciar intencionalmente el estado interno pre-recovery
        stub._commit_side = -1
        stub._commit_until = time.time() + 15.0
        stub._consecutive_turns = 4
        stub._turn_sign_history = [-1.0, -1.0, -1.0, -1.0]
        stub._consecutive_empty = 3
        stub._consecutive_empty_recoveries = 2
        stub._consecutive_blocked = 5
        stub._plan_path_world = np.ones((10, 2))
        stub._plan_pose = Pose(1.0, 2.0, 0.5)

        # Ejecutar limpieza de endpoint directamente
        stub._reset_recovery_state()

        self.assertEqual(stub._commit_side, 0, "_commit_side debe quedar en 0")
        self.assertEqual(stub._commit_until, 0.0, "_commit_until debe quedar en 0.0")
        self.assertEqual(stub._consecutive_turns, 0, "_consecutive_turns debe resetearse a 0")
        self.assertEqual(len(stub._turn_sign_history), 0, "_turn_sign_history debe quedar vacío")
        self.assertEqual(stub._consecutive_empty, 0, "_consecutive_empty debe ser 0")
        self.assertEqual(stub._consecutive_empty_recoveries, 0, "_consecutive_empty_recoveries debe ser 0")
        self.assertEqual(stub._consecutive_blocked, 0, "_consecutive_blocked debe ser 0")
        self.assertIsNone(stub._plan_path_world, "_plan_path_world debe invalidarse (None)")
        self.assertIsNone(stub._plan_pose, "_plan_pose debe invalidarse (None)")

    def test_steep_slope_vetoes_180_and_cascades_safely(self):
        """(e) En pendiente peligrosa (pitch=10.0° >= 8.0°), 180° debe ser vetado incluso
        si los laterales están bloqueados, cayendo limpiamente a barrido ciego sin vuelco."""
        stub = _build_recovery_stub(tilt_pitch_deg=10.0, tilt_roll_deg=0.0)

        # Herradura donde solo 180° estaría libre, pero está en rampa
        pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
        pmap.value[:] = 1.0
        pmap.conf[:] = 1.0
        for x in np.arange(-0.35, 0.70, pmap.cfg.resolution_m_per_px):
            for y in np.arange(-0.70, 0.70, pmap.cfg.resolution_m_per_px):
                if x > 0.0 or abs(y) > 0.10:
                    f, c = pmap.world_to_cell(float(x), float(y))
                    if 0 <= f < pmap.n and 0 <= c < pmap.n:
                        pmap.value[f, c] = 0.0

        stub.pmap = pmap
        stub._sent.clear()
        stub._recover_informado()

        # 180° fue vetado por pendiente, y sin VLM cayó a barrido ciego
        self.assertEqual(stub.stats.recoveries_por_mapa, 0, "No debe tomar 180° por mapa en pendiente")
        self.assertEqual(stub.stats.recoveries_ciegas, 1, "Debe caer a barrido ciego de forma segura")
        turn_180 = [c for c in stub._sent if "180" in c.reason]
        self.assertEqual(len(turn_180), 0, "No se debió enviar ningún giro de 180°")


if __name__ == "__main__":
    unittest.main()
