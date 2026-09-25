"""Tests para la exclusión del rumbo 0° en recovery y chequeo de clearance en _unstick (Fase 2).

Verifica:
  (a) Con excluir_frente=True y un mapa donde 0° tiene el mejor score, el candidato elegido NO es 0°.
  (b) _unstick con un BEV cuya franja frontal está sin observar (-1.0) cancela el avance forzado y
      no manda ningún DriveCommand con linear > 0.
"""

from __future__ import annotations

import types
import numpy as np
import pytest

from .bridge import Bridge
from .navigation import DriveCommand, PathFollower
from .odometry import Pose
from .persistent_map import MapConfig, PersistentMap


def _build_recovery_stub(pmap: PersistentMap) -> types.SimpleNamespace:
    stub = types.SimpleNamespace()
    stub.resolution = 0.03
    stub.forward_range = 2.0
    stub.side_range = 2.0
    stub.follower = PathFollower(angular_sign=-1.0, turn_speed=0.35,
                                 max_linear=0.30, max_angular=0.45)
    stub.unstick_forward_s = 0.5
    stub.unstick_min_clearance_m = 0.9  # ASUMIDO
    stub.front_near_m = 0.40
    stub.front_half_width_m = 0.22
    stub.front_traversable_thresh = 0.26
    stub.front_min_free_ratio = 0.35
    stub.recovery_headings_deg = [0.0, 45.0, -45.0, 90.0, -90.0, 180.0]
    stub.recovery_min_cobertura_pct = 25.0
    stub.recovery_min_libre_pct = 30.0
    stub.recovery_goal_weight = 1.2
    stub.recovery_clearance_weight = 1.0
    stub.heading_search_radius_m = 1.5
    stub.use_map = True
    stub.use_vlm_recovery = False
    stub.use_recovery_scan = False
    stub._stop_requested = False
    stub.pmap = pmap

    stub.stats = types.SimpleNamespace(
        unstucks=0, recoveries_por_mapa=0, recoveries_por_vlm=0,
        empty_plans=0, stops=0, blocked=0, near_regime_activations=0
    )

    pose_holder = {"pose": Pose(0.0, 0.0, 0.0)}

    class _Odo:
        @property
        def pose(self):
            return pose_holder["pose"]

        def update(self, _raw, *a, **kw):
            return pose_holder["pose"]

        def current_roll_pitch(self, *a, **kw):
            return None

    stub.odometry = _Odo()

    sent: list[DriveCommand] = []
    stub.send = lambda cmd: sent.append(cmd)
    stub._sent = sent

    class _Client:
        def telemetry(self):
            return types.SimpleNamespace(raw={})

        def front_frame(self):
            return np.zeros((8, 8, 3), dtype=np.uint8), 0.0

    stub.client = _Client()

    class _Perception:
        def process(self, _rgb, *a, **kw):
            # Por defecto, BEV despejado
            return types.SimpleNamespace(
                traversability=np.ones((240, 240), dtype=np.float32),
                observed=np.ones((240, 240), dtype=bool),
            )

    stub.perception = _Perception()
    stub.heading_est = types.SimpleNamespace(reset_track=lambda: None)
    stub._consecutive_turns = 0
    stub._turn_sign_history = []
    stub._consecutive_empty_recoveries = 0
    stub._last_scan_pose = None
    stub._last_scan_time = 0.0
    stub._scan_count_at_stuck = 0

    # Vincular métodos reales de Bridge
    for name in ("_map_free_and_coverage", "_evaluar_candidatos_recovery_mapa",
                 "_is_front_blocked", "_unstick", "_safe_reset_recovery_state",
                 "_is_tilt_too_steep_for_recovery", "_get_goal_relative_bearing_deg",
                 "_girar_hacia", "_recover_informado"):
        if hasattr(Bridge, name):
            setattr(stub, name, types.MethodType(getattr(Bridge, name), stub))

    return stub


def test_excluir_frente_no_elige_cero():
    """(a) Con excluir_frente=True y un mapa donde 0° tiene el mejor score,
    el elegido NO es 0°."""
    # Mapa totalmente libre donde 0° tiene goal_align máximo (meta al frente)
    pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    pmap.value[:] = 1.0
    pmap.conf[:] = 1.0

    stub = _build_recovery_stub(pmap)
    stub._get_goal_relative_bearing_deg = lambda: 0.0  # Meta justo al frente

    # 1. Sin excluir frente: 0° debe ganar por score
    cand_nominal = stub._evaluar_candidatos_recovery_mapa(veto_tilt=False, razon_tilt="", excluir_frente=False)
    assert cand_nominal is not None
    assert cand_nominal["heading"] == 0.0, f"Sin excluir frente debía ganar 0°, dio {cand_nominal['heading']}"

    # 2. Con excluir_frente=True: 0° queda descartado y se elige un rumbo alternativo
    cand_excluido = stub._evaluar_candidatos_recovery_mapa(veto_tilt=False, razon_tilt="", excluir_frente=True)
    assert cand_excluido is not None
    assert cand_excluido["heading"] != 0.0, f"Con excluir_frente=True NO debe elegir 0°, dio {cand_excluido['heading']}"
    assert abs(cand_excluido["heading"]) in (45.0, 90.0, 180.0)


def test_unstick_cancela_si_frente_sin_observar():
    """(b) _unstick con un BEV cuya franja frontal está sin observar (-1)
    no manda ningún DriveCommand con linear > 0."""
    pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    stub = _build_recovery_stub(pmap)

    # Simular percepción que devuelve celdas sin observar (-1.0) en el BEV instantáneo
    bev_unobserved = -np.ones((240, 240), dtype=np.float32)

    class _PerceptionUnobserved:
        def process(self, _rgb, *a, **kw):
            return types.SimpleNamespace(
                traversability=bev_unobserved,
                observed=np.zeros((240, 240), dtype=bool),
            )

    stub.perception = _PerceptionUnobserved()

    stub._sent.clear()
    stub._unstick()

    # Verificar que no se envió ningún comando de avance (linear > 0)
    avances_enviados = [cmd for cmd in stub._sent if cmd.linear > 0.0]
    assert len(avances_enviados) == 0, (
        f"Se enviaron comandos de avance forzado con frente sin observar: {avances_enviados}"
    )


def test_unstick_cancela_si_clearance_bajo():
    """Verifica que si el clearance frontal es menor a unstick_min_clearance_m (0.9m),
    _unstick cancela el avance forzado inmediatamente."""
    pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    stub = _build_recovery_stub(pmap)

    # BEV con obstáculo a 0.5 m (bloqueado entre 0.40m y 0.60m)
    # clearance resultante será ~0.40 m < 0.90 m
    bev_blocked = np.ones((240, 240), dtype=np.float32)
    # Colocar no-transitable a ~0.5m del rover
    r_obstacle = 240 - 1 - int(0.50 / stub.resolution)
    bev_blocked[r_obstacle - 5 : r_obstacle + 5, :] = 0.0

    class _PerceptionBlocked:
        def process(self, _rgb, *a, **kw):
            return types.SimpleNamespace(
                traversability=bev_blocked,
                observed=np.ones((240, 240), dtype=bool),
            )

    stub.perception = _PerceptionBlocked()

    stub._sent.clear()
    stub._unstick()

    avances_enviados = [cmd for cmd in stub._sent if cmd.linear > 0.0]
    assert len(avances_enviados) == 0, (
        f"Se enviaron comandos de avance forzado con clearance bajo (0.5m < 0.9m): {avances_enviados}"
    )
