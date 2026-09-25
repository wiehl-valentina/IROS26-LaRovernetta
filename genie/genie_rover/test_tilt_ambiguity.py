"""Pruebas unitarias offline del gate de ambiguedad de tilt (Paso 1).

Verifica:
(a) pitch=0: el score que llega a project_score_to_bev es identico al de entrada.
(b) pitch=±8°: una fila de pixeles NO transitables cerca del horizonte (que en una de las hipotesis
    cae a >2.5 m o es invalida) queda NaN; verificado con ambos signos de pitch (+8° y -8°).
(c) pitch=±8°: una franja NO transitable cuyo rayo cae a ~0.5 m en todas las hipotesis sigue
    siendo obstaculo en el BEV (no se anula); verificado con ambos signos de pitch (+8° y -8°).
(d) Pixeles transitables nunca se vuelven NaN, incluso en zonas de horizonte ambiguo.
(e) Por debajo del umbral tilt_ambiguity_min_deg (1.5°), el gate permanece inactivo.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import yaml

from genie_rover.perception import PerceptionPipeline


def _build_test_pipeline(downscale: int = 1) -> PerceptionPipeline:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "frodobot_rover.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    pipe = PerceptionPipeline.__new__(PerceptionPipeline)
    cam = cfg["camera"]
    proj = cfg["projection"]

    pipe.camera_k = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    pipe.dist = None
    pipe.image_size = tuple(cam["image_size"])
    pipe._rectifiers = {}
    pipe._warned_sizes = set()
    pipe.base_camera_pose = np.asarray(cam["pose"], dtype=np.float64).reshape(4, 4)
    pipe.camera_pose = pipe.base_camera_pose.copy()
    pipe.tilt_threshold_rad = math.radians(float(cam.get("tilt_threshold_deg", 0.5)))
    pipe._cached_roll = None
    pipe._cached_pitch = None

    # Parametros del gate de ambiguedad
    pipe.tilt_ambiguity_enabled = bool(cam.get("tilt_ambiguity_enabled", True))
    pipe.tilt_ambiguity_min_deg = float(cam.get("tilt_ambiguity_min_deg", 1.5))
    pipe.tilt_ambiguity_min_rad = math.radians(pipe.tilt_ambiguity_min_deg)
    pipe.tilt_ambiguity_obstacle_thresh = float(cam.get("tilt_ambiguity_obstacle_thresh", 0.5))

    pipe.ground_z = float(proj.get("ground_z", 0.0))
    pipe.resolution = float(proj["resolution_m_per_px"])
    pipe.forward_range = float(proj["forward_range_m"])
    pipe.side_range = float(proj["side_range_m"])
    pipe.max_ray = float(proj.get("max_ray_distance_m", 2.5))
    pipe.projection_downscale = downscale
    return pipe


class DummyRunner:
    def __init__(self, score: np.ndarray):
        self._score = score

    def traversability(self, img: np.ndarray) -> np.ndarray:
        return self._score.copy()


def test_pitch_zero_identical():
    """(a) pitch=0: el score que llega a project_score_to_bev es idéntico al de entrada."""
    pipe = _build_test_pipeline(downscale=1)
    h, w = pipe.image_size[1], pipe.image_size[0]
    score_in = np.random.RandomState(42).uniform(0.0, 1.0, size=(h, w)).astype(np.float32)

    pipe.runner = DummyRunner(score_in)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    res = pipe.process(rgb, roll_rad=0.0, pitch_rad=0.0)

    assert res.stats["tilt_ambiguo_px"] == 0, "No debe anular ningún pixel con pitch=0"
    assert math.isclose(res.stats["distancia_confiable_m"], pipe.max_ray, rel_tol=1e-5), \
        "Con pitch=0 distancia_confiable_m debe ser max_ray"
    assert np.array_equal(res.image_traversability, score_in)


@pytest.mark.parametrize("pitch_deg", [8.0, -8.0])
def test_pitch_near_horizon_nan(pitch_deg: float):
    """(b) pitch=±8°: una fila de píxeles NO transitables cerca del horizonte queda NaN;

    con ambos signos de pitch (+8° y -8°).
    """
    pipe = _build_test_pipeline(downscale=1)
    h, w = pipe.image_size[1], pipe.image_size[0]

    # Score base transitable (1.0), excepto una fila en y=500 (cerca del horizonte)
    score_in = np.ones((h, w), dtype=np.float32)
    score_in[500, :] = 0.0  # Obstáculo candidato

    pipe.runner = DummyRunner(score_in)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    res = pipe.process(rgb, roll_rad=0.0, pitch_rad=math.radians(pitch_deg))

    # La fila y=500 cae a >2.5m o es inválida en una de las 3 hipótesis (A, B1 o B2)
    assert res.stats["tilt_ambiguo_px"] >= w, \
        f"La fila y=500 debía anularse para pitch={pitch_deg}°, pero tilt_ambiguo_px={res.stats['tilt_ambiguo_px']}"

    # Distancia confiable para 8° debe ser ~0.74 m
    dconf = res.stats["distancia_confiable_m"]
    assert 0.70 <= dconf <= 0.78, f"dconf={dconf} fuera de rango esperado ~0.74 m"


@pytest.mark.parametrize("pitch_deg", [8.0, -8.0])
def test_near_obstacle_persists_in_bev(pitch_deg: float):
    """(c) pitch=±8°: una franja NO transitable cuyo rayo cae a ~0.5 m en ambas hipótesis

    sigue siendo obstáculo en el BEV (no se anula ni se vuelve NaN).
    """
    pipe = _build_test_pipeline(downscale=1)
    h, w = pipe.image_size[1], pipe.image_size[0]

    score_in = np.ones((h, w), dtype=np.float32)
    # y in [760, 780] corresponde a distancias entre 0.31m y 1.16m en las 3 hipótesis (todas < max_ray)
    score_in[760:780, :] = 0.0

    pipe.runner = DummyRunner(score_in)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    res = pipe.process(rgb, roll_rad=0.0, pitch_rad=math.radians(pitch_deg))

    # Los pixeles a ~0.5m no deben anularse
    assert res.stats["tilt_ambiguo_px"] == 0, \
        f"Pixeles a ~0.5m no debían anularse para pitch={pitch_deg}°, pero se anularon {res.stats['tilt_ambiguo_px']}"

    # En el BEV proyectado debe registrarse la presencia del obstáculo
    bev_obstaculos = np.sum((res.traversability >= 0.0) & (res.traversability < 0.5))
    assert bev_obstaculos > 0, f"El obstáculo cercano a ~0.5m no apareció en el BEV para pitch={pitch_deg}°"


@pytest.mark.parametrize("pitch_deg", [8.0, -8.0])
def test_traversable_pixels_never_nan(pitch_deg: float):
    """(d) píxeles transitables nunca se vuelven NaN, incluso en la zona ambigua del horizonte."""
    pipe = _build_test_pipeline(downscale=1)
    h, w = pipe.image_size[1], pipe.image_size[0]

    # Todo el campo visual es transitable
    score_in = np.full((h, w), 0.9, dtype=np.float32)

    pipe.runner = DummyRunner(score_in)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    res = pipe.process(rgb, roll_rad=0.0, pitch_rad=math.radians(pitch_deg))

    assert res.stats["tilt_ambiguo_px"] == 0, "No debe anular píxeles transitables (score >= 0.5)"


def test_gate_inactive_below_min_deg():
    """(e) Verificación de umbral mínimo: por debajo de tilt_ambiguity_min_deg (1.5°), el gate no actúa."""
    pipe = _build_test_pipeline(downscale=1)
    h, w = pipe.image_size[1], pipe.image_size[0]

    score_in = np.zeros((h, w), dtype=np.float32)  # Todo obstáculo
    pipe.runner = DummyRunner(score_in)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # pitch = 1.0° < 1.5°
    res = pipe.process(rgb, roll_rad=0.0, pitch_rad=math.radians(1.0))
    assert res.stats["tilt_ambiguo_px"] == 0
    assert math.isclose(res.stats["distancia_confiable_m"], pipe.max_ray, rel_tol=1e-5)
