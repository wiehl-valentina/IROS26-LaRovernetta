"""Pruebas unitarias para logging de diagnóstico (Fase 3: DIAG 1 y DIAG 2).

Verifica:
  1. DIAG 1: Al entrar a _retroceso_y_recover se imprime la línea de diagnóstico
     con clearance, mapa (-90..+90) y mitades del BEV instantáneo, y se guardan
     los artefactos RGB/BEV en debug_dir si está configurado.
  2. DIAG 2: Con log_tilt=False (por defecto), no se genera archivo CSV de tilt.
  3. DIAG 2: Con log_tilt=True, se genera tilt_log.csv en debug_dir con encabezado
     timestamp,pitch_deg,roll_deg,linear,angular,front_is_blocked y se registran
     las muestras correctamente a ~5 Hz.
"""

from __future__ import annotations

import csv
import math
import time
import types
from pathlib import Path
import numpy as np
import pytest

from .bridge import Bridge
from .navigation import DriveCommand, PathFollower
from .odometry import Pose
from .persistent_map import MapConfig, PersistentMap


def _build_diag_stub(debug_dir: Path | None = None, log_tilt: bool = False) -> types.SimpleNamespace:
    stub = types.SimpleNamespace()
    stub.resolution = 0.03
    stub.forward_range = 2.0
    stub.side_range = 2.0
    stub.heading_search_radius_m = 1.5
    stub.front_near_m = 0.40
    stub.front_half_width_m = 0.22
    stub.front_traversable_thresh = 0.26
    stub.front_min_free_ratio = 0.35
    stub.retroceso_max_m = 0.4
    stub.retroceso_paso_m = 0.2
    stub.retroceso_linear = -0.18
    stub.retroceso_min_libre_pct = 55.0
    stub.retroceso_min_cobertura_pct = 30.0
    stub.allow_reverse = False
    stub.dry_run = True
    stub._consecutive_blocked = 4
    stub._stop_requested = False
    stub.debug_dir = debug_dir
    stub.log_tilt = log_tilt
    stub._tilt_csv_file = None
    stub._tilt_csv_writer = None
    stub._tilt_csv_path = None
    stub._last_tilt_log_time = 0.0
    stub._last_front_blocked = False

    if log_tilt and debug_dir:
        stub._tilt_csv_path = debug_dir / "tilt_log.csv"
        stub._tilt_csv_file = open(stub._tilt_csv_path, "w", newline="", buffering=1)
        stub._tilt_csv_writer = csv.writer(stub._tilt_csv_file)
        stub._tilt_csv_writer.writerow(["timestamp", "pitch_deg", "roll_deg", "linear", "angular", "front_is_blocked"])

    stub.stats = types.SimpleNamespace(iterations=12, near_regime_activations=0, retrocesos=0)
    stub.pmap = PersistentMap(MapConfig(size_m=6.0, resolution_m_per_px=0.03))
    stub.pmap.value[:] = 1.0
    stub.pmap.conf[:] = 1.0

    class _MockOdo:
        pose = Pose(0.0, 0.0, 0.0)
        last_pitch = math.radians(3.5)
        last_roll = math.radians(-2.1)
        def current_roll_pitch(self, *a, **kw):
            return self.last_roll, self.last_pitch

    stub.odometry = _MockOdo()

    sent: list[DriveCommand] = []
    stub._sent = sent

    # Bind methods
    for name in ("_map_free_and_coverage", "_is_tilt_too_steep_for_recovery",
                 "_get_estimated_tilt_deg", "_retroceso_y_recover",
                 "_maybe_log_tilt", "_close_tilt_log", "send"):
        if hasattr(Bridge, name):
            setattr(stub, name, types.MethodType(getattr(Bridge, name), stub))

    stub._recover_informado = lambda *a, **kw: None
    return stub


def test_diag1_retroceso_y_recover_logging_and_debug(tmp_path, capsys):
    """Verifica que _retroceso_y_recover emite el log de diagnóstico DIAG 1 y guarda RGB/BEV."""
    stub = _build_diag_stub(debug_dir=tmp_path)
    bev = np.ones((60, 60), dtype=np.float32)
    # Lado izquierdo bloqueado en franja cercana, lado derecho libre
    bev[:, :30] = 0.1
    bev[:, 30:] = 0.9
    rgb = np.full((30, 30, 3), 120, dtype=np.uint8)

    stub._retroceso_y_recover(bev, rgb=rgb)

    out = capsys.readouterr().out
    assert "REGIMEN CERCANO:" in out
    assert "mapa: [" in out
    assert "-90°" in out and "+90°" in out
    assert "bev_fresco[0.4-1.25m]: izq=" in out

    # Verificar guardado en debug_dir
    dumped_rgb = tmp_path / "00012_rgb.jpg"
    dumped_bev = tmp_path / "00012_bev.npy"
    assert dumped_rgb.exists(), "Debe guardar el frame RGB en debug_dir"
    assert dumped_bev.exists(), "Debe guardar el BEV en debug_dir"
    loaded_bev = np.load(dumped_bev)
    assert np.array_equal(loaded_bev, bev)


def test_diag2_tilt_logging_disabled_by_default(tmp_path):
    """Con log_tilt=False, no se crea el archivo tilt_log.csv."""
    stub = _build_diag_stub(debug_dir=tmp_path, log_tilt=False)
    stub.send(DriveCommand(0.25, 0.10, "test drive"))
    assert not (tmp_path / "tilt_log.csv").exists()


def test_diag2_tilt_logging_enabled(tmp_path):
    """Con log_tilt=True, se crea tilt_log.csv y se escriben las muestras con pitch/roll."""
    stub = _build_diag_stub(debug_dir=tmp_path, log_tilt=True)
    stub._last_front_blocked = True
    stub.send(DriveCommand(0.30, -0.15, "test drive 1"))

    # Esperar > 0.18s para cumplir con rate limit de ~5 Hz
    time.sleep(0.20)
    stub._last_front_blocked = False
    stub.send(DriveCommand(0.00, 0.00, "test stop"))
    stub._close_tilt_log()

    csv_path = tmp_path / "tilt_log.csv"
    assert csv_path.exists()
    rows = list(csv.reader(csv_path.open()))
    assert len(rows) == 3  # header + 2 filas
    assert rows[0] == ["timestamp", "pitch_deg", "roll_deg", "linear", "angular", "front_is_blocked"]

    # Fila 1: linear=0.30, angular=-0.15, front_is_blocked=True
    assert rows[1][1] == "3.50"  # pitch_deg
    assert rows[1][2] == "-2.10"  # roll_deg
    assert rows[1][3] == "0.30"
    assert rows[1][4] == "-0.15"
    assert rows[1][5] == "True"

    # Fila 2: linear=0.00, angular=0.00, front_is_blocked=False
    assert rows[2][3] == "0.00"
    assert rows[2][4] == "0.00"
    assert rows[2][5] == "False"
