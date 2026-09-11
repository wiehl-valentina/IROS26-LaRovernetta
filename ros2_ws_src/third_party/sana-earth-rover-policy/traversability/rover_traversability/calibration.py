"""Camera calibration for the Earth Rover Mini+ front camera.

These values are NOT used by the image-space steering policy in this package —
they are shipped so the next stage of your roadmap (projecting the
traversability mask to a bird's-eye-view costmap and running the vendored
``genie_path_planner`` on it) has correct rover-specific numbers. The
``genie/configs/stretch_path_planner.yaml`` in this repo carries Hello Robot
Stretch intrinsics, which are wrong for the rover.

Conventions:
- ``K``: 3x3 pinhole intrinsics for the 1024x576 front frame (derived from the
  camera's field of view; distortion is treated as negligible).
- ``dist``: OpenCV 5-term distortion coefficients (all zero — undistortion is
  a no-op for this camera model).
- ``T_base_camera``: 4x4 pose of the camera in the rover base frame — the
  camera sits ~14 cm above ground, ~11 cm forward of base origin, pitched
  ~8 degrees down.
"""

from __future__ import annotations

import io
from importlib import resources

import numpy as np


def _load_npy(name: str) -> np.ndarray:
    data = resources.files("rover_traversability").joinpath(f"data/{name}").read_bytes()
    return np.load(io.BytesIO(data))


def load_camera_K() -> np.ndarray:
    """3x3 float64 pinhole intrinsics for the 1024x576 Mini+ front frame."""
    return _load_npy("mini_camera_K.npy")


def load_camera_dist() -> np.ndarray:
    """(5,) float64 OpenCV distortion coefficients (zeros for the Mini+)."""
    return _load_npy("mini_camera_dist.npy")


def load_T_base_camera() -> np.ndarray:
    """4x4 float64 camera pose in the rover base frame."""
    return _load_npy("mini_T_base_camera.npy")


def horizontal_fov_deg(frame_width_px: int = 1024) -> float:
    """Horizontal field of view implied by K, for a frame of the given width."""
    K = load_camera_K()
    fx = float(K[0, 0])
    import math

    return math.degrees(2.0 * math.atan((frame_width_px / 2.0) / fx))
