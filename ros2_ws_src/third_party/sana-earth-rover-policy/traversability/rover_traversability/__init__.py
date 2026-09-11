"""rover-traversability: SAM-TP traversability perception for the Earth Rover Mini+.

Importing this package never pulls in torch — everything except
``TraversabilityPredictor`` works on a machine with only numpy/pillow/requests.
The predictor (and torch/sam2 with it) loads lazily on first access.
"""

import os

# Must be set before torch is imported anywhere in the process: SAM2's Hiera
# backbone uses ops without an MPS kernel, and without this flag inference
# crashes on Apple Silicon instead of falling back to CPU for those ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .calibration import load_camera_dist, load_camera_K, load_T_base_camera
from .client import CommandResult, RoverClient
from .geo import HeadingEstimator, gps_bearing_and_distance, wrap_angle_deg
from .images import ImageDecodeError, PayloadError, to_rgb
from .mission import MissionResult, MissionRunner
from .policy import CommandDecision, PolicyConfig, suggest_command
from .strategy import TraversabilityStrategy
from .weights import (
    CheckpointNotFoundError,
    SamNotInstalledError,
    resolve_checkpoint,
    resolve_config,
)

_LAZY_PREDICTOR_EXPORTS = (
    "TraversabilityPredictor",
    "TraversabilityResult",
    "CheckpointMismatchError",
)

__all__ = [
    "CommandDecision",
    "CommandResult",
    "CheckpointNotFoundError",
    "HeadingEstimator",
    "ImageDecodeError",
    "MissionResult",
    "MissionRunner",
    "PayloadError",
    "PolicyConfig",
    "RoverClient",
    "SamNotInstalledError",
    "TraversabilityStrategy",
    "gps_bearing_and_distance",
    "load_camera_K",
    "load_camera_dist",
    "load_T_base_camera",
    "resolve_checkpoint",
    "resolve_config",
    "suggest_command",
    "to_rgb",
    "wrap_angle_deg",
    *_LAZY_PREDICTOR_EXPORTS,
]


def __getattr__(name):  # PEP 562: lazy import so `import rover_traversability` stays torch-free
    if name in _LAZY_PREDICTOR_EXPORTS:
        from . import predictor as _predictor

        return getattr(_predictor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
