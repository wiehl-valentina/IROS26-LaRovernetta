"""sana — mission v2: decoupled perception/control on top of rover_traversability.

Two threads plus one pure state machine. See the "Mission v2" section of the
README for the design, and config.py for the field-proven motion values ported
from auto-navigation-mini.
"""

from .config import MissionV2Config

__all__ = ["MissionV2Config", "MissionV2Runner"]


def __getattr__(name):
    # Lazy import: MissionV2Runner pulls in the predictor (and torch with it)
    # only when constructed, mirroring the vendored package's behavior.
    if name == "MissionV2Runner":
        from .runner import MissionV2Runner

        return MissionV2Runner
    raise AttributeError(f"module 'sana' has no attribute {name!r}")
