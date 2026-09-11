"""State shared between the perception, control and viewer threads.

One lock, one stop event, immutable snapshots swapped by reference — that is
the whole synchronization surface of mission v2. Writers build a snapshot
outside the lock and swap it in; readers grab the reference and use it
lock-free (snapshots are frozen and their arrays are never mutated after
publication).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MaskSnapshot:
    """One perception result. Written by the perception thread only."""

    mask: object                 # HxW float32 numpy array, never mutated
    overlay_jpeg: bytes          # pre-encoded JPEG for the viewer
    mono_ts: float               # time.monotonic() at publication
    inference_s: float
    seq: int


class SharedState:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._mask: Optional[MaskSnapshot] = None
        self._decision: Optional[dict] = None

    # ------------------------------------------------------------- perception

    def publish_mask(self, snap: MaskSnapshot) -> None:
        with self._lock:
            self._mask = snap

    def latest_mask(self) -> Optional[MaskSnapshot]:
        with self._lock:
            return self._mask

    # ---------------------------------------------------------------- control

    def publish_decision(self, record: dict) -> None:
        with self._lock:
            self._decision = record

    def latest_decision(self) -> Optional[dict]:
        with self._lock:
            return self._decision
