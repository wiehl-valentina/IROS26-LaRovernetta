"""Shared test helpers for the sana suite.

Kept in a uniquely-named module (NOT conftest.py) so this suite and the
vendored traversability/tests can be collected in the same pytest run —
two conftest modules cannot both be imported by name.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# missions/level2.py does the same insert; tests import `sana` the same way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ------------------------------------------------------------------ mask makers

def make_mask(h=120, w=160, value=1.0):
    return np.full((h, w), value, dtype=np.float32)


def half_open_mask(side: str, h=120, w=160):
    """Left or right half drivable, other half blocked."""
    m = np.zeros((h, w), dtype=np.float32)
    if side == "left":
        m[:, : w // 2] = 1.0
    else:
        m[:, w // 2 :] = 1.0
    return m


# -------------------------------------------------------------- decision stub

@dataclass(frozen=True)
class DecisionStub:
    """Duck-typed stand-in for rover_traversability.policy.CommandDecision."""

    linear: float = 0.22
    angular: float = 0.0
    stop: bool = False
    reason: str = "forward"
    corridor_scores: tuple = ()
    best_corridor: int = -1


# ------------------------------------------------------------- fake predictor

@dataclass
class FakeResult:
    mask: np.ndarray
    logits: np.ndarray
    overlay: np.ndarray
    image: np.ndarray
    device: str = "fake"
    inference_s: float = 0.01


class FakePredictor:
    """Returns a fixed mask; records what it was asked to predict."""

    def __init__(self, mask=None):
        self.mask = mask if mask is not None else make_mask()
        self.calls: list = []

    def predict(self, payload):
        self.calls.append(payload)
        h, w = self.mask.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        return FakeResult(
            mask=self.mask,
            logits=self.mask * 20 - 10,
            overlay=img,
            image=img,
        )


# --------------------------------------------------------------- fake session

@dataclass
class FakeResponse:
    status_code: int = 200
    _json: dict | None = None
    text: str = ""

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """requests.Session stand-in: programmable per-endpoint responses.

    Thread-safe on purpose — test_runner_threads exercises real threads.
    A queue with a single response repeats it forever; with more than one it
    pops until one remains.
    """

    def __init__(self):
        self.responses: dict[str, list] = {}
        self.requests: list = []               # (method, url, json_payload, thread_ident)
        self._lock = threading.Lock()

    def queue(self, endpoint: str, *responses):
        self.responses.setdefault(endpoint, []).extend(responses)

    def _pop(self, url: str) -> FakeResponse:
        for endpoint, q in self.responses.items():
            if url.endswith(endpoint) and q:
                return q.pop(0) if len(q) > 1 else q[0]
        return FakeResponse(status_code=404, text="not queued")

    def get(self, url, timeout=None):
        with self._lock:
            self.requests.append(("GET", url, None, threading.get_ident()))
            return self._pop(url)

    def post(self, url, json=None, timeout=None):
        with self._lock:
            self.requests.append(("POST", url, json, threading.get_ident()))
            return self._pop(url)

    def calls_to(self, endpoint: str, method: str | None = None):
        with self._lock:
            return [r for r in self.requests
                    if r[1].endswith(endpoint) and (method is None or r[0] == method)]
