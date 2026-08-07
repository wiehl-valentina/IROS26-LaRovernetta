"""Shared fixtures. Every test here runs WITHOUT torch, the checkpoint, or network."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest


# ----------------------------------------------------------------- mask makers

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


def bimodal_mask(h=120, w=160):
    """Openings at far left and far right, wall in the middle."""
    m = np.zeros((h, w), dtype=np.float32)
    m[:, : w // 5] = 1.0
    m[:, -w // 5 :] = 1.0
    return m


# --------------------------------------------------------------- fake predictor

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


# ------------------------------------------------------------------ fake session

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
    """requests.Session stand-in: programmable per-endpoint responses."""

    def __init__(self):
        self.responses: dict[str, list] = {}   # endpoint suffix -> queue of FakeResponse
        self.requests: list = []               # (method, url, json_payload)

    def queue(self, endpoint: str, *responses):
        self.responses.setdefault(endpoint, []).extend(responses)

    def _pop(self, url: str) -> FakeResponse:
        for endpoint, queue in self.responses.items():
            if url.endswith(endpoint) and queue:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        return FakeResponse(status_code=404, text="not queued")

    def get(self, url, timeout=None):
        self.requests.append(("GET", url, None))
        return self._pop(url)

    def post(self, url, json=None, timeout=None):
        self.requests.append(("POST", url, json))
        return self._pop(url)


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture
def fake_predictor():
    return FakePredictor()
