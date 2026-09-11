"""Fixtures for the sana test suite.

Every test here runs WITHOUT torch, the checkpoint, or network — same
discipline as traversability/tests. The actual fakes live in helpers.py
(a uniquely-named module, so both test suites can share one pytest run).
"""

from __future__ import annotations

import pytest

from helpers import FakePredictor, FakeSession


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture
def fake_predictor():
    return FakePredictor()
