"""Tests for the live viewer: bind an ephemeral port, fetch the three routes."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from sana.shared import MaskSnapshot, SharedState
from sana.viewer import start_viewer


@pytest.fixture
def served():
    shared = SharedState()
    server = start_viewer(shared, port=0)  # ephemeral port
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield shared, base
    server.shutdown()


def fetch(url):
    with urllib.request.urlopen(url, timeout=2) as resp:
        return resp.status, resp.headers.get("Content-Type"), resp.read()


def test_index_serves_html(served):
    _, base = served
    status, ctype, body = fetch(base + "/")
    assert status == 200 and "text/html" in ctype
    assert b"overlay.jpg" in body and b"decision.json" in body


def test_overlay_404_before_first_frame_then_serves_bytes(served):
    shared, base = served
    with pytest.raises(urllib.error.HTTPError) as err:
        fetch(base + "/overlay.jpg")
    assert err.value.code == 404

    shared.publish_mask(MaskSnapshot(
        mask=None, overlay_jpeg=b"\xff\xd8fakejpeg", mono_ts=0.0,
        inference_s=0.02, seq=1,
    ))
    status, ctype, body = fetch(base + "/overlay.jpg")
    assert status == 200 and ctype == "image/jpeg"
    assert body == b"\xff\xd8fakejpeg"


def test_decision_json_roundtrips(served):
    shared, base = served
    record = {"state": "pursue", "linear": 0.25, "tick": 7}
    shared.publish_decision(record)
    status, ctype, body = fetch(base + "/decision.json")
    assert status == 200 and "application/json" in ctype
    assert json.loads(body) == record


def test_unknown_route_404(served):
    _, base = served
    with pytest.raises(urllib.error.HTTPError) as err:
        fetch(base + "/etc/passwd")
    assert err.value.code == 404
