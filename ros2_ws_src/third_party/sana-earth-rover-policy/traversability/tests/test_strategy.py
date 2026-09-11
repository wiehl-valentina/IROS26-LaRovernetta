import inspect

import numpy as np

from rover_traversability.client import RoverClient
from rover_traversability.strategy import TraversabilityStrategy

from conftest import FakePredictor, FakeResponse, FakeSession, make_mask


def _strategy(session, predictor, **kwargs):
    client = RoverClient(base_url="http://localhost:8000", session=session)
    return TraversabilityStrategy(client=client, predictor=predictor, **kwargs)


def _control_posts(session):
    return [r for r in session.requests if r[0] == "POST" and r[1].endswith("/control")]


def test_dry_run_sends_nothing(fake_session, capsys):
    s = _strategy(fake_session, FakePredictor(make_mask(value=1.0)), drive=False)
    s.analyze("ignored-by-fake")
    assert _control_posts(fake_session) == []
    assert "[dry-run]" in capsys.readouterr().out


def test_drive_sends_policy_command(fake_session):
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    s = _strategy(fake_session, FakePredictor(make_mask(value=1.0)), drive=True)
    s.analyze("frame")
    posts = _control_posts(fake_session)
    assert len(posts) == 1
    cmd = posts[0][2]["command"]
    assert cmd["linear"] > 0


def test_drive_blocked_actively_stops(fake_session):
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    s = _strategy(fake_session, FakePredictor(make_mask(value=0.0)), drive=True)
    s.analyze("frame")
    posts = _control_posts(fake_session)
    assert posts, "blocked decision must SEND a zero command, not go silent"
    assert posts[0][2]["command"] == {"linear": 0.0, "angular": 0.0, "lamp": 0}


def test_none_payload_noops(fake_session):
    s = _strategy(fake_session, FakePredictor(), drive=True)
    s.analyze(None)
    assert fake_session.requests == []


def test_predictor_error_stops_when_driving(fake_session):
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))

    class ExplodingPredictor:
        def predict(self, payload):
            raise RuntimeError("boom")

    s = _strategy(fake_session, ExplodingPredictor(), drive=True)
    s.analyze("frame")
    posts = _control_posts(fake_session)
    assert posts and posts[0][2]["command"]["linear"] == 0.0


def test_on_decision_callback(fake_session):
    seen = []
    s = _strategy(
        fake_session,
        FakePredictor(make_mask(value=1.0)),
        drive=False,
        on_decision=lambda result, decision: seen.append(decision),
    )
    s.analyze("frame")
    assert len(seen) == 1 and seen[0].reason == "forward"


def test_duck_types_team_rover_strategy():
    """Must satisfy programs.genai_rover_api.RoverStrategy without importing it."""
    assert hasattr(TraversabilityStrategy, "get_image_payload")
    assert hasattr(TraversabilityStrategy, "analyze")
    sig = inspect.signature(TraversabilityStrategy.analyze)
    assert list(sig.parameters) == ["self", "payload"]
    sig = inspect.signature(TraversabilityStrategy.get_image_payload)
    assert list(sig.parameters) == ["self"]


def test_overlay_saving(tmp_path, fake_session):
    s = _strategy(
        fake_session,
        FakePredictor(make_mask(value=1.0)),
        drive=False,
        save_overlays_dir=str(tmp_path / "out"),
    )
    s.analyze("frame")
    saved = list((tmp_path / "out").glob("overlay_*.png"))
    assert len(saved) == 1
