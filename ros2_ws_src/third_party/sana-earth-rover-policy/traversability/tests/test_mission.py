import base64
from io import BytesIO

from PIL import Image

from rover_traversability.client import RoverClient
from rover_traversability.mission import MissionRunner, _parse_checkpoints

from conftest import FakePredictor, FakeResponse, FakeSession, make_mask

LAT0, LON0 = -34.9215, -57.9545


def _frame_b64():
    img = Image.new("RGB", (16, 12), (0, 128, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _checkpoints_body(done=0):
    return {
        "checkpoints_list": [
            # lat/lon as STRINGS, like the real endpoint
            {"id": 1, "sequence": 1, "latitude": str(LAT0 + 0.0004), "longitude": str(LON0)},
            {"id": 2, "sequence": 2, "latitude": str(LAT0 + 0.0008), "longitude": str(LON0)},
        ],
        "latest_scanned_checkpoint": done,
    }


def test_parse_checkpoints_casts_strings():
    cps, latest = _parse_checkpoints(_checkpoints_body(done=1))
    assert latest == 1
    assert isinstance(cps[0].latitude, float)
    assert cps[0].sequence == 1 and cps[1].sequence == 2


def _runner(session, **kwargs):
    client = RoverClient(base_url="http://localhost:8000", session=session)
    return MissionRunner(
        client=client,
        predictor=FakePredictor(make_mask(value=1.0)),
        interval_s=0.0,
        **kwargs,
    )


def test_no_pending_checkpoints(fake_session):
    fake_session.queue("/checkpoints-list", FakeResponse(200, _checkpoints_body(done=2)))
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    result = _runner(fake_session).run()
    assert result.completed
    assert result.steps == 0


def test_drives_toward_checkpoint_and_stops_on_max_steps(fake_session):
    fake_session.queue("/checkpoints-list", FakeResponse(200, _checkpoints_body()))
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    fake_session.queue(
        "/data",
        FakeResponse(200, {"latitude": LAT0, "longitude": LON0,
                           "speed": 0.5, "orientation": 0.0}),
    )
    fake_session.queue("/v2/screenshot", FakeResponse(200, {"front_frame": _frame_b64()}))

    result = _runner(fake_session, max_steps=3).run()
    assert not result.completed
    assert result.reason == "max_steps reached"
    assert result.steps == 3
    # It drove: at least one non-zero motion command went out.
    moves = [
        r[2]["command"] for r in fake_session.requests
        if r[0] == "POST" and r[1].endswith("/control")
    ]
    assert any(cmd["linear"] > 0 for cmd in moves)
    # Final act of run() is always a stop.
    assert moves[-1]["linear"] == 0.0 and moves[-1]["angular"] == 0.0


def test_checkpoint_rejection_keeps_driving(fake_session):
    """Backend says 'not close enough' -> not an error, keep going."""
    fake_session.queue("/checkpoints-list", FakeResponse(200, _checkpoints_body()))
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    # Rover already within arrive_attempt_m of checkpoint 1 (~44m away by default).
    fake_session.queue(
        "/data",
        FakeResponse(200, {"latitude": LAT0 + 0.00039, "longitude": LON0,
                           "speed": 0.5, "orientation": 0.0}),
    )
    fake_session.queue("/v2/screenshot", FakeResponse(200, {"front_frame": _frame_b64()}))
    fake_session.queue(
        "/checkpoint-reached",
        FakeResponse(400, {"error": "Bot is not within 6 meters from the checkpoint"}),
    )
    result = _runner(fake_session, max_steps=2, arrive_attempt_m=8.0).run()
    assert result.steps == 2  # rejection did not abort the loop
    attempted = [r for r in fake_session.requests if r[1].endswith("/checkpoint-reached")]
    assert attempted  # it did try


def test_mission_completion(fake_session):
    fake_session.queue("/checkpoints-list", FakeResponse(200, _checkpoints_body(done=1)))
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    fake_session.queue(
        "/data",
        FakeResponse(200, {"latitude": LAT0 + 0.00079, "longitude": LON0,
                           "speed": 0.5, "orientation": 0.0}),
    )
    fake_session.queue("/v2/screenshot", FakeResponse(200, {"front_frame": _frame_b64()}))
    fake_session.queue(
        "/checkpoint-reached",
        FakeResponse(200, {"message": "ok", "mission_completed": True}),
    )
    result = _runner(fake_session).run()
    assert result.completed
    assert result.reason == "mission completed"


def test_goal_behind_arc_turns(fake_session):
    """Goal offset > turn_in_place_deg -> slow arc, not a spin and not a predict."""
    body = {
        "checkpoints_list": [
            {"id": 1, "sequence": 1, "latitude": str(LAT0 - 0.0004), "longitude": str(LON0)},
        ],
        "latest_scanned_checkpoint": 0,
    }
    fake_session.queue("/checkpoints-list", FakeResponse(200, body))
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    # Heading north (orientation 0 as cold-start), goal due south -> offset 180.
    fake_session.queue(
        "/data",
        FakeResponse(200, {"latitude": LAT0, "longitude": LON0,
                           "speed": 0.0, "orientation": 0.0}),
    )
    result = _runner(fake_session, max_steps=1).run()
    d = result.history[0]["decision"]
    assert d.reason == "arc_turn_to_goal"
    assert d.linear > 0  # arc, not a pure spin (GPS-COG needs motion)
