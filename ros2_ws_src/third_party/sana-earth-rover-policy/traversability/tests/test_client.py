import base64
from io import BytesIO

from PIL import Image

from rover_traversability.client import RoverClient

from conftest import FakeResponse, FakeSession


def _client(session):
    return RoverClient(base_url="http://localhost:8000", session=session)


def _frame_b64():
    img = Image.new("RGB", (8, 6), (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_send_command_payload_shape_and_clamping(fake_session):
    fake_session.queue("/control", FakeResponse(200, {"message": "ok"}))
    c = _client(fake_session)
    result = c.send_command(2.0, -3.0, lamp=1)
    assert result.accepted
    method, url, payload = fake_session.requests[-1]
    assert method == "POST" and url.endswith("/control")
    assert payload == {"command": {"linear": 1.0, "angular": -1.0, "lamp": 1}}


def test_watchdog_500_does_not_raise(fake_session):
    fake_session.queue("/control", FakeResponse(500, text="safety stop took priority"))
    result = _client(fake_session).send_command(0.5, 0.0)
    assert not result.accepted
    assert result.status == 500


def test_connection_error_does_not_raise():
    import requests

    class BoomSession:
        def post(self, *a, **k):
            raise requests.ConnectionError("down")

        def get(self, *a, **k):
            raise requests.ConnectionError("down")

    c = RoverClient(base_url="http://localhost:8000", session=BoomSession())
    assert not c.send_command(1, 0).accepted
    assert c.get_data() is None
    assert c.get_front_frame_b64() is None


def test_stop_retries_until_accepted(fake_session):
    fake_session.responses["/control"] = [
        FakeResponse(500, text="rejected"),
        FakeResponse(200, {"message": "ok"}),
        FakeResponse(200, {"message": "ok"}),
    ]
    result = _client(fake_session).stop(retries=3)
    assert result.accepted


def test_front_frame_parsing(fake_session):
    b64 = _frame_b64()
    fake_session.queue("/v2/screenshot", FakeResponse(200, {"front_frame": b64}))
    c = _client(fake_session)
    assert c.get_front_frame_b64() == b64
    frame = c.get_front_frame()
    assert frame is not None and frame.shape == (6, 8, 3)


def test_front_frame_falls_back_to_v2_front(fake_session):
    b64 = _frame_b64()
    fake_session.queue("/v2/screenshot", FakeResponse(404))
    fake_session.queue("/v2/front", FakeResponse(200, {"front_frame": b64}))
    assert _client(fake_session).get_front_frame_b64() == b64


def test_base_url_env(monkeypatch):
    monkeypatch.setenv("ROVER_BASE_URL", "http://example.com:9999/")
    c = RoverClient()
    assert c.base_url == "http://example.com:9999"


def test_checkpoint_reached_rejection_is_not_error(fake_session):
    fake_session.queue(
        "/checkpoint-reached",
        FakeResponse(400, {"error": "Bot is not within 6 meters from the checkpoint"}),
    )
    result = _client(fake_session).checkpoint_reached()
    assert not result.accepted
    assert result.status == 400
