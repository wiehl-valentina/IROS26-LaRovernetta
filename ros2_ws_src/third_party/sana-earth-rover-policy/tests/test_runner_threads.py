"""Real-thread integration test for MissionV2Runner with fakes.

Asserts the anti-race invariants the design promises:
- exactly ONE thread ever POSTs /control (the control/main thread),
- the last /control write is a stop,
- nothing is written to /control after run() returns,
- the perception thread actually ran the predictor from another thread.
"""

from __future__ import annotations

import dataclasses
import threading
import time

from helpers import FakePredictor, FakeResponse, FakeSession

from sana.config import MissionV2Config
from sana.runner import MissionV2Runner

CP1 = {"id": 1, "sequence": 1, "latitude": "10.0", "longitude": "20.0"}


def make_sessions():
    control = FakeSession()
    control.queue("/data", FakeResponse(_json={
        "latitude": 10.00003, "longitude": 20.0, "speed": 0.5,
        "orientation": 180.0, "battery": 90,
        "rpms": [[30, 30, 30, 30, 0]],
    }))
    control.queue("/checkpoints-list", FakeResponse(_json={
        "checkpoints_list": [CP1], "latest_scanned_checkpoint": None,
    }))
    control.queue("/checkpoint-reached",
                  FakeResponse(_json={"mission_completed": True}))
    control.queue("/control", FakeResponse(_json={"message": "ok"}))

    perception = FakeSession()
    perception.queue("/v2/screenshot", FakeResponse(_json={"front_frame": "abc"}))
    return control, perception


def test_mission_completes_with_single_control_writer(tmp_path):
    control_session, perception_session = make_sessions()
    predictor = FakePredictor()
    cfg = dataclasses.replace(MissionV2Config(), control_hz=50.0, overlay_every_n=0)

    runner = MissionV2Runner(
        cfg=cfg,
        base_url="http://fake:8000",
        runs_dir=str(tmp_path),
        viewer=False,
        predictor=predictor,
        control_session=control_session,
        perception_session=perception_session,
    )
    result = runner.run(max_seconds=5.0)

    assert result.completed and result.reason == "completed"
    assert result.checkpoints_reached == 1

    posts = control_session.calls_to("/control", "POST")
    assert posts, "mission never wrote /control"
    # Single-writer invariant: every /control write from one thread (this one).
    writer_idents = {p[3] for p in posts}
    assert writer_idents == {threading.get_ident()}
    # The final write is a stop.
    last_cmd = posts[-1][2]["command"]
    assert (last_cmd["linear"], last_cmd["angular"]) == (0.0, 0.0)

    # Nothing touches /control after run() returns.
    n_posts = len(posts)
    time.sleep(0.3)
    assert len(control_session.calls_to("/control", "POST")) == n_posts

    # Perception really ran, from a different thread, against its own session.
    assert predictor.calls, "perception never ran the predictor"
    shots = perception_session.calls_to("/v2/screenshot", "GET")
    assert shots and all(s[3] != threading.get_ident() for s in shots)
    # The perception session never wrote a command.
    assert perception_session.calls_to("/control", "POST") == []

    # Run artifacts exist, including the auto-generated report.
    run_dir = tmp_path / result.run_dir.split("/")[-1]
    assert (run_dir / "decisions.jsonl").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "report.html").exists()


def test_dry_run_writes_logs_but_no_commands(tmp_path):
    control_session, perception_session = make_sessions()
    cfg = dataclasses.replace(MissionV2Config(), control_hz=50.0, overlay_every_n=0)

    runner = MissionV2Runner(
        cfg=cfg,
        base_url="http://fake:8000",
        runs_dir=str(tmp_path),
        viewer=False,
        dry_run=True,
        predictor=FakePredictor(),
        control_session=control_session,
        perception_session=perception_session,
    )
    result = runner.run(max_steps=10)

    assert control_session.calls_to("/control", "POST") == []
    assert control_session.calls_to("/checkpoint-reached", "POST") == []
    run_dir = tmp_path / result.run_dir.split("/")[-1]
    lines = (run_dir / "decisions.jsonl").read_text().strip().splitlines()
    assert len(lines) == result.ticks
