"""Tick-by-tick tests for sana/control.py against fake HTTP and perception.

No threads here: tests drive ControlLoop.tick(now) directly with controlled
monotonic times and inspect what got POSTed to the fake session.
"""

from __future__ import annotations

import numpy as np

from helpers import FakeResponse, FakeSession, make_mask
from rover_traversability.client import RoverClient

from sana.config import MissionV2Config
from sana.control import ControlLoop, GpsOutlierFilter
from sana.fsm import State
from sana.shared import MaskSnapshot, SharedState

CFG = MissionV2Config()

# Rover start and two checkpoints: CP1 ~55 m south of the rover, CP2 farther.
ROVER = {"latitude": 10.0005, "longitude": 20.0}
CP1 = {"id": 1, "sequence": 1, "latitude": "10.0", "longitude": "20.0"}
CP2 = {"id": 2, "sequence": 2, "latitude": "9.999", "longitude": "20.0"}


def telemetry(lat=ROVER["latitude"], lon=ROVER["longitude"], speed=0.8,
              orientation=180.0, battery=90, rpms=((30, 30, 30, 30, 0),)):
    return {
        "latitude": lat, "longitude": lon, "speed": speed,
        "orientation": orientation, "battery": battery,
        "rpms": [list(r) for r in rpms],
    }


def checkpoints_body(latest=None, cps=(CP1, CP2)):
    return {"checkpoints_list": [dict(c) for c in cps],
            "latest_scanned_checkpoint": latest}


def build_loop(session: FakeSession, cfg=CFG, dry_run=False):
    session.queue("/control", FakeResponse(_json={"message": "ok"}))
    client = RoverClient(base_url="http://fake:8000", session=session)
    shared = SharedState()
    loop = ControlLoop(cfg, client, shared, dry_run=dry_run)
    return loop, shared


def publish_mask(shared, mono_ts, mask=None, seq=1):
    shared.publish_mask(MaskSnapshot(
        mask=mask if mask is not None else make_mask(),
        overlay_jpeg=b"", mono_ts=mono_ts, inference_s=0.02, seq=seq,
    ))


def control_posts(session):
    return session.calls_to("/control", "POST")


# ------------------------------------------------------------------- driving

def test_pursue_sends_policy_forward_command(fake_session):
    fake_session.queue("/data", FakeResponse(_json=telemetry()))
    fake_session.queue("/checkpoints-list", FakeResponse(_json=checkpoints_body()))
    loop, shared = build_loop(fake_session)
    loop._refresh_target()
    publish_mask(shared, mono_ts=0.0)

    out = loop.tick(0.1)
    assert out.state == State.PURSUE
    posts = control_posts(fake_session)
    assert posts, "no /control write"
    cmd = posts[-1][2]["command"]
    assert cmd["linear"] > 0


def test_stale_mask_sends_stop(fake_session):
    fake_session.queue("/data", FakeResponse(_json=telemetry()))
    fake_session.queue("/checkpoints-list", FakeResponse(_json=checkpoints_body()))
    loop, shared = build_loop(fake_session)
    loop._refresh_target()
    publish_mask(shared, mono_ts=0.0)

    out = loop.tick(CFG.stale_mask_s + 1.0)
    assert out.reason == "stale_mask"
    cmd = control_posts(fake_session)[-1][2]["command"]
    assert cmd["linear"] == 0.0 and cmd["angular"] == 0.0


def test_physical_speed_cap_applies_to_sent_command(fake_session):
    fake_session.queue("/data", FakeResponse(_json=telemetry(speed=2.0)))
    fake_session.queue("/checkpoints-list", FakeResponse(_json=checkpoints_body()))
    loop, shared = build_loop(fake_session)
    loop._refresh_target()
    publish_mask(shared, mono_ts=0.0)

    out = loop.tick(0.1)
    assert "speed_capped" in out.guard_notes
    cmd = control_posts(fake_session)[-1][2]["command"]
    assert cmd["linear"] == 0.5 * CFG.min_linear


def test_dead_wheels_trigger_recovery_backup(fake_session):
    data = telemetry(rpms=((0, 0, 0, 0, 0),))
    fake_session.queue("/data", FakeResponse(_json=data))
    fake_session.queue("/checkpoints-list", FakeResponse(_json=checkpoints_body()))
    loop, shared = build_loop(fake_session)
    loop._refresh_target()

    t = 0.0
    while loop.fs.state != State.RECOVER_BACKUP:
        publish_mask(shared, mono_ts=t, seq=int(t * 10) + 1)
        loop.tick(t)
        t += 0.2
        assert t < 30, f"never recovered (state={loop.fs.state})"
    publish_mask(shared, mono_ts=t, seq=999)
    out = loop.tick(t)
    assert out.linear < 0  # backing up
    cmd = control_posts(fake_session)[-1][2]["command"]
    assert cmd["linear"] < 0


# ------------------------------------------------------------------- arrival

def test_arrival_retry_then_advance_to_next_checkpoint(fake_session):
    near_cp1 = telemetry(lat=10.00003, lon=20.0)
    fake_session.queue("/data", FakeResponse(_json=near_cp1))
    # First list: nothing scanned. After the accepted arrival the backend
    # reports CP1 as scanned; queue pops the first, then repeats the second.
    fake_session.queue("/checkpoints-list",
                       FakeResponse(_json=checkpoints_body(latest=None)),
                       FakeResponse(_json=checkpoints_body(latest=1)))
    fake_session.queue("/checkpoint-reached",
                       FakeResponse(status_code=400, text="not within 15 meters"),
                       FakeResponse(_json={"mission_completed": False}))
    loop, shared = build_loop(fake_session)
    loop._refresh_target()
    assert loop.target.sequence == 1

    publish_mask(shared, mono_ts=0.0)
    out = loop.tick(0.1)                       # enters ARRIVE, first attempt -> 400
    assert out.state == State.ARRIVE and out.attempt_checkpoint
    assert loop.checkpoints_reached == 0

    publish_mask(shared, mono_ts=1.2)
    out = loop.tick(0.1 + CFG.arrive_retry_s + 0.1)   # retry -> 200
    assert out.attempt_checkpoint
    publish_mask(shared, mono_ts=1.4)
    out = loop.tick(1.5 + CFG.arrive_retry_s)  # feedback consumed, re-routed
    assert loop.checkpoints_reached == 1
    assert loop.target.sequence == 2
    assert loop.fs.state in (State.PURSUE, State.ALIGN)


# ---------------------------------------------------------- heading / filters

def test_heading_estimator_reset_after_reverse():
    session = FakeSession()
    loop, _ = build_loop(session)
    original = loop.heading

    loop.fs.last_linear = -0.2                # commanding reverse
    loop._update_heading(10.0, 20.0, 0.9, None)
    assert loop.heading is original           # frozen, not replaced yet

    loop.fs.last_linear = 0.2                 # reverse ended
    loop._update_heading(10.0, 20.0, 0.9, None)
    assert loop.heading is not original       # fresh estimator re-acquires


def test_gps_outlier_filter_rejects_then_reaccepts():
    f = GpsOutlierFilter(max_jump_m=500.0, accept_after=5)
    assert f.filter(10.0, 20.0) == (10.0, 20.0)
    # A >500 m teleport: substituted with the previous fix...
    for _ in range(4):
        assert f.filter(15.0, 20.0) == (10.0, 20.0)
    # ...until enough consecutive rejections prove the world really moved.
    assert f.filter(15.0, 20.0) == (15.0, 20.0)
    # Normal small motion passes straight through.
    assert f.filter(15.0001, 20.0) == (15.0001, 20.0)


def test_send_throttling_skips_unchanged_commands(fake_session):
    fake_session.queue("/data", FakeResponse(_json=telemetry()))
    fake_session.queue("/checkpoints-list", FakeResponse(_json=checkpoints_body()))
    loop, shared = build_loop(fake_session)
    loop._refresh_target()

    publish_mask(shared, mono_ts=0.0)
    loop.tick(0.1)
    n_after_first = len(control_posts(fake_session))
    publish_mask(shared, mono_ts=0.2)
    loop.tick(0.3)                            # same command, within keepalive
    assert len(control_posts(fake_session)) == n_after_first
    publish_mask(shared, mono_ts=0.3)
    loop.tick(0.3 + CFG.send_keepalive_s + 0.1)   # keepalive refresh
    assert len(control_posts(fake_session)) == n_after_first + 1


def test_dry_run_never_posts(fake_session):
    near_cp1 = telemetry(lat=10.00003, lon=20.0)
    fake_session.queue("/data", FakeResponse(_json=near_cp1))
    fake_session.queue("/checkpoints-list", FakeResponse(_json=checkpoints_body()))
    loop, shared = build_loop(fake_session, dry_run=True)
    loop._refresh_target()

    publish_mask(shared, mono_ts=0.0)
    for i in range(5):
        loop.tick(0.1 + i * 0.2)
    assert control_posts(fake_session) == []
    assert fake_session.calls_to("/checkpoint-reached", "POST") == []


def test_battery_floor_finishes_mission(fake_session):
    fake_session.queue("/data", FakeResponse(_json=telemetry(battery=10)))
    fake_session.queue("/checkpoints-list", FakeResponse(_json=checkpoints_body()))
    loop, shared = build_loop(fake_session)
    loop._refresh_target()
    publish_mask(shared, mono_ts=0.0)

    out = loop.tick(0.1)
    assert out.done and out.done_reason == "battery_low"
