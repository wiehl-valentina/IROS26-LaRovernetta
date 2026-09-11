"""Table-style tests for the pure mission-v2 state machine (sana/fsm.py)."""

from __future__ import annotations

import pytest

from helpers import DecisionStub
from sana.config import MissionV2Config
from sana.fsm import ControlInputs, FsmState, State, step

CFG = MissionV2Config()


def inputs(t=0.0, **kw):
    base = dict(
        has_gps=True,
        distance_m=50.0,
        goal_offset_deg=0.0,
        mask_decision=DecisionStub(),
        mask_age_s=0.1,
        perception_alive=True,
        speed_ms=0.8,
        battery_pct=90.0,
        mean_abs_rpm=40.0,
    )
    base.update(kw)
    return ControlInputs(t=t, **base)


def fresh(state=State.WIGGLE, t=0.0):
    fs = FsmState()
    fs.state = state
    fs.entered_t = t
    return fs


# ------------------------------------------------------------------- WIGGLE

def test_wiggle_drives_forward_without_heading():
    fs = fresh()
    out = step(CFG, fs, inputs(t=1.0, goal_offset_deg=None))
    assert out.reason == "wiggle"
    assert out.linear == CFG.wiggle_linear and out.angular == 0.0
    assert fs.state == State.WIGGLE


def test_wiggle_exits_to_pursue_when_heading_acquired_and_goal_ahead():
    fs = fresh()
    out = step(CFG, fs, inputs(t=2.0, goal_offset_deg=5.0))
    assert fs.state == State.PURSUE
    assert out.linear == DecisionStub().linear


def test_wiggle_exits_to_align_when_goal_far_off():
    fs = fresh()
    step(CFG, fs, inputs(t=2.0, goal_offset_deg=90.0))
    assert fs.state == State.ALIGN
    assert fs.align_mode == "kturn"


def test_wiggle_times_out_into_pursue():
    fs = fresh()
    out = step(CFG, fs, inputs(t=CFG.wiggle_max_s + 0.1, goal_offset_deg=None))
    assert fs.state == State.PURSUE
    assert out.reason == DecisionStub().reason


def test_no_gps_at_startup_ends_mission():
    fs = fresh()
    # The FSM measures the outage from when it first OBSERVES it.
    step(CFG, fs, inputs(t=0.0, has_gps=False, goal_offset_deg=None, distance_m=None))
    out = step(CFG, fs, inputs(t=CFG.no_gps_giveup_s + 1, has_gps=False,
                               goal_offset_deg=None, distance_m=None))
    assert out.done and out.done_reason == "no_gps"


# -------------------------------------------------------------------- ALIGN

def test_align_mode_selection_by_offset():
    for offset, mode in [(35.0, "arc"), (90.0, "kturn"), (170.0, "pivot")]:
        fs = fresh()
        step(CFG, fs, inputs(t=0.0, goal_offset_deg=offset))
        assert fs.align_mode == mode, offset


def test_align_hysteresis_enters_30_exits_8():
    fs = fresh(State.PURSUE)
    step(CFG, fs, inputs(t=0.0, goal_offset_deg=31.0))
    assert fs.state == State.ALIGN
    # 15 deg: inside the entry threshold but above exit — keep aligning.
    step(CFG, fs, inputs(t=1.0, goal_offset_deg=15.0))
    assert fs.state == State.ALIGN
    step(CFG, fs, inputs(t=2.0, goal_offset_deg=7.0))
    assert fs.state == State.PURSUE


def test_align_turn_sign_locked_across_180_wrap():
    fs = fresh(State.PURSUE)
    step(CFG, fs, inputs(t=0.0, goal_offset_deg=170.0))
    assert fs.turn_sign == -1.0  # goal right -> turn right (negative angular)
    out = step(CFG, fs, inputs(t=0.5, goal_offset_deg=-179.0))
    assert fs.state == State.ALIGN
    assert fs.turn_sign == -1.0  # wrap noise must not flip the direction
    assert out.angular <= 0.0 or out.linear > 0.0  # probe phases have angular 0


def test_align_arc_keeps_wheels_rolling_forward():
    fs = fresh(State.PURSUE)
    out = step(CFG, fs, inputs(t=0.0, goal_offset_deg=-40.0))
    assert out.reason == "align_arc"
    assert out.linear > 0
    assert abs(out.angular) <= out.linear  # skid-steer: no counter-rotation
    assert out.angular > 0  # goal left -> turn left -> positive angular


def test_align_kturn_alternates_phases_same_rotation_sign():
    fs = fresh(State.PURSUE)
    out_fwd = step(CFG, fs, inputs(t=0.0, goal_offset_deg=90.0))
    entered = fs.entered_t
    out_fwd = step(CFG, fs, inputs(t=entered + 0.5, goal_offset_deg=90.0))
    assert out_fwd.reason == "align_kturn_fwd" and out_fwd.linear > 0
    out_rev = step(CFG, fs, inputs(t=entered + CFG.kturn_phase_s + 0.1,
                                   goal_offset_deg=90.0))
    assert out_rev.reason == "align_kturn_rev" and out_rev.linear < 0
    assert out_fwd.angular == out_rev.angular  # same rotation direction


def test_align_pivot_alternates_spin_and_forward_probe():
    fs = fresh(State.PURSUE)
    step(CFG, fs, inputs(t=0.0, goal_offset_deg=170.0))
    entered = fs.entered_t
    spin = step(CFG, fs, inputs(t=entered + 0.5, goal_offset_deg=170.0))
    assert spin.reason == "align_pivot" and spin.linear == 0.0
    assert abs(spin.angular) == CFG.max_angular
    probe = step(CFG, fs, inputs(t=entered + CFG.pivot_phase_s + 0.2,
                                 goal_offset_deg=170.0))
    assert probe.reason == "align_probe"
    assert probe.linear == CFG.min_linear and probe.angular == 0.0


def test_align_stall_boosts_then_kicks_in_reverse():
    fs = fresh(State.PURSUE)
    step(CFG, fs, inputs(t=0.0, goal_offset_deg=170.0))
    entered = fs.entered_t
    boosted = step(CFG, fs, inputs(t=entered + 0.4, goal_offset_deg=170.0,
                                   mean_abs_rpm=0.0))
    assert boosted.reason == "align_stall_boost"
    assert abs(boosted.angular) == CFG.max_angular
    kicked = step(CFG, fs, inputs(t=entered + 0.4 + CFG.align_kick_after_s + 0.1,
                                  goal_offset_deg=170.0, mean_abs_rpm=0.0))
    assert kicked.reason == "align_kick" and kicked.linear < 0


def test_align_timeout_falls_back_to_pursue():
    fs = fresh(State.PURSUE)
    step(CFG, fs, inputs(t=0.0, goal_offset_deg=90.0))
    entered = fs.entered_t
    step(CFG, fs, inputs(t=entered + CFG.align_timeout_s + 0.1,
                         goal_offset_deg=None))
    assert fs.state == State.PURSUE


# ------------------------------------------------------------------- PURSUE

def test_pursue_passes_policy_decision_through():
    fs = fresh(State.PURSUE)
    d = DecisionStub(linear=0.21, angular=-0.1, reason="turning_to_corridor")
    out = step(CFG, fs, inputs(t=0.0, mask_decision=d))
    assert (out.linear, out.angular, out.reason) == (0.21, -0.1, "turning_to_corridor")


def test_pursue_blocked_decision_stops():
    fs = fresh(State.PURSUE)
    out = step(CFG, fs, inputs(t=0.0, mask_decision=DecisionStub(stop=True, reason="blocked")))
    assert (out.linear, out.angular) == (0.0, 0.0)
    assert out.reason.startswith("blocked")


def test_pursue_stale_mask_stops_but_stays():
    fs = fresh(State.PURSUE)
    out = step(CFG, fs, inputs(t=0.0, mask_age_s=CFG.stale_mask_s + 1))
    assert (out.linear, out.angular) == (0.0, 0.0)
    assert out.reason == "stale_mask"
    assert fs.state == State.PURSUE


def test_pursue_enters_arrive_inside_radius():
    fs = fresh(State.PURSUE)
    out = step(CFG, fs, inputs(t=0.0, distance_m=CFG.arrive_attempt_m - 1))
    assert fs.state == State.ARRIVE
    assert out.attempt_checkpoint


# ------------------------------------------------------------------- ARRIVE

def test_arrive_attempts_are_paced():
    fs = fresh(State.PURSUE)
    out1 = step(CFG, fs, inputs(t=0.0, distance_m=4.0))
    out2 = step(CFG, fs, inputs(t=0.2, distance_m=4.0))
    out3 = step(CFG, fs, inputs(t=0.2 + CFG.arrive_retry_s, distance_m=4.0))
    assert out1.attempt_checkpoint and not out2.attempt_checkpoint
    assert out3.attempt_checkpoint


def test_arrive_accepted_with_next_checkpoint_reroutes():
    fs = fresh(State.ARRIVE)
    step(CFG, fs, inputs(t=5.0, distance_m=40.0, goal_offset_deg=10.0,
                         arrival_accepted=True, has_next_checkpoint=True))
    assert fs.state == State.PURSUE


def test_arrive_accepted_last_checkpoint_completes():
    fs = fresh(State.ARRIVE)
    out = step(CFG, fs, inputs(t=5.0, arrival_accepted=True,
                               mission_completed=True, has_next_checkpoint=False))
    assert out.done and out.done_reason == "completed"


def test_arrive_exit_hysteresis():
    fs = fresh(State.ARRIVE)
    step(CFG, fs, inputs(t=5.0, distance_m=CFG.arrive_exit_m - 0.5))
    assert fs.state == State.ARRIVE
    step(CFG, fs, inputs(t=6.0, distance_m=CFG.arrive_exit_m + 0.5))
    assert fs.state == State.PURSUE


def test_arrive_next_checkpoint_also_in_radius_no_infinite_loop():
    fs = fresh(State.ARRIVE)
    out = step(CFG, fs, inputs(t=5.0, distance_m=3.0, goal_offset_deg=0.0,
                               arrival_accepted=True, has_next_checkpoint=True))
    assert fs.state == State.ARRIVE  # re-routed PURSUE -> ARRIVE, terminated cleanly
    assert not out.done


# ----------------------------------------------------------- STUCK / RECOVERY

def drive_until_stuck(fs, t0=0.0):
    """Forward command with dead wheels until the FSM enters recovery."""
    t = t0
    step(CFG, fs, inputs(t=t))  # establishes last_linear > gate
    while fs.state not in (State.RECOVER_BACKUP, State.DONE):
        t += 0.2
        step(CFG, fs, inputs(t=t, mean_abs_rpm=0.0))
        assert t < t0 + 30, "never entered recovery"
    return t


def test_stuck_detection_enters_backup_then_turn_then_wiggle():
    fs = fresh(State.PURSUE)
    t = drive_until_stuck(fs)
    assert fs.state == State.RECOVER_BACKUP
    out = step(CFG, fs, inputs(t=t + 0.2, mean_abs_rpm=0.0))
    assert out.linear < 0 and out.reason == "backup"
    t_turn = t + CFG.backup_s + 0.1
    out = step(CFG, fs, inputs(t=t_turn, mean_abs_rpm=0.0))
    assert fs.state == State.RECOVER_TURN and out.linear == 0.0
    t_wig = t_turn + CFG.recover_turn_s + 0.1
    step(CFG, fs, inputs(t=t_wig, mean_abs_rpm=0.0, goal_offset_deg=None))
    assert fs.state == State.WIGGLE


def test_recovery_turn_sign_alternates_between_events():
    fs = fresh(State.PURSUE)
    t = drive_until_stuck(fs)
    first_sign = fs.recovery_turn_sign
    # ride through backup + turn back to wiggle, then get stuck again
    t = t + CFG.backup_s + CFG.recover_turn_s + 1.0
    step(CFG, fs, inputs(t=t, goal_offset_deg=None))
    t = drive_until_stuck(fs, t0=t + 0.2)
    assert fs.recovery_turn_sign == -first_sign


def ride_recovery_back_to_pursue(fs, t):
    """Step with healthy wheels until the recovery FSM lands back in PURSUE."""
    deadline = t + 30.0
    while fs.state != State.PURSUE:
        t += 0.5
        step(CFG, fs, inputs(t=t, goal_offset_deg=5.0))
        assert t < deadline, f"never returned to PURSUE (state={fs.state})"
    return t


def test_three_stuck_events_within_window_give_up():
    fs = fresh(State.PURSUE)
    t = 0.0
    for _ in range(CFG.max_stuck_events - 1):
        t = drive_until_stuck(fs, t0=t + 1.0)
        t = ride_recovery_back_to_pursue(fs, t)
    t = drive_until_stuck(fs, t0=t + 1.0)
    assert fs.state == State.DONE
    assert fs.done_reason == "stuck_giveup"


# -------------------------------------------------------------- GLOBAL GUARDS

def test_battery_floor_ends_mission():
    fs = fresh(State.PURSUE)
    out = step(CFG, fs, inputs(t=0.0, battery_pct=CFG.battery_floor_pct - 1))
    assert out.done and out.done_reason == "battery_low"
    # DONE is terminal and always outputs a stop.
    out2 = step(CFG, fs, inputs(t=1.0, battery_pct=90.0))
    assert out2.done and (out2.linear, out2.angular) == (0.0, 0.0)


def test_dead_perception_ends_mission():
    fs = fresh(State.PURSUE)
    out = step(CFG, fs, inputs(t=0.0, perception_alive=False))
    assert out.done and out.done_reason == "perception_dead"


def test_gps_lost_mid_mission_parks_and_resumes():
    fs = fresh(State.PURSUE)
    step(CFG, fs, inputs(t=0.0))  # had gps once
    step(CFG, fs, inputs(t=1.0, has_gps=False, distance_m=None, goal_offset_deg=None))
    out = step(CFG, fs, inputs(t=1.5 + CFG.gps_lost_stop_s, has_gps=False,
                               distance_m=None, goal_offset_deg=None))
    assert (out.linear, out.angular) == (0.0, 0.0) and out.reason == "gps_lost"
    assert fs.state == State.PURSUE
    out = step(CFG, fs, inputs(t=2.0 + CFG.gps_lost_stop_s))
    assert out.reason != "gps_lost"


def test_physical_speed_cap_coasts_instead_of_braking():
    fs = fresh(State.PURSUE)
    out = step(CFG, fs, inputs(t=0.0, speed_ms=CFG.speed_cap_ms * CFG.speed_cap_tol + 0.1))
    assert out.linear == pytest.approx(0.5 * CFG.min_linear)
    assert "speed_capped" in out.guard_notes


def test_hard_clamps_bound_any_output():
    fs = fresh(State.PURSUE)
    wild = DecisionStub(linear=5.0, angular=-9.0)
    out = step(CFG, fs, inputs(t=0.0, mask_decision=wild))
    assert abs(out.linear) <= CFG.max_linear
    assert abs(out.angular) <= CFG.max_angular
    assert "clamped" in out.guard_notes
