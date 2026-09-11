"""Pure control state machine for mission v2. No I/O, no clocks, no network.

The control loop calls ``step(cfg, fs, inp)`` at ~5 Hz with a monotonic
timestamp and everything it observed this tick; the FSM mutates its ``FsmState``
and returns the command to send. All HTTP (telemetry, /control,
/checkpoint-reached) happens in the caller, which feeds results back through
``ControlInputs`` on the next tick — that is what keeps this module trivially
testable without threads, torch or network.

States:

    WIGGLE  - forward burst until GPS course-over-ground heading is acquired
              (a rover standing still has no trustworthy heading; the
              magnetometer is not trusted, see rover_traversability.geo).
    ALIGN   - goal is far off to the side/behind. Sub-mode chosen on entry by
              |offset|: ARC (<45 deg, keeps moving so COG stays alive),
              KTURN (45-135, fwd/rev phases rotating the same way, wheels
              never counter-rotate), PIVOT (>=135, spin bursts + forward
              probes because spinning in place gets no COG update).
    PURSUE  - drive the traversability corridor policy with goal bias.
    ARRIVE  - within attempt radius: slow straight + /checkpoint-reached
              retries; the backend enforcing the true radius is the protocol.
    RECOVER_BACKUP / RECOVER_TURN - wheels stalled under a forward command:
              back up, turn (alternating sides), then re-acquire heading.
    DONE    - terminal; the output is always a stop.

Global guards (battery, dead perception, GPS loss, stuck, physical speed cap,
hard clamps) run every tick regardless of state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(str, Enum):
    WIGGLE = "wiggle"
    ALIGN = "align"
    PURSUE = "pursue"
    ARRIVE = "arrive"
    RECOVER_BACKUP = "recover_backup"
    RECOVER_TURN = "recover_turn"
    DONE = "done"


@dataclass
class ControlInputs:
    """Everything the control loop observed this tick. Times are monotonic."""

    t: float
    has_gps: bool = False
    distance_m: Optional[float] = None       # to the current checkpoint
    goal_offset_deg: Optional[float] = None  # wrapped [-180, 180]; + = goal to the RIGHT
    mask_decision: Optional[object] = None   # CommandDecision-like (linear/angular/stop/reason)
    mask_age_s: Optional[float] = None
    perception_alive: bool = True
    speed_ms: Optional[float] = None         # GPS-measured physical speed
    battery_pct: Optional[float] = None
    mean_abs_rpm: Optional[float] = None     # mean |rpm| of the 4 wheels
    arrival_accepted: Optional[bool] = None  # result of last /checkpoint-reached
    mission_completed: bool = False
    has_next_checkpoint: bool = False


@dataclass
class Output:
    linear: float
    angular: float
    reason: str
    state: State
    attempt_checkpoint: bool = False
    done: bool = False
    done_reason: Optional[str] = None
    guard_notes: tuple = ()                  # e.g. ("speed_capped",)


@dataclass
class FsmState:
    """Mutable across ticks; owned by the control loop."""

    state: State = State.WIGGLE
    entered_t: float = 0.0
    done_reason: Optional[str] = None
    # align episode
    align_mode: Optional[str] = None         # "arc" | "kturn" | "pivot"
    turn_sign: float = 0.0                   # +1 = left, -1 = right (SDK: angular+ = LEFT)
    stall_since: Optional[float] = None
    # stuck tracking
    last_linear: float = 0.0                 # what we commanded last tick
    stuck_since: Optional[float] = None
    stuck_events: list = field(default_factory=list)  # monotonic timestamps
    recovery_turn_sign: float = -1.0         # alternates each recovery (start: right)
    # gps availability tracking
    gps_lost_since: Optional[float] = None
    ever_had_gps: bool = False
    # arrival pacing
    last_arrival_attempt_t: float = -1e9


# --------------------------------------------------------------------- helpers


def _enter(fs: FsmState, state: State, t: float) -> None:
    fs.state = state
    fs.entered_t = t
    fs.align_mode = None
    fs.stall_since = None


def _finish(fs: FsmState, t: float, reason: str) -> Output:
    fs.done_reason = reason
    _enter(fs, State.DONE, t)
    return Output(0.0, 0.0, f"done_{reason}", State.DONE, done=True, done_reason=reason)


def _pick_align_mode(cfg, offset_deg: float) -> str:
    a = abs(offset_deg)
    if a < cfg.arc_turn_max_deg:
        return "arc"
    if a < cfg.kturn_max_deg:
        return "kturn"
    return "pivot"


def _enter_align(cfg, fs: FsmState, t: float, offset_deg: float) -> None:
    _enter(fs, State.ALIGN, t)
    fs.align_mode = _pick_align_mode(cfg, offset_deg)
    # Goal to the right (offset > 0) -> turn right -> negative angular.
    # Locked for the whole episode so +/-180 wrap noise can't flip it mid-turn.
    fs.turn_sign = -1.0 if offset_deg > 0 else 1.0


def _route_by_offset(cfg, fs: FsmState, t: float, offset_deg: Optional[float]) -> None:
    if offset_deg is not None and abs(offset_deg) > cfg.align_enter_deg:
        _enter_align(cfg, fs, t, offset_deg)
    else:
        _enter(fs, State.PURSUE, t)


# ---------------------------------------------------------------- global guards


def _global_guards(cfg, fs: FsmState, inp: ControlInputs) -> Optional[Output]:
    if inp.battery_pct is not None and inp.battery_pct <= cfg.battery_floor_pct:
        return _finish(fs, inp.t, "battery_low")

    if not inp.perception_alive:
        return _finish(fs, inp.t, "perception_dead")

    # GPS availability. At startup (never had a fix) give up after a while;
    # mid-mission a lost fix just parks the rover until it returns.
    if inp.has_gps:
        fs.ever_had_gps = True
        fs.gps_lost_since = None
    else:
        if fs.gps_lost_since is None:
            fs.gps_lost_since = inp.t
        lost_for = inp.t - fs.gps_lost_since
        if not fs.ever_had_gps and lost_for > cfg.no_gps_giveup_s:
            return _finish(fs, inp.t, "no_gps")
        if fs.ever_had_gps and lost_for > cfg.gps_lost_stop_s:
            return Output(0.0, 0.0, "gps_lost", fs.state)

    # Wheel stall under a forward command -> recovery. The gate is a fraction
    # of min_linear (auto-navigation-mini gates at 0.3 with max_linear=0.25,
    # so its detector can never fire).
    gate = cfg.stuck_cmd_fraction * cfg.min_linear
    if (
        fs.state in (State.WIGGLE, State.PURSUE, State.ARRIVE)
        and fs.last_linear >= gate
        and inp.mean_abs_rpm is not None
        and inp.mean_abs_rpm < cfg.stuck_rpm_eps
    ):
        if fs.stuck_since is None:
            fs.stuck_since = inp.t
        elif inp.t - fs.stuck_since >= cfg.stuck_after_s:
            fs.stuck_since = None
            fs.stuck_events = [
                e for e in fs.stuck_events if inp.t - e <= cfg.stuck_window_s
            ]
            fs.stuck_events.append(inp.t)
            if len(fs.stuck_events) >= cfg.max_stuck_events:
                return _finish(fs, inp.t, "stuck_giveup")
            fs.recovery_turn_sign = -fs.recovery_turn_sign
            _enter(fs, State.RECOVER_BACKUP, inp.t)
    else:
        fs.stuck_since = None

    return None


# ---------------------------------------------------------------- state handlers


def _step_wiggle(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    heading_ok = inp.goal_offset_deg is not None
    if heading_ok:
        _route_by_offset(cfg, fs, inp.t, inp.goal_offset_deg)
        return _dispatch(cfg, fs, inp)
    if inp.t - fs.entered_t > cfg.wiggle_max_s:
        # Continue without heading: PURSUE still avoids obstacles, and the
        # first stretch of forward motion will bootstrap the COG anyway.
        _enter(fs, State.PURSUE, inp.t)
        return _dispatch(cfg, fs, inp)
    return Output(cfg.wiggle_linear, 0.0, "wiggle", fs.state)


def _step_align(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    offset = inp.goal_offset_deg
    if offset is not None and abs(offset) <= cfg.align_exit_deg:
        _enter(fs, State.PURSUE, inp.t)
        return _dispatch(cfg, fs, inp)
    if inp.t - fs.entered_t > cfg.align_timeout_s:
        # Heading feedback is missing or not converging; pursue rather than
        # spin forever — forward motion is what regenerates the heading.
        _enter(fs, State.PURSUE, inp.t)
        return _dispatch(cfg, fs, inp)

    sign = fs.turn_sign
    elapsed = inp.t - fs.entered_t
    if fs.align_mode == "arc":
        lin, ang = cfg.arc_linear, sign * min(cfg.arc_angular, cfg.arc_linear)
        reason = "align_arc"
    elif fs.align_mode == "kturn":
        # Alternating forward/reverse phases, same angular sign both ways so
        # the rotation accumulates in one direction and the wheels never
        # counter-rotate (skid-steer: tight turn without a pivot).
        phase = (elapsed % (2.0 * cfg.kturn_phase_s)) < cfg.kturn_phase_s
        lin = cfg.min_linear if phase else -cfg.min_linear
        ang = sign * cfg.max_angular
        reason = "align_kturn_fwd" if phase else "align_kturn_rev"
    else:  # pivot
        cycle = elapsed % (cfg.pivot_phase_s + cfg.probe_phase_s)
        if cycle < cfg.pivot_phase_s:
            lin, ang = 0.0, sign * cfg.max_angular
            reason = "align_pivot"
        else:
            lin, ang = cfg.min_linear, 0.0
            reason = "align_probe"

    # Stall escalation: wheels not moving during the align episode. The timer
    # runs on wheel evidence alone — a zero-angular probe phase must NOT reset
    # it, or a pivot whose phase equals align_kick_after_s could never kick.
    # Boost to max_angular while commanding a turn; if the stall persists, a
    # short reverse pulse breaks static friction (momentum kick — dynamic
    # friction << static friction).
    if inp.mean_abs_rpm is not None and inp.mean_abs_rpm < cfg.align_stall_rpm:
        if fs.stall_since is None:
            fs.stall_since = inp.t
        stalled_for = inp.t - fs.stall_since
        if stalled_for > cfg.align_kick_after_s:
            lin = -cfg.min_linear
            ang = sign * cfg.max_angular
            reason = "align_kick"
            if stalled_for > cfg.align_kick_after_s + cfg.align_kick_s:
                fs.stall_since = inp.t  # kick done; restart the cycle
        elif abs(ang) > 0.05:
            ang = sign * cfg.max_angular
            reason = "align_stall_boost"
    else:
        fs.stall_since = None

    return Output(lin, ang, reason, fs.state)


def _step_pursue(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    if inp.distance_m is not None and inp.distance_m <= cfg.arrive_attempt_m:
        _enter(fs, State.ARRIVE, inp.t)
        return _dispatch(cfg, fs, inp)
    if inp.goal_offset_deg is not None and abs(inp.goal_offset_deg) > cfg.align_enter_deg:
        _enter_align(cfg, fs, inp.t, inp.goal_offset_deg)
        return _dispatch(cfg, fs, inp)

    if inp.mask_decision is None or (
        inp.mask_age_s is not None and inp.mask_age_s > cfg.stale_mask_s
    ):
        return Output(0.0, 0.0, "stale_mask", fs.state)

    d = inp.mask_decision
    if d.stop:
        return Output(0.0, 0.0, f"blocked_{d.reason}", fs.state)
    return Output(d.linear, d.angular, d.reason, fs.state)


def _step_arrive(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    if inp.arrival_accepted:
        # Consume the flag: the re-routed state may dispatch back into ARRIVE
        # within this same tick (next checkpoint already inside the radius)
        # and must not treat the old acceptance as its own.
        inp.arrival_accepted = None
        if inp.mission_completed or not inp.has_next_checkpoint:
            return _finish(fs, inp.t, "completed")
        # Target already advanced by the caller; re-route on fresh geometry.
        _route_by_offset(cfg, fs, inp.t, inp.goal_offset_deg)
        return _dispatch(cfg, fs, inp)
    if inp.distance_m is not None and inp.distance_m > cfg.arrive_exit_m:
        _enter(fs, State.PURSUE, inp.t)
        return _dispatch(cfg, fs, inp)

    attempt = inp.t - fs.last_arrival_attempt_t >= cfg.arrive_retry_s
    if attempt:
        fs.last_arrival_attempt_t = inp.t

    # Slow straight toward the point; do NOT re-enter ALIGN here — bearings go
    # wild this close to the target and the backend owns the real radius.
    # The mask still vetoes: a blocked or stale scene stops the rover (the
    # checkpoint attempt itself is harmless and keeps retrying).
    d = inp.mask_decision
    if d is None or (inp.mask_age_s is not None and inp.mask_age_s > cfg.stale_mask_s):
        return Output(0.0, 0.0, "stale_mask", fs.state, attempt_checkpoint=attempt)
    if d.stop:
        return Output(0.0, 0.0, f"blocked_{d.reason}", fs.state, attempt_checkpoint=attempt)
    return Output(cfg.arrive_linear, 0.0, "arrive", fs.state, attempt_checkpoint=attempt)


def _step_recover_backup(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    if inp.t - fs.entered_t >= cfg.backup_s:
        _enter(fs, State.RECOVER_TURN, inp.t)
        return _dispatch(cfg, fs, inp)
    return Output(-cfg.min_linear, 0.0, "backup", fs.state)


def _step_recover_turn(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    if inp.t - fs.entered_t >= cfg.recover_turn_s:
        # Heading is unreliable after reversing + turning: re-acquire it.
        _enter(fs, State.WIGGLE, inp.t)
        return _dispatch(cfg, fs, inp)
    return Output(0.0, fs.recovery_turn_sign * cfg.max_angular, "recover_turn", fs.state)


_HANDLERS = {
    State.WIGGLE: _step_wiggle,
    State.ALIGN: _step_align,
    State.PURSUE: _step_pursue,
    State.ARRIVE: _step_arrive,
    State.RECOVER_BACKUP: _step_recover_backup,
    State.RECOVER_TURN: _step_recover_turn,
}


def _dispatch(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    return _HANDLERS[fs.state](cfg, fs, inp)


# ------------------------------------------------------------------ entry point


def step(cfg, fs: FsmState, inp: ControlInputs) -> Output:
    """Advance one control tick. Mutates ``fs``; returns the command to send."""
    if fs.state == State.DONE:
        return Output(0.0, 0.0, f"done_{fs.done_reason}", State.DONE,
                      done=True, done_reason=fs.done_reason)

    guard = _global_guards(cfg, fs, inp)
    out = guard if guard is not None else _dispatch(cfg, fs, inp)
    out = _post_filter(cfg, out, inp)
    fs.last_linear = out.linear
    return out


def _post_filter(cfg, out: Output, inp: ControlInputs) -> Output:
    """Always-last output filters, in order: physical speed cap, hard clamps.

    The speed cap is the belt-and-suspenders guarantee from task #281 of
    auto-navigation-mini: max_linear is firmware *command* space (multiplied
    ~3-5x by the Mini+), so the only way to bound the *physical* speed is to
    compare the GPS-measured one. When tripped, the command drops to just
    above the motor deadband (coast) instead of a hard zero — braking to zero
    at 1.65 m/s pitches the rover nose-down.
    """
    lin, ang, notes = out.linear, out.angular, list(out.guard_notes)

    if (
        lin > 0.0
        and inp.speed_ms is not None
        and inp.speed_ms > cfg.speed_cap_ms * cfg.speed_cap_tol
    ):
        lin = min(lin, 0.5 * cfg.min_linear)
        notes.append("speed_capped")

    clamped_lin = max(-cfg.max_linear, min(cfg.max_linear, lin))
    clamped_ang = max(-cfg.max_angular, min(cfg.max_angular, ang))
    if clamped_lin != lin or clamped_ang != ang:
        notes.append("clamped")

    if clamped_lin == out.linear and clamped_ang == out.angular and not notes:
        return out
    return Output(
        clamped_lin,
        clamped_ang,
        out.reason,
        out.state,
        attempt_checkpoint=out.attempt_checkpoint,
        done=out.done,
        done_reason=out.done_reason,
        guard_notes=tuple(notes),
    )
