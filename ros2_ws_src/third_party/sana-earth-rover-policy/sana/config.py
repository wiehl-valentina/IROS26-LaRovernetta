"""Mission v2 configuration.

Motion values come from auto-navigation-mini's field measurements
(Divyesh's config.py, 2026-08-04):

- The Mini+ firmware multiplies the linear command by ~3-5x: cmd=0.45 measured
  2.39 m/s physical, so max_linear=0.25 is ~1.3-1.5 m/s real.
- The motor deadband is ~0.15 in command space (0.15 -> 0.04 m/s actual);
  min_linear=0.20 sits firmly outside it.
- max_angular=0.4: at 1.0 rad/s the rover tipped onto its side during pivots
  (skid-steer physics, centrifugal force on the outer wheels).

Three fixes over what auto-navigation-mini ships:
- its wheel-stuck gate (command >= 0.3 with max_linear=0.25) can never fire;
  here the gate is a fraction of min_linear.
- its GPS outlier filter rejects forever after a real >500 m jump; here we
  re-accept after N consecutive rejections (lives in control.py).
- its align collapses to a constant 0.3 rad/s, below the ~0.35 pivot deadband
  its own config documents; here turns use max_angular and escalate on stall.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MissionV2Config:
    # ----------------------------------------------------------------- motion
    max_linear: float = 0.25
    min_linear: float = 0.20
    max_angular: float = 0.4

    # ------------------------------------------------------------------ align
    align_enter_deg: float = 30.0     # enter ALIGN above this |goal offset|
    align_exit_deg: float = 8.0       # exit ALIGN below this (hysteresis)
    align_timeout_s: float = 20.0     # align episode longer than this -> PURSUE
    arc_turn_max_deg: float = 45.0    # |offset| <  45 -> ARC (keep moving while turning)
    kturn_max_deg: float = 135.0      # 45..135 -> KTURN; >= 135 -> PIVOT
    arc_linear: float = 0.20
    arc_angular: float = 0.20         # <= arc_linear: both wheels keep rolling forward
    kturn_phase_s: float = 1.2        # duration of each K-turn fwd/rev phase
    pivot_phase_s: float = 1.2        # pivot in place
    probe_phase_s: float = 0.8        # short forward burst to refresh GPS-COG heading
    align_stall_rpm: float = 1.0      # turning with rpms below this = stalled
    align_kick_after_s: float = 1.2   # sustained stall -> reverse pulse (momentum kick)
    align_kick_s: float = 0.4
    bearing_ema_alpha: float = 0.3    # goal-offset smoothing (compass jitters +/-13 deg)

    # ----------------------------------------------------------------- wiggle
    wiggle_linear: float = 0.20       # forward burst until GPS-COG is acquired
    wiggle_max_s: float = 8.0         # give-up: continue without heading (WARN)
    no_gps_giveup_s: float = 15.0     # no fix at startup -> DONE(no_gps)
    gps_lost_stop_s: float = 10.0     # fix lost while driving -> stop and wait

    # ---------------------------------------------------------- stuck/recovery
    stuck_rpm_eps: float = 5.0        # mean |rpm| below this = wheels not moving
    stuck_cmd_fraction: float = 0.9   # gate: command >= 0.9*min_linear (fixed gate)
    stuck_after_s: float = 3.0
    backup_s: float = 2.0
    recover_turn_s: float = 1.5
    max_stuck_events: int = 3         # events within stuck_window_s -> give up
    stuck_window_s: float = 60.0

    # ---------------------------------------------------------------- arrival
    arrive_attempt_m: float = 6.0     # start attempting /checkpoint-reached
    arrive_exit_m: float = 8.0        # hysteresis: back to PURSUE beyond this
    arrive_retry_s: float = 1.0       # the backend's 400 IS the designed retry loop
    arrive_linear: float = 0.20       # slow straight; the backend owns the true radius

    # ----------------------------------------------------------------- guards
    speed_cap_ms: float = 1.5         # PHYSICAL cap measured via GPS (task #281)
    speed_cap_tol: float = 1.1        # trips at 1.65 m/s (GPS jitters +/-0.2)
    battery_floor_pct: float = 15.0
    stale_mask_s: float = 3.0         # older mask -> (0,0) this tick
    perception_dead_s: float = 30.0   # perception dead -> end mission
    control_reject_s: float = 5.0     # /control rejecting continuously -> end mission

    # -------------------------------------------------------------- loops/send
    control_hz: float = 5.0
    send_change_eps: float = 0.02     # smaller command change is not re-sent...
    send_keepalive_s: float = 0.9     # ...unless this long since last send (~4 s latch)

    # -------------------------------------------------------------- perception
    renormalize_percentile: bool = False  # per-frame p5-p95 stretch (soft scenes)

    # ------------------------------------------------------------------- demo
    viewer_port: int = 8001           # :8000 is the SDK
    overlay_every_n: int = 5          # save an overlay PNG every N frames

    def policy(self):
        """Vendored PolicyConfig carrying our motion limits."""
        from rover_traversability.policy import PolicyConfig

        return PolicyConfig(
            max_linear=self.max_linear,
            min_linear=self.min_linear,
            max_angular=self.max_angular,
        )
