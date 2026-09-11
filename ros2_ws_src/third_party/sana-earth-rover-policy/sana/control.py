"""Control loop: telemetry -> heading -> FSM -> throttled /control writes.

Runs in the MAIN thread so Ctrl-C lands in the one place that owns /control
and its final stop. Everything decision-shaped lives in the pure FSM
(sana/fsm.py); this module does the I/O around it: telemetry parsing, the GPS
outlier filter, heading estimation (with the reverse-motion trap handled),
checkpoint bookkeeping, command send throttling, and the JSONL log.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from math import degrees
from pathlib import Path
from typing import Callable, Optional

from rover_traversability.geo import (
    HeadingEstimator,
    gps_bearing_and_distance,
    wrap_angle_deg,
)
from rover_traversability.policy import suggest_command

from .fsm import ControlInputs, FsmState, State, step
from .shared import SharedState

log = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    id: int
    sequence: int
    latitude: float
    longitude: float


def parse_checkpoints(body: dict) -> tuple[list[Checkpoint], int]:
    """Parse /checkpoints-list; lat/lon arrive as STRINGS there.

    Same contract as rover_traversability.mission._parse_checkpoints — copied
    rather than imported because that helper is private to a package that
    mirrors an external PR.
    """
    raw = body.get("checkpoints_list") or []
    cps = [
        Checkpoint(
            id=int(c.get("id", i)),
            sequence=int(c.get("sequence", i + 1)),
            latitude=float(c["latitude"]),
            longitude=float(c["longitude"]),
        )
        for i, c in enumerate(raw)
    ]
    cps.sort(key=lambda c: c.sequence)
    latest = body.get("latest_scanned_checkpoint")
    return cps, (int(latest) if latest is not None else 0)


class GpsOutlierFilter:
    """Reject physically impossible GPS jumps; recover from real teleports.

    auto-navigation-mini rejects any fix >500 m from the previous one (the
    "Kenya bounce": foreign RTM packets / cached fixes during satellite loss)
    but does so forever — after a genuine relocation every subsequent fix is
    far from the frozen anchor and gets rejected until restart. Here N
    consecutive rejections mean the world moved, not the data: accept and
    re-anchor.
    """

    def __init__(self, max_jump_m: float = 500.0, accept_after: int = 5) -> None:
        self.max_jump_m = float(max_jump_m)
        self.accept_after = int(accept_after)
        self._last: Optional[tuple[float, float]] = None
        self._rejections = 0

    def filter(self, lat, lon):
        if lat is None or lon is None:
            return None, None
        lat, lon = float(lat), float(lon)
        if self._last is not None:
            _, dist = gps_bearing_and_distance(self._last[0], self._last[1], lat, lon)
            if dist > self.max_jump_m:
                self._rejections += 1
                if self._rejections < self.accept_after:
                    return self._last  # substitute, don't drop
                log.warning("GPS jumped >%.0fm for %d fixes — accepting new position",
                            self.max_jump_m, self._rejections)
        self._rejections = 0
        self._last = (lat, lon)
        return lat, lon


def _mean_abs_rpm(data: dict) -> Optional[float]:
    """Mean |rpm| of the 4 wheels from the /data ``rpms`` array.

    Format (SDK): ``[[fl, fr, rl, rr, ts], ...]`` — last entry is freshest.
    """
    rpms = data.get("rpms")
    if not rpms or not isinstance(rpms, (list, tuple)):
        return None
    latest = rpms[-1]
    if not isinstance(latest, (list, tuple)) or len(latest) < 4:
        return None
    try:
        return sum(abs(float(r)) for r in latest[:4]) / 4.0
    except (TypeError, ValueError):
        return None


class ControlLoop:
    def __init__(
        self,
        cfg,
        client,                                   # RoverClient, exclusive to this thread
        shared: SharedState,
        perception_alive: Callable[[], bool] = lambda: True,
        dry_run: bool = False,
        jsonl_path: Optional[Path] = None,
        log_every: int = 25,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.shared = shared
        self.perception_alive = perception_alive
        self.dry_run = dry_run
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.log_every = log_every

        self.fs = FsmState()
        self.heading = HeadingEstimator()
        self.gps_filter = GpsOutlierFilter()
        self.policy_cfg = cfg.policy()

        self.ticks = 0
        self.checkpoints_reached = 0
        self.target: Optional[Checkpoint] = None
        self._offset_ema: Optional[float] = None
        self._arrival_feedback: Optional[dict] = None
        self._was_reversing = False
        self._last_sent: Optional[tuple[float, float]] = None
        self._last_sent_t = -1e9
        self._reject_since: Optional[float] = None
        self._start_mono: Optional[float] = None
        self._jsonl_file = None

    # ------------------------------------------------------------- lifecycle

    def run(self, max_steps: Optional[int] = None,
            max_seconds: Optional[float] = None) -> str:
        """Drive ticks until the FSM finishes. Returns the finish reason."""
        period = 1.0 / self.cfg.control_hz
        self._start_mono = time.monotonic()
        if self.jsonl_path:
            self._jsonl_file = self.jsonl_path.open("a")

        self._refresh_target()
        if self.target is None:
            return "no_pending_checkpoints"

        try:
            while not self.shared.stop.is_set():
                t0 = time.monotonic()
                out = self.tick(t0)
                if out.done:
                    return out.done_reason or "done"
                if self._reject_since is not None and (
                    t0 - self._reject_since > self.cfg.control_reject_s
                ):
                    return "control_rejected"
                if max_steps is not None and self.ticks >= max_steps:
                    return "max_steps"
                if max_seconds is not None and t0 - self._start_mono >= max_seconds:
                    return "max_seconds"
                elapsed = time.monotonic() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
            return "stopped"
        finally:
            if self._jsonl_file:
                self._jsonl_file.close()

    # ------------------------------------------------------------------ tick

    def tick(self, now: float):
        if self._start_mono is None:
            self._start_mono = now
        self.ticks += 1
        data = self.client.get_data() or {}
        lat, lon = self.gps_filter.filter(data.get("latitude"), data.get("longitude"))
        self._last_fix = (lat, lon)
        speed = data.get("speed")
        battery = data.get("battery")
        rpm = _mean_abs_rpm(data)

        heading_deg = self._update_heading(lat, lon, speed, data.get("orientation"))

        distance_m = None
        goal_offset = None
        if lat is not None and lon is not None and self.target is not None:
            bearing_rad, distance_m = gps_bearing_and_distance(
                lat, lon, self.target.latitude, self.target.longitude
            )
            if heading_deg is not None:
                raw = wrap_angle_deg(degrees(bearing_rad) - heading_deg)
                goal_offset = self._smooth_offset(raw)

        snap = self.shared.latest_mask()
        mask_age = now - snap.mono_ts if snap is not None else None
        decision = None
        if snap is not None:
            decision = suggest_command(snap.mask, self.policy_cfg,
                                       goal_offset_deg=goal_offset)

        feedback = self._arrival_feedback or {}
        self._arrival_feedback = None
        inp = ControlInputs(
            t=now,
            has_gps=lat is not None,
            distance_m=distance_m,
            goal_offset_deg=goal_offset,
            mask_decision=decision,
            mask_age_s=mask_age,
            perception_alive=self._perception_ok(now, snap),
            speed_ms=speed,
            battery_pct=battery,
            mean_abs_rpm=rpm,
            arrival_accepted=feedback.get("accepted"),
            mission_completed=feedback.get("completed", False),
            has_next_checkpoint=feedback.get("has_next", False),
        )
        out = step(self.cfg, self.fs, inp)

        if out.attempt_checkpoint:
            self._attempt_checkpoint()

        sent = self._send(out, now)
        self._record(out, inp, decision, snap, sent, now)
        return out

    # --------------------------------------------------------------- helpers

    def _perception_ok(self, now: float, snap) -> bool:
        if not self.perception_alive():
            return False
        last_seen = snap.mono_ts if snap is not None else self._start_mono
        return (now - last_seen) <= self.cfg.perception_dead_s

    def _update_heading(self, lat, lon, speed, orientation):
        """GPS-COG heading with the reverse-motion trap handled.

        COG during reverse motion points 180 deg the wrong way, and even the
        anchor drifts. While commanding reverse we feed speed 0 so the
        estimator's min-speed guard freezes the COG; when reverse ends, the
        estimator is replaced so the next forward meters re-acquire cleanly.
        """
        reversing = self.fs.last_linear < 0
        if self._was_reversing and not reversing:
            self.heading = HeadingEstimator()
            self._offset_ema = None
        self._was_reversing = reversing
        return self.heading.update(
            lat, lon, 0.0 if reversing else speed, orientation
        )

    def _smooth_offset(self, raw_deg: float) -> float:
        """Shortest-path EMA — the magnetometer jitters +/-13 deg per tick and
        unsmoothed offsets flip the align turn decision every tick."""
        if self._offset_ema is None:
            self._offset_ema = raw_deg
        else:
            delta = wrap_angle_deg(raw_deg - self._offset_ema)
            self._offset_ema = wrap_angle_deg(
                self._offset_ema + self.cfg.bearing_ema_alpha * delta
            )
        return self._offset_ema

    def _refresh_target(self) -> None:
        body = self.client.get_checkpoints_list() or {}
        cps, done = parse_checkpoints(body)
        self.target = next((c for c in cps if c.sequence > done), None)
        self._offset_ema = None

    def _attempt_checkpoint(self) -> None:
        if self.dry_run:
            log.info("[dry-run] would POST /checkpoint-reached")
            return
        res = self.client.checkpoint_reached()
        if not res.accepted:
            self._arrival_feedback = {"accepted": False}
            return
        self.checkpoints_reached += 1
        body = res.body or {}
        self._refresh_target()
        self._arrival_feedback = {
            "accepted": True,
            "completed": bool(body.get("mission_completed")),
            "has_next": self.target is not None,
        }
        log.info("checkpoint reached (%d so far); next: %s",
                 self.checkpoints_reached,
                 f"seq {self.target.sequence}" if self.target else "none")

    def _send(self, out, now: float) -> Optional[bool]:
        """Throttled /control write.

        Field-measured send cadence (auto-navigation-mini): re-sending an
        unchanged command at 10 Hz makes the motors constantly reset; sending
        it once lets the rover auto-stop after ~4 s of silence. Changed
        commands go out immediately; unchanged ones are refreshed about once
        per second.
        """
        if self.dry_run:
            return None
        cmd = (out.linear, out.angular)
        changed = (
            self._last_sent is None
            or abs(cmd[0] - self._last_sent[0]) > self.cfg.send_change_eps
            or abs(cmd[1] - self._last_sent[1]) > self.cfg.send_change_eps
        )
        if not changed and now - self._last_sent_t < self.cfg.send_keepalive_s:
            return None
        res = self.client.send_command(out.linear, out.angular)
        self._last_sent = cmd
        self._last_sent_t = now
        if res.accepted:
            self._reject_since = None
        elif self._reject_since is None:
            self._reject_since = now
        return res.accepted

    def _record(self, out, inp: ControlInputs, decision, snap, sent, now: float) -> None:
        record = {
            "ts": time.time(),
            "mono": now,
            "tick": self.ticks,
            "state": out.state.value,
            "reason": out.reason,
            "lat": self._last_fix[0],
            "lon": self._last_fix[1],
            "linear": round(out.linear, 3),
            "angular": round(out.angular, 3),
            "sent": sent,
            "dry_run": self.dry_run,
            "distance_m": None if inp.distance_m is None else round(inp.distance_m, 2),
            "goal_offset_deg": None if inp.goal_offset_deg is None
            else round(inp.goal_offset_deg, 1),
            "speed_ms": inp.speed_ms,
            "battery_pct": inp.battery_pct,
            "rpm_mean": None if inp.mean_abs_rpm is None else round(inp.mean_abs_rpm, 1),
            "mask_age_s": None if inp.mask_age_s is None else round(inp.mask_age_s, 2),
            "mask_seq": snap.seq if snap else None,
            "inference_s": snap.inference_s if snap else None,
            "corridor_scores": [round(float(s), 3) for s in decision.corridor_scores]
            if decision else None,
            "best_corridor": decision.best_corridor if decision else None,
            "guards": list(out.guard_notes),
            "checkpoint_seq": self.target.sequence if self.target else None,
            "checkpoints_reached": self.checkpoints_reached,
            "attempted_checkpoint": out.attempt_checkpoint,
        }
        self.shared.publish_decision(record)
        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(record) + "\n")
            self._jsonl_file.flush()
        if self.ticks % self.log_every == 1:
            dist = f"{inp.distance_m:.1f}m" if inp.distance_m is not None else "?"
            log.info("tick %d [%s] dist=%s cmd=(%.2f, %.2f) %s",
                     self.ticks, out.state.value, dist,
                     out.linear, out.angular, out.reason)
