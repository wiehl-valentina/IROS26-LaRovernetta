"""Simple GPS-checkpoint mission runner: traversability + goal bearing.

The loop each step:
1. Read telemetry (lat/lon/speed/orientation) and the front frame.
2. Estimate heading via GPS course-over-ground (see geo.py for why the
   magnetometer is not trusted).
3. Compute the bearing offset to the next checkpoint.
4. If the goal is far to the side/behind: arc-turn toward it (a slow arc, not
   a pure spin, so GPS-COG keeps updating — a rover spinning in place gets no
   heading feedback).
5. Otherwise: SAM-TP mask -> goal-biased steering command.
6. Within ``arrive_attempt_m`` of the checkpoint, try POST /checkpoint-reached;
   the backend enforces the real radius and rejects with 400 while too far —
   that is the designed retry loop, keep driving.

This is deliberately the *simple* mission solution. The roadmap upgrade path
(docs/ROADMAP.md) replaces step 5 with BEV projection + the vendored
genie_path_planner.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from math import degrees
from typing import Callable, Optional

from .client import RoverClient
from .geo import HeadingEstimator, gps_bearing_and_distance, wrap_angle_deg
from .policy import CommandDecision, PolicyConfig, suggest_command

log = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    id: int
    sequence: int
    latitude: float
    longitude: float


@dataclass
class MissionResult:
    completed: bool
    checkpoints_reached: int
    steps: int
    reason: str
    history: list = field(default_factory=list)


def _parse_checkpoints(body: dict) -> tuple[list[Checkpoint], int | None]:
    """Parse /checkpoints-list. Lat/lon arrive as STRINGS — cast them."""
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
    return cps, (int(latest) if latest is not None else None)


class MissionRunner:
    def __init__(
        self,
        client: RoverClient | None = None,
        predictor=None,
        policy: PolicyConfig | None = None,
        arrive_attempt_m: float = 8.0,
        interval_s: float = 0.5,
        turn_in_place_deg: float = 70.0,
        arc_turn_linear: float = 0.15,
        max_steps: int | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> None:
        self.client = client or RoverClient()
        if predictor is None:
            from .predictor import TraversabilityPredictor

            predictor = TraversabilityPredictor()
        self.predictor = predictor
        self.policy = policy or PolicyConfig()
        self.arrive_attempt_m = float(arrive_attempt_m)
        self.interval_s = float(interval_s)
        self.turn_in_place_deg = float(turn_in_place_deg)
        self.arc_turn_linear = float(arc_turn_linear)
        self.max_steps = max_steps
        self.on_step = on_step
        self.heading = HeadingEstimator()

    # ------------------------------------------------------------------ public

    def run(self) -> MissionResult:
        """Drive checkpoints until the mission completes. Always stops the rover."""
        try:
            return self._run_inner()
        finally:
            self.client.stop()

    # ----------------------------------------------------------------- internal

    def _next_checkpoint(self) -> tuple[Optional[Checkpoint], int]:
        body = self.client.get_checkpoints_list()
        if not body:
            return None, 0
        cps, latest = _parse_checkpoints(body)
        done = latest or 0
        for cp in cps:
            if cp.sequence > done:
                return cp, done
        return None, done

    def _run_inner(self) -> MissionResult:
        target, done_count = self._next_checkpoint()
        if target is None:
            return MissionResult(
                completed=done_count > 0,
                checkpoints_reached=done_count,
                steps=0,
                reason="no pending checkpoints (mission not started, or already complete)",
            )
        log.info("next checkpoint: seq %s at (%s, %s)",
                 target.sequence, target.latitude, target.longitude)

        steps = 0
        reached = done_count
        history: list = []

        while True:
            if self.max_steps is not None and steps >= self.max_steps:
                return MissionResult(False, reached, steps, "max_steps reached", history)
            steps += 1
            t_start = time.perf_counter()

            data = self.client.get_data() or {}
            lat, lon = data.get("latitude"), data.get("longitude")
            heading_deg = self.heading.update(
                lat, lon, data.get("speed"), data.get("orientation")
            )

            goal_offset_deg: float | None = None
            distance_m: float | None = None
            if lat is not None and lon is not None:
                bearing_rad, distance_m = gps_bearing_and_distance(
                    float(lat), float(lon), target.latitude, target.longitude
                )
                if heading_deg is not None:
                    goal_offset_deg = wrap_angle_deg(degrees(bearing_rad) - heading_deg)

            # Arrival attempt — backend enforces the true radius.
            if distance_m is not None and distance_m <= self.arrive_attempt_m:
                res = self.client.checkpoint_reached()
                if res.accepted:
                    reached += 1
                    body = res.body or {}
                    if body.get("mission_completed"):
                        return MissionResult(True, reached, steps, "mission completed", history)
                    target, _ = self._next_checkpoint()
                    if target is None:
                        return MissionResult(True, reached, steps, "all checkpoints done", history)
                    log.info("checkpoint reached; next: seq %s", target.sequence)
                    continue
                # else: not close enough yet per the backend — keep driving.

            decision = self._decide(goal_offset_deg)
            if decision.stop:
                self.client.stop()
            else:
                self.client.send_command(decision.linear, decision.angular)

            step_info = {
                "step": steps,
                "distance_m": distance_m,
                "heading_deg": heading_deg,
                "goal_offset_deg": goal_offset_deg,
                "decision": decision,
            }
            history.append(step_info)
            if self.on_step is not None:
                self.on_step(step_info)

            elapsed = time.perf_counter() - t_start
            if elapsed < self.interval_s:
                time.sleep(self.interval_s - elapsed)

    def _decide(self, goal_offset_deg: float | None) -> CommandDecision:
        # Goal far off to the side or behind: the camera can't see there, so
        # arc-turn toward it (motion keeps GPS-COG heading alive).
        if goal_offset_deg is not None and abs(goal_offset_deg) > self.turn_in_place_deg:
            turn = -self.policy.max_angular if goal_offset_deg > 0 else self.policy.max_angular
            return CommandDecision(
                linear=self.arc_turn_linear,
                angular=turn,
                stop=False,
                reason="arc_turn_to_goal",
            )

        frame = self.client.get_front_frame()
        if frame is None:
            return CommandDecision(0.0, 0.0, True, "no_frame")
        try:
            result = self.predictor.predict(frame)
        except Exception as exc:
            log.error("prediction failed: %s", exc)
            return CommandDecision(0.0, 0.0, True, "predict_error")
        return suggest_command(result.mask, self.policy, goal_offset_deg=goal_offset_deg)
