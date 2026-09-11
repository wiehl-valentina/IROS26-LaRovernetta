"""Mission v2 runner: wires clients, threads and the run directory together.

Layout of a run:

    runs/<YYYYmmdd_HHMMSS>/
    ├── decisions.jsonl      one line per control tick
    ├── overlay_NNNNN.png    perception overlays every N frames
    ├── result.json          final summary (also feeds the report)
    └── report.html          self-contained post-run report (sana/report.py)

Threading: perception and (optionally) the live viewer run as daemon threads;
the control loop runs in the CALLING thread so Ctrl-C lands in the single
/control writer. The runner's ``finally`` sets the shared stop event and sends
the final stop command.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import MissionV2Config
from .control import ControlLoop
from .perception import PerceptionLoop
from .shared import SharedState

log = logging.getLogger(__name__)


@dataclasses.dataclass
class MissionV2Result:
    completed: bool
    reason: str
    ticks: int
    checkpoints_reached: int
    run_dir: str


class MissionV2Runner:
    def __init__(
        self,
        cfg: Optional[MissionV2Config] = None,
        base_url: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        dry_run: bool = False,
        start_mission: bool = False,
        runs_dir: str = "runs",
        viewer: bool = True,
        # test seams — production leaves these None
        predictor=None,
        control_session=None,
        perception_session=None,
    ) -> None:
        self.cfg = cfg or MissionV2Config()
        self.base_url = base_url
        self.checkpoint = checkpoint
        self.device = device
        self.dry_run = dry_run
        self.start_mission = start_mission
        self.runs_dir = Path(runs_dir)
        self.viewer_enabled = viewer
        self._predictor = predictor
        self._control_session = control_session
        self._perception_session = perception_session
        self.run_dir: Optional[Path] = None

    # ------------------------------------------------------------------- run

    def run(self, max_steps: Optional[int] = None,
            max_seconds: Optional[float] = None) -> MissionV2Result:
        from rover_traversability.client import RoverClient

        self.run_dir = self.runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        started_ts = time.time()

        predictor, warmup_s = self._build_predictor()
        cfg = self._derate_for_slow_inference(warmup_s)

        control_client = RoverClient(base_url=self.base_url,
                                     session=self._control_session)
        perception_client = RoverClient(base_url=self.base_url,
                                        session=self._perception_session)

        if self.start_mission and not self.dry_run:
            res = control_client.start_mission()
            log.info("/start-mission -> accepted=%s (%s)", res.accepted, res.detail)

        shared = SharedState()
        perception = PerceptionLoop(perception_client, predictor, shared, cfg,
                                    run_dir=self.run_dir)
        perception_thread = threading.Thread(
            target=perception.run, name="perception", daemon=True
        )

        control = ControlLoop(
            cfg,
            control_client,
            shared,
            perception_alive=perception_thread.is_alive,
            dry_run=self.dry_run,
            jsonl_path=self.run_dir / "decisions.jsonl",
        )

        viewer_server = self._start_viewer(shared, cfg)
        perception_thread.start()

        reason = "error"
        try:
            reason = control.run(max_steps=max_steps, max_seconds=max_seconds)
        finally:
            shared.stop.set()
            if not self.dry_run:
                control_client.stop()
            perception_thread.join(timeout=2.0)
            if viewer_server is not None:
                viewer_server.shutdown()

        result = MissionV2Result(
            completed=reason == "completed",
            reason=reason,
            ticks=control.ticks,
            checkpoints_reached=control.checkpoints_reached,
            run_dir=str(self.run_dir),
        )
        self._write_result(result, control_client, cfg, warmup_s, started_ts)
        self._write_report()
        return result

    # --------------------------------------------------------------- helpers

    def _build_predictor(self):
        if self._predictor is not None:
            return self._predictor, 0.0
        from rover_traversability import TraversabilityPredictor

        log.info("loading SAM-TP...")
        predictor = TraversabilityPredictor(
            checkpoint=self.checkpoint, device=self.device
        )
        warmup_s = predictor.warmup()
        log.info("model ready (warmup %.2fs)", warmup_s)
        return predictor, warmup_s

    def _derate_for_slow_inference(self, warmup_s: float) -> MissionV2Config:
        """CPU-only fallback: with 1-4 s/frame the mask is always seconds old,
        so widen the staleness budget and slow the rover proportionally."""
        cfg = self.cfg
        if warmup_s > 1.0:
            cfg = dataclasses.replace(
                cfg,
                stale_mask_s=max(cfg.stale_mask_s, 2.5 * warmup_s),
                max_linear=min(cfg.max_linear, 0.15),
            )
            log.warning(
                "slow inference (%.1fs): stale_mask_s=%.1f, max_linear=%.2f",
                warmup_s, cfg.stale_mask_s, cfg.max_linear,
            )
        return cfg

    def _start_viewer(self, shared: SharedState, cfg):
        if not self.viewer_enabled:
            return None
        try:
            from .viewer import start_viewer
        except ImportError:
            return None
        try:
            server = start_viewer(shared, port=cfg.viewer_port)
            log.info("live viewer at http://localhost:%d", cfg.viewer_port)
            return server
        except Exception:
            log.exception("viewer failed to start; continuing without it")
            return None

    def _write_result(self, result: MissionV2Result, client, cfg,
                      warmup_s: float, started_ts: float) -> None:
        from .control import parse_checkpoints

        body = client.get_checkpoints_list() or {}
        cps, done = parse_checkpoints(body)
        payload = {
            **dataclasses.asdict(result),
            "started_ts": started_ts,
            "ended_ts": time.time(),
            "dry_run": self.dry_run,
            "warmup_s": warmup_s,
            "config": dataclasses.asdict(cfg),
            "checkpoints": [dataclasses.asdict(c) for c in cps],
            "latest_scanned_checkpoint": done,
        }
        (self.run_dir / "result.json").write_text(json.dumps(payload, indent=2))

    def _write_report(self) -> None:
        try:
            from .report import write_report

            out = write_report(self.run_dir)
            log.info("report: %s", out)
        except ImportError:
            pass  # report module ships separately; runs stay valid without it
        except Exception:
            log.exception("report generation failed (run data is intact)")
