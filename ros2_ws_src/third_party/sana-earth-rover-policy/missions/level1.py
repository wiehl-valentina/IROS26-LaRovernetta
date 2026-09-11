#!/usr/bin/env python3
"""Mission level 1: drive the active mission's GPS checkpoints (baseline).

Perception (SAM-TP) -> corridor policy with goal bias -> /control, in one
serial loop. All the logic lives in rover-traversability; this script adds
what field runs need: per-step JSONL logging, periodic overlays and a final
summary, all under runs/<timestamp>/.

This is the reference/baseline runner. For the decoupled-loop version with
field behaviors (align maneuvers, stuck recovery, live viewer, HTML report)
use missions/level2.py.

Usage:
    python missions/level1.py --start-mission --yes-i-want-the-rover-to-move
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from rover_traversability import TraversabilityPredictor
from rover_traversability.client import RoverClient
from rover_traversability.mission import MissionRunner
from rover_traversability.policy import PolicyConfig

MOVE_FLAG = "--yes-i-want-the-rover-to-move"


class RecordingPredictor:
    """Wraps the predictor keeping the last result, so overlays can be saved
    from the on_step callback without touching MissionRunner."""

    def __init__(self, inner: TraversabilityPredictor) -> None:
        self.inner = inner
        self.last = None

    def predict(self, payload):
        self.last = self.inner.predict(payload)
        return self.last


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(MOVE_FLAG, dest="move", action="store_true",
                   help="explicit confirmation that the rover will move")
    p.add_argument("--base-url", default=None,
                   help="SDK URL (default: $ROVER_BASE_URL or localhost:8000)")
    p.add_argument("--checkpoint", default=None, help="path to the .pt (default: cache/HF)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--start-mission", action="store_true",
                   help="POST /start-mission before driving")
    p.add_argument("--arrive-attempt-m", type=float, default=8.0)
    p.add_argument("--interval", type=float, default=0.5, help="seconds between steps")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-linear", type=float, default=0.5)
    p.add_argument("--max-angular", type=float, default=0.5)
    p.add_argument("--overlay-every", type=int, default=10,
                   help="save an overlay every N steps (0 = never)")
    p.add_argument("--runs-dir", default="runs")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.move:
        print(f"This mission MOVES the rover. Re-run adding {MOVE_FLAG}")
        return 2

    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    client = RoverClient(base_url=args.base_url)
    predictor = RecordingPredictor(
        TraversabilityPredictor(checkpoint=args.checkpoint, device=args.device)
    )
    print("warmup...", flush=True)
    warmup_s = predictor.inner.warmup()
    print(f"warmup: {warmup_s:.2f}s on {predictor.inner._device}")

    if args.start_mission:
        res = client.start_mission()
        print(f"/start-mission -> accepted={res.accepted} ({res.detail})")

    log_file = (run_dir / "decisions.jsonl").open("a")

    def on_step(info: dict) -> None:
        d = info["decision"]
        record = {
            "ts": time.time(),
            "step": info["step"],
            "distance_m": info["distance_m"],
            "heading_deg": info["heading_deg"],
            "goal_offset_deg": info["goal_offset_deg"],
            "linear": d.linear,
            "angular": d.angular,
            "stop": d.stop,
            "reason": d.reason,
            "corridor_scores": [round(float(s), 3) for s in d.corridor_scores],
            "best_corridor": d.best_corridor,
            "inference_s": getattr(predictor.last, "inference_s", None),
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if info["step"] % 20 == 1:
            dist = f"{info['distance_m']:.1f}m" if info["distance_m"] is not None else "?"
            print(f"step {info['step']}: dist={dist} cmd=({d.linear:.2f}, {d.angular:.2f}) {d.reason}")
        if args.overlay_every and info["step"] % args.overlay_every == 0 and predictor.last is not None:
            from PIL import Image
            Image.fromarray(predictor.last.overlay).save(run_dir / f"overlay_{info['step']:05d}.png")

    runner = MissionRunner(
        client=client,
        predictor=predictor,
        policy=PolicyConfig(max_linear=args.max_linear, max_angular=args.max_angular),
        arrive_attempt_m=args.arrive_attempt_m,
        interval_s=args.interval,
        max_steps=args.max_steps,
        on_step=on_step,
    )

    try:
        result = runner.run()
    except KeyboardInterrupt:
        client.stop()
        print("\ninterrupted — stop sent")
        return 130
    finally:
        log_file.close()

    summary = dataclasses.asdict(result)
    summary.pop("history", None)
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
