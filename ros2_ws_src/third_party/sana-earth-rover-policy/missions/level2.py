#!/usr/bin/env python3
"""Mission level 2: decoupled perception/control with field-proven behaviors.

Same GPS-checkpoint mission as level1.py, rebuilt on the sana package:
5 Hz control decoupled from SAM-TP inference, align/K-turn/pivot maneuvers,
stuck recovery, physical speed cap, live viewer and a post-run HTML report.

Usage:
    python missions/level2.py --dry-run                      # no movement at all
    python missions/level2.py --start-mission --yes-i-want-the-rover-to-move
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sana.config import MissionV2Config  # noqa: E402

MOVE_FLAG = "--yes-i-want-the-rover-to-move"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(MOVE_FLAG, dest="move", action="store_true",
                   help="explicit confirmation that the rover will move")
    p.add_argument("--dry-run", action="store_true",
                   help="full stack (perception, FSM, viewer, logs) but never "
                        "POSTs /control or /checkpoint-reached")
    p.add_argument("--base-url", default=None,
                   help="SDK URL (default: $ROVER_BASE_URL or localhost:8000)")
    p.add_argument("--checkpoint", default=None, help="path to the SAM-TP .pt")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--start-mission", action="store_true",
                   help="POST /start-mission before driving")
    p.add_argument("--max-linear", type=float, default=None)
    p.add_argument("--max-angular", type=float, default=None)
    p.add_argument("--arrive-attempt-m", type=float, default=None)
    p.add_argument("--control-hz", type=float, default=None)
    p.add_argument("--renormalize", action="store_true",
                   help="per-frame percentile stretch for soft-confidence scenes")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--viewer-port", type=int, default=None)
    return p


def build_config(args) -> MissionV2Config:
    overrides = {}
    for arg_name, field in [
        ("max_linear", "max_linear"),
        ("max_angular", "max_angular"),
        ("arrive_attempt_m", "arrive_attempt_m"),
        ("control_hz", "control_hz"),
        ("viewer_port", "viewer_port"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field] = value
    if args.renormalize:
        overrides["renormalize_percentile"] = True
    return dataclasses.replace(MissionV2Config(), **overrides)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if not args.move and not args.dry_run:
        print(f"This mission MOVES the rover. Re-run with {MOVE_FLAG} "
              f"(or --dry-run to watch without moving).")
        return 2

    from sana.runner import MissionV2Runner

    runner = MissionV2Runner(
        cfg=build_config(args),
        base_url=args.base_url,
        checkpoint=args.checkpoint,
        device=args.device,
        dry_run=args.dry_run,
        start_mission=args.start_mission,
        runs_dir=args.runs_dir,
        viewer=not args.no_viewer,
    )
    try:
        result = runner.run(max_steps=args.max_steps, max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        # The runner's finally already sent the stop command.
        print("\ninterrupted — stop sent")
        return 130

    print(json.dumps(dataclasses.asdict(result), indent=2))
    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
