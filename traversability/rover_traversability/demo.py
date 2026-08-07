"""Runnable demos: python -m rover_traversability.demo <command>

Commands:
  predict IMAGE     one-shot inference on an image file, saves the overlay
  live              fetch frames from the SDK, predict, save overlays — sends NOTHING
  drive             reactive obstacle-avoidance loop — MOVES THE ROVER (opt-in flag)
  mission           GPS-checkpoint mission loop — MOVES THE ROVER (opt-in flag)
"""

from __future__ import annotations

import argparse
import sys
import time

MOVE_FLAG = "--yes-i-want-the-rover-to-move"


def _build_predictor(args):
    from .predictor import TraversabilityPredictor

    print("Loading SAM-TP...", flush=True)
    predictor = TraversabilityPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        contrast_refine=not args.no_refine,
    )
    warm = predictor.warmup()
    print(f"Model ready on {predictor.device} (warmup inference: {warm:.2f}s)")
    return predictor


def cmd_predict(args) -> int:
    from PIL import Image

    predictor = _build_predictor(args)
    result = predictor.predict(args.image)
    drivable_pct = 100.0 * float((result.mask > 0.5).mean())
    print(f"inference: {result.inference_s:.3f}s on {result.device}")
    print(f"drivable pixels: {drivable_pct:.1f}%")

    from .policy import suggest_command

    decision = suggest_command(result.mask)
    print(f"suggested command: {decision}")

    Image.fromarray(result.overlay).save(args.out)
    print(f"overlay written to {args.out}")
    return 0


def cmd_live(args) -> int:
    from .client import RoverClient
    from .strategy import TraversabilityStrategy

    predictor = _build_predictor(args)
    strategy = TraversabilityStrategy(
        client=RoverClient(),
        predictor=predictor,
        drive=False,
        save_overlays_dir=args.save_dir,
    )
    print(f"Live dry-run: overlays -> {args.save_dir}, no commands sent. Ctrl-C to stop.")
    frames = 0
    try:
        while args.max_frames is None or frames < args.max_frames:
            payload = strategy.get_image_payload()
            if payload:
                strategy.analyze(payload)
                frames += 1
            else:
                print("no frame from SDK (is it running? mission started?)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    print(f"done ({frames} frames)")
    return 0


def _require_move_flag(args, what: str) -> bool:
    if not args.i_want_the_rover_to_move:
        print(f"REFUSING to {what}: this sends real motion commands.")
        print(f"Re-run with {MOVE_FLAG} in a safe, open area with a finger on Ctrl-C.")
        return False
    return True


def cmd_drive(args) -> int:
    if not _require_move_flag(args, "drive"):
        return 2
    from .client import RoverClient
    from .strategy import TraversabilityStrategy

    client = RoverClient()
    predictor = _build_predictor(args)
    strategy = TraversabilityStrategy(client=client, predictor=predictor, drive=True,
                                      save_overlays_dir=args.save_dir)
    print("Driving (reactive avoidance only, no goal). Ctrl-C stops the rover.")
    iterations = 0
    try:
        while args.max_iterations is None or iterations < args.max_iterations:
            payload = strategy.get_image_payload()
            if payload:
                strategy.analyze(payload)
            iterations += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        client.stop()
        print("rover stopped")
    return 0


def cmd_mission(args) -> int:
    if not _require_move_flag(args, "run a mission"):
        return 2
    from .client import RoverClient
    from .mission import MissionRunner

    client = RoverClient()
    if args.start_mission:
        res = client.start_mission()
        print(f"start-mission: accepted={res.accepted} {res.detail}")

    predictor = _build_predictor(args)

    def report(info: dict) -> None:
        d = info["decision"]
        dist = info["distance_m"]
        goal = info["goal_offset_deg"]
        dist_s = "?" if dist is None else f"{dist:6.1f}m"
        goal_s = "?" if goal is None else f"{goal:+6.1f}deg"
        print(
            f"step {info['step']:4d} | dist {dist_s} | goal_off {goal_s} | "
            f"{d.reason}: lin={d.linear:+.2f} ang={d.angular:+.2f}"
        )

    runner = MissionRunner(
        client=client,
        predictor=predictor,
        arrive_attempt_m=args.arrive_attempt_m,
        interval_s=args.interval,
        max_steps=args.max_steps,
        on_step=report,
    )
    print("Mission running. Ctrl-C stops the rover.")
    try:
        result = runner.run()
    except KeyboardInterrupt:
        client.stop()
        print("interrupted; rover stopped")
        return 1
    print(
        f"mission result: completed={result.completed} "
        f"checkpoints={result.checkpoints_reached} steps={result.steps} ({result.reason})"
    )
    return 0 if result.completed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rover_traversability.demo")
    parser.add_argument("--checkpoint", default=None, help="path to the SAM-TP .pt checkpoint")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    parser.add_argument("--no-refine", action="store_true", help="disable contrast refinement")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="one-shot inference on an image file")
    p.add_argument("image")
    p.add_argument("--out", default="overlay.png")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("live", help="live overlay loop against the SDK (sends nothing)")
    p.add_argument("--save-dir", default="trav_out")
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(fn=cmd_live)

    p = sub.add_parser("drive", help="reactive driving loop (MOVES THE ROVER)")
    p.add_argument(MOVE_FLAG, dest="i_want_the_rover_to_move", action="store_true")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--max-iterations", type=int, default=None)
    p.set_defaults(fn=cmd_drive)

    p = sub.add_parser("mission", help="GPS checkpoint mission (MOVES THE ROVER)")
    p.add_argument(MOVE_FLAG, dest="i_want_the_rover_to_move", action="store_true")
    p.add_argument("--start-mission", action="store_true", help="POST /start-mission first")
    p.add_argument("--arrive-attempt-m", type=float, default=8.0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=None)
    p.set_defaults(fn=cmd_mission)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
