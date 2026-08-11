"""capture_test.py — safe, dry-run frame capture from a live rover.

Only reuses existing package code:
    rover_traversability.client.RoverClient  - HTTP to the SDK (GET only, here)
    rover_traversability.images.to_rgb       - decode the frame
    rover_traversability.policy.suggest_command - OPTIONAL reference logging

SAFETY: this script never sends /control. It never even imports
RoverClient.send_command usage — the only client calls made are
get_front_frame_b64() and get_data(), both GETs. The optional --with-policy
path calls suggest_command() purely to log what the policy *would* decide;
that decision is never sent anywhere.

Usage:
    python -m testing.capture_test --save-dir dataset/session01
    python -m testing.capture_test --save-dir dataset/session01 --interval 30 --with-policy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rover_traversability.client import RoverClient
from rover_traversability.images import to_rgb
from rover_traversability.policy import PolicyConfig, suggest_command


def _build_predictor(checkpoint: str | None, device: str | None):
    from rover_traversability import TraversabilityPredictor  # lazy, torch-optional

    print("Loading SAM-TP for reference logging only (no commands will be sent)...")
    predictor = TraversabilityPredictor(checkpoint=checkpoint, device=device)
    predictor.warmup()
    print(f"Model ready on {predictor.device}.")
    return predictor


def capture_loop(args) -> int:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    client = RoverClient()
    predictor = None
    if args.with_policy:
        try:
            predictor = _build_predictor(args.checkpoint, args.device)
        except Exception as exc:
            # ImportError / SamNotInstalledError / CheckpointNotFoundError / ...
            print(f"--with-policy requested but unavailable ({exc}); "
                  f"continuing WITHOUT policy logging.")
            predictor = None

    policy_cfg = PolicyConfig()
    frame_idx = 0
    print(f"Capturing to {save_dir} every {args.interval:.0f}s. "
          f"DRY-RUN ONLY — no /control commands are ever sent. Ctrl-C to stop.")

    try:
        while args.max_frames is None or frame_idx < args.max_frames:
            t_start = time.time()
            payload = client.get_front_frame_b64()

            if not payload:
                print("no frame from SDK (is it running? mission started?)")
            else:
                ts = time.time()
                frame_idx += 1
                stem = f"frame_{frame_idx:05d}_{int(ts)}"
                image_path = save_dir / f"{stem}.jpg"

                try:
                    rgb = to_rgb(payload)
                except Exception as exc:
                    print(f"decode failed, skipping frame: {exc}")
                    continue

                from PIL import Image
                Image.fromarray(rgb).save(image_path, format="JPEG", quality=92)

                telemetry = client.get_data() or {}
                record = {
                    "timestamp": ts,
                    "frame_name": image_path.name,
                    "resolution": {"height": int(rgb.shape[0]), "width": int(rgb.shape[1])},
                    "telemetry": telemetry,
                }

                if predictor is not None:
                    try:
                        result = predictor.predict(rgb)
                        decision = suggest_command(result.mask, policy_cfg)
                        record["policy_reference"] = {
                            "note": "default PolicyConfig() — reference only, "
                                    "the real tuning happens offline in policy_test.py",
                            "reason": decision.reason,
                            "linear": decision.linear,
                            "angular": decision.angular,
                            "stop": decision.stop,
                            "best_corridor": decision.best_corridor,
                            "corridor_scores": list(decision.corridor_scores),
                        }
                    except Exception as exc:
                        record["policy_reference_error"] = str(exc)

                (save_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
                print(f"[{frame_idx}] saved {image_path.name} "
                      f"({rgb.shape[1]}x{rgb.shape[0]})")

            elapsed = time.time() - t_start
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)
    except KeyboardInterrupt:
        print("interrupted by user")

    print(f"done. {frame_idx} frame(s) saved to {save_dir}. No commands were ever sent.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m testing.capture_test",
        description="Dry-run periodic frame capture from the rover, for later offline policy testing.",
    )
    p.add_argument("--save-dir", required=True,
                    help="folder to write frames + sidecar JSON into")
    p.add_argument("--interval", type=float, default=30.0,
                    help="seconds between captures (default: 30)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--with-policy", action="store_true",
                    help="also run SAM-TP + suggest_command() per frame for reference "
                         "logging only (requires torch/sam2/checkpoint; never sends commands)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    return capture_loop(args)


if __name__ == "__main__":
    sys.exit(main())
