"""policy_test.py — run PolicyConfig(s) over a captured dataset, offline.

Reuses (no logic duplicated):
    rover_traversability.policy.PolicyConfig / suggest_command / CommandDecision

Design choice: SAM-TP inference runs ONCE per image and is cached to disk
(testing/common.py MaskCache) — policy_tuner.py can then sweep hundreds of
PolicyConfig combinations over the same dataset in seconds, because
suggest_command() is pure numpy and cheap; the model itself is never re-run
for a config that only changes policy thresholds.

Also importable: policy_tuner.py calls evaluate_config()/write_results()
directly instead of shelling out, so a single mask cache and a single
predictor instance serve the whole search.

Usage:
    python -m testing.policy_test --images dataset/session01 --out results/config_001
    python -m testing.policy_test --images dataset/session01 --out results --configs configs.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rover_traversability.policy import PolicyConfig, suggest_command

from .common import MaskCache, decision_to_row, draw_policy_overlay, list_dataset


def predictor_or_none(checkpoint=None, device=None):
    """Best-effort predictor; policy_test.py can still run purely off a
    pre-populated mask cache if torch/sam2/checkpoint aren't available."""
    try:
        from rover_traversability import TraversabilityPredictor
        predictor = TraversabilityPredictor(checkpoint=checkpoint, device=device)
        predictor.warmup()
        print(f"predictor ready on {predictor.device}")
        return predictor
    except Exception as exc:
        print(f"predictor unavailable ({exc}) — relying entirely on the mask "
              f"cache; any uncached frame will raise.")
        return None


def load_goal_offsets(records) -> dict:
    """Pulls goal_offset_deg out of sidecar metadata, if present (capture_test.py
    doesn't log one by default — telemetry alone isn't a bearing to a goal —
    but a hand-written labels/metadata file can add it per frame for testing
    goal-biased configs offline). Purely optional."""
    out = {}
    for rec in records:
        v = rec.metadata.get("goal_offset_deg")
        if v is not None:
            out[rec.name] = float(v)
    return out


def evaluate_config(
    cfg: PolicyConfig,
    records,
    mask_cache: MaskCache,
    out_dir: Path | None = None,
    save_overlays: bool = True,
    goal_offsets: dict | None = None,
) -> tuple[list[dict], dict]:
    """Run suggest_command(mask, cfg) over every frame in `records`.

    Returns (rows, summary): `rows` is one dict per frame (-> results.csv),
    `summary` is the aggregate metrics for this config (-> summary.csv row).
    """
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    reason_counts = {"forward": 0, "turning_to_corridor": 0, "blocked": 0, "no_data": 0}
    left = right = straight = 0
    linear_vals: list[float] = []
    angular_vals: list[float] = []
    best_scores: list[float] = []
    prev_angular = None
    oscillation_flips = 0
    moving_frames = 0

    for rec in records:
        rgb, mask = mask_cache.get(rec)
        goal_deg = goal_offsets.get(rec.name) if goal_offsets else None
        decision = suggest_command(mask, cfg, goal_offset_deg=goal_deg)

        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1

        if not decision.stop:
            linear_vals.append(decision.linear)
            angular_vals.append(decision.angular)
            if decision.angular > 0.03:
                left += 1
            elif decision.angular < -0.03:
                right += 1
            else:
                straight += 1
            if 0 <= decision.best_corridor < len(decision.corridor_scores):
                best_scores.append(decision.corridor_scores[decision.best_corridor])
            if prev_angular is not None:
                flipped = (prev_angular > 0.03 and decision.angular < -0.03) or \
                          (prev_angular < -0.03 and decision.angular > 0.03)
                if flipped:
                    oscillation_flips += 1
            prev_angular = decision.angular
            moving_frames += 1

        rows.append(decision_to_row(rec.name, decision, {
            "goal_offset_deg": goal_deg,
            "telemetry": json.dumps(rec.metadata.get("telemetry", {})),
        }))

        if save_overlays and out_dir is not None:
            overlay = draw_policy_overlay(rgb, mask, decision, cfg, goal_deg)
            overlay.save(out_dir / f"{rec.name}_overlay.jpg", quality=88)

    n = len(records)
    summary = {
        "n_frames": n,
        "forward_pct": 100.0 * reason_counts.get("forward", 0) / n,
        "turning_pct": 100.0 * reason_counts.get("turning_to_corridor", 0) / n,
        "blocked_pct": 100.0 * reason_counts.get("blocked", 0) / n,
        "no_data_pct": 100.0 * reason_counts.get("no_data", 0) / n,
        "left_pct": 100.0 * left / n,
        "right_pct": 100.0 * right / n,
        "straight_pct": 100.0 * straight / n,
        "avg_linear": (sum(linear_vals) / len(linear_vals)) if linear_vals else 0.0,
        "avg_abs_angular": (sum(abs(a) for a in angular_vals) / len(angular_vals)) if angular_vals else 0.0,
        "avg_best_score": (sum(best_scores) / len(best_scores)) if best_scores else 0.0,
        "oscillation_rate": oscillation_flips / max(1, moving_frames - 1),
        "stop_pct": 100.0 * (reason_counts.get("blocked", 0) + reason_counts.get("no_data", 0)) / n,
    }
    return rows, summary


def write_results(rows: list[dict], summary: dict, cfg: PolicyConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if rows:
        with open(out_dir / "results.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m testing.policy_test",
        description="Evaluate PolicyConfig(s) offline over a captured image dataset.",
    )
    p.add_argument("--images", required=True,
                    help="folder produced by capture_test.py (or plain images)")
    p.add_argument("--out", required=True, help="output folder for overlays/results")
    p.add_argument("--configs", default=None,
                    help="JSON file: a single {param: value} object, or a list of "
                         "{name, params} objects. Default: PolicyConfig() defaults.")
    p.add_argument("--cache-dir", default=None,
                    help="mask cache dir (default: <images>/.mask_cache)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-overlays", action="store_true")
    args = p.parse_args(argv)

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else images_dir / ".mask_cache"

    records = list_dataset(images_dir)
    print(f"loaded {len(records)} frame(s) from {images_dir}")

    predictor = predictor_or_none(args.checkpoint, args.device)
    mask_cache = MaskCache(cache_dir, predictor=predictor)
    goal_offsets = load_goal_offsets(records)

    if args.configs:
        raw = json.loads(Path(args.configs).read_text())
        configs = raw if isinstance(raw, list) else [{"name": "config_001", "params": raw}]
    else:
        configs = [{"name": "config_001", "params": {}}]

    summary_rows = []
    for entry in configs:
        name = entry.get("name", "config")
        cfg = PolicyConfig(**entry.get("params", {}))
        cfg_out = out_dir / name
        rows, summary = evaluate_config(
            cfg, records, mask_cache, out_dir=cfg_out,
            save_overlays=not args.no_overlays, goal_offsets=goal_offsets,
        )
        write_results(rows, summary, cfg, cfg_out)
        summary_rows.append({"config": name, **summary})
        print(f"{name}: forward={summary['forward_pct']:.1f}% "
              f"turn={summary['turning_pct']:.1f}% stop={summary['stop_pct']:.1f}% "
              f"avg_score={summary['avg_best_score']:.2f} osc={summary['oscillation_rate']:.2f}")

    if summary_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"summary written to {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
