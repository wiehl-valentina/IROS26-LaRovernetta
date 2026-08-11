"""policy_tuner.py — progressive search over PolicyConfig, ranked by a
safety-first score.

Reuses policy_test.py's evaluate_config()/write_results() directly (in-process,
no subprocess) so a single mask cache and a single predictor serve the whole
search — that's what makes sweeping many configs cheap (see testing/common.py
MaskCache and the module docstring in policy_test.py).

Search strategy (per your spec, section 5): NOT a full grid over all 14
params at once. Coordinate/progressive search: optimize one parameter at a
time (holding the others fixed at the current best), fix it, move to the
next. DEFAULT_STAGES below is the exact order you listed; extend or reorder
it freely — the search loop doesn't care how many stages there are.

Scoring (section 4 — must NOT just reward "goes forward fastest"):
  - Without labels: a heuristic combining (a) average confidence of the
    chosen corridor when moving, (b) inverse of left/right oscillation
    frame-to-frame, and (c) an "activity" term that peaks at a moderate
    forward_pct and falls off in BOTH directions — so a config that never
    stops scores no better than one that's over-cautious.
  - With --labels (a hand-made frame,expected_reason[,expected_side] CSV —
    see compare_to_labels docstring): real accuracy against ground truth
    dominates, and "should have stopped but didn't" is penalized twice as
    hard as "stopped when it could have gone", because a false-safe decision
    is worse than a false-stop.

Usage:
    python -m testing.policy_tuner --images dataset/session01 --out tuning_results
    python -m testing.policy_tuner --images dataset/session01 --out tuning_results \
        --labels dataset/session01/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rover_traversability.policy import PolicyConfig

from .common import list_dataset, MaskCache
from .policy_test import evaluate_config, load_goal_offsets, predictor_or_none, write_results

# Progressive search order + candidate values, exactly as specified: tune
# roi_top first, fix the best, tune drivable_thresh, fix it, etc. Values for
# stop_center_fraction/max_linear were not given explicitly in the spec's
# example but the stage order was ("optimizar stop_center_fraction",
# "optimizar k_angular", "optimizar max_linear") — grids chosen as reasonable
# neighborhoods around the PolicyConfig() defaults.
DEFAULT_STAGES: list[tuple[str, list[float]]] = [
    ("roi_top", [0.45, 0.50, 0.55, 0.60, 0.65]),
    ("drivable_thresh", [0.40, 0.45, 0.50, 0.55, 0.60]),
    ("stop_center_fraction", [0.30, 0.35, 0.40, 0.45, 0.50]),
    ("k_angular", [0.8, 1.0, 1.2, 1.4, 1.6]),
    ("max_linear", [0.3, 0.4, 0.5, 0.6]),
]


def score_summary(summary: dict) -> float:
    """Unsupervised, safety-first heuristic score in roughly [0, 100].

    Deliberately NOT "higher forward_pct / higher avg_linear is better" —
    that alone would reward blind driving. Confidence and stability are
    weighted above raw activity, and activity itself is scored against a
    healthy midpoint rather than maximized.
    """
    confidence = summary["avg_best_score"] * 100.0
    stability = max(0.0, 100.0 - summary["oscillation_rate"] * 200.0)
    fwd = summary["forward_pct"]
    activity = max(0.0, 100.0 - abs(fwd - 65.0))  # peaks at 65% forward
    return 0.5 * confidence + 0.3 * stability + 0.2 * activity


def compare_to_labels(rows: list[dict], labels_path: Path) -> dict:
    """Score against manually-labeled ground truth.

    labels_path: CSV with a `frame` column matching FrameRecord.name, and
    `expected_reason` in {forward, turning_to_corridor, blocked}. This is the
    information NOTHING in the pipeline can infer automatically — someone
    has to look at the overlay/frame and decide "was there really a safe
    forward path here, or should this have stopped/turned". Optional
    `expected_side` (left/right/straight) refines it further but isn't
    required for the safety-priority scoring below.

    A "should have stopped but drove" (unsafe_misses) is weighted double
    against a "stopped but could have driven" (overcautious_misses), per the
    stated priority: safety over speed.
    """
    labels: dict[str, dict] = {}
    with open(labels_path, newline="") as fh:
        for r in csv.DictReader(fh):
            labels[r["frame"]] = r

    total = correct = unsafe_misses = overcautious = 0
    for row in rows:
        lbl = labels.get(row["frame"])
        if lbl is None:
            continue
        total += 1
        expected_stop = lbl["expected_reason"] == "blocked"
        actual_stop = row["stop"] in (True, "True", "true")
        if lbl["expected_reason"] == row["reason"]:
            correct += 1
        elif expected_stop and not actual_stop:
            unsafe_misses += 1
        elif actual_stop and not expected_stop:
            overcautious += 1

    if total == 0:
        return {"labeled_frames": 0, "accuracy": None, "safety_penalized_accuracy": None}

    accuracy = correct / total
    penalized = max(0.0, (correct - 2 * unsafe_misses)) / total
    return {
        "labeled_frames": total,
        "correct": correct,
        "unsafe_misses": unsafe_misses,
        "overcautious_misses": overcautious,
        "accuracy": accuracy,
        "safety_penalized_accuracy": penalized,
    }


def progressive_search(
    records,
    mask_cache: MaskCache,
    base_cfg: PolicyConfig,
    stages,
    out_dir: Path,
    labels_path: Path | None = None,
    goal_offsets: dict | None = None,
) -> tuple[PolicyConfig, list[dict]]:
    current_params = asdict(base_cfg)
    history: list[dict] = []
    trial_idx = 0

    for param_name, values in stages:
        print(f"\n== optimizing {param_name} (others fixed) ==")
        best_val, best_score, best_summary = None, float("-inf"), None

        for v in values:
            trial_idx += 1
            trial_params = dict(current_params)
            trial_params[param_name] = v
            cfg = PolicyConfig(**trial_params)
            trial_name = f"config_{trial_idx:03d}_{param_name}_{v}"
            cfg_out = out_dir / f"stage_{param_name}" / trial_name

            rows, summary = evaluate_config(
                cfg, records, mask_cache, out_dir=cfg_out,
                save_overlays=True, goal_offsets=goal_offsets,
            )

            if labels_path:
                label_stats = compare_to_labels(rows, labels_path)
                pa = label_stats.get("safety_penalized_accuracy")
                score = 100.0 * pa if pa is not None else score_summary(summary)
                summary.update(label_stats)
            else:
                score = score_summary(summary)

            summary["score"] = score
            write_results(rows, summary, cfg, cfg_out)
            history.append({"stage": param_name, "trial": trial_name, "value": v, **summary})

            print(f"  {trial_name}: score={score:.1f} fwd={summary['forward_pct']:.0f}% "
                  f"turn={summary['turning_pct']:.0f}% stop={summary['stop_pct']:.0f}% "
                  f"conf={summary['avg_best_score']:.2f} osc={summary['oscillation_rate']:.2f}")

            if score > best_score:
                best_score, best_val, best_summary = score, v, summary

        current_params[param_name] = best_val
        print(f"-> best {param_name} = {best_val} (score {best_score:.1f})")

    final_cfg = PolicyConfig(**current_params)
    return final_cfg, history


def write_ranking(history: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(history, key=lambda r: r["score"], reverse=True)
    fieldnames = list(ranked[0].keys()) if ranked else []
    with open(out_dir / "all_trials_ranked.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    print("\n" + "=" * 78)
    print(f"{'CONFIG':<28}{'SCORE':>8}{'FORWARD':>10}{'TURN':>8}{'STOP':>8}{'CONF':>8}")
    print("-" * 78)
    for r in ranked[:15]:
        print(f"{r['trial']:<28}{r['score']:>8.1f}{r['forward_pct']:>9.1f}%"
              f"{r['turning_pct']:>7.1f}%{r['stop_pct']:>7.1f}%{r['avg_best_score']:>8.2f}")
    print("=" * 78)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m testing.policy_tuner",
        description="Progressive search + ranking over PolicyConfig, evaluated on a captured dataset.",
    )
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-config", default=None,
                    help="JSON with starting {param: value} overrides (default: PolicyConfig() defaults)")
    p.add_argument("--labels", default=None,
                    help="CSV with frame,expected_reason[,expected_side] for supervised scoring")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else images_dir / ".mask_cache"

    records = list_dataset(images_dir)
    print(f"loaded {len(records)} frame(s) from {images_dir}")

    predictor = predictor_or_none(args.checkpoint, args.device)
    mask_cache = MaskCache(cache_dir, predictor=predictor)
    goal_offsets = load_goal_offsets(records)

    base_params = json.loads(Path(args.base_config).read_text()) if args.base_config else {}
    base_cfg = PolicyConfig(**base_params)

    labels_path = Path(args.labels) if args.labels else None
    if labels_path and not labels_path.is_file():
        print(f"--labels path does not exist: {labels_path}; falling back to unsupervised scoring")
        labels_path = None

    final_cfg, history = progressive_search(
        records, mask_cache, base_cfg, DEFAULT_STAGES, out_dir,
        labels_path=labels_path, goal_offsets=goal_offsets,
    )

    write_ranking(history, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_config.json").write_text(json.dumps(asdict(final_cfg), indent=2))
    print(f"\nbest config written to {out_dir / 'best_config.json'}:")
    print(json.dumps(asdict(final_cfg), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
