"""Image-space steering policy over a traversability mask. No calibration needed.

The mask is the SAM-TP output: HxW float32 in [0, 1], 1 = drivable. The policy
looks at the lower part of the image (near ground), splits it into vertical
corridors, and steers toward the most drivable corridor — optionally biased
toward a GPS goal direction. It is pure numpy, stateless and deterministic.

Two field-proven rules are inherited from the auto-navigation-mini research
stack's BEV controller:
- Steering targets the score-weighted centroid *within* the chosen corridor,
  never the centroid of the whole view — a bimodal mask (two openings) must
  not aim the rover at the wall between them.
- The stop test is fraction-based (a percentage of the center corridor must be
  blocked), never "any blocked pixel" — real model output always has some
  blocked pixels, and an any-pixel test halts the rover on every frame.

Command conventions match the SDK: linear/angular in [-1, 1], angular
positive = LEFT turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Horizontal FOV of the Mini+ front camera (from the packaged intrinsics:
# 2*atan(512 / 488.924) for the 1024-wide frame). Used to map a goal bearing
# offset into an image x position.
DEFAULT_HFOV_DEG = 92.7


@dataclass(frozen=True)
class PolicyConfig:
    roi_top: float = 0.55            # analyze rows below this fraction of height
    drivable_thresh: float = 0.5     # mask value above which a pixel counts as drivable
    num_corridors: int = 9           # odd number of vertical bands across the ROI
    stop_center_fraction: float = 0.40  # stop if >= this fraction of center band is blocked
    min_corridor_score: float = 0.35    # a corridor below this is not worth turning toward
    bottom_weight: float = 2.0       # near (bottom) rows weighted up, linear ramp 1 -> this
    min_valid_pixels: int = 200      # fewer finite ROI pixels than this -> stop ("no_data")
    max_linear: float = 0.5
    min_linear: float = 0.15
    max_angular: float = 0.5
    k_angular: float = 1.2           # steering gain on normalized image-x offset
    hfov_deg: float = DEFAULT_HFOV_DEG
    goal_sigma: float = 0.5          # width of the goal-bias gaussian in normalized x
    goal_bias_floor: float = 0.2     # off-goal corridors keep this fraction of their score


@dataclass(frozen=True)
class CommandDecision:
    linear: float
    angular: float
    stop: bool
    reason: str                       # "forward" | "turning_to_corridor" | "blocked" | "no_data"
    corridor_scores: tuple = field(default=())
    best_corridor: int = -1


def _goal_offset_to_image_x(goal_offset_deg: float, hfov_deg: float) -> float:
    """Map a bearing offset (deg, positive = goal to the right) to normalized
    image x in [-1, 1] via the pinhole model."""
    half = math.radians(hfov_deg / 2.0)
    x = math.tan(math.radians(goal_offset_deg)) / math.tan(half)
    return float(np.clip(x, -1.0, 1.0))


def suggest_command(
    mask: np.ndarray,
    cfg: PolicyConfig = PolicyConfig(),
    goal_offset_deg: float | None = None,
) -> CommandDecision:
    """Turn a traversability mask into a (linear, angular) command.

    Args:
        mask: HxW float array, values in [0, 1], 1 = drivable.
        cfg: tuning knobs.
        goal_offset_deg: bearing to the goal relative to the rover's heading
            (positive = goal is to the RIGHT). None = pure obstacle avoidance.
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim != 2 or m.size == 0:
        return CommandDecision(0.0, 0.0, True, "no_data")

    h, w = m.shape
    roi = m[int(cfg.roi_top * h):, :]
    if roi.shape[0] < 2:
        roi = m

    finite = np.isfinite(roi)
    if int(finite.sum()) < cfg.min_valid_pixels:
        return CommandDecision(0.0, 0.0, True, "no_data")
    roi = np.nan_to_num(roi, nan=0.0, posinf=1.0, neginf=0.0)

    # Column scores, near rows weighted up.
    row_w = np.linspace(1.0, cfg.bottom_weight, roi.shape[0], dtype=np.float32)[:, None]
    col_score = (roi * row_w).sum(axis=0) / (row_w.sum() + 1e-9)  # (W,)

    # Corridor scores.
    n = max(3, int(cfg.num_corridors) | 1)  # odd, >= 3
    bounds = np.linspace(0, w, n + 1, dtype=int)
    corridor_scores = np.array(
        [float(col_score[bounds[i]:bounds[i + 1]].mean()) for i in range(n)],
        dtype=np.float32,
    )
    center_idx = n // 2

    # Stop test on the raw center corridor (fraction-based, see module docstring).
    center_band = roi[:, bounds[center_idx]:bounds[center_idx + 1]]
    center_blocked_frac = float((center_band < cfg.drivable_thresh).mean())
    center_open = center_blocked_frac < cfg.stop_center_fraction

    if not center_open and float(corridor_scores.max()) < cfg.min_corridor_score:
        return CommandDecision(
            0.0, 0.0, True, "blocked",
            tuple(round(float(s), 3) for s in corridor_scores), -1,
        )

    # Goal bias: multiply corridor scores by a gaussian centered on the goal's
    # image position. Traversability dominates; the goal breaks ties.
    biased = corridor_scores.copy()
    if goal_offset_deg is not None:
        gx = _goal_offset_to_image_x(goal_offset_deg, cfg.hfov_deg)
        centers = (bounds[:-1] + bounds[1:]) / 2.0
        centers_norm = (centers - w / 2.0) / (w / 2.0)
        gauss = np.exp(-0.5 * ((centers_norm - gx) / cfg.goal_sigma) ** 2)
        biased = corridor_scores * (cfg.goal_bias_floor + (1.0 - cfg.goal_bias_floor) * gauss)

    # Best corridor; ties resolved toward the center.
    order = np.argsort(-biased, kind="stable")
    best = int(min((i for i in order if biased[i] == biased[order[0]]),
                   key=lambda i: abs(i - center_idx)))

    if float(corridor_scores[best]) < cfg.min_corridor_score:
        return CommandDecision(
            0.0, 0.0, True, "blocked",
            tuple(round(float(s), 3) for s in corridor_scores), best,
        )

    # Target x = score-weighted centroid inside the chosen corridor.
    band_cols = col_score[bounds[best]:bounds[best + 1]]
    xs = np.arange(bounds[best], bounds[best + 1], dtype=np.float32)
    weight_sum = float(band_cols.sum())
    target_x = float((xs * band_cols).sum() / weight_sum) if weight_sum > 1e-6 else float(xs.mean())

    offset_norm = (target_x - (w - 1) / 2.0) / ((w - 1) / 2.0)
    # Image-right target => turn right => negative angular (positive = LEFT).
    angular = float(np.clip(-cfg.k_angular * offset_norm, -cfg.max_angular, cfg.max_angular))

    scores_out = tuple(round(float(s), 3) for s in corridor_scores)

    if not center_open:
        # Can't drive forward, but a side corridor is viable: rotate toward it.
        turn = cfg.max_angular if best < center_idx else -cfg.max_angular
        return CommandDecision(0.0, float(turn), False, "turning_to_corridor", scores_out, best)

    center_score = float(np.clip(corridor_scores[center_idx], 0.0, 1.0))
    linear = cfg.min_linear + (cfg.max_linear - cfg.min_linear) * center_score
    linear *= 1.0 - 0.5 * abs(angular) / max(cfg.max_angular, 1e-6)
    return CommandDecision(round(linear, 3), round(angular, 3), False, "forward", scores_out, best)
