"""Shared utilities for the traversability testing/optimization tools.

Nothing here duplicates policy logic. It is reused as-is from the main
package:
    rover_traversability.images.to_rgb          - decode any payload to RGB
    rover_traversability.policy.PolicyConfig     - tunable policy parameters
    rover_traversability.policy.suggest_command  - the actual policy under test
    rover_traversability.policy.CommandDecision

This module only adds: dataset I/O (image + capture_test.py sidecar JSON), a
mask cache so the expensive SAM-TP inference runs at most once per distinct
image regardless of how many PolicyConfig combinations get evaluated on top
of it, and an overlay renderer for policy_test.py.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rover_traversability.images import to_rgb
from rover_traversability.policy import CommandDecision

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SIDECAR_SUFFIX = ".json"


# --------------------------------------------------------------------- dataset

@dataclass
class FrameRecord:
    """One captured frame plus whatever metadata capture_test.py wrote for it."""

    path: Path
    name: str  # stem, used as the stable key everywhere (rows, overlays, cache)
    metadata: dict = field(default_factory=dict)

    @property
    def sidecar_path(self) -> Path:
        return self.path.with_suffix(SIDECAR_SUFFIX)


def list_dataset(images_dir: str | Path) -> list[FrameRecord]:
    """Every image in a folder, sorted chronologically (filenames sort that way
    because capture_test.py zero-pads a monotonic index).

    Picks up the capture_test.py sidecar JSON if present. Works fine on a
    folder of plain images with no sidecars too (metadata stays empty) so a
    hand-collected set of frames/masks can also be evaluated.
    """
    d = Path(images_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    records = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        meta: dict = {}
        sidecar = p.with_suffix(SIDECAR_SUFFIX)
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text())
            except (json.JSONDecodeError, OSError):
                meta = {}
        records.append(FrameRecord(path=p, name=p.stem, metadata=meta))
    if not records:
        raise FileNotFoundError(
            f"no images ({', '.join(IMAGE_EXTENSIONS)}) found in {d}"
        )
    return records


def frame_content_hash(path: Path) -> str:
    """Stable cache key for a frame's *content*, independent of filename —
    so re-running capture with different naming still hits the cache."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- mask caching

class MaskCache:
    """Runs SAM-TP once per distinct image — inference, not policy math, is
    the expensive part of this pipeline. Cached masks live on disk so repeat
    runs of policy_test.py / policy_tuner.py over the same dataset never touch
    torch again after the first pass.
    """

    def __init__(self, cache_dir: str | Path, predictor=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = predictor

    def get(self, record: FrameRecord) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rgb_image, mask). Decodes/predicts and caches on a miss."""
        rgb = to_rgb(str(record.path))
        key = frame_content_hash(record.path)
        cached = self.cache_dir / f"{key}.npy"
        if cached.is_file():
            return rgb, np.load(cached)
        if self.predictor is None:
            raise RuntimeError(
                f"no cached mask for {record.name} and no predictor was "
                "given to compute one. Install torch + the vendored sam2 "
                "package and pass --checkpoint, or pre-populate the cache "
                "with a run that has a predictor available."
            )
        result = self.predictor.predict(rgb)
        np.save(cached, result.mask)
        return rgb, result.mask


# -------------------------------------------------------------------- overlay

def draw_policy_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    decision: CommandDecision,
    cfg,
    goal_offset_deg: float | None = None,
) -> Image.Image:
    """Green/red traversability blend + ROI line + corridor grid & scores +
    the chosen corridor/direction + the config's key thresholds — everything
    needed to eyeball one (image, config) decision without re-reading code.
    """
    h, w = mask.shape
    base = Image.fromarray(rgb).convert("RGB").resize((w, h))
    m = np.clip(mask, 0.0, 1.0)[..., None]
    color = np.concatenate(
        [255.0 * (1.0 - m), 255.0 * m, np.zeros_like(m)], axis=-1
    ).astype(np.uint8)
    blended = Image.blend(base, Image.fromarray(color), alpha=0.4)
    draw = ImageDraw.Draw(blended, "RGBA")

    roi_y = int(cfg.roi_top * h)
    draw.line([(0, roi_y), (w, roi_y)], fill=(245, 166, 35, 255), width=2)

    n = max(3, int(cfg.num_corridors) | 1)
    bounds = np.linspace(0, w, n + 1, dtype=int)
    center_idx = n // 2
    scores = decision.corridor_scores or tuple(0.0 for _ in range(n))

    for i in range(n):
        x0, x1 = int(bounds[i]), int(bounds[i + 1])
        is_best = i == decision.best_corridor
        is_center = i == center_idx
        outline = (57, 217, 138, 255) if is_best else (255, 255, 255, 80)
        draw.rectangle([x0, roi_y, x1 - 1, h - 1], outline=outline,
                        width=3 if is_best else 1)
        if is_center:
            draw.line([(x0, roi_y), (x0, h)], fill=(255, 255, 255, 120))
        if i < len(scores):
            draw.text((x0 + 4, roi_y + 4), f"{scores[i]:.2f}",
                       fill=(255, 255, 255, 230))

    label = f"{decision.reason} | lin={decision.linear:+.2f} ang={decision.angular:+.2f}"
    if goal_offset_deg is not None:
        label += f" | goal={goal_offset_deg:+.0f}deg"
    cfg_label = (
        f"roi_top={cfg.roi_top} thresh={cfg.drivable_thresh} "
        f"stop_frac={cfg.stop_center_fraction} min_score={cfg.min_corridor_score} "
        f"k_ang={cfg.k_angular} max_lin={cfg.max_linear}"
    )
    draw.rectangle([0, 0, w, 18], fill=(0, 0, 0, 160))
    draw.text((4, 3), label, fill=(255, 255, 255, 255))
    draw.rectangle([0, h - 16, w, h], fill=(0, 0, 0, 160))
    draw.text((4, h - 15), cfg_label, fill=(200, 210, 225, 255))

    cx, cy = w // 2, h - 30
    if decision.stop:
        draw.line([(cx - 10, cy - 10), (cx + 10, cy + 10)], fill=(255, 92, 92, 255), width=4)
        draw.line([(cx - 10, cy + 10), (cx + 10, cy - 10)], fill=(255, 92, 92, 255), width=4)
    else:
        dx = -20 if decision.angular > 0.05 else (20 if decision.angular < -0.05 else 0)
        draw.line([(cx, cy), (cx + dx, cy - 24)], fill=(57, 217, 138, 255), width=4)

    return blended


def decision_to_row(name: str, decision: CommandDecision, extra: dict) -> dict:
    row = {
        "frame": name,
        "linear": decision.linear,
        "angular": decision.angular,
        "stop": decision.stop,
        "reason": decision.reason,
        "best_corridor": decision.best_corridor,
        "corridor_scores": json.dumps(list(decision.corridor_scores)),
    }
    row.update(extra)
    return row
