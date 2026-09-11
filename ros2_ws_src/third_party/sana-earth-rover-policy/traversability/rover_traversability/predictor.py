"""SAM-TP inference wrapper: image in, traversability mask out.

Model: SAM 2.1 (Hiera-tiny image branch) with the prompt encoder replaced by a
learned "traversability prompt" (GeNIE SAM-TP), fine-tuned on Earth Rover Mini+
footage. The custom prompt encoder ignores all point prompts — the model is
effectively a single-purpose semantic segmenter and its output is deterministic
per image.

The sam2 package itself is vendored in this repo under ./genie and must be
installed first (see README). This module is imported lazily by the package so
that machines without torch can still use the policy/client/mission utilities.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .images import to_rgb
from .weights import SamNotInstalledError, resolve_checkpoint, resolve_config

# Must be set before torch import: some SAM2 ops have no MPS kernel and need
# the CPU-fallback escape hatch on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# hydra (used inside sam2's build_sam2) mutates global state during model
# construction and is not re-entrant. Serialize construction; use one
# predictor per process.
_BUILD_LOCK = threading.Lock()


class CheckpointMismatchError(RuntimeError):
    """The checkpoint's architecture does not match the tiny inference config."""


def pick_device() -> str:
    """Best available torch device: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def refine_traversability_by_contrast(
    trav: np.ndarray,
    rgb: np.ndarray,
    drivable_thresh: float = 0.5,
    darkness_ratio: float = 0.65,
    min_reference_pixels: int = 500,
) -> np.ndarray:
    """Mark pixels-much-darker-than-the-ground as obstacles, even if SAM-TP said drivable.

    Why this exists: SAM-TP was trained as "ground vs. above-ground" and can
    label dark objects sitting on light ground (other rovers, chairs, low
    obstacles) as drivable. Fix: use the pixels it called drivable as a
    per-frame reference for what the ground looks like (median luminance) and
    downgrade drivable pixels dramatically darker than that reference. The
    reference is recomputed every frame, so it adapts to lighting.

    Args:
        trav: HxW float32 traversability in [0, 1] (post-sigmoid).
        rgb:  HxWx3 uint8 frame, same spatial size.
        drivable_thresh: pixels above this count as "reference ground".
        darkness_ratio: pixels darker than median_reference * this are downgraded.
        min_reference_pixels: skip refinement below this much reference signal.
    """
    t = np.asarray(trav, dtype=np.float32)
    if rgb is None or rgb.shape[:2] != t.shape:
        return t

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b  # ITU-R BT.601, [0, 255]

    ground_mask = t > float(drivable_thresh)
    if int(ground_mask.sum()) < int(min_reference_pixels):
        return t

    median_ground_lum = float(np.median(lum[ground_mask]))
    if median_ground_lum < 30.0:
        # Everything is dark (dusk, indoors) — "darker than ground" means nothing.
        return t

    obstacle_mask = ground_mask & (lum < median_ground_lum * float(darkness_ratio))
    if not obstacle_mask.any():
        return t

    out = t.copy()
    out[obstacle_mask] = 0.0
    return out


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(logits, dtype=np.float32), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the frame with green (drivable) / red (blocked). Pure numpy."""
    m = np.clip(mask, 0.0, 1.0)[..., None]
    color = np.concatenate([255.0 * (1.0 - m), 255.0 * m, np.zeros_like(m)], axis=-1)
    out = (1.0 - alpha) * image.astype(np.float32) + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class TraversabilityResult:
    mask: np.ndarray        # HxW float32 in [0, 1], 1 = drivable
    logits: np.ndarray      # HxW float32 raw model output
    overlay: np.ndarray     # HxWx3 uint8 RGB debug view (green = drivable)
    image: np.ndarray       # decoded input frame, HxWx3 uint8 RGB
    device: str
    inference_s: float


class TraversabilityPredictor:
    """Loads SAM-TP once and predicts per-frame traversability.

    Usage:
        predictor = TraversabilityPredictor()          # resolves weights (see weights.py)
        result = predictor.predict(base64_or_path_or_array)
        result.mask     # HxW float32, 1 = drivable
        result.overlay  # HxWx3 uint8 for eyeballing
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: str | Path | None = None,
        device: str | None = None,
        hf_repo: str | None = None,
        contrast_refine: bool = True,
        overlay_alpha: float = 0.45,
    ) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise SamNotInstalledError(
                "torch and/or the vendored sam2 package are missing. From the "
                "repository root run:\n"
                "    pip install torch torchvision\n"
                "    pip install --no-build-isolation -e ./genie\n"
            ) from exc

        cfg_path = resolve_config(config)
        ckpt_path = resolve_checkpoint(checkpoint, hf_repo=hf_repo)
        self._device = device or pick_device()
        self._contrast_refine = bool(contrast_refine)
        self._overlay_alpha = float(overlay_alpha)

        with _BUILD_LOCK:
            try:
                model = build_sam2(str(cfg_path), str(ckpt_path), device=self._device)
            except (RuntimeError, KeyError) as exc:
                raise CheckpointMismatchError(
                    f"Failed to load checkpoint {ckpt_path} against the tiny SAM-TP "
                    f"inference config ({cfg_path.name}, Hiera-tiny embed_dim=96). "
                    "You are probably pointing at a different-sized checkpoint — "
                    "e.g. the public GeNIE checkpoint_2.pt is base+ sized and will "
                    "NOT load. The expected file is checkpoint_finetuned_v2.pt."
                ) from exc

        # Built once, reused every frame (constructing per-call costs real latency).
        self._predictor = SAM2ImagePredictor(sam_model=model, mask_threshold=0.0)

    @property
    def device(self) -> str:
        return self._device

    def predict(self, payload) -> TraversabilityResult:
        """Run SAM-TP on anything images.to_rgb accepts (base64/path/bytes/array)."""
        import torch
        from PIL import Image

        rgb = to_rgb(payload)
        pil = Image.fromarray(rgb)
        w, h = pil.size

        # Dummy bottom-edge points: the SAM2ImagePredictor API requires a
        # prompt, but SAM-TP's learned prompt encoder ignores it entirely.
        pts = np.array(
            [(0, h - 1), (w - 1, h - 1), ((w - 1) // 2, h - 1)], dtype=np.float32
        )
        labels = np.ones(len(pts), dtype=np.int32)

        t0 = time.perf_counter()
        with torch.inference_mode():
            self._predictor.reset_predictor()
            self._predictor.set_image(pil)
            masks, _iou, _low_res = self._predictor.predict(
                point_coords=pts,
                point_labels=labels,
                multimask_output=False,
                return_logits=True,
                normalize_coords=False,
            )
        inference_s = time.perf_counter() - t0

        logits = np.asarray(masks[0], dtype=np.float32)
        mask = _sigmoid(logits)
        if self._contrast_refine:
            mask = refine_traversability_by_contrast(mask, rgb)

        return TraversabilityResult(
            mask=mask,
            logits=logits,
            overlay=_make_overlay(rgb, mask, self._overlay_alpha),
            image=rgb,
            device=self._device,
            inference_s=inference_s,
        )

    def warmup(self) -> float:
        """Run one dummy inference to compile kernels; returns seconds taken."""
        dummy = np.full((576, 1024, 3), 128, dtype=np.uint8)
        return self.predict(dummy).inference_s
