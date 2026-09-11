"""Perception loop: camera frame -> SAM-TP -> published mask snapshot.

Runs in its own daemon thread, naturally paced by inference latency (~3-6 Hz
on Apple MPS, slower on CPU). It NEVER sends rover commands — the control
loop is the single /control writer by construction. It owns its own
RoverClient (requests.Session is not documented thread-safe) and is the only
thread that touches the predictor (hydra global state is not re-entrant).
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .shared import MaskSnapshot, SharedState

log = logging.getLogger(__name__)


def renormalize_percentile(
    mask: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0, floor: float = 0.45
) -> np.ndarray:
    """Per-frame percentile stretch for soft-confidence scenes.

    The fine-tuned checkpoint outputs mid-range probabilities (~0.4-0.6) on
    out-of-distribution scenes instead of saturating; stretching p5-p95 back
    to [0, 1] restores the contrast the downstream thresholds expect. The
    ``floor`` guard skips the stretch when even p95 is low — a genuinely
    blocked scene must not be inflated into a drivable-looking one.
    """
    lo, hi = np.percentile(mask, [p_lo, p_hi])
    if hi < floor or hi - lo < 1e-6:
        return mask
    return np.clip((mask - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _encode_jpeg(overlay: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


class PerceptionLoop:
    def __init__(
        self,
        client,                      # RoverClient, exclusive to this thread
        predictor,                   # exclusive to this thread
        shared: SharedState,
        cfg,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.client = client
        self.predictor = predictor
        self.shared = shared
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.frames = 0
        self.errors = 0

    def run(self) -> None:
        while not self.shared.stop.is_set():
            try:
                self._one_frame()
            except Exception:
                # Never let a transient decode/HTTP error kill perception —
                # the control loop's stale-mask guard handles the gap.
                self.errors += 1
                log.exception("perception frame failed")
                time.sleep(0.3)
        log.info("perception loop exiting after %d frames (%d errors)",
                 self.frames, self.errors)

    def _one_frame(self) -> None:
        payload = self.client.get_front_frame_b64()
        if not payload:
            time.sleep(0.3)
            return

        result = self.predictor.predict(payload)
        mask = result.mask
        if self.cfg.renormalize_percentile:
            mask = renormalize_percentile(mask)

        self.frames += 1
        self.shared.publish_mask(
            MaskSnapshot(
                mask=mask,
                overlay_jpeg=_encode_jpeg(result.overlay),
                mono_ts=time.monotonic(),
                inference_s=float(result.inference_s),
                seq=self.frames,
            )
        )

        if (
            self.run_dir is not None
            and self.cfg.overlay_every_n
            and self.frames % self.cfg.overlay_every_n == 0
        ):
            from PIL import Image

            Image.fromarray(result.overlay).save(
                self.run_dir / f"overlay_{self.frames:05d}.png"
            )
