"""Drop-in strategy for the team's existing RoverLoop.

``TraversabilityStrategy`` duck-types ``programs.genai_rover_api.RoverStrategy``
(same two methods, same signatures) WITHOUT importing any team code, so this
package never creates an import-time dependency on the rest of the repo.

Swap one line in programs/genai_rover_api.py's main():

    from rover_traversability import TraversabilityStrategy
    strategy = TraversabilityStrategy(drive=True)   # instead of Base64ImageStrategy()

By default ``drive=False``: the strategy predicts and PRINTS what it would do,
but sends nothing — run it against the live SDK risk-free first.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

from .client import RoverClient
from .policy import CommandDecision, PolicyConfig, suggest_command

log = logging.getLogger(__name__)


class TraversabilityStrategy:
    def __init__(
        self,
        client: RoverClient | None = None,
        predictor=None,
        policy: PolicyConfig | None = None,
        drive: bool = False,
        save_overlays_dir: str | None = None,
        on_decision: Callable[[object, CommandDecision], None] | None = None,
    ) -> None:
        self.client = client or RoverClient()
        if predictor is None:
            # Lazy import keeps this module importable without torch; failing
            # here (construction) beats failing mid-mission.
            from .predictor import TraversabilityPredictor

            predictor = TraversabilityPredictor()
        self.predictor = predictor
        self.policy = policy or PolicyConfig()
        self.drive = bool(drive)
        self.save_overlays_dir = save_overlays_dir
        self.on_decision = on_decision
        self._frame_count = 0

    # -- RoverStrategy interface (duck-typed) --------------------------------

    def get_image_payload(self) -> Optional[str]:
        return self.client.get_front_frame_b64()

    def analyze(self, payload: Optional[str]) -> None:
        if not payload:
            return
        try:
            result = self.predictor.predict(payload)
        except Exception as exc:
            log.error("prediction failed: %s", exc)
            if self.drive:
                self.client.stop()
            return

        decision = suggest_command(result.mask, self.policy)
        self._frame_count += 1
        self._maybe_save_overlay(result)
        if self.on_decision is not None:
            self.on_decision(result, decision)

        if not self.drive:
            print(
                f"[dry-run] {decision.reason}: linear={decision.linear:+.2f} "
                f"angular={decision.angular:+.2f} (inference {result.inference_s:.2f}s "
                f"on {result.device})"
            )
            return

        if decision.stop:
            # Actively stop: the rover repeats its last command forever, so
            # "send nothing" is NOT a stop.
            self.client.stop()
        else:
            self.client.send_command(decision.linear, decision.angular)

    # -- helpers ---------------------------------------------------------------

    def _maybe_save_overlay(self, result) -> None:
        if not self.save_overlays_dir:
            return
        try:
            from PIL import Image

            os.makedirs(self.save_overlays_dir, exist_ok=True)
            path = os.path.join(
                self.save_overlays_dir, f"overlay_{int(time.time())}_{self._frame_count:05d}.png"
            )
            Image.fromarray(result.overlay).save(path)
        except Exception as exc:
            log.warning("overlay save failed: %s", exc)
