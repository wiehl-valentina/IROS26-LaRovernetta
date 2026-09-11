"""Minimal HTTP client for the Earth Rovers SDK running on localhost.

Talks only to the SDK's public HTTP endpoints — it never imports SDK code, so
it keeps working across SDK reorganizations. Command sending NEVER raises:
newer SDK versions (v6.1+) have a control watchdog that rejects motion with
HTTP 500 while a safety stop is pending, and a control loop must ride through
that rather than crash.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import requests

from .images import to_rgb

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"


@dataclass
class CommandResult:
    accepted: bool
    status: int | None = None
    detail: str = ""
    body: dict | None = None


class RoverClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 5.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("ROVER_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = float(timeout)
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ frames

    def get_front_frame_b64(self) -> str | None:
        """Raw base64 of the front camera frame (no data: prefix), or None."""
        for endpoint in ("/v2/screenshot", "/v2/front"):
            try:
                resp = self._session.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("frame fetch failed (%s): %s", endpoint, exc)
                return None
            if resp.status_code == 404:
                continue  # older/newer SDK without this endpoint
            if resp.status_code != 200:
                log.warning("frame fetch %s -> HTTP %s: %s",
                            endpoint, resp.status_code, resp.text[:200])
                return None
            try:
                return resp.json().get("front_frame")
            except ValueError:
                log.warning("frame fetch %s returned non-JSON body", endpoint)
                return None
        return None

    def get_front_frame(self) -> np.ndarray | None:
        """Decoded HxWx3 uint8 RGB front frame, or None."""
        b64 = self.get_front_frame_b64()
        if not b64:
            return None
        try:
            return to_rgb(b64)
        except Exception as exc:
            log.warning("frame decode failed: %s", exc)
            return None

    # --------------------------------------------------------------- telemetry

    def get_data(self) -> dict | None:
        """Telemetry blob from GET /data (lat/lon/orientation/speed/battery/...)."""
        try:
            resp = self._session.get(f"{self.base_url}/data", timeout=self.timeout)
            if resp.status_code != 200:
                log.warning("/data -> HTTP %s", resp.status_code)
                return None
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("/data failed: %s", exc)
            return None

    # ----------------------------------------------------------------- control

    def send_command(self, linear: float, angular: float, lamp: int = 0) -> CommandResult:
        """POST /control. Values clamped to [-1, 1]. Never raises."""
        payload = {
            "command": {
                "linear": float(np.clip(linear, -1.0, 1.0)),
                "angular": float(np.clip(angular, -1.0, 1.0)),
                "lamp": int(lamp),
            }
        }
        return self._post("/control", payload, timeout=2.0)

    def stop(self, retries: int = 3) -> CommandResult:
        """Send a zero command until accepted (the rover latches its last command)."""
        result = CommandResult(accepted=False, detail="not attempted")
        for _ in range(max(1, retries)):
            result = self.send_command(0.0, 0.0)
            if result.accepted:
                return result
        return result

    # ----------------------------------------------------------------- mission

    def start_mission(self) -> CommandResult:
        return self._post("/start-mission", {})

    def get_checkpoints_list(self) -> dict | None:
        """GET /checkpoints-list. NOTE: lat/lon come back as strings here."""
        try:
            resp = self._session.get(f"{self.base_url}/checkpoints-list", timeout=self.timeout)
            if resp.status_code != 200:
                log.warning("/checkpoints-list -> HTTP %s: %s",
                            resp.status_code, resp.text[:200])
                return None
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("/checkpoints-list failed: %s", exc)
            return None

    def checkpoint_reached(self) -> CommandResult:
        """POST /checkpoint-reached. The SDK reads current GPS itself; the
        backend rejects with 400 if the rover isn't close enough — treat a
        rejected result as "keep driving", not an error."""
        return self._post("/checkpoint-reached", {})

    # ----------------------------------------------------------------- helpers

    def _post(self, endpoint: str, payload: dict, timeout: float | None = None) -> CommandResult:
        try:
            resp = self._session.post(
                f"{self.base_url}{endpoint}", json=payload, timeout=timeout or self.timeout
            )
        except requests.RequestException as exc:
            return CommandResult(accepted=False, status=None, detail=str(exc))
        body: dict | None
        try:
            body = resp.json()
        except ValueError:
            body = None
        detail = "" if resp.ok else (resp.text[:300] or f"HTTP {resp.status_code}")
        return CommandResult(accepted=resp.ok, status=resp.status_code, detail=detail, body=body)
