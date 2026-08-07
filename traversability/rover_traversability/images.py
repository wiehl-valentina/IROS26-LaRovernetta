"""Decode anything the SDK or the team's control loop hands us into an RGB array.

The team's ``RoverLoop`` passes an ``Optional[str]`` payload that is either a
raw base64 string (from ``GET /v2/screenshot`` — no ``data:`` prefix) or a
filesystem path. The SDK's encoding depends on the ``IMAGE_FORMAT`` env var
(defaults to png server-side, while client code often assumes jpeg), so we
sniff magic bytes instead of trusting any declared mime type.
"""

from __future__ import annotations

import base64
import binascii
import os
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class PayloadError(ValueError):
    """The payload could not be interpreted as an image in any supported form."""


class ImageDecodeError(PayloadError):
    """The payload yielded bytes, but the bytes are not a decodable image."""


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

_MAGIC_SIGNATURES = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


def sniff_format(data: bytes) -> str | None:
    """Best-effort image format from magic bytes; None if unrecognized."""
    for magic, name in _MAGIC_SIGNATURES:
        if data[: len(magic)] == magic:
            return name
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _from_bytes(data: bytes) -> np.ndarray:
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        magic_hex = data[:12].hex(" ") if data else "<empty>"
        raise ImageDecodeError(
            f"Got {len(data)} bytes but they are not a decodable image "
            f"(sniffed format: {sniff_format(data)}, first bytes: {magic_hex}). "
            "If this came from the SDK, check that the endpoint returned a "
            "frame and not an error body."
        ) from exc
    return np.asarray(img.convert("RGB"))


def _from_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise PayloadError(f"Unsupported array shape {arr.shape}; expected HxW, HxWx3 or HxWx4.")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        scale = 255.0 if (finite.size == 0 or finite.max() <= 1.0) else 1.0
        arr = np.clip(np.nan_to_num(arr) * scale, 0.0, 255.0)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _from_str(payload: str) -> np.ndarray:
    if payload.startswith("data:"):
        try:
            _, b64_part = payload.split(",", 1)
        except ValueError as exc:
            raise PayloadError("Malformed data: URI (no comma separator).") from exc
        return _from_bytes(base64.b64decode(b64_part))

    if os.path.exists(payload):
        with open(payload, "rb") as fh:
            return _from_bytes(fh.read())

    # Looks like a path the caller typo'd? Don't mis-diagnose it as bad base64
    # (this is the JpgImageStrategy("screenshots/imagen.jpg") failure mode).
    if len(payload) < 512 and payload.lower().endswith(_IMAGE_EXTENSIONS):
        raise FileNotFoundError(f"Image file not found: {payload}")

    compact = "".join(payload.split())
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PayloadError(
            "String payload is neither an existing file path, a data: URI, "
            f"nor valid base64 (length {len(payload)}, starts with {payload[:24]!r})."
        ) from exc
    return _from_bytes(data)


def to_rgb(payload) -> np.ndarray:
    """Decode a payload into an HxWx3 uint8 RGB array.

    Accepts: np.ndarray (HxW / HxWx3 / HxWx4, uint8 or float), PIL.Image,
    bytes/bytearray of an encoded image, a filesystem path (str or PathLike),
    a raw base64 string, or a ``data:`` URI.
    """
    if payload is None:
        raise PayloadError("Payload is None.")
    if isinstance(payload, np.ndarray):
        return _from_array(payload)
    if isinstance(payload, Image.Image):
        return np.asarray(payload.convert("RGB"))
    if isinstance(payload, (bytes, bytearray)):
        return _from_bytes(bytes(payload))
    if isinstance(payload, os.PathLike):
        with open(payload, "rb") as fh:
            return _from_bytes(fh.read())
    if isinstance(payload, str):
        return _from_str(payload)
    raise PayloadError(f"Unsupported payload type: {type(payload).__name__}")
