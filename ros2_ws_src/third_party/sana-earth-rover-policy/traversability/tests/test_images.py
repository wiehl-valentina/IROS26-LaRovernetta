import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from rover_traversability.images import ImageDecodeError, PayloadError, sniff_format, to_rgb


def _encoded(fmt="PNG", size=(20, 10)) -> bytes:
    img = Image.new("RGB", size, (10, 200, 30))
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_raw_base64_roundtrip():
    b64 = base64.b64encode(_encoded("PNG")).decode()
    arr = to_rgb(b64)
    assert arr.shape == (10, 20, 3)
    assert arr.dtype == np.uint8


def test_base64_with_whitespace():
    b64 = base64.b64encode(_encoded("JPEG")).decode()
    noisy = "\n".join(b64[i : i + 40] for i in range(0, len(b64), 40))
    assert to_rgb(noisy).shape == (10, 20, 3)


def test_data_uri():
    b64 = base64.b64encode(_encoded("PNG")).decode()
    arr = to_rgb(f"data:image/png;base64,{b64}")
    assert arr.shape == (10, 20, 3)


def test_file_path(tmp_path):
    p = tmp_path / "frame.jpg"
    p.write_bytes(_encoded("JPEG"))
    assert to_rgb(str(p)).shape == (10, 20, 3)
    assert to_rgb(p).shape == (10, 20, 3)  # PathLike too


def test_png_bytes_when_jpeg_assumed():
    """The SDK defaults to IMAGE_FORMAT=png while client code assumes jpeg —
    decoding must not care."""
    data = _encoded("PNG")
    assert sniff_format(data) == "png"
    assert to_rgb(data).shape == (10, 20, 3)


def test_sniff_magic_bytes():
    assert sniff_format(_encoded("JPEG")) == "jpeg"
    assert sniff_format(_encoded("PNG")) == "png"
    assert sniff_format(_encoded("WEBP")) == "webp"
    assert sniff_format(b"garbage") is None


def test_ndarray_passthrough_and_promotions():
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    assert to_rgb(rgb).shape == (5, 6, 3)
    gray = np.zeros((5, 6), dtype=np.uint8)
    assert to_rgb(gray).shape == (5, 6, 3)
    rgba = np.zeros((5, 6, 4), dtype=np.uint8)
    assert to_rgb(rgba).shape == (5, 6, 3)
    floats = np.ones((5, 6, 3), dtype=np.float32)  # [0,1] -> scaled
    assert to_rgb(floats).max() == 255


def test_pil_image():
    img = Image.new("RGB", (6, 5), (1, 2, 3))
    assert to_rgb(img).shape == (5, 6, 3)


def test_missing_image_path_is_file_not_found():
    with pytest.raises(FileNotFoundError):
        to_rgb("screenshots/imagen_typo.jpg")


def test_garbage_string_is_payload_error():
    with pytest.raises(PayloadError):
        to_rgb("!!! not base64 and not a path !!!")


def test_valid_b64_of_non_image_is_decode_error():
    b64 = base64.b64encode(b"clearly not an image at all........").decode()
    with pytest.raises(ImageDecodeError):
        to_rgb(b64)


def test_none_rejected():
    with pytest.raises(PayloadError):
        to_rgb(None)
