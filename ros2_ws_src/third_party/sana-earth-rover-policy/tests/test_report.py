"""Tests for the post-run report generator against a synthetic run directory."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sana.report import write_report


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "20260811_120000"
    d.mkdir()

    records = []
    states = ["wiggle"] * 3 + ["pursue"] * 10 + ["recover_backup"] * 2 + \
             ["recover_turn"] * 2 + ["pursue"] * 5 + ["arrive"] * 3
    for i, state in enumerate(states):
        records.append({
            "ts": 1e9 + i * 0.2, "mono": i * 0.2, "tick": i + 1,
            "state": state, "reason": state,
            "lat": 10.0 + i * 1e-5, "lon": 20.0 + (i % 3) * 1e-6,
            "linear": 0.25, "angular": 0.0,
            "inference_s": 0.15 + (i % 5) * 0.01,
            "guards": ["speed_capped"] if i == 7 else [],
            "attempted_checkpoint": state == "arrive",
        })
    (d / "decisions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    (d / "result.json").write_text(json.dumps({
        "completed": True, "reason": "completed", "ticks": len(records),
        "checkpoints_reached": 1, "run_dir": str(d),
        "started_ts": 1e9, "ended_ts": 1e9 + len(records) * 0.2,
        "dry_run": False,
        "config": {"arrive_attempt_m": 6.0},
        "checkpoints": [
            {"id": 1, "sequence": 1, "latitude": 10.00025, "longitude": 20.0},
        ],
    }))

    from PIL import Image

    for n in (5, 10, 15):
        img = np.zeros((36, 64, 3), dtype=np.uint8)
        img[..., 1] = 128
        Image.fromarray(img).save(d / f"overlay_{n:05d}.png")
    return d


def test_report_is_selfcontained_html(run_dir):
    out = write_report(run_dir)
    assert out.name == "report.html"
    html = out.read_text()

    assert "<svg" in html                       # GPS trace rendered
    assert "CP1" in html                        # checkpoint marker
    assert "data:image/jpeg;base64," in html    # filmstrip inlined
    assert "recover_backup" in html             # legend / occupancy
    assert "Stuck recoveries</td><td>1" in html
    assert "Checkpoints reached</td><td>1" in html
    # Self-contained: no external resource loads.
    assert 'src="http' not in html and "href=" not in html


def test_report_without_gps_or_overlays(tmp_path):
    d = tmp_path / "empty_run"
    d.mkdir()
    (d / "decisions.jsonl").write_text(json.dumps({
        "ts": 1.0, "mono": 0.0, "tick": 1, "state": "wiggle",
        "reason": "wiggle", "lat": None, "lon": None,
        "linear": 0.2, "angular": 0.0,
    }) + "\n")
    (d / "result.json").write_text(json.dumps({
        "completed": False, "reason": "max_steps", "checkpoints_reached": 0,
        "started_ts": 0.0, "ended_ts": 1.0, "config": {}, "checkpoints": [],
    }))
    html = write_report(d).read_text()
    assert "No GPS trace recorded" in html
    assert "No overlays saved" in html
