"""Post-run mission report: one self-contained report.html per run directory.

Input: a runs/<timestamp>/ directory with decisions.jsonl, result.json and
optional overlay_*.png files. Output: report.html with an inline-SVG GPS
trace colored by FSM state, checkpoint markers with the attempt radius, a
filmstrip of overlays (base64 data URIs) and a stats table. Everything is
inlined — the file works from a USB stick with no network.

Usage: generated automatically at mission end, or manually:

    python -m sana.report runs/20260811_153000
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
from pathlib import Path

from rover_traversability.geo import gps_bearing_and_distance

STATE_COLORS = {
    "wiggle": "#f2c14e",
    "align": "#4ea5f2",
    "pursue": "#63c74d",
    "arrive": "#b06ef2",
    "recover_backup": "#f2734e",
    "recover_turn": "#f24e88",
    "done": "#999999",
}
_EARTH_R = 6_371_000.0


def _load_run(run_dir: Path) -> tuple[list[dict], dict]:
    records = []
    jsonl = run_dir / "decisions.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    result = {}
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    return records, result


# ------------------------------------------------------------------ SVG trace

def _project(lat, lon, lat0, lon0, mean_lat_rad):
    x = math.radians(lon - lon0) * math.cos(mean_lat_rad) * _EARTH_R
    y = -math.radians(lat - lat0) * _EARTH_R  # SVG y grows downward
    return x, y


def _trace_svg(records: list[dict], result: dict, size: int = 560) -> str:
    fixes = [(r["lat"], r["lon"], r.get("state", "pursue"))
             for r in records if r.get("lat") is not None and r.get("lon") is not None]
    checkpoints = result.get("checkpoints", [])
    if len(fixes) < 2:
        return "<p class='muted'>No GPS trace recorded.</p>"

    lat0 = sum(f[0] for f in fixes) / len(fixes)
    lon0 = sum(f[1] for f in fixes) / len(fixes)
    mean_lat = math.radians(lat0)

    pts = [(_project(la, lo, lat0, lon0, mean_lat), st) for la, lo, st in fixes]
    cp_pts = [(_project(c["latitude"], c["longitude"], lat0, lon0, mean_lat), c)
              for c in checkpoints]

    xs = [p[0][0] for p in pts] + [c[0][0] for c in cp_pts]
    ys = [p[0][1] for p in pts] + [c[0][1] for c in cp_pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 5.0)
    pad = span * 0.12
    lo_x, lo_y = min(xs) - pad, min(ys) - pad
    scale = size / (span + 2 * pad)

    def sxy(x, y):
        return (x - lo_x) * scale, (y - lo_y) * scale

    # Consecutive same-state runs become one polyline each.
    segments: list[tuple[str, list]] = []
    for (xy, state) in pts:
        if segments and segments[-1][0] == state:
            segments[-1][1].append(xy)
        else:
            prev_tail = [segments[-1][1][-1]] if segments else []
            segments.append((state, prev_tail + [xy]))

    parts = [f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" '
             f'class="trace" role="img" aria-label="GPS trace">']
    attempt_r = (result.get("config", {}).get("arrive_attempt_m", 6.0)) * scale
    for (x, y), c in cp_pts:
        px, py = sxy(x, y)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{attempt_r:.1f}" '
                     f'fill="#b06ef2" fill-opacity="0.12" stroke="#b06ef2" '
                     f'stroke-dasharray="4 3"/>')
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#b06ef2"/>')
        parts.append(f'<text x="{px + 8:.1f}" y="{py + 4:.1f}" class="cp">'
                     f'CP{c.get("sequence", "?")}</text>')
    for state, seg in segments:
        if len(seg) < 2:
            continue
        points = " ".join(f"{sxy(x, y)[0]:.1f},{sxy(x, y)[1]:.1f}" for x, y in seg)
        color = STATE_COLORS.get(state, "#ccc")
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                     f'stroke-width="2.5" stroke-linecap="round"/>')
    sx, sy = sxy(*pts[0][0])
    ex, ey = sxy(*pts[-1][0])
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="none" '
                 f'stroke="#fff" stroke-width="2"/>')
    parts.append(f'<rect x="{ex - 5:.1f}" y="{ey - 5:.1f}" width="10" height="10" '
                 f'fill="#fff"/>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ filmstrip

def _filmstrip(run_dir: Path, max_frames: int = 12, width: int = 240) -> str:
    overlays = sorted(run_dir.glob("overlay_*.png"))
    if not overlays:
        return "<p class='muted'>No overlays saved for this run.</p>"
    step = max(1, len(overlays) // max_frames)
    picked = overlays[::step][:max_frames]

    from PIL import Image

    cells = []
    for path in picked:
        img = Image.open(path).convert("RGB")
        h = int(img.height * width / img.width)
        img = img.resize((width, h))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        cells.append(f'<figure><img src="data:image/jpeg;base64,{b64}" '
                     f'alt="{path.name}"><figcaption>{path.stem}</figcaption></figure>')
    return f'<div class="strip">{"".join(cells)}</div>'


# ----------------------------------------------------------------------- stats

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round(q / 100 * (len(vals) - 1)))))
    return vals[idx]


def _stats(records: list[dict], result: dict) -> list[tuple[str, str]]:
    duration = (result.get("ended_ts", 0) or 0) - (result.get("started_ts", 0) or 0)
    monos = [r["mono"] for r in records if r.get("mono") is not None]
    span = (monos[-1] - monos[0]) if len(monos) > 1 else 0.0
    hz = (len(monos) - 1) / span if span > 0 else 0.0

    fixes = [(r["lat"], r["lon"]) for r in records
             if r.get("lat") is not None and r.get("lon") is not None]
    dist = sum(
        gps_bearing_and_distance(a[0], a[1], b[0], b[1])[1]
        for a, b in zip(fixes, fixes[1:])
    )

    state_time: dict[str, float] = {}
    for prev, cur in zip(records, records[1:]):
        if prev.get("mono") is not None and cur.get("mono") is not None:
            dt = cur["mono"] - prev["mono"]
            state_time[prev.get("state", "?")] = (
                state_time.get(prev.get("state", "?"), 0.0) + dt
            )
    occupancy = ", ".join(
        f"{s} {t:.0f}s" for s, t in sorted(state_time.items(), key=lambda kv: -kv[1])
    ) or "—"

    inference = [r["inference_s"] for r in records if r.get("inference_s")]
    stuck_events = sum(
        1 for prev, cur in zip(records, records[1:])
        if prev.get("state") != "recover_backup" and cur.get("state") == "recover_backup"
    )
    attempts = sum(1 for r in records if r.get("attempted_checkpoint"))
    capped = sum(1 for r in records if "speed_capped" in (r.get("guards") or []))

    return [
        ("Outcome", f"{result.get('reason', '?')} "
                    f"({'completed' if result.get('completed') else 'not completed'})"),
        ("Duration", f"{duration:.0f} s"),
        ("Control ticks / achieved rate", f"{len(records)} ticks / {hz:.1f} Hz"),
        ("Distance traveled (GPS)", f"{dist:.1f} m"),
        ("Checkpoints reached", str(result.get("checkpoints_reached", 0))),
        ("Arrival attempts", str(attempts)),
        ("Inference mean / p95",
         f"{(sum(inference) / len(inference)):.3f} s / {_percentile(inference, 95):.3f} s"
         if inference else "—"),
        ("Stuck recoveries", str(stuck_events)),
        ("Speed-cap ticks", str(capped)),
        ("Time per state", occupancy),
        ("Dry run", "yes" if result.get("dry_run") else "no"),
    ]


# ----------------------------------------------------------------------- html

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Mission report — {name}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #14161a;
          color: #e8e8e8; margin: 0 auto; max-width: 1100px; padding: 1.5rem; }}
  h1 {{ font-size: 1.3rem; color: #63c74d; }}
  h2 {{ font-size: 1.05rem; margin-top: 2rem; border-bottom: 1px solid #333;
        padding-bottom: .3rem; }}
  .muted {{ color: #888; }}
  .cols {{ display: flex; gap: 2rem; flex-wrap: wrap; align-items: flex-start; }}
  svg.trace {{ background: #0c0e10; border-radius: 8px; max-width: 100%; }}
  svg.trace text.cp {{ fill: #b06ef2; font-size: 12px; }}
  table {{ border-collapse: collapse; font-size: .92rem; }}
  td {{ padding: .3rem .7rem; border-bottom: 1px solid #2a2d33; }}
  td:first-child {{ color: #999; white-space: nowrap; }}
  .legend span {{ display: inline-block; margin-right: 1rem; font-size: .85rem; }}
  .legend i {{ display: inline-block; width: 12px; height: 12px;
               border-radius: 2px; margin-right: .35rem; vertical-align: -1px; }}
  .strip {{ display: flex; gap: .5rem; overflow-x: auto; padding-bottom: .5rem; }}
  .strip figure {{ margin: 0; text-align: center; }}
  .strip img {{ border-radius: 4px; display: block; }}
  .strip figcaption {{ font-size: .7rem; color: #888; margin-top: .2rem; }}
</style>
</head>
<body>
<h1>Sana Rover Policy — mission report</h1>
<p class="muted">{name}</p>

<h2>GPS trace by state</h2>
<div class="cols">
  <div>{trace}</div>
  <div>
    <div class="legend">{legend}</div>
    <p class="muted">Circle = start, square = end. Dashed circle = checkpoint
    attempt radius.</p>
  </div>
</div>

<h2>Stats</h2>
<table>{stats_rows}</table>

<h2>What SAM-TP saw (green = drivable)</h2>
{filmstrip}
</body>
</html>
"""


def write_report(run_dir) -> Path:
    run_dir = Path(run_dir)
    records, result = _load_run(run_dir)
    legend = "".join(
        f'<span><i style="background:{color}"></i>{state}</span>'
        for state, color in STATE_COLORS.items()
    )
    stats_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in _stats(records, result)
    )
    html = _TEMPLATE.format(
        name=run_dir.name,
        trace=_trace_svg(records, result),
        legend=legend,
        stats_rows=stats_rows,
        filmstrip=_filmstrip(run_dir),
    )
    out = run_dir / "report.html"
    out.write_text(html)
    return out


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m sana.report runs/<timestamp>")
        return 2
    out = write_report(args[0])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
