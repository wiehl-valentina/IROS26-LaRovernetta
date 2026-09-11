# rover-traversability

> 🇦🇷 **[Guía rápida en español → README.es.md](README.es.md)**

**Learned traversability perception + mission navigation for the Earth Rover Mini+.**

One camera frame in → per-pixel drivability out → steering command out. The
model is **SAM-TP** (SAM 2.1 with a learned traversability prompt, the same
architecture already vendored in this repo under `genie/`), **fine-tuned on
~50k Earth Rover Mini+ frames** — so it knows what *this* rover's ground looks
like.

Green = drivable, red = blocked (real output on a frame from this repo's
`screenshots/`):

- `predict` → mask + overlay (perception only)
- `suggest_command` → `{linear, angular}` steering from the mask (no calibration needed)
- `MissionRunner` → drives GPS checkpoints using mask + goal bearing
- `TraversabilityStrategy` → drop-in replacement for the strategies in
  `programs/genai_rover_api.py` (one-line swap)

Everything is additive: this package imports **nothing** from the rest of the
repo and talks to the rover only through the SDK's HTTP API on
`http://localhost:8000` (override with `ROVER_BASE_URL`).

---

## Install

From the repository root:

```bash
# 1. torch (platform-specific — CPU/MPS wheels are fine, CUDA if you have it)
pip install torch torchvision

# 2. the vendored GeNIE/SAM-TP code (provides the `sam2` package)
pip install --no-build-isolation -e ./genie

# 3. this package
pip install -e './traversability[hf]'
```

`[hf]` adds `huggingface_hub` for automatic weight download; skip it if you
place the checkpoint manually.

Requires Python ≥ 3.10. Without torch installed, the package still imports and
the policy/client/mission utilities and tests all work — only
`TraversabilityPredictor` needs torch+sam2.

## Weights

The fine-tuned checkpoint (`checkpoint_finetuned_v2.pt`, ~130 MB) is not in
git. The predictor finds it in this order:

1. `TraversabilityPredictor(checkpoint="/path/to/file.pt")`
2. `SAMTP_CHECKPOINT=/path/to/file.pt` (environment variable)
3. `~/.cache/rover_traversability/checkpoint_finetuned_v2.pt`
4. Auto-download from Hugging Face:
   [`sanatem/samtp-mini-traversability`](https://huggingface.co/sanatem/samtp-mini-traversability)
   (public — no token needed). Override the repo with `SAMTP_HF_REPO`.

sha256 of the known-good file:
`44e508da3d36a63431f8197f16784c980abf43ea94fc4e524bcd19d0646692bd`

> ⚠️ The public GeNIE checkpoint (`checkpoint_2.pt` from the paper's Google
> Drive) is a **base+** sized model and will NOT load — this package uses the
> tiny config that matches the fine-tuned checkpoint.

## Quickstart

### Level 0 — one image, no rover

```bash
python -m rover_traversability.demo predict screenshots/imagen.jpg --out overlay.png
```

Prints latency, drivable %, and the steering command the policy would send;
writes the green/red overlay.

### Level 1 — live against the SDK, sends nothing

```bash
hypercorn main:app          # the SDK, as usual
python -m rover_traversability.demo live --save-dir trav_out
```

Watch overlays accumulate in `trav_out/` while the rover sits still. This is
the risk-free way to see what the model thinks of a real scene.

### Level 2 — one-line swap in your existing loop

In `programs/genai_rover_api.py`'s `main()`:

```python
from rover_traversability import TraversabilityStrategy

strategy = TraversabilityStrategy(drive=True)   # instead of Base64ImageStrategy()
loop = RoverLoop(strategy=strategy, sleep_seconds=0.5, max_iterations=None)
```

`drive=False` (the default) is a dry-run: it predicts and prints, but never
sends a command. Your `RoverLoop` is untouched.

### Level 3 — GPS checkpoint mission

```bash
python -m rover_traversability.demo mission --start-mission \
    --yes-i-want-the-rover-to-move
```

Reads `/checkpoints-list`, estimates heading from GPS course-over-ground,
biases the drivable-corridor steering toward the next checkpoint, calls
`/checkpoint-reached` when close (the backend enforces the true radius —
rejections just mean "keep driving"), repeats until the mission completes.

## Performance (measured, 1024×576 frames)

| Device | Latency/frame | Effective rate |
| --- | --- | --- |
| Apple M-series GPU (MPS) | 0.16–0.23 s | 4–6 Hz |
| Apple M-series CPU | ~0.44 s | ~2.3 Hz |
| CUDA GPU | faster than both | — |

First inference after load is slower (~1–5 s kernel warmup) — call
`predictor.warmup()` before the loop.

## Safety notes

- **The rover latches its last command.** Silence is not a stop. Everything in
  this package that decides "stop" actively sends `{linear: 0, angular: 0}`
  (with retries), and the demo loops always send a stop on exit/Ctrl-C.
- `drive`/`mission` demos refuse to run without
  `--yes-i-want-the-rover-to-move`. First runs: open area, finger on Ctrl-C.
- Newer SDK versions (v6.1+) reject motion commands with HTTP 500 while a
  safety stop is pending; the client treats that as a refused command, never a
  crash.
- The model can mislabel **dark objects on light ground** as drivable
  (it was trained "ground vs above-ground"). A per-frame luminance-contrast
  refinement mitigates this and is ON by default — keep it on.

## What's in the box

| Module | What it does |
| --- | --- |
| `predictor.py` | `TraversabilityPredictor` — loads SAM-TP once, `predict()` → mask/logits/overlay |
| `policy.py` | `suggest_command(mask, cfg, goal_offset_deg)` — corridor steering, pure numpy, every threshold in `PolicyConfig` |
| `mission.py` | `MissionRunner` — checkpoints + GPS-COG heading + goal-biased steering |
| `geo.py` | bearing/distance + `HeadingEstimator` (GPS course-over-ground; the Mini+ magnetometer is not trustworthy — see docstring) |
| `client.py` | `RoverClient` — thin HTTP client for the SDK, never raises on command failures |
| `strategy.py` | `TraversabilityStrategy` — duck-types your `RoverStrategy` |
| `images.py` | decodes base64 / data-URI / path / bytes / arrays; sniffs magic bytes (the SDK may send PNG when you expect JPEG — including `screenshots/imagen.jpg` in this repo, which is a PNG in disguise) |
| `calibration.py` + `data/` | Mini+ camera intrinsics/extrinsics for your future BEV work |
| `weights.py` | checkpoint resolution chain described above |

## Where to take it next

See **[docs/ROADMAP.md](docs/ROADMAP.md)** — tune the policy, fine-tune the
model on your own footage ([docs/FINETUNING.md](docs/FINETUNING.md)), and
replace the simple corridor policy with the BEV planner you already have
vendored in `genie/`.

## Tests

```bash
pip install -e './traversability[dev]'
pytest traversability/tests    # 67 tests, no torch/checkpoint/network needed
```

## Attribution

- SAM 2 / SAM 2.1: Meta Platforms, Apache-2.0.
- SAM-TP architecture + GeNIE planner: Wang, Liu, Chen, Da, Qian, Man, Soh
  (as vendored in `./genie`).
- Mini+ fine-tuned weights: FrodoBots research (see the HF model card).
  Training data derives from FrodoBots Mini footage (Mini-4K is CC-BY-SA) —
  keep weight redistribution private/team-internal.
