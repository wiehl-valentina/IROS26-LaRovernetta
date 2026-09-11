# Roadmap: from "it drives" to your own navigation stack

This package is deliberately the *simple* solution — good enough to complete a
basic checkpoint mission, small enough to read in an afternoon. Everything
below is ordered so each step teaches the thing the next step needs. The
expensive part you can't easily reproduce (a model trained on tens of
thousands of labeled rover frames — that needs data + GPU-weeks you don't
have) is already done and handed to you. The rest is yours.

## Step 0 — Run it and read it

1. `demo predict` on a few saved frames → look at the overlays.
2. `demo live` with the rover on and parked → watch what the model thinks in
   your actual test environment.
3. Read `policy.py` top to bottom (~150 lines). It is the entire
   decision-making of this package. Understand `roi_top`, corridors, the
   fraction-based stop rule, and the goal bias.

## Step 1 — Tune the policy (your first real contribution)

The default thresholds were NOT tuned in your environment. Field-tune them:

- Log decisions: pass `on_decision=` to `TraversabilityStrategy` (or `on_step=`
  to `MissionRunner`) and record `corridor_scores` per frame alongside the
  saved overlays.
- Watch for: stopping too eagerly (`stop_center_fraction` too low), grazing
  obstacles (`drivable_thresh` too low, or `roi_top` looking too far ahead),
  oscillating left/right (`k_angular` too high), crawling (`max_linear`).
- Every knob is in `PolicyConfig` (frozen dataclass) — change values in one
  place, nothing else moves.

Deliverable: your own `PolicyConfig(...)` for your test field, with notes on
why each change was made.

## Step 2 — Fine-tune SAM-TP on YOUR footage

The model was trained on Mini+ footage from other places. Your competition
environment (surfaces, curbs, vegetation) will have failure cases. Fixing them
by adding your own training data is the single highest-leverage ML work you
can do — full recipe in **[FINETUNING.md](FINETUNING.md)**.

Deliverable: `checkpoint_yourteam_v1.pt` on your own HF repo, and
`SAMTP_HF_REPO` pointed at it. No code changes needed.

## Step 3 — Replace the corridor policy with a real planner

You already have the planner: `genie_path_planner` (installed with
`pip install -e ./genie`) plans paths on a bird's-eye-view (BEV) costmap —
read `genie/README.md`, it documents the whole pipeline. What it needs and
where to get it:

| Planner input | Where it comes from |
| --- | --- |
| BEV traversability grid | project `result.mask` to the ground plane using the camera calibration **shipped in this package** (`load_camera_K()`, `load_T_base_camera()` — real Mini+ values; the `genie/configs/stretch_path_planner.yaml` intrinsics are for a different robot, don't use them) |
| goal (x right, y forward, meters) | you already compute bearing+distance in `mission.py` — convert with basic trig |
| `PlannerConfig` | `genie_path_planner.planner` defaults are a fine start |

Then: `plan_on_bev(...)` → path as (x, y) waypoints in meters → a pure-pursuit
controller (pick a lookahead point on the path, steer proportionally to its
bearing) → the same `client.send_command()`.

Suggested order: first produce and *visualize* BEV masks offline from saved
frames (get the homography right before anything moves), then plan on them
offline, then close the loop.

Deliverable: a `bev_mission.py` that replaces `suggest_command` with
project → plan → pursue.

## Step 4 — Robustness (competition-day items)

In rough priority order:

- **Recovery behaviors**: what happens when the rover is stuck (commands sent,
  GPS not moving)? Detect it (no GPS progress over N seconds while
  commanding forward) and script an escape (back up, turn, retry).
- **Temporal smoothing**: single-frame masks flicker. Average the last 2–3
  masks (or BEV grids, ego-motion-warped) before deciding.
- **Command-age safety**: if inference latency spikes, the rover keeps
  executing the stale command. Derate `max_linear` when the last mask is old.
- **Battery/telemetry gates**: read `/data` battery and stop the mission
  cleanly below a threshold.

## Step 5 — Ideas with headroom

- A VLM supervisor (you already have Gemini wired) that runs at 0.2 Hz and
  nudges the goal ("path on the left looks better") while SAM-TP handles the
  20x-faster reflexes. Slow brain / fast brain.
- Fuse the wheel-RPM odometry from `/data` into heading when GPS drops.
- Log every mission run (frames + telemetry + decisions) from day one — that
  archive becomes your fine-tuning dataset (Step 2) for free.
