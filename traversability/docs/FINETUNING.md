# Fine-tuning SAM-TP on your own footage

The checkpoint you received is a full model state dict — it is directly usable
as the starting point for further training. This is the intended iteration
loop: collect frames where the model is wrong → label them → fine-tune → swap
one env var.

## What you need

| Thing | Where |
| --- | --- |
| Training code | Meta's SAM2 repo: `https://github.com/facebookresearch/sam2` — the `training/` package (NOT vendored in `./genie`, clone it separately) |
| Training configs | **already in this repo**: `genie/sam2/configs/sam2.1_training_tiny/` — written for exactly this architecture (tiny backbone + `want_custom_prompt_encoder: 2`) |
| Init checkpoint | `checkpoint_finetuned_v2.pt` (what this package runs) |
| Labeled data | you make this — see below |
| GPU | a real one (1024-res training; a laptop won't do). University cluster, Colab Pro, Lightning/HF cloud GPUs — a few hours on a single A100/4090-class card is enough for a small fine-tune |

## 1. Collect frames

You are already generating them: every `demo live` / mission run can save
frames. Prioritize frames where the current model is *wrong* (look at the
overlays — obstacles marked green, ground marked red). A few hundred
well-chosen frames beat thousands of redundant ones.

```bash
python -m rover_traversability.demo live --save-dir capture_session_01
```

(Also save the raw frames, not just overlays — `TraversabilityResult.image`
has the decoded frame if you script it yourself.)

## 2. Label

Binary masks: drivable ground = white (255), everything else = black (0). Same
resolution as the frame, PNG. Tools: CVAT, Label Studio, Roboflow, or
SAM-assisted annotation (segment with vanilla SAM2, then correct — much faster
than painting from scratch).

Layout (MOSE / PNG-VOS style, what the training configs expect — each "video"
can just be a folder of unrelated frames):

```
dataset/
  JPEGImages/
    session01/
      00000.jpg
      00001.jpg
  Annotations/
    session01/
      00000.png
      00001.png
```

## 3. Train

In your clone of `facebookresearch/sam2`:

1. Copy `genie/sam2/configs/sam2.1_training_tiny/sam2.1_custom2.yaml` (from
   this repo) into the training configs.
2. Point it at your data and the init checkpoint:
   - `img_folder` / `gt_folder` → your `JPEGImages` / `Annotations`
   - checkpoint init → path to `checkpoint_finetuned_v2.pt`
     (`ckpt_state_dict_keys: ['model']` — already set in the config)
3. Reference hyperparameters (what produced the checkpoint you have):
   resolution 1024, batch 8, AdamW, `base_lr 5e-6`, `vision_lr 3e-6`,
   5 epochs, 1 frame per sample, 1 object per frame.
4. For a small "patch" fine-tune on a few hundred frames: keep the LR, drop to
   2–3 epochs, and hold out ~10% of frames to eyeball for regressions.

Output: `checkpoint.pt` with the same `{"model": state_dict}` format.

## 4. Validate before it touches the rover

```bash
SAMTP_CHECKPOINT=/path/to/your_new.pt \
  python -m rover_traversability.demo predict heldout_frame.jpg --out check.png
```

Compare overlays old-vs-new on your held-out frames — especially frames the
old model got RIGHT (catastrophic forgetting is the classic failure). Then
`demo live`, then `demo drive` in an open area.

## 5. Ship it

Upload to your own private HF model repo and switch:

```bash
hf repo create yourteam/samtp-yourteam --repo-type model --private
hf upload yourteam/samtp-yourteam your_new.pt checkpoint_finetuned_v2.pt
export SAMTP_HF_REPO=yourteam/samtp-yourteam
```

(Keep the filename or set `SAMTP_HF_FILENAME`.) Record the sha256 next to your
results so "which model was this run?" is always answerable.

## Licensing reminder

The base weights descend from SAM 2.1 (Apache-2.0) and FrodoBots Mini footage
(the public Mini-4K dataset is CC-BY-SA). If you publish fine-tuned weights,
carry the provenance note and attribution from the base model card
(`sanatem/samtp-mini-traversability`) with them.
