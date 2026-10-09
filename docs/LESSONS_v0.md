# Lessons from v0: the Social-Affective Filter (SAF), April–October 2026

v0 is archived at git tag **`v0-saf-final`**. Everything below is recoverable with `git checkout v0-saf-final`, including the full per-layer docs with their resolved-issue histories. Derived data artifacts remain on the SSD at `/Volumes/Extreme SSD/social_robotics/` (`full_run_2026_06_18/`, `bench_v0/`). **The raw Ego4D videos were deleted on October 8, 2026.** Clip ids and re-download instructions are in `DELETED_2026-10-08.json` next to them. Per-layer datasets were published to Hugging Face under `louisye/social-robotics-*`.

This file keeps what v0 *taught*, so that no one has to rediscover it.

## What v0 was

- **Node 01–02:** Ego4D acquisition plus a social-presence filter (YOLOv8-pose, then a `qwen2.5vl` multi-person gate).
- **02b:** bystander-anchored "reaction segments".
- **Six feature layers:**

  | Layer | Signal | Method |
  |---|---|---|
  | 03a | Gaze | L2CS-Net |
  | 03b | Categorical emotion | HSEmotion + a Gemma reasoner |
  | 03c | Prosody | emotion2vec+ |
  | 03d | Proxemics | Depth Anything + SAM |
  | 03e | Nods/shakes | Head pose |
  | 03f | Flinches | YOLO-pose |

- **Downstream:** a joined segment dataset, and **SocialRobotics-Bench v0**, a human-rated benchmark for "is this reaction feedback on the wearer's action, and what is its valence?"

At the end: 991 clips, 23,378 segments (19,053 real + controls), 15,946 QA pairs.

## The decisive negative result

- **Human rating was impossible.** On trying to rate the 349 benchmark moments, the maintainer could not tell what emotion was displayed or what task it was about. **0 / 349** moments were rated.
- **A frontier model failed the same way.** Gemini pre-seeds came out **130 approving / 67 neutral / 0 disapproving**. Rationales showed ordinary engagement read as approval ("picks up card carefully", "hands card carefully"), and the wearer and the bystander confused.
- **The corpus barely contained signal** (19,053 real segments):

  | Channel | Coverage |
  |---|---|
  | Emotion | Readable on **5.1%**, non-zero on **1.1%** |
  | Head gestures | **18 nods and 36 shakes** in the entire corpus |
  | Proxemics | **98% "Neutral"** |
  | Prosody | 100% coverage only because it was *ambient* room audio, never attributed to a bystander |

  None of these channels was ever validated against ground truth.

## Root causes (why more engineering would not have fixed it)

1. **No outcome variable.** The thesis is about reward, but no task success label existed anywhere. The only possible validation was human judgment, which is unscalable and, here, ill-posed.
2. **The phenomenon was absent from the data.** The thesis needs three things together: someone attempts something with an outcome, someone watches, and the watcher's reaction depends on that outcome. The top Ego4D scenarios in the corpus were cleaning, construction, grocery shopping, cards, cooking and eating, which is steady-state co-activity where bystanders are not evaluating the wearer.
3. **Segment selection selected proximity, not consequence.** 02b picked moments when a bystander was *close*, not moments when the wearer's action *had an effect*.
4. **Categorical emotion was the wrong intermediate.** Eight basic-emotion classes on distant egocentric faces were noise. Hand-built channels each had to be gated down to roughly 1% yield to be honest.
5. **Order of work.** Four months of extraction engineering before any evaluation existed. → The new project builds the eval harness first ([`03_eval_harness.md`](03_eval_harness.md)).

## Technical findings worth keeping

- **Optical flow on egocentric video tracks the wearer's ego-motion and passing objects**, not social moments. The flow-peak "climax" was uniform-random within its bystander cluster (p10 0.06, p90 0.95), and 43.5% of its windows held no bystander detection.
- **Wearer-vs-observer window mismatch.** Windows anchored on the actor's motion missed the observer, who was detected a median of ~7–10 s away. The bug recurred independently in four layers. Anchor measurement windows to the *observer*.
- **HSEmotion on distant, blurry egocentric faces is near-uniform**: a median top-class probability of 0.18 vs. the 1/8 = 0.125 baseline. A confidence gate (≥ 0.4) cut yield to 2/50 clips. BlazeFace false-positives on non-face crops (backpacks, distant bodies).
- **Gaze-derived nods are noise**: 0.25% precision, fabricating a "nod" on 57% of non-nod windows, because the vestibulo-ocular reflex decouples gaze from head motion. Use head pose only.
- **Multi-window counting duplicates.** Re-anchored windows photocopied the same reaction: 34,781 raw "nods" → 1,701 distinct (clip, person) reactions → **30** trustworthy head-pose gestures. Always dedupe by distinct (person, measurement window), and drop untracked (negative-id) phantom tracks.
- **Tracking explodes on long clips**: up to 829 spurious tracks per clip, median 48. Per-clip caps are mandatory for any per-person heavy model.
- **Sparse sampling starves kinematics.** 1 frame per 3 s is enough for keep/drop decisions but not for motion. Window-dense re-detection took flinch detections from 0 → 7 on 25 clips.
- **Social-presence false positives in Ego4D:** faces on TVs and monitors, side-by-side stereo captures, the wearer's own chin, dogs. Each needed its own gate. After the fixes, 33.6% of processed Ego4D videos passed the filter.

## Operations (still apply; the tooling survives in `tools/` and `src/shared/`)

- **Multi-hour runs must be detached** (`tools/daemonize.py`, PPID 1). Agent-harness background tasks are reaped after ~1–2 h. Wrap them in `tools/run_supervised.sh` to survive native macOS crashes ("Python quit unexpectedly").
- **Ollama:** the Python client ignores its timeout, so call the HTTP API with an enforced timeout (`src/shared/vlm_client.py`). Set `num_ctx` small (4096) for single-image prompts; the 128k default allocated ~52 GB. Pinned tags drift, so resolve against `ollama list` (`src/models_config.py`). Set `OLLAMA_NUM_PARALLEL` > 1 for concurrent calls.
- **Video decode:** never `cap.set(POS_FRAMES)` per frame on H.264. Every seek re-decodes from the keyframe, 10–20× wasted work. Decode sequentially with `grab()`/`read()`. Random seeks are fine only for a handful of widely spaced frames.
- **MPS multi-process:** 3 subprocess workers sharing one GPU gave ~3× throughput on decode-bound layers. Stagger the model loads and keep a free-RAM floor.
- **Storage:** no video on the internal SSD, ever. Point `TMPDIR` and `HF_HOME` at the external SSD before running anything heavy.

## What v0 artifacts can still be used for

- **The 991-clip Ego4D segment dataset:** a false-positive set (no evaluative reaction expected) for the new harness. Using it with the new encoders means re-downloading the chosen clips first. See [`02_data_sources.md`](02_data_sources.md).
- **The `bench_v0` rating UI** (at the tag): a working local video + form tool, if a QA-of-our-code view is ever needed.
- **emotion2vec+**: a working on-Mac audio encoder, reusable as an *embedding*, never as emotion categories.
