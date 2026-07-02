# Layer 03a (Attention/Engagement) — Functional Validation & Spot-Check Report

**Date**: June 3, 2026
**Goal**: Confirm Layer 03a *runs end-to-end* on real Node-02 output and that its gaze/attention scores *pass a visual smell test* — i.e., the score matches where the bystander is actually looking. This is a correctness probe, not a scale or accuracy benchmark.
**Host**: Mac Studio (M4 Max, 64 GB), MPS.

## Terminology
The videos that pass Node 02 (social-presence + task labeling) and are retained in `filtered_manifest.json` are referred to here as **"filtered clips"** (synonyms in code/docs: the *retained* / *reservoir* set, *social-positive* clips). The current reservoir holds **1,000** filtered clips.

## Sample
10 filtered clips, sampled evenly across the Node-02 `social_presence_score` range (1.2 → 272) from the 703 clips with `duration ≤ 2500 s` and `bystander-frames ≤ 300`. The high-runtime outliers (the 833-track / multi-hour clips) were deliberately excluded so the run finished in ~12 min; see Issue 2 for why those are also the clips where 03a is least reliable.

## Run summary
- **10 / 10 clips produced a result. 0 failed-to-score. 0 errors. 0 crashes** (single wrapper attempt, `rc=0`).
- L2CS-Net (ResNet50, Gaze360 weights) loaded and ran on MPS — `model_used = l2cs_net_3d_gaze` on every clip (not the dummy fallback), so scores are real.
- Resumable per-clip (atomic write + `processed_ids`); ran under the standard auto-restart wrapper.
- **Mechanically, the layer works.** The issues below are about *output quality*, not stability.

## Per-clip results (sorted by score)
`score` = `aggregate.mean_attention_all_persons` (mean per-person `average_attention_score`).

| score | clip | tracks | engaged | persons | trace pts | 02-score |
|---|---|---|---|---|---|---|
| 0.16 | `10167fcf` | 1 | no | 1 | 123 | 1.2 |
| 0.23 | `14f5014d` | 6 | yes | 6 | 16072 | 272.1 |
| 0.27 | `5f4e4722` | 3 | no | 3 | 543 | 5.2 |
| 0.32 | `067b03df` | 7 | yes | 7 | 1994 | 26.4 |
| 0.32 | `6fd026d8` | 13 | yes | 13 | 6524 | 80.1 |
| 0.33 | `0a6d4809` | 11 | yes | 11 | 5863 | 56.7 |
| 0.33 | `3f503d0b` | 7 | yes | 7 | 1708 | 16.3 |
| 0.38 | `0518d285` | 29 | yes | 29 | 12911 | 112.5 |
| 0.39 | `02c40de9` | 6 | yes | 6 | 3837 | 39.9 |
| 0.41 | `25ffbde8` | 5 | yes | 5 | 927 | 9.2 |

**Highest mean**: `25ffbde8` (0.41) · **Lowest mean**: `10167fcf` (0.16) · **Failed-to-score**: none.

### Per-sample score distribution (50,502 samples across all 10 clips)
- `score == 0.00`: **43%** (gaze ≥60° off both camera and hands)
- `0 < score < 0.70`: **36%**
- `score ≥ 0.70`: **21%**
- **Every clip's peak single-sample score is 0.80–1.00** — the model *does* fire high when it judges someone is looking; the low clip means are the average being dragged down by the 43% zero-samples.

> **Read this carefully:** "highest/lowest *mean*" is dominated by clip length, not engagement — the shortest clips have the highest means (fewer not-looking samples to average in). The meaningful signal is the **per-sample trace** (peaks + sustained windows), not `average_attention_score`. See Issue 3.

## Spot checks (gaze overlay: green = bystander bbox, red arrow = gaze direction, label = score/target/pitch/yaw)

### 1. Highest-scoring clip `25ffbde8` @ peak (score 0.92, "Camera") — ❌ the "bystander" is a DOG
![highest peak — dog](frames/sc03a_highest_peak.jpg)
L2CS-Net regressed a confident "looking at camera" (0.92) on a **dog's face** filling the frame while the wearer pets it. The dog is a Node-02 bystander track (YOLO-pose + qwen confirmed it as a "person"); 03a has no human-face check, so it scored it as an attentive human. **The single highest-scoring track in the batch is invalid.** → Issue 1 (03a) + new Issue 2 (02).

### 2. Within the same clip `25ffbde8`, same track @ low (score 0.00) — ❌ box on empty terrain
![highest clip low — empty box](frames/sc03a_highest_low.jpg)
The *same* `person_id` track's box at t=13 s sits on **empty canyon ground**; the real hikers are center-frame, outside the box. One track's boxes land on a dog at t=31 s and empty terrain at t=13 s — stale/drifting boxes under fast egocentric motion (boxes are from Node 02's 1/3-FPS ByteTrack, reused with up to 2 s tolerance). → Issue 2.

### 3. Known-social clip `0a6d4809` "Talking with friends" @ peak (score 0.99, "Camera") — ⚠️ distant, glass-occluded, facing each other
![talking peak — distant glass](frames/sc03a_talking_peak.jpg)
Two people behind a glass patio door, far away and partially occluded by mullions, are scored 0.99 "looking at the wearer" — but they are clearly oriented toward *each other*. Gaze regression on tiny/distant/occluded faces is unreliable; a near-max "at camera" score here is not trustworthy. → Issue 1.

### 4. Lowest-scoring clip `10167fcf` @ its peak (score 0.80, "Camera") — ⚠️ profile view, offset box
![lowest peak — desk profile](frames/sc03a_lowest_peak.jpg)
A man at a desk in profile, looking at his laptop (not obviously at the wearer), scored 0.80 "at camera"; the green box is also offset left of him. The clip *mean* (0.16) is plausibly correct (he mostly isn't looking at the wearer), but the 0.80 peak is a marginal/false high.

## Smell-test verdict (initial run)
**Runs: yes. Trustworthy output (initial): not yet.** Across 4 visual checks, the highest-confidence outputs were a dog (0.92), distant people facing each other (0.99), and a profile-view desk worker (0.80). The mechanics were sound, but three issues made the raw scores unreliable. All four issues (3 in 03a + 1 in 02) were then **remediated and re-validated** below.

## Post-fix re-run (v2) & resolution
Implemented the recommended fixes and re-ran 03a on the same 10 clips (`03a_attention_result_10_v2.json`):
- **03a Issue 1 — face-presence gate** (BlazeFace short-range, conf ≥ 0.5): crops with no resolvable human face now score `0.0 / target "NoFace"` instead of a bogus gaze. The **dog crop is rejected** (validated directly), and `NoFace` rates run 6–95% per clip — the high rates reflect Ego4D's many distant/profile/occluded bystanders, exactly where L2CS gaze was untrustworthy.
- **03a Issue 2 — bbox re-detection**: YOLO-pose re-detects person boxes on each sampled frame and IoU-matches the track, replacing the up-to-2 s-stale manifest box before cropping.
- **03a Issue 3 — length-invariant metrics**: added `attended_fraction` (share of samples ≥ 0.5) and `engaged_attention_score` (mean over score > 0), since `average_attention_score` tracks clip length.
- **02 Issue 2 — dog flag**: clip `25ffbde8` marked `flagged_invalid` in `filtered_manifest.json` (no 02 re-run); 03a skips flagged clips. Confirmed skipped in the v2 run.

**v2 outcome (per-clip mean → attended_fraction / engaged_attention_score):** the means collapse toward 0 (NoFace=0 dominates) but the new metrics carry the real signal — e.g. `6fd026d8` attended 0.23 / **engaged 0.80**, `14f5014d` attended 0.35 / engaged 0.50. Re-spot-checking the v2 high scorers: `14f5014d` (0.99, POV_Actor_Hands) is a real man looking down at his hands; `6fd026d8` (1.00, Camera) is a real golfer — **both real humans, no dog.**

**Residual (pre-existing, not regressed):** L2CS still over-confidently scores hard poses (sunglasses, profile, bent-over) when a face *is* detected — this is the documented L2CS gaze-noise limitation (03a "Model Selection Rationale"), now bounded to real human faces by the gate. The face gate's aggressiveness (short-range model misses distant faces) is a recall/precision tradeoff; both gates are env-toggleable (`SR_03A_FACE_GATE`, `SR_03A_BBOX_REDETECT`) and the threshold is tunable (`SR_03A_MIN_FACE_CONF`).

## Issues resolved (transitioned per the Issue-Resolution skill)
- **`docs/03a` Issues 1, 2, 3 → Resolved** (#2 face gate, #3 bbox re-detect, #4 engagement metrics). Unresolved section now empty.
- **`docs/02` Issue 2 (non-human) → Resolved #17** (dog-flag remediation). 02 Issue 1 (multi-person false-negative) remains Unresolved — its fix requires a 02 re-run that was deliberately deferred.

## Artifacts
- `manifest_10.json` — the 10-clip input sample (dog clip now `flagged_invalid`)
- `03a_attention_result_10.json` — initial 03a output; `03a_attention_result_10_v2.json` — post-fix output (face gate + re-detect + new fields)
- `frames/sc03a_*.jpg` — initial gaze-overlay stills; `frames/sc03a_v2_*.jpg` — post-fix stills (real humans)
- Runner: `scratch/run_03a_10.py`; overlay tool: `scratch/spotcheck_03a.py`; face-gate validator: `scratch/validate_facegate.py`
