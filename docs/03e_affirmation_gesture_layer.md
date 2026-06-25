# AI Task Breakdown: Affirmation Gesture Layer (03e)

## Objective
The **Affirmation Gesture Layer** parses the most explicit non-verbal heuristic an infant uses for moral validation: Head Nodding and Head Shaking. Even if an adult is smiling, a shaking head signals "No." This layer extracts rhythmic, high-frequency spatial oscillations from the bystander's head tracking data to explicitly classify affirmation or negation.

---

## 📥 Input Requirements
- **`03a_attention_result.json`** (required cross-layer): This layer acts as a direct mathematical extension of 03a. It uses **only the head-pose** vectors (`head_pitch_rad`/`head_yaw_rad`, from 03a's MediaPipe FaceLandmarker); the L2CS-Net **gaze** vectors (`pitch_rad`/`yaw_rad`) are **not** used — they were proven to be noise (Resolved #11). A window with no head-pose sample is reported `no_head_pose` (unmeasured). Every emitted gesture carries `signal_source="head_pose"`. The pipeline strictly requires this file and must fail-fast (`RuntimeError`) if missing or empty, to prevent silent failures and wasted compute.
- **`filtered_manifest.json`**: For reaction window boundaries.

---

## 🛠️ Implementation Strategy

### 1. Data Reuse Pipeline
Do not re-run inference on the videos. Load the raw 3D head pose arrays from Layer 03a. For each sample in the reaction window, we have `(timestamp, pitch, yaw)`.
- **Window anchoring**: 03e does **not** read the wearer-climax reaction window directly — 03a only sampled each bystander where it was *detected*, so an unaligned window holds no trace samples and the layer scored **0/50** before this was fixed. It anchors via the shared `bystander_measurement_window` helper (`src/shared/bystander_window.py`; counts upstream 03a trace samples, `min = 5`, `± 2 s` pad), re-anchoring to the bystander-detection span when the strict window is too sparse (Resolved #1; see docs/03 Cross-Layer § Shared Helper).
- **Per-segment scoring + guardrails**: 03e scores one reaction *segment* at a time (the multi-window climax — docs/03 Cross-Layer § Multi-Window Reaction Segments), so a bystander's per-segment gestures form a reaction *trajectory* through the task rather than one label. It applies the two mandatory multi-window guardrails: **untracked** bystanders are skipped (Node-02 gives an untracked box a negative person id — 03d Resolved #4; positive-id tracks, even brief single-detection ones, are kept, preserving the single-detection anchoring of Resolved #1), and each **distinct `(person, measurement_window)` is scored once** (a segment that re-anchors to an already-scored window is dropped). Without these, a single untracked window was emitted up to **70×** (Resolved #6).

### 2. Time-Series Signal Processing (SciPy)
Nodding and shaking have distinct frequency signatures—they are rhythmic oscillations occurring roughly between 1Hz and 3Hz.
- **Uniform Resampling**: Layer 03a utilizes an adaptive stride (e.g., 8 FPS baseline, up to 32 FPS bursts). Because `scipy.signal.filtfilt` requires uniform sampling intervals to avoid frequency distortion, we must interpolate the raw signals onto a fixed-dt grid using `scipy.interpolate.interp1d` before filtering.
- **NaN Gap Limiting (Model-Calibrated)**: When the gaze model loses face tracking, `NaN` gaps occur. While we linearly interpolate across these gaps, bridging long absences creates a straight-line ramp that, after detrending and bandpassing, can fabricate spurious oscillations. We track the `interpolated_fraction` and short-circuit bystanders exceeding a threshold. *Crucially, this threshold is re-keyed to the upstream model* (e.g., `0.3` for L2CS-Net, `0.2` for CrossGaze) to match each model's specific tracking-loss distribution profile.
- **Saccade Suppression (FPS-Gated Median Smoothing)**: We consume the bystander's **head-pose** pitch/yaw signal (gaze is discarded as noise — Resolved #11). To suppress high-frequency jitter from aliasing into the 1-3Hz band, we apply a rank-order `medfilt(kernel_size=3)`. *Why it is gated:* This is strictly gated to activate only at high sampling rates (`fps >= 32.0`). At lower cadences, a 3-sample median window spans too much of a genuine nod's period and physically demolishes the tone.
- **Dynamic Bandpass Filtering**: We isolate human communication gestures (1-3Hz). However, because the effective FPS can be low, a static 1.5Hz cutoff might mathematically erase genuine 2.0Hz nods. We use a three-tier Nyquist-aware strategy that dynamically adjusts the frequency band based on the effective sampling rate.
- **Nodding & Shaking Detection**: Look for rhythmic variance. Use peak-finding (`find_peaks`) to identify rhythmic extrema on Pitch (nodding) and Yaw (shaking).
- **No frequency gate (it does not work here)**: a 1–3 Hz band gate on the detected `pitch_oscillation_hz` was tried and **rejected** — at this corpus's low / variable sampling a genuine 1 Hz nod is detected at ~**0.7 Hz** (the trustworthy *head-pose* nods too, median 0.73 Hz), overlapping slow drift, so any threshold rejects real nods along with the noise. Trustworthiness instead comes from using **head pose only** — gaze is discarded as noise (Resolved #11), so every emitted gesture is `signal_source="head_pose"`.
- **Ambiguous Gesture Classification**: If simultaneous pitch and yaw oscillations occur (a diagonal wobble), the gesture is classified as `ambiguous_wobble` when both confidence scores exceed `0.6` and are within a `0.15` delta. This prevents arbitrary bias toward nods or shakes.
- **Energy-Weighted Confidence**: A purely count-based confidence score treats vigorous nods and imperceptible micro-nods identically. The pipeline computes the RMS amplitude of the bandpass-filtered signal and applies it as a normalized multiplicative factor to the count-based confidence score.

### 3. Emotion Corollary
Combine this with Layer 03b. A "Smile" + "Nod" = Absolute positive validation. A "Smile" + "Shake" = Playful invalidation or disbelief. 

---

## 📤 Output Schema and Integration

To optimize resume cycles and make filtering decisions explicit, skipped videos (e.g., missing attention data or lacking tasks) are written as "sentinel" entries with a `skipped_reason` rather than being silently ignored.

**Example Output Data (`03e_affirmation_gesture_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03e_affirmation_gesture",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "segment_index": 0,
      "reaction_window_sec": [12.0, 15.0],
      "per_person": [
        {
          "person_id": 0,
          "pitch_oscillation_hz": 2.1,
          "yaw_oscillation_hz": 0.2,
          "interpolated_fraction": 0.04,
          "gesture_detected": "affirming_nod",
          "confidence": 0.94,
          "measurement_window_sec": [12.0, 15.0],
          "signal_source": "head_pose"
        }
      ]
    }
  ]
}
```
*(Sentinel Example: `{"video_id": "clip_2", "layer": "03e_affirmation_gesture", "tasks_analyzed": [], "skipped_reason": "no_attention_data"}`)*

> **Per-segment trajectory**: a task expands into one `tasks_analyzed` entry per reaction segment (`segment_index` + `reaction_window_sec`), so a bystander's entries across a task's segments form its **reaction trajectory** through the task (docs/03 § Multi-Window Reaction Segments). After the distinct-window dedup + genuine-track filter, each entry is a genuine moment, not a duplicate. A consumer wanting one per-task verdict aggregates a bystander's segment gestures (e.g. recency-weighted).

## Verification & Validation Check
- **Singular Video Test**: Plot the Pitch and Yaw arrays explicitly on a matplotlib line graph for a known head-nod video. Visually identify the sine-wave signature of the nod on the Pitch axis and verify the SciPy peak-finding logic successfully counted the nods.
- **Batch Test**: Pass 50 clips through the signal processor. Assert that the script executes in milliseconds (as it uses pre-computed vectors) and gracefully handles `NaN` values where L2CS-Net lost tracking on the face, filling defaults safely to avoid breaking the Pandas merge step on the **Mac Studio (M4 Max, 64 GB unified memory)**.

## 🚀 Implementation Accomplishments (April 2026)

The Affirmation Gesture Layer has been implemented successfully in `src/layer_03e_affirmation_gesture/pipeline.py`.

- **Signal Extraction:** Built a pipeline that parses the `pitch_rad` and `yaw_rad` attention traces from `03a_attention_result.json` filtered strictly to the task's reaction window boundaries.
- **Bandpass Filtering:** Employed `scipy.signal.butter` and `filtfilt` to isolate the 1-3Hz frequencies characteristic of human nodding and shaking. A three-tier Nyquist-aware strategy selects the appropriate filter configuration based on the effective sampling rate.
- **Zero-Crossing Detection:** Used `find_peaks` to identify rhythmic extrema (peaks and troughs) with specific prominence thresholds to compute frequency oscillations safely without running inference on raw video.
- **Uniform Resampling:** Added `scipy.interpolate.interp1d` to resample non-uniform timestamps (from 03a's adaptive stride of 0.2s-0.5s) onto a fixed-dt grid before applying `filtfilt`, which requires uniform spacing.
- **Hard Dependency Validation:** The pipeline now raises a `RuntimeError` at startup if `03a_attention_result.json` is missing or empty, enforcing the documented hard dependency contract.
- **Ambiguous Gesture Classification:** Added `ambiguous_wobble` classification when both pitch and yaw oscillate with similar confidence (within 0.15 threshold), preventing arbitrary nod/shake bias.

## 🧪 Resolved Issues & Implementation Refinements

*(All historical implementation refinements regarding Nyquist limits, adaptive resampling, saccade suppression, interpolation thresholding, and missing-dependency enforcement have been standardized and integrated directly into the core Implementation Strategy.)*

1. **03e Read the Wearer-Climax Reaction Window but 03a Only Sampled Bystander-Detection Windows — 0/50 Yield (Resolved - June 14)**:
   - **Problem**: The June 14 50-clip smell test (`e2e_reports/2026_06_14_layer03e_50/`) scored **0/50**. `process_video` filtered 03a's `attention_trace` to the strict `task_reaction_window_sec` (wearer optical-flow climax ±1 s), but 03a's window-restricted sampling only runs gaze within ±2 s of each **bystander detection** timestamp, and detections sit a median ~6.8 s from the wearer's climax (docs/03d Resolved #1). So the reaction window held **zero** trace samples on 17 of the 23 clips with 03a data → `insufficient_trace`. This is the identical wearer-vs-bystander anchoring mismatch that 03b (Resolved #2) and 03d (Resolved #1) fixed; 03e never received it.
   - **Solution** (Option A): Added `_measurement_window()` mirroring 03d's `_bystander_measurement_window` (**June 14: both now delegate to the shared `src/shared/bystander_window.py` helper — see docs/03 Cross-Layer § Shared Helper**) — keep the reaction window when it already holds ≥ `MIN_TRACE_POINTS` trace samples; otherwise anchor to the bystander DETECTION nearest `task_climax_sec` ± `ANCHOR_SPAN_DETECTIONS`, padded by `WINDOW_PAD_SEC` (= 03a's 2 s sampling pad) and bounded by `MAX_ANCHOR_SPAN_SEC` (= 30 s). Each scored record now carries `measurement_window_sec` + `window_source`, and the sentinel reason is aggregated via `_aggregate_skip_reason` (`span_capped` / `insufficient_trace` / `mixed_skip`). **Validated on the June 14 re-run against the existing 03a traces (no 03a re-run needed): 0 → 20 scored clips, 83 person-task rows (70 `bystander_anchored`, 13 `reaction_window`), all anchored spans ≤ 28 s.** *(This restores yield; the now-visible gestures remain subject to Unresolved Issue 1 (gaze vs head pose) and the new NoFace-zero artifact (Unresolved Issue 2) — i.e. the fix makes 03e produce output, not yet trustworthy output.)*

2. **`filtfilt` padlen Crash on Short Windows Aborted the Whole Clip (Resolved - June 14)**:
   - **Problem**: The 6 June-14 clips whose window overlapped 03a's samples (the only scorable clips) all raised `ValueError: The length of the input vector x must be greater than padlen, which is 15` from `scipy.signal.filtfilt`: a windowed + resampled per-person signal of ~11–15 samples is shorter than the 2nd-order Butterworth bandpass `padlen` (15), and filtfilt **raises** rather than degrading. With no per-person `try/except`, the exception aborted the **entire clip** (no record, not even a sentinel) — one short-trace person poisoned every co-occurring person, so the run wrote 44 records not 50 and the scorable clips yielded nothing.
   - **Solution** (Option A): Added `_safe_filtfilt()` (returns `None` when `len(sig) <= 3·max(len(a),len(b))`); `_detect_oscillation` falls back to the raw-detrended peak-finding path on a too-short signal instead of raising. Wrapped the per-person body in `try/except` so any unexpected failure records that person's skip (`error`) and can never abort the clip. **Validated: the June 14 re-run produced 50 records with 0 crashes (empty error log).**

3. **NoFace / Lost-Tracking Samples Were Encoded as `pitch=yaw=0.0`, Fabricating Step-Edge "Gestures" (Resolved - June 14)**:
   - **Problem**: With Resolved #1/#2 in place the June 14 re-run emitted 27 gestures, but spot-checking showed they were artifacts, not head motion. 03a writes lost-tracking samples into `attention_trace` as `pitch_rad = yaw_rad = 0.0` with `target = "NoFace"` (**not `NaN`**), so `_fill_nan` / `interpolated_fraction` never flagged them (all 27 reported `interpolated_fraction = 0.0`) and the step transitions in and out of those zero-plateaus were counted by `find_peaks` as rhythmic extrema. The visually-verified exemplar `0780244d` pid5 (`affirming_nod`, conf 1.0) was a flat-zero plateau + a single spike + a 0→1.05-rad step — not an oscillation.
   - **Solution** (Option A, 03e-load-side variant — no 03a re-run): in `process_video`, samples whose `target == "NoFace"` are mapped to `NaN` when building the pitch/yaw arrays, so they flow through the existing `_fill_nan` + `interpolated_fraction` guard instead of forming real-valued steps. **Validated on the June 14 re-run against the existing 03a traces: detected gestures 27 → 6 (the NoFace step-edge false positives — including `0780244d` pid5 — are gone, that bystander now correctly `over_interpolated`); 7 clips became honest `over_interpolated` sentinels; `interpolated_fraction` now spans 0–0.27 (was uniformly 0.0).** The 6 survivors are gaze-derived and remain subject to Unresolved Issue 1. Covered by `test_noface_samples_treated_as_missing` and `test_small_noface_fraction_still_detects`.

4. **Nod/Shake Was Derived From GAZE, Not Head Pose — Confident False Nods From Eye Motion (Resolved - June 14)**:
   - **Problem**: 03e's premise (nod = pitch oscillation, shake = yaw oscillation) assumes HEAD orientation, but `process_video` read 03a's `pitch_rad`/`yaw_rad`, which are L2CS **gaze** angles (where the *eyes* point). Gaze moves with saccades / smooth-pursuit while the head is still, so non-nod eye motion aliased into the 1–3 Hz band — on the post-Resolved-#3 re-run all 6 surviving `affirming_nod`s were gaze artifacts, not head motion (e.g. `0780244d` pid5's head-crop strip showed the bystander looking around, not nodding).
   - **Solution** (Option A): New shared `src/shared/head_pose.py` (`HeadPoseEstimator`) runs MediaPipe **FaceLandmarker** on the same face crop 03a already uses for gaze, decomposing `facial_transformation_matrixes` → head Euler **pitch (nod) / yaw (shake)**. 03a emits additive `head_pitch_rad`/`head_yaw_rad` per trace sample (gated by `ENABLE_HEAD_POSE` + the `face_landmarker.task` asset; `null` when FaceLandmarker finds no face — its own tracking-loss). 03e now **prefers head pose** (`signal_source` provenance), mapping `null` head pose → `NaN` (Resolved #3 semantics) and falling back to gaze only when a window has no head pose at all. **Validated on the 13 gesture-producing clips (`e2e_reports/2026_06_14_layer03e_headpose/`): gaze false nods 6 → 1 — the 5 clips with any head-pose coverage are rejected (head pitch flat → `none`, or `over_interpolated`); the lone survivor (`0780244d` pid1) is an honest gaze-fallback, flagged `signal_source = "gaze"` because FaceLandmarker resolved no face in its window.** *(Amplitude re-tuning of the 1–3 Hz bands / `PEAK_PROMINENCE` / `RMS_THRESHOLD` for head-pose scale, and validation against true nod footage, are deferred — the Ego4D sample is thin on deliberate nods. The sparse head-pose coverage this surfaced is accepted as the honest gaze-fallback floor — Resolved #5.)* **(Superseded by Resolved #11 — gaze is now discarded entirely, not preferred-then-fallback; it was proven to be noise.)**

5. **Head-Pose Coverage Is Sparse on Small Bystander Faces — Accepted as the Honest Gaze-Fallback Floor (Resolved - June 14)**:
   - **Problem**: The June 14 head-pose run (`e2e_reports/2026_06_14_layer03e_headpose/`) showed MediaPipe FaceLandmarker resolves head pose on only **~8%** of trace samples (42% `NoFace`, **50% face-present-but-no-landmarks** on small / steeply-angled / motion-blurred bystander faces), so most per-person windows fall back to gaze or `over_interpolated`. Resolved #4 therefore rejects gaze artifacts only *where head pose exists*, which is rare on this egocentric corpus.
   - **Solution** (Option C — accept the honest floor): **No code change.** 03e already emits per-vector `signal_source` provenance (Resolved #4), so head-pose-derived gestures (`signal_source="head_pose"`) and the lower-confidence gaze-fallback gestures (`signal_source="gaze"`) are explicitly distinguishable, letting downstream consumers (Layer 04) weight gaze-derived gestures lower or filter them. The recovery alternatives (crop-upscaling, a small-face 6DoF model) were considered not worth the added compute/dependency given this corpus's low absolute gesture yield. The sparse-coverage characteristic is an accepted modality floor, not a defect — 03e stays honest (provenance, no false confidence) about what it could and could not measure. **(Superseded by Resolved #11: the gaze fallback was subsequently proven to be noise — 0.25 % precision — and is now discarded; the honest floor is head-pose-only, and a 6DoF recovery of the missing head pose was tested and deferred.)**

6. **Multi-Window Over-Counting — Distinct-Window Dedup + Untracked-Track Filter (Resolved - June 24)**:
   - **Problem**: The bystander-aware multi-window climax (02 Resolved #22) expands each task into ~13 reaction segments, and 03e re-anchors every segment to the bystander's nearest detection (Resolved #1) — so for a sparse track *all* its segments collapse onto the **same** measurement window. On the top-200 (climax-populated) re-run this reported the identical reaction once per segment: **34,781 "nods"** that were only ~1,701 distinct `(clip, person)` reactions counted a median **13× (max 70×)** each — e.g. `1fe55d7f` person `-124` (an *untracked* negative-id box, 03d Resolved #4) was the same 4 s window scored **70 times**. Nod:shake ran ~37:1, implausible for real affirmation.
   - **Solution**: Two cross-layer guardrails in the per-segment loop (docs/03 § Multi-Window Reaction Segments): **(1) distinct-window dedup** — a per-clip `(person_id, measurement_window)` set; a segment that re-derives an already-scored window is skipped; **(2) untracked-track filter** — skip negative person ids (untracked boxes), keeping genuine positive-id tracks even when brief (preserving the single-detection anchoring of Resolved #1, since the dedup already prevents over-counting). A **frequency gate** (reject sub-1 Hz "nods") was prototyped and **rejected**: the detected `pitch_oscillation_hz` is biased low at this corpus's variable sampling — a genuine 1 Hz nod, *including the trustworthy head-pose ones*, is detected at ~**0.7 Hz** — so it overlaps slow drift and cannot separate them. **Validated on the top-200 re-run**: nods **34,781 → 1,023** (one per genuine `(clip, person)`; per-person cross-segment count median 13× → **1×**; **0** untracked nods), runtime **4,070 s → 37 s**, suite **33/33** (`test_dedup_and_untracked_filter`). The residual 1,023 nods are **1,013 gaze-fallback / 10 head-pose** — i.e. the multi-window over-count is fixed and what remains is the sparse-head-pose gaze-fallback floor, **since resolved by discarding gaze entirely (Resolved #11)**, *not* a multi-window artifact.

11. **Gaze Is Noise — Head-Pose-Only Gestures + 6DoF Recovery Tested & Deferred (Resolved - June 24)**:
   - **Problem**: After the multi-window fix (Resolved #6), **1,013 of 1,023** detected "nods" were `signal_source="gaze"` (only 10 head-pose). Resolved #4/#5 had kept gaze as a *flagged, lower-confidence fallback*. A corpus concordance test — run 03e twice over the 03a trace (once normal, once with head pose nulled to **force** gaze) and compare on the windows that were head-pose — proved gaze is not merely low-confidence, it is **anti-signal**. Versus head pose as ground truth, gaze-nod **precision = 0.25 %** (3/1180), it **missed 70 %** of real nods, and it **fabricated a "nod" on 57 %** of non-nod windows. The mechanism is physiological and corpus-independent: the **vestibulo-ocular reflex** counter-rotates the eyes to stabilise gaze *against* head rotation, so gaze barely moves *during* a nod (missed) yet swings for reading/scanning saccades (false) — gaze and head-nod are decoupled, so gaze can never proxy a nod.
   - **6DoF recovery tested, not assumed**: gaze being dead, the only route to denser trustworthy gestures is recovering head pose where FaceLandmarker's 468-landmark step fails. Probed empirically: bystander faces are **large when found (~90 px median)** so *size* isn't the blocker — but recovery did **not validate** against the FaceLandmarker reference on dense windows: **solvePnP** from RetinaFace's 5 landmarks gave pitch `r=+0.29`, **6DRepNet** `r=+0.08` (yaw control ~0); face *re-detection* with the available detector was itself only ~24 %. A robust attempt (MediaPipe BlazeFace detector + a within-window temporal validation) is a real sub-project with an uncertain ceiling, so it is **deferred**, not built.
   - **Solution** (head pose only): 03e no longer reads gaze at all. A window with no FaceLandmarker `head_pitch_rad` sample is reported **unmeasured** (`no_head_pose`), *never* a false neutral that would dilute the reward signal; the existing `interpolated_fraction` guard then naturally restricts gestures to *dense* head-pose windows. **Validated on the top-200 re-run**: gestures **1,023 / 18 / 22 → 10 nod / 15 shake / 5 wobble**, all `signal_source="head_pose"`, over **2,089 trustworthy head-pose windows** (the rest measured neutrals); 68 clips have no head pose anywhere and are honestly `no_head_pose`. Sparse but trustworthy — 03e's honest floor on small-bystander egocentric video. **Supersedes the gaze-fallback of Resolved #4/#5** (kept below for history; do not re-add gaze). Suite **34/34** (`test_gaze_only_window_is_unmeasured_not_scored`).

### 🧪 Test Suite Results (25/25 Passed across `test_layer_03e.py` + `test_head_pose.py`)

A comprehensive verification suite in `tests/test_layer_03e.py` validates the following:
- **Nod/Shake/None Classification:** Synthetic sine-wave verification at ~2Hz.
- **NaN Resilience:** Gap patching using `np.interp` before filtration.
- **Hard Dependency Validation:** Raising `RuntimeError` on missing dependency files.
- **Ambiguous Wobble:** Correct classification of simultaneous equal-amplitude oscillations.
- **Non-Uniform Sampling:** Verification of detection accuracy after uniform resampling.
- **Upstream Gaze-Model Calibration:** `MAX_INTERPOLATED_FRACTION` re-keys to 0.3 / 0.2 / 0.25 against `processing_meta.model_used` across L2CS-Net, CrossGaze, 3DGazeNet, unknown-model, fallback-only, and missing-meta branches.
- **Read-Window Anchoring (Resolved #1):** keeps a dense reaction window, anchors (padded) to the climax-nearest detection when the window is empty, caps an over-long span, and recovers an offset trace end-to-end.
- **filtfilt Short-Signal Guard + Per-Person Isolation (Resolved #2):** `_safe_filtfilt` returns `None` below padlen; a short window falls back without crashing; a per-person exception yields a sentinel, not a clip abort.
- **NoFace-as-Missing (Resolved #3):** `target='NoFace'` samples become NaN — a NoFace-dominated window is rejected (`over_interpolated`), while a few NoFace samples are bridged without killing a genuine nod.
- **Head-Pose Only (Resolved #4 / #11):** 03e uses `head_*` exclusively — an oscillating gaze with a flat head scores `none`; a real head-pitch oscillation is a nod; `null` head pose is missing (NaN), not zero; a window with no head pose at all is `no_head_pose` (unmeasured — gaze is never used).
- **Head-Pose Estimator (`test_head_pose.py`):** Euler decomposition (pitch = X-axis nod, yaw = Y-axis shake); degrades to `None` when the asset is missing (never raises).

## ⚠️ Unresolved Issues & Suggestions

_None at this time — the head-pose-coverage limitation is accepted as the honest **head-pose-only** floor; gaze was proven to be noise and discarded, and a 6DoF recovery was tested and deferred (Resolved #11)._

> **June 14 update:** the smell-test-blocking findings (window mismatch; `filtfilt` crash) **and** the NoFace-zero step-edge artifact are now **resolved above** (Resolved #1–#3). With them fixed, the June 14 re-run yields **6 detected gestures** (down from 27) — all now gaze-derived, i.e. squarely this Issue 1. A dedicated head-pose source (Option A) is what remains to make 03e's output *trustworthy*, not merely present.
