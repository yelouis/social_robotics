# AI Task Breakdown: Reasonable Emotion Layer (03b)

## Objective
The **Reasonable Emotion Layer** leverages the affective reactions of bystanders to deduce the success, failure, or general outcome of an action performed by the POV actor. 

Crucially, this layer analyzes **emotion transitions and durations** over time (e.g., a fleeting shock turning into sustained applause) purely within the dynamically bounded temporal moment of the action. It then outputs both granular temporal slices and a **Late-Stage Weighted Average** representing final task success.

---

## 📥 Input Requirements
- **`filtered_manifest.json`** (required): This layer iterates through the `identified_tasks` array provided by Node 02. For each task, it retrieves:
  - **Contextual Task Label** (`task_label`): To know what action the bystander is reacting to.
  - **Temporal Alignment** (`task_reaction_window_sec`): Derived dynamically via Optical Flow peak detection (+ VLM refinement) and Action Velocity profiling, this is the exact timestamp boundary within which we sample bystander reactions for a given task.
- **Cross-layer (optional)**: `03a_attention_result.json` — Used to weight emotion readings. A bystander who isn't even looking at the POV actor (attention score < 0.3) during the reaction window is flagged as unreliable.

---

## 🛠️ Implementation Strategy

Because a single video might contain multiple tasks, this sequence occurs iteratively for *each* task inside the `identified_tasks` array.

### Step 1: Expectation Generation (Gemma 4)
Using the extracted Contextual Task (e.g., "Juggling apples"), prompt a locally running LLM (**Gemma 4** via Ollama) to generate baseline emotional expectations. On the **Mac Studio (M4 Max, 64 GB unified memory)** target host, the current default `gemma4:26b` (27B-class, ~15 GB) leaves substantial headroom (see Resolved Issue #16). Concurrent tracking arrays for multiple bystanders coexist without swapping.

**Structured Prompt Template:**
```text
You are analyzing bystander reactions to this task: "{task_label}"

Generate sets of emotions a bystander would display if:
1. The outcome is POSITIVE (successful, impressive).
2. The outcome is NEGATIVE (failed, dangerous).

Respond in EXACTLY this JSON format:
{
  "positive_emotions": ["emotion1", "emotion2"],
  "negative_emotions": ["emotion1", "emotion2"],
  "neutral_baseline": ["bored", "neutral", "distracted"]
}
```

### Step 2: Temporal Sampling & Pairwise Chunking
We track the bystander strictly within the `task_reaction_window_sec` for the current task.
1. Run a SOTA emotion model (specifically **HSEmotion-PyTorch**) on the bounding box faces at **3-5 FPS** across the reaction window.
   *Architectural Note:* We explicitly use HSEmotion-PyTorch instead of alternatives like Py-Feat because HSEmotion runs natively on Apple Silicon via Metal Performance Shaders (MPS). CPU-bound emotion models create severe inference bottlenecks on the Mac Studio, so maintaining MPS acceleration is critical for this layer's throughput.
2. Form an Emotion Timeseries (e.g., `Neutral (0.5s) -> Shock (0.5s) -> Joy (2.0s)`).
3. **Chunking into Pairs**: Break the sequence into consecutive Pairwise Transitions. Emotion logic is evaluated strictly one transition at a time to anchor LLM reasoning.
   - Transition 1: `Neutral -> Shock`
   - Transition 2: `Shock -> Joy`

### Step 3: Accumulated History Evaluation (Gemma 4)

**State-Change Filter:** To optimize performance and reduce latency, the pipeline first checks if the emotion label changed between the two samples. If `start_emotion == end_emotion`, the LLM call is skipped entirely, and the previous temporal slice's duration is simply extended.

For genuine emotional transitions, the pair is fed sequentially to Gemma 4. To maintain chronological coherence, the prompt **cumulatively injects previously analyzed pairs as historical context**.

**LLM JSON Defense:** To defend against LLM hallucinatory drift (e.g., malformed JSON or added commentary), both Step 1 and Step 3 utilize strict Pydantic schemas, Ollama's `format="json"`, and a multi-attempt retry loop with escalating temperatures. If all retries fail, the system gracefully degrades to a deterministic, rule-based fallback classification.

**Evaluation Prompt:**
```text
Task: "{task_label}"
Predicted Positive Emotions: {positive_emotions}
Predicted Negative Emotions: {negative_emotions}

Previous Reaction History (for context):
{accumulated_history} 
*(e.g., "1. Neutral -> Shock (Classified: Anticipation/Neutral)")*

Current Transition being Evaluated:
During the climax of the task, the bystander's emotion transitioned explicitly from {emotion_start} to {emotion_end}. 

Considering the history, does this new transition indicate a positive or negative task execution?

Respond in EXACTLY this JSON format:
{
  "classified_direction": "positive" | "negative" | "neutral",
  "reasoning": "Briefly map this specific transition to the task outcome context."
}
```

### Step 4: Temporal Slices & Overarching Task Scalar
For every evaluated pair, we extract a slice scalar: `slice_success_scalar = magnitude * direction_sign` (where positive=1, negative=-1, neutral=0).

#### Final Video-Level Aggregate Score (Per Task)
Instead of just leaving researchers with raw temporal slices, we calculate a final overarching score representing the outcome of that specific task.

We use a **Late-Stage Weighted Average**:
Emotions resolve over time. A "shock" reflex at the beginning of a window is less indicative of the final success than the "joy" at the end of the window. Therefore, we weight the `slice_success_scalar` by both its **duration** and its **chronological position** (how late it occurs relative to the task climax).

$$ \text{Task Success Score} = \frac{\sum_{i=1}^{n} (S_i \times D_i \times W_i)}{\sum_{i=1}^{n} (D_i \times W_i)} $$

Where:
- $S_i$ = The `slice_success_scalar` of slice $i$
- $D_i$ = Duration of slice $i$ in seconds
- $W_i$ = Chronological Weight in seconds: $W_i = \max(0.1,\; t_{\text{start}_i} - t_{\text{climax}})$. The `max(0.1, ...)` floor prevents the first slice from being zeroed out entirely. Later slices mathematically overpower early reflexes.

*(Note: If there are multiple bystanders, this final score is averaged across all bystander scores for that task, weighted by their 03a attention span.)*

---

## 📤 Output Schema and Integration
The layer outputs a structured array mapping the bystander's emotional journey *per task* within the video.

**Example Output Data (`03b_reasonable_emotion_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03b_reasonable_emotion",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "task_label": "Juggling apples",
      "task_reaction_window_sec": [6.2, 8.2],
      "per_person": [
        {
          "person_id": 0,
          "temporal_slices": [
            {
              "slice_id": 1,
              "window_sec": [6.2, 6.7],
              "transition_pair": ["neutral", "surprise"],
              "terminal_magnitude": 0.85,
              "classified_direction": "neutral",
              "slice_success_scalar": 0.0
            },
            {
              "slice_id": 2,
              "window_sec": [6.7, 8.2],
              "transition_pair": ["surprise", "joy"],
              "terminal_magnitude": 0.92,
              "classified_direction": "positive",
              "slice_success_scalar": 0.92
            }
          ],
          "late_stage_weighted_success_score": 0.81
        }
      ],
      "task_aggregate_score": 0.81
    }
  ]
}
```

## Verification & Validation Check
To ensure the LLM reasoning is chronologically sound and empirically reliable:
- **Singular Video Test**: Run the emotion layer for a specific video ID. Dump the exact prompt string sent to Gemma 4 and its exact JSON return to the console. Manually review the `classified_direction` logical mapping against the input `transition_pair` to verify the prompt is unbroken.
- **Batch Test**: Run the step on an entire `filtered_manifest.json` batch. Parse the final output and check the total distribution of `task_aggregate_score` values. If >95% of the values are strictly exactly positive or exactly negative, review the Gemma 4 temperature settings as the model may have collapsed into a predictable output path. Performance profiling should verify that processing scales continuously on the **Mac Studio (M4 Max, 64 GB unified memory)**.

## 🚀 Implementation Accomplishments (April 2026)

The initial implementation of the Reasonable Emotion Layer is complete:

- **Pipeline Created**: Built `ReasonableEmotionPipeline` in `src/layer_03b_reasonable_emotion/pipeline.py` which extracts `task_reaction_window_sec` bounds, dynamically samples bounding box crops at ~3 FPS, and chunks them into consecutive pairwise transitions.
- **Late-Stage Weighted Average**: Fully implemented the mathematical calculation from the spec, weighting later emotional segments heavier than early reflexes using the chronological weight `W_i = max(0.1, t_start_i - t_climax)`.
- **Attention-Weighted Aggregation**: Multi-bystander `task_aggregate_score` is computed as an attention-weighted mean using 03a scores, matching the doc specification.
- **Automated Testing Suite**: Implemented robust Pytest tests with mocked manifests, verifying schema conformance, mathematical correctness, and emotion classification semantics.

## 🧪 Resolved Issues & Implementation Refinements

1. **Transition Evaluation Concurrency (Resolved - May 23)**:
   - **Problem**: The emotion layer sequentially calls the local Gemma 4 LLM to evaluate every emotional transition for each bystander. This created a severe processing bottleneck, adding 3-5 seconds of latency per transition sequentially.
   - **Solution**: Implemented sequential evaluation with cross-bystander parallelization in `_process_task` in [pipeline.py](file:///Users/louisye/Desktop/Louis/social_robotics/src/layer_03b_reasonable_emotion/pipeline.py) using a `ThreadPoolExecutor`. Emotion frames are still sampled sequentially to avoid thread-safety violations on `cv2.VideoCapture`. Once sampled, the chronological history evaluations for each bystander's emotional journey are submitted as concurrent tasks. This maintains sequential context within each bystander's chain while overlapping the I/O-heavy local LLM calls across different bystanders, tasks, and videos.

2. **Reaction Window Anchored to Bystander Presence (Resolved - June 04)**:
   - **Problem**: `_process_task` sampled emotion strictly inside `task_reaction_window_sec` — anchored to the optical-flow peak (the *wearer's* kinetic climax), which in egocentric Ego4D is usually not when a bystander is on camera. In the June 4 10-clip smell test, **8 of 10 clips scored nothing**: the nearest bystander detection was 8–110 s from the reaction window, so every sample was dropped by the 2 s match tolerance.
   - **Solution**: Added per-bystander window anchoring in the new `_collect_emotion_timeseries` (`src/layer_03b_reasonable_emotion/pipeline.py`). For each bystander, if any of its detections fall within `MATCH_TOL` of the optical-flow window the original window is kept; otherwise the sampling window is re-anchored to the bystander's detection timestamp **nearest the climax** (preserving the velocity-derived window width). Re-validated on the same 10 clips: **scored clips rose from 2/10 to 8/10**, sampling now occurs where bystanders are actually present.

3. **Face-Presence Gate Before Emotion Inference (Resolved - June 04)**:
   - **Problem**: `_sample_emotions` cropped the Node-02 **full-body** bystander box and passed it to HSEmotion (a face model). On the v1 run the crops were a hiker's backpack and a distant body; HSEmotion returned near-uniform magnitudes (~0.18, vs the 1/8 = 0.125 baseline) and labels flipped neutral→fear→joy→surprise→disgust within 2 s — pure noise, with no confidence gate.
   - **Solution**: Added a MediaPipe BlazeFace gate (`models/mediapipe/blaze_face_short_range.tflite`, reusing 03a's model) in `_extract_face`: each bystander crop is face-detected and cropped to the largest face before HSEmotion; crops with no detected face emit no sample. The gate is active only when a real emotion model is loaded (the mock/test path is unaffected) and is env-toggleable (`SR_03B_FACE_GATE`, `SR_03B_MIN_FACE_CONF`). **Caveat surfaced by re-validation** (see Unresolved Issue 1): on this Ego4D sample the bystander crops are so distant/motion-blurred that BlazeFace still false-positives on non-face regions and HSEmotion stays near-uniform (~0.18), so the gate alone does not yet make the emotion signal trustworthy — a residual data-quality limitation is filed below.

4. **ONNX Emotion Backend + No Silent Mock (Resolved - June 04)**:
   - **Problem**: `from hsemotion.facial_emotions import HSEmotionRecognizer` failed to load under this environment for three compounding reasons — torch ≥ 2.6 `weights_only=True`, a CUDA-pickled checkpoint needing `map_location`, and a timm-version pickle mismatch (`DepthwiseSeparableConv` / `timm.layers`). On failure `emotion_detector` became `None` and `_sample_emotions` **silently** substituted seeded-RNG **mock** emotions — producing fake-but-real-looking scores with no warning.
   - **Solution**: The emotion backend now **prefers the ONNX build** (`from hsemotion_onnx.facial_emotions import HSEmotionRecognizer`), falling back to the torch build, then to mock. `emotion_source` (`hsemotion_onnx` | `hsemotion_torch` | `mock`) is recorded on the pipeline and emitted on every output record, and a **loud warning** is printed at init if the model fails to load (Option C, done regardless). Validated: the smell-test re-run reports `emotion_source = hsemotion_onnx` with real weights. Adds `hsemotion-onnx` + `onnxruntime` (see `docs/ml_dependencies.md`).

5. **03b Self-Populates Climax Metadata (Resolved - June 04)**:
   - **Problem**: `filtered_manifest.json` ships with `task_temporal_metadata = {}` (Node 02 defers it, 02 Resolved Issue #8) and `_process_task` returns `None` without `task_reaction_window_sec`, but 03b never called `populate_climax_for_manifest` — so run standalone on the production manifest it scored **0 clips**.
   - **Solution**: `run()` now calls `shared.climax_extraction.populate_climax_for_manifest(self.input_manifest_path, skip_vlm=True)` before reading the manifest (idempotent — a no-op once every task has metadata), fulfilling the Layer-03 contract from 02 Resolved Issue #8. Defensive: a failure logs and proceeds with existing metadata so a missing optional dep cannot crash the run.

6. **Single-Pass, Face-Batched Emotion Sampling (Resolved - June 04)**:
   - **Problem** (was Unresolved Issue 1): `_process_task` looped over each bystander and called `_sample_emotions` separately, re-opening/decoding the video from frame 0 once per bystander and using a per-sample random `cap.set(POS_FRAMES)` seek — both wasteful and, on H.264, seek-fragile.
   - **Solution**: Replaced the per-bystander loop with `_collect_emotion_timeseries`, which builds the union of all bystanders' per-window sample timestamps and walks the video in a **single sequential `grab()`/`retrieve()` pass** (no random seeks — matching the 03a Resolved Issue #5 lesson that `cap.set` is keyframe-approximate on H.264). At each timestamp the active bystanders' face crops are collected and run through HSEmotion as a **batch** via `predict_multi_emotions`. This integrates with the face gate (Resolved Issue #3) and window anchoring (Resolved Issue #2) in one pass. The tested aggregation math (`_evaluate_bystander_transitions`, late-stage weighting) is unchanged; the two math unit tests were updated to monkeypatch the new `_collect_emotion_timeseries` and the suite stays green (12/12).

7. **Emotion-Confidence Gate (Resolved - June 05)**:
   - **Problem** (was Unresolved Issue 1): Even with window anchoring (Resolved Issue #2) + the face gate (Resolved Issue #3), HSEmotion magnitudes on the Ego4D sample stayed near the 1/8 = 0.125 uniform baseline (median **0.18**) because bystander crops are distant/blurry and BlazeFace false-positives on non-face regions. 03b therefore emitted near-zero **noise** scores, with the magnitude consumed as `terminal_magnitude` under no confidence floor (a 0.18 "fear" weighted like a 0.9 "fear").
   - **Solution**: Added `MIN_EMOTION_CONF` (default **0.4**, env `SR_03B_MIN_EMOTION_CONF`) in `_collect_emotion_timeseries`: emotion samples whose top softmax probability is below the threshold are dropped, so a task with no confident sample scores **nothing (honest)** instead of feeding noise downstream. Validated on the June 5 50-clip run: scored clips fell to **2/50**, and both are high-face-quality close-frontal-face clips (`137d8616`: 278 px/0.90 conf; `43bd06f3`: 140 px/0.97 conf) — i.e. the gate suppresses the noise while retaining the genuine signal. This makes explicit that 03b is intrinsically **low-yield on distant-bystander egocentric footage**; the cost of that is addressed by the face-quality pre-filter (Resolved Issue #8 below).

8. **Bystander-Face-Quality Pre-Filter — Productionized & Wired (Resolved - June 06)**:
   - **Problem** (was Unresolved Issue 1): 03b is intrinsically **low-yield on egocentric footage** (Resolved Issue #7: only 2/50 clips produce a confident emotion score, because most bystander crops are too distant/blurry for HSEmotion to resolve), while the climax optical-flow pass (`shared.climax_extraction`) is the **dominant per-clip cost** that every Layer 03 pays first (~0.57× realtime; ~14 days for the 1,000-clip / 600 h reservoir before the 02 Resolved Issue #18 speedup). Paying climax on the ~84–96 % of clips that will score nothing is wasted work. A cheap pre-pass that decodes only the sparse bystander-detection frames already recorded in the manifest existed only as a throwaway prototype (`scratch/face_quality_prefilter.py`) — validated on the June 5 50-clip run (threshold **face ≥ 120 px, conf ≥ 0.8, ≥ 3 face-frames** keeps both clips that scored with **0 false negatives** while skipping 84 % → **6.2× climax saving**) but **not productionized or wired into the run flow**, so every 03 run still paid full climax on dead clips.
   - **Solution**: Productionized the prototype as a shared module `src/shared/face_quality_prefilter.py` and wired it into the Layer-03 cost path (Option A — standalone pre-pass that annotates + filters the manifest, reusable by every face-based 03 layer):
     - **`populate_face_quality_for_manifest(manifest_path)`** annotates each entry with a `bystander_face_quality` field (`best_face_px`, `best_face_conf`, `n_face_frames`, `n_checked`) computed by running BlazeFace (`models/mediapipe/blaze_face_short_range.tflite`) over only the sparse bystander-detection frames (≤ `MAX_FRAMES_PER_CLIP = 24` widely-spaced seeks; **~0.9 s/clip → ~15 min for 1,000**). It is **idempotent** (skips entries that already carry the field unless `force=True`), writes the manifest back at a save-point cadence so a crash mid-pass loses no progress, and is **defensive**: if mediapipe is unavailable it logs and no-ops (returns 0) so a missing optional dep cannot crash the run.
     - **`passes_face_quality(entry, …)`** is the gate predicate, with env-overridable thresholds (`SR_FACE_QUALITY_MIN_PX = 120`, `SR_FACE_QUALITY_MIN_CONF = 0.8`, `SR_FACE_QUALITY_MIN_FRAMES = 3`) and a master toggle (`SR_FACE_QUALITY_GATE = 1`). **Fail-open semantics** keep the gate honest: a clip that was never scored (no field) is processed, not silently dropped; only a present-but-failing record (checked → no resolvable face, including the all-zero "no bystanders" record) is skipped.
     - **`populate_climax_for_manifest()` gained an `entry_filter` predicate** (`src/shared/climax_extraction.py`): entries for which it returns False are dropped from `todo` **in the main process before Pool dispatch**, so gated clips are never opened and there is no Pool-pickling concern. 03b's `run()` passes `entry_filter=passes_face_quality`, so optical flow is **never paid** on a clip the gate will skip.
     - **03b `run()` now** (1) calls `populate_face_quality_for_manifest` first, (2) gates climax via `entry_filter`, and (3) skips gated clips in the processing loop with a logged count — and falls the gate open on any pre-filter exception so it can never silently discard work.
     - *Seek nuance (locked in a code comment)*: unlike climax/03a — where `cap.set(POS_FRAMES)` is the keyframe-approximate anti-pattern fixed by sequential `grab()` (02 Resolved Issue #18, 03a Resolved Issue #5) — the pre-filter **correctly uses random seeks**, because its samples are a handful of widely-spaced timestamps across the whole clip, so seeking is far cheaper than decoding every intervening frame and exact frame accuracy is irrelevant to a "is a face resolvable here?" check.
     - **Validation**: the productionized `compute_face_quality_for_entry` is **bit-identical** to the prototype on spot-checked clips incl. both scorers (`43bd06f3`: 140 px/0.968/19; `137d8616`: 278 px/0.897/8); the productionized gate **reproduces the report exactly** — keeps **8/50**, skips **84 % → 6.2× climax saving**, with **both scorers kept (0 false negatives)**. Added `tests/test_face_quality_prefilter.py` (10 tests: gate threshold/fail-open/toggle/custom-threshold + the climax `entry_filter` skip/no-op); full 03b + pre-filter suites green (**22/22**). Combined with the 02 Resolved Issue #18 climax speedup, the full-corpus 03 climax drops from **~14 days → roughly hours** with no change to results.

## ⚠️ Unresolved Issues & Suggestions

None at this time. All documented issues for this layer have been resolved (see the section above).

