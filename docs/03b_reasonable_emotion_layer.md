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

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Bystander faces in egocentric footage are too low-quality for reliable emotion (face gate alone insufficient)
**Status**: ⚠️ Confirmed Unresolved — Surfaced by the June 4 post-fix re-validation (`e2e_reports/2026_06_04_layer03b/`, v2 run). With window anchoring (Resolved Issue #2) and the BlazeFace face gate (Resolved Issue #3) in place, **8/10 clips now score** and emotions come from a real ONNX model — yet HSEmotion magnitudes remain near the 1/8 = 0.125 uniform baseline (median **0.18**, max 0.38), so task scores stay near zero (−0.11 to +0.09). Inspection of the post-gate crops shows why: Node-02 bystander boxes are full-body and distant, so at the sampled timestamps the face region is tiny / motion-blurred, **BlazeFace false-positives on non-face regions** (e.g. a dark blurred blob was scored "Neutral" 0.176 with all 8 classes within 0.06–0.18), and HSEmotion is correctly uncertain. The emotion signal is therefore still largely noise — a data-quality limitation exposed now that the plumbing works, not a code bug.

**Option A (recommended)**: **Confidence-gate the emotion magnitude** — discard emotion samples whose top softmax probability is below a threshold (~0.3–0.4) so near-uniform guesses (and BlazeFace false positives) produce no sample; a task with no confident sample scores nothing (honest) rather than emitting noise.
  - *Pros*: Cheap (no new model); directly removes the ~0.18 noise; complements the face gate; env-tunable.
  - *Cons*: On this Ego4D sample it will likely drop most/all clips to "no score" — correct, but means 03b yields little on distant-bystander egocentric footage.

**Option B**: **Require a minimum face size** (e.g. detected face box ≥ 60–80 px) before scoring — reject tiny/distant faces where neither BlazeFace nor HSEmotion is reliable.
  - *Pros*: Targets the root cause (resolution); reduces the BlazeFace false-positive rate.
  - *Cons*: Threshold needs tuning against the bystander-distance distribution; still yields little on far-away bystanders.

**Option C**: **Accept that 03b is low-yield on egocentric data** — keep the gates, treat 03b as firing only on the rare close, frontal-face bystander, and use 03a's face-quality / `attended_fraction` signals to pre-select clips worth running 03b on.
  - *Pros*: No further code; honest about the modality limit.
  - *Cons*: Most filtered clips produce no emotion signal; 03b's contribution to the dataset is sparse.

Your selection: _____

