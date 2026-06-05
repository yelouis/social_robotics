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

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Redundant Video Decoding for Multiple Bystanders
**Status**: ⚠️ Confirmed Unresolved — Verified in [pipeline.py](file:///Users/louisye/Desktop/Louis/social_robotics/src/layer_03b_reasonable_emotion/pipeline.py#L386-L480): the pipeline loops over each tracked bystander separately, resulting in the video file being opened and decoded from frame 0 multiple times (once per bystander) and using random seeks (`cap.set`) which adds I/O latency.

**Option A (recommended)**: **Single-Pass Video Decoding with Batched Model Inference** — Modify `_process_task` and `_sample_emotions` to decode the video file once, extract the cropped frames for all active bystanders at the sampled timestamp, and pass them as a batch to HSEmotion.
  - *Pros*: Avoids opening the video multiple times and performing random seeks; speeds up video I/O and frame decoding overhead by N-times; utilizes PyTorch batched tensor execution on MPS.
  - *Cons*: Complex codebase refactoring of frame sampling and cropping logic.

**Option B**: **In-Memory Frame Caching** — Cache decoded video frames or bystander crops in memory or temporary files so subsequent bystander loops read from cache.
  - *Pros*: Avoids re-decoding the video.
  - *Cons*: Consumes significant RAM/disk if frames are cached, increasing memory pressure on 24GB hosts.

Your selection: _____

---

### Issue 2: Reaction window decoupled from bystander presence → most clips score nothing
**Status**: ⚠️ Confirmed Unresolved — Verified in the June 4 10-clip smell test (`e2e_reports/2026_06_04_layer03b/`): **8 of 10 clips failed to score**. `_process_task` samples bystander emotion strictly inside `task_reaction_window_sec`, which `shared/climax_extraction.py` anchors to the **optical-flow peak — the wearer's kinetic climax**. In egocentric Ego4D that moment is usually *not* when a bystander is on camera: across the 8 failures the nearest bystander detection was **8–110 s away** from the reaction window (e.g. `6b3988dd`: pedestrian detected at t=87 s, reaction window at t=116 s on an empty street). With no bystander box within the `_sample_emotions` 2.0 s tolerance, every sample is skipped, `<2` samples remain, and the task returns `None`.

**Option A (recommended)**: **Anchor the reaction window to bystander presence, not just wearer kinetics** — intersect the optical-flow window with the intervals where bystanders are actually detected, or select the bystander-present span nearest the climax, before sampling emotion.
  - *Pros*: Directly fixes the dominant failure; emotion is only sampled where a bystander exists; keeps the climax as a tie-breaker.
  - *Cons*: A task whose bystander is never present near any kinetic event still yields nothing (arguably correct); needs a rule for choosing among multiple bystander spans.

**Option B**: **Use a per-bystander reaction window** computed from that bystander's own detection timestamps overlapping the task, decoupling 03b from the wearer-centric optical-flow climax entirely.
  - *Pros*: Maximizes recall of real reactions; each bystander scored over the time they are actually visible.
  - *Cons*: Loses the "reaction *to the task climax*" semantics the layer was designed around; larger change.

**Option C**: **Widen the window / match tolerance** to the nearest bystander detection when the window is empty.
  - *Pros*: Smallest change.
  - *Cons*: Samples emotion far from the actual task moment, weakening the causal "reaction to outcome" interpretation.

Your selection: _____

---

### Issue 3: Full-body / non-face crops fed to HSEmotion → near-uniform emotion noise
**Status**: ⚠️ Confirmed Unresolved — Verified in the June 4 smell test. `_sample_emotions` crops the Node-02 **bystander bounding box (a full-body YOLO-pose person box)** and passes it to HSEmotion, which expects a face. On the 2 clips that scored, the crops were a hiker's **backpack/back (no face)** and a **distant full-body** desk occupant; HSEmotion returned magnitudes of **0.16–0.25** (the 8-class uniform baseline is 0.125 — i.e. the model is guessing) and the labels flipped neutral→fear→joy→surprise→disgust within 2 s. The resulting `task_aggregate_score` (0.04 on both) is classifier noise, and the magnitude is consumed as `terminal_magnitude` with no confidence gate, so a 0.18 "fear" is weighted like a 0.9 "fear".

**Option A (recommended)**: **Add a face-detection + crop gate** (BlazeFace via MediaPipe, mirroring 03a Resolved Issue #2) — detect the face within the bystander box, crop to it before HSEmotion, and emit no emotion sample when no face is found.
  - *Pros*: Feeds HSEmotion an actual face; removes the back-of-person/full-body noise; reuses the 03a gate pattern + model.
  - *Cons*: One detector pass per sample; distant/occluded bystanders yield no emotion (arguably correct).

**Option B**: **Confidence-gate the emotion magnitude** — discard samples whose top softmax probability is below a threshold (e.g. < 0.4) as "no reliable emotion".
  - *Pros*: Cheap; filters the near-uniform guesses without a new model.
  - *Cons*: Does not fix *what* is cropped; a confidently-wrong emotion on a non-face crop still passes; threshold needs tuning.

Your selection: _____

---

### Issue 4: HSEmotion torch backend fails to load (torch ≥2.6 / CUDA-pickle / timm) → silent mock-emotion fallback
**Status**: ⚠️ Confirmed Unresolved — Verified June 4. `from hsemotion.facial_emotions import HSEmotionRecognizer` + construct fails in this environment for three compounding reasons: (1) torch ≥ 2.6 defaults `weights_only=True`, refusing the pickled `timm` EfficientNet checkpoint; (2) the checkpoint is **CUDA-pickled**, so `torch.load` needs `map_location` on this MPS/CPU host; (3) **timm version mismatch** — `DepthwiseSeparableConv has no attribute 'conv_s2d'` on timm 1.x, while timm 0.6.x lacks `timm.layers` the pickle imports. When it fails, `emotion_detector` is set to `None` and `_sample_emotions` **silently** uses a seeded-RNG **mock** emotion distribution (pipeline.py:600–611) — producing fake-but-real-looking scores with no prominent warning. The smell-test run only got real emotions by injecting the ONNX backend.

**Option A (recommended)**: **Switch 03b to the ONNX HSEmotion backend** (`hsemotion-onnx`, `from hsemotion_onnx.facial_emotions import HSEmotionRecognizer`) — same `predict_emotions` API, ONNX weights via onnxruntime, no torch/timm pickle. Validated working (`enet_b2_8`) in this run.
  - *Pros*: Sidesteps all three breakages; drop-in; CPU/ARM friendly.
  - *Cons*: Adds `onnxruntime`; loses native MPS for the emotion model (small model, negligible).

**Option B**: **Pin the compatible timm + patch the load** — pin timm to the checkpoint's version and wrap HSEmotion's `torch.load` with `weights_only=False, map_location=...`.
  - *Pros*: Keeps the torch backend / MPS.
  - *Cons*: Brittle version pinning that can conflict with other layers; still pickle-fragile.

**Option C (do regardless)**: **Fail loudly instead of silently mocking** — if the emotion model cannot load, log a prominent warning and tag results `emotion_source: "mock"` (or refuse to run) so a run is never silently fake.
  - *Pros*: Prevents the silent-mock landmine; trivial.
  - *Cons*: None material; complements A or B.

Your selection: _____

---

### Issue 5: 03b never populates climax metadata → 0 results on the shipped manifest
**Status**: ⚠️ Confirmed Unresolved — Verified June 4. `filtered_manifest.json` ships with `task_temporal_metadata = {}` (Node 02 defers it, 02 Resolved Issue #8), and `_process_task` returns `None` when `task_reaction_window_sec` is missing. Per Issue #8 the *first Layer 03 to consume the manifest* must call `shared.climax_extraction.populate_climax_for_manifest`, but **03b never does** — so run standalone on the production manifest it scores **0 clips**. The smell-test runner had to call it first (cost: 988 s for 10 short clips).

**Option A (recommended)**: **03b calls `populate_climax_for_manifest(manifest_path, ...)` at the top of `run()`** (it is idempotent — a no-op once populated), fulfilling the documented Layer-03 contract.
  - *Pros*: Makes 03b self-sufficient; matches 02 Resolved Issue #8; idempotent and cheap on already-populated manifests.
  - *Cons*: The first Layer-03 run pays the optical-flow cost (already the documented design).

**Option B**: **A Layer-03 orchestrator populates climax once** before invoking any 03x layer.
  - *Pros*: Single chokepoint; shared across 03a–03g.
  - *Cons*: Requires an orchestrator that does not yet exist; 03b still broken if run directly.

Your selection: _____

