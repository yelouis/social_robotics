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
Using the extracted Contextual Task (e.g., "Juggling apples"), prompt a locally running LLM (**Gemma 4** via Ollama) to generate baseline emotional expectations. This lightweight local LLM inference is well-suited for the **24GB RAM Mac mini M4 Pro**, allowing concurrent tracking arrays without swapping.

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
1. Run a SOTA emotion model (e.g., Py-Feat) on the bounding box faces at **3-5 FPS** across the reaction window.
2. Form an Emotion Timeseries (e.g., `Neutral (0.5s) -> Shock (0.5s) -> Joy (2.0s)`).
3. **Chunking into Pairs**: Break the sequence into consecutive Pairwise Transitions. Emotion logic is evaluated strictly one transition at a time to anchor LLM reasoning.
   - Transition 1: `Neutral -> Shock`
   - Transition 2: `Shock -> Joy`

### Step 3: Accumulated History Evaluation (Gemma 4)
Each emotional transition pair is fed sequentially to Gemma 4. However, to maintain chronological coherence, the prompt **cumulatively injects previously analyzed pairs as historical context**.

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
- **Batch Test**: Run the step on an entire `filtered_manifest.json` batch. Parse the final output and check the total distribution of `task_aggregate_score` values. If >95% of the values are strictly exactly positive or exactly negative, review the Gemma 4 temperature settings as the model may have collapsed into a predictable output path. Performance profiling should verify that processing scales continuously on the **Mac mini M4 Pro (24GB RAM)**.

## 🚀 Implementation Accomplishments (April 2026)

The initial implementation of the Reasonable Emotion Layer is complete:

- **Pipeline Created**: Built `ReasonableEmotionPipeline` in `src/layer_03b_reasonable_emotion/pipeline.py` which extracts `task_reaction_window_sec` bounds, dynamically samples bounding box crops at ~3 FPS, and chunks them into consecutive pairwise transitions.
- **Late-Stage Weighted Average**: Fully implemented the mathematical calculation from the spec, weighting later emotional segments heavier than early reflexes using the chronological weight `W_i = max(0.1, t_start_i - t_climax)`.
- **Attention-Weighted Aggregation**: Multi-bystander `task_aggregate_score` is computed as an attention-weighted mean using 03a scores, matching the doc specification.
- **Automated Testing Suite**: Implemented robust Pytest tests with mocked manifests, verifying schema conformance, mathematical correctness, and emotion classification semantics.

## 🧪 Resolved Issues & Implementation Refinements

1. **Attention Weighting Applied at Wrong Level (Resolved - April 26)**:
   - **Problem**: The original implementation multiplied each individual bystander's `late_stage_weighted_success_score` by their 03a attention score (`person_score *= attention_score`). This corrupted the `[-1, +1]` semantic range of the score — a fully positive result (`+1.0`) from a bystander with `0.8` attention would become `0.8`, indistinguishable from a genuinely mixed reaction.
   - **Solution**: Removed the per-person multiplication. Attention scores are now used exclusively to weight the **multi-bystander aggregate**, matching the doc spec: *"this final score is averaged across all bystander scores for that task, weighted by their 03a attention span."* Individual `late_stage_weighted_success_score` values remain in the pure `[-1, +1]` range.

2. **Mock Emotions Overwriting PyFeat Results (Resolved - April 26)**:
   - **Problem**: The `_sample_emotions` method had a fallback mock emotion generator (random distribution based on temporal progress) that ran **unconditionally** after the PyFeat `try/except` block. If PyFeat was successfully loaded and returned real results, the mock code would immediately overwrite them with random values.
   - **Solution**: Added a `detected` flag. The mock generator now only runs inside an `if not detected:` guard, ensuring that when PyFeat is wired in, its results are preserved.

3. **`surprise` Misclassified as Positive (Resolved - April 26)**:
   - **Problem**: The hardcoded fallback expectations placed `surprise` in the `positive_emotions` list. However, the doc's own schema example explicitly shows `["neutral", "surprise"]` classified as `"neutral"` with `slice_success_scalar: 0.0`. Surprise is psychologically ambivalent — it carries no intrinsic positive/negative valence.
   - **Solution**: Moved `surprise` from `positive_emotions` to `neutral_baseline`. Added `"amusement"` and `"relief"` as the replacement positive entries. Added a dedicated `test_surprise_classified_as_neutral` test to lock this behavior.

4. **Unused Imports (Resolved - April 26)**:
   - **Problem**: Four imports (`tempfile`, `math`, `sys`, `torch`) were present at the top of `pipeline.py` but never referenced in the module body. These were carried over from the 03a attention layer template.
   - **Solution**: Removed all four unused imports.

5. **Math Test Not Asserting Computed Values (Resolved - April 26)**:
   - **Problem**: `test_late_stage_weighted_math` had an elaborate comment computing the expected value `0.875` but the actual assertions only checked `is not None` and `len(slices) == 3`. The mathematical correctness of the weighted average was never verified.
   - **Solution**: Rewrote the test with correct hand-calculated expectations (accounting for 3 slices, not 2). The test now asserts exact `slice_success_scalar` values for each slice, verifies `classified_direction` labels, and checks the final `late_stage_weighted_success_score` against the analytically computed value `round(0.70 / 1.30, 2) = 0.54`.

6. **Py-Feat CPU-Only on macOS (Resolved - April 27)**:
   - **Problem**: Py-Feat currently lacks native MPS (Metal Performance Shaders) acceleration. On the Mac mini M4 Pro, all inference runs on CPU. The original pipeline also had only a `pass` stub inside the `feat_detector` block — PyFeat was never actually called.
   - **Solution**: Wired the full `Detector.detect_image()` call into `_sample_emotions`. Face crops are written to a temporary PNG via `cv2.imwrite`, passed to PyFeat, and the dominant emotion column (highest probability) is extracted as the label with its probability as the magnitude. A `detected` flag ensures the mock fallback only runs when PyFeat is absent or fails. The `Detector` is instantiated once at `__init__` time with `device="cpu"` and reused across all videos, avoiding repeated model-loading overhead.

7. **LLM JSON Output Fragility — Steps 1 & 3 (Resolved - April 27)**:
   - **Problem**: The multi-step pairwise evaluation structure relies on Gemma 4 generating valid JSON responses. LLMs are prone to hallucinatory drift (adding commentary outside the JSON block, omitting required keys, or producing malformed brackets), which would cause `json.loads()` to crash.
   - **Solution**: Implemented three layers of defense: Pydantic schemas for strict field validation, Ollama `format="json"` to force valid JSON emission, and a retry mechanism with escalating temperature. Graceful degradation to rule-based classification ensures pipeline stability.

8. **Ollama Model Tag Mismatch (Resolved - April 30)**:
   - **Problem**: The pipeline was hardcoded to `gemma4:e4b`, but the local Ollama instance only contained the `gemma4:latest` tag, causing a "model not found" error during E2E testing.
   - **Solution**: Updated `OLLAMA_MODEL` to `gemma4:latest` to match the environment.

9. **E2E Validation on Ego4D Test Set (Resolved - April 30)**:
   - **Problem**: The module lacked end-to-end verification against real-world egocentric data to confirm logical coherence and schema compliance.
   - **Solution**: Conducted a full E2E test using clip `001e3e4e-2743-47fc-8564-d5efd11f9e90.mp4`. The system correctly ingested 03a attention scores (0.23 avg) and processed a "Loading laundry" task. The LLM correctly classified `neutral -> fear` and `neutral -> anger` transitions as `negative`. The final `late_stage_weighted_success_score` (-0.39) logically reflected the mixed-to-negative emotional journey. Output schema conformance was 100% verified.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Transition Evaluation Latency
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py` (lines 373-403): the `_process_task` method iterates over every consecutive emotion pair in the timeseries and calls `_llm_evaluate_transition()` for each pair, even when the emotion label does not change between consecutive samples. At ~3 FPS sampling within a 4-second reaction window, this produces ~12 samples → ~11 transition pairs → 11 LLM calls per bystander per task. With Gemma 4 running locally on the M4 Pro, each call takes ~10-15 seconds, resulting in **~2-3 minutes per reaction window per bystander**. For a batch of 100 videos with 2 tasks and 1.5 bystanders each, this totals ~7.5 hours of pure LLM inference.

**Option A (recommended)**: **State-Change Filter** — Before calling `_llm_evaluate_transition()`, check if `start_emotion['emotion'] == end_emotion['emotion']`. If the emotion label hasn't changed, skip the LLM call entirely and extend the previous slice's `window_sec` end time instead. Only trigger LLM evaluation when a genuine emotion transition occurs. This would reduce the 11 calls per window to typically 2-4 calls (matching the number of actual emotion state changes).
  - *Pros*: Directly addresses the root cause; 3-5x speedup on typical windows; no change to output semantics (same-emotion "transitions" carry zero information); preserves the accumulated history context for genuine transitions.
  - *Cons*: If PyFeat produces jittery labels (e.g., `joy → neutral → joy` due to noise), legitimate micro-transitions could be merged. A minimum-duration threshold (e.g., emotion must persist for ≥0.5s to count as a state change) would mitigate this.

**Option B**: **Batch LLM Evaluation** — Instead of evaluating transitions one-by-one with accumulated history, construct a single prompt containing ALL transitions for a given reaction window and send it to Gemma 4 in one call. The model would return a JSON array of per-transition classifications.
  - *Pros*: Reduces LLM calls from ~11 to 1 per bystander per task; maintains full chronological context in a single prompt.
  - *Cons*: Significantly longer prompt (may hit Gemma 4's context window limits at 8K tokens for complex windows); harder to validate individual transition classifications; the model may "drift" on later transitions in a long prompt; requires restructuring the Pydantic validation to handle arrays.

**Option C**: **Hybrid: State-Change Filter + Cached History** — Combine Option A's state-change filter with a local rule-based pre-classifier. Only escalate to the LLM when the rule-based classifier returns `"uncertain"` (e.g., the terminal emotion doesn't appear in either the positive or negative expectation lists).
  - *Pros*: Maximum speedup (LLM only called for genuinely ambiguous transitions); rule-based fallback handles obvious cases (e.g., `neutral → joy` is always positive) instantly.
  - *Cons*: More complex implementation; the rule-based pre-classifier may disagree with the LLM on edge cases, creating inconsistency; requires defining the "uncertain" boundary.

Your selection: Proceed with Option A.
