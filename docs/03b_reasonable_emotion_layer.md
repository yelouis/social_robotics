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

10. **Transition Evaluation Latency (Resolved - May 05)**:
    - **Problem**: The `_process_task` method in `pipeline.py` iterated over every consecutive emotion pair in the timeseries and called `_llm_evaluate_transition()` for each pair, even when the emotion label did not change between consecutive samples. This caused ~11 LLM calls per bystander per task, resulting in ~2-3 minutes latency per reaction window per bystander.
    - **Solution**: Implemented a State-Change Filter. Before calling `_llm_evaluate_transition()`, the pipeline now checks if `start_emotion['emotion'] == end_emotion['emotion']`. If the emotion label hasn't changed, it skips the LLM call entirely and extends the previous slice's `window_sec` end time instead. Genuine emotion transitions are still evaluated by the LLM, preserving chronological context while providing a 3-5x speedup.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Unused `re` Import in `pipeline.py`
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03b_reasonable_emotion/pipeline.py` line 5: `import re` is present but the `re` module is never referenced anywhere in the module body. This is the same class of dead-import issue previously resolved as Issue #4 (April 26), reintroduced during a later edit (likely the LLM-JSON-defense work in Issue #7, since regex-based JSON salvage is a common pattern that may have been considered then dropped).

**Option A (recommended)**: **Drop the import outright** — Remove the `import re` line.
  - *Pros*: Zero behavioral risk; restores the post-April-26 cleanliness; static analyzers (`ruff`, `pyflakes`) stop flagging the file.
  - *Cons*: None.

**Option B**: **Use `re` to harden raw LLM JSON parsing** — Pre-strip Markdown code fences (` ```json ... ``` `) from `response.message.content` before `json.loads`, since `format="json"` does not always prevent fence wrapping on every Ollama backend.
  - *Pros*: Adds a fourth layer of defense to Issue #7's JSON-fragility fix; fences are a real failure mode for some Gemma builds.
  - *Cons*: Extends scope beyond a cleanup; needs its own test; if pydantic + `format="json"` is already covering the observed cases, this is gold-plating.

Your selection: _____

---

### Issue 2: `KNOWN_EMOTIONS` is Dead Code with Misleading Docstring
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03b_reasonable_emotion/pipeline.py` lines 28-35. The comment above the constant states: *"every LLM response is validated against this set. Anything outside it is silently normalized to 'neutral'."* Neither validation nor normalization is implemented anywhere in the file. `ExpectationSchema` (lines 37-48) only enforces field structure (list typing, key presence) — it does not constrain vocabulary. `_sample_emotions` consults `PYFEAT_EMOTION_COLUMNS` (line 56), not `KNOWN_EMOTIONS`. The set is referenced exactly once, in a generator comprehension at line 557, where it is used to lowercase-filter PyFeat columns — a use that is functionally redundant with `PYFEAT_EMOTION_COLUMNS` itself.

This is hazardous because future contributors reading the docstring will assume LLM-emitted emotions like `"jubilation"` or `"shock"` get normalized, when in reality they flow through into `transition_pair` and `_rule_based_classify` unchanged (and silently land in the `neutral` bucket via fallthrough, but for the wrong reason).

**Option A (recommended)**: **Implement the normalization the comment promises** — Add a `_normalize_emotion(label: str) -> str` helper that returns the input if it is in `KNOWN_EMOTIONS`, else `"neutral"`. Apply it inside `_sample_emotions` (after PyFeat or mock label assignment) and inside `_llm_evaluate_transition` (to the `emotion_start`/`emotion_end` arguments before they enter the prompt).
  - *Pros*: Aligns code with the existing comment, which was clearly the original intent; protects downstream rule-based fallback from out-of-vocabulary labels; keeps `KNOWN_EMOTIONS` as the single source of truth.
  - *Cons*: Requires a test asserting that an OOV label maps to `neutral`; minor risk of masking genuine PyFeat columns if they ever expand beyond `KNOWN_EMOTIONS`.

**Option B**: **Delete `KNOWN_EMOTIONS` and the misleading comment** — Drop lines 28-35 and rely on `PYFEAT_EMOTION_COLUMNS` plus pydantic structural validation alone.
  - *Pros*: Smallest diff; no new behavior to test.
  - *Cons*: Leaves the LLM path with no vocabulary guard at all; an LLM emitting `"shock"` (not in any expectation list) silently classifies as `neutral` via fallthrough.

Your selection: _____

---

### Issue 3: Per-Sample Temp-PNG Round-Trip in PyFeat Path
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03b_reasonable_emotion/pipeline.py` lines 542-563. For each emotion sample (~3 FPS × reaction window × bystander count), the code writes the BGR face crop to a `tempfile.NamedTemporaryFile(suffix=".png")` via `cv2.imwrite`, passes the path string to `self.feat_detector.detect_image(tmp_path)`, then `unlink`s the file. On the Mac mini M4 Pro target, PyFeat's CPU inference (per Issue #6) is already the dominant cost, but each sample now also pays for: (a) PNG encode in OpenCV, (b) two filesystem syscalls per sample (write + unlink), and (c) PIL/imageio decode inside PyFeat. For a manifest of N tasks × M bystanders, the cost scales linearly and is pure overhead.

PyFeat's `Detector` exposes lower-level entry points (`detect_emotions(frame_array, faces)` and `_predict_emotions(face_aligned)`) that accept numpy arrays directly, bypassing the disk round-trip entirely.

**Option A (recommended)**: **Switch to in-memory PyFeat API** — Replace `cv2.imwrite` + `detect_image(path)` with a direct call passing the BGR (or RGB-converted) ndarray to `detect_emotions` / `_predict_emotions`. Drop the `tempfile` import block entirely.
  - *Pros*: Eliminates per-sample disk IO and the unlink-on-error cleanup mess (lines 564-570); reduces wall-clock time per video meaningfully on long manifests; removes a TOCTOU-style failure surface.
  - *Cons*: PyFeat's lower-level API is less stable across versions than `detect_image`; would need a version-pin assertion or a fallback path; requires verifying that the face has already been detected (the bbox is supplied by Node 02, so the detector currently runs face-detection redundantly inside `detect_image`).

**Option B**: **Keep `detect_image` but reuse a single temp path** — Allocate one persistent temp file at `__init__` time and overwrite it per sample. Skip `unlink` until pipeline shutdown.
  - *Pros*: Trivial diff; preserves the existing PyFeat call shape.
  - *Cons*: Still pays for PNG encode + decode every sample; only saves the unlink syscall; doesn't address the redundant face-detection-inside-`detect_image` issue.

**Option C**: **Batch crops into a single `detect_image` call per task** — Collect every sample's crop into a list and submit them in one PyFeat batch at the end of `_sample_emotions`.
  - *Pros*: Amortizes PyFeat model warm-up / batch tensor overhead; potentially largest speedup.
  - *Cons*: Most invasive refactor; loses the early-exit on per-sample failure; PyFeat's batch API expects either a directory of images or a video path, neither of which fits the current crop-array shape cleanly.

Your selection: _____

---

### Issue 4: `VideoCapture` Not Released When `cap.isOpened()` Returns False
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03b_reasonable_emotion/pipeline.py` lines 318-321. After `cap = cv2.VideoCapture(str(video_path))`, the `if not cap.isOpened()` branch returns `None` without calling `cap.release()`. The very next branch at lines 325-327 (the `fps == 0 or total_frames == 0` case) does correctly call `cap.release()` before returning, so the omission is an inconsistency, not a uniform pattern. While `cv2.VideoCapture` is generally tolerant of being garbage-collected un-released, on long batch runs over manifests with corrupted files this can leak file descriptors and (on some FFmpeg backends) hold a transient memory mapping until GC.

**Option A (recommended)**: **Add the `cap.release()` call** — One-line fix mirroring the sibling branch.
  - *Pros*: Trivial; makes the two early-exit branches symmetric; defends against backend-specific FD leaks.
  - *Cons*: None.

**Option B**: **Wrap the whole `process_video` body in a `try / finally: cap.release()`** — Guarantees release on any exit path, including the bystander/task early-returns at lines 308-316 (which currently never opened a cap, so don't need release, but a uniform pattern is more defensible).
  - *Pros*: Eliminates this entire class of bug; resilient to future early-return additions.
  - *Cons*: Slightly larger diff; the lines-308-316 exits would call `release()` on an undefined `cap` unless the variable is initialized to `None` first.

Your selection: _____

---

### Issue 5: Global `random.seed()` Mutation in `_sample_emotions`
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03b_reasonable_emotion/pipeline.py` line 507: `random.seed(int(sum(sum(b) for b in b_bboxes)))`. This is called on every invocation of `_sample_emotions` and mutates the **global** `random` module state. After fix #6 (April 27) wired in real PyFeat inference, the mock-emotion fallback (lines 572-583, the only consumer of `random` in this method) is now a degraded code path — but the `random.seed()` call still runs unconditionally **before** the PyFeat branch, so even successful PyFeat runs reseed the global RNG. Any other module in the same Python process that relies on `random` (e.g., test fixtures, augmentation libraries, ID generators) will inherit the bbox-derived seed.

**Option A (recommended)**: **Use a local `random.Random(seed)` instance** — Replace `random.seed(...)` + `random.uniform(...)` + `random.choice(...)` with `rng = random.Random(seed)` and `rng.uniform(...)` / `rng.choice(...)`. Confine the determinism to this method.
  - *Pros*: Eliminates global state leakage entirely; preserves the deterministic-mock property; trivial diff.
  - *Cons*: None.

**Option B**: **Move the seed call inside the `if not detected:` mock branch** — At least skip the global mutation when PyFeat is the actual emotion source.
  - *Pros*: Smallest diff; PyFeat-success path no longer pollutes global RNG.
  - *Cons*: Still leaks state every time PyFeat fails or is absent; doesn't fix the underlying anti-pattern.

**Option C**: **Delete the mock fallback entirely** — Now that PyFeat is wired in (Issue #6), make `feat_detector is None` a hard failure rather than a silent degradation to random emotions.
  - *Pros*: Removes a confusing legacy code path; surfaces missing-PyFeat misconfigurations loudly; eliminates the seed call as a side effect.
  - *Cons*: Breaks integration tests that rely on the mock for video-less determinism; Pytest fixtures in `tests/test_layer_03b.py` would need monkeypatching of `_sample_emotions` (which they already do for the math/surprise tests, but not for `test_schema_conformance`).

Your selection: _____
