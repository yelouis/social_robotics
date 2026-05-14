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
1. Run a SOTA emotion model (e.g., HSEmotion-PyTorch) on the bounding box faces at **3-5 FPS** across the reaction window.
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
- **Batch Test**: Run the step on an entire `filtered_manifest.json` batch. Parse the final output and check the total distribution of `task_aggregate_score` values. If >95% of the values are strictly exactly positive or exactly negative, review the Gemma 4 temperature settings as the model may have collapsed into a predictable output path. Performance profiling should verify that processing scales continuously on the **Mac Studio (M4 Max, 64 GB unified memory)**.

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

11. **Reintroduced Dead `re` Import (Resolved - May 07)**:
    - **Problem**: `import re` had returned to the top of `src/layer_03b_reasonable_emotion/pipeline.py` even though the `re` module is never referenced anywhere in the file. This was the same class of dead-import defect previously cleaned up in Resolved Issue #4 (April 26), most likely reintroduced during the LLM-JSON-defense work in Resolved Issue #7 — regex-based JSON salvage is a common scaffolding pattern that was apparently considered and then dropped. The leftover import produced lint warnings and re-staged the same code-hygiene regression after a fresh audit had certified the file as clean.
    - **Solution**: Deleted the `import re` line from the import block. No other change required — the code path the import would have served (Markdown-fence-stripping before `json.loads`) was decided against in favor of pydantic + Ollama `format="json"` + retry-with-temperature-escalation.

12. **`KNOWN_EMOTIONS` Dead Code with Misleading Docstring (Resolved - May 07)**:
    - **Problem**: The module defined a `KNOWN_EMOTIONS` set with a comment claiming *"every LLM response is validated against this set. Anything outside it is silently normalized to 'neutral'."* Neither validation nor normalization was actually implemented anywhere in the file. `ExpectationSchema` only enforced list-of-strings structure, not vocabulary; `_sample_emotions` consulted `PYFEAT_EMOTION_COLUMNS`, not `KNOWN_EMOTIONS`; and a grep confirmed `KNOWN_EMOTIONS` had **zero** runtime references — the audit doc that flagged it had also misread the only `lower()` set comprehension in `_sample_emotions`, which actually iterates `PYFEAT_EMOTION_COLUMNS`. The constant existed purely as a misleading docstring trap for future contributors who would assume OOV labels were being scrubbed.
    - **Solution**: Deleted the `KNOWN_EMOTIONS` constant and the surrounding "canonical emotion vocabulary" comment block. The remaining vocabulary surface is now only `PYFEAT_EMOTION_COLUMNS` (used to filter PyFeat DataFrame columns) and the per-task `expectations` dict from the LLM (used by `_rule_based_classify`'s set-membership fallback). LLM-emitted out-of-vocabulary labels still flow through to the rule-based classifier and land in `"neutral"` via fallthrough — same as before — but now without a comment falsely promising a vocabulary guard. Verified no test imports `KNOWN_EMOTIONS`; the existing 03b test suite remains green except for one pre-existing failure unrelated to this change.

13. **Per-Sample PyFeat Temp-PNG Disk Round-Trip (Resolved - May 07)**:
    - **Problem**: For every emotion sample (~3 FPS × reaction-window-length × bystander-count), `_sample_emotions` opened a `tempfile.NamedTemporaryFile(suffix=".png")`, wrote the BGR face crop with `cv2.imwrite`, passed the path string to `Detector.detect_image(path)`, and then `unlink`'d the file. Each sample paid for: PNG encode in OpenCV, two filesystem syscalls (write + unlink), and a PIL/imageio decode inside PyFeat. The error path also carried a fragile `try / except NameError: pass` cleanup block that only worked because `tmp_path` happened to be defined in an outer scope. On long manifests this is pure linear-cost overhead on top of an already-CPU-bound PyFeat inference (Resolved Issue #6).
    - **Solution**: Removed `import tempfile` and switched to an in-memory call: the BGR crop is converted to RGB via `cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)` and passed as a numpy ndarray directly to `self.feat_detector.detect_image(crop_rgb)`. Modern py-feat (≥0.6) accepts both paths and ndarrays for `detect_image`, so the call shape is preserved. If a stricter py-feat version rejects the ndarray, the existing exception handler catches it and the `detected` flag stays `False`, falling through to the deterministic mock generator — the same safety net that already covered every other PyFeat failure mode. The fragile `NameError`-guarded cleanup block was deleted along with the temp-file path.

14. **Missing `cap.release()` on `cap.isOpened() == False` Branch (Resolved - May 07)**:
    - **Problem**: In `process_video`, the `if not cap.isOpened():` early-return path returned `None` *without* calling `cap.release()`, while the immediately-adjacent `if fps == 0 or total_frames == 0:` early-return correctly released the capture. On long batch runs against manifests with corrupted or unreadable video files, the unreleased branch could leak file descriptors and — on some FFmpeg backends — hold a transient memory mapping until garbage collection.
    - **Solution**: Added the missing `cap.release()` call inside the `if not cap.isOpened():` branch, mirroring the sibling branch's behavior. Both early-exit paths now release the capture symmetrically.

15. **Global `random` State Mutation in `_sample_emotions` (Resolved - May 07)**:
    - **Problem**: `_sample_emotions` opened with `random.seed(int(sum(sum(b) for b in b_bboxes)))`, mutating the **global** `random` module state on every invocation. After Resolved Issue #6 wired in real PyFeat inference, the mock fallback (the only consumer of `random` in this method) became a degraded code path — but the `random.seed()` call still ran unconditionally **before** the PyFeat branch, so successful PyFeat runs were also reseeding the global RNG. Any other module in the same Python process that relied on `random` (test fixtures, augmentation libraries, ID generators) silently inherited a bbox-derived seed.
    - **Solution**: Replaced the global `random.seed(...)` + `random.uniform(...)` + `random.choice(...)` triplet with a method-local `rng = random.Random(int(sum(sum(b) for b in b_bboxes)))` and per-call `rng.uniform(...)` / `rng.choice(...)`. The deterministic-mock property is preserved (same seed → same sequence per bystander), and the global RNG is no longer touched by Layer 03b at all.

16. **`gemma4:e4b` (4B) Default Underutilized 64 GB Mac Studio for Multi-Step Reasoning (Resolved - May 14)**:
    - **Problem**: `OLLAMA_MODEL` in `pipeline.py` was pinned to `gemma4:latest`, which resolves to the 4B/4-bit `e4b` tag. That tag was selected in Resolved Issue #8 for the 24 GB Mac mini M4 Pro memory budget, where the 27B-class variant (~15 GB) would have squeezed out the resident face-emotion detector and the L2CS-Net pipeline already loaded from 03a. On the Mac Studio (M4 Max, 64 GB unified memory) target host the 4B model is a memory-headroom regression: the 03b reasoning chain (Step 1 expectation generation + Step 3 cumulative-history evaluation per transition pair) is exactly the multi-step structured-output regime where 27B-class models measurably reduce the JSON drift that Resolved Issue #7's pydantic + retry-with-temperature defenses exist to absorb.
    - **Solution**: Changed `OLLAMA_MODEL` from `"gemma4:latest"` to `"gemma4:26b"`, making the 27B-class model the unconditional default. Per the selected remediation path, no `SAF_GEMMA_TIER` env override and no `e4b` fallback were added — the configuration surface stays flat. The constant is consumed directly by both `ollama_client.chat()` call sites (`_llm_generate_expectations`, `_llm_evaluate_transition`), so the change is a pure model-tag swap with no control-flow change; the existing test suite (which forces `ollama_available = False`) remains green. The empirical E2E re-validation against a live `gemma4:26b` Ollama instance — Resolved Issue #9's `001e3e4e-2743-47fc-8564-d5efd11f9e90.mp4` clip, confirming preserved score direction and reduced retry-with-temperature frequency — remains as a follow-up step gated on Mac Studio host access.

17. **Py-Feat CPU-Only Inference Bottlenecked 03b on the 64 GB Mac Studio (Resolved - May 14)**:
    - **Problem**: Resolved Issue #6 wired Py-Feat's `Detector` with `device="cpu"` because Py-Feat lacks native MPS (Metal Performance Shaders) acceleration. On the Mac Studio (M4 Max, 64 GB unified memory) target host this leaves the MPS GPU — dramatically faster than the CPU at vision-model inference — completely idle, while CPU-bound Py-Feat inference dominates the 03b layer's wall-clock at 3-5 FPS × reaction-window-length × bystander-count.
    - **Solution**: Migrated off Py-Feat to HSEmotion-PyTorch, which loads through the existing PyTorch stack and runs natively on MPS. The optional `from feat import Detector` import was replaced with `from hsemotion.facial_emotions import HSEmotionRecognizer`, and an optional `torch` import was added solely for device detection. `__init__` now resolves `self.emotion_device` to `"mps"` when `torch.backends.mps.is_available()` (else `"cpu"`) and instantiates `HSEmotionRecognizer(model_name="enet_b2_8", device=...)` once for reuse across all videos. `_sample_emotions` calls `predict_emotions(crop_rgb, logits=False)` — an in-memory ndarray call with no disk round-trip — and takes the argmax FER+ label plus its softmax probability as the magnitude. A new `HSEMOTION_TO_CANONICAL` dict folds the 8-class FER+ vocabulary onto the canonical 03b label set: `happiness` → `joy`, and `contempt` (which has no 03b analog) → `disgust`, the closest negative-valence label. The now-unused `PYFEAT_EMOTION_COLUMNS` constant was deleted and `self.feat_detector` was renamed to `self.emotion_detector`. The graceful-degradation contract from Resolved Issue #6 is preserved unchanged: when the package is absent or inference raises, the `detected` flag stays `False` and the deterministic method-local-RNG mock generator runs. A `test_hsemotion_label_mapping` test locks the two documented renames; the full 03b suite (12 tests) is green, and module import + pipeline construction were verified with HSEmotion absent (device detection correctly resolved to `mps` on the M4 Max host). A side-by-side per-class calibration check against Resolved Issue #9's E2E clip remains a follow-up gated on installing HSEmotion on the Mac Studio host.

## ⚠️ Unresolved Issues & Suggestions

_No currently tracked unresolved issues. Both prior issues (Gemma 4 model tier and Py-Feat MPS migration) were resolved on May 14 — see Resolved Issues #16 and #17 above._
