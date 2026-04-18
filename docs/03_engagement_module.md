# AI Task Breakdown: The Engagement Module (`analyze_engagement.py`)

## Objective
Develop the logic to extract the actor's observable "Cognitive State" (`Focused`, `Neutral`, `Startled`) from filtered clips using a Temporal Vision-Language Model. Only clips that passed Node 1's attention filter are processed here.

## ⚠️ Critical: Use the Third-Person Paired Video
The Charades-Ego egocentric clip shows the world **from the actor's own eyes** — you **cannot** see the actor's face, shoulders, or torso in this view. The engagement module **must** load the **third-person paired video** (via `tp_video_id` / `tp_video_path` from `paired_clips_manifest.csv`) where the actor's full body and facial expressions are visible to the camera.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Environment & Data Ingestion
- [x] **Action**: Create `analyze_engagement.py`.
- [x] **Action**: Load `attention_results.json` and filter for `passed_attention == true`.
- [x] **Action**: Look up `tp_video_path` from `paired_clips_manifest.csv`.

### Task 2: Temporal Sampling Logic
- [x] **Action**: Sample frames from the **third-person video** within the `primary_window` at 1 FPS (max 5 frames).
- [x] **Action**: Implement `sample_frames_base64` using `cv2.CAP_PROP_POS_MSEC` for precise seeking.
- [x] **Action**: Encode frames as base64 strings for Ollama compatibility.

### Task 3: VLM Integration & Inference
- [x] **Action**: Integrate **Ollama via `ollama.chat()`** with the `moondream` model.
- [x] **Action**: Implement `generate_engagement_prompt()` for strict-JSON instructions.
- [x] **Action**: Execute inference using base64 image strings in the message payload.

### Task 4: Output Parsing & Formatting
- [x] **Action**: Write a robust parser `parse_vlm_response(response_text)` for `FOCUSED`, `NEUTRAL`, `STARTLED`.
- [x] **Action**: Implement error handling: Catch `JSONDecodeError` or unmapped responses and assign `UNKNOWN`.
- [x] **Action**: Save output to `engagement_results.json`.
- [x] **Action**: Derive `confidence`: `1.0` for success (if unspecified), `0.3` for `UNKNOWN`.
- **Success Criteria**: The VLM categorizes a batch of clips into the three designated cognitive states using the third-person view, without causing Mac OOM (Out of Memory) crashes or swapping freezes.

## ✅ Status & Completion
- **Pivot**: Successfully switched from `mlx-vlm` to `Ollama (Moondream)` for cognitive state classification due to persistent library conflicts.
- **Task 1**: Completed. `analyze_engagement.py` handles manifest ingestion and filtering accurately.
- **Task 2**: Completed. Implemented `sample_frames_base64` using `cv2` with 1 FPS sampling and precise seeking.
- **Task 3**: Completed. Ollama integration via `chat()` API with `moondream` is stable.
- **Task 4**: Completed. Results exported to `engagement_results.json` with refined confidence scoring and error logging.

**Progress Summary (Verification 2026-04-18)**:
- **Test Sub-batch**: 2 clips processed via `venv/bin/python3`.
- **Consistency**: Both clips classified as `FOCUSED` with confidence 0.65-0.8.
- **Robustness**: Verified `1.0` default confidence for successful parses and `0.3` for `UNKNOWN`.

## ✅ Resolved Pipeline Issues (Audit 2026-04-18)

| Issue | Root Cause | Resolution |
|-------|------------|------------|
| **VLM API Bug** | `ollama.generate()` failed with base64 images. | Switched to `ollama.chat()` with message payloads. |
| **Confidence Logic** | Defaulted to 0.5 instead of 1.0/0.3 spec. | Updated `analyze_engagement.py` to match requirements. |
| **Task Numbering** | Task 2 was missing from the instructions. | Re-organized `03_engagement_module.md` for clarity. |
| **Module Import** | `ModuleNotFoundError` for `ollama` in `vlm_env`. | Identified `venv` as the correct environment; documented usage. |
| **Library Conflicts** | `mlx-vlm` and `transformers` v4.49+ clashed. | Fixed by pivoting to Ollama/Moondream backend. |
