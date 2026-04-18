# AI Task Breakdown: Real-Time Sanity Checks & Eng Workflow

## Objective
Implement infrastructure safety mechanisms, visual debugging, and metric checks to prevent "Silent Failures" commonly caused by extreme memory pressure on 24GB unified memory environments.

## Status: ✅ COMPLETED
- **Implemented**: April 18, 2026
- **Safety Library**: `saf_pipeline_utils.py`
- **Visual Tools**: `debug_overlay.py`

## Agent Instructions: Step-by-Step Tasks

### Task 1: Visual Debugger Overlay (`debug_overlay.py`)
- **Action**: Create `debug_overlay.py` utilizing `OpenCV`. [DONE]
- **Action**: Write rendering function `annotate_frame`. [DONE]
- **Action**: Utilize `cv2.line`, `cv2.circle`, and `cv2.putText` to draw pose skeleton, action labels, and VLM state. [DONE]
- **Action**: Render egocentric and third-person views side-by-side. [DONE]
- **Action**: Write pipeline hook `export_debug_video`. [DONE]

### Task 2: Distribution Monitoring (The Flatline Check)
- **Action**: Implement `DistributionMonitor` with rolling buffer. [DONE]
- **Action**: Implement anomaly detection for `std < 0.1`. [DONE]
- **Action**: Raise `ModelCollapseError` logic. [DONE]

### Task 3: Confidence Routing (Gemma 4 E2B)
- **Action**: Ensure `Engagement` and `Flinch` nodes augment returns with `Confidence_Score`. [DONE]
- **Action**: Use **Gemma 4** to cross-check VLM and detect hallucinations. [DONE]
- **Action**: Write `ConfidenceRouter` logic to flag clips with `min_confidence < 0.6` to `manual_review/`. [DONE]

### Task 4: The "Reaction Lag" Temporal Filter
- **Action**: Write `validate_reaction_lag(action_start_timestamp, flinch_frame_timestamp)`. [DONE]
- **Action**: Implement `100ms < lag < 5000ms` logic. [DONE]

## Success Criteria
- ✅ Pipeline safeguardsmodularized in `saf_pipeline_utils.py`.
- ✅ Intercepts hallucinated, collapsed, or physically impossible system conclusions.
