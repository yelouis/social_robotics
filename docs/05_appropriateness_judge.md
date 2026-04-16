# AI Task Breakdown: The Appropriateness Judge (`judge_alignment.py`)

## Objective
Build `judge_alignment.py`, the synthesis module that mathematically combines Attention, Engagement, and Flinch data into a single "Social Reward" scalar ($R_s$). Formats the results into a Parquet file for embodied AI leaderboard submission.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Data Synthesis
- **Action**: Create `judge_alignment.py`.
- **Action**: Write an orchestrator function `compile_human_state(ego_video_id)` that loads and joins the cached outputs of all three upstream nodes:
  1. `attention_results.json` → `passed_attention`, `activity_fraction`, `primary_window`
  2. `engagement_results.json` → `state`, `confidence`
  3. `flinch_results.json` → `flinch`, `v_peak`, `confidence`
- **Action**: Join on `ego_video_id`. Handle missing entries defensively — if a clip is present in attention but missing from engagement (e.g., VLM crashed), flag it as `INCOMPLETE` and exclude from the final export.

### Task 2: Reward Calculation Logic
- **Action**: Define the weighting dictionary:
  ```python
  STATE_WEIGHTS = {
      "FOCUSED": 1.0,
      "NEUTRAL": 0.5,
      "STARTLED": 0.0,
      "UNKNOWN": 0.0
  }
  ```
- **Action**: Write the core function `calculate_social_reward(attention_bool, state_str, flinch_bool)`.
- **Action**: Execute the exact formula:
  `R_s = (int(attention_bool) * STATE_WEIGHTS[state_str]) - (int(flinch_bool) * 2.0)`
- **Rationale**: `NEUTRAL` is weighted at `0.5` (not `0.8` as in a prior draft) to create clearer separation between actively focused and passively present. A flinch always dominates with a `-2.0` penalty, driving $R_s$ deeply negative as a strong safety signal. `Distracted` was removed — the attention module's boolean already handles that case.
- **Action**: Log the resulting float per clip.

### Task 3: VLM Justification (Optional, Gated by Flag)
- **Action**: Write an optional function `generate_justification(compiled_state, r_s)` gated by a `--justify` CLI flag (disabled by default to save VLM calls).
- **Action**: Prompt template:
  *"A person was performing a household task. Their attention was [True/False], their cognitive state appeared [State], and they [did/did not] physically flinch (V_peak={v_peak}). The computed social reward is {R_s}. In one sentence, explain whether this suggests the person was comfortable during the activity."*

### Task 4: Parquet Export Formatting (The "No-Pixels" Rule)
- **Constraint**: The exported dataset must strictly be a **Derived Signal Dataset**. The `.parquet` file must **not** contain raw video clips, `video_bytes`, extracted frames, or any part of the original Charades-Ego / Charades media, to comply with the AI2 non-redistribution license.
- **Action**: Write a formatting function `export_to_parquet(results_list, output_path="~/charades_ego_data/exports/reward_dataset.parquet")`.
- **Action**: Standardize the data into a Pandas DataFrame with the following schema:

  | Column | Type | Description |
  |--------|------|-------------|
  | `ego_video_id` | str | Charades-Ego egocentric clip ID |
  | `tp_video_id` | str | Paired third-person Charades clip ID |
  | `timestamp_start` | float | Start of the analysis window (seconds) |
  | `timestamp_end` | float | End of the analysis window (seconds) |
  | `action_annotations` | str | Original Charades action annotation string for the window |
  | `cognitive_state` | str | VLM-derived state: FOCUSED / NEUTRAL / STARTLED |
  | `flinch_detected` | bool | Kinematic flinch override flag |
  | `v_peak` | float | Peak upper-body velocity (pixels/sec) |
  | `social_reward` | float | Final $R_s$ scalar |

- **Action**: Invoke `df.to_parquet(output_path, engine="pyarrow")` to save.
- **Success Criteria**: A structurally valid `.parquet` file is generated containing strictly derived annotation metadata (with zero video/pixel data), ready for leaderboard submission.
