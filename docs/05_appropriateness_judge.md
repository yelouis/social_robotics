# AI Task Breakdown: The Appropriateness Judge (`judge_alignment.py`)

## Objective
Build `judge_alignment.py`, the synthesis module that mathematically combines Attention, Engagement, and Flinch data into a single "Social Reward" scalar ($R_s$). Formats the results into a Parquet file for embodied AI leaderboard submission.

## Status: ✅ COMPLETED
- **Implemented**: April 18, 2026
- **Primary Orchestrator**: `judge_alignment.py`
- **Reasoning Model**: Gemma 4 E4B (via Ollama)
- **Export Format**: Apache Parquet (Snappy Compression)

## Agent Instructions: Step-by-Step Tasks

### Task 1: Data Synthesis & Model Selection
- **Action**: Create `judge_alignment.py`. [DONE]
- **Constraint**: Utilize **Gemma 4 E4B** for reasoning and synthesis. [DONE]
- **Action**: Write an orchestrator function `compile_human_state(ego_video_id)` that loads and joins the cached outputs. [DONE]
- **Action**: Join on `ego_video_id`. Handle missing entries defensively. [DONE]

### Task 2: Reward Calculation Logic
- **Action**: Define the weighting dictionary. [DONE]
- **Action**: Write the core function `calculate_social_reward(attention_bool, state_str, flinch_bool)`. [DONE]
- **Action**: Execute the exact formula: `R_s = (int(attention_bool) * STATE_WEIGHTS[state_str]) - (int(flinch_bool) * 2.0)`. [DONE]
- **Action**: Log the resulting float per clip. [DONE]

### Task 3: Gemma 4 Justification (Optional, Gated by Flag)
- **Action**: Write an optional function `generate_justification(compiled_state, r_s)` using **Gemma 4 E4B**. [DONE]
- **Action**: Prompt template implemented as requested. [DONE]

### Task 4: Parquet Export Formatting (The "No-Pixels" Rule)
- **Constraint**: Exported dataset must be a **Derived Signal Dataset** (no pixels). [DONE]
- **Action**: Write a formatting function `export_to_parquet`. [DONE]
- **Action**: Standardize the data into a Pandas DataFrame with the required schema. [DONE]
- **Action**: Invoke `df.to_parquet(output_path, engine="pyarrow")`. [DONE]

## Success Criteria
- ✅ Valid `.parquet` file generated containing strictly derived annotation metadata.
- ✅ Social Reward correctly penalizes flinches and rewards attentive focus.
