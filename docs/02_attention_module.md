# AI Task Breakdown: The Attention Module (`filter_attention.py`)

## Objective
Build `filter_attention.py`, the first node of the waterfall filter, responsible for verifying if a bystander is visually attending to the robotic task using purely metadata and lightweight 2D heuristics.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Module Boilerplate & Data Loading
- **Action**: Create `filter_attention.py` with standard data science imports (e.g., `numpy`, `pandas`, `json`).
- **Action**: Write a function `load_ego4d_metadata(filepath)` that parses the Ego4D "Hands and Objects" annotations and "Looking At Me" (LAM) pre-annotations into a readable tabular format structure.

### Task 2: Object of Interest (OOI) Extraction
- **Action**: Write a function `extract_ooi_bbox(clip_metadata, frame_idx)` that returns the bounding box `[x1, y1, x2, y2]` of the primary active task object for a given frame.

### Task 3: 2D Gaze Vector Calculation
- **Action**: Implement a function `calculate_gaze_vector(frame)` using a lightweight model like OpenFace 2.0 or PyGaze.
- **Constraint**: For the 16GB RAM constraints, prioritize extracting pre-calculated 2D gaze vectors directly from the dataset's LAM metadata if available. Only fallback to running frame-by-frame inference if metadata is missing.

### Task 4: Intersection Heuristic Logic
- **Action**: Write a geometric function `check_gaze_intersection(gaze_vector, ooi_bbox) -> bool` to determine if the 2D vector raycasts into the bounds of the OOI box.
- **Action**: Write the core filter function `evaluate_attention(clip_frames, min_threshold=0.60)`. It must calculate the percentage of total frames where the gaze intersects the OOI. If `< 60%`, flag the clip as `ATTENTION_FALSE`.

### Task 5: Output & Pipeline Interfacing
- **Action**: Write the evaluation results to an output file (e.g., `attention_results.json`) indicating `{clip_id: string, passed_attention: bool}`.
- **Success Criteria**: The script successfully processes a batch of clip metadata and discards clips failing the 60% intersection threshold rapidly (within seconds/minutes, not hours).
