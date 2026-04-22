# Project Overview: Extensible Social Feature Extraction Pipeline
## 📌 Grounding Document for AI Generation

**Read This First**: If you are an AI assistant or coding agent attached to this project workspace, this is your foundational grounding document. By referencing this file, you understand the core goal and pipeline architecture of the `social_robotics` project.

---

## 🎯 Global Project Goal
The objective of this pipeline is to extract **social-related features and metadata from any video**—with a specific focus on first-person (POV/egocentric) videos, such as those found in **Ego4D, EPIC-KITCHENS, Charades-Ego, and EgoProceL**. 

Rather than relying on closed-circuit rules or fixed pairings of videos, the system behaves as an extensible data processing engine. It ingests raw videos from multiple repositories, runs multiple parallel validation and extraction filters over them, and outputs **dehydrated social metadata result files**. In the future, this data will be aggregated and pushed to platforms like Hugging Face.

### Output format
The final result of the pipeline is a dehydrated dataset. Because of license constraints of various video dataset platforms, we NEVER export raw video pixels. The final exported files strictly contain:
1. Identifying metadata (how to re-hydrate the dataset with original sources).
2. The exact social features and values derived by our independent extraction layers.

---

## 🧩 Pipeline Architecture

The pipeline follows a multi-stage approach. Steps 2, 3, and 4 are modular by design.

1. **Dataset Acquisition** (`01_dataset_acquisition.md`)
   - Initial ingestion point to pull raw video sources (e.g., from Ego4d, EPIC-KITCHENS, Charades-Ego, or EgoProceL) to the local disk.

2. **Filtering & Task Labeling** (`02_filtering_and_labeling.md`)
   - *Social Presence Filter*: We strictly filter the dataset for videos containing *more than one person* (excluding the POV cameraperson). 
   - *Task Labeling*: Analyzes what task the POV person is currently undertaking. The video is discarded if there is no clear task being performed.

3. **Social Feature Layers** (`03_social_layer_architecture.md`)
   - The core engine room of the project. These are **independent, on-going extraction layers** written as separate modules. 
   - Current layers include the Attention Layer (03a) for gaze tracking, the Acoustic Prosody Layer (03c) for non-verbal vocal cues, and the Motor Resonance Layer (03f) for sympathetic flinch detection. See the full registry in `03_social_layer_architecture.md`.
   - Every active layer parses the filtered datasets, evaluates the video chunks, and appends its outputs to the centralized metadata record.

4. **Dehydrated Export** (`04_dehydrated_export.md`)
   - Merges the extraction schemas, removes any underlying raw data (pixels/audio), and prepares the output `.parquet` and dataset cards for Hugging Face or other platforms.

### Supporting Documents
- **ML Dependencies** (`ml_dependencies.md`): Centralized registry of all model weights, libraries, and system tools required across the pipeline. Must be updated when any layer adds or changes a model dependency.
