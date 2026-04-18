# AI Task Breakdown: Hugging Face Hub Export (`export_hf_dataset.py`)

## Objective
Finalize the pipeline by uploading the sanitized Parquet dataset of social rewards (derived from Charades-Ego) to the Hugging Face Hub. Per the AI2 Charades-Ego license (non-commercial, no redistribution), zero raw video frames or clips are included — only derived annotations and computed signals.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Hugging Face Authentication & Environment
- **Action**: Verify the installation of the `huggingface_hub` Python library (e.g., `pip install huggingface_hub`).
- **Action**: Implement an environment variable check in `export_hf_dataset.py` for the user's `HF_TOKEN`.
- **Action**: Write logic to gracefully prompt the user or halt execution if the authentication token is missing.

### Task 2: Dataset Repository Initialization
- **Action**: Import `HfApi` from the `huggingface_hub` library.
- **Action**: Write a function `initialize_hf_repo(repo_id)` (e.g., target ID: `louisye/charades-ego-social-reward-vla`).
- **Action**: Use the API to create a new Dataset repository on Hugging Face if it does not already exist (`repo_type="dataset"`).

### Task 3: Automatic Dataset Card Generation (`README.md`)
- **Action**: Create a multi-line python string template for a Hugging Face Dataset Card (`README.md`). It must include:
  - Embedded YAML front matter specifying: `task_categories: [reinforcement-learning]`, `tags: [robotics, vla, social-reward, charades-ego]`, `license: other`.
  - A "Dataset Description" section explaining that this is a **derived signal dataset** computed from the Charades-Ego + original Charades paired-view videos using the Modular Social-Affective Filter (SAF) pipeline.
  - A clear statement that Charades-Ego is a **proxy** dataset: the reward signals measure human comfort/focus during daily manipulation tasks that transfer to robotics settings.
  - The `Social Reward` ($R_s$) formula and the `STATE_WEIGHTS` dictionary.
  - The parquet column schema (matching `05_appropriateness_judge.md` Task 4).
  - **License compliance notice**: This repository contains zero raw video or redistributed media. It hosts only computed annotations and reward scalars. The original Charades-Ego and Charades videos can be obtained directly from AI2's public S3 bucket — no credentials required. Links must be provided.
- **Action**: Write a function `upload_dataset_card(repo_id, readme_content)` that commits this generated `README.md` to the root of the repository.

### Task 4: Parquet Data Upload (The "No-Pixels" Rule)
- **Constraint**: Ensure the finalized `.parquet` file strictly contains the metadata schema defined in `05_appropriateness_judge.md` (`ego_video_id`, `tp_video_id`, `timestamp_start`, `timestamp_end`, `action_annotations`, `cognitive_state`, `flinch_detected`, `v_peak`, `social_reward`) and absolutely zero `video_bytes` or original media.
- **Action**: Write the core upload function `upload_parquet_to_hub(local_parquet_path, repo_id)`.
- **Action**: Point the function to `~/charades_ego_data/exports/reward_dataset.parquet`.
- **Action**: Use `HfApi().upload_file` to push the parquet file to a standardized `data/` directory inside the Hugging Face repository.

### Task 5: Hydration Script Implementation (`load_dataset.py`)
- **Constraint**: Because we avoid redistributing videos, supply a "Hydration Script" that lets end-users reconstruct the full dataset locally. Unlike Ego4D, Charades-Ego requires **no credentials** — the videos are publicly hosted on AI2's S3 bucket.
- **Action**: Implement a `load_dataset.py` script that:
  1. Downloads the derived metadata parquet from Hugging Face using the `datasets` library.
  2. Prompts the user to confirm download of the video archives (with size estimates):
     ```python
      CHARADES_EGO_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo_v1_480.tar" # ~11GB
      CHARADES_TP_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip" # ~15GB
     ```
  3. Extracts and indexes both ego and third-person videos locally.
  4. Merges the video paths with the parquet metadata by `ego_video_id` and `tp_video_id` to produce a hydrated, queryable dataset object.
- **Action**: Use `HfApi().upload_file` to push `load_dataset.py` to the root of the Hugging Face repository.

## ✅ Resolved Pipeline Issues (Audit 2026-04-17)

| Issue | Root Cause | Resolution |
|-------|------------|------------|
| **Hydration URL** | S3 bucket uses `.zip` for original Charades, not `.tar`. | Updated URLs in Task 5 to match `download_charades_ego.sh`. |
| **Size Estimates** | Initial estimates were inaccurate. | Updated ego/tp estimates to reflect actual data sizes. |

### Pipeline Success
- **Success Criteria**: The pipeline outputs a success message containing the live URL to the newly created Hugging Face Dataset repository, which correctly hosts: (1) the Dataset Card, (2) the derived Parquet data with zero pixels, and (3) the `load_dataset.py` hydration script with public S3 download links for both ego and third-person video archives.
