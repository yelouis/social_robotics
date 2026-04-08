# AI Task Breakdown: Hugging Face Hub Export (`export_hf_dataset.py`)

## Objective
Finalize the pipeline by automatically uploading the sanitized, formatted Parquet dataset of social rewards to the Hugging Face Hub, ensuring it is ready for public use and meets the Embodied Data Leaderboard schema requirements.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Hugging Face Authentication & Environment
- **Action**: Verify the installation of the `huggingface_hub` Python library (e.g., `pip install huggingface_hub`).
- **Action**: Implement an environment variable check in `export_hf_dataset.py` for the user's `HF_TOKEN`.
- **Action**: Write logic to gracefully prompt the user or halt execution if the authentication token is missing.

### Task 2: Dataset Repository Initialization
- **Action**: Import `HfApi` from the `huggingface_hub` library.
- **Action**: Write a function `initialize_hf_repo(repo_id)` (e.g., target ID: `louisye/ego4d-social-reward-vla`).
- **Action**: Use the API to create a new Dataset repository on Hugging Face if it does not already exist (`repo_type="dataset"`).

### Task 3: Automatic Dataset Card Generation (`README.md`)
- **Action**: Create a multi-line python string template for a Hugging Face Dataset Card (`README.md`). It must include:
  - Embedded YAML metadata tags specifying the dataset is for `Reinforcement Learning from Human Feedback (RLHF)` and `Robotics / VLA`.
  - A brief description of the "Modular Social-Affective Filter" methodology and hardware constraint mitigations.
  - An explanation of the calculated `Social Reward` ($R_s$) formula.
  - A crucial compliance notice explicitly stating that the repository hosts a **Derived Work** of metadata annotations, safely adhering to the "Not Redistributing the Database" mandate of the Ego4D license.
- **Action**: Write a function `upload_dataset_card(repo_id, readme_content)` that commits this generated `README.md` to the root of the repository.

### Task 4: Parquet Data Upload (The "No-Pixels" Rule)
- **Constraint**: Ensure the finalized `.parquet` file generated in Module 5 strictly contains metadata (`video_id`, `timestamp_start`, `timestamp_end`, `$R_s$`, `human_state_metadata`) and absolutely zero `video_bytes` or original media.
- **Action**: Write the core upload function `upload_parquet_to_hub(local_parquet_path, repo_id)`.
- **Action**: Point the function to the finalized `.parquet` file.
- **Action**: Use `HfApi().upload_file` to push the parquet file directly to a standardized `data/` directory inside the Hugging Face repository.

### Task 5: Hydration Script Implementation (`load_dataset.py`)
- **Constraint**: Because we strictly avoid redistributing videos, we must follow the community standard by supplying a "Hydration Script" alongside our dataset.
- **Action**: Implement a `load_dataset.py` script that handles the reconstruction of the full dataset on the end-user's machine.
- **Action**: Include logic to download your derived metadata (the $R_s$ scores) from Hugging Face.
- **Action**: Include logic to securely prompt the user for their Ego4D credentials.
- **Action**: Import or invoke the official `ego4d` CLI to "hydrate" the dataset, automatically downloading only the specific clips defined by the metadata locally.
- **Action**: Use `HfApi().upload_file` to push `load_dataset.py` to the root of the Hugging Face repository.

### Pipeline Success
- **Success Criteria**: The pipeline outputs a success message containing the live URL to the newly created Hugging Face Dataset repository, which correctly hosts the Dataset Card, the derived Parquet data containing no pixels, and the `load_dataset.py` hydration script.
