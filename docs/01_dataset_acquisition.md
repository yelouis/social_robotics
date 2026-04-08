# AI Task Breakdown: Dataset Acquisition

## Objective
Implement an automated script or set of instructions to acquire the Ego4D v2.1 dataset focusing specifically on social and audiovisual benchmarks for VLA training.

## ⚠️ Critical Action for Your M1 Pro (SSD Wear Prevention)
**DO NOT** run this data extraction or project pipeline directly on your Mac's internal 512GB SSD. The constant read/write cycles from `ffmpeg` and VLM inference will rapidly accelerate SSD wear.
- **Requirement**: Utilize a 1TB+ External NVMe SSD (e.g., Samsung T7 or SanDisk Extreme).
- **Action**: Configure your paths to set the `--output_directory` to that external drive.
- **Action**: Symlink your Python virtual environment (e.g., `venv` or `conda` env) to the external drive if possible to push all heavy OS I/O operations off the internal drive.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Environment Setup
- **Action**: Check if the `ego4d` CLI tool is installed. If not, install it via the official Ego4D instructions (e.g., `pip install ego4d`).
- **Action**: Ensure the user has authenticated with Ego4D and downloaded their necessary credentials and license keys to authorize the CLI.

### Task 2: Data Retrieval Script
- **Action**: Create a bash script (e.g., `download_ego4d_social.sh`).
- **Action**: Add the following command to the script to download only clips with social interaction metadata:
  ```bash
  ego4d --output_directory="/Volumes/YourExternalDrive/ego4d_data" --datasets clips --benchmarks social
  ```
- **Constraint**: Ensure the script validates the existence of the external drive output directory (e.g., `/Volumes/YourExternalDrive/ego4d_data`).

### Task 3: Verification
- **Action**: Add a verification step in the script that checks the total size and file count of the downloaded dataset to ensure social metadata (`.json` or `.csv`) and corresponding clips are actually present.
- **Success Criteria**: The generated external drive folder contains the necessary video clips and metadata files without downloading the entire terabytes-large full dataset.

### Task 4: Ingestion Node (`ingest_node.py`) & Auto-Cleanup
- **Action**: Create an `ingest_node.py` script responsible for handling localized frame extraction (e.g., using `cv2` or `ffmpeg`) for downstream visual modules.
- **Constraint**: Implement strict **"Auto-Cleanup" logic**. Because of SSD wear constraints, immediately after frame metadata is generated and the video state is passed to RAM for the next module, you **must delete all extracted frames** from the external drive.
- **Success Criteria**: The system seamlessly extracts necessary visual frames dynamically, but never accumulates a backlog of `.jpg`/`.png` image artifacts taking up drive space.
