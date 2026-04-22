# AI Task Breakdown: Dataset Acquisition

## Objective
Download and stage large-scale first-person POV (egocentric) videos—such as Ego4d or other relevant datasets—onto the local system for ingestion into the pipeline.

## Core Supported Datasets
The system is built to parse First-Person POV interactions. While the pipeline is designed to be relatively dataset-agnostic, the following are the primary supported sources:

- **Ego4D**: The primary target standard representing unstructured, daily interaction.
- **EPIC-KITCHENS-100**: Academic download via University of Bristol.
- **Charades-Ego**: Open download from Allen AI.
- **EgoProceL**: Open GitHub repository + links to source datasets.

## ⚠️ Critical Action for Hardware Management
Video datasets are inherently massive (often spanning terabytes). 
**DO NOT** run these data pipelines directly on strict internal SSD setups if space/wear are concerns. 
- Ensure your `OUTPUT_DIR` maps out to the external **2TB SSD called "Extreme SSD"**. This is explicitly designated to handle the large size of the Ego4D and other video datasets.
- **Hardware Profile**: As we are running a **Mac mini M4 Pro with 24GB RAM**, be strictly mindful of memory ceilings when unpacking or indexing these massive datasets in Python. Rely on streaming processors or chunked extraction.

## Recommended Implementation Steps for Agents

### Task 1: Environment Definition
- Setup environment variables. Ensure the ingestion directory is properly symlinked or mapped out to the **"Extreme SSD" (2TB external storage)**.
- Download necessary command line tools (e.g. `aws-cli`, `wget`, `curl`).

### Task 2: Dataset Registration & Extraction 
- Most datasets require script-based downloaders.
- Write a `download_videos.sh` logic step. For Ego4D, this might employ their native python CLI package.
- Extract or mount the video `.mp4` chunks into a localized `/raw_videos` buffer folder so that the downstream module (`02_filtering_and_labeling`) can index the raw files.

### Task 3: Local Manifest Initialization
- Generate an initial local registry or manifest (e.g., `local_video_registry.json`) parsing all the successfully downloaded `.mp4` files.
- Each video should be assigned a UUID or dataset-specific ID pointing to its exact path on disk.

## Success Criteria
You should successfully hydrate a local folder filled with MP4 video clips, paired with an index/manifest referencing every file path accurately before moving to the Filtering module.

## Verification & Validation Check
To ensure data ingestion was successful without relying on blind trust, perform the following validation steps:
- **Singular Video Test**: Pick one UUID from the localized manifest and use a lightweight media player or python script (e.g., OpenCV) to explicitly read the file path and extract the first frame. If it opens without corruption, the test passes.
- **Batch Test**: Write a script to iterate over the entire initialized `local_video_registry.json`. For each entry, check that the file exists on the **"Extreme SSD"**, has a size greater than 0 bytes, and that the file extension matches `.mp4`. This ensures no silent download failures occurred across the massive file list.

## 🚀 Implementation Accomplishments (April 2026)

The initial implementation of the Dataset Acquisition module is complete:

- **SSD Integration**: Fully mapped to the **Extreme SSD** mount point (`/Volumes/Extreme SSD`).
- **Existing Data Recognition**: Implemented logic to scan the SSD for pre-existing `charades_ego_data` and `ego4d_data` folders, preventing redundant downloads of nearly **15,700 videos**.
- **Automated Registry**: Created `src/dataset_acquisition/registry.py` which generates a comprehensive `local_video_registry.json` manifest.
- **Verification Framework**: Added `tests/test_dataset_acquisition.py` to ensure video files are readable and non-corrupt using OpenCV frame extraction tests.
- **Smart Downloader**: Implemented a requirement-aware downloader in `src/dataset_acquisition/downloader.py` that skips datasets if keys (like AWS for Ego4D) are missing but data is already present on disk.


### 🐛 Bug Fixes & Refinements

During code review and validation, a few critical issues were identified and resolved:

1. **Subdirectory Indexing Failure (`downloader.py`)**:
   - **Problem**: The `is_already_downloaded()` method used `.glob("*.mp4")`, which only scanned the top-level directory. It failed to recognize existing datasets if videos were nested in subdirectories, leading to redundant downloads.
   - **Solution**: Changed `.glob` to `.rglob("*.mp4")` to search recursively.

2. **Missing Extraction Logic (`downloader.py`)**:
   - **Problem**: The `CharadesEgoDownloader` successfully downloaded the massive `.zip` file but lacked the logic to extract it. This prevented downstream modules from finding the raw `.mp4` chunks.
   - **Solution**: Integrated the `zipfile` module to extract contents directly into the designated `OUTPUT_DIR` after downloading.

3. **Duplicate Registry Entries & OS Hidden Files (`registry.py`)**:
   - **Problem**: Overlapping paths in `DATASET_PATHS` caused the registry script to index the same dataset multiple times (resulting in ~15.7k entries instead of the actual ~7.8k). Additionally, macOS hidden files (starting with `._`) could be incorrectly indexed.
   - **Solution**: Implemented a `seen_paths` set to track absolute paths and skip duplicates. Added logic to explicitly skip files starting with `._`. Running the fixed script successfully pruned the duplicate count to 7,861 unique videos.

## 🧪 Test Batch Results (April 2026)

To verify the pipeline without downloading terabytes of data, a test batch was executed using a simulated acquisition process:

- **Test Scope**: 5 videos from Charades-Ego and available samples from Ego4D.
- **Process**:
    1. Isolated 5 non-empty videos from the SSD.
    2. Copied them to a dedicated `test_raw_videos/` directory to simulate a fresh download.
    3. Ran the registry script to generate `test_video_registry.json`.
    4. Performed automated verification using `pytest`.
- **Results**:
    - **Registry**: Successfully indexed the 5 test videos while correctly filtering out macOS metadata files (`._`).
    - **Validation**: All 5 videos passed the existence, size, and frame extraction tests.
- **Command**: `REGISTRY_FILE=test_video_registry.json pytest tests/test_dataset_acquisition.py`
