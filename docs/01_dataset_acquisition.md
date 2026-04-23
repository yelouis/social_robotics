# AI Task Breakdown: Dataset Acquisition

## Objective
Download and stage large-scale first-person POV (egocentric) videos—such as Ego4d or other relevant datasets—onto the local system for ingestion into the pipeline.

## Core Supported Datasets
The system is built to parse First-Person POV interactions. While the pipeline is designed to be relatively dataset-agnostic, the following are the primary supported sources:

- **Ego4D**: The primary target standard representing unstructured, daily interaction.
- **EPIC-KITCHENS-100**: Academic download via University of Bristol.
- **Charades-Ego**: Open download from Allen AI.
- **EgoProceL**: Open GitHub repository + links to source datasets.

## 🔝 Dataset Download Priority
Due to storage constraints and project focus, datasets are prioritized in the following order:
1.  **Ego4D** (Highest Priority: Primary target for unstructured interaction)
2.  **Charades-Ego** (Second Priority: Large-scale egocentric dataset)
3.  **EPIC-KITCHENS-100** (Third Priority: Specialized task-based interaction)
4.  **EgoProceL** (Meta-dataset repository)

## ⚠️ Critical Action for Hardware Management
Video datasets are inherently massive (often spanning terabytes). 
**DO NOT** run these data pipelines directly on strict internal SSD setups if space/wear are concerns. 
- Ensure your `OUTPUT_DIR` maps out to the external **2TB SSD called "Extreme SSD"**. This is explicitly designated to handle the large size of the Ego4D and other video datasets.
- **Hardware Profile**: As we are running a **Mac mini M4 Pro with 24GB RAM**, be strictly mindful of memory ceilings when unpacking or indexing these massive datasets in Python. Rely on streaming processors or chunked extraction.

## Streaming Filtering Strategy (Storage Optimization)
To mitigate the massive storage requirements of these datasets, the system employs a **Streaming Filtering** strategy. 

- **Download-Filter-Discard (Batched)**: Videos are processed in small batches (e.g., 50 UIDs at a time). For each batch, the system downloads the raw files, runs the social filter, and then immediately deletes non-compliant videos.
- **Keep Criterion**: Only videos with **more than one person** (excluding the camera wearer) are persisted on the "Extreme SSD".
- **Resumability**: To avoid re-downloading videos that were previously discarded, the system maintains a `processed_uids.json` file. Once a UID has been evaluated (kept or purged), it is marked as processed and never requested from the downloader again.

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

## 🧪 Resolved Issues & Implementation Refinements (April 2026)

1. **Storage Capacity vs. Massive Datasets (Resolved)**:
   - **Problem**: Large-scale datasets like Ego4D and EPIC-KITCHENS-100 can exceed the 1.6 TiB available on the Extreme SSD.
   - **Solution**: Implemented the **Streaming Filtering Strategy**. Videos are now evaluated for social presence immediately after download or extraction. Non-social videos are purged on-the-fly, ensuring that only relevant data (estimated to be <20% of the raw volume) is persisted.

2. **Download Throughput (Mitigated)**:
   - **Problem**: Standard python-based downloads were throttled or inefficient.
   - **Solution**: The system now supports prioritized downloads (Ego4D > Charades-Ego > EPIC) and uses the more robust `ego4d` CLI for the primary dataset.

3. **macOS Hidden Files on External SSD (Resolved)**:
   - **Problem**: Git operations on the Extreme SSD generated `non-monotonic index` errors due to `._` files.
   - **Solution**: Implemented explicit filtering in `registry.py` to skip files starting with `._` and documented the use of `dot_clean` for SSD maintenance.

4. **Streaming Filter Diverges from Pipeline Filter (Resolved)**:
   - **Problem**: The streaming acquisition filter and the main processing pipeline had independent, divergent detection logic.
   - **Solution**: Consolidated all detection logic into `src/shared/social_presence.py`. Both modules now use the same YOLO-based engine, ensuring consistent behavior across the entire lifecycle of a video clip.

5. **SSD Storage Overflow & Re-download Loops (Resolved - April 22)**:
   - **Problem**: The raw Ego4D CLI download of 9,821 videos (~5-10TB) would fill the 2TB Extreme SSD before filtering could occur. Additionally, deleting discarded videos caused the CLI to re-download them on the next run.
   - **Solution**: Implemented **UID-based Batching** and **Processed UID Tracking**. The script `src/dataset_acquisition/run_selective_download.py` now requests Ego4D videos in batches of 50. After each batch, the filter runs immediately and purges "bad" videos. A persistent `processed_uids.json` file ensures that discarded UIDs are never re-requested, effectively "finishing" the dataset in chunks without ever exceeding SSD capacity.

6. **Ego4D Camera Wearer Detection (Resolved - April 22)**:
   - **Problem**: YOLOv8 incorrectly identified the camera wearer's own limbs (arms/legs) and "ghost torsos" (full-frame background artifacts) as bystanders.
   - **Solution**: Implemented a **Refined Anti-Wearer Heuristic**:
     - **Limb Filtering**: Ignored boxes touching the bottom edge without a visible head/shoulders.
     - **Ghost Exclusion**: Ignored full-height boxes starting exactly at the top edge (`y1=0`), a common egocentric false positive.
     - **Confidence Boost**: Increased YOLO confidence threshold from 0.25 to **0.50**.
     - **Temporal Consistency**: Required at least **2 frames** of social presence per video to confirm (filtering out 1-frame glitches).
