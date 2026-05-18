# AI Task Breakdown: Dataset Acquisition

## Objective
Download and stage large-scale first-person POV (egocentric) videos—such as Ego4d or other relevant datasets—onto the local system for ingestion into the pipeline.

## Core Supported Datasets
The system is built to parse First-Person POV interactions. While the pipeline is designed to be relatively dataset-agnostic, the following are the primary supported sources:

- **Ego4D**: The primary target standard representing unstructured, daily interaction.
- **Charades-Ego**: Open download from Allen AI.
- **EPIC-KITCHENS-100**: (Deferred) Academic download via University of Bristol.
- **EgoProceL**: (Deferred) Open GitHub repository + links to source datasets.

> **Note on Scope**: Focus will exclusively remain on Ego4D and Charades-Ego to reach statistical significance before attempting complex integrations with gated or nested meta-datasets.

## 🔝 Dataset Download Priority
Due to storage constraints and project focus, datasets are prioritized in the following order:
1.  **Ego4D** (Highest Priority: Primary target for unstructured interaction)
2.  **Charades-Ego** (Second Priority: Large-scale egocentric dataset)
3.  *(Deferred)* **EPIC-KITCHENS-100** (Third Priority: Specialized task-based interaction)
4.  *(Deferred)* **EgoProceL** (Meta-dataset repository)

## ⚠️ Critical Action for Hardware Management
Video datasets are inherently massive (often spanning terabytes). 
**DO NOT** run these data pipelines directly on strict internal SSD setups if space/wear are concerns. 
- Ensure your `OUTPUT_DIR` maps out to the external **2TB SSD called "Extreme SSD"**. This is explicitly designated to handle the large size of the Ego4D and other video datasets.
- **Environment Variables**: PyTorch and HuggingFace default to caching on the internal SSD, which will quickly cause `[Errno 28] No space left on device` OS crashes. **You must export the following environment variables** to point to the external drive before running any scripts:
  - `export TMPDIR="/Volumes/Extreme SSD/tmp"`
  - `export TEMP="/Volumes/Extreme SSD/tmp"`
  - `export TMP="/Volumes/Extreme SSD/tmp"`
  - `export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"`
- **Hardware Profile**: As we are running a **Mac Studio (M4 Max, 64 GB unified memory)**, the memory ceiling is more generous than the prior Mac mini M4 Pro (24 GB), but streaming/chunked extraction is still the correct architectural default. Multi-terabyte datasets can still exceed physical RAM during indexing, and other layers (03d, 03f, 03g) co-reside in unified memory during the E2E run.
- **Dynamic Memory Unloading**: The pipeline avoids unnecessary model load/unload overhead on the 64GB Mac Studio by lazily retaining the YOLOv8 and MediaPipe models in MPS/unified memory between batches. It explicitly unloads models via `self.filterer.detector.unload()` only when system memory pressure exceeds 75%, adapting dynamically to contention.

## Streaming Filtering Strategy (Storage Optimization)
To mitigate the massive storage requirements of these datasets, the system employs a **Streaming Filtering** strategy. 

- **Download-Filter-Discard (Batched)**: Videos are processed in small batches. The system dynamically calculates safe batch sizes (between 10 and 500 UIDs) based on `shutil.disk_usage()` of the "Extreme SSD" at the start of each cycle. It downloads the raw files, runs the social filter, and then immediately deletes non-compliant videos.
- **Keep Criterion**: Only videos with **more than one person** (excluding the camera wearer) are persisted on the "Extreme SSD".
- **Resumability**: To avoid re-downloading videos that were previously discarded, the system maintains a `processed_uids.json` file. Once a UID has been evaluated (kept or purged), it is marked as processed and never requested from the downloader again.
- **Single Source of Truth**: The acquisition filter and the main processing pipeline both utilize `src/shared/social_presence.py` to ensure consistent social presence detection heuristics across the entire lifecycle.

### Filtering Heuristics & Performance
To optimize filtering accuracy and throughput during the streaming process, the following designs are enforced in `social_presence.py`:
- **Batched Inference**: The social filter accumulates frames into an internal batch array to leverage YOLO's native GPU batch inference, significantly improving throughput over sequential processing.
- **Wearer Detection Refinement**: YOLOv8 can incorrectly identify the camera wearer's limbs as external bystanders. This is mitigated by combining a geometric anti-wearer heuristic (bottom-edge exclusion, 0.50 confidence floor, 2-frame consistency) with **MediaPipe Hands**, which filters out YOLO person detections that heavily overlap (>40%) with detected hand bounding boxes.
- **State Isolation**: ByteTrack tracker state is explicitly reset on a per-video basis (`model.predictor.trackers.reset()`) to prevent `person_id` assignments and trajectory history from bleeding across different videos in the same batch.

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

## 🚀 Implementation Status

The Dataset Acquisition module is fully operational at `src/dataset_acquisition/`.
- **SSD Integration**: Fully mapped to the **Extreme SSD** (`/Volumes/Extreme SSD`).
- **Streaming Filter**: Implemented "Download-Filter-Discard" batching logic in `run_selective_download.py` to manage multi-terabyte datasets within a 2TB limit.
- **Smart Registry**: Automated indexing of 7,861 unique videos via `registry.py`, with explicit filtering for macOS metadata.
- **Requirement-Aware Downloader**: `downloader.py` handles authentication-gated datasets (Ego4D) and open-source datasets (Charades-Ego) with integrated extraction logic.

## 🧪 Resolved Issues & Implementation Refinements

*All past resolved issues have been successfully integrated into the system design documentation above.*

## ⚠️ Unresolved Issues & Suggestions

_No unresolved issues at this time._
