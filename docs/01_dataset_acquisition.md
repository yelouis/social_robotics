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

## 🚀 Implementation Status

The Dataset Acquisition module is fully operational at `src/dataset_acquisition/`.
- **SSD Integration**: Fully mapped to the **Extreme SSD** (`/Volumes/Extreme SSD`).
- **Streaming Filter**: Implemented "Download-Filter-Discard" batching logic in `run_selective_download.py` to manage multi-terabyte datasets within a 2TB limit.
- **Smart Registry**: Automated indexing of 7,861 unique videos via `registry.py`, with explicit filtering for macOS metadata.
- **Requirement-Aware Downloader**: `downloader.py` handles authentication-gated datasets (Ego4D) and open-source datasets (Charades-Ego) with integrated extraction logic.

## 🧪 Resolved Issues & Implementation Refinements

1. **Subdirectory Indexing Failure (Resolved - April 21)**:
   - **Problem**: The `is_already_downloaded()` method in `downloader.py` used `.glob("*.mp4")`, which failed to detect videos nested in subdirectories, causing redundant downloads.
   - **Solution**: Changed scanning logic to `.rglob("*.mp4")` to search all subfolders recursively.

2. **Missing Extraction Logic (Resolved - April 21)**:
   - **Problem**: `CharadesEgoDownloader` downloaded large `.zip` files but lacked extraction code, preventing the pipeline from accessing raw `.mp4` chunks.
   - **Solution**: Integrated the `zipfile` module to automatically extract contents into the `OUTPUT_DIR` upon download completion.

3. **Duplicate Registry Entries & OS Metadata (Resolved - April 21)**:
   - **Problem**: Overlapping paths in the registry caused nearly 8,000 duplicate entries, and macOS hidden files (`._`) were incorrectly indexed as valid videos.
   - **Solution**: Implemented a `seen_paths` set to ensure absolute uniqueness and added explicit string filtering to skip files starting with `._`.

4. **Storage Capacity vs. Massive Datasets (Resolved - April 22)**:
   - **Problem**: Ego4D and EPIC-KITCHENS-100 can exceed the 1.6 TiB available on the SSD if downloaded in full.
   - **Solution**: Implemented the **Streaming Filtering Strategy**. Videos are evaluated immediately after download and non-social clips are purged on-the-fly, reducing the persisted volume to <20% of the raw dataset.

5. **Download Throughput (Mitigated - April 22)**:
   - **Problem**: Standard Python-based downloads were inefficient for massive file counts.
   - **Solution**: Prioritized the download queue (Ego4D > Charades-Ego) and integrated the native `ego4d` CLI for high-throughput AWS-backed transfers.

6. **Streaming Filter Logic Divergence (Resolved - April 22)**:
   - **Problem**: The acquisition filter and the main processing pipeline used independent, divergent detection logic, leading to inconsistent datasets.
   - **Solution**: Consolidated all detection heuristics into `src/shared/social_presence.py`, ensuring a "Single Source of Truth" for the entire lifecycle of a video clip.

7. **SSD Storage Overflow & Re-download Loops (Resolved - April 22)**:
   - **Problem**: Deleting discarded videos caused the Ego4D CLI to re-download them on subsequent runs, leading to infinite download loops and SSD overflow.
   - **Solution**: Implemented **UID-based Batching** and **Processed UID Tracking**. A persistent `processed_uids.json` ledger ensures that purged videos are never re-requested.

8. **Ego4D Camera Wearer Detection (Resolved - April 22)**:
   - **Problem**: YOLOv8 incorrectly identified the camera wearer's own limbs or background torso-like artifacts as external bystanders.
   - **Solution**: Implemented a **Refined Anti-Wearer Heuristic** involving bottom-edge exclusion, a 0.50 confidence floor, and a 2-frame temporal consistency requirement.

9. **Robust Batch Filtering & Error Recovery (Resolved - April 23)**:
   - **Problem**: Batched acquisition occasionally missed videos if the CLI nested them unexpectedly, and transient network errors would crash multi-hour runs.
   - **Solution**: Implemented **Tethered UID Mapping** for recursive existence verification and wrapped the batch execution loop in granular exception handling to ensure the pipeline continues to the next set of UIDs after a failure.

10. **Dynamic Batch Sizing (Resolved - May 04)**:
    - **Problem**: The batch size for dataset downloads was hardcoded to `50` in `run_selective_download.py`, which failed to adapt to varying available SSD space, risking overflow or underutilization.
    - **Solution**: Implemented Query-Based Dynamic Sizing using `shutil.disk_usage()`. The system now automatically calculates the safe batch size based on available storage at the start of each cycle, clamped between 10 and 200.

11. **Parallel Frame Filtering Latency (Resolved - May 04)**:
    - **Problem**: The social presence filter processed videos sequentially frame-by-frame, underutilizing compute resources and causing bottlenecks during the YOLO inference phase.
    - **Solution**: Implemented Batched Frame-Level Parallelism in `social_presence.py`. The system now accumulates frames into an internal batch array and leverages YOLO's native GPU batch inference API, significantly improving throughput without multi-process complexity.

12. **Dataset Support Expansion Scope (Resolved - May 04)**:
    - **Problem**: EgoProceL and EPIC-KITCHENS-100 downloaders were stubbed out, leaving ambiguity on whether their absence was a bug or intentional.
    - **Solution**: Deferred non-Ego4D datasets. Documented that the focus will exclusively remain on Ego4D and Charades-Ego to reach statistical significance before attempting complex integrations with gated or nested meta-datasets.

13. **Wearer Detection Refinement (Hand Occlusion False Positives) (Resolved - May 04)**:
    - **Problem**: The geometric anti-wearer heuristic produced false positives when the wearer's hands occluded or overlapped with bystander bounding boxes, mistakenly identifying hands as bystanders.
    - **Solution**: Integrated MediaPipe Hands to generate accurate hand localization and filter out YOLO person detections that overlap heavily (>40%) with the detected hand bounding boxes.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: ByteTrack Tracker State Bleeding Across Videos
**Status**: ⚠️ Confirmed Unresolved — Verified in `social_presence.py` (line 107): the `model.track()` call uses `persist=True`, which tells ByteTrack to maintain internal tracker state across invocations. Because the `SocialPresenceDetector` instance is reused across multiple videos (via `StreamingFilter` in `filterer.py`), tracker IDs and trajectory history from Video A bleed into Video B. This can cause incorrect `person_id` assignments (e.g., a bystander in Video B inherits the ID of someone in Video A) and potentially corrupt the temporal consistency check (`min_consistency=2`), since stale tracked objects from the previous video may count toward the frame threshold of the new video.

**Option A (recommended)**: **Explicit Tracker Reset** — Call `self.model.predictor.trackers[0].reset()` (or `self._model = None` to force full reload) at the top of each `detect()` invocation before processing a new video. This ensures a clean tracker state per video.
  - *Pros*: Minimal code change (1-2 lines); directly fixes the root cause; no performance penalty since ByteTrack initialization is near-instant.
  - *Cons*: Relies on Ultralytics internal API (`predictor.trackers`), which may change across versions; requires a version-pinned dependency guard.

**Option B**: **Disable Persistence** — Change `persist=True` to `persist=False` in the `.track()` call. ByteTrack will reinitialize its state on every batch call within the same video.
  - *Pros*: Simplest possible fix; eliminates cross-video contamination entirely.
  - *Cons*: Breaks within-video temporal ID consistency, since ByteTrack will reset between every batch of 16 frames. Person IDs will not be stable across batches within a single video, degrading the quality of `person_id` in downstream layers that depend on it.

**Option C**: **Per-Video Detector Instantiation** — Create a new `SocialPresenceDetector` instance for each video in the `StreamingFilter.check_social_presence()` method, then call `unload()` after.
  - *Pros*: Guarantees complete isolation between videos; no reliance on internal APIs.
  - *Cons*: Reloads the YOLO model (~500ms) and MediaPipe Hands model for every single video, adding significant latency to batch acquisition of thousands of videos.

Your selection: Proceed with Option A

---

### Issue 2: Unused Imports in Selective Download Script
**Status**: ⚠️ Confirmed Unresolved — Verified in `run_selective_download.py` (line 8): `EpicKitchensDownloader` and `EgoProceLDownloader` are imported but never used in active code. They only appear inside triple-quoted comment blocks (lines 96-131, 133-139), which are string literals, not actual code. This causes lint warnings and adds unnecessary import-time overhead (each downloader class imports its own dependencies).

**Option A (recommended)**: **Remove Unused Imports** — Delete the `EpicKitchensDownloader` and `EgoProceLDownloader` imports from line 8. If they are needed in the future when those datasets are re-enabled, they can be re-added at that time.
  - *Pros*: Clean lint; reduces import-time side effects; trivial change.
  - *Cons*: None meaningful; easily reversible.

Your selection:Proceed with Option A

---

### Issue 3: SocialPresenceDetector Not Unloaded After Batch Filtering
**Status**: ⚠️ Confirmed Unresolved — Verified in `downloader.py` (lines 238-274): after the Ego4D batch filtering loop completes, the `StreamingFilter` (and its underlying `SocialPresenceDetector` holding a YOLO model + MediaPipe Hands model in GPU/MPS memory) is never explicitly unloaded. On the M4 Pro with 24GB unified memory, the YOLO model and MediaPipe graph remain resident across all subsequent batches for the lifetime of the process. Over long multi-batch runs (potentially hundreds of batches), this contributes to memory pressure and may cause MPS out-of-memory errors or system swapping.

**Option A (recommended)**: **Explicit Unload After Each Batch** — Call `self.filterer.detector.unload()` at the end of the filtering loop inside `Ego4DDownloader.download()` (after line 274). The lazy-loading property pattern already handles re-initialization on the next batch.
  - *Pros*: Frees ~300-500MB of MPS memory between batches; leverages the existing `unload()` method; no architectural change needed.
  - *Cons*: Adds ~1-2s of model reload time at the start of each subsequent batch.

**Option B**: **Unload Only on Low Memory** — Check available memory (via `psutil` or `os.sysconf`) before each batch, and only call `unload()` if memory is below a threshold (e.g., 4GB free).
  - *Pros*: Avoids unnecessary reload overhead when memory is abundant.
  - *Cons*: Adds a dependency on memory introspection; more complex; threshold tuning is hardware-specific.

Your selection: Proceed with Option A
