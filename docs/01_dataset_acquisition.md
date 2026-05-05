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

10. **Progress Bar Overflow (Resolved - April 27)**:
    - **Problem**: The `tqdm` progress bar incorrectly showed 300% completion because update calls for disabled datasets (EPIC, EgoProceL) were still executing outside of comment blocks.
    - **Solution**: Moved all `pbar.update(1)` calls for inactive tasks inside their respective triple-quoted comment blocks.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Dynamic Batch Sizing
**Status**: ⚠️ Confirmed Unresolved — The batch size is hardcoded to `50` in `run_selective_download.py`. This means the system cannot adapt to varying available SSD space. If the SSD has 500GB free, 50 videos per batch is overly conservative; if it has only 10GB free, 50 videos could overflow mid-batch.

**Option A (recommended)**: **Query-Based Dynamic Sizing** — At the start of each batch cycle, query `shutil.disk_usage("/Volumes/Extreme SSD")` to determine available space. Divide available bytes by the average per-video size (derived from the previous batch's mean file size, or a conservative default of 500MB) to compute the max safe batch size, clamped to a `[10, 200]` range.
  - *Pros*: Fully adaptive; prevents overflow and maximizes throughput; zero external dependencies.
  - *Cons*: Relies on accurate average file size estimation; first batch uses a conservative default until calibrated.

**Option B**: **Tiered Static Sizing** — Replace the single `50` constant with a lookup table: `{">500GB": 200, ">100GB": 100, ">50GB": 50, "<50GB": 20}`, checked once at startup.
  - *Pros*: Simpler to implement; no per-batch overhead.
  - *Cons*: Cannot react to space changes mid-run; requires manual tuning of tier boundaries.

Your selection: Proceed with Option A

---

### Issue 2: Parallel Frame Filtering
**Status**: ⚠️ Confirmed Unresolved — The social presence filter processes videos sequentially in a `for` loop. On the Mac mini M4 Pro (10 CPU cores, 24GB RAM), this underutilizes available compute during the YOLO inference phase.

**Option A (recommended)**: **`concurrent.futures.ProcessPoolExecutor` with Worker Pool** — Wrap the per-video filter call in a process pool (e.g., `max_workers=4`). Each worker loads its own YOLO model instance. The main process collects results and writes to the manifest atomically.
  - *Pros*: 3-4x throughput improvement on 10-core M4 Pro; each worker is memory-isolated; straightforward implementation with stdlib.
  - *Cons*: 4 YOLO instances × ~200MB ≈ 800MB additional memory; requires careful locking on the shared `processed_uids.json` and manifest files.

**Option B**: **`multiprocessing.Pool` with Shared Model** — Use `torch.multiprocessing` with `fork` start method to share the YOLO model's read-only weights across workers.
  - *Pros*: Lower memory overhead (shared weights); potentially faster startup.
  - *Cons*: `fork` is unreliable on macOS (Apple discourages it); MPS tensors cannot be shared across forked processes; higher crash risk.

**Option C**: **Batched Frame-Level Parallelism** — Keep sequential video processing but batch-submit sampled frames to YOLO using its native batch inference API (`model.predict(frames, batch=N)`).
  - *Pros*: No multi-process complexity; leverages YOLO's internal GPU batching; minimal code change.
  - *Cons*: Limited speedup (only parallelizes within a single video, not across videos); memory spike on large frame batches.

Your selection: Proceed with Option C

---

### Issue 3: Dataset Support Expansion (EgoProceL & EPIC-KITCHENS-100)
**Status**: ⚠️ Confirmed Unresolved — The downloaders for EgoProceL and EPIC-KITCHENS-100 are stubbed out (commented/disabled) in `run_selective_download.py`. Only Ego4D and Charades-Ego are operational.

**Option A (recommended)**: **Deferred Until Ego4D Saturation** — Continue with Ego4D and Charades-Ego until the filtered dataset reaches a statistically significant size (e.g., >5,000 social clips). Document this as a deliberate prioritization, not a bug. Enable EPIC-KITCHENS-100 next (it has structured task annotations similar to Ego4D), then EgoProceL last (meta-dataset, requires recursive source resolution).
  - *Pros*: Focused engineering effort; avoids splitting attention across 4 dataset integrations; Ego4D alone provides sufficient diversity for initial research.
  - *Cons*: Delays cross-dataset generalization validation.

**Option B**: **Parallel Dataset Integration Sprint** — Implement all four dataset downloaders simultaneously with a unified `DatasetAdapter` interface.
  - *Pros*: Maximum dataset diversity immediately; identifies cross-dataset schema issues early.
  - *Cons*: High engineering cost; EPIC-KITCHENS-100 requires university-gated access; EgoProceL's nested source resolution is complex.

Your selection: Let's only do Ego4D for now.

---

### Issue 4: Wearer Detection Refinement (Hand Occlusion False Positives)
**Status**: ⚠️ Confirmed Unresolved — The geometric anti-wearer heuristic (edge exclusion, confidence floor, temporal consistency) is effective but still produces occasional false positives when the wearer's hands occlude or overlap with bystander bounding boxes.

**Option A (recommended)**: **Ego-Hand Segmentation Mask via EgoHOS** — Integrate the [EgoHOS](https://github.com/owenzlz/EgoHOS) egocentric hand-object segmentation model to generate per-frame hand masks. Subtract these masks from the YOLO `person` detections before social presence evaluation. EgoHOS is specifically trained on Ego4D data and runs on PyTorch MPS.
  - *Pros*: Purpose-built for egocentric hand removal; high precision on Ego4D; published research backing.
  - *Cons*: Adds ~300MB model weight; increases per-frame inference time by ~50ms; requires integration testing on the M4 Pro memory budget.

**Option B**: **IoU Overlap Suppression** — If a YOLO `person` detection has >40% IoU with the bottom-center "wearer zone" (the lower-third, center-half of the frame), suppress it. This extends the current geometric heuristic without adding a new model.
  - *Pros*: Zero additional dependencies; trivial to implement; no memory cost.
  - *Cons*: Still purely geometric; fails when hands are raised or extended into the upper frame; lower precision than learned segmentation.

Your selection: Proceed with Option A.
