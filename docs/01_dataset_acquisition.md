# AI Task Breakdown: Dataset Acquisition

## Objective
Download and stage large-scale first-person POV (egocentric) videos—such as Ego4d or other relevant datasets—onto the local system for ingestion into the pipeline.

## ⚠️ Critical Action for Hardware Management
Video datasets are inherently massive (often spanning terabytes). 
**DO NOT** run these data pipelines directly on strict internal SSD setups if space/wear are concerns. 
- Ensure your `OUTPUT_DIR` maps out to the external **2TB SSD called "Extreme SSD"**. This is explicitly designated to handle the large size of the Ego4D and other video datasets.
- **Hardware Profile**: As we are running a **Mac mini M4 Pro with 24GB RAM**, be strictly mindful of memory ceilings when unpacking or indexing these massive datasets in Python. Rely on streaming processors or chunked extraction.

## Target Video Paradigm
The system is built to parse First-Person POV interactions. While the pipeline is designed to be relatively dataset-agnostic, the target standard is **Ego4D**. 
These videos represent unstructured, daily interaction. 

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
