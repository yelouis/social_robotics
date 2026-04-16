# AI Task Breakdown: Dataset Acquisition (Charades-Ego + Original Charades)

## Objective
Download **both** the Charades-Ego egocentric videos **and** the original Charades third-person videos from AI2's public S3 bucket via `wget`. Build a unified clip manifest that maps every egocentric clip to its paired third-person video. No proprietary CLI tool or license key is required — all downloads are public.


## ⚠️ Critical Action for Your M4 Pro Mac mini (SSD Wear Prevention)
**DO NOT** run this data extraction or project pipeline directly on your Mac's internal SSD. The constant read/write cycles from `ffmpeg` and VLM inference will rapidly accelerate SSD wear.
- **Requirement**: Utilize a 2TB External NVMe SSD (e.g., Samsung T7 or SanDisk Extreme).
- **Action**: Configure your paths to set `OUTPUT_DIR` to that external drive (e.g., `/Volumes/Extreme SSD/charades_ego_data`).
- **Action**: Symlink your Python virtual environment (e.g., `venv` or `conda` env) to the external drive if possible to push all heavy OS I/O operations off the internal drive.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Environment Setup
- **Action**: Verify that `wget` (or `curl`), `tar`, and `unzip` are available in the shell. Install via `brew install wget` if missing.
- **Action**: Verify Python dependencies: `pandas`, `opencv-python` (for downstream modules). Install via `pip install pandas opencv-python`.
- **Note**: Charades-Ego is publicly hosted on AI2's S3 bucket. No credentials, API keys, or license agreements are required to download.

### Task 2: Download Script (`download_charades_ego.sh`)
- **Action**: Create a bash script `download_charades_ego.sh`.
- **Action**: Add the following commands to the script:
  ```bash
  OUTPUT_DIR="/Volumes/Extreme SSD/charades_ego_data"
  mkdir -p "$OUTPUT_DIR"/{annotations,ego_videos,tp_videos}

  # --- Guard: external drive check ---
  if [[ "$OUTPUT_DIR" != /Volumes/* ]]; then
    echo "ERROR: OUTPUT_DIR must be on an external drive (/Volumes/...)." && exit 1
  fi

  # 1. Download & extract annotations (CSV + class mapping, ~5MB)
  wget -nc -P "$OUTPUT_DIR" https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo.zip
  unzip -o "$OUTPUT_DIR/CharadesEgo.zip" -d "$OUTPUT_DIR/annotations"

  # 2. Download egocentric videos at 480p (~22GB)
  wget -nc -P "$OUTPUT_DIR" https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo_v1_480.tar
  tar -xf "$OUTPUT_DIR/CharadesEgo_v1_480.tar" -C "$OUTPUT_DIR/ego_videos"

  # 3. Download original Charades third-person videos at 480p (~22GB)
  #    These are the PAIRED third-person views needed by Modules 03 & 04.
  wget -nc -P "$OUTPUT_DIR" https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.tar
  tar -xf "$OUTPUT_DIR/Charades_v1_480.tar" -C "$OUTPUT_DIR/tp_videos"

  echo "Download complete. Total expected: ~44GB of video + annotations."
  ```
- **Constraint**: The `-nc` (no-clobber) flag allows safe re-runs without re-downloading files already present.

### Task 3: Paired Manifest Generation (`build_manifest.py`)
- **Action**: Create a Python script `build_manifest.py` that reads `CharadesEgo_v1_train.csv` (and `CharadesEgo_v1_test.csv`) using `pandas`.
- **Action**: Filter to only rows where `egocentric == 'Yes'`. For each ego clip, resolve the paired third-person video using the `charades_video` column.
- **Action**: Build a unified dataframe with columns: `ego_video_id`, `ego_video_path`, `tp_video_id`, `tp_video_path`, `actions`, `length`, `scene`, `script`.
  - `ego_video_path` = `$OUTPUT_DIR/ego_videos/{ego_video_id}.mp4`
  - `tp_video_path` = `$OUTPUT_DIR/tp_videos/{tp_video_id}.mp4`
- **Action**: Validate that both video files actually exist on disk. Log any missing pairs.
- **Action**: Persist as `paired_clips_manifest.csv` to `$OUTPUT_DIR/annotations/`.
- **Success Criteria**: Every row in the manifest has a confirmed ego + third-person video pair on disk, ready for downstream dual-view processing.

### Task 4: Ingestion Node (`ingest_node.py`) & Auto-Cleanup
- **Action**: Create an `ingest_node.py` script responsible for handling localized frame extraction (e.g., using `cv2` or `ffmpeg`) for downstream visual modules. Load video paths from `paired_clips_manifest.csv`.
- **Action**: The ingest node should accept a `view` argument (`"ego"` or `"tp"`) to extract frames from the correct video of the pair, since Module 03/04 need third-person frames while Module 02 needs no frames at all.
- **Constraint**: Implement strict **"Auto-Cleanup" logic**. Because of SSD wear constraints, immediately after frame metadata is generated and the video state is passed to RAM for the next module, you **must delete all extracted frames** from the external drive.
- **Success Criteria**: The system seamlessly extracts necessary visual frames dynamically, but never accumulates a backlog of `.jpg`/`.png` image artifacts taking up drive space.
