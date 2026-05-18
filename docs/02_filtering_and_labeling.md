# AI Task Breakdown: Video Filtering and Labeling

## Objective
To ensure computational efficiency and relevance, not all downloaded videos make it to the advanced Social Feature Layers. This module performs two critical filtration passes to weed out unusable data from first-person (POV) perspectives.

---

## 1. The Social Presence Filter
Since our project relies on investigating social-affective interactions, a video of a person entirely alone is useless. 

**Criteria**: 
The video **must** contain more than one person. Since this is an egocentric/POV video, the person whose perspective is being shown (the camera wearer) *does not count* towards this total. There must be at least one other visible human actor present in the frame. 

To achieve this, the system uses a **Multi-Modal Verification Pipeline**:
- **Pose Keypoint Validation (YOLOv8-Pose)**: The system utilizes `yolov8n-pose.pt`. Detections are only kept if they contain at least one head keypoint (nose, eye, or ear) AND at least one shoulder keypoint above a minimum confidence threshold. This effectively eliminates false positives from the wearer's own outstretched limbs or vaguely humanoid equipment.
- **VLM Early-Exit Verification Gate**: Pose-confirmed candidate frames are sent to a fast Vision-Language Model (e.g., `moondream`) to answer a simple multi-person YES/NO prompt. The pipeline tracks consecutive VLM confirmations. Once a video hits the required consistency threshold, the filter short-circuits (early-exit) and marks the video as kept.
- **Geometric Anti-Wearer Heuristic**: As a baseline defense, the system drops detections touching the bottom edge of the frame without a head/torso, or full-height detections starting at the absolute top (`y1=0`).
- **Temporal Consistency**: A video is only "KEPT" if social presence is detected in at least **2 sampled frames** to filter out momentary glitches.

> [!IMPORTANT]
> **Integration Note**: To optimize storage on the "Extreme SSD", this filter is integrated directly into the **Dataset Acquisition** module via a **batched UID strategy**. Videos are downloaded in small groups (e.g., 50), filtered immediately, and non-social videos are purged. A persistent `processed_uids.json` tracks completion so that purged videos are never re-downloaded.

**Task Requirements**:
- Evaluate the raw videos from the initial manifest.
- Drop any video where zero exterior individuals are found.
- **Persist bounding boxes**: For every frame sampled during detection, the bounding box coordinates and timestamps of detected `person` instances **must be written to the output manifest**, not just the keep/drop decision. Downstream Social Layers (03a Attention, 03b Reasonable Emotion, etc.) depend on this data to isolate and crop bystander regions without re-running detection.
- *Potential Implementation Idea*: Utilize standard lightweight Object Detection (e.g. YOLOv8) on heavily downsampled frames to check for bounding boxes labeled `person`.

---

## 2. Contextual Task Labeling
A social interaction has fundamentally different mechanics if the POV person is reading a book versus cooking a meal with a knife. For downstream layers to accurately process features (like "Flinch" physics), we must define the task context. 

Because unstructured Ego4D videos can be lengthy, a single video may contain **multiple distinct tasks**.

**Criteria**:
We must actively generate task labels spanning the duration of the clip by extracting them from the **Ego4D Metadata**. If no clear tasks are identified in the metadata (e.g., descriptions like "Idling" or "No clear task") for the entire duration, the video is dropped.

**Task Requirements**:
- **Metadata Extraction**: Parse the Ego4D `scenarios` and `descriptions` (from `ego4d.json` and benchmark-specific annotation files) to classify the sequential actions.
- **Taxonomy Mapping**: Map the natural language descriptions and scenario tags to a structured array of `identified_tasks`.
- **Velocity Mapping**: Since metadata doesn't explicitly provide "velocity," we use a predefined lookup table to map task labels to physical velocities:
  - **Fast / Instinctual**: slipping, dropping, throwing, impact.
  - **Medium**: standard manual labor, walking, talking.
  - **Slow / Cognitive**: reading, writing, thinking, solving.
- *Drop Rule*: If the metadata lists "Idling," "No clear task," or "Ambiguous activity" (or is missing task-level annotations) for the entire video duration, it must be discarded.

---

## 3. Temporal Task Climax Identification
Knowing *what* the task is isn't enough; downstream affective layers need to know exactly *when* it happens so they can measure bystander reactions accurately.

**Criteria**:
Identify the exact timestamp or narrow temporal window where the "climax" or core action of each task occurs (e.g., the exact moment the dropped glass shatters, or the exact moment the basketball leaves the hands).

**Task Requirements**:
- **Hybrid Optical Flow + VLM Climax Detection**: A two-stage approach that requires zero additional model downloads:
  - **Stage 1 — Optical Flow Variance Peak**: Use OpenCV's `cv2.calcOpticalFlowFarneback` to compute dense optical flow. To optimize throughput and precision, this is implemented as a **Two-Pass Hierarchical Detection**: a coarse pass at ~5 FPS identifies the rough peak window within the task's temporal bounds, followed by a native 30 FPS dense pass within ±1s of the coarse peak for full temporal precision. The frame with the highest flow magnitude is the kinetic `task_climax_sec`. This is highly accurate for abrupt physical actions (dropping, slipping, throwing).
  - **Stage 2 — VLM Refinement (slow/cognitive tasks)**: For tasks classified as `"slow"` velocity (e.g., solving a puzzle, reading a sign), sample 3-5 candidate frames around the optical flow peak and use the existing **moondream** VLM to score each frame's proximity to the action climax, using the `task_label` as context.
- **Dynamic Action Velocity & Delay Buffers**: Human reactions vary wildly depending on the abruptness of a task. The system maps the metadata-derived `task_label` to a "Velocity" class. We use this to compute a dynamic `task_reaction_window_sec`:
  - **Fast / Instinctual** (e.g., slipping, dropping an item): Requires a short buffer. Climax + `[0.5s to 2.0s]`.
  - **Medium** (e.g., throwing a ball, standard interaction): Climax + `[1.0s to 3.0s]`.
  - **Slow / Cognitive** (e.g., solving a puzzle, reading a sign): Climax + `[2.0s to 6.0s]`.

---

## Output
This module outputs a `filtered_manifest.json` that the downstream **Social Feature Extraction Layers** will consume. Only clips that survived both the Social Presence check and Contextual Task extraction make it onto this ledger.

### `filtered_manifest.json` Schema Definition
All downstream Social Feature Layers depend on this exact contract. Any additions to this schema must be **additive only**—never remove or rename existing keys.

```json
{
  "video_id": "ego4d_clip_10293",
  "source_dataset": "ego4d",
  "video_path": "/Volumes/Extreme SSD/raw_videos/ego4d_clip_10293.mp4",
  "fps": 30.0,
  "duration_sec": 45.0,
  "identified_tasks": [
    {
      "task_id": "t_01",
      "task_label": "Chopping vegetables",
      "task_confidence": 0.92,
      "task_velocity": "medium",
      "task_start_sec": 0.0,
      "task_end_sec": 45.0,
      "task_temporal_metadata": {
        "task_climax_sec": 5.2,
        "task_reaction_window_sec": [6.2, 8.2],
        "climax_extraction_method": "optical_flow_peak+vlm_refinement",
        "optical_flow_peak_magnitude": 42.8,
        "vlm_climax_confidence": 0.85
      }
    },
    {
      "task_id": "t_02",
      "task_label": "Dropping a plate",
      "task_confidence": 0.88,
      "task_velocity": "fast",
      "task_start_sec": 0.0,
      "task_end_sec": 45.0,
      "task_temporal_metadata": {
        "task_climax_sec": 24.1,
        "task_reaction_window_sec": [24.6, 26.1],
        "climax_extraction_method": "optical_flow_peak",
        "optical_flow_peak_magnitude": 78.3
      }
    }
  ],
  "bystander_detections": [
    {
      "person_id": 0,
      "timestamps_sec": [0.0, 0.5, 1.0, 1.5],
      "bounding_boxes": [
        [120, 80, 340, 510],
        [125, 78, 345, 512],
        [130, 82, 350, 515],
        [128, 80, 348, 513]
      ],
      "detection_confidence": [0.94, 0.91, 0.93, 0.90]
    }
  ],
  "hand_detections": [
    {
      "timestamp_sec": 0.0,
      "hand_boxes": [
        [410, 520, 560, 670],
        [610, 540, 760, 690]
      ]
    },
    {
      "timestamp_sec": 0.5,
      "hand_boxes": [
        [415, 522, 565, 672]
      ]
    }
  ]
}
```

> **Co-indexing rule**: Within each `bystander_detections` entry, `timestamps_sec[i]`, `bounding_boxes[i]`, and `detection_confidence[i]` are strictly co-indexed. The i-th element of each array describes the same sampled frame.

> **`hand_detections` contract**: One entry per sampled frame that contained MediaPipe-detected wearer hands. `timestamp_sec` matches a sampled frame timestamp from `bystander_detections`, and `hand_boxes` is a list of `[x1, y1, x2, y2]` integer pixel coordinates. Frames with no detected hands are simply omitted. Layer 03a (Attention) consumes this field for occlusion suppression — never remove or rename it.

## Verification & Validation Check
To ensure the filtering mechanisms are robust and correct:
- **Singular Video Test**: Execute the filter module against a single chosen `.mp4` video that is visually verified to have multiple interacting people. Inspect the generated JSON output to confirm the `bystander_detections` bounding boxes accurately surround the actors.
- **Batch Test**: Run the node over a subset folder (e.g., 50 videos). Aggregate the output to check distribution metrics (e.g., % dropped due to no social presence, % dropped due to idling). Assert that every entry in the `filtered_manifest.json` has `identified_tasks` with non-empty attributes, effectively proving the system scaled accurately without unhandled `None` crashes. Furthermore, utilize the **Mac Studio (M4 Max, 64 GB unified memory)** environment effectively by ensuring the batched VLM calls stay within the unified-memory budget. With 64 GB available, both YOLOv8 and Qwen2.5-VL co-reside, so the pipeline runs a single-pass interleaved architecture on hosts with ≥48 GB unified memory (see Resolved Issue #19), falling back to the two-pass architecture from Resolved Issue #11 on smaller hosts such as the 24 GB Mac mini.

---

## 🚀 Implementation Status

The filtering and labeling pipeline is fully operational within the `src/dataset_acquisition` and `src/filtering_and_labeling` modules. 
- **Social Presence Filter**: Successfully implemented using YOLOv8-Pose (`yolov8n-pose.pt`) with head+shoulder keypoint validation, the geometric Anti-Wearer Heuristic, and a fast-VLM early-exit verification gate (see Resolved Issue #22).
- **Contextual Task Labeling**: Shifted from VLM-based labeling to **Ego4D Metadata Extraction** using `ego4d.json`. This provides ground-truth task contexts and significantly reduces computational overhead.
- **Temporal Climax Identification**: Implemented a hybrid two-stage approach using dense optical flow (`cv2.calcOpticalFlowFarneback`) and VLM-based refinement for slow tasks.

## 🧪 Resolved Issues & Implementation Refinements

1. **Pipeline Resumability & Error Isolation (Resolved - April 22)**:
   - **Problem**: Long batch runs were fragile; a single error in one video would crash the entire process.
   - **Solution**: Implemented incremental state saving (write-after-each-video) and isolated per-video errors to `02_filter_errors.json`.

2. **Unified Memory Constraints & Dual-Architecture Filtering (Resolved - May 02 / May 13)**:
   - **Problem**: The pipeline originally interleaved YOLO and VLM calls per video, which forced both models to reside in memory, risking OOM kills on 24GB unified memory hosts. However, enforcing a strict two-pass architecture (YOLO pass on all videos -> unload -> VLM pass) wasted SSD read bandwidth on 64GB hosts.
   - **Solution**: The pipeline now implements a memory-gated dual architecture. On hosts with ≥48 GB unified memory, it uses a **single-pass interleaved architecture** to process both YOLO and VLM steps in one read. On hosts with <48 GB, it automatically falls back to the **two-pass architecture**, safely isolating model memory footprints.

3. **VLM Model Tiering by Host Capacity (Resolved - May 13)**:
   - **Problem**: Stage-2 climax refinement was pinned to Qwen2.5-VL 3B to fit on 24GB hosts, leaving accuracy on the table for 64GB hosts.
   - **Solution**: Promoted the 7B tier to the default on 64GB hosts. The pipeline integrates with the central tier-per-host registry to auto-select `qwen2.5vl:7b` (≥48 GB) or `qwen2.5vl:3b` (<48 GB).

4. **MediaPipe Graph Initialization Crash on Apple Silicon (Resolved - May 14)**:
   - **Problem**: `MediaPipe` 0.10.11 failed due to `protobuf` incompatibilities on Apple Silicon.
   - **Solution**: Downgraded to `0.10.9` to restore stability for the wearer occlusion suppression layer.


## ⚠️ Unresolved Issues & Suggestions

### Issue 1: YOLO-Pose Fires on People Displayed on TVs / Monitors / Photos
**Status**: ⚠️ Confirmed Unresolved — Discovered during the May 16 100-video E2E run. The highest-scoring "best" video in the run was `002ad105-bd9a-4858-953e-54e88dc7587e` (Ego4D, `social_presence_score = 346.67`, 48 bystander tracks, 440 frames-with-detections). Spot-checking the frames at the top three detection timestamps (`462s`, `700s`, `1045s`) shows the camera wearer is alone in a hotel room watching YouTube on a TV — the 48 "bystanders" are all faces and torsos rendered on the TV screen (YouTube thumbnails, a soccer-coach broadcast, a vlog face). `SocialPresenceDetector._has_head_and_shoulder()` (`src/shared/social_presence.py:101`) is satisfied because the on-screen people genuinely have visible head + shoulder keypoints from YOLO-pose's perspective, and the geometric Anti-Wearer Heuristic (`src/shared/social_presence.py:265-269`) only rejects bottom-edge / top-edge limb stubs. There is currently no "is this a real 3D body or a pixel-rendered display surface?" gate, so any egocentric video that lingers on a TV / phone screen / framed photograph / poster will pass the filter with arbitrarily high scores and pollute downstream Layer 03a / 03f training data with non-social signal.

**Option A (recommended)**: **Screen-Detection Pre-Filter via YOLO General Class Set** — Run a second YOLO-bbox pass (or extend the existing one) on the same sampled frames using COCO class `tv` (class id 62) and `laptop` (63). For every confirmed `bystander` person box, compute IoU against the union of `tv` / `laptop` boxes in the same frame; drop any person box whose IoU ≥ 0.5 with a screen box.
  - *Pros*: Cheap (one extra class set, same model invocation), local fix to `social_presence.py`, no new model dependencies. Eliminates the dominant on-TV false-positive class observed in the E2E run. Composes cleanly with the existing pose-keypoint gate.
  - *Cons*: Misses framed photographs and small phone screens that COCO doesn't reliably detect. A person standing in front of a TV (their real torso below the screen, the broadcast face above) can be incorrectly dropped if IoU is too lax — requires tuning.

**Option B**: **Re-enable VLM Verification with a Screen-Aware Prompt** — Turn `SAF_VLM_VERIFY_SOCIAL=1` (default) back on and tighten the Moondream prompt to ask *"two or more **real physical** people, not people shown on TVs, phones, photographs, or paintings"*. This was already partially built (Resolved Issue #22) but the current YES/NO prompt does not call out screen content.
  - *Pros*: Generalizes beyond TVs to phones, photos, posters, magazines, mirrors. Zero new model weights — Moondream is already loaded.
  - *Cons*: Adds 1–10 s of VLM latency per candidate frame (the reason `SAF_VLM_VERIFY_SOCIAL=0` was used in this E2E run). Moondream's screen-vs-real discrimination has not been quantitatively validated against Ego4D.

**Option C**: **Depth-Anything Sanity Check (Borrow From Layer 03d)** — For each candidate bystander box, sample the median depth inside the box vs. the median depth of the surrounding TV-shaped rectangle; reject the candidate if the inside-vs-outside depth delta is ≤ a small threshold (i.e. the "person" is on the same flat plane as the wall behind them).
  - *Pros*: Generalizes to all flat-surface displays without needing a fixed class list. Reuses the Depth-Anything model that the production tier already loads for Layer 03d.
  - *Cons*: Requires sequencing Layer 02 after Layer 03d's depth pass, or duplicating depth inference inside Layer 02. ~300 ms / frame extra latency at the `medium` tier; closer to 2 s at `large`.

Your selection: Proceed with Option B.

---

### Issue 2: Non-Ego4D Datasets Silently Dropped at Metadata Stage
**Status**: ⚠️ Confirmed Unresolved — Verified in `FilteringPipeline.contextual_task_labeling` (`src/filtering_and_labeling/pipeline.py:287-330`) and `process_video_vlm_pass` (`pipeline.py:204-236`). During the May 16 E2E run, 58 of 80 Charades-Ego videos passed the YOLO social-presence filter but were then dropped from `filtered_manifest.json` because `contextual_task_labeling()` looks the video up in `self.metadata` (loaded from `EGO4D_METADATA_PATH`) and returns an empty list when the UID is not an Ego4D `video_uid` — Charades-Ego IDs like `00V9V` / `04DU1EGO` are never present in the Ego4D metadata. With no `identified_tasks`, `process_video_vlm_pass` returns `None` and the entry is never appended to `filtered_manifest.json`. Resolved Issue #9 ("Ego4D Metadata Integration", May 02) replaced the prior VLM-based labeling path with a metadata-only path, which effectively reduced the four "Core Supported Datasets" (Ego4D, Charades-Ego, EPIC-KITCHENS, EgoProceL) listed in `01_dataset_acquisition.md` down to one — Ego4D — for any video that should appear in the dehydrated export. The README and the project overview still advertise the full four-dataset set, so this gap is silent at the docs level too.

**Option A (recommended)**: **Per-Dataset Labeling Strategy Registry** — Add a small `_LABELER_BY_DATASET` table keyed on `entry["dataset"]` that maps Ego4D → existing metadata extractor, Charades-Ego → its action-class CSV (`Charades_Ego_v1_train.csv` / `_test.csv` ship with the dataset), EPIC-KITCHENS → its `EPIC_100_train.csv` verb/noun classes, and EgoProceL → its YAML procedure annotations. `contextual_task_labeling` dispatches on `entry["dataset"]` instead of always hitting `self.metadata`.
  - *Pros*: Restores the four advertised datasets without re-introducing the slow VLM labeling path. Keeps the `task_confidence = 1.0` ground-truth marker for all four sources, since each native annotation file is authoritative for that dataset.
  - *Cons*: Three new annotation files must be downloaded and indexed on first run. Velocity heuristic (`_get_velocity_from_label`) needs new keyword mappings for kitchen / procedural verbs.

**Option B**: **Fallback to Lightweight VLM Labeling When Metadata Is Missing** — Keep the metadata-only fast path for Ego4D, but when `video_id not in self.metadata`, dispatch to a one-shot Moondream call asking for a 1-line task label instead of returning `[]`.
  - *Pros*: Restores all non-Ego4D datasets with zero new dataset files. Naturally extends to any future ego dataset without code changes.
  - *Cons*: Reverts to a probabilistic labeler for ~50 % of the pipeline output (Charades-Ego alone is the largest source). `task_confidence` semantics get muddied — would need a per-task `task_label_source` field ("metadata" vs "vlm") to keep ground-truth consumers honest. Re-introduces the latency Resolved Issue #9 was designed to eliminate.

Your selection: Do not process Charades-Ego for now. Just do Ego4D.

---

### Issue 3: Filtering VLM Pinned to Ollama Tag `qwen2.5vl:7b` That Is Not Locally Installed
**Status**: ⚠️ Confirmed Unresolved — Verified at `src/models_config.py:73-78` (`"filtering_vlm": {"medium": ("qwen2.5vl:7b", ...)}`) and at `FilteringPipeline.__init__` (`pipeline.py:46`). On the production Mac Studio host (auto-detected tier `medium`), the pipeline resolves `self.vlm_model = "qwen2.5vl:7b"`, but `ollama list` shows only `qwen2.5vl:latest` (which is the same 8.3 B-parameter Qwen2.5-VL weights, just under the default tag). Every Stage-2 VLM climax-refinement call (slow-velocity tasks only) then fails with `Error: model 'qwen2.5vl:7b' not found`, gets caught by the broad `except Exception` in `temporal_climax_identification` (`pipeline.py:488-490`) which prints `"VLM Refinement failed: ..."` and silently falls back to `optical_flow_peak`, never setting `vlm_climax_confidence`. The same tag-mismatch pattern exists for `layer_03b_ollama` (`gemma4:26b` registered, only `gemma4:latest` installed) and will fail the same way once Layer 03b runs end-to-end.

**Option A (recommended)**: **Pull the Pinned Tags Once and Drop the Latent Bug** — Run `ollama pull qwen2.5vl:7b && ollama pull gemma4:26b` on the production host (and document this as a one-time setup step in `ml_dependencies.md`). The tagged versions are byte-identical to `:latest` today, so this is a metadata-only pull; future tag drift then becomes the only failure mode.
  - *Pros*: Zero code changes. Restores the documented Stage-2 climax-refinement path immediately. Makes `models_config.py` accurate again.
  - *Cons*: Adds an out-of-band manual setup step that's easy to forget on a new host. Doesn't address future tag mismatches.

**Option B**: **Make `get_model()` Resolve Ollama Tags Lazily Against `ollama list`** — Have `get_model("filtering_vlm")` query `ollama list` (cached for the process lifetime) and substitute `:latest` when the exact tag is missing.
  - *Pros*: Bulletproof against tag drift. Same code works on any developer machine without manual `ollama pull` choreography.
  - *Cons*: Adds an Ollama runtime dependency at the module level (currently `models_config.py` has zero runtime deps). Hides genuine tag mismatches that might be intentional (e.g. an A/B test against a specific tag).

**Option C**: **Tighten `except Exception` to Log Loudly Instead of Silently Falling Back** — Keep the current tag, but in `temporal_climax_identification`'s except block, route the error through `self.log_error(video_id, e)` so it lands in `02_filter_errors.json` instead of just printing to stdout. Pair with a startup-time `ollama list` sanity check that emits a hard-stop banner if `self.vlm_model` is missing.
  - *Pros*: Surfaces the bug instead of letting it silently degrade output quality. Independent of A/B above and can be combined.
  - *Cons*: Doesn't actually fix the broken VLM path — only makes its failure visible.

Your selection: Proceed with Option B.

---

### Issue 4: Optical-Flow Climax Extraction Throughput on Long Ego4D Videos
**Status**: ⚠️ Confirmed Unresolved — Verified during the May 16 E2E run. With `SAF_VLM_VERIFY_SOCIAL=0` and the full pipeline (`FilteringPipeline.run` → `_run_single_pass` → `temporal_climax_identification`), the first Ego4D video in the registry (`000786a7-…`, 415 s duration, 215 MB, 1 scenario in Ego4D metadata) had not finished its Task Refinement step after **3+ minutes** of wall-clock time, while the comparable YOLO-only social filter on the same video finished in 50.1 s. `temporal_climax_identification` (`pipeline.py:345-427`) issues a `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)` for every coarse-pass frame at ~5 FPS over the whole task range, plus another `cap.set` per dense-pass frame at native FPS in the ±1 s window around the coarse peak. On Ego4D `full_scale` MP4 files served from the Extreme SSD, each per-frame seek issues a fresh decoder reset; OpenCV does not retain GOP state across `cap.set` calls, so a 415 s video at 30 FPS triggers ~2 100 seeks for the coarse pass and ~60 more for the dense pass — every one of which round-trips through the H.264 decoder. The slowest videos in the run (e.g. `002c3b5c-…`, 2 758 s / 46 minutes / 625 MB) would project to >15 minutes of climax extraction per video. With Ego4D videos comprising 100 % of pipeline-eligible content (per Issue #2), this throughput limit caps the realistic dataset size at hundreds of videos per host-day, not thousands.

**Option A (recommended)**: **Sequential Decode With Frame-Counter, Drop `cap.set` Entirely** — Restructure `temporal_climax_identification` to mirror the pattern already used in `SocialPresenceDetector.detect()` (`shared/social_presence.py:206-217`): one `cap.read()` loop walking frames forward, with a `current_frame_idx` counter that gates whether each decoded frame contributes to the coarse-pass or dense-pass flow computation. The dense pass becomes a second forward walk starting from the coarse-peak frame's position rather than a seek.
  - *Pros*: Eliminates all per-frame seeks; reduces decoder cost from O(frames × IDR distance) to O(frames). Already proven on the same SSD by the YOLO filter path. No model dependencies.
  - *Cons*: Coarse pass becomes strictly sequential, so it cannot terminate early on the climax peak — must read every frame in the task range. For short tasks this is a wash; for hour-long Ego4D videos it's a net win because the existing `cap.set` path already touches every frame anyway.

**Option B**: **Down-Sample with FFmpeg `select=isnan(prev_selected_t)+gte(t-prev_selected_t,0.2)` Pipe** — Replace OpenCV's seek-and-decode with an external FFmpeg invocation that emits frames at the desired coarse-pass rate over a pipe, which OpenCV consumes via `cv2.VideoCapture` against a `ffmpeg:` pipe URL.
  - *Pros*: FFmpeg's `select` filter respects GOP boundaries and is dramatically faster on long videos than `cap.set` round-trips.
  - *Cons*: Adds an FFmpeg subprocess dependency for the climax stage. Pipe-mode VideoCapture is finicky on macOS and harder to debug than direct file reads.

**Option C**: **Defer Climax Extraction to a Layer-03 Co-Resident Pass** — Move `temporal_climax_identification` out of Layer 02 and into a thin shared utility that runs alongside Layer 03d (Depth) or 03g (Shared Reality), both of which already do a single sequential decode of every frame for their own features. Layer 02 would emit `identified_tasks` without `task_temporal_metadata`, and the climax window would be filled in by whichever 03 layer ran first.
  - *Pros*: Zero extra disk reads — the climax flow is computed as a free side product of an already-required pass. Reduces Layer 02 to a pure metadata-and-filter stage.
  - *Cons*: Cross-layer schema contract change — `filtered_manifest.json` becomes a partial record until at least one Layer 03 has run, breaking the current "Layer 02 output is consumable in isolation" invariant.

Your selection: Proceed with Option C.
