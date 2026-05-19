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

> [!IMPORTANT]
> **Execution Location**: Climax extraction is no longer executed by Layer 02 itself. Per Resolved Issue #8 below, Layer 02 emits `task_temporal_metadata = {}`, and the first Layer 03 pipeline to consume the manifest invokes `shared.climax_extraction.populate_climax_for_manifest()` to fill in the optical-flow peak, dynamic reaction window, and optional VLM refinement in place. Subsequent Layer 03 pipelines find the metadata cached and skip the optical-flow pass entirely.

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
- **Batch Test**: Run the node over a subset folder (e.g., 50 videos). Aggregate the output to check distribution metrics (e.g., % dropped due to no social presence, % dropped due to idling). Assert that every entry in the `filtered_manifest.json` has `identified_tasks` with non-empty attributes, effectively proving the system scaled accurately without unhandled `None` crashes. Furthermore, utilize the **Mac Studio (M4 Max, 64 GB unified memory)** environment effectively by ensuring the batched VLM calls stay within the unified-memory budget. With 64 GB available, both YOLOv8 and Qwen2.5-VL co-reside, so the pipeline runs a single-pass interleaved architecture on hosts with ≥48 GB unified memory, falling back to the two-pass architecture on smaller hosts such as the 24 GB Mac mini (see Resolved Issue #2).

---

## 🚀 Implementation Status

The filtering and labeling pipeline is fully operational within the `src/dataset_acquisition` and `src/filtering_and_labeling` modules.
- **Social Presence Filter**: YOLOv8-Pose (`yolov8n-pose.pt`) with head+shoulder keypoint validation, the geometric Anti-Wearer Heuristic, and a four-stage Moondream verification cascade — screen-aware multi-person prompt (Resolved Issue #5), once-per-video side-by-side stereo gate (Resolved Issue #10), per-detection wearer-chin gaze gate for bottom-20% bboxes (Resolved Issue #11), and MediaPipe Tasks API hand-landmarker for occlusion suppression (Resolved Issue #9). Sample rate defaults to 1 frame per 3 seconds (Resolved Issue #11).
- **Contextual Task Labeling**: Ego4D Metadata Extraction from `ego4d.json` is the only labeling path; non-Ego4D entries are skipped at intake until per-dataset labelers are written (Resolved Issue #6). Obvious-solo Ego4D videos (scenarios like `"Reading books"`, `"Watching tv"`, etc.) are skipped before YOLO via the `_SOLO_SCENARIO_SUBSTRINGS` allowlist (Resolved Issue #12).
- **Temporal Climax Identification**: Implementation lives in `src/shared/climax_extraction.py` and is invoked by the first Layer 03 pipeline to consume the manifest, not by Layer 02 itself (Resolved Issue #8).

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

5. **Screen-Aware VLM Verification Prompt (Resolved - May 17)**:
   - **Problem**: During the May 16 100-video E2E run, YOLO-pose's head+shoulder gate (`SocialPresenceDetector._has_head_and_shoulder()`, `src/shared/social_presence.py:101`) confidently fired on people rendered on TVs, monitors, and framed photographs because the on-screen faces genuinely satisfy the keypoint criterion. The geometric Anti-Wearer Heuristic only rejects bottom-edge / top-edge limb stubs, leaving no gate for "real 3D body vs pixel-rendered display surface". The single highest-scoring video in the run (Ego4D `002ad105-…`) was the camera wearer alone in a hotel room watching YouTube; all 48 "bystander" tracks were faces on the TV. The pre-existing Moondream YES/NO prompt in `_vlm_confirms_multiple_people` asked only "two or more distinct people" without excluding screens, so even with `SAF_VLM_VERIFY_SOCIAL=1` the VLM would have confirmed.
   - **Solution**: Tightened the Moondream prompt in `src/shared/social_presence.py` to explicitly require "two or more real, physical people actually present in the scene" and to enumerate the negative classes ("Do NOT count people shown on TVs, monitors, phone screens, photographs, posters, paintings, magazines, or reflections in mirrors"). The pipeline default (`SAF_VLM_VERIFY_SOCIAL=1` in `FilteringPipeline.__init__`) was already on — this resolution fixes the prompt content so that switching it back on actually filters out the on-TV class. Moondream remains the lightweight VLM (no new model weights); per-candidate-frame latency is unchanged from the original verification gate.

6. **Non-Ego4D Datasets Skipped at Intake Pending Per-Dataset Labelers (Resolved - May 17)**:
   - **Problem**: `FilteringPipeline.contextual_task_labeling` looked every video up in `self.metadata` (loaded from `EGO4D_METADATA_PATH`) and returned `[]` whenever the UID was not an Ego4D `video_uid`. In the May 16 E2E run, 58 of 80 Charades-Ego videos cleared the YOLO social filter and were then silently dropped from `filtered_manifest.json` because `process_video_vlm_pass` returns `None` on empty `identified_tasks`. The acquisition layer continued advertising four "Core Supported Datasets" (Ego4D, Charades-Ego, EPIC-KITCHENS, EgoProceL), masking the gap.
   - **Solution**: Added a `_SUPPORTED_DATASETS = ("ego4d",)` allowlist plus an `_is_supported_dataset()` gate at the top of `_run_single_pass` and `_run_two_pass` in `src/filtering_and_labeling/pipeline.py`. Non-Ego4D entries are now skipped before YOLO runs (avoiding the wasted social-filter pass) with an explicit per-video log line naming the unsupported dataset. The metadata-only ground-truth path (`task_confidence = 1.0`) is preserved for Ego4D; per-dataset labelers for Charades-Ego / EPIC-KITCHENS / EgoProceL are tracked separately for when their annotation parsers are written.

7. **Lazy Ollama Tag Resolution Against Local `ollama list` (Resolved - May 17)**:
   - **Problem**: The tier registry pinned the Stage-2 climax-refinement VLM to `qwen2.5vl:7b` on `medium`/`large` hosts (`src/models_config.py`), but the production Mac Studio had only `qwen2.5vl:latest` installed locally. Every slow-velocity VLM refinement call failed with `Error: model 'qwen2.5vl:7b' not found`, was swallowed by the broad `except Exception` in the climax stage, and silently fell back to `optical_flow_peak` without `vlm_climax_confidence`. The same latent bug existed for `layer_03b_ollama` (`gemma4:26b` registered, only `gemma4:latest` installed) and would fail identically once 03b ran end-to-end.
   - **Solution**: Added `_resolve_ollama_tag()` in `src/models_config.py`. `get_model()` now checks whether the configured tag (`qwen2.5vl:*`, `gemma4:*`, `moondream:*`) is present in `ollama list`; if the exact tag is missing but `<base>:latest` is installed, it substitutes `:latest` and emits a one-shot stderr warning so the fallback is visible without hiding genuine missing-model errors. The local-tag set is cached for the process lifetime so the `ollama.list()` call happens at most once per pipeline run. Hosts without the `ollama` Python client gracefully degrade to the original behavior (no fallback, identical to before).

8. **Climax Extraction Deferred to Layer 03 Co-Resident Pass (Resolved - May 17)**:
   - **Problem**: `temporal_climax_identification` issued a `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)` for every coarse-pass frame at ~5 FPS over each task range plus a dense pass at native FPS in the ±1 s window around the coarse peak. On Ego4D `full_scale` MP4 files served from the Extreme SSD, each seek triggered a fresh H.264 decoder reset because OpenCV does not retain GOP state across `cap.set` calls. In the May 16 E2E run the first 415 s Ego4D video had not finished its Task Refinement step after 3+ minutes; the 2 758 s videos in the registry projected to >15 minutes of climax extraction per video. With non-Ego4D datasets already skipped (Resolved Issue #6), this throughput limit alone capped the realistic dataset size at hundreds of videos per host-day.
   - **Solution**: Removed the optical-flow pass from `FilteringPipeline.process_video_vlm_pass` and extracted the two-pass coarse/dense flow + slow-task VLM refinement logic into `src/shared/climax_extraction.py`. Layer 02 now emits `identified_tasks` with `task_temporal_metadata = {}`; any Layer 03 pipeline that needs the reaction window calls `shared.climax_extraction.populate_climax_for_manifest(manifest_path, vlm_model=…)` (or `compute_task_climax_for_video()` when it already holds an open `cv2.VideoCapture`) before its own feature extraction. The shared utility is idempotent — tasks with non-empty metadata are skipped — so only the first Layer 03 to run for a given manifest pays the cost, and subsequent layers consume the cached metadata. The cross-layer schema contract change is acknowledged: `filtered_manifest.json` is a partial record until at least one Layer 03 has been invoked.

9. **MediaPipe `solutions` Namespace Removed in 0.10.30+ (Resolved - May 18)**:
   - **Problem**: `SocialPresenceDetector.mp_hands` (`src/shared/social_presence.py:66`) referenced `mp.solutions.hands.Hands(...)`, but the installed `mediapipe==0.10.35` no longer exposes the `solutions` namespace at the top-level module (only `tasks`, `Image`, `ImageFormat`). The first VLM-positive frame in every Ego4D video raised `AttributeError: module 'mediapipe' has no attribute 'solutions'`, which was caught by `SocialPresenceDetector.detect()`'s outer except and returned empty bystander lists. Because `FilteringPipeline.process_video_vlm_pass` hard-wires `return_hands=True`, the production Node 02 pipeline crashed on every single Ego4D video in the May 17 100-video E2E run instead of falling back. The standalone scratch runner only succeeded after forcing `return_hands=False`.
   - **Solution**: Migrated `SocialPresenceDetector` to the MediaPipe Tasks API. Replaced the `mp.solutions.hands.Hands(...)` constructor with `mp.tasks.vision.HandLandmarker.create_from_options(...)` using `RunningMode.IMAGE` (which avoids the monotonic-timestamp constraint of `VIDEO` mode so a single detector instance can be reused across videos). Downloaded the `hand_landmarker.task` float16 bundle (~7 MB) once into `models/mediapipe/hand_landmarker.task` and registered it in `src/models_config.py` under the new `social_presence_hand_landmarker` key; the path is resolved against `_PROJECT_ROOT` so MediaPipe's `model_asset_path` is robust to caller CWD. The per-frame call site (`social_presence.py:301`) wraps the BGR frame as `mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)` and iterates `hand_results.hand_landmarks` (list-of-lists) instead of the legacy `multi_hand_landmarks.landmark`. The normalized landmark coordinate shape is unchanged (`.x`/`.y` in `[0, 1]`), so `_extract_hand_boxes`'s pixel-scaling math required only the structural rename. Verified end-to-end on a Charades-Ego sample: hand bounding boxes round-trip into `filtered_manifest.hand_detections` with no AttributeError on `return_hands=True`.

10. **Stereo-Format Egocentric Captures Confirmed via Moondream Gate (Resolved - May 18)**:
    - **Problem**: The highest-scoring video in the May 17 Ego4D run (`0219271c-7641-4e17-a00c-81c42e0d4779`, score `843.32`, 984 frame-detections, 7 "bystanders") was a side-by-side stereoscopic capture of a single person crafting alone — the 2880×1080 frame rendered the same scene twice across left and right halves. YOLO-pose detected the wearer's own arms + torso in each stereo half and treated them as two separate persons, and the tightened Moondream multi-person prompt from Resolved Issue #5 also confirmed because the duplicate humans are not on a screen or in a photograph — they are literally rendered side-by-side. This single video accounted for >93% of the entire run's positive-detection mass.
    - **Solution**: Added `_vlm_confirms_side_by_side_stereo()` to `SocialPresenceDetector` and gated it once per video at the first pose-positive frame. The Moondream prompt asks explicitly: *"Is this image a side-by-side stereoscopic capture where the left half and the right half show the same scene rendered twice (as if from a stereo camera rig)?"*. On `YES`, the detector logs a "Stereo gate rejected" line and short-circuits the entire video (returns `([], [])` for `return_hands` callers, `False` for `fast_mode`). To minimize wall-time tax, the stereo check uses a `stereo_checked` local flag so each video pays at most one extra Moondream call regardless of how many pose-positive frames follow. Refactored the existing `_vlm_confirms_multiple_people` helper into a shared `_vlm_ask_yes_no(frame_bgr, prompt, default_on_error)` so both stereo and gaze-direction checks (Resolved Issue #11) share the same image-resize + tempfile + ollama-chat plumbing. `default_on_error=False` for the stereo prompt — a VLM outage falls back to the existing multi-person gate rather than letting one rare class drop every bottom-edge bystander.

11. **Wearer's Own Chin Passing YOLO-Pose + Moondream Multi-Person Gate (Resolved - May 18)**:
    - **Problem**: The lowest-scoring passing video in the May 17 run, `0030b1e9-c6a6-4809-a495-8d45791f9775` (single person trimming a hedge alone with a head-mounted GoPro), produced 18 frame-detections across 3 "bystanders" because the wearer's own chin / lower face peeked into the bottom edge as they looked down. YOLO-pose detected the lower-jaw + collar region as a separate person (conf 0.61–0.88), the keypoint head+shoulder gate passed because mouth/jaw plus collar count as head+shoulder, and the Moondream multi-person prompt confirmed because there *is* a human face visible — Moondream couldn't tell from the cropped patch that the face it sees belongs to the camera wearer. The Geometric Anti-Wearer Heuristic in `social_presence.py` only rejects bbox bottom-edges touching the literal bottom pixel row plus full-height `y1=0` ghosts, both of which miss the partial-chin signature.
    - **Solution**: Added `_vlm_face_oriented_upward(frame_bgr, bbox)` to `SocialPresenceDetector` that crops the candidate bbox (with a 5% pad in both axes for context) and asks Moondream: *"Is this person's face oriented upward toward the camera, as if you are looking up at them from below? This pose is the signature of an egocentric camera operator's own chin or lower face peeking into the frame from underneath."*. The gate fires only when (a) `vlm_verify` is on and (b) the bbox `y2 > 0.80 * img_h` (bottom 20% of the frame). On `YES`, the detection is skipped via `continue` before it reaches `frame_detections`, so the per-person aggregation in `FilteringPipeline.social_presence_filter` never sees it. The companion sampling-rate change drops the social-presence sample rate from 1 FPS to 1 frame per 3 seconds (`sample_rate_fps=1/3.0`) across `SocialPresenceDetector.detect()`, `FilteringPipeline.social_presence_filter`, and `StreamingFilter.check_social_presence`. At 1 FPS the per-video VLM budget was dominated by re-checking the same wearer chin in nearly every sampled frame; sampling at 1/3 FPS reclaims that budget while still clearing the `min_consistency=2` gate on real multi-person videos. `default_on_error=False` so a VLM outage falls back to the YOLO-pose + multi-person Moondream gate rather than silently dropping every bottom-edge real bystander.

12. **Ego4D Scenario-Tag Prefilter Skips Obvious Solo Activities Before YOLO (Resolved - May 18)**:
    - **Problem**: In the May 17 100-video Ego4D E2E run, only 3 / 100 videos passed the social-presence filter (and 2 of the 3 were false positives — see Resolved Issues #10 and #11). Independently verified that of the 97 rejected videos, the 3 longest-running (e.g. `011ee98a-…`, 9314 s duration, 795 s YOLO+VLM time) were true-negative solo activity. At ~108 s/video on a Mac Studio, a 10k social-interaction corpus needed ~10 host-days to drain the solo-dominated Ego4D `full_scale` slice — the YOLO+VLM cost was being burned on videos the metadata could already classify as solo. The originally-proposed Option A spec'd a `num_persons` scenario tag, but inspecting `ego4d.json` (9821 videos) revealed Ego4D has no explicit `num_persons` field; the only person-count signal is the human-readable `scenarios` list (e.g. `"Reading books"`, `"Talking with family members"`).
    - **Solution**: Added a `_SOLO_SCENARIO_SUBSTRINGS` allowlist and an `_is_likely_solo_by_metadata(video_id)` gate on `FilteringPipeline`. The allowlist is intentionally conservative — only 16 substrings covering activities that are overwhelmingly solo (`"crafting"`, `"reading books"`, `"watching tv"`, `"listening to music"`, `"sleeping"`, `"daily hygiene"`, `"fixing pc"`, `"on a screen"`, `"play with cellphone"`, `"playing games / video games"`, `"assembling a puzzle"`, `"writing on book"`, `"reviewing flash cards"`, `"crossword puzzle"`, `"potting plants"`, `"talking on the phone"`). A video is filtered out only when ALL its `scenarios` strings match a solo keyword (case-insensitive substring); any mixed or ambiguous tag (cooking, eating, walking on street, commuting) falls through to YOLO+VLM. The gate runs after the `_is_supported_dataset` check and before YOLO in both `_run_single_pass` and `_run_two_pass`, logging an explicit `"Skipping {video_id}: Ego4D scenarios flagged as solo"` line per skipped entry. On the full Ego4D registry, 1786 / 9821 videos (18%) match the solo allowlist and skip the multimodal pass entirely. The deviation from the original Option A spec (scenario substrings instead of a literal `num_persons` field) is structural — Ego4D's metadata schema doesn't carry person counts — but the user intent is satisfied: skip the easy-solo majority before paying YOLO+VLM cost. Future per-dataset labelers (Resolved Issue #6) can add their own solo-scenario allowlists keyed off the `dataset` field.
