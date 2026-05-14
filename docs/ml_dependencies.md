# ML Dependencies Tracking

This document outlines the machine learning dependencies, models, and libraries required for the pipeline. It should be updated if any new models or library dependencies are added or removed.

## 1. Local Models
*Note: Heavy model weights should preferably be offloaded to the 2TB "Extreme SSD" if local internal storage is constrained. The videos themselves MUST be saved to "Extreme SSD".*

| Model Name | Purpose | Approximate Size | Recommended Save Location | License |
|---|---|---|---|---|
| **YOLOv8** (`yolov8n` or `yolov8s`) | Social Presence Filter (Bystander Box Detection) | ~6-25 MB | Internal SSD (`~/.cache/ultralytics`) or Local Project | AGPL-3.0 |
| **YOLOv8-pose** (`yolov8n-pose.pt`) | High-performance pose estimation (Motor Resonance) | ~6.5 MB | Internal SSD or Local Project | AGPL-3.0 |
| **L2CS-Net** (or CrossGaze) | 3D Gaze / Head Pose Estimation | ~200 MB | Internal SSD or Local Project | MIT |
| **Qwen2.5-VL** (via Ollama) | Primary VLM for visual classification + Task Climax VLM refinement (Node 02) | ~3-10 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **moondream** (via Ollama) | Fast VLM alternative (Lightweight) | ~1.6 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **Gemma 4** `gemma4:e4b` (via Ollama) | Fast multimodal model for action expectation & reasoning logic | ~2.5 GB (4-bit) | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **emotion2vec+** `large` (via FunASR) | Acoustic Prosody — primary SER model (9-class emotion + 768-dim embeddings) | ~600 MB | Internal SSD or Local Project | MIT |
| **SenseVoice** (`SenseVoiceSmall`) | Acoustic Prosody — supplementary Audio Event Detection (laughter, applause, crying) | ~500 MB | Internal SSD or Local Project | Apache-2.0 |
| **Depth Anything V2-Small** (`vits`) | Proxemic Kinematics (Relative Depth Delta Mapping) | ~100 MB | Internal SSD | Apache-2.0 |
| **Depth Anything V1-Large** (`vitl`) | Proxemic Kinematics — fallback if V2-Small quality is insufficient | ~1.3 GB | "Extreme SSD" | Apache-2.0 |
| **Metric3D v2** | Absolute Metric Depth (Alternative to Depth Anything) | ~1.5 GB | "Extreme SSD" | Apache-2.0 |
| **RTMPose** (via MMPose) | High-performance pose estimation (Deferred/Optional fallback) | ~100 MB (model only) | Internal SSD or Local Project | Apache-2.0 |

> [!NOTE]
> **All models in this table are license-clean** (MIT, Apache-2.0, or AGPL-3.0). No CC-BY-NC or non-commercial restrictions remain. The exported `social_metadata.parquet` can be distributed freely.

> [!IMPORTANT]
> **Task Climax Detection (Node 02)**: This pipeline does NOT use a dedicated highlight detection model. Instead, it uses a **hybrid Optical Flow + VLM refinement** approach: (1) OpenCV's `cv2.calcOpticalFlowFarneback` finds the kinetic peak within each task segment, (2) for slow/cognitive tasks, **Qwen2.5-VL** VLM refines the climax timestamp using the task label. This requires zero new model downloads.

> [!IMPORTANT]
> **Gemma 4 variant**: The documented "~2.5 GB" refers specifically to the `gemma4:e4b` (4B parameter, 4-bit quantized) variant, which remains the current production default. On the **Mac Studio (M4 Max, 64 GB unified memory)** target host, larger variants (`gemma4:26b` at ~15 GB or `gemma4:31b` at ~18 GB) are now physically loadable without OOM, but switching the production default requires re-validating the 03b prompt suite and re-tuning Resolved Issue #7's JSON-schema retries — see the Unresolved Issues section of `03b_reasonable_emotion_layer.md` for the upgrade trade-off matrix.

## 2. Core Libraries

| Library | Purpose | Approximate Install Size | Location | Notes |
|---|---|---|---|---|
| **Ollama** | Main engine for running Gemma 4, moondream, etc. | ~500 MB (binary + runtime) | System Install | |
| **FunASR** | Framework for emotion2vec+ inference | ~50 MB | Python virtual environment | `pip install -U funasr` |
| **Py-Feat** | SOTA Emotion inference on bystander faces | ~300-500 MB (incl. models) | Python virtual environment | ⚠️ CPU-only on macOS (no MPS/GPU support) |
| **MediaPipe** | Egocentric Hand Detection tracker | ~100 MB | Python virtual environment | |
| **Pandas** | Dehydrated CSV/Dataframe handling | ~100 MB | Python virtual environment | |
| **transformers** | Loading Hugging Face models like Depth Anything V2 | ~150 MB | Python virtual environment | |
| **huggingface_hub** | Interacting with Hugging Face exports and model caching | ~20 MB | Python virtual environment | |
| **PyTorch** (`torch`, `torchvision`) | Required for YOLO, L2CS-Net, Depth Anything, Py-Feat, etc. | ~2-3 GB | Python virtual environment | MPS backend works on Apple Silicon (validated on M4 Max / Mac Studio) |
| **OpenCV** (`opencv-python`) | Frame extraction, Optical Flow (Task Climax Detection), and UI video playback | ~150 MB | Python virtual environment | |
| **Librosa** | Audio feature extraction (Acoustic Prosody) | ~50 MB | Python virtual environment | |
| **SciPy** | Signal processing (Affirmation Gestures) | ~150 MB | Python virtual environment | |
| **MMPose** (+ mmengine, mmcv) | Framework for RTMPose (Deferred/Optional) | ~500 MB - 1 GB (total stack) | Python virtual environment | Requires source install; MMCV compilation historically failed on Apple Silicon (M4 Pro). Re-verify on M4 Max + current `mmcv-full` releases before re-introducing this dependency. |

## 3. System Tools

| Tool | Purpose | Location |
|---|---|---|
| **aws-cli** | Downloading Ego4D datasets from S3 buckets | System Install |
| **git** | Cloning EgoProceL and other GitHub repositories | System Install |
| **wget / curl** | General dataset ingestion and script downloads | System Install |
| **ffmpeg** | Audio extraction and video frame manipulation | System Install |

> **Storage Directive**: For all extremely large datasets (e.g., ego4d output directories, which can be Terabytes) and heavy VLM/LLM model blobs, map the storage volumes explicitly to the **2TB "Extreme SSD"** to preserve the internal Mac Studio SSD (M4 Max host) and handle the massive size of the videos. This directive is independent of host RAM — it applies equally to the Mac Studio host and any legacy Mac mini fallback.

---

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Default Model Tier Calibrated to 24 GB Mac Mini, Underutilizes 64 GB Mac Studio Budget
**Status**: ⚠️ Confirmed Unresolved — Every Ollama/Hugging Face model in the "Local Models" table was selected at the smallest viable tier specifically to fit within the prior 24 GB unified-memory ceiling: `gemma4:e4b` (4B/4-bit instead of 27B), `yolov8n-pose` (nano instead of x/large), Depth Anything **V2-Small** (~25M params instead of V1-Large/Metric3D v2), SAM **ViT-Base** (~93M params instead of ViT-Huge). With the new **Mac Studio (M4 Max, 64 GB unified memory)** host, the steady-state combined resident set of all 03 layers' larger variants would still fit within ~40 GB (leaving ~24 GB for OS + cached intermediates), unlocking measurable quality gains that the smaller tiers cannot match. The current defaults are now a memory-headroom regression on the new host.

**Option A (recommended)**: **Tier-Per-Host Configuration Layer** — Introduce a single `models_config.yaml` (or `models_config.py`) that maps each layer to a (small, medium, large) tier with explicit model identifiers and approximate sizes. Add a startup banner that prints the active tier + total estimated resident set. Default to "medium" on hosts with ≥48 GB unified memory (auto-detected via `psutil.virtual_memory().total`), "small" otherwise. Layer pipelines read their model ID from this config instead of hardcoding.
  - *Pros*: One configuration surface controls every layer's quality/memory trade-off; auto-detection prevents accidental OOM on smaller hosts; researchers can override per-experiment via env var.
  - *Cons*: Requires touching every 03x layer constructor (8 files); adds a startup dependency that must be validated against the existing fail-fast pattern; medium-tier weights need to be downloaded and validated for each upgraded model.

**Option B**: **Per-Layer Issues Filed in Each 03x Doc** — Keep `ml_dependencies.md` as a flat catalog and let each layer's own doc carry its own model-tier upgrade decision (e.g., 03b decides Gemma 4 27B vs 4B in its own unresolved issues block).
  - *Pros*: Preserves the current "layer is the unit of work" architecture; smaller PRs; each tier change can be benchmarked independently.
  - *Cons*: No single source of truth for the host's total resident-set budget; risk of two layers independently choosing "large" and collectively exceeding the 40 GB target.

**Option C**: **Stay Conservative; Only Upgrade Layers Where Quality Wins Are Measurable** — Run quality benchmarks (FER+ for emotion, COCO-keypoints for pose, NYUv2 for depth) before deciding which layers benefit, and leave the others on small-tier.
  - *Pros*: Avoids spending memory headroom on layers where the upgrade is imperceptible; matches the project's "Skip on failure / minimal-overhead" philosophy.
  - *Cons*: Slowest to implement; benchmark harnesses don't currently exist for several layers; quality wins from a larger model can be non-linear and only show up on edge cases that aren't in the benchmark.

Your selection: Proceed with Option A.

---

### Issue 2: RTMPose / MMPose Compilation Status on M4 Max Unverified
**Status**: ⚠️ Confirmed Unresolved — The MMPose row's "Notes" column documents that `mmcv` compilation historically failed on Apple Silicon M4 Pro. RTMPose remains in the Local Models table as "Deferred/Optional" and Layer 03f Resolved Issue #1/#6 substituted YOLOv8-pose explicitly because of the compilation failure. The underlying `mmcv-full` build chain has changed on PyPI between 2024 and 2026, and the M4 Max ships a different LLVM/Clang baseline than the M4 Pro it superseded. The compilation status is now stale — it may or may not still fail.

**Option A (recommended)**: **Re-Attempt `pip install mmcv-full` on M4 Max in a Clean venv** — Spin up a new Python 3.10 venv, attempt the install, log the outcome, and document the result inline in this doc. If it succeeds, file a separate issue in `03f_motor_resonance_layer.md` to A/B RTMPose-l vs YOLOv8n-pose on accuracy.
  - *Pros*: Cheap (one venv); produces a binary answer to the deferred RTMPose path; documents the result for future maintainers regardless of outcome.
  - *Cons*: A successful build does not guarantee inference quality wins — that still requires the 03f benchmark; if compilation still fails the time is purely diagnostic.

**Option B**: **Drop MMPose From the Table Entirely** — Treat the YOLOv8-pose substitution (03f Resolved Issue #1) as permanent and remove the row to stop misleading future contributors.
  - *Pros*: Eliminates dead documentation; simplifies the dependency surface.
  - *Cons*: Loses the explicit deferral context; future contributors who don't read 03f's history may re-propose RTMPose without knowing why it was rejected.

Your selection: Proceed with Option B.
