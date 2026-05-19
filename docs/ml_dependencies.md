# ML Dependencies Tracking

This document outlines the machine learning dependencies, models, and libraries required for the pipeline. It should be updated if any new models or library dependencies are added or removed.

## 1. Local Models
*Note: Heavy model weights should preferably be offloaded to the 2TB "Extreme SSD" if local internal storage is constrained. The videos themselves MUST be saved to "Extreme SSD".*

| Model Name | Purpose | Approximate Size | Recommended Save Location | License |
|---|---|---|---|---|
| **YOLOv8** (`yolov8n` or `yolov8s`) | Social Presence Filter (Bystander Box Detection) | ~6-25 MB | Internal SSD (`~/.cache/ultralytics`) or Local Project | AGPL-3.0 |
| **YOLOv8-pose** (`yolov8n-pose.pt`) | High-performance pose estimation (Motor Resonance) | ~6.5 MB | Internal SSD or Local Project | AGPL-3.0 |
| **MediaPipe HandLandmarker** (`hand_landmarker.task`, float16) | Egocentric wearer hand detection for occlusion suppression (Layer 02 → Layer 03a). MediaPipe Tasks API bundle (replaces the legacy `mp.solutions.hands` namespace removed in mediapipe>=0.10.30). | ~7 MB | Local Project (`models/mediapipe/`) | Apache-2.0 |
| **L2CS-Net** (or CrossGaze) | 3D Gaze / Head Pose Estimation | ~200 MB | Internal SSD or Local Project | MIT |
| **Qwen2.5-VL** (via Ollama) | Primary VLM for visual classification + Task Climax VLM refinement (Node 02) | ~3-10 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **moondream** (via Ollama) | Fast VLM alternative (Lightweight) | ~1.6 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **Gemma 4** `gemma4:26b` (via Ollama) | Multi-step action expectation & reasoning logic (Layer 03b) | ~15 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **emotion2vec+** `large` (via FunASR) | Acoustic Prosody — primary SER model (9-class emotion + 768-dim embeddings) | ~600 MB | Internal SSD or Local Project | MIT |
| **SenseVoice** (`SenseVoiceSmall`) | Acoustic Prosody — supplementary Audio Event Detection (laughter, applause, crying) | ~500 MB | Internal SSD or Local Project | Apache-2.0 |
| **Depth Anything V2-Small** (`vits`) | Proxemic Kinematics (Relative Depth Delta Mapping) | ~100 MB | Internal SSD | Apache-2.0 |
| **Depth Anything V1-Large** (`vitl`) | Proxemic Kinematics — fallback if V2-Small quality is insufficient | ~1.3 GB | "Extreme SSD" | Apache-2.0 |
| **Metric3D v2** | Absolute Metric Depth (Alternative to Depth Anything) | ~1.5 GB | "Extreme SSD" | Apache-2.0 |

> [!NOTE]
> **All models in this table are license-clean** (MIT, Apache-2.0, or AGPL-3.0). No CC-BY-NC or non-commercial restrictions remain. The exported `social_metadata.parquet` can be distributed freely.

> [!IMPORTANT]
> **Tier-per-host selection**: The specific model id loaded for each row is resolved at runtime by `src/models_config.py`. The active tier auto-detects from `psutil.virtual_memory().total` (≥48 GB → `medium`; otherwise → `small`) and can be pinned via `SR_MODEL_TIER=small|medium|large`. The rows above describe the **`medium`-tier defaults** that the Mac Studio (M4 Max, 64 GB) actually loads in production; the `small` tier substitutes `gemma4:e4b`, `yolov8n-pose`, Depth Anything V2-Small, SAM ViT-Base, `qwen2.5vl:3b`, and `emotion2vec_plus_base` for the 24 GB Mac mini fallback. The startup banner prints the active tier + estimated resident set on the first import of `models_config`.

> [!IMPORTANT]
> **Task Climax Detection (Node 02)**: This pipeline does NOT use a dedicated highlight detection model. Instead, it uses a **hybrid Optical Flow + VLM refinement** approach: (1) OpenCV's `cv2.calcOpticalFlowFarneback` finds the kinetic peak within each task segment, (2) for slow/cognitive tasks, **Qwen2.5-VL** VLM refines the climax timestamp using the task label. This requires zero new model downloads.

> [!IMPORTANT]
> **Gemma 4 variant**: Layer 03b's `OLLAMA_MODEL` defaults to `gemma4:26b` (27B-class, ~15 GB resident) on the **Mac Studio (M4 Max, 64 GB unified memory)** target host — the multi-step structured-output reasoning chain measurably benefits from the larger model, and the 64 GB budget has ample headroom. The prior `gemma4:e4b` (4B parameter, 4-bit quantized, ~2.5 GB) default was chosen for the legacy 24 GB Mac mini M4 Pro budget; it is no longer wired as a fallback. See Resolved Issue #16 in `03b_reasonable_emotion_layer.md` for the migration rationale and the pending live E2E re-validation step.

> [!NOTE]
> **emotion2vec+ variant**: The "Acoustic Prosody" layer (03c) pins the SER model to the `large` variant (`iic/emotion2vec_plus_large`, ~600 MB, 9-class) — the current production default. FunASR also ships `iic/emotion2vec_plus_base` (~300 MB, lower quality) and `iic/emotion2vec_plus_seed` (~2 GB, transformer-XL backbone, stronger embeddings on long cross-speaker utterances). The `seed` variant's ~2 GB footprint was disqualifying on the legacy 24 GB Mac mini M4 Pro once L2CS, Py-Feat, and Depth Anything V2 + SAM were simultaneously resident, but on the **Mac Studio (M4 Max, 64 GB unified memory)** target host it is loadable with no displacement of any other 03 layer. It remains a **documented candidate, not a production change** — promoting `seed` would require A/B'ing 9-class dominant-emotion accuracy against human labels and re-checking that the SenseVoice low-confidence gate (`< 0.6`) does not need recalibration. See `03c_acoustic_prosody_layer.md` Resolved Issue #19 for the retention rationale.

## 2. Core Libraries

| Library | Purpose | Approximate Install Size | Location | Notes |
|---|---|---|---|---|
| **Ollama** | Main engine for running Gemma 4, moondream, etc. | ~500 MB (binary + runtime) | System Install | |
| **FunASR** | Framework for emotion2vec+ inference | ~50 MB | Python virtual environment | `pip install -U funasr` |
| **HSEmotion-PyTorch** | SOTA Emotion inference on bystander faces (Layer 03b) | ~50-100 MB (incl. `enet_b2_8` weights) | Python virtual environment | Native MPS acceleration on Apple Silicon; `pip install hsemotion`. Replaced Py-Feat, which was CPU-only on macOS. |
| **MediaPipe** | Egocentric Hand Detection tracker | ~100 MB | Python virtual environment | |
| **Pandas** | Dehydrated CSV/Dataframe handling | ~100 MB | Python virtual environment | |
| **transformers** | Loading Hugging Face models like Depth Anything V2 | ~150 MB | Python virtual environment | |
| **huggingface_hub** | Interacting with Hugging Face exports and model caching | ~20 MB | Python virtual environment | |
| **PyTorch** (`torch`, `torchvision`) | Required for YOLO, L2CS-Net, Depth Anything, HSEmotion-PyTorch, etc. | ~2-3 GB | Python virtual environment | MPS backend works on Apple Silicon (validated on M4 Max / Mac Studio) |
| **OpenCV** (`opencv-python`) | Frame extraction, Optical Flow (Task Climax Detection), and UI video playback | ~150 MB | Python virtual environment | |
| **Librosa** | Audio feature extraction (Acoustic Prosody) | ~50 MB | Python virtual environment | |
| **SciPy** | Signal processing (Affirmation Gestures) | ~150 MB | Python virtual environment | |

## 3. System Tools

| Tool | Purpose | Location |
|---|---|---|
| **aws-cli** | Downloading Ego4D datasets from S3 buckets | System Install |
| **git** | Cloning EgoProceL and other GitHub repositories | System Install |
| **wget / curl** | General dataset ingestion and script downloads | System Install |
| **ffmpeg** | Audio extraction and video frame manipulation | System Install |

> **Storage Directive**: For all extremely large datasets (e.g., ego4d output directories, which can be Terabytes) and heavy VLM/LLM model blobs, map the storage volumes explicitly to the **2TB "Extreme SSD"** to preserve the internal Mac Studio SSD (M4 Max host) and handle the massive size of the videos. This directive is independent of host RAM — it applies equally to the Mac Studio host and any legacy Mac mini fallback.

