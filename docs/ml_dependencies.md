# ML Dependencies Tracking

This document outlines the machine learning dependencies, models, and libraries required for the pipeline. It should be updated if any new models or library dependencies are added or removed.

## 1. Local Models
*Note: Heavy model weights should preferably be offloaded to the 2TB "Extreme SSD" if local internal storage is constrained. The videos themselves MUST be saved to "Extreme SSD".*

| Model Name | Purpose | Approximate Size | Recommended Save Location | License |
|---|---|---|---|---|
| **YOLOv8** (`yolov8n` or `yolov8s`) | Social Presence Filter (Bystander Box Detection) | ~6-25 MB | Internal SSD (`~/.cache/ultralytics`) or Local Project | AGPL-3.0 |
| **L2CS-Net** (or CrossGaze) | 3D Gaze / Head Pose Estimation | ~200 MB | Internal SSD or Local Project | MIT |
| **moondream** (via Ollama) | Fast VLM for visual classification + Task Climax VLM refinement (Node 02) | ~1.6 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **Qwen2.5-VL** (via Ollama) | VLM Alternative (Heavy) | ~3-10 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **Gemma 4** `gemma4:e4b` (via Ollama) | Fast multimodal model for action expectation & reasoning logic | ~2.5 GB (4-bit) | "Extreme SSD" (`OLLAMA_MODELS` external directory) | Apache-2.0 |
| **emotion2vec+** `large` (via FunASR) | Acoustic Prosody — primary SER model (9-class emotion + 768-dim embeddings) | ~600 MB | Internal SSD or Local Project | MIT |
| **SenseVoice** (`SenseVoiceSmall`) | Acoustic Prosody — supplementary Audio Event Detection (laughter, applause, crying) | ~500 MB | Internal SSD or Local Project | Apache-2.0 |
| **Depth Anything V2-Small** (`vits`) | Proxemic Kinematics (Relative Depth Delta Mapping) | ~100 MB | Internal SSD | Apache-2.0 |
| **Depth Anything V1-Large** (`vitl`) | Proxemic Kinematics — fallback if V2-Small quality is insufficient | ~1.3 GB | "Extreme SSD" | Apache-2.0 |
| **Metric3D v2** | Absolute Metric Depth (Alternative to Depth Anything) | ~1.5 GB | "Extreme SSD" | Apache-2.0 |
| **RTMPose** (via MMPose) | High-performance pose estimation (Motor Resonance) | ~100 MB (model only) | Internal SSD or Local Project | Apache-2.0 |

> [!NOTE]
> **All models in this table are license-clean** (MIT, Apache-2.0, or AGPL-3.0). No CC-BY-NC or non-commercial restrictions remain. The exported `social_metadata.parquet` can be distributed freely.

> [!IMPORTANT]
> **Task Climax Detection (Node 02)**: This pipeline does NOT use a dedicated highlight detection model. Instead, it uses a **hybrid Optical Flow + VLM refinement** approach: (1) OpenCV's `cv2.calcOpticalFlowFarneback` finds the kinetic peak within each task segment, (2) for slow/cognitive tasks, `moondream` VLM refines the climax timestamp using the task label. This requires zero new model downloads.

> [!IMPORTANT]
> **Gemma 4 variant**: The documented "~2.5 GB" refers specifically to the `gemma4:e4b` (4B parameter, 4-bit quantized) variant. Do NOT pull larger variants (`gemma4:26b` at ~15 GB or `gemma4:31b` at ~18 GB) as they will consume the entire 24GB unified memory of the Mac mini M4 Pro and cause system instability.

## 2. Core Libraries

| Library | Purpose | Approximate Install Size | Location | Notes |
|---|---|---|---|---|
| **Ollama** | Main engine for running Gemma 4, moondream, etc. | ~500 MB (binary + runtime) | System Install | |
| **FunASR** | Framework for emotion2vec+ inference | ~50 MB | Python virtual environment | `pip install -U funasr` |
| **Py-Feat** | SOTA Emotion inference on bystander faces | ~300-500 MB (incl. models) | Python virtual environment | ⚠️ CPU-only on macOS (no MPS/GPU support) |
| **MediaPipe** | Egocentric Hand Detection tracker | ~100 MB | Python virtual environment | |
| **Pandas** | Dehydrated CSV/Dataframe handling | ~100 MB | Python virtual environment | |
| **huggingface_hub** | Interacting with Hugging Face exports | ~20 MB | Python virtual environment | |
| **PyTorch** (`torch`, `torchvision`) | Required for YOLO, L2CS-Net, Depth Anything, Py-Feat, etc. | ~2-3 GB | Python virtual environment | MPS backend works on M4 Pro |
| **OpenCV** (`opencv-python`) | Frame extraction, Optical Flow (Task Climax Detection), and UI video playback | ~150 MB | Python virtual environment | |
| **Librosa** | Audio feature extraction (Acoustic Prosody) | ~50 MB | Python virtual environment | |
| **SciPy** | Signal processing (Affirmation Gestures) | ~150 MB | Python virtual environment | |
| **MMPose** (+ mmengine, mmcv) | Framework for RTMPose and other pose estimators | ~500 MB - 1 GB (total stack) | Python virtual environment | Requires source install; MMCV compilation may need flags on Apple Silicon |

## 3. System Tools

| Tool | Purpose | Location |
|---|---|---|
| **aws-cli** | Downloading Ego4D datasets from S3 buckets | System Install |
| **wget / curl** | General dataset ingestion and script downloads | System Install |
| **ffmpeg** | Audio extraction and video frame manipulation | System Install |

> **Storage Directive**: For all extremely large datasets (e.g., ego4d output directories, which can be Terabytes) and heavy VLM/LLM model blobs, map the storage volumes explicitly to the **2TB "Extreme SSD"** to preserve the internal Mac mini M4 Pro storage and handle the massive size of the videos.
