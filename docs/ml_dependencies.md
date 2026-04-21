# ML Dependencies Tracking

This document outlines the machine learning dependencies, models, and libraries required for the pipeline. It should be updated if any new models or library dependencies are added or removed.

## 1. Local Models
*Note: Heavy model weights should preferably be offloaded to the 2TB "Extreme SSD" if local internal storage is constrained. The videos themselves MUST be saved to "Extreme SSD".*

| Model Name | Purpose | Approximate Size | Recommended Save Location |
|---|---|---|---|
| **YOLOv8** (`yolov8n` or `yolov8s`) | Social Presence Filter (Bystander Box Detection) | ~6-25 MB | Internal SSD (`~/.cache/ultralytics`) or Local Project |
| **BayesianVSLNet** | Temporal Task Climax Identification | ~100-300 MB | Internal SSD or Local Project |
| **L2CS-Net** (or CrossGaze) | 3D Gaze / Head Pose Estimation | ~200 MB | Internal SSD or Local Project |
| **moondream** (via Ollama) | Fast VLM for visual classification | ~1-2 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) |
| **Qwen2.5-VL** (via Ollama) | VLM Alternative (Heavy) | ~3-10 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) |
| **Gemma 4** (via Ollama) | Fast LLM for action expectation & reasoning logic | ~2-5 GB | "Extreme SSD" (`OLLAMA_MODELS` external directory) |

## 2. Core Libraries

| Library | Purpose | Approximate Install Size | Location |
|---|---|---|---|
| **Ollama** | Main engine for running Gemma 4, moondream, etc. | ~500 MB (binary + runtime) | System Install |
| **Py-Feat** | SOTA Emotion inference on bystander faces | ~300-500 MB (incl. models) | Python virtual environment |
| **MediaPipe** | Egocentric Hand Detection tracker | ~100 MB | Python virtual environment |
| **Pandas** | Dehydrated CSV/Dataframe handling | ~100 MB | Python virtual environment |
| **huggingface_hub** | Interacting with Hugging Face exports | ~20 MB | Python virtual environment |
| **PyTorch** (`torch`, `torchvision`) | Required for YOLO, BayesianVSLNet, L2CS-Net, Py-Feat | ~2-3 GB | Python virtual environment |
| **OpenCV** (`opencv-python`) | Frame extraction & UI video playback | ~150 MB | Python virtual environment |

> **Storage Directive**: For all extremely large datasets (e.g., ego4d output directories, which can be Terabytes) and heavy VLM model blobs, map the storage volumes explicitly to the **2TB "Extreme SSD"** to preserve the internal Mac mini M4 Pro storage and handle the massive size of the videos.
