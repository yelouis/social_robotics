# Workflow: Local Model Environment Setup
**Goal**: Configure the M4 Pro for hybrid VLA inference (Ollama + MLX-VLM).

### Step 1: Core Tooling
- [ ] **Action**: Ensure `brew` is installed.
- [ ] **Action**: Install Ollama via `brew install --cask ollama`.
- [ ] **Action**: Launch Ollama and set `OLLAMA_KEEP_ALIVE="-1"` to prevent model eviction.

### Step 2: Logic Layer (Gemma 4)
- [ ] **Action**: Pull the reasoning-heavy variant: `ollama pull gemma4:12b`.
- [ ] **Action**: Pull the high-speed utility variant: `ollama pull gemma4:e4b`.
- [ ] **Verification**: Run a "Thinking Mode" test: `ollama run gemma4:12b "<|think|> Test Rs calculation logic."`

### Step 3: Vision Layer (MLX-VLM)
- [ ] **Action**: Create a virtual env on the external SSD: `/Volumes/Extreme SSD/saf_env`.
- [ ] **Action**: `pip install mlx-vlm huggingface_hub`.
- [ ] **Action**: Download Qwen 2.5-VL to the external drive:
  `mlx_vlm.download --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit --path "/Volumes/Extreme SSD/models/"`

### Step 4: Final Verification
- [ ] **Action**: Check total memory pressure.
- [ ] **Success Criteria**: Agent must confirm models are reachable and external drive I/O is active.