#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/Extreme SSD/social_robotics"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
export TMPDIR="/Volumes/Extreme SSD/tmp"
export SR_NO_MODEL_BANNER=1

echo "[pipeline $(date)] === Starting holoassist_pool pipeline ==="

# 1. memguard status
echo "[pipeline $(date)] Stage 1: checking memory guard status..."
./venv/bin/python -m shared.memguard --status

# 2. Features: siglip-b16-224
echo "[pipeline $(date)] Stage 2: extracting siglip-b16-224 features..."
mkdir -p "$ROOT/features/holoassist_pool/siglip-b16-224"
bash tools/run_supervised.sh "$ROOT/features/holoassist_pool/siglip-b16-224/progress.json" \
    ./venv/bin/python -m features.extract --dataset holoassist_pool --encoder siglip-b16-224

# 3. Features: e2v-plus-large
echo "[pipeline $(date)] Stage 3: extracting e2v-plus-large features..."
mkdir -p "$ROOT/features/holoassist_pool/e2v-plus-large"
bash tools/run_supervised.sh "$ROOT/features/holoassist_pool/e2v-plus-large/progress.json" \
    ./venv/bin/python -m features.extract --dataset holoassist_pool --encoder e2v-plus-large

# 4. Pseudolabeling and quality rows
echo "[pipeline $(date)] Stage 4: scoring pool and generating pseudolabels..."
./venv/bin/python -m transfer.pseudolabel

echo "[pipeline $(date)] === HoloAssist pool pipeline complete ==="
