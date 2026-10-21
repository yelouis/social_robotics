#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/Extreme SSD/social_robotics"
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
export TMPDIR="/Volumes/Extreme SSD/tmp"
export SR_NO_MODEL_BANNER=1

echo "[pipeline $(date)] === Starting holoassist_audible pipeline ==="

# 1. memguard status
echo "[pipeline $(date)] Stage 1: checking memory guard status..."
./venv/bin/python -m shared.memguard --status

# 2. Features: siglip-b16-224
echo "[pipeline $(date)] Stage 2: extracting siglip-b16-224 features..."
mkdir -p "$ROOT/features/holoassist_audible/siglip-b16-224"
bash tools/run_supervised.sh "$ROOT/features/holoassist_audible/siglip-b16-224/progress.json" \
    ./venv/bin/python -m features.extract --dataset holoassist_audible --encoder siglip-b16-224

# 3. Features: e2v-plus-large
echo "[pipeline $(date)] Stage 3: extracting e2v-plus-large features..."
mkdir -p "$ROOT/features/holoassist_audible/e2v-plus-large"
bash tools/run_supervised.sh "$ROOT/features/holoassist_audible/e2v-plus-large/progress.json" \
    ./venv/bin/python -m features.extract --dataset holoassist_audible --encoder e2v-plus-large

# 4. Local Judge (qwen2.5vl:7b) train split
echo "[pipeline $(date)] Stage 4: running local judge (qwen2.5vl:7b) on train split..."
mkdir -p "$ROOT/judge/holoassist_audible/qwen2.5vl_7b"
bash tools/run_supervised.sh "$ROOT/judge/holoassist_audible/qwen2.5vl_7b/progress_train.json" \
    ./venv/bin/python -m judge.vlm_judge --dataset holoassist_audible --split train --backend ollama

# 5. Local Judge (qwen2.5vl:7b) test split
echo "[pipeline $(date)] Stage 5: running local judge (qwen2.5vl:7b) on test split..."
bash tools/run_supervised.sh "$ROOT/judge/holoassist_audible/qwen2.5vl_7b/progress_test.json" \
    ./venv/bin/python -m judge.vlm_judge --dataset holoassist_audible --split test --backend ollama

# 6. Large Local Judge (gemma4:26b) test split
echo "[pipeline $(date)] Stage 6: running large local judge (gemma4:26b) on test split..."
mkdir -p "$ROOT/judge/holoassist_audible/gemma4_26b"
bash tools/run_supervised.sh "$ROOT/judge/holoassist_audible/gemma4_26b/progress.json" \
    ./venv/bin/python -m judge.vlm_judge --dataset holoassist_audible --split test --backend ollama-large

# 7. Probes
echo "[pipeline $(date)] Stage 7: running probes for holoassist_audible..."
./venv/bin/python -m harness.probes --dataset holoassist_audible --notes "Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md)"

# 8. Report
echo "[pipeline $(date)] Stage 8: generating report for holoassist_audible..."
./venv/bin/python tools/report_holoassist.py --dataset holoassist_audible

echo "[pipeline $(date)] === holoassist_audible pipeline finished successfully ==="
