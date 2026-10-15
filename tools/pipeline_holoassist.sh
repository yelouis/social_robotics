#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/Extreme SSD/social_robotics"
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "[pipeline $(date)] pipeline orchestrator started."

# 1. Wait for current train judge supervisor (PID 78507) to finish
echo "[pipeline $(date)] waiting for train judge (PID 78507) to finish..."
while kill -0 78507 2>/dev/null; do
    sleep 30
done

N_TR=$(python3 -c "import json; print(len(json.load(open('$ROOT/judge/holoassist/qwen2.5vl_7b/progress_train.json'))))" 2>/dev/null || echo 0)
echo "[pipeline $(date)] train judge finished with $N_TR items."

# 2. Run local judge on test split
echo "[pipeline $(date)] starting local judge on test split (qwen2.5vl:7b)..."
bash tools/run_supervised.sh "$ROOT/judge/holoassist/qwen2.5vl_7b/progress_test.json" \
    ./venv/bin/python -m judge.vlm_judge --dataset holoassist --split test --backend ollama

N_TE=$(python3 -c "import json; print(len(json.load(open('$ROOT/judge/holoassist/qwen2.5vl_7b/progress_test.json'))))" 2>/dev/null || echo 0)
echo "[pipeline $(date)] test judge finished with $N_TE items."

# 3. Run large local judge on test split (gemma4:26b)
echo "[pipeline $(date)] starting large judge on test split (gemma4:26b)..."
bash tools/run_supervised.sh "$ROOT/judge/holoassist/gemma4_26b/progress.json" \
    ./venv/bin/python -m judge.vlm_judge --dataset holoassist --split test --backend ollama-large

N_LG=$(python3 -c "import json; print(len(json.load(open('$ROOT/judge/holoassist/gemma4_26b/progress.json'))))" 2>/dev/null || echo 0)
echo "[pipeline $(date)] large judge finished with $N_LG items."

# 4. Run probes
echo "[pipeline $(date)] running probes for HoloAssist..."
./venv/bin/python -m harness.probes --dataset holoassist

# 5. Generate report and TSV
echo "[pipeline $(date)] generating HoloAssist report and examples TSV..."
./venv/bin/python tools/report_holoassist.py

echo "[pipeline $(date)] ALL STAGES FINISHED SUCCESSFULLY."
