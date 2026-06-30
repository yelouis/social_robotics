#!/bin/bash
# Supervised relaunch loop for the top-200 density landing.
#
# Long Layer-03 runs occasionally hit a silent macOS native crash (MPS/ffmpeg
# fault, no traceback) that kills the whole process tree — exactly what took down
# the first unsupervised attempt at ~57/200 clips of 03f. The landing is fully
# resumable (densify skips an existing dense manifest; the parallel runner uses
# deterministic shards whose workers resume from shard{i}.result.json), so the
# fix is supervision, not pipeline changes: relaunch until "LANDING DONE", with
# PYTHONFAULTHANDLER (a Python-level fault now dumps a traceback) and a
# no-progress guard that aborts a deterministic poison-clip crash-loop.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PY="$ROOT/venv/bin/python"
RUN="$ROOT/e2e_reports/2026_06_29_density_landing_top200"
LOG="${1:-$RUN/supervise.log}"
MAX_ATTEMPTS="${SR_MAX_ATTEMPTS:-30}"

# total records across both layers' shard outputs + merged outputs (resume progress)
count_records() {
  "$PY" - <<'PYEOF' 2>/dev/null || echo 0
import json, glob, os
run = os.environ["RUN"]
tot = 0
for p in glob.glob(os.path.join(run, "*.json.parallel", "shard*.result.json")) + \
         glob.glob(os.path.join(run, "03?_*result*.json")):
    try:
        tot += len(json.load(open(p)))
    except Exception:
        pass
print(tot)
PYEOF
}
export RUN

prev=-1
stale=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[supervise] === attempt $attempt $(date '+%F %T') ===" | tee -a "$LOG"
  PYTHONFAULTHANDLER=1 TK_SILENCE_DEPRECATION=1 caffeinate -dimsu \
    "$PY" -u "$ROOT/tools/run_density_landing.py" >> "$LOG" 2>&1
  rc=$?
  if grep -q "LANDING DONE" "$LOG"; then
    echo "[supervise] LANDING DONE after $attempt attempt(s)" | tee -a "$LOG"
    break
  fi
  cur=$(count_records)
  if [ "$cur" -le "$prev" ]; then stale=$((stale + 1)); else stale=0; fi
  echo "[supervise] attempt $attempt rc=$rc records=$cur (prev=$prev, stale=$stale)" | tee -a "$LOG"
  prev=$cur
  if [ "$stale" -ge 3 ]; then
    echo "[supervise] ABORT — no record progress across 3 consecutive relaunches (poison clip?)" | tee -a "$LOG"
    break
  fi
  sleep 10
done
echo "[supervise] supervisor exiting $(date '+%F %T')" | tee -a "$LOG"
