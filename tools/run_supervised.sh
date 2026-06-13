#!/usr/bin/env bash
# Supervised runner — resolves docs/03_social_layer_architecture.md Unresolved
# Issue 1 (Option A): long unattended layer runs die silently (the recurring
# macOS "Python quit unexpectedly" native-crash mode) and an operator cannot
# tell a dead run from a slow one. Every layer pipeline is already resumable
# (per-video atomic writes + resume-by-default), so recovery is one relaunch
# away — this wrapper performs that relaunch automatically and surfaces a
# genuinely-stuck run loudly.
#
# What it adds around any resumable runner:
#   - caffeinate -dimsu : blocks system/disk sleep for the run's lifetime
#   - PYTHONFAULTHANDLER : a native fault dumps a Python traceback into the log
#                          instead of dying silently
#   - relaunch loop      : re-invokes the (resumable) runner until it exits 0
#   - no-progress guard   : counts records in the result JSON between attempts;
#                          aborts after 2 consecutive relaunches that add zero
#                          records (a deterministic "poison clip" crash-loop —
#                          the edge the docs/01 acquisition wrapper never hit,
#                          because layer pipelines mark a video processed only
#                          AFTER success, not before).
#
# Usage:
#   tools/run_supervised.sh <result_json> <runner command...>
# Example:
#   tools/run_supervised.sh \
#       e2e_reports/2026_06_10_layer03d_50_postfix/03d_result_50.json \
#       ./venv/bin/python tools/run_03d_50.py
#
# Env:
#   SR_SUPERVISE_MAX_ATTEMPTS  hard ceiling on relaunches (default 50)
#   SR_SUPERVISE_LOG           supervisor log path (default supervise_<ts>.log)
set -uo pipefail

PROGRESS_FILE="${1:-}"
if [ -z "$PROGRESS_FILE" ] || [ "$#" -lt 2 ]; then
    echo "usage: run_supervised.sh <result_json> <runner command...>" >&2
    exit 2
fi
shift

MAX_ATTEMPTS="${SR_SUPERVISE_MAX_ATTEMPTS:-50}"
LOG="${SR_SUPERVISE_LOG:-supervise_$(date +%Y%m%d_%H%M%S).log}"

# caffeinate is macOS-only; degrade gracefully (e.g. CI/Linux) to a no-op prefix.
if command -v caffeinate >/dev/null 2>&1; then
    NOSLEEP=(caffeinate -dimsu)
else
    NOSLEEP=()
fi

count_records() {
    python3 - "$PROGRESS_FILE" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(len(d) if isinstance(d, list) else 0)
except Exception:
    print(0)
PY
}

log() { echo "[supervise $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&2; }

log "supervising: $* (progress=$PROGRESS_FILE, max_attempts=$MAX_ATTEMPTS, log=$LOG)"

stale=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    before="$(count_records)"
    log "attempt $attempt/$MAX_ATTEMPTS — $before records so far"
    PYTHONFAULTHANDLER=1 "${NOSLEEP[@]}" "$@" 2>&1 | tee -a "$LOG"
    rc="${PIPESTATUS[0]}"
    after="$(count_records)"

    if [ "$rc" -eq 0 ]; then
        log "runner exited 0 (clean) — $after records total. DONE."
        exit 0
    fi

    log "runner died (exit $rc) — progressed ${before} -> ${after} records this attempt."
    if [ "$after" -le "$before" ]; then
        stale=$((stale + 1))
        log "no progress this attempt (zero-progress streak = $stale)."
        if [ "$stale" -ge 2 ]; then
            log "ABORT: 2 consecutive relaunches with zero new records — likely a deterministic poison clip crashing at the same spot. Inspect $LOG (PYTHONFAULTHANDLER traceback) and the failing clip."
            exit 1
        fi
    else
        stale=0
    fi
done

log "ABORT: reached MAX_ATTEMPTS=$MAX_ATTEMPTS without a clean exit."
exit 1
