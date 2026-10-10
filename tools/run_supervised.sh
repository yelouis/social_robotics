#!/usr/bin/env bash
# Supervised runner: long unattended runs die silently (the recurring macOS
# "Python quit unexpectedly" native-crash mode) and an operator cannot tell a
# dead run from a slow one. If the runner is resumable (per-item atomic writes +
# resume-by-default), recovery is one relaunch away — this wrapper performs that
# relaunch automatically and surfaces a genuinely-stuck run loudly.
# (History: docs/LESSONS_v0.md, "Operations".)
#
# What it adds around any resumable runner:
#   - caffeinate -dimsu : blocks system/disk sleep for the run's lifetime
#   - PYTHONFAULTHANDLER : a native fault dumps a Python traceback into the log
#                          instead of dying silently
#   - relaunch loop      : re-invokes the (resumable) runner until it exits 0
#   - no-progress guard   : counts records in the result JSON between attempts;
#                          aborts after 2 consecutive relaunches that add zero
#                          records (a deterministic "poison clip" crash-loop —
#                          runners that mark an item processed only AFTER
#                          success can otherwise crash-loop forever).
#
# Usage:
#   tools/run_supervised.sh <result_json> <runner command...>
# Example:
#   tools/run_supervised.sh results/features.json \
#       ./venv/bin/python -m <resumable runner>
#
# Env:
#   SR_SUPERVISE_MAX_ATTEMPTS  hard ceiling on relaunches (default 50)
#   SR_SUPERVISE_LOG           supervisor log path (default supervise_<ts>.log)
#   SR_MEMWAIT_SLEEP_S         sleep seconds after exit 75 memory deferral (default 600)
#   SR_MAX_MEM_DEFERRALS       max consecutive memory deferrals before abort (default 72)
set -uo pipefail

PROGRESS_FILE="${1:-}"
if [ -z "$PROGRESS_FILE" ] || [ "$#" -lt 2 ]; then
    echo "usage: run_supervised.sh <result_json> <runner command...>" >&2
    exit 2
fi
shift

MAX_ATTEMPTS="${SR_SUPERVISE_MAX_ATTEMPTS:-50}"
LOG="${SR_SUPERVISE_LOG:-supervise_$(date +%Y%m%d_%H%M%S).log}"
MEMWAIT_SLEEP_S="${SR_MEMWAIT_SLEEP_S:-600}"
MAX_MEM_DEFERRALS="${SR_MAX_MEM_DEFERRALS:-72}"

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
mem_deferrals=0
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

    if [ "$rc" -eq 75 ]; then
        mem_deferrals=$((mem_deferrals + 1))
        log "runner deferred by memory guard (exit 75, deferral $mem_deferrals/$MAX_MEM_DEFERRALS) — sleeping ${MEMWAIT_SLEEP_S}s before retry"
        if [ "$mem_deferrals" -ge "$MAX_MEM_DEFERRALS" ]; then
            log "ABORT: memory guard: deferred $mem_deferrals times; giving up"
            exit 75
        fi
        sleep "$MEMWAIT_SLEEP_S"
        continue
    fi

    mem_deferrals=0
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
