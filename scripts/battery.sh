#!/usr/bin/env bash
set -u

SLOW=0
for arg in "$@"; do
    if [ "$arg" = "--slow" ]; then
        SLOW=1
    fi
done

max_exit=0

# G1: Lint
./venv/bin/ruff check src tests tools
g1_exit=$?
echo "G1 lint: exit $g1_exit"
if [ "$g1_exit" -gt "$max_exit" ]; then
    max_exit=$g1_exit
fi

# G2: Fast tests
SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/
g2_exit=$?
echo "G2 tests: exit $g2_exit"
if [ "$g2_exit" -gt "$max_exit" ]; then
    max_exit=$g2_exit
fi

# G3: Harness self-test
if [ ! -f "src/harness/scorecard.py" ]; then
    echo "G3 selftest: skipped (harness not built)"
    g3_exit=0
else
    PYTHONPATH=src SR_NO_MODEL_BANNER=1 ./venv/bin/python -m harness.scorecard --selftest
    g3_exit=$?
    echo "G3 selftest: exit $g3_exit"
fi
if [ "$g3_exit" -gt "$max_exit" ]; then
    max_exit=$g3_exit
fi

# G4: Slow tests
if [ "$SLOW" -eq 1 ]; then
    SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m slow tests/
    g4_exit=$?
    echo "G4 slow: exit $g4_exit"
    if [ "$g4_exit" -gt "$max_exit" ]; then
        max_exit=$g4_exit
    fi
fi

exit "$max_exit"
