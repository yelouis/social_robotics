#!/bin/zsh
# SRB v0 — engine pre-pass over the held-out manifest (docs/07 §7 v0):
#   harvest (download + Node-02) -> 02b (captions ON) -> 03a(restrict) ->
#   03e -> 03c -> densify -> 03f -> 03d -> segment join -> candidate export
#   -> rating kits.
# Every stage is resumable; rerunning this script continues where it stopped.
set -e
ROOT="/Users/louisye/Desktop/Louis/social_robotics"
PY="$ROOT/venv/bin/python"
B="/Volumes/Extreme SSD/social_robotics/bench_v0"
M="$B/heldout_manifest.json"

echo "=== harvest (download + Node-02) $(date) ==="
cd "$ROOT/bench" && $PY harvest_heldout.py

echo "=== 02b (captions+controls+wearer) $(date) ==="
cd "$ROOT/src" && $PY -m layer_02b_task_climax.pipeline "$M" --workers 4

echo "=== 03a (segment-restricted) $(date) ==="
SR_03A_SEGMENT_RESTRICT=1 $PY -m layer_03a_attention.pipeline "$M" "$B/03a_attention_result.json" --workers 3

echo "=== 03e $(date) ==="
$PY -c "
import sys; sys.path.insert(0, '.')
from layer_03e_affirmation_gesture.pipeline import AffirmationGesturePipeline
AffirmationGesturePipeline('$M', '$B/03e_affirmation_gesture_result.json', '$B/03a_attention_result.json').run()"

echo "=== 03c $(date) ==="
$PY -m layer_03c_acoustic_prosody.pipeline --manifest "$M" --output "$B/03c_acoustic_prosody_result.json"

echo "=== densify $(date) ==="
$PY "$ROOT/tools/densify_manifest.py" --in "$M" --out "$B/heldout_manifest.dense.json"

echo "=== 03f $(date) ==="
$PY "$ROOT/tools/run_parallel_layer.py" --layer 03f --manifest "$B/heldout_manifest.dense.json" \
    --output "$B/03f_motor_resonance_result.json" --workers 3

echo "=== 03d $(date) ==="
$PY "$ROOT/tools/run_parallel_layer.py" --layer 03d --manifest "$B/heldout_manifest.dense.json" \
    --output "$B/03d_proxemic_kinematics_result.json" --workers 3

echo "=== segment join $(date) ==="
cp "$M" "$B/filtered_manifest.json"
$PY -m layer_04_dehydrated_export.segment_dataset "$B" --manifest "$M" \
    --out-dir "$B/segment_dataset" --git-sha "$(git -C $ROOT rev-parse --short HEAD)"

echo "=== candidate export $(date) ==="
cd "$ROOT/bench" && $PY export_candidates.py

echo "=== rating kits $(date) ==="
$PY make_rating_kit.py

echo "=== SRB PRE-PASS DONE $(date) ==="
