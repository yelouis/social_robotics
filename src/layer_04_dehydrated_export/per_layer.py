"""Per-layer dehydrated publishing.

The combined Layer-04 export merges every 03* layer into one wide table. This
module instead emits ONE dataset per layer — each carrying the manifest base
columns + only that layer's feature columns — for publishing as an independent
Hugging Face dataset with a detailed, column-accurate interpretation card.

Two layer-specific behaviors the combined export deliberately does NOT apply:
  1. Rows with no measured signal for *this* layer are dropped. In the combined
     table such a row may still be useful (another layer scored it); in a
     per-layer dataset it is dead weight (only manifest metadata, or a
     `skipped_reason` sentinel). `drop_unmeasured_rows` removes them and then
     prunes any column left entirely empty.
  2. The internal `03x_` layer-id prefix is stripped from the published headers
     — handled downstream by `DehydratedExporter.export_parquet`.
"""
import json
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

try:
    from .aggregator import DataAggregator
    from .export import DehydratedExporter
except ImportError:  # src/ on path rather than package import
    from layer_04_dehydrated_export.aggregator import DataAggregator
    from layer_04_dehydrated_export.export import DehydratedExporter

# Manifest/identity columns the aggregator joins onto every row — never a
# per-layer "signal" and never dropped.
MANIFEST_COLUMNS = ("video_id", "source_dataset", "task_labels", "duration_sec", "fps")
# String values that mean "no data" even though the cell is technically non-null.
_EMPTY_TOKENS = ("", "[]", "{}")
_LAYER_ID_RE = re.compile(r"^03[a-z]_")


def published_layer_name(layer_name: str) -> str:
    """`03f_motor_resonance` -> `motor_resonance` (the published, prefix-free name)."""
    return _LAYER_ID_RE.sub("", layer_name)


def repo_slug(layer_name: str, prefix: str = "social-robotics") -> str:
    """`03f_motor_resonance` -> `social-robotics-motor-resonance`."""
    return f"{prefix}-{published_layer_name(layer_name).replace('_', '-')}"


def _is_empty(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(v, str) and v.strip() in _EMPTY_TOKENS


def drop_unmeasured_rows(df: pd.DataFrame, manifest_columns=MANIFEST_COLUMNS) -> pd.DataFrame:
    """Drop rows that carry no measured signal for this layer, then prune any
    column left entirely empty.

    A row is *unmeasured* when every non-manifest column is NaN or an empty-JSON
    sentinel (`"[]"`/`"{}"`/`""`) — i.e. the video was either absent from the
    layer's output or present only as a `skipped_reason` sentinel. `skipped_reason`
    itself is not counted as signal (its presence marks the absence of signal)."""
    manifest = set(manifest_columns)
    signal_cols = [c for c in df.columns
                   if c not in manifest and not c.endswith("_skipped_reason")]
    if not signal_cols:
        return df.copy()

    def has_signal(row) -> bool:
        return any(not _is_empty(row[c]) for c in signal_cols)

    kept = df[df.apply(has_signal, axis=1)].copy()

    # Prune columns that are entirely empty across the surviving rows (e.g. the
    # now-vestigial skipped_reason, or a metric no surviving row populated).
    for col in list(kept.columns):
        if col in manifest:
            continue
        if kept[col].map(_is_empty).all():
            kept = kept.drop(columns=[col])
    return kept


# ---------------------------------------------------------------------------
# Dataset-card content (detailed interpretation), keyed by PUBLISHED names.
# ---------------------------------------------------------------------------
_MANIFEST_DESCRIPTIONS = {
    "video_id": ("string", "Ego4D source-clip UUID. The rehydration key — map back to your own legally-obtained Ego4D copy (`<video_id>.mp4`)."),
    "source_dataset": ("string", "Origin corpus (`ego4d`)."),
    "task_labels": ("string", "Comma-joined VLM task label(s) — the activity the camera-wearer performed."),
    "duration_sec": ("float", "Source clip duration (seconds)."),
    "fps": ("float", "Source clip frame rate."),
}

_LAYER_SPECS = {
    "attention": dict(
        lid="03a", title="Attention / Engagement",
        tagline="Does the bystander look at the camera-wearer? Per-bystander gaze + head-pose engagement around each task.",
        method="L2CS-Net 3D gaze + MediaPipe FaceLandmarker head pose, sampled adaptively at 8→16 FPS.",
        signal="Per-bystander visual attention toward the POV actor (gaze raycast to the camera / the wearer's hands), scored 0–1 over an adaptive temporal trace.",
        tags=["gaze-estimation", "attention"],
        caveats=[
            "Trace timestamps (in the `*_raw` column) are **not uniformly spaced** (adaptive stride); resample onto a fixed-dt grid before frequency-domain analysis.",
            "A missing row would mean no bystander face was trackable; such rows are excluded from this per-layer dataset.",
        ],
        columns={
            "attention_mean_attention_all_persons": ("float (0–1)", "Mean `average_attention_score` across tracked bystanders. Higher = more visually focused on the wearer/task."),
            "attention_any_person_engaged": ("bool", "True if any bystander crossed the engagement threshold."),
            "attention_num_bystanders_tracked": ("float (int)", "Number of bystanders for whom a gaze trace was produced."),
            "attention_per_person_raw": ("JSON string", "Per-bystander detail: `person_id`, `average_attention_score`, `peak_engagement_timestamp_sec`, `attention_variance`, `sustained_engagement_sec`, `is_engaged`, `gaze_target_classification` (`Camera`/`POV_Actor_Hands`/`Unknown`), and `attention_trace` of `{t, score, pitch_rad, yaw_rad, target}` (head-pose Euler angles in radians)."),
            "attention_processing_meta_model_used": ("string", "Gaze model id (e.g. `l2cs_net_3d_gaze`)."),
            "attention_processing_meta_sampling_fps_effective": ("float", "Baseline sampling rate (8 FPS)."),
            "attention_processing_meta_sampling_fps_burst": ("float", "Boosted rate during fast attention transitions (16 FPS)."),
            "attention_processing_meta_sampling_strategy": ("string", "e.g. `adaptive_8_to_16_fps`."),
        },
    ),
    "reasonable_emotion": dict(
        lid="03b", title="Reasonable Emotion",
        tagline="Was the bystander's facial-emotion trajectory contextually appropriate to the task?",
        method="HSEmotion face emotion + an LLM (Gemma) reasoning pass over emotion transitions.",
        signal="A 0–1 'reasonableness' score for how the bystander's emotional response evolved during a task's reaction window (late-stage-weighted success of emotion transitions toward a contextually-appropriate direction).",
        tags=["emotion-recognition", "facial-emotion"],
        caveats=[
            "**Very sparse** — bystander faces are rarely both visible and emotion-legible in egocentric video. Treat as illustrative, not statistically representative.",
        ],
        columns={
            "reasonable_emotion_avg_task_score": ("float (0–1)", "Mean across tasks of `task_aggregate_score` — higher = more contextually reasonable emotional trajectory."),
            "reasonable_emotion_emotion_source": ("string", "Which emotion backend produced the underlying labels (face-emotion model vs VLM fallback)."),
            "reasonable_emotion_tasks_analyzed_raw": ("JSON string", "Per-task detail: `task_label`, `task_reaction_window_sec`, per-person `temporal_slices` (`{window_sec, transition_pair, terminal_magnitude, classified_direction, slice_success_scalar}`), `late_stage_weighted_success_score`, `task_aggregate_score`."),
        },
    ),
    "acoustic_prosody": dict(
        lid="03c", title="Acoustic Prosody",
        tagline="Ambient vocal tone around each task — alarming vs soothing — as corroborating context.",
        method="emotion2vec+ / SenseVoice speech-emotion + librosa acoustic features on the task audio window.",
        signal="A signed −1…+1 prosody scalar summarizing the acoustic tone of the task window (negative = alarming/discouraging, ~0 = neutral, positive = soothing/positive).",
        tags=["audio", "speech-emotion-recognition", "prosody"],
        caveats=[
            "**Ambient audio, NOT bystander-attributed.** Egocentric audio is dominated by the camera-wearer; there is no speaker separation. Use `prosody_scalar` only as **corroboration** for a per-bystander visual signal, never standalone.",
            "Check `audio_present` in the raw column before fusing; tasks with no audio should be excluded, not read as confident-neutral.",
        ],
        columns={
            "acoustic_prosody_avg_prosody_scalar": ("float (−1…+1)", "Mean across tasks of `prosody_scalar`. Sign = valence of ambient tone; magnitude = intensity."),
            "acoustic_prosody_tasks_analyzed_raw": ("JSON string", "Per-task detail: `prosody_metrics` (`max_amplitude_dbFS`, `pitch_contour_variance`, 9-class `emotion_scores`, `dominant_emotion`, `audio_present`), `classified_acoustic_tone` (`Alarming`/`Soothing`/`Discouraging`/`Neutral`), `prosody_scalar`."),
        },
    ),
    "proxemic_kinematics": dict(
        lid="03d", title="Proxemic Kinematics",
        tagline="Did the bystander move toward (approach) or away from (retreat) the camera-wearer?",
        method="Depth-Anything monocular depth + SAM masks + YOLO bbox-scale delta over the task window.",
        signal="A signed proxemic vector per bystander combining bbox-scale change (40%) and monocular depth change (60%): positive = approach, negative = retreat; plus a discrete approach/avoidance class.",
        tags=["depth-estimation", "proxemics", "kinematics"],
        caveats=[
            "`person_id` (in the raw column) is **not a stable identity** across the clip; a vector is a within-window delta, not a tracked trajectory.",
        ],
        columns={
            "proxemic_kinematics_max_abs_proxemic_vector": ("float (signed)", "Largest-magnitude proxemic vector across bystanders. **Positive = approach, negative = retreat.** Micro-movements (|Δ|<~0.05) are deadbanded to zero."),
            "proxemic_kinematics_max_proxemic_confidence": ("float (0–1)", "Max confidence across bystanders."),
            "proxemic_kinematics_any_approach_detected": ("bool", "True if any bystander was classified `Approach_Intervention`."),
            "proxemic_kinematics_any_avoidance_detected": ("bool", "True if any bystander was classified `Avoidance`."),
            "proxemic_kinematics_tasks_analyzed_raw": ("JSON string", "Per-task / per-person detail: `bbox_scale_delta_pct`, `depth_anything_v2_delta`, `proxemic_vector`, `classified_action`, `proxemic_confidence`, `optical_flow_noise`."),
        },
    ),
    "affirmation_gesture": dict(
        lid="03e", title="Affirmation Gesture",
        tagline="Did the bystander nod (affirm) or shake their head (negate)?",
        method="1–3 Hz Butterworth band-pass + SciPy peak-finding on 03a's head-pose pitch/yaw traces (no extra inference).",
        signal="Detection of communicative head nods (`affirming_nod`) and shakes (`negating_shake`) from rhythmic head-pose oscillation during the task window.",
        tags=["gesture-recognition", "head-pose"],
        caveats=[
            "Derived from 03a's head-pose trace, which is sparse on egocentric footage — so this layer is low-yield by nature.",
            "`interpolated_fraction` in the raw column flags how much of the window was bbox-interpolated; high values are lower-trust.",
        ],
        columns={
            "affirmation_gesture_any_nod_detected": ("bool", "True if any bystander produced an `affirming_nod`."),
            "affirmation_gesture_any_shake_detected": ("bool", "True if any bystander produced a `negating_shake`."),
            "affirmation_gesture_max_gesture_confidence": ("float (0–1)", "Max gesture confidence across bystanders."),
            "affirmation_gesture_tasks_analyzed_raw": ("JSON string", "Per-task / per-person detail: `pitch_oscillation_hz`, `yaw_oscillation_hz`, `interpolated_fraction`, `gesture_detected` (`affirming_nod`/`negating_shake`/`ambiguous_wobble`/none), `confidence`."),
        },
    ),
    "motor_resonance": dict(
        lid="03f", title="Motor Resonance",
        tagline="Did the bystander flinch in sympathy with a sudden camera jolt, or mirror the wearer's motion?",
        method="YOLOv8-pose dense in-window sampling + Farneback optical flow; impulse + dominant-jolt resonance gate.",
        signal="Sympathetic motor responses: an impulsive bystander 'flinch' time-locked to the wearer's dominant camera jolt (motor resonance), and postural mirroring of the wearer's vertical motion.",
        tags=["pose-estimation", "optical-flow", "motor-resonance"],
        caveats=[
            "`max_ego_chaos_score` describes the **wearer's camera**, not the bystander — it is context for the flinch detector, not a bystander reaction.",
            "Motor-resonance is deliberately conservative (impulse ≥ 3× the bystander's own median, time-locked to the single dominant jolt, ≥ 2 genuine detections), so camera motion through a fixed bbox cannot manufacture a flinch.",
        ],
        columns={
            "motor_resonance_any_motor_resonance": ("bool", "True if any bystander produced an impulsive flinch time-locked to the dominant ego jolt."),
            "motor_resonance_max_empathy_scalar": ("float (0–1)", "Max empathy scalar; non-zero only when a resonance fired."),
            "motor_resonance_any_mirroring": ("bool", "True if any bystander's spine-angle change correlated with sustained downward camera motion."),
            "motor_resonance_max_mirroring_scalar": ("float (0–1)", "Max mirroring-correlation strength across bystanders."),
            "motor_resonance_max_ego_chaos_score": ("float (~0–1)", "Max per-task ego-kinetic chaos — the **camera-wearer's** motion intensity (context), NOT a bystander signal."),
            "motor_resonance_tasks_analyzed_raw": ("JSON string", "Per-task / per-person detail: `ego_kinetic_chaos_score`, `bystander_pose_velocity_peak`, `resonance_delay_sec`, `motor_resonance_detected`, `empathy_scalar`, `mirroring_detected`, `mirroring_scalar`."),
        },
    ),
}


def _col_table(rows):
    out = ["| column | type | meaning |", "|---|---|---|"]
    for name, typ, desc in rows:
        out.append(f"| `{name}` | {typ} | {desc} |")
    return "\n".join(out)


def dataset_card(df: pd.DataFrame, layer_name: str, repo_id: str) -> str:
    """Build a detailed dataset card whose column table reflects the ACTUAL
    columns present in the published DataFrame (so it always matches the parquet,
    even after `drop_unmeasured_rows` prunes empty columns)."""
    key = published_layer_name(layer_name)
    spec = _LAYER_SPECS.get(key)
    if spec is None:
        raise KeyError(f"No card spec for layer '{layer_name}' (published name '{key}'). "
                       f"Add an entry to per_layer._LAYER_SPECS.")
    feat_cols = [c for c in df.columns if c not in MANIFEST_COLUMNS]
    manifest_rows = [(c, *_MANIFEST_DESCRIPTIONS[c]) for c in MANIFEST_COLUMNS if c in df.columns]
    feat_rows = []
    for c in feat_cols:
        typ, desc = spec["columns"].get(c, ("", "(layer-specific column)"))
        feat_rows.append((c, typ, desc))
    tags = "\n".join(f"- {t}" for t in (["egocentric", "social-interactions", "ego4d", "human-robot-interaction"] + spec["tags"]))
    caveats = "\n".join(f"- {c}" for c in spec["caveats"])
    has_raw = any(c.endswith("_raw") for c in feat_cols)
    raw_example = ""
    if has_raw:
        raw_example = f"""
### The `*_raw` JSON column
The `*_raw` column holds the full nested per-task / per-person detail as a JSON string. Parse it with:
```python
import json, pandas as pd
df = pd.read_parquet("hf://datasets/{repo_id}/social_metadata.parquet")
raw_col = next(c for c in df.columns if c.endswith("_raw"))
detail = json.loads(df.iloc[0][raw_col])
```
"""
    return f"""---
license: mit
pretty_name: "Social Robotics — {spec['title']} ({spec['lid']})"
language:
- en
tags:
{tags}
size_categories:
- n<1K
configs:
- config_name: default
  data_files: social_metadata.parquet
---

# Social Robotics: {spec['title']} (`{spec['lid']}`)

> {spec['tagline']}

One layer of the **Social-Affective Filter (SAF)** — dehydrated social-signal metadata extracted from egocentric (first-person) video so robots can learn to read human reactions. **No raw pixels and no audio.** Each row is one source video, keyed by `video_id`; rehydrate against your own legally-obtained Ego4D copies (below).

- **Rows:** {len(df)} — videos in the 50-clip evaluation slice for which **this layer produced a measured signal** (videos it could not measure are excluded from this per-layer dataset; the layers still join 1:1 on `video_id`).
- **Signal:** {spec['signal']}
- **Method:** {spec['method']}

## ⚠️ Read this first — interpretation caveats
{caveats}
- Egocentric footage is legitimately low-yield (small/sparse bystander faces, heavy camera motion); we publish honest measurements only, never fabricated zeros.

## Columns

### Identity & manifest (shared across all SAF datasets)
{_col_table(manifest_rows)}

### {spec['title']} signal
{_col_table(feat_rows)}
{raw_example}
## How to load
```python
import pandas as pd
df = pd.read_parquet("hf://datasets/{repo_id}/social_metadata.parquet")
# or:  from datasets import load_dataset;  ds = load_dataset("{repo_id}")
```

## Rehydration — mapping back to video
`video_id` is the **Ego4D clip UUID**. With your own licensed Ego4D copy, the file is `<video_id>.mp4`; timestamps in the `*_raw` columns index into that clip. A helper (`rehydrate_dataset.py`) is included. We never redistribute source media — obtain Ego4D under its own license.

## Provenance
Generated by the SAF pipeline (`export_metadata.json` records `schema_version` + `pipeline_git_sha`). Headers are the descriptive layer name + metric; the pipeline's internal `{spec['lid']}_` layer-id prefix is stripped at publish time. License **MIT** (this metadata only; Ego4D videos remain under the Ego4D license).
"""


def build_layer_dataset(layer_name, result_path, manifest_path, out_dir,
                        owner="louisye", repo_prefix="social-robotics",
                        git_sha="unknown", rehydrate_script=None):
    """Aggregate one layer over the manifest, drop unmeasured rows, export the
    prefix-stripped parquet + metadata, and write the dataset card + rehydrate
    helper into ``out_dir``. No network. Returns a summary dict (incl. repo_id)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_id = f"{owner}/{repo_slug(layer_name, repo_prefix)}"

    # Aggregate just this layer: a temp dir holding the manifest + the single
    # canonically-named result file (the aggregator globs `03*_result.json`).
    tmp = Path(tempfile.mkdtemp(prefix="saf_per_layer_"))
    try:
        shutil.copy(manifest_path, tmp / "filtered_manifest.json")
        shutil.copy(result_path, tmp / f"{layer_name}_result.json")
        agg = DataAggregator(str(tmp))
        df = agg.aggregate()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    df = drop_unmeasured_rows(df)
    if df.empty:
        raise ValueError(f"{layer_name}: no measured rows after filtering — nothing to publish.")
    # Attach provenance to the FINAL (filtered) frame so export_metadata reflects it.
    df = agg.add_export_metadata(df, active_layers=[layer_name.split("_", 1)[0]], git_sha=git_sha)

    pq, meta = DehydratedExporter(str(out_dir)).export_parquet(df)  # strips 03x_, writes parquet+metadata

    published = pd.read_parquet(pq)  # read back to get the actual published columns
    (out_dir / "README.md").write_text(dataset_card(published, layer_name, repo_id))
    if rehydrate_script and Path(rehydrate_script).exists():
        shutil.copy(rehydrate_script, out_dir / "rehydrate_dataset.py")

    return {"layer_name": layer_name, "repo_id": repo_id, "parquet": str(pq),
            "n_rows": int(published.shape[0]), "n_cols": int(published.shape[1]),
            "columns": list(published.columns), "out_dir": str(out_dir)}


def publish_layer_dataset(out_dir, repo_id, token):
    """Create the PUBLIC dataset repo and upload the staged folder. Network."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required to publish.") from e
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True, token=token)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset", token=token,
                      commit_message="Publish per-layer dehydrated dataset (parquet + card)")
    info = api.dataset_info(repo_id, token=token)
    return {"repo_id": repo_id, "private": info.private,
            "files": sorted(s.rfilename for s in info.siblings)}


def publish_all_layers(results_dir, manifest_path=None, out_root=None, owner="louisye",
                       repo_prefix="social-robotics", git_sha="unknown",
                       upload=False, token=None):
    """Build (and optionally publish) one dataset per `03*_result.json` found in
    ``results_dir`` — the same directory the combined Layer-04 aggregator scans."""
    results_dir = Path(results_dir)
    manifest_path = Path(manifest_path) if manifest_path else results_dir / "filtered_manifest.json"
    out_root = Path(out_root) if out_root else results_dir / "per_layer_datasets"
    rehydrate = Path(__file__).parent / "rehydrate_dataset.py"

    summaries = []
    for result_path in sorted(results_dir.glob("03*_result.json")):
        layer_name = result_path.stem.replace("_result", "")
        out_dir = out_root / published_layer_name(layer_name)
        summary = build_layer_dataset(layer_name, result_path, manifest_path, out_dir,
                                      owner=owner, repo_prefix=repo_prefix, git_sha=git_sha,
                                      rehydrate_script=rehydrate)
        if upload:
            summary["upload"] = publish_layer_dataset(out_dir, summary["repo_id"], token)
        summaries.append(summary)
    return summaries
