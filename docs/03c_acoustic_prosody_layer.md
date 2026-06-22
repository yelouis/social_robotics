# AI Task Breakdown: Acoustic Prosody Layer (03c)

## Objective
The **Acoustic Prosody Layer** draws on developmental psychology, specifically how infants respond to *Infant-Directed Speech* and *Alarm Tones*. Before lexical comprehension (understanding the actual words), infants deduce right and wrong strictly from the tone, pitch, and abruptness of a caregiver's response. This layer analyzes the non-verbal acoustic payload of the bystander's response immediately following a task.

---

## 📥 Input Requirements
- **`filtered_manifest.json`** (required): Needs the `task_reaction_window_sec` for each task so we know exactly when to slice the audio.
- **Raw Audio Chunk**: Extracted from the source `.mp4` file explicitly within the bounded task reaction window.
- **Cross-layer (optional)**: None. This layer evaluates ambient sound regardless of visual attention.

---

## 🛠️ Implementation Strategy

### 1. Audio Slicing & Pre-processing
Use an `ffmpeg` subprocess to extract the audio track spanning the exact `task_reaction_window_sec` and resample it to 16kHz (mono), saving to a uniquely named temporary file (`tempfile.mkstemp`). This avoids loading multi-gigabyte `.mp4` files into memory. 
- **Resource Management**: File I/O overhead for these bounded ~6.0s clips is negligible, so beta `torchaudio` streaming was explicitly deferred in favor of the robust FFmpeg subprocess. A strict `try/finally` block guarantees temporary `.wav` files are deleted even if downstream processing exceptions occur. Guardrails ensure zero-byte/header-only files (<44 bytes) are skipped.

### 2. Speech Emotion Recognition (emotion2vec+)
Instead of transcribing words, we run a State-of-the-Art Speech Emotion Recognition (SER) model to capture the acoustic flavor:
- **Primary SER Model**: Loaded dynamically via the cross-layer registry (`src/models_config.py`). Defaults to **emotion2vec+ large** (`iic/emotion2vec_plus_large`) on standard/high-memory hosts, with a fallback to `base` for legacy 24GB hosts.
- **Fail-Fast Initialization**: The pipeline enforces a strict fail-fast policy (`raise RuntimeError(...) from e`) if FunASR or its dependencies fail to load, preventing silent degradation to neutral outputs. Models use `disable_update=True` to load from offline cache, eliminating cold-start download latency.
- **Mechanism**: The model extracts a **9-class emotion probability distribution**. Output labels are defensively parsed to handle composite strings (e.g., splitting `'生气/angry'` to `angry`).
- **Heuristic Mapping**:
  - High `angry` + high `fearful` + Sudden Volume Spike = **Alarming / Deterrent**
  - High `happy` + high `surprised` + Melodic Pitch Contour = **Soothing / Encouraging**
  - High `sad` or high `disgusted` + Low Volume = **Discouraging / Negative**
- **Volume-bonus task gating**: the "Sudden Volume Spike" term in the Alarming rule is **withheld when the task label is an inherently-loud activity** (cooking, cleaning, laundry, blacksmithing, construction, power tools, …; the `high_volume_expected_task_keywords` set). In egocentric footage the loudest sound is usually the wearer's own object-manipulation, so raw loudness is a poor alarm proxy — without this gate, loud mechanical clatter fabricated false **Alarming** classifications on otherwise-calm clips (Resolved #2).
- **SER-Confidence Floor**: Any non-Neutral tone additionally requires the dominant SER class to clear `min_ser_confidence` (default 0.5) — a sub-floor softmax is effectively guessing, and bonus terms (pitch variance, volume) must not assemble a tone out of noise. The pitch-variance term itself only contributes when `happy` clears the floor. Mirrors 03b's `MIN_EMOTION_CONF` honesty gate.

### 3. Supplementary Audio Event Detection (SenseVoice)
We run a secondary `SenseVoiceSmall` pass (also managed via `models_config.py`) to detect non-speech social cues like laughter, applause, and crying.
- **Conditional Trigger**: Only runs when the dominant emotion confidence from `emotion2vec+` falls below 0.6.
- **Memory Optimization**: On standard hosts (<48GB unified memory), SenseVoice is **lazy-loaded** to save ~200MB of resident memory. On high-memory hosts (e.g., 64GB Mac Studio), it is **eager-loaded** at initialization to eliminate mid-run latency spikes.
- **Robust Parsing**: Matches bracketed event tokens explicitly (e.g., `<|laughter|>`) using a module-level dictionary to prevent false positives from spoken transcription words.

### 4. Acoustic Features & Heuristic Configuration
Calculate deterministic acoustic features using `librosa`:
- **Volume Spike (dB)**: Measure the delta between the pre-climax ambient noise floor and the peak amplitude.
- **Pitch Variance**: Calculate the fundamental frequency (f0) using `librosa.pyin`, properly isolating voiced frames via `~np.isnan`.
- **Dynamic Configuration**: All heuristic thresholds (volume cutoffs, pitch weights, confidence gates) are extracted into a frozen dataclass (`src/layer_03c_acoustic_prosody/config.py`). This allows overriding logic without editing core pipeline source code.

---

## 📤 Output Schema and Integration
The layer outputs an isolated JSON mapping the acoustic payload per task.

> **Scope (Issue 3 — ambient context, not bystander-attributed)**: 03c scores the *entire* ambient audio window with no speaker separation. In egocentric footage that audio is dominated by the camera-wearer (their own speech, breathing, object-handling), not the bystander, so `prosody_scalar` is **ambient acoustic context, not a bystander-attributed verdict**. Downstream fusion (Layer 04 / the 03 aggregator) must consume it only as **corroboration** — weighted by agreement with a per-bystander visual layer (03b face emotion) for the same task/window — never as a standalone bystander signal. The `audio_present` flag (Issue 1) marks tasks with no audio track so consumers exclude them from fusion rather than read them as confident-neutral.

**Example Output Data (`03c_acoustic_prosody_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03c_acoustic_prosody",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "task_reaction_window_sec": [6.2, 8.2],
      "prosody_metrics": {
        "max_amplitude_dbFS": -12.4,
        "pitch_contour_variance": 0.85,
        "emotion_scores": {
          "angry": 0.72, "happy": 0.05, "sad": 0.02,
          "surprised": 0.08, "fearful": 0.03, "neutral": 0.05,
          "disgusted": 0.02, "other": 0.02, "unknown": 0.01
        },
        "dominant_emotion": "angry",
        "dominant_emotion_confidence": 0.72,
        "audio_present": true
      },
      "classified_acoustic_tone": "Alarming",
      "prosody_scalar": -0.9
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Extract the 2-second audio chunk of a known "yell" video. Run the Python `funasr` emotion2vec+ inference and print the 9-class emotion scores to the console. Listen to the `.wav` slice to manually confirm the model caught the exact peak of the shout.
- **Batch Test**: Run over 100 clips. Verify that videos classified as "Alarming" correlate with high `angry` + `fearful` scores and high delta in `max_amplitude_dbFS`. Ensure audio loading does not bottleneck the **Mac Studio (M4 Max, 64 GB unified memory)**; chunked torchaudio streaming is no longer strictly required (Resolved Issue #12 confirms the FFmpeg subprocess path is adequate), but the test should still verify steady-state memory does not balloon over a 100-clip run.

## 🚀 Implementation Accomplishments

The Acoustic Prosody Layer has been implemented and successfully integrated:
- **Audio Extraction**: Designed to use `ffmpeg` as a subprocess to rapidly slice and resample bounded audio windows (16kHz, mono) directly into temporary `.wav` files. This avoids loading multi-gigabyte video files entirely into memory.
- **Acoustic Features**: Leveraged `librosa` to compute deterministic features (`max_amplitude_dbFS` and `pitch_contour_variance`).
- **SER Model Integration**: Established the pipeline structure for `funasr.AutoModel` running `iic/emotion2vec_plus_large`. It extracts 9-class probabilities seamlessly.
- **Robust Heuristics**: Finalized the mathematical mappings to correlate acoustic payload probabilities with discrete scalar outcomes ("Alarming", "Soothing", "Discouraging", "Neutral").
- **Verification Framework**: Fully mocked test suite using `pytest` implemented in `tests/test_layer_03c.py` ensuring pipeline math and schema output validations pass successfully.

## 🧪 Resolved Issues & Implementation Refinements

1. **No-Audio Clips Emitted a *Confident* Neutral (Resolved - June 08)**:
   - **Problem**: When the source `.mp4` had no audio stream (29 of 50 clips in the June 8 50-clip smell test), `_extract_audio_chunk` returned `None` and `_process_task` (`src/layer_03c_acoustic_prosody/pipeline.py`) emitted a stub with `max_amplitude_dbFS = -100.0` **but also** `dominant_emotion = "neutral"`, `dominant_emotion_confidence = 1.0`, `classified_acoustic_tone = "Neutral"` — byte-for-byte identical to a clip whose audio was genuinely quiet/neutral. Downstream multi-layer fusion could not distinguish "no acoustic data" from "a confident neutral acoustic reaction," so silent clips would be read as confident evidence; the `-100.0` dBFS sentinel was the only (easily-missed) tell.
   - **Solution** (Option A): The no-audio stub branch now emits an explicit no-data marker — `audio_present: false`, `dominant_emotion_confidence: 0.0`, and all-zero `emotion_scores` — instead of a confident neutral; the success path emits `audio_present: true` for symmetry. This mirrors 03a's `NoFace` target and the 03b emotion-confidence honesty gate, letting downstream fusion exclude `audio_present == false` tasks rather than treat them as evidence. Covered by `test_no_audio_task_emits_explicit_marker` and a new `audio_present` assertion in `test_schema_conformance`.

2. **Loud Ambient Task Noise Produced a False "Alarming" (Resolved - June 08)**:
   - **Problem**: `_classify_acoustic_tone` computed `alarming_score = angry + fearful + high_volume_bonus` (`+0.3` when `dBFS > -20`). On the June 8 run both clips classified **Alarming** had `angry + fearful ≈ 0` — `573fc64b` "Cooking" (`other = 0.82`, dBFS −11.9) and `04f2cec1` "Cleaning/laundry" (`disgusted = 0.41`, dBFS −15.8) — i.e. the volume bonus fired on loud *mechanical task noise* (clatter), not an alarmed voice. In egocentric footage the loudest sound is usually the wearer's own object manipulation, so raw loudness is a poor alarm proxy.
   - **Solution** (task-conditional refinement of Option A, per the user's selection): rather than a single global `angry+fearful` floor, the volume bonus is made **conditional on the task being performed**, because some tasks are unavoidably loud. Added `high_volume_expected_task_keywords` to `Layer03cConfig` (cooking, laundry, cleaning, blacksmith, construction/renovation, yardwork/shoveling, machinery/power tools, …) and `_task_expects_high_volume(task_label)` (case-insensitive substring match). `_classify_acoustic_tone` gained an `expect_high_volume` flag and withholds `high_volume_bonus` for inherently-loud tasks, so loud mechanical noise can no longer fabricate an Alarming. Re-classifying the recorded 50-run data confirmed the fix is surgical: **exactly the two false-Alarming clips flipped Alarming → Neutral, with no other classification changed.** Covered by `test_task_expects_high_volume_keyword_match` and `test_volume_bonus_suppressed_for_loud_task_end_to_end`.

3. **Ambient Audio Not Attributed to the Bystander (Resolved - June 08)**:
   - **Problem**: 03c scores the whole 16 kHz mono window with no speaker separation, so in egocentric footage the detected emotion may belong to the camera-wearer (their own speech / breathing / object-handling), not the bystander reacting to the task — a conceptual attribution limit (the acoustic analog of 03e Issue 1's gaze-vs-head-pose).
   - **Solution** (Option A — scope, not a new model): the layer is now explicitly documented as **ambient acoustic context, not a bystander-attributed verdict** (see the Scope note under *Output Schema and Integration*). Downstream fusion (Layer 04 / the 03 aggregator) must consume `prosody_scalar` only as **corroboration**, weighted by agreement with a per-bystander visual layer (03b face emotion) for the same task/window — empirically viable, since the June 8 run already showed 03b/03c agreement on `43bd06f3` (sad). No 03c pipeline behavior change; the `audio_present` marker (Resolved #1) keeps no-data tasks out of that fusion. Speaker diarization / source attribution was deferred — mono egocentric audio usually cannot separate sources.

4. **Soothing Fabricated From a Sub-Noise SER Read — SER-Confidence Floor (Resolved - June 09)**:
   - **Problem**: Surfaced by the June 9 post-fix 50-clip re-spot-check (`e2e_reports/2026_06_09_layer03c_50_postfix/`). `_classify_acoustic_tone` gated only on the *composite* tone score (`min_dominant_tone_score = 0.3`), never on the SER *dominant-emotion confidence*. Clip `072e3481` "Walking on street" had `happy = 0.317` barely edging `neutral = 0.261` — a near coin-flip — yet `soothing_score = 0.317 + pitch_contour_variance(0.65) × 0.5 = 0.64` cleared the gate and emitted **Soothing `+1.0`**, the same full-strength scalar as the genuine `6b3988dd` (`happy = 0.994`). The pitch-variance bonus could assemble a winning tone out of softmax noise — the symmetric twin of the volume-bonus false-Alarming closed in Resolved #2, which had only fixed the Alarming side.
   - **Solution** (Option A): Added `Layer03cConfig.min_ser_confidence` (default **0.5**). `_classify_acoustic_tone` now forces **Neutral** whenever even the dominant SER class is below the floor (symmetric across Alarming/Soothing/Discouraging), and the `pitch_variance_soothing_weight` term contributes **only when `happy` itself clears the floor** — melody corroborates a confident happy, it cannot create one. Mirrors 03b's `MIN_EMOTION_CONF` honesty gate. Re-classifying the recorded June 9 run confirmed the fix is surgical: **exactly `072e3481` flipped Soothing → Neutral; all genuine Soothings (happy 0.53–0.99) and the sad-0.90 Discouraging were unchanged.** Covered by `test_soothing_blocked_below_ser_confidence_floor` and `test_pitch_term_requires_confident_happy`; `test_config_volume_cutoff_override_changes_alarming_bonus` and `test_volume_bonus_suppressed_for_loud_task_end_to_end` updated to clear the floor so their volume-bonus mechanics remain reachable.

5. **`disgusted` Was Unmapped — Confident Disgust Read as Neutral (Resolved - June 09)**:
   - **Problem**: The tone heuristic consumed only `angry`/`fearful`/`happy`/`surprised`/`sad`/`neutral`; the SER's `disgusted` (and `other`/`unknown`) classes fed no tone score. On the June 9 re-run, `04f2cec1` "Cleaning/laundry" had `disgusted = 0.41` as its top emotion yet classified Neutral — silently discarding a negative-valence vocalization class (a recoiling "ugh") with clear developmental meaning.
   - **Solution** (Option A): `negative_score` now includes `disgusted` alongside `sad` (`negative_score = sad + disgusted + low_volume_bonus`), so **confident** disgust yields Discouraging (−0.5), consistent with the existing sad→Discouraging mapping. The Resolved #4 SER floor intentionally keeps *weak* disgust from over-firing — `04f2cec1` (0.41 < 0.5) correctly stays Neutral, validated on the recorded run (0 unintended flips). `other`/`unknown` remain unmapped by design: they carry no valence. Covered by `test_confident_disgust_maps_to_discouraging` and `test_weak_disgust_stays_neutral_via_floor`.

## ⚠️ Unresolved Issues & Suggestions

_None at this time — the June 9 post-fix re-spot-check findings (SER-confidence floor, disgust mapping) have been resolved above (Resolved #4–5)._