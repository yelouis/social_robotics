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
Use `FFmpeg` or `librosa` to extract the audio track spanning the exact `task_reaction_window_sec`. Resample the audio to 16kHz, as required by most SOTA speech models.

### 2. Speech Emotion Recognition (emotion2vec+)
Instead of transcribing words (which VLMs can do), we run a State-of-the-Art Speech Emotion Recognition (SER) model to capture the acoustic flavor:
- **Primary SER Model**: **emotion2vec+ large** (`iic/emotion2vec_plus_large`) via the FunASR framework. MIT licensed. ACL 2024. Achieves SOTA on IEMOCAP and multiple languages.
- **Mechanism**: Feed the 16kHz audio slice into emotion2vec+ to extract a **9-class emotion probability distribution** (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown) plus an optional 768-dim emotion embedding.
- **Heuristic Mapping**:
  - High `angry` + high `fearful` + Sudden Volume Spike = **Alarming / Deterrent** (e.g., a sharp "Hey! Stop!").
  - High `happy` + high `surprised` + Melodic Pitch Contour = **Soothing / Encouraging** (e.g., "Good job!").
  - High `sad` + Low Volume = **Discouraging / Negative**.
- **Supplementary Audio Event Detection**: Optionally run **SenseVoice** alongside for detecting non-speech audio events (laughter, applause, crying, coughing) that provide additional social context.

### 3. Pitch Contour & Amplitude Variance (Librosa)
To supplement the emotion2vec+ embedding, calculate deterministic acoustic features using `librosa`:
- **Volume Spike (dB)**: Measure the delta between the pre-climax ambient noise floor and the peak amplitude within the reaction window.
- **Pitch Variance**: Calculate the fundamental frequency (f0). A highly melodic voice has smooth variance, while a bark or yell has abrupt, broken pitch contours.

---

## 📤 Output Schema and Integration
The layer outputs an isolated JSON mapping the acoustic payload per task.

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
        "dominant_emotion_confidence": 0.72
      },
      "classified_acoustic_tone": "Alarming",
      "prosody_scalar": -0.9
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Extract the 2-second audio chunk of a known "yell" video. Run the Python `funasr` emotion2vec+ inference and print the 9-class emotion scores to the console. Listen to the `.wav` slice to manually confirm the model caught the exact peak of the shout.
- **Batch Test**: Run over 100 clips. Verify that videos classified as "Alarming" correlate with high `angry` + `fearful` scores and high delta in `max_amplitude_dbFS`. Ensure audio loading does not bottleneck the **24GB RAM Mac mini M4 Pro**; use chunked torchaudio streaming interfaces where possible.

## 🚀 Implementation Accomplishments

The Acoustic Prosody Layer has been implemented and successfully integrated:
- **Audio Extraction**: Designed to use `ffmpeg` as a subprocess to rapidly slice and resample bounded audio windows (16kHz, mono) directly into temporary `.wav` files. This avoids loading multi-gigabyte video files entirely into memory.
- **Acoustic Features**: Leveraged `librosa` to compute deterministic features (`max_amplitude_dbFS` and `pitch_contour_variance`).
- **SER Model Integration**: Established the pipeline structure for `funasr.AutoModel` running `iic/emotion2vec_plus_large`. It extracts 9-class probabilities seamlessly.
- **Robust Heuristics**: Finalized the mathematical mappings to correlate acoustic payload probabilities with discrete scalar outcomes ("Alarming", "Soothing", "Discouraging", "Neutral").
- **Verification Framework**: Fully mocked test suite using `pytest` implemented in `tests/test_layer_03c.py` ensuring pipeline math and schema output validations pass successfully.

## 🧪 Resolved Issues & Implementation Refinements

1. **Missing `__init__.py` (Resolved - April 28)**:
   - **Problem**: The `src/layer_03c_acoustic_prosody/` directory lacked an `__init__.py` file, preventing the module from being imported by the test suite or cross-layer consumers.
   - **Solution**: Added an empty `__init__.py` to match the project's modular package structure.

2. **Temp File Collision on Concurrent Runs (Resolved - April 28)**:
   - **Problem**: Audio extraction used deterministic filenames in `/tmp`, leading to silent overwrites or reading stale data when processing multiple videos concurrently.
   - **Solution**: Replaced hardcoded paths with `tempfile.mkstemp(suffix=".wav", prefix="prosody_")` to ensure guaranteed-unique, OS-assigned paths.

3. **`librosa.pyin` Voiced Flag Misuse (Resolved - April 28)**:
   - **Problem**: The code used a float probability array as a boolean mask for pitch filtering, causing incorrect calculations and potential `IndexError`.
   - **Solution**: Changed filtering logic to `f0[~np.isnan(f0)]`, which correctly isolates voiced frames across all numpy versions.

4. **Empty WAV Header-Only Guard Missing (Resolved - April 28)**:
   - **Problem**: FFmpeg produces 44-byte header-only files when reaction windows exceed video duration. The original `getsize > 0` check allowed these invalid files through, crashing downstream models.
   - **Solution**: Updated guard to `os.path.getsize(out_wav) > 44` to ensure only files with actual sample data are processed.

5. **No Error Log File — Architecture Policy Violation (Resolved - April 28)**:
   - **Problem**: The layer silently swallowed per-video exceptions, violating the architectural requirement for persistent error logging (`<layer>_errors.json`).
   - **Solution**: Implemented a `_log_error()` method and wrapped the processing loop in a `try/except` block to record structured tracebacks to `03c_acoustic_prosody_errors.json`.

6. **Missing Python Dependencies — `funasr`, `librosa`, `torchaudio` (Resolved - April 29)**:
   - **Problem**: Core ML dependencies were missing from the environment, causing the pipeline to silently degrade to all-Neutral stubs.
   - **Solution**: Installed required packages and implemented explicit `RuntimeError` raises at initialization time to ensure the pipeline "fails fast" if dependencies are absent.

7. **First Run Latency — emotion2vec+ Model Download (Resolved - April 29)**:
   - **Problem**: Automatic downloading of the ~600MB SER model on the first run caused significant cold-start latency and appeared to hang.
   - **Solution**: Pre-fetched model weights and configured the pipeline to use `disable_update=True` for instantaneous offline loading.

8. **FFmpeg Binary Missing (Resolved - April 29)**:
   - **Problem**: The Mac mini lacked `ffmpeg`, causing all audio slicing operations to fail silently.
   - **Solution**: Installed `ffmpeg` via Homebrew and added a `shutil.which("ffmpeg")` startup check to enforce its presence.

9. **Test Manifest Latency — Ego4D Full-Length Videos (Resolved - April 30)**:
   - **Problem**: E2E validation against full Ego4D clips (>900s) was prohibitively slow for iterative development.
   - **Solution**: Developed a `mock_filtered_manifest.json` with pre-defined task windows and targeted shorter clips to reduce validation cycles to sub-minute durations.

10. **SER Label Parsing Failure — 'Chinese/English' Format (Resolved - April 30)**:
    - **Problem**: The `emotion2vec+` model returns composite labels (e.g., `'生气/angry'`), which mismatched the English-only lookup keys and resulted in `0.0` scores.
    - **Solution**: Updated `_run_ser_model` to split labels and extract the English component, ensuring correct mapping to social heuristics.

11. **Supplementary Audio Event Detection (SenseVoice) (Resolved - May 05)**:
    - **Problem**: The pipeline exclusively used `emotion2vec+` for 9-class speech emotion recognition, missing critical non-speech social cues like laughter, applause, and crying.
    - **Solution**: Integrated a sequential `SenseVoiceSmall` inference pass that runs conditionally when `emotion2vec+`'s dominant emotion confidence is below 0.6. Extracted non-speech event labels are now appended as a new `audio_events` array in the output schema.

12. **torchaudio Streaming for Long Clips (Resolved - May 05)**:
    - **Problem**: The pipeline extracts audio via an `ffmpeg` subprocess to slice and resample into a temporary `.wav` file on disk, which was flagged as potentially inefficient compared to `torchaudio.io.StreamReader` zero-copy streaming.
    - **Solution**: Deferred and retained the current `ffmpeg` architecture. Since `task_reaction_window_sec` bounding ensures maximum clip sizes of ~6.0s (192KB), temporary file I/O overhead is negligible. This avoids introducing beta PyTorch streaming APIs and preserves system stability.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Partial Fail-Fast — `AutoModel` Load Errors Silently Swallowed
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03c_acoustic_prosody/pipeline.py` lines 64-79. The previously-resolved Issue #6 (April 29) raises `RuntimeError` only on `ImportError`. The outer `except Exception as e` clause at lines 78-79 catches every other failure mode of `AutoModel(model="iic/emotion2vec_plus_large", ...)` and `AutoModel(model="iic/SenseVoiceSmall", ...)` — corrupted weights, network failures during a forced re-download, OOM at model load, missing ModelScope cache, etc. — and merely logs them. Because `self.funasr_available` remains `False` and `self.model` remains `None`, `_run_ser_model` (lines 188-189) silently returns the all-Neutral default, exactly the symptom Issue #6 was supposed to eliminate.

**Option A (recommended)**: **Re-raise as `RuntimeError` with actionable context** — Replace `logger.error(f"Failed to load funasr model: {e}")` with `raise RuntimeError(f"Failed to load funasr SER model: {e}. Verify the ModelScope cache at ~/.cache/modelscope or re-run with disable_update=False to force re-download.") from e`.
  - *Pros*: Restores the fail-fast contract from Issue #6 across all failure modes, not just `ImportError`; preserves the original exception chain via `from e`; surfaces operational problems at startup rather than after thousands of mis-classified videos.
  - *Cons*: A transient ModelScope outage now hard-fails the entire pipeline; small risk if the user wanted graceful degradation for partial-stack environments (but that intent contradicts Issue #6).

**Option B**: **Keep degradation but mark output explicitly** — Allow init to proceed, but add a top-level `"layer_status": "degraded"` field on every output record when `self.funasr_available is False`, and stamp every emotion-score block with `"source": "fallback_neutral"` to make the degraded state visible in downstream analysis.
  - *Pros*: Preserves liveness on partial-stack hosts; downstream layers (03b emotion fusion, 03g shared reality) can detect and weight-down degraded records.
  - *Cons*: Doubles output schema surface area; downstream consumers need new branches; conflicts with the architectural posture established by Issue #6.

Your selection: Proceed with Option A.

---

### Issue 2: Temp `.wav` Leak on Mid-Pipeline Exception
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03c_acoustic_prosody/pipeline.py`. `_process_task` calls `_safe_remove(wav_path)` only at line 338, after a successful run through `_extract_librosa_features` (line 325), `_run_ser_model` (line 326), and `_run_sensevoice_model` (line 333). If any of those three raises an unexpected exception (e.g., librosa OOM on a malformed WAV, funasr CUDA allocation failure, SenseVoice tokenizer crash), the exception propagates to the outer `try/except` in `run()` (lines 381-388), which logs it via `_log_error` and `continue`s — but never reaches the line-338 cleanup. Each such failure orphans the `prosody_<random>.wav` file in the OS temp directory. On a 10K-video manifest with even a 1% fail rate, that's 100 leaked files; on macOS `/tmp` survives until reboot, and on Linux `/var/tmp` survives indefinitely.

**Option A (recommended)**: **Wrap `_process_task` body in `try / finally: _safe_remove(wav_path)`** — Move the cleanup into a `finally` block that runs unconditionally after `wav_path` is assigned. Guard against the early-return-before-extraction path by initializing `wav_path = None` and gating `_safe_remove` on `wav_path is not None`.
  - *Pros*: Guarantees cleanup on every code path including exception, early return, and success; minimal diff; no behavior change on the happy path.
  - *Cons*: Requires careful placement of the `wav_path = None` initializer above the try-block; must verify `_safe_remove(None)` is either prevented or made idempotent.

**Option B**: **Use a `contextlib.contextmanager` for the temp file** — Refactor `_extract_audio_chunk` into a context manager (`@contextmanager def _temp_audio_chunk(...)`) that yields the path and `finally`-cleans on `__exit__`.
  - *Pros*: Most idiomatic Python; the `with` block in `_process_task` makes the cleanup obligation visually obvious; reusable from future tests.
  - *Cons*: Larger refactor; changes the public-ish helper signature; harder to compose with the early-return-on-extraction-failure stub branch (lines 308-323).

**Option C**: **Add a startup-time sweep of stale `prosody_*.wav` files** — In `__init__`, glob `tempfile.gettempdir()` for `prosody_*.wav` older than 1 hour and unlink them.
  - *Pros*: Self-healing across runs; cleans up debt from prior crashes too.
  - *Cons*: Treats the symptom, not the cause; introduces a startup glob over `/tmp` that other processes' files share; potential race with concurrent pipeline instances.

Your selection: Proceed with Option A.

---

### Issue 3: SenseVoice Event Detection Uses Fragile Substring Matching
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03c_acoustic_prosody/pipeline.py` lines 232-245. `_run_sensevoice_model` lowercases the SenseVoice `text` field and then uses raw `in` substring matching for both bracketed event tokens (`<|laughter|>`) and bare English words (`laughter`, `applause`, `cough`, `crying`, `sneeze`). The bare-word arm produces false positives on transcribed speech: any utterance containing "slaughter*", "laughtered", "applauded", "coughed", "crying out", "cried", or "sneezed at" matches. SenseVoice is a transcription-capable model, so its `text` regularly contains spoken-word content alongside the special-token annotations — the bare substrings are not reliable event signals.

**Option A (recommended)**: **Match only the bracketed event tokens** — Drop the `or "laughter" in text_lower`-style bare-word arms; check exclusively for `<|laughter|>`, `<|applause|>`, `<|cough|>`, `<|crying|>`, `<|sneeze|>`. Compile a fixed map `EVENT_TOKENS = {"<|laughter|>": "laughter", ...}` and iterate.
  - *Pros*: Eliminates the false-positive class entirely; aligns with how SenseVoice actually emits non-speech events; simpler code; still O(n) on text length.
  - *Cons*: Slightly stricter — if a future SenseVoice version changes the bracket convention (e.g., `[laughter]`), this misses events; mitigated by adding a comment pinning the assumption to the current funasr SenseVoice version.

**Option B**: **Use word-boundary regex** — Replace `in` with `re.search(r"\blaughter\b", text_lower)` for each event word.
  - *Pros*: Catches the bare-word events while excluding "slaughter", "laughtered", etc.
  - *Cons*: Still false-positives on legitimate transcriptions like "the laughter died down"; doesn't address the root issue that bare words ≠ event annotations.

Your selection: Proceed with Option A.

---

### Issue 4: SenseVoice Model Eagerly Loaded Despite Conditional Use
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03c_acoustic_prosody/pipeline.py` line 70: `self.sensevoice_model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True)` runs unconditionally at init. The model is invoked only inside the `if dominant_confidence < 0.6` branch at line 332 of `_process_task`. For any manifest where emotion2vec+ produces mostly high-confidence outputs (the common case on clean audio), SenseVoice occupies ~150-200MB of resident memory for the entire pipeline lifetime without ever running. On the 24GB Mac mini M4 Pro, this is meaningful pressure when 03c runs alongside 03a (DeiT/CLIP attention) and 03b (PyFeat).

**Option A (recommended)**: **Lazy-load SenseVoice on first low-confidence sample** — Replace the eager `AutoModel(...)` call at line 70 with a `self.sensevoice_model = None` placeholder. In `_run_sensevoice_model`, lazy-init via `if self.sensevoice_model is None: self.sensevoice_model = AutoModel(...)` on the first call.
  - *Pros*: Zero memory cost on high-confidence-only manifests; init latency moves to first-use only; preserves the existing call sites unchanged.
  - *Cons*: First low-confidence sample pays the model-load cost (one-time per pipeline run); the 03c init no longer fails fast on broken SenseVoice weights — that failure is deferred to the first invocation.

**Option B**: **Make SenseVoice opt-in via constructor flag** — Add `enable_sensevoice: bool = True` to `__init__` and gate both the model load and the conditional invocation on it.
  - *Pros*: Explicit control for orchestration scripts; trivial to disable in memory-constrained runs; easy to A/B test the SenseVoice contribution.
  - *Cons*: Adds a config knob the user must remember; doesn't solve the eager-load cost in the default `True` case.

**Option C**: **Drop SenseVoice entirely and surface the limitation** — Remove the SenseVoice model and the `audio_events` array. Document that non-speech event detection is out of scope for 03c and would require a future 03h layer.
  - *Pros*: Maximum memory savings; smaller code surface; makes 03c a pure prosody/SER layer with single responsibility.
  - *Cons*: Reverses the previously-resolved Issue #11 (May 5); loses an integration that may be load-bearing for downstream social-cue fusion.

Your selection: Proceed with Option A.

---

### Issue 5: Hardcoded Acoustic Heuristic Thresholds
**Status**: ⚠️ Confirmed Unresolved — Verified in `src/layer_03c_acoustic_prosody/pipeline.py` `_classify_acoustic_tone` and `_process_task`. The following constants are baked into source: `-20.0` dBFS high-volume cutoff (line 266), `-35.0` dBFS low-volume cutoff (line 267), `0.3` volume bonus weight (lines 269, 271), `0.5` pitch-variance weight on soothing score (line 270), `0.3` minimum dominant-tone score for non-Neutral classification (line 292), `0.6` `dominant_emotion_confidence` threshold for SenseVoice gating (line 332), `10000.0` pitch-variance normalization divisor (line 165). None of these reference `src/config.py` or any configuration source. Empirical tuning against new datasets requires source edits, which complicates A/B comparisons and makes the heuristics opaque to non-Python collaborators.

**Option A (recommended)**: **Promote thresholds to a `Layer03cConfig` dataclass** — Define a frozen dataclass in `src/config.py` (or `src/layer_03c_acoustic_prosody/config.py`) holding all seven constants with their current values as defaults. Accept an optional `config: Layer03cConfig = Layer03cConfig()` parameter in `__init__` and reference fields throughout `_classify_acoustic_tone` / `_process_task`.
  - *Pros*: Tuning becomes data, not code; the dataclass docstring becomes the single source of truth for what each threshold means; trivial to override in tests via `Layer03cConfig(high_volume_dbfs=-15.0)`; aligns with the project's existing `config.py` pattern.
  - *Cons*: Diff touches every reference site; needs a one-line migration in any orchestration script that currently calls `AcousticProsodyPipeline(...)` (defaults preserve current behavior).

**Option B**: **Load thresholds from a YAML/JSON config file** — Add a `--config` CLI flag pointing at a config file, parse with `pydantic` for validation.
  - *Pros*: External config is the most flexible; allows checked-in named profiles (`profile_indoor.yaml`, `profile_outdoor.yaml`); non-Python users can tune.
  - *Cons*: Pulls in YAML parsing as a dependency (or hand-rolls JSON); overkill for seven scalars; harder to keep in sync with code-side validation.

**Option C**: **Defer — current thresholds are empirically tuned and unlikely to need frequent changes** — Document the thresholds and their rationale in this doc, but leave them in source.
  - *Pros*: Zero diff; preserves whatever empirical tuning produced these values.
  - *Cons*: First request to tune for a new dataset re-opens this issue; the implicit "they're empirically tuned" claim is currently uncited in code comments.

Your selection: Proceed with Option A.