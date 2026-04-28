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

## 🧪 Resolved Issues & Implementation Refinements (April 2026)

During code review and validation of the initial implementation, the following critical issues were identified and resolved:

1. **Missing `__init__.py` (Resolved)**:
   - **Problem**: The `src/layer_03c_acoustic_prosody/` directory lacked an `__init__.py` file. While the CLI entrypoint (`if __name__ == "__main__"`) worked, the module could not be imported by the test suite (`from src.layer_03c_acoustic_prosody.pipeline import ...`) or by any future cross-layer consumer, breaking Python's package resolution.
   - **Solution**: Added an empty `__init__.py` to match the pattern established by all sibling layer modules (`03a`, `03b`).

2. **Temp File Collision on Concurrent Runs (Resolved)**:
   - **Problem**: The original `_extract_audio_chunk` used a deterministic filename constructed from `os.path.basename(video_path)` + start/end seconds in the system `/tmp` directory (e.g., `temp_audio_clip.mp4_1.0_3.0.wav`). If two pipeline instances processed the same video concurrently, or if a previous run crashed leaving the file behind, the second process would silently overwrite or read a stale/partial `.wav`.
   - **Solution**: Replaced with `tempfile.mkstemp(suffix=".wav", prefix="prosody_")`, which returns a guaranteed-unique, OS-assigned path. The file descriptor is immediately closed so `ffmpeg` can write to the path.

3. **`librosa.pyin` Voiced Flag Misuse (Resolved)**:
   - **Problem**: The original code did `f0[voiced_flag]` to filter pitch values to only voiced frames. However, `librosa.pyin` returns `voiced_flag` as a float probability array (not a boolean mask), and `f0` contains `NaN` for unvoiced frames. Using a float array as an index produced incorrect filtering and potential `IndexError` on some numpy versions.
   - **Solution**: Changed to `f0[~np.isnan(f0)]`, which correctly filters to only the voiced frames regardless of the `voiced_flag` dtype.

4. **Empty WAV Header-Only Guard Missing (Resolved)**:
   - **Problem**: When `ffmpeg` extracts audio from a reaction window that extends beyond the video's actual duration (common in the manifest — e.g., video `61EI0` has duration 17.09s but task t_04's window is `[18.04, 22.04]`), it produces a valid `.wav` file containing only the 44-byte WAV header and zero audio samples. The original code checked `os.path.getsize(out_wav) > 0`, which passed for these header-only files, causing downstream `librosa` and `funasr` to process silence or crash.
   - **Solution**: Changed the guard to `os.path.getsize(out_wav) > 44` (the standard WAV header size), ensuring only files with actual audio data are forwarded to the models.

5. **No Error Log File — Architecture Policy Violation (Resolved)**:
   - **Problem**: The `03_social_layer_architecture.md` Failure & Resumability Policy requires every layer to log per-video errors to a `<layer>_errors.json` file. The initial implementation silently swallowed per-video exceptions via bare `continue` statements, making debugging impossible for batch runs.
   - **Solution**: Added `_log_error()` method (mirroring `03b`'s pattern) that writes structured error entries with tracebacks to `03c_acoustic_prosody_errors.json`. Wrapped the per-video task loop in `try/except` to catch and log errors without halting the batch.

6. **Missing Python Dependencies — `funasr`, `librosa`, `torchaudio` (Resolved)**:
   - **Problem**: `funasr` and `librosa` were not installed in the workspace virtual environment. The pipeline silently degraded to producing all-Neutral results, making it appear functional while producing no real analysis.
   - **Solution**:
     - Installed all required packages: `venv/bin/pip install librosa funasr torchaudio`.
     - **Hardened startup validation**: Replaced the silent `try/except ImportError` fallbacks with explicit `RuntimeError` raises that include actionable install commands (e.g., `"Install with: venv/bin/pip install funasr torchaudio"`). The pipeline now fails fast at `__init__` time if any dependency is missing.
     - Added 3 new unit tests (`test_init_raises_without_ffmpeg`, `test_init_raises_without_librosa`, `test_init_raises_without_funasr`) verifying that the startup validation triggers correctly.

7. **First Run Latency — emotion2vec+ Model Download (Resolved)**:
   - **Problem**: The `emotion2vec+ large` model weights (~600MB) would be downloaded from ModelScope on the very first execution, causing an apparent hang with no user feedback.
   - **Solution**: Pre-fetched the model weights by running a one-shot initialization script. The weights are now cached at `~/.cache/modelscope/hub/models/iic/emotion2vec_plus_large/`. Subsequent runs use `disable_update=True` for instantaneous offline loading.

8. **FFmpeg Not Installed (Resolved)**:
   - **Problem**: The Mac mini did not have `ffmpeg` installed. Audio slicing silently failed and every task produced a "Neutral" stub.
   - **Solution**:
     - Installed via `brew install ffmpeg` (v8.1 now at `/opt/homebrew/bin/ffmpeg`).
     - Added `shutil.which("ffmpeg")` check in `_init_models()` that raises `RuntimeError` with install instructions if the binary is absent. This prevents silent degradation on a fresh machine.

