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

*None at this time.*