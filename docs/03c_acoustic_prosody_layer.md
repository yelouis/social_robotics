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
  - High `sad` + Low Volume = **Discouraging / Negative**

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
- **Batch Test**: Run over 100 clips. Verify that videos classified as "Alarming" correlate with high `angry` + `fearful` scores and high delta in `max_amplitude_dbFS`. Ensure audio loading does not bottleneck the **Mac Studio (M4 Max, 64 GB unified memory)**; chunked torchaudio streaming is no longer strictly required (Resolved Issue #12 confirms the FFmpeg subprocess path is adequate), but the test should still verify steady-state memory does not balloon over a 100-clip run.

## 🚀 Implementation Accomplishments

The Acoustic Prosody Layer has been implemented and successfully integrated:
- **Audio Extraction**: Designed to use `ffmpeg` as a subprocess to rapidly slice and resample bounded audio windows (16kHz, mono) directly into temporary `.wav` files. This avoids loading multi-gigabyte video files entirely into memory.
- **Acoustic Features**: Leveraged `librosa` to compute deterministic features (`max_amplitude_dbFS` and `pitch_contour_variance`).
- **SER Model Integration**: Established the pipeline structure for `funasr.AutoModel` running `iic/emotion2vec_plus_large`. It extracts 9-class probabilities seamlessly.
- **Robust Heuristics**: Finalized the mathematical mappings to correlate acoustic payload probabilities with discrete scalar outcomes ("Alarming", "Soothing", "Discouraging", "Neutral").
- **Verification Framework**: Fully mocked test suite using `pytest` implemented in `tests/test_layer_03c.py` ensuring pipeline math and schema output validations pass successfully.

## ⚠️ Unresolved Issues & Suggestions

_None at this time — all tracked issues have been successfully integrated into the system architecture._