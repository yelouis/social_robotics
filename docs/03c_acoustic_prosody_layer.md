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
