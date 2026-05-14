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
- **Batch Test**: Run over 100 clips. Verify that videos classified as "Alarming" correlate with high `angry` + `fearful` scores and high delta in `max_amplitude_dbFS`. Ensure audio loading does not bottleneck the **Mac Studio (M4 Max, 64 GB unified memory)**; chunked torchaudio streaming is no longer strictly required (Resolved Issue #12 confirms the FFmpeg subprocess path is adequate), but the test should still verify steady-state memory does not balloon over a 100-clip run.

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

13. **Partial Fail-Fast — `AutoModel` Load Errors Silently Swallowed (Resolved - May 07)**:
    - **Problem**: The `_init_models` outer `except Exception as e` clause caught every non-`ImportError` failure mode of `AutoModel(...)` (corrupt weights, ModelScope outage, OOM, missing cache) and merely logged it. `self.funasr_available` remained `False` and `_run_ser_model` silently returned the all-Neutral default — re-introducing the symptom Issue #6 was supposed to eliminate.
    - **Solution**: Replaced the swallow-and-log with `raise RuntimeError("Failed to load funasr SER model: {e}. Verify the ModelScope cache at ~/.cache/modelscope or re-run with disable_update=False to force re-download.") from e`. The original exception is preserved via `from e`. Added `test_init_raises_when_automodel_load_fails` to lock in the new contract.
    - **Model Instructions/Context**: All 03 inference layers must fail fast on load errors — silent degradation to neutral/zero output corrupts every downstream score without surfacing the root cause. Use `raise RuntimeError(..., context) from e` rather than `logger.error` for any model-load failure.

14. **Temp `.wav` Leak on Mid-Pipeline Exception (Resolved - May 07)**:
    - **Problem**: `_process_task` only called `_safe_remove(wav_path)` after a successful happy-path completion. Any exception from `_extract_librosa_features`, `_run_ser_model`, or `_run_sensevoice_model` propagated to `run()`'s outer `try/except`, leaving the `prosody_<random>.wav` orphaned in the OS temp directory until reboot (macOS) or indefinitely (Linux `/var/tmp`).
    - **Solution**: Wrapped the `_process_task` body in `try / finally`, with `wav_path = None` initialized above the `try` so the `finally`'s `if wav_path is not None: self._safe_remove(wav_path)` is safe on every exit path including the early `task_reaction_window_sec` validation return. Added `test_wav_cleaned_up_on_processing_exception` to enforce cleanup under simulated librosa OOM.
    - **Model Instructions/Context**: Any helper that allocates an OS-level temp resource (file, GPU buffer, subprocess handle) must release it from a `finally` block in the calling scope, not from a trailing line in the happy path. This pattern should be reused for any future audio/video chunk extractors.

15. **SenseVoice Event Detection Uses Fragile Substring Matching (Resolved - May 07)**:
    - **Problem**: `_run_sensevoice_model` previously matched both bracketed event tokens (`<|laughter|>`) and bare English words (`laughter`, `applause`, `cough`, `crying`, `sneeze`) via raw substring `in`. Because SenseVoice is transcription-capable, its `text` field regularly contains spoken-word content; "slaughter", "laughtered", "applauded", "the laughter died down", "cried", and "sneezed at" all produced false-positive event labels.
    - **Solution**: Dropped the bare-word arms entirely. Introduced a module-level `SENSEVOICE_EVENT_TOKENS = {"<|laughter|>": "laughter", "<|applause|>": "applause", "<|cough|>": "cough", "<|crying|>": "crying", "<|sneeze|>": "sneeze"}` and replaced the if-chain with a single comprehension over the map. Added `test_sensevoice_only_matches_bracketed_tokens` and `test_sensevoice_matches_bracketed_token_emission` to lock in the bracketed-only semantics.
    - **Model Instructions/Context**: The token map is keyed on the funasr SenseVoice bracket convention as of May 2026. If a future SenseVoice version changes the bracket grammar (e.g., `[laughter]`), update `SENSEVOICE_EVENT_TOKENS` rather than reintroducing bare-word matching.

16. **SenseVoice Model Eagerly Loaded Despite Conditional Use (Resolved - May 07)**:
    - **Problem**: `self.sensevoice_model = AutoModel(model="iic/SenseVoiceSmall", ...)` ran unconditionally during `_init_models`, but SenseVoice is only invoked when `emotion2vec+`'s dominant-emotion confidence falls below the gating threshold. On clean-audio manifests this paid ~150-200MB of resident memory for the entire pipeline lifetime without ever firing the model.
    - **Solution**: Replaced the eager `AutoModel(...)` call with a `self.sensevoice_model = None` placeholder and stashed `AutoModel` itself on `self._AutoModel` for deferred construction. `_run_sensevoice_model` now lazy-initializes the model on its first invocation, so high-confidence-only runs never pay the load cost. Added `test_sensevoice_lazy_loaded_on_first_low_confidence` to verify both first-call construction and second-call reuse.
    - **Model Instructions/Context**: Conditional, secondary models in this codebase should be lazy-loaded by default. The first-use latency penalty is a one-time, per-process cost; the steady-state memory savings compound across concurrent layers (03a + 03b + 03c) on the 24GB Mac mini M4 Pro budget. Apply this pattern to any future fallback/gated model integrations.

17. **Hardcoded Acoustic Heuristic Thresholds (Resolved - May 07)**:
    - **Problem**: Seven heuristic constants were baked into `pipeline.py` source (volume cutoffs, volume bonuses, pitch-variance weights, dominant-tone minimum, SenseVoice confidence gate, pitch-variance normalization divisor). Empirical tuning required source edits, which complicated A/B comparisons and made the heuristics opaque to non-Python collaborators.
    - **Solution**: Created `src/layer_03c_acoustic_prosody/config.py` housing a frozen `Layer03cConfig` dataclass with all seven constants as fields (defaults preserve previous behavior). `AcousticProsodyPipeline.__init__` now accepts an optional `config: Layer03cConfig = Layer03cConfig()` parameter. `_classify_acoustic_tone`, `_process_task`, and `_extract_librosa_features` all read from `self.config.*` instead of magic numbers. Added `test_config_thresholds_override_classification` and `test_config_volume_cutoff_override_changes_alarming_bonus` to verify override propagation.
    - **Model Instructions/Context**: Future layer pipelines should follow this pattern — house tunable thresholds in a frozen dataclass at `<layer_dir>/config.py`, accept it as an optional constructor parameter, and reference fields via `self.config.<field_name>`. Defaults must preserve current behavior so downstream orchestration scripts continue to work without changes.

18. **SenseVoice Lazy-Load Gate Cost-Neutral on 64 GB Mac Studio (Resolved - May 13)**:
    - **Problem**: Resolved Issue #16 replaced the eager `AutoModel(model="iic/SenseVoiceSmall", ...)` construction with a lazy load, driven by the 24 GB Mac mini M4 Pro memory budget where keeping ~150-200 MB of SenseVoice resident cost more than the rarely-paid first-call latency. On the **Mac Studio (M4 Max, 64 GB unified memory)** target host, 200 MB is ~0.3% of available memory, so the steady-state cost is negligible — but every clip that triggers the low-confidence gate (the most acoustically ambiguous clips, exactly where SenseVoice is most useful) still paid the one-time ~2-3 s SenseVoice load latency mid-run at the worst possible moment.
    - **Solution**: Added a `HIGH_MEMORY_HOST_BYTES = 48 * 2**30` class constant and a `host_can_eager_load_sensevoice()` staticmethod to `AcousticProsodyPipeline`, mirroring the `HIGH_MEMORY_HOST_BYTES` / `host_can_retain_resident()` pattern already established in `src/layer_03a_attention/pipeline.py`. `_init_models` now branches after the emotion2vec+ load: on hosts at or above the 48 GB gate it eager-constructs `self.sensevoice_model` (restoring pre-Resolved-Issue-#16 behavior), and on smaller hosts it leaves `self.sensevoice_model = None` so the lazy path in `_run_sensevoice_model` still applies. The host check imports `psutil` and falls back to `False` (lazy) when `psutil` is absent. Eager construction is wrapped in `try/except`: because SenseVoice is a supplementary model, a load failure logs and falls back to the lazy path rather than hard-failing the pipeline. Added `test_host_can_eager_load_sensevoice_gate`, `test_sensevoice_eager_loaded_on_high_memory_host`, `test_sensevoice_stays_lazy_on_standard_memory_host`, and `test_eager_load_failure_falls_back_to_lazy` to lock in both host-class branches and the graceful-degradation contract.
    - **Model Instructions/Context**: The 48 GB unified-memory host gate is the project-wide boundary between the 24 GB Mac mini M4 Pro and the 64 GB Mac Studio M4 Max. Reuse the `HIGH_MEMORY_HOST_BYTES = 48 * 2**30` constant plus a `host_can_*()` staticmethod (with the `psutil`-absent → conservative-default fallback) for any future host-class branch. Eager-loading is only safe to add for *supplementary* models whose load failure can degrade gracefully; primary inference models must still fail fast per Resolved Issue #13.

19. **emotion2vec+ `seed` Variant Documented as Candidate; `large` Retained (Resolved - May 14)**:
    - **Problem**: The 03c SER model is pinned to `iic/emotion2vec_plus_large` (~600 MB, 9-class). FunASR also ships `iic/emotion2vec_plus_base` (~300 MB, lower quality) and `iic/emotion2vec_plus_seed` (~2 GB, transformer-XL backbone), the latter producing stronger embeddings on long utterances with cross-speaker variation — relevant to the cross-bystander aggregate scoring path that 03b consumes. The `seed` variant's ~2 GB footprint was disqualifying on the legacy 24 GB Mac mini M4 Pro once L2CS, Py-Feat, and Depth Anything V2 + SAM were all simultaneously resident during the E2E run, but on the **Mac Studio (M4 Max, 64 GB unified memory)** target host it is loadable with no displacement of any other 03 layer. The open question was whether the new memory headroom alone justified migrating off — or A/B-validating against — the `large` pin.
    - **Solution**: Decision recorded to **stay on the `large` variant**; no production code change was made. A `> [!NOTE]` callout was added to `docs/ml_dependencies.md` (alongside the existing Gemma 4 variant note) documenting `base` and `seed` as available FunASR variants, recording that `seed` is now memory-affordable on the M4 Max host but remains **untested** against the 03c acoustic validation subset. Promoting `seed` would require A/B'ing 9-class dominant-emotion accuracy against human labels and re-checking that the SenseVoice low-confidence gate (`< 0.6`, Resolved Issue #11) does not need recalibration — work that carries migration risk (e.g. `seed`'s 768-dim embedding may have different distributional properties) with no measured benefit until a concrete downstream failure justifies it. This entry plus the `ml_dependencies.md` note close the "no documented reason beyond inertia" gap for the `large` pin.

## ⚠️ Unresolved Issues & Suggestions

_None at this time — all tracked issues have transitioned to the Resolved section above._