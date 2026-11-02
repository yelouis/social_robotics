# 05: Wave B, Multimodal Human Reactions to Robot Failures and Explanations (REFLEX): the contract

**What this answers:**
1. When a physical robot fails during a collaborative manipulation task, do the human partner's spontaneous **facial, gaze, head-pose and posture reactions** (`react-face`) distinguish normal execution from robot failure, and do they add information to the action-only model (`dataset = reflex`)?
2. After the robot fails and explains what happened, do the human partner's **facial/postural (`react-face`) and vocal prosody (`react-nonverbal`) reactions** predict whether the human-robot pair will successfully resolve the failure (`Resolved?`, a **hidden outcome** that has not occurred yet at the end of the action window; `dataset = reflex_resolve`)?

**Why REFLEX replaces Option E (`AM-FED+` / beverage tasting) and `BAD`:**
- On October 23, 2026, the maintainer selected Issue 1 Option E (`amfed@affectiva.com` and the beverage-tasting database author). Affectiva's address bounced immediately (`550 5.4.1 Recipient address rejected`) and its dataset page is gone; the beverage-tasting author (`M5`) did not reply. Meanwhile, `BAD` (Issue 4) requires an institutional IRB protocol that an unaffiliated researcher cannot obtain.
- On November 1, 2026, the maintainer approved pivoting Wave B to **REFLEX** (Naoum et al., KTH, HRI 2025; [Zenodo 14590226](https://zenodo.org/records/14590226), DOI `10.5281/zenodo.14590226`, **CC BY 4.0**, open access, 37.22 GB, 55 participants):
  1. It requires **no email, EULA signature, YouTube scraping or IRB**.
  2. It provides **facial Action Units, gaze, head rotation/velocity, facial valence/arousal, body posture, and speech prosody** extracted from the unmasked raw recordings before GDPR video anonymization, fulfilling **`DW2` (`react-face`)**.
  3. Its videos (`video_cam1.mp4`) have the **human participant segmented out with a black mask** while the **Baxter robot and tabletop objects remain visible** (`arXiv:2502.14185` §IV-1). Thus the action-only controls (`judge`, `judge-large`, `action-probe`) see the robot's physical action with the human's reactions masked by construction.
  4. It directly prepares the reaction reader for **`DW6` (Stage A live emotional feedback on the Pollen Robotics MicroDuck)** by training and validating `react-face` on people reacting to a physical robot's behavior in real time.

**Status:** pre-registered November 1, 2026 (`00_thesis.md` entry 28). Every rule below—items, windows, splits, feature columns, probe settings, and pass/kill verdicts—is fixed before `Dataset.zip` is downloaded and extracted. The build spec is [`agent_execution_guide.md`](agent_execution_guide.md) (items B1–B4).

---

## 1. Archive & Download Specification (`tools/download_reflex.py`)

- **Source:** Zenodo record `14590226` (`https://zenodo.org/records/14590226`, v1.1.0, March 2, 2025; CC BY 4.0).
- **Direct file URL:** `https://zenodo.org/api/records/14590226/files/Dataset.zip/content`.
- **Exact size:** `37,219,808,401` bytes (`37.22 GB`).
- **Exact MD5 checksum:** `334caac04f3e607ac91182dea7a03929`.
- **Rate-limit constraint (measured November 1, 2026):** Zenodo returns `HTTP 429: TOO MANY REQUESTS` on parallel multi-connection downloads. `tools/download_reflex.py` **must** use a single sequential resumable stream (`curl -L -C - --retry 5 --retry-delay 10` or single-stream HTTP `Range` resume) to `DATA_ROOT/raw/reflex/Dataset.zip`.
- **Verification & extraction:**
  1. Verify `os.path.getsize(zip_path) == 37219808401` and MD5 `== "334caac04f3e607ac91182dea7a03929"`.
  2. Extract `Dataset.zip` into `DATA_ROOT/raw/reflex/`, skipping any archive entry whose path starts with `__MACOSX/` or whose basename starts with `._` (AppleDouble metadata inside the zip).
  3. Verify all **5 strategy directories** and **55 participant directories** under `DATA_ROOT/raw/reflex/Dataset/`:
     - `C1-Fixed-Low` (`C1-1` … `C1-11`)
     - `C2-Fixed-Medium` (`C2-1` … `C2-11`)
     - `C3-Fixed-High` (`C3-1` … `C3-11`)
     - `D1-Decay-Smooth` (`D1-1` … `D1-11`) — *Note: `DatasetGuide.md` calls D1 `Decay-Slow`, but the actual directory name in `Dataset.zip` is `D1-Decay-Smooth`.*
     - `D2-Decay-Rapid` (`D2-1` … `D2-11`)
  4. Verify that every one of the 55 participant directories contains the **10 required files**:
     `analysis.csv`, `body.csv`, `facetorch.csv`, `gaze.csv`, `hume.csv`, `openface.csv`, `questionnaire.csv`, `speech.csv`, `time.csv`, and `video_cam1.mp4`.
     *(Note: `video_cam2.mp4` is missing for `C1-Fixed-Low/C1-10` and `D1-Decay-Smooth/D1-3` in the upstream archive and is not used by any step in Wave B.)*
  5. Delete `DATA_ROOT/raw/reflex/Dataset.zip` after verification to reclaim 37.22 GB on the SSD, and write `DATA_ROOT/raw/reflex/DOWNLOAD.json` recording `zenodo_record`, `doi`, `license`, `zip_bytes`, `zip_md5`, `n_participants` (`55`), `verified` (`true`), `archive_deleted` (`true`), and `ts`.

---

## 2. Time & Frame Alignment Contract (Preventing Unit-Boundary Errors)

Across each participant directory `<P>`, REFLEX stores data in three coordinate systems that must be mapped explicitly:

1. **1-indexed frames (`Frame` = `1 … N`):**
   - `time.csv` (`Frame` column, `1 … N`), `analysis.csv` (`Start Frame`, `End Frame`), and `openface.csv` (`frame` column, `1 … N`) use **1-indexed** frame numbers corresponding 1-to-1 with the frames of `video_cam1.mp4`.
2. **0-indexed frames (`0 … N − 1`):**
   - `facetorch.csv` (`Frame ID` column) and `body.csv` (`Frame` column) use **0-indexed** frame numbers (`frame_0idx = frame_1idx - 1`).
3. **Container seconds vs. Wall-clock seconds:**
   - `video_cam1.mp4` is encoded at a fixed container frame rate (`fps ≈ 4.4444` Hz, i.e. `40/9` fps via OpenCV `VideoWriter`, no audio stream). When `ffmpeg -ss <t> -i video_cam1.mp4` extracts frames for `FrameEncoder` (`siglip-b16-224-u9`) and `VLMJudge`, it seeks in **container seconds**:
     $$t_{\text{container}}(\text{frame}_{1\text{idx}}) = \frac{\text{frame}_{1\text{idx}} - 1}{\text{fps}_{\text{container}}}$$
     where $\text{fps}_{\text{container}}$ and $D_{\text{container}}$ are read from `video_cam1.mp4` via `ffprobe`.
   - `time.csv` (`Seconds` column) and `speech.csv` (`BeginTime`, `EndTime` columns) use **elapsed wall-clock seconds** (`0.0 … ~1560.0` s). Because the live camera capture rate fluctuated slightly around 4.44 Hz, wall-clock seconds drift by several seconds from container seconds over a 25-minute session.
   - **Rule:**
     - `Item.action_window_sec` and `Item.reaction_window_sec` store **container seconds** `[round((sf - 1) / fps, 4), round(min(D_container, (ef - 1) / fps), 4)]` so `FrameEncoder` and `VLMJudge` seek to the exact annotated frames `[sf, ef]` in `video_cam1.mp4`.
     - `Item.meta["action_frames"] = [sf, ef]` and `Item.meta["reaction_frames"] = [rsf, ref]` store the exact **1-indexed inclusive frame bounds** for `ReflexFaceEncoder`.
     - `Item.meta["wall_action_window_sec"]` and `Item.meta["wall_reaction_window_sec"]` store the **wall-clock seconds** `[time.csv.Seconds[sf], time.csv.Seconds[ef]]` for matching utterances in `speech.csv` (`ReflexSpeechEncoder`, `react-spoke`, `react-full`).

---

## 3. Datasets, Items and Splits (`src/sources/reflex.py`)

In each participant's `analysis.csv`, robot-failure episodes are annotated as contiguous 4-row blocks in exact state order `Pre` → `Failure` → `Explanation` → `Resolution` (604 episodes total across the 55 participants; 2,416 rows). Let `episode_idx` (`0 … K − 1`) be the 0-based index of the 4-row episode block within that participant's `analysis.csv`.

### 3.1 Group Partition (Shared Across Both REFLEX Datasets)

- **`group_id`:** the participant ID string (e.g. `"C1-1"`, `"D1-3"`), **55 unique groups**.
- **Split rule (`03_eval_harness.md` §5):** sort the 55 unique participant IDs lexicographically, shuffle once with `numpy.random.default_rng(0)`, assign the first `round(0.7 * 55) = 38` participants to `train` and the remaining `17` participants to `test`.
- Both `splits/reflex.json` and `splits/reflex_resolve.json` use this identical 38-train / 17-test participant partition so no participant ever straddles train and test in either dataset or between datasets.

### 3.2 Dataset 1: `reflex` (Immediate Reaction to Robot Failure: `Pre` vs. `Failure`)

Each valid 4-row episode `(Pre, Failure, Explanation, Resolution)` in `analysis.csv` yields **two items**:

1. **`pre` (`label = 1`, robot performing normally before failure):**
   - `item_id`: `"reflex:<participant>:<episode_idx>:pre"`
   - `action_frames`: `[Pre.Start Frame, Pre.End Frame]` (strictly `< Failure.Start Frame`).
   - `reaction_frames`: `[Pre.Start Frame, Pre.End Frame]`.
   - `label_source`: `"reflex.analysis_state_pre_vs_failure"`.
2. **`fail` (`label = 0`, robot failure):**
   - `item_id`: `"reflex:<participant>:<episode_idx>:fail"`
   - `action_frames`: `[Failure.Start Frame, Failure.End Frame]`.
   - `reaction_frames`: `[Failure.Start Frame, min(Explanation.End Frame, Failure.Start Frame >= 1 and Failure.End Frame + 22, max_frame)]`, i.e. `min(Explanation.End Frame, Failure.End Frame + 22, max_frame)`. At 4.44 Hz, 22 frames is $\approx 5.0\text{ s}$ after failure end, matching HoloAssist's `[start, end + 5.0]` s observer reaction window (Lesson L4) while staying within the explanation window before resolution begins.
   - `label_source`: `"reflex.analysis_state_pre_vs_failure"`.

- **`context_text` (identical for `pre` and `fail` of an episode):**
  `"Task: human-robot collaborative object sorting. Step: <action> object."`
  where `<action>` is `Action.strip().lower()` (`"pick"`, `"carry"`, or `"place"`).
  - **Leakage rule (L12):** `analysis.csv`'s `Object` column contains annotator failure notes such as `"White Cube - Drop"`, `"Red Toy (Nothing inside)"`, and `"Cup (Wrong Cup)"`. `Object`, `State`, `Explanation Level`, and `Resolved?` are **never** included in `context_text` or `meta.noun` (`meta.verb = <action>`, `meta.noun = "object"`). Because `pre` and `fail` share the exact same `(verb, noun)` in every episode, `step-prior` on `reflex` is 0.50 by construction.
- **Judge question (`DATASET_QUESTIONS["reflex"]`), verbatim:**
  `Is the robot performing the step normally at this point, with no failure yet?`
- **Expected yield:** 604 episodes $\rightarrow$ **1,208 candidate items** across 55 participants (minus any episode skipped for `invalid_frames` or `invalid_window`, recorded in `DATA_ROOT/items/reflex/stats.json`).

### 3.3 Dataset 2: `reflex_resolve` (Hidden-Outcome Resolution: `Failure + Explanation` $\rightarrow$ `Resolved?`)

Each valid 4-row episode `(Pre, Failure, Explanation, Resolution)` where `Failure.Resolved?` (trimmed uppercase) is `"TRUE"` or `"FALSE"` yields **one item**:

- **`item_id`:** `"reflex_resolve:<participant>:<episode_idx>"`
- **`label`:** `1` if `Failure.Resolved?.strip().upper() == "TRUE"` (477 episodes), `0` if `"FALSE"` (116 episodes). The 11 episodes with any other value in `Resolved?` are skipped with reason `missing_resolved_label` in `DATA_ROOT/items/reflex_resolve/stats.json`.
- **`label_source`:** `"reflex.analysis_resolved"`.
- **`action_frames` (what `judge`, `judge-large`, and `action-probe` see):**
  `[Failure.Start Frame, Explanation.End Frame]`.
  - **Why this is a strict hidden-outcome window:** it ends before `Resolution` begins, and the human participant is black-masked in `video_cam1.mp4`. Neither the human's reaction nor the resolution outcome is visible in the action window.
- **`reaction_frames` (what `react-face`, `react-nonverbal`, `react-spoke`, and `react-full` see):**
  `[Failure.Start Frame, Resolution.End Frame]`.
- **`context_text`:**
  `"Task: human-robot collaborative object sorting. Step: <action> object. Explanation level: <explanation_level>."`
  where `<action>` is `Action.strip().lower()` and `<explanation_level>` is `Explanation Level.strip().lower()`.
  - **`meta.verb`** = `<action>`, **`meta.noun`** = `<explanation_level>` (`Object` is never used in `context_text` or `meta.noun`).
  - **`step-prior` (Lesson L12):** `step_prior_scores()` fits the hierarchical Bayesian prior ($m = 5$: base rate $p_0 \to \text{verb} \to (\text{verb}, \text{noun})$, LOO on train) on `(action, explanation_level)` and enters `action_best`, `action-fusion`, and `fusion`.
- **Judge question (`DATASET_QUESTIONS["reflex_resolve"]`), verbatim:**
  `Will the human and the robot successfully resolve this failure and complete the step?`
- **Expected yield:** **593 candidate items** (477 `label = 1`, 116 `label = 0`; 11 skipped as `missing_resolved_label`) across 55 participants.

### 3.4 Speech Metadata on Both Datasets

For every item in `reflex` and `reflex_resolve`, `src/sources/reflex.py` reads the participant's `speech.csv` and finds all rows where:
- `Id.strip().lower() in {"participant", "robot/participant"}`, and
- `float(BeginTime) <= wall_reaction_window_sec[1]` and `float(EndTime) >= wall_reaction_window_sec[0]`.

It sets:
- `meta["spoke"] = 1` if at least one matching row exists, else `0`;
- `meta["transcript"]` = the space-joined `Transcript.strip()` strings of matching rows (excluding `"NA"`), or `""`.

---

## 4. Encoders (`src/features/face.py`, `src/features/extract.py`)

Wave B uses no new pip packages and downloads no new models:

| Encoder id | Modality | Dim | Spec |
|---|---|---|---|
| `siglip-b16-224-u9` | Action window visual (`video_cam1.mp4`, human black-masked) | `768` (`float32`) | Existing `FrameEncoder(encoder_id="siglip-b16-224-u9")`: 9 uniformly spaced frames over `item.action_window_sec`, L2-normalized per frame and mean-pooled |
| `reflex-face-v1` | Reaction window facial AUs, gaze, head pose/velocity, valence/arousal, arm posture (`openface.csv`, `facetorch.csv`, `body.csv`) | `153` (`float32`) | `ReflexFaceEncoder` (`src/features/face.py`), defined in §4.1 below |
| `reflex-speech-v1` | Reaction window participant vocal prosody (`speech.csv`) | `96` (`float32`) | `ReflexSpeechEncoder` (`src/features/face.py`), defined in §4.2 below |

### 4.1 `ReflexFaceEncoder` (`encoder_id = "reflex-face-v1"`, 153-d `float32`)

Given an item with `participant_dir = Path(item.video_path).parent` and 1-indexed inclusive `rsf, ref = item.meta["reaction_frames"]` (with column names stripped of leading/trailing whitespace on CSV load):

1. **OpenFace 2.0 (`openface.csv`, 141 dims):**
   - Filter rows where `rsf <= int(float(row["frame"])) <= ref`, `int(float(row["success"])) == 1`, and `float(row["confidence"]) >= 0.7`.
   - **Honest null (L3):** if `0` rows pass this filter, raise `RuntimeError(f"no valid OpenFace frames in [{rsf}, {ref}]")`.
   - Extract the **46 columns** in exact order:
     - Gaze (8): `gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y`
     - Head rotation (3): `pose_Rx, pose_Ry, pose_Rz`
     - Continuous AU intensity (17): `AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r`
     - Binary AU presence (18): `AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c, AU09_c, AU10_c, AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, AU23_c, AU25_c, AU26_c, AU28_c, AU45_c`
   - Across the $M \ge 1$ valid frames, compute `mean` (46-d), `std` (`ddof=0`, 46-d), and `max` (46-d) $\rightarrow 138$ dims.
   - Head angular velocity variability (3 dims): on the 3 head rotation columns `pose_Rx, pose_Ry, pose_Rz`, if $M \ge 2$, compute `np.std(np.diff(pose_mat, axis=0), axis=0, ddof=0)`; if $M = 1$, `[0.0, 0.0, 0.0]`.
2. **Facetorch (`facetorch.csv`, 6 dims):**
   - Filter rows where `rsf - 1 <= int(float(row["Frame ID"])) <= ref - 1` and both `Valence` and `Arousal` are finite numbers (not `""` or `"NA"`).
   - If `0` rows pass, raise `RuntimeError(f"no valid Facetorch frames in [{rsf - 1}, {ref - 1}]")`.
   - Compute `mean` (2-d), `std` (`ddof=0`, 2-d), and `max` (2-d) of `[Valence, Arousal]` $\rightarrow 6$ dims.
3. **Body posture (`body.csv`, 6 dims):**
   - Filter rows where `rsf - 1 <= int(float(row["Frame"])) <= ref - 1` and `Crossed Arms`, `Arms behind back` are in `{"TRUE", "FALSE"}` (mapped to `1.0` / `0.0`).
   - If `0` rows pass, raise `RuntimeError(f"no valid Body frames in [{rsf - 1}, {ref - 1}]")`.
   - Compute `mean` (2-d), `std` (`ddof=0`, 2-d), and `max` (2-d) of `[Crossed Arms, Arms behind back]` $\rightarrow 6$ dims.
4. Concatenate `[openface_141d, facetorch_6d, body_6d]` as `np.float32` of shape `(153,)`. Raise `RuntimeError` if any entry is non-finite.

### 4.2 `ReflexSpeechEncoder` (`encoder_id = "reflex-speech-v1"`, 96-d `float32`)

Given an item with `participant_dir = Path(item.video_path).parent` and `wr_s, wr_e = item.meta["wall_reaction_window_sec"]`:

1. Read `speech.csv` and select rows where `Id.strip().lower() in {"participant", "robot/participant"}`, `float(BeginTime) <= wr_e`, `float(EndTime) >= wr_s`, and all 48 Hume prosody columns (`Admiration` … `Triumph`, columns 6 through 53 inclusive) are finite numbers (not `"NA"`).
2. **Honest null (L3):** if `0` rows match, raise `RuntimeError("no participant speech in reaction window")`.
3. Compute `mean` (48-d) and `max` (48-d) across matching utterances and concatenate $\rightarrow$ `np.float32` of shape `(96,)`.

---

## 5. Probe Conditions & Fusion on REFLEX (`src/harness/probes.py`)

For `dataset in ("reflex", "reflex_resolve")`:

- **Action-only controls (`action_best` candidates):**
  - `judge`: `qwen2.5vl:7b` on `action_window_sec` (4 frames, no audio).
  - `judge-large`: `gemma4:26b` on `action_window_sec` (test split only; enters `action_best` at $\ge 90\%$ coverage).
  - `judge-frontier`: `gemini-3.6-flash` (if cached at $\ge 90\%$ coverage; not required to run).
  - `action-probe`: `LogisticRegression` on `siglip-b16-224-u9` features.
  - `step-prior`: `step_prior_scores()` on `(meta.verb, meta.noun)`.
  - `action-fusion`: `LogisticRegression` on `[logit(judge), logit(step-prior), X_action]`, fit on `fusion_tr_mask` and scored on `fusion_te_mask`.
- **Reaction-only conditions:**
  - `react-face`: `LogisticRegression` on `reflex-face-v1` (153-d). **Headline reaction condition on both `reflex` and `reflex_resolve`.**
  - `react-nonverbal`: `LogisticRegression` on `reflex-speech-v1` (96-d) over items with participant speech (plus diagnostic `react-nonverbal|spoke=1` on test items with `meta.spoke == 1`). If fewer than 2 classes have speech on train or test, reported honestly as `partial` or `not run` with derived `notes` (`03_eval_harness.md` §6).
  - `react-spoke`: `1.0 - float(meta.spoke)` (context row).
  - `react-full`: TF-IDF + `LogisticRegression` on `meta.transcript` (leakage context row; `not run` with derived notes if train transcripts have $< 2$ classes or empty vocabulary).
- **Fusion (`fusion`):**
  - Masks: `fusion_tr_mask = valid_j_tr & valid_act_tr & valid_face_tr`, `fusion_te_mask = valid_j_te & valid_act_te & valid_face_te`.
  - On `reflex`: `[logit(judge), logit(step-prior), X_action, X_face]`.
  - On `reflex_resolve`: `[logit(judge), logit(step-prior), X_action, X_face, spoke_1d, X_speech_zf]`, where `spoke_1d = float(meta.spoke)` and `X_speech_zf` is `reflex-speech-v1` standardized on `valid_speech_tr` items and zero-filled (`0.0`) on items without participant speech (if `valid_speech_tr` has $< 2$ classes, `spoke_1d` and `X_speech_zf` are omitted).
- **Ablation $\Delta$ (`fusion_minus_action_fusion`) and `fusion_minus_action_best`:**
  - Paired group-bootstrap $\Delta$ AUROC (`1,000` resamples, `default_rng(0)`).
- **Shuffled-label controls (one-sided leak check, `ci_low > 0.5` fails per `03_eval_harness.md` §11.9):**
  - `action-probe:shuffled`, `react-face:shuffled`, `react-nonverbal:shuffled`, `step-prior:shuffled`, `action-fusion:shuffled`, `fusion:shuffled`.
- **Exported Stage A MicroDuck artifact (`DW6` preparation):**
  - When `run_probes` fits `react-face` on `reflex` and `reflex_resolve`, it saves the fitted `StandardScaler` (`mean`, `scale`) and `LogisticRegression` (`coef`, `intercept`) to `DATA_ROOT/transfer/{dataset}_face_probe.npz`.

---

## 6. Pre-Registered Predictions, Pass Bar and Kill Rule

Per `00_thesis.md` (H1, Decision 20, and Decision 28) and `03_eval_harness.md` §11.8:

1. **Clause 1 ("reaction-only beats `action_best`"):**
   - Tested on `react-face`: **held** iff `react-face`'s `ci_low > action_best` AUROC.
2. **Clause 2 ("reactions add to the action-only model"):**
   - Tested on `fusion_minus_action_fusion` (the ablation $\Delta$): **held** iff `ci_low > 0`.
3. **H1 Pass Bar on REFLEX (`reflex` and `reflex_resolve`):**
   - **Met** on a dataset iff `react-face` AUROC $\ge 0.65$ **and** `fusion_minus_action_fusion` $\ge +0.03$ with `ci_low > 0`.
4. **Restated Kill Rule (Decision 20 & Decision 28):**
   - If **neither** `reflex` **nor** `reflex_resolve` achieves a positive ablation gain (`fusion_minus_action_fusion` `ci_low > 0`) **and** `react-face` fails to beat chance (`ci_low <= 0.5`) on both datasets, then human facial/postural reactions do not carry usable reward signal on REFLEX; stop and write up the negative result.
   - Conversely, if `react-face` has `ci_low > 0.5` or `fusion_minus_action_fusion` has `ci_low > 0` on at least one REFLEX dataset, the exported `reflex_face_probe.npz` / `reflex_resolve_face_probe.npz` proceeds to `DW3` (H2 transfer) and `DW6` (Stage A live MicroDuck feedback).
