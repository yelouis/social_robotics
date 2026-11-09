# 05: Wave B, Multimodal Human Reactions to Robot Failures and Explanations (REFLEX): the contract

**What this answers:**
1. When a physical robot fails during a collaborative manipulation task, do the human partner's spontaneous **facial, gaze, head-pose and posture reactions** (`react-face`) distinguish normal execution from robot failure, and do they add information to the action-only model (`dataset = reflex`)?
2. After the robot fails and explains what happened, do the human partner's **facial/postural (`react-face`) and vocal prosody (`react-nonverbal`) reactions** predict whether the human-robot pair will successfully resolve the failure (`Resolved?`, a **hidden outcome** that has not occurred yet at the end of the action window; `dataset = reflex_resolve`)?

**Why REFLEX replaces Option E (`AM-FED+` / beverage tasting) and `BAD`:**
- On October 23, 2026, the maintainer selected Issue 1 Option E (`amfed@affectiva.com` and the beverage-tasting database author). Affectiva's address bounced immediately (`550 5.4.1 Recipient address rejected`) and its dataset page is gone; the beverage-tasting author (`M5`) did not reply. Meanwhile, `BAD` (Issue 4) requires an institutional IRB protocol that an unaffiliated researcher cannot obtain.
- On November 1, 2026, the maintainer approved pivoting Wave B to **REFLEX** (paper: Khanna, Naoum et al., KTH, HRI 2025, arXiv:2502.14185; dataset: Naoum and Khanna; author order corrected November 5, 2026, see `01_prior_work.md`; [Zenodo 14590226](https://zenodo.org/records/14590226), DOI `10.5281/zenodo.14590226`, **CC BY 4.0**, open access, 37.22 GB, 55 participants):
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

---

## 7. Result of v1 (November 4, 2026), and the designer's post-hoc notes

**The pre-registered verdict is PROCEED** (`docs/evals/2026-11-04_reflex_h1.md`; rows `2026-11-05T05:49:35Z`–`05:49:51Z`). It rests on one clause: `reflex` `react-face` 0.609 [0.569, 0.644], `ci_low > 0.5`.
- **Every other pre-registered check failed.** The H1 pass bar is not met on either dataset, and clauses 1 and 2 did not hold on either. The ablation Δ is +0.023 [−0.004, 0.045] on `reflex` and +0.036 [−0.002, 0.078] on `reflex_resolve`.
- **The implementation followed this spec faithfully.** The designer recomputed:
  - all 1,812 items' frames and container and wall-clock windows, with 0 mismatches;
  - the split (38 / 17 participants, `default_rng(0)`);
  - the encoders against §4;
  - `react-face` (0.608 [0.569, 0.644]), the exported probe (0.6085) and the 50-seed permutation null.
- **The verdict stands. These notes do not change it.** They record what the designer found, so that no one builds on more than v1 supports. What to do next is **Issue 12** (`ongoing_general_errors.md`).

**Flaws in this spec:**
1. **§3.2's windows differ in length by class, and §4.1's summaries grow with length.**
   - A `fail` item's reaction window runs to `Failure.End + 22` frames (mean 45.7 frames), but a `pre` item's is just the `Pre` state (mean 35.5).
   - `reflex-face-v1` takes each column's `max` and `std` over the window. Both grow with the number of frames.
   - **This window-length effect is the whole of the PROCEED clause** (diagnostics below).
2. **The masked video still shows the participant's body.** `video_cam1.mp4` blacks the participant out but keeps their silhouette, so leaning, turning, and a hand raised to the head are visible. Verified on the frames of `reflex:D2-1:10:pre` and `reflex:D1-1:4:fail`. The "action-only" controls (`judge`, `judge-large`, `action-probe`, `action-fusion`) therefore see body reactions, and the opening section's "masked by construction" (point 3) is wrong.
3. **§3.3's reaction window includes the outcome.** It runs through `Resolution`, while the action window stops before it, so the reaction channels watch the resolution happen. The action-only channels do not.
4. **§3.3's outcome is set mostly by the step, not the explanation.** Resolution by step: `pick` 219/219, `carry` 198/220, `place` 71/165. By explanation level it is 74–88% at every level.
5. **Facts in §2–§4 that the archive contradicts:**
   - the container rate is 4.4 fps for all 55 videos, not 40/9;
   - the 11 "other" `Resolved?` values are participant `C3-2`'s 11 episodes. Their header is misspelled `Resloved?`, and all 11 are `TRUE`. B1 normalized the header, so `reflex_resolve` has 604 items (488 / 116), not 593;
   - `speech.csv` has no `Transcript` column. The text is in `Text` (1,105 participant utterances, median 7 words), so `meta.transcript` is empty and `react-full` did not run;
   - 12 utterances are labeled `robot/ participant` (with a space), which §3.4's rule does not match.
6. **§3.4's speech rule reads the robot's voice as the participant's** (found November 8, 2026).
   - Across all 55 participants, of the 1,093 `speech.csv` rows that §3.4's rule (`Id.strip().lower() in {"participant", "robot/participant"}`) selects as participant speech, 622 are `robot/participant` rows that include the robot's own voice, and only 471 are `participant` rows (4 of which contain robot-like phrasing; `docs/evals/2026-11-08_reflex_v2_posthoc.json`).
   - On `reflex_resolve`, 367 of the 466 items with any §3.4 speech row (0.788, or 79%) contain at least one mixed `robot/participant` segment.
   - Restricted to `participant` rows alone, `react-nonverbal` on `reflex_resolve` falls from 0.660 [0.545, 0.766] (140 test items) to 0.501 [0.309, 0.734] (66 test items; exploratory), and before resolution (`[Failure.Start, Explanation.End]`) it is not estimable because only two unresolved test items remain (`docs/evals/2026-11-08_reflex_v2_posthoc.json`).

**Exploratory diagnostics** (measured after the result was seen, so they are not evidence for any verdict):
- **Sources:** cached features, or the B2 encoders on new windows. Probes are the harness's (`StandardScaler` + `LogisticRegression(C=1.0, class_weight="balanced")`); CIs are grouped, 1,000 resamples. Item **B4b** reproduces every number in code.
- **`reflex`:**
  - **window length alone** (shorter → `pre`, sign set on train): **0.735 [0.675, 0.796]**, better than `react-face`;
  - **`react-face` on its 50 `_mean` dims only:** 0.515 [0.471, 0.564];
  - **`react-face` with equal-length windows** (per episode, both items cut to the shorter length, median 36 frames): 0.525 [0.473, 0.572] start-anchored, 0.511 [0.457, 0.561] end-anchored;
  - **[window length + face] − [window length]:** −0.069 [−0.103, −0.030].
- **`reflex_resolve`:**
  - **reactions cut at the end of `Explanation`** (before the outcome): `react-face` 0.545 [0.461, 0.634]; `react-nonverbal` 0.643 [0.480, 0.843] (104 items, 16 negatives);
  - **`step-prior` alone:** 0.941 [0.886, 0.980] on the face-scored items. **[step-prior + face] − step-prior:** −0.121 [−0.193, −0.059];
  - **`react-nonverbal:shuffled`'s leak-check FAIL** (0.603 [0.507, 0.691]) is a single-seed false alarm. Over 50 seeds the null averages 0.5100 (SD 0.0626), seed 0 is 0.6025, and the true-label 0.660 beats all 50 (L14).
- **Reading:** once window length is equalized, faces carry no detectable signal on REFLEX in either dataset. The exported `reflex_face_probe.npz` mostly encodes window length, so it cannot serve as a live reaction reader (DW6).

**What a v2 would have to fix** (only if Issue 12 selects it):
- equal-length reaction windows for both classes, with a window-length-only control;
- `reflex_resolve` reaction windows that end where the action window ends;
- transcripts from `Text`, and `Id` matched after removing all whitespace;
- a fusion that cannot lose to its own `step-prior` input.
- The silhouette leak cannot be removed with the released video.

**Decision (November 5, 2026; Issue 12 = A, the maintainer):**
- **v1 is Wave B's result.** PROCEED stays as the recorded verdict, with these notes beside it, and the write-up reports both together (`06_writeup.md` §2).
- **No v2 is run.** Wave C (H2 transfer) and DW7 are not selected, and Stage A (DW6) is paused until a validated reaction reader exists.

**Exploratory v2 (November 7–8, 2026; run at the maintainer's request in chat, outside the guide's queue; not pre-registered)**
- **What v2 is:** A 47-feature reaction encoder (`ReflexFaceV2Encoder` in `tools/diag_reflex_v2.py`; `docs/evals/2026-11-08_reflex_v2_posthoc.json`) combining 9 gaze-label proportions, late-minus-early deltas, and transition rates from `gaze.csv`, 34 mean and late-minus-early delta features across 17 facial expression channels from `hume.csv`, and 4 Valence/Arousal mean and delta features from `facetorch.csv`, evaluated across four episode-matched or original window schemes (`equal_start` 0.759 [0.724, 0.795], `action_equal_end` 0.736 [0.691, 0.787], `equal_end` 0.695 [0.642, 0.753], `orig` 0.722 [0.680, 0.763] on `reflex`, with paired `fusion_minus_action_fusion` gains of +0.056 [+0.029, +0.083], +0.068 [+0.040, +0.099], +0.029 [+0.005, +0.053], and +0.044 [+0.017, +0.068]).
- **Chosen on the test participants:** The 47 features, 17 Hume channels, and four window schemes were selected after about 80 test-set AUROC evaluations on the same 17 REFLEX test participants on November 7–8, 2026 (over 40 random 38/17 participant re-splits on `action_equal_end`, the already-selected specification averages 0.763 with 5th–95th percentiles [0.723, 0.787], which tests split sensitivity but not feature-selection overfitting).
- **Gaze-vs-face split:** The gain over `action-fusion` comes from the dataset authors' semi-manually validated categorical gaze labels (`gaze.csv`, annotated during failure episodes), not from facial expressions: on `action_equal_end`, the 9 gaze features alone score 0.720 [0.689, 0.752] with a fusion gain of +0.053 [+0.025, +0.084], whereas the 38 Hume + Facetorch face features score 0.618 [0.575, 0.664] with a fusion gain of +0.004 [−0.007, +0.017] (on `equal_start`, gaze-only scores 0.718 [0.673, 0.758] with gain +0.038 [+0.014, +0.063], and face-only scores 0.643 [0.595, 0.697] with gain +0.012 [−0.001, +0.029]; robot-gaze share in `action_equal_end` test windows is 0.061 in `Pre` vs. 0.128 in `Failure`).
- **Phase placebo and onset straddle:** A no-failure placebo comparing the early half vs. the late half of `Pre` within each episode (where no failure has occurred anywhere) scores 0.679 [0.631, 0.732] with the same 47 features (0.638 [0.601, 0.677] gaze-only, 0.604 [0.558, 0.653] face-only; applying the `equal_start` probe directly to the placebo windows scores 0.390 [0.336, 0.450]), and straddling the exact failure onset (last frames of `Pre` vs. first frames of `Failure`) drops to 0.598 [0.542, 0.650] for all 47 features, 0.600 [0.552, 0.651] for gaze-only, and 0.495 [0.431, 0.569] for face-only.
- **`reflex_resolve` step-proxy checks:** On `reflex_resolve` (`[Failure.Start, Explanation.End]`, `n_test` = 187), v2 scores 0.696 [0.601, 0.790] (window length alone 0.652 [0.581, 0.723]; `[length + v2] − length` +0.104 [+0.020, +0.189]; last 10 frames 0.658 [0.560, 0.754]; last 20 frames 0.661 [0.575, 0.749]), but `step-prior` alone scores 0.937 [0.883, 0.980], adding v2 to `step-prior` reduces AUROC by −0.062 [−0.112, −0.005], `fusion` (0.775 [0.668, 0.889] vs. `action-fusion` 0.743 [0.632, 0.874], gain +0.031 [+0.002, +0.064]) trails `action_best` (`step-prior`) by −0.163 [−0.287, −0.024], v2 predicts the physical step itself on test (`pick` 0.852, `place` 0.838, `carry` 0.709), and within `place` items (`n_test` = 51, the only step with balanced outcomes) v2 is at chance at 0.505 [0.309, 0.705].

It does not change the recorded verdict.

---

## 8. Pre-registered confirmatory test: automatic OpenFace gaze on phase-matched object handlings (November 9, 2026; Issue 13 = C)

**Why this test exists (Issue 13 = C, selected by the maintainer November 9, 2026):**
- The exploratory v2 evaluation (§7) showed that the within-episode `reflex` gain (+0.053 [+0.025, +0.084] on `action_equal_end`) came from `gaze.csv` rather than facial expressions (+0.004 [−0.007, +0.017]), and left two questions unresolved:
  1. **Automatic camera gaze vs. dataset gaze labels:** `gaze.csv` was generated and manually validated by the dataset authors only during failure episodes (`02_data_sources.md`). Can an automatic camera gaze tracker (`openface.csv`) recover that signal without human validation?
  2. **Same-phase object handling vs. within-episode phase shift:** Within each failure episode, `Pre` is the sub-step preceding the failure (`Approach` before `Pick`, `Approach + Pick` before `Carry`, or `Approach + Pick + Carry` before `Place`), so comparing `Pre` against `Failure` within an episode mixes failure with the robot's normal progression from table to shelf (`placebo_early_vs_late_pre` = 0.679 [0.631, 0.732]). Does automatic gaze distinguish **successful vs. failed object handlings at the exact same physical phase and equal duration**?
- **Pre-registration disclosure (L17):** This specification was written on November 9, 2026 after a label-blind inspection of `analysis.csv` episode schedules, `video_cam1.mp4` contact sheets for one training participant (`C1-1`), and valid-frame coverage counts (`success == 1` and `confidence >= 0.7`) in `openface.csv`, **before any OpenFace gaze angle or vector value was read or any feature or probe was computed**.
- **Scope:** It is evaluated **once** by `tools/eval_reflex_openface_gaze.py`, writing `docs/evals/2026-11-09_reflex_openface_gaze.json`. `DATA_ROOT` and `results/scorecard.jsonl` remain untouched; Wave B's recorded v1 verdict (`PROCEED`, §7) is never changed.

### 8.1 Phase-matched contrast (`reflex_same_phase`: 275 pairs = 550 items across 55 participants)

Across all 55 participants (`38` train, `17` test in the locked `splits/reflex.json`), `analysis.csv` contains the same five pairs of segments where the positive segment (`label = 1`) is the `Pre` interval of a downstream-failure episode in which the robot **completes** a physical step (`Pick` at the table or `Carry` to the shelf) and the negative segment (`label = 0`) is the `Failure` interval of another episode of the **same participant** in which the **same physical step** fails:

| Pair ID | Physical phase (`verb`) | Positive (`label = 1`, successful handling) | Negative (`label = 0`, failed handling at same phase) |
|---|---|---|---|
| `pick_1` | `pick` | Round `2`, Object `A`, `Carry`, `State = Pre` (approaches & successfully grasps/lifts Object A at table) | Round `1`, Object `B`, `Pick`, `State = Failure` (fails to grasp/lift Object B at table) |
| `pick_2` | `pick` | Round `3`, Object `B`, `Carry`, `State = Pre` (approaches & successfully grasps/lifts Object B at table) | Round `4`, Object `A`, `Pick`, `State = Failure` (fails to grasp/lift Object A at table) |
| `carry_1` | `carry` | Round `1`, Object `D`, `Place`, `State = Pre` (successfully carries Object D from table to shelf) | Round `1`, Object `B`, `Carry`, `State = Failure` (drops/fails while carrying Object B to shelf) |
| `carry_2` | `carry` | Round `3`, Object `A`, `Place`, `State = Pre` (successfully carries Object A from table to shelf) | Round `2`, Object `A`, `Carry`, `State = Failure` (drops/fails while carrying Object A to shelf) |
| `carry_3` | `carry` | Round `4`, Object `C`, `Place`, `State = Pre` (successfully carries Object C from table to shelf) | Round `3`, Object `B`, `Carry`, `State = Failure` (drops/fails while carrying Object B to shelf) |

- **1-to-1 mapping to `items/reflex/items.jsonl`:** Every positive segment maps to the existing `:pre` item (`label = 1`) and every negative segment maps to the existing `:fail` item (`label = 0`) of that `(Round No., Object, Action)` episode in `items/reflex/items.jsonl` (`190` train pairs = `380` train items across 38 train participants; `85` test pairs = `170` test items across 17 test participants).
- **Equal-length end-anchored windows (`K` frames per pair):**
  - For each pair `(pos_it, neg_it)` of participant `p`, let `[psf, pef]` and `[nsf, nef]` be `pos_it.meta["action_frames"]` and `neg_it.meta["action_frames"]`, and let `K = min(pef - psf + 1, nef - nsf + 1)`.
  - Both items are evaluated on their terminal `K`-frame window ending at the completion or failure of the physical step:
    - `pos_window = (pef - K + 1, pef)` (the final `K` frames of `Carry.Pre` = the completed grasp/lift at the table for `pick`, or the final `K` frames of `Place.Pre` = the completed carry to the shelf for `carry`);
    - `neg_window = (nef - K + 1, nef)` (the final `K` frames of `Pick.Failure` or `Carry.Failure`).
  - By construction, `len(pos_window) == len(neg_window) == K` for every pair, so window length is identical for positive and negative items (`AUROC = 0.500`).
- **Physical-phase assignment for `step-prior`:**
  - In `items/reflex/items.jsonl`, `meta["verb"]` stores the episode's failure verb (`"carry"` on `Carry.Pre`, `"place"` on `Place.Pre`), which would trivially separate cross-episode pairs if left unchanged.
  - On `reflex_same_phase`, both items of `pick_1` and `pick_2` are assigned `meta["verb"] = "pick"`, `meta["noun"] = "object"`, and both items of `carry_1`, `carry_2`, and `carry_3` are assigned `meta["verb"] = "carry"`, `meta["noun"] = "object"`. Because each pair has one positive and one negative item of the same physical phase, `step-prior` has a 50% base rate in both phases and zero label leakage.

### 8.2 Automatic OpenFace gaze encoder (`ReflexOpenFaceGazeEncoder`, `reflex-openface-gaze-v1`, 26-d)

- **Input:** `openface.csv` in the participant's directory (`1`-indexed `frame` column), stripped of whitespace around column names.
- **Quality filter:** Keep rows with `rf >= rsf`, `rf <= ref`, `int(float(success)) == 1`, and `float(confidence) >= 0.7`. If `n_valid == 0` in `[rsf, ref]`, raise `RuntimeError(f"no valid OpenFace frames in [{rsf}, {ref}]")` (honest null).
- **Columns (`8` continuous eye-gaze columns):**
  `OPENFACE_GAZE_COLS = ("gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y")`.
- **Feature vector (`26` length-invariant features, `dtype=np.float32`):**
  1. **Means (`8` dims, `of_<col>_mean`):** `np.mean(sub, axis=0)` over valid frames in `[rsf, ref]`.
  2. **Standard deviations (`8` dims, `of_<col>_std`):** `np.std(sub, axis=0)` if `n_valid >= 2` else `np.zeros(8, dtype=np.float32)`.
  3. **Late-minus-early deltas (`8` dims, `of_<col>_delta`):** `np.mean(sub[half:], axis=0) - np.mean(sub[:half], axis=0)` where `half = n_valid // 2` if `n_valid >= 2` else `np.zeros(8, dtype=np.float32)`.
  4. **Angular speed mean and delta (`2` dims, `of_gaze_angle_speed_mean`, `of_gaze_angle_speed_delta`):**
     Let `gx = sub[:, 6]` (`gaze_angle_x`) and `gy = sub[:, 7]` (`gaze_angle_y`). For `n_valid >= 2`, define frame-to-frame angular step `step_speed = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)` (`len = n_valid - 1`):
     - `of_gaze_angle_speed_mean = float(np.mean(step_speed))` if `len(step_speed) >= 1` else `0.0`;
     - `of_gaze_angle_speed_delta = float(np.mean(step_speed[shalf:]) - np.mean(step_speed[:shalf]))` where `shalf = len(step_speed) // 2` if `len(step_speed) >= 2` else `0.0`.
- **Coverage on `reflex_same_phase` (verified label-blind before feature extraction):** `376 / 380` valid train items (`186 / 190` complete train pairs) and `157 / 170` valid test items (`77 / 85` complete test pairs).

### 8.3 Probe, controls, confirmatory rule, and diagnostic matrix

- **Probe and bootstrap (fixed across the repo):**
  - `StandardScaler` fit on valid train items only + `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")`.
  - Grouped bootstrap CIs (`auroc_ci` and `delta_auroc_ci`, `1,000` resamples, `default_rng(0)`, grouped by participant `group_id`).
- **Action-only baseline on `reflex_same_phase`:**
  - Uses the existing cached `siglip-b16-224-u9` (`action-probe`), cached `qwen2.5vl_7b` (`judge`), cached `gemma4_26b` (`judge-large`), physical-phase `step-prior`, and `action-fusion` (`[logit(judge), logit(step-prior), X_action]`) on the `380` train and `170` test items.
  - `openface_fusion` concatenates `[logit(judge), logit(step-prior), X_action, X_openface_gaze]` on items where judge, action, and OpenFace gaze are all valid (`mtr`, `mte`), and is compared against `action-fusion` on the same valid mask (`same_phase.openface_fusion_minus_action_fusion`) and against `action_best` (`same_phase.openface_fusion_minus_action_best`).
- **Pre-registered confirmatory decision rule (evaluated once on `same_phase`):**
  1. **Clause 2 (positive ablation gain):** **held** iff `same_phase.openface_fusion_minus_action_fusion` has `ci_low > 0.0`.
  2. **Standalone above-chance check:** **held** iff `same_phase.openface_gaze` has `ci_low > 0.5`.
  3. **H1 pass bar on `same_phase`:** **met** iff `same_phase.openface_gaze` point AUROC `>= 0.65` **and** `same_phase.openface_fusion_minus_action_fusion` point Δ `>= +0.03` with `ci_low > 0.0`.
  4. **Interpretation:**
     - If both (1) and (2) hold (and survive the pair-complete check `same_phase.openface_gaze_pair_complete`), automatic camera gaze adds genuine phase-matched information about robot task failure on REFLEX.
     - If either (1) or (2) fails (`ci_low <= 0.0` on the ablation gain or `ci_low <= 0.5` standalone), automatic camera gaze does **not** support a phase-matched reaction signal on REFLEX; the confirmatory result is reported as negative alongside v1 and exploratory v2.
- **Secondary diagnostic checks computed in the same single run (`docs/evals/2026-11-09_reflex_openface_gaze.json`):**
  1. **Pair-complete and phase-stratum checks on `same_phase`:**
     - `same_phase.openface_gaze_pair_complete` and `same_phase.openface_fusion_minus_action_fusion_pair_complete` (restricted to the `186` train and `77` test pairs where both items of the pair have valid OpenFace frames);
     - `same_phase.openface_gaze_pick` (`pick_1` + `pick_2`, `68` test items before exclusions) and `same_phase.openface_gaze_carry` (`carry_1` + `carry_2` + `carry_3`, `102` test items before exclusions).
  2. **Semi-manual `gaze.csv` (`GAZE_IDX`, 9-d) and automatic face-only (`FACE_IDX`, 38-d) on `same_phase`:**
     - `same_phase.dataset_gaze`, `same_phase.dataset_gaze_fusion`, `same_phase.dataset_gaze_fusion_minus_action_fusion`, `same_phase.dataset_gaze_pick`, `same_phase.dataset_gaze_carry`;
     - `same_phase.face_only`, `same_phase.face_only_fusion`, `same_phase.face_only_fusion_minus_action_fusion`;
     - `same_phase.v2_all`, `same_phase.v2_all_fusion`, `same_phase.v2_all_fusion_minus_action_fusion`.
     - *Purpose:* tests whether even the dataset authors' semi-manual `gaze.csv` labels (which scored `0.720` with `+0.053` gain within-episode) survive phase-matching or collapse once `Pre` and `Failure` are matched to the same physical phase.
  3. **Automatic `openface_gaze` on the original within-episode `reflex` (`1,208` items, `action_equal_end` and `equal_start`):**
     - `reflex.openface_gaze_action_equal_end`, `reflex.openface_fusion_action_equal_end`, `reflex.openface_fusion_minus_action_fusion_action_equal_end`, `reflex.openface_fusion_minus_action_best_action_equal_end`;
     - `reflex.openface_gaze_equal_start`, `reflex.openface_fusion_equal_start`, `reflex.openface_fusion_minus_action_fusion_equal_start`, `reflex.openface_fusion_minus_action_best_equal_start`;
     - `reflex.placebo_early_vs_late_pre_openface_gaze` and `reflex.onset_straddle_openface_gaze`.
     - *Purpose:* directly compares automatic `openface.csv` gaze against semi-manual `gaze.csv` (`0.720` / `+0.053` on `action_equal_end`, `0.718` / `+0.038` on `equal_start`) on the exact same within-episode windows.
  4. **Uncorrupted full-object handlings (`Gap 2`: Round 1 Object C, and `Gap 4`: Round 2 Object B) vs. failed episodes (`uncorrupted_gap`):**
     - In every participant, Round 1 Object C occurs in the gap between `(Round 1, Object B, Carry)` (`Post End Frame`) and `(Round 1, Object D, Place)` (`Pre Start Frame`), and Round 2 Object B occurs in the gap between `(Round 2, Object A, Carry)` (`Post End Frame`) and `(Round 2, Object C, Pick)` (`Pre Start Frame`; `54/55` participants have Round 2 Object C).
     - `gaze.csv` has **zero rows** in these uncorrupted successful object handlings because `gaze.csv` was annotated only during failure episodes, whereas `openface.csv` was tracked continuously across the full video.
     - Define two pairs per participant (or one for `D2-8`, which lacks Round 2 Object C):
       - `gap_1`: Positive (`label = 1`) = terminal `K` frames of Round 1 Object C's handling gap (`[r1d_pre_start - K, r1d_pre_start - 1]`) vs. Negative (`label = 0`) = terminal `K` frames of `(Round 1, Object D, Place, Failure)` (`[nef - K + 1, nef]`), where `K = r1d_place_fail_end - r1d_place_fail_start + 1`;
       - `gap_2` (for the 54 participants with Round 2 Object C): Positive (`label = 1`) = terminal `K` frames of Round 2 Object B's handling gap (`[r2c_pre_start - K, r2c_pre_start - 1]`) vs. Negative (`label = 0`) = terminal `K` frames of `(Round 3, Object A, Place, Failure)` (`[nef - K + 1, nef]`), where `K = r3a_place_fail_end - r3a_place_fail_start + 1`.
     - Report `uncorrupted_gap.openface_gaze` and `uncorrupted_gap.openface_gaze_pair_complete`.

### 8.4 Result of the pre-registered confirmatory test (November 9, 2026)

**The pre-registered confirmatory verdict is `NOT SUPPORTED`** (`docs/evals/2026-11-09_reflex_openface_gaze.json`; `tools/eval_reflex_openface_gaze.py`).
- **Primary confirmatory result on `same_phase` (`380` train items [`376` OpenFace-valid, `186` complete pairs]; `170` test items [`157` OpenFace-valid, `77` complete pairs] across `17` test participants):**
  - **Window length and `step-prior` controls:** Equal-length end-anchored windows have `same_phase.window_length` = **0.500 [0.500, 0.500]** (`170` test items), and physical-phase `same_phase.step_prior` = **0.500 [0.500, 0.500]** (`170` test items). Action-only controls score `same_phase.action_probe` = **0.831 [0.786, 0.891]**, `same_phase.judge` = **0.601 [0.540, 0.661]**, `same_phase.judge_large` = **0.500 [0.500, 0.500]**, and `same_phase.action_fusion` = **0.861 [0.814, 0.918]** (`170` test items).
  - **Standalone OpenFace gaze (`same_phase.openface_gaze`, 26-d):** **0.717 [0.653, 0.789]** (`157` valid test items; `ci_low` = 0.653 > 0.5, above chance and $\ge 0.65$).
  - **Multimodal fusion and paired ablation gain:** `same_phase.openface_fusion` scores **0.891 [0.851, 0.941]** (`157` test items), for a paired ablation gain `same_phase.openface_fusion_minus_action_fusion` of **+0.035 [−0.002, +0.063]** (`same_phase.openface_fusion_minus_action_best` = **+0.037 [−0.001, +0.067]**).
  - **Pre-registered decision rule (§8.3):** Although standalone `same_phase.openface_gaze` (0.717 [0.653, 0.789]) exceeds chance (`standalone_above_chance = True`) and its point ablation gain (+0.035) exceeds +0.03, the paired ablation CI covers 0.0 (`ci_low` = −0.002 $\le 0.0$; `ablation_gain_positive = False`, `confirmatory_supported = False`, `h1_pass_bar_met = False`), so the pre-registered confirmatory verdict is **`NOT SUPPORTED`**.
- **Pair-complete and phase-stratum checks on `same_phase`:**
  - Restricted to the `77` test pairs where both items of the pair have valid OpenFace frames (`154` test items), `same_phase.openface_gaze_pair_complete` scores **0.730 [0.660, 0.803]** and `same_phase.openface_fusion_minus_action_fusion_pair_complete` scores **+0.040 [+0.004, +0.068]** (indicating that the ~3.5–4.0 point fusion gain is right at the boundary of statistical resolution across 17 test participants).
  - Within physical phases, `same_phase.openface_gaze_pick` scores **0.662 [0.512, 0.806]** (`63` valid test items) and `same_phase.openface_gaze_carry` scores **0.745 [0.648, 0.838]** (`94` valid test items).
- **Same-phase comparison with semi-manual `gaze.csv` (`GAZE_IDX`, 9-d) and automatic `face_only` (`FACE_IDX`, 38-d) on `same_phase` (`170` test items):**
  - The dataset authors' semi-manual `gaze.csv` labels score `same_phase.dataset_gaze` = **0.802 [0.741, 0.862]** (`same_phase.dataset_gaze_pick` = **0.734 [0.635, 0.827]** on `68` items; `same_phase.dataset_gaze_carry` = **0.847 [0.783, 0.909]** on `102` items), raising `action-fusion` to `same_phase.dataset_gaze_fusion` = **0.924 [0.894, 0.958]** with a paired ablation gain `same_phase.dataset_gaze_fusion_minus_action_fusion` of **+0.063 [+0.029, +0.091]** (`same_phase.dataset_gaze_fusion_minus_action_best` = **+0.063 [+0.029, +0.091]**).
  - Automatic facial expressions (`same_phase.face_only`, 38 Hume + Facetorch features) score **0.708 [0.658, 0.756]** standalone, but `same_phase.face_only_fusion` (**0.862 [0.822, 0.911]**) adds zero gain over `action-fusion` (`same_phase.face_only_fusion_minus_action_fusion` = **+0.001 [−0.022, +0.024]**; all 47 v2 features `same_phase.v2_all` = **0.829 [0.781, 0.881]**, `same_phase.v2_all_fusion` = **0.917 [0.888, 0.952]**, gain **+0.056 [+0.015, +0.095]**).
- **Within-episode `reflex` comparison (`1,208` items) and uncorrupted gap check (`uncorrupted_gap`):**
  - On the original within-episode `reflex` task (`Pre` vs. `Failure` of the same episode), automatic `openface.csv` gaze scores `reflex.openface_gaze_action_equal_end` = **0.604 [0.534, 0.669]** (`347` valid test items; vs. 0.720 [0.689, 0.752] for semi-manual `gaze.csv`), with `reflex.openface_fusion_action_equal_end` = **0.798 [0.748, 0.845]** and zero ablation gain (`reflex.openface_fusion_minus_action_fusion_action_equal_end` = **+0.007 [−0.021, +0.030]**, `reflex.openface_fusion_minus_action_best_action_equal_end` = **−0.003 [−0.028, +0.019]**). On `equal_start` (`348` valid test items), `reflex.openface_gaze_equal_start` scores **0.648 [0.584, 0.714]** (`reflex.openface_fusion_equal_start` = **0.789 [0.736, 0.843]**, `reflex.openface_fusion_minus_action_fusion_equal_start` = **−0.002 [−0.021, +0.016]**, `reflex.openface_fusion_minus_action_best_equal_start` = **−0.010 [−0.029, +0.009]**).
  - On the within-episode early-vs-late `Pre` placebo (`341` valid test items), `reflex.placebo_early_vs_late_pre_openface_gaze` scores **0.587 [0.551, 0.623]**, and on the failure-onset straddle (`344` valid test items), `reflex.onset_straddle_openface_gaze` scores **0.603 [0.545, 0.660]**.
  - On the uncorrupted full-object handlings (`Gap 2` Round 1 Object C and `Gap 4` Round 2 Object B vs. matched-duration `Place.Failure`, where `gaze.csv` has zero rows), `uncorrupted_gap.openface_gaze` scores **0.679 [0.550, 0.800]** (`63` valid test items; `uncorrupted_gap.openface_gaze_pair_complete` = **0.671 [0.541, 0.787]** on `62` items).


