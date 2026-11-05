# Agent Execution Guide: Active Build — Wave B: REFLEX Multimodal Human-Robot Reactions (B1–B4), November 1, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:**
  - On October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
  - Waves A and D v1 (`A1`–`A15`, `D1`–`D4`; commits `651039b` … `65882bf`) are complete and verified by the designer (§1.5, §5.1).
  - On November 1, 2026, after Option E for Issue 1 (`M4` `amfed@affectiva.com` bounced; `M5` taste-liking database author did not reply) failed to yield access and `BAD` (Issue 4) remained gated behind an institutional IRB, the maintainer approved pivoting Wave B to the open-access **REFLEX** dataset ([Zenodo 14590226](https://zenodo.org/records/14590226), DOI `10.5281/zenodo.14590226`, **CC BY 4.0**, 37.22 GB, 55 participants reacting to a physical Baxter robot across 604 failure episodes; `00_thesis.md` entry 28).
- **What is approved right now:** **Wave B (`B1` → `B2` → `B3` → `B4`)**, specified in [`05_wave_b_reflex.md`](05_wave_b_reflex.md) and §2–§3 below.
  - It evaluates human facial/gaze/head/posture reactions (`react-face`, encoder `reflex-face-v1`, fulfilling `DW2`) and vocal prosody reactions (`react-nonverbal`, encoder `reflex-speech-v1`) on two datasets derived from REFLEX:
    1. `reflex` (1,208 candidate items): immediate reaction to a robot manipulation failure (`Pre` `label = 1` vs. `Failure` `label = 0`).
    2. `reflex_resolve` (593 candidate items): hidden-outcome collaborative resolution (`[Failure.Start Frame, Explanation.End Frame]` action window before resolution $\rightarrow$ `Resolved?` `TRUE` `label = 1` vs. `FALSE` `label = 0`).
  - It also exports the fitted `react-face` linear probes (`DATA_ROOT/transfer/{reflex,reflex_resolve}_face_probe.npz`) for downstream use in `DW6` (Stage A live emotional feedback on the Pollen Robotics MicroDuck).

**Every path, number, field name and literal string in this guide and in [`05_wave_b_reflex.md`](05_wave_b_reflex.md) is a decision, not a suggestion.**

---

## 0. Standing constraints (apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare (`--slow` when the item touches or runs models). G2 must pass **3 times in a row** with `TMPDIR` on the SSD.
   - Read every exit code bare.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only (no `match`, no runtime `X | Y`).
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies and models are frozen.** No new pip packages, no new model downloads, and no Ollama pulls. The only approved data download is `Dataset.zip` from Zenodo `14590226` in item **B1** (`05_wave_b_reflex.md` §1).
4. **Data lives under `DATA_ROOT`** (`/Volumes/Extreme SSD/social_robotics`; exFAT, 1.0 TiB free). No video or audio on the internal disk. Never delete anything you did not create (except deleting `DATA_ROOT/raw/reflex/Dataset.zip` after B1 verifies extraction, as required by `05_wave_b_reflex.md` §1). All globs over the SSD must use `shared.fsutil.data_glob` (Lesson L13).
5. **Long runs (> 30 min) run detached** (`tools/daemonize.py` + `tools/run_supervised.sh`; confirm `PPID 1`). Runners are resumable and end with the count line (`items_in=... items_out=... excluded=... elapsed_s=...`).
6. **Memory (`03_eval_harness.md` §12).**
   - Never run two model-loading jobs at once.
   - Run `PYTHONPATH=src ./venv/bin/python -m shared.memguard --status` before each model job and record its output. Exit 75 is never a pass.
   - **`gemma4:26b` is shared with `animated_infographics`; never unload it.**
7. **The scorecard is append-only.** Nulls carry derived reasons. Reports are generated from rows. Verdicts follow the pinned rules (`03_eval_harness.md` §11.8; `05_wave_b_reflex.md` §6).
8. **Pre-registered rules are never changed after their result is seen.** That includes `05_wave_b_reflex.md` §1–§6, the pass bars and the kill rule.
9. **Commits:**
   - one item = one Conventional Commit on `main` (`feat(b1): …`, `feat(b2): …`, `feat(b3): …`, `feat(b4): …`), with the WHY, the red and green runs, and any missed validation bar in the body (L14);
   - stage by explicit paths only;
   - record the resolution line in `docs/ongoing_general_errors.md` §3;
   - push with `/usr/bin/git push origin main`;
   - no branches, no PRs, never amend.
10. **When a contract and this guide disagree, or a bar cannot be met, STOP and file a new issue** (the next number is **Issue 11**) in `docs/ongoing_general_errors.md`, with options.
11. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer.

---

## 1. Verified baseline (November 1, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1.
- **External SSD (`/Volumes/Extreme SSD`):** 1.0 TiB free (`df -h`).
- **Memory guard (`python -m shared.memguard --status`):** Available memory `57.6 GB` (pressure level `1`), heavy lock `free`, Ollama models `(none loaded)`.
- **`venv`, Python 3.9.6:** torch 2.8.0, transformers 4.46.3, funasr 1.3.1, scikit-learn 1.6.1, scipy 1.13.1, numpy 2.0.2, huggingface_hub 0.36.2, ruff 0.15.11, pytest 8.4.2.

### 1.2 Repository

- `HEAD` at baseline verification: `65882bf` (`docs: record Issue 1 = E and the October 23 access requests (M4, M5)`), plus this November 1, 2026 Wave B specification commit.

### 1.3 Data on the SSD

| What | Where | State |
|---|---|---|
| Oops! | `raw/oops/`, `items/oops/`, `features/oops/`, `judge/oops/` | complete |
| HoloAssist (`holoassist`, `holoassist_audible`, `holoassist_pool`) | `raw/holoassist/`, `items/…`, `features/…`, `judge/…` | complete; per-session audibility in `runs/holoassist_session_audibility.jsonl` |
| RoboReward (all of it) | `raw/roboreward/` (`DOWNLOAD.json`: 54,140 files, 43,204,474,744 bytes) | complete; `items/roboreward/` (4,000 train pool + 1,517 test); `features/roboreward/siglip-b16-224-u9/`; `judge/roboreward/gemma4_26b/` |
| Wave D artefacts | `transfer/{labeler,scaler,heads,curve_scores}.npz`, `transfer/holoassist_pool_pseudolabels.jsonl` | complete (v1) |
| REFLEX | `raw/reflex/` (`DOWNLOAD.json`: 55 participants, 665 files); `items/reflex/` (1,208 items); `items/reflex_resolve/` (604 items); `features/{reflex,reflex_resolve}/{reflex-face-v1,reflex-speech-v1,siglip-b16-224-u9}/`; `judge/{reflex,reflex_resolve}/{qwen2.5vl_7b,gemma4_26b}/` | complete (B1–B3) |

### 1.4 Gates (run bare November 4, 2026 on B3, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result on B3 (November 4, 2026) |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · `All checks passed!` |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **115 passed, 11 deselected**, 3 runs out of 3 (73.08 s, 69.92 s, 69.17 s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS (`planted signal auroc=0.830 [0.807, 0.852]`, `null ci=[0.459, 0.523]`, `grouping ratio=4.66`, `paired delta ci_low=0.138`) |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **11 passed, 115 deselected** (205.56 s) |

### 1.5 Headline results (the scorecard is the source)

| Dataset | Result | Row `ts` |
|---|---|---|
| `oops` | ablation gain +0.015 [−0.001, 0.029]; `action-probe` 0.779; `judge-large` 0.708 | `2026-10-21T04:12:36–42Z` |
| `holoassist` | `step-prior` 0.751; `action-fusion` 0.682; no reaction Δ (the audio was inaudible in most sessions) | `2026-10-21T04:12:57Z–04:13:02Z` |
| `holoassist_audible` | `react-nonverbal` 0.654; **ablation gain +0.021 [0.006, 0.034]**; fusion − `step-prior` −0.065 | `2026-10-21T16:50:20–30Z` |
| `roboreward` (Wave D v1) | **NOT SUPPORTED**: `gain:react:N=100` +0.0009 [−0.0006, 0.0024]; `gain:truth:N=100` −0.0507; `robot-full` 0.657; `judge-large` 0.747 | `2026-10-22T14:22:28Z–14:24:54Z` |

---

## 2. Execution order

Execute strictly in order, one commit per item on `main`:

1. **`B1`** — REFLEX download (`tools/download_reflex.py`), extraction, and item/split builder (`src/sources/reflex.py`) for `reflex` and `reflex_resolve`.
2. **`B2`** — REFLEX reaction encoders (`src/features/face.py`: `reflex-face-v1` and `reflex-speech-v1`), CLI wiring (`src/features/extract.py`), judge questions (`src/judge/vlm_judge.py`), and probe conditions (`src/harness/probes.py`).
3. **`B3`** — REFLEX feature extraction (`reflex-face-v1`, `reflex-speech-v1`, `siglip-b16-224-u9`) and VLM judge runs (`qwen2.5vl:7b` train+test, `gemma4:26b` test) on `reflex` and `reflex_resolve`.
4. **`B4`** — Wave B probe evaluation, scorecard rows, exported MicroDuck face probes (`DATA_ROOT/transfer/{reflex,reflex_resolve}_face_probe.npz`), evaluation report (`tools/report_reflex.py`), and Wave B close-out.

---

## 3. The items

### Item B1 — REFLEX download, verification, extraction, and item/split builder (`reflex` and `reflex_resolve`)

**Read first:** [`05_wave_b_reflex.md`](05_wave_b_reflex.md) §1–§3, [`02_data_sources.md`](02_data_sources.md) → REFLEX, [`03_eval_harness.md`](03_eval_harness.md) §3 & §5.

**Deliverables:**

1. **`tools/download_reflex.py`:**
   - Resumable single-stream download of `https://zenodo.org/api/records/14590226/files/Dataset.zip/content` to `DATA_ROOT/raw/reflex/Dataset.zip`.
   - **Rate-limit constraint (`05_wave_b_reflex.md` §1):** Zenodo returns `HTTP 429` on multi-connection downloads. Use a single sequential stream (`curl -L -C - --retry 5 --retry-delay 10 -o <zip_path> <url>` or single-connection Python HTTP with `Range` header on resume) and update `DATA_ROOT/raw/reflex/progress.json` so `tools/run_supervised.sh` can monitor progress.
   - Verify exact byte size `37219808401` and MD5 `334caac04f3e607ac91182dea7a03929`.
   - Extract `Dataset.zip` into `DATA_ROOT/raw/reflex/`, skipping any zip member whose path starts with `__MACOSX/` or whose basename starts with `._`.
   - Verify all 5 strategy folders (`C1-Fixed-Low`, `C2-Fixed-Medium`, `C3-Fixed-High`, `D1-Decay-Smooth`, `D2-Decay-Rapid`) and all 55 participant folders (`11` per strategy), confirming that every participant directory contains all 10 required files (`analysis.csv`, `body.csv`, `facetorch.csv`, `gaze.csv`, `hume.csv`, `openface.csv`, `questionnaire.csv`, `speech.csv`, `time.csv`, `video_cam1.mp4`).
   - Delete `DATA_ROOT/raw/reflex/Dataset.zip` after verification and write `DATA_ROOT/raw/reflex/DOWNLOAD.json` with `{zenodo_record: "14590226", doi: "10.5281/zenodo.14590226", license: "CC-BY-4.0", zip_bytes: 37219808401, zip_md5: "334caac04f3e607ac91182dea7a03929", n_participants: 55, verified: true, archive_deleted: true, ts: ...}`.
   - If `DATA_ROOT/raw/reflex/DOWNLOAD.json` already exists with `verified: true` and all 55×10 files exist on disk, exit 0 idempotently without re-downloading (unless `--force` is passed).

2. **`src/sources/reflex.py`:**
   - CLI: `PYTHONPATH=src ./venv/bin/python -m sources.reflex --dataset {reflex,reflex_resolve,all} [--force-split]`.
   - Implements the exact item, time-mapping, metadata, and split rules of `05_wave_b_reflex.md` §2–§3:
     - Uses `ffprobe` on each participant's `video_cam1.mp4` to read container `fps` (`r_frame_rate` / `avg_frame_rate`) and container `duration`.
     - Maps 1-indexed frame bounds `[sf, ef]` to container seconds `[round((sf - 1) / fps, 4), round(min(duration, (ef - 1) / fps), 4)]` for `action_window_sec` and `reaction_window_sec`.
     - Maps 1-indexed frame bounds `[sf, ef]` to wall-clock seconds via `time.csv` (`Seconds` column) for `meta["wall_action_window_sec"]` and `meta["wall_reaction_window_sec"]`.
     - Computes `meta["spoke"]` (`0` or `1`) and `meta["transcript"]` from `speech.csv` rows where `Id.strip().lower() in {"participant", "robot/participant"}` and `[BeginTime, EndTime]` overlaps `wall_reaction_window_sec`.
     - **Never** includes `Object`, `State`, or `Resolved?` in `context_text` (`05_wave_b_reflex.md` §3.2–§3.3).
     - Validates items with `validate_items()`, writes `DATA_ROOT/items/{reflex,reflex_resolve}/items.jsonl` and `stats.json` (recording total episodes seen, items kept per class, and skip counts by reason: `missing_video`, `invalid_frames`, `invalid_window`, `missing_resolved_label`), and writes `splits/{reflex,reflex_resolve}.json` using the shared 55-participant grouped 70/30 partition (`38` train participants / `17` test participants, `default_rng(0)`).

3. **`tests/test_reflex_source.py`:**
   - Fast unit tests on a synthetic REFLEX directory fixture (2 strategies, 4 participants, synthetic `analysis.csv`, `time.csv`, `speech.csv`, and tiny `ffmpeg`-generated `video_cam1.mp4`) verifying:
     - 1-indexed frame to container-second conversion (`(sf - 1) / fps`) vs. wall-clock second lookup in `time.csv`;
     - `reflex` `pre` (`label=1`) vs `fail` (`label=0`, reaction window capped at `min(Explanation.End Frame, Failure.End Frame + 22, max_frame)`);
     - `reflex_resolve` (`label=1` on `TRUE`, `label=0` on `FALSE`, skipping empty/other `Resolved?` with reason `missing_resolved_label`, action window `[Failure.Start Frame, Explanation.End Frame]` vs. reaction window `[Failure.Start Frame, Resolution.End Frame]`);
     - `Object` leakage test: a banned failure word (`"Drop"`, `"Wrong"`, `"Fail"`) placed in `analysis.csv`'s `Object` column never appears in `context_text` or `meta.noun`;
     - identical train/test participant sets between `reflex` and `reflex_resolve` (`0` straddling groups).

**Validation (must all pass before committing `B1`):**
- Run `tools/download_reflex.py` detached via `tools/daemonize.py` + `tools/run_supervised.sh` (confirm `PPID 1`).
- Confirm `DATA_ROOT/raw/reflex/DOWNLOAD.json` has `verified: true` and `archive_deleted: true`, and `df -h "/Volumes/Extreme SSD"` shows ≥ 900 GiB free.
- Run `PYTHONPATH=src ./venv/bin/python -m sources.reflex --dataset all` and verify:
  - `validate_items()` returns `0` errors on both `reflex` and `reflex_resolve`;
  - `splits/reflex.json` and `splits/reflex_resolve.json` have `38` train groups, `17` test groups, and `0` straddling groups;
  - no item's `context_text` contains any substring from `{"drop", "wrong", "fail", "true", "false", "resolved"}` (case-insensitive).
- Run `scripts/battery.sh` bare (with G2 3 times in a row).
- Commit as `feat(b1): download REFLEX and build reflex and reflex_resolve items and splits` and push to `main`.

---

### Item B2 — REFLEX reaction encoders (`src/features/face.py`), CLI wiring (`src/features/extract.py`), judge questions (`src/judge/vlm_judge.py`), and probe conditions (`src/harness/probes.py`)

**Read first:** [`05_wave_b_reflex.md`](05_wave_b_reflex.md) §4–§5, [`03_eval_harness.md`](03_eval_harness.md) §6, §8, §9.

**Deliverables:**

1. **`src/features/face.py`:**
   - Implement `ReflexFaceEncoder` (`encoder_id = "reflex-face-v1"`, output shape `(153,)`, `dtype = np.float32`) and `ReflexSpeechEncoder` (`encoder_id = "reflex-speech-v1"`, output shape `(96,)`, `dtype = np.float32`) following `05_wave_b_reflex.md` §4.1–§4.2 to the letter:
     - Strip whitespace from all CSV column headers on load (`openface.csv` columns have leading spaces like `" frame"`).
     - Cache parsed per-participant CSV tables in memory keyed by `str(participant_dir)` so extracting all items of a participant reads each CSV only once.
     - Enforce the 1-indexed (`openface.csv` `frame` in `[rsf, ref]`) vs. 0-indexed (`facetorch.csv` `Frame ID` and `body.csv` `Frame` in `[rsf - 1, ref - 1]`) mapping.
     - Enforce the honest-null rule (Lesson L3): raise `RuntimeError` if `0` valid rows match in `openface.csv` (`success == 1` and `confidence >= 0.7`), `facetorch.csv`, `body.csv`, or `speech.csv` (for `reflex-speech-v1`), or if any output element is `NaN`/`Inf`. Never write a zero vector on missing data.

2. **`src/features/extract.py`:**
   - Add `"reflex-face-v1"` and `"reflex-speech-v1"` to `--encoder` choices and `run_extraction()`.
   - For `"reflex-face-v1"` and `"reflex-speech-v1"`, pass `item` (or `item.video_path, item.reaction_window_sec, item.meta`) to the encoder; do not enter `memguard` for these lightweight CSV encoders (keep `memguard` for `siglip-b16-224`, `siglip-b16-224-u9`, and `e2v-plus-large`).

3. **`src/judge/vlm_judge.py`:**
   - Add the two verbatim questions to `DATASET_QUESTIONS` (`03_eval_harness.md` §8; `05_wave_b_reflex.md` §3.2–§3.3):
     - `"reflex": "Is the robot performing the step normally at this point, with no failure yet?"`
     - `"reflex_resolve": "Will the human and the robot successfully resolve this failure and complete the step?"`

4. **`src/harness/probes.py`:**
   - Add `REFLEX_DATASETS = ("reflex", "reflex_resolve")` support per `05_wave_b_reflex.md` §5:
     - Keep `oops`, `holoassist`, and `holoassist_audible` paths 100% unchanged (their config hashes and scorecard values must remain identical).
     - On `REFLEX_DATASETS`:
       - Load action features from `siglip-b16-224-u9`, face reaction features from `reflex-face-v1` (condition `react-face` + `react-face:shuffled`), and speech prosody features from `reflex-speech-v1` (condition `react-nonverbal` + `react-nonverbal:shuffled` + diagnostic `react-nonverbal|spoke=1`).
       - Handle sparse speech gracefully (`03_eval_harness.md` §6): if `valid_re_tr` has `< 2` classes or `0` items, or if `valid_re_te` has `< 2` classes on test, write a clean `not run` or `partial: ...; metric undefined: ...` row for `react-nonverbal` and `react-nonverbal:shuffled` rather than raising from `LogisticRegression.fit` or `auroc_ci`. Likewise, if `react-full` TF-IDF has an empty vocabulary or `< 2` classes, write a clean `not run` row with derived notes.
       - Compute `step-prior` and `step-prior:shuffled` via `step_prior_scores()` on `(meta.verb, meta.noun)`, entering `action_best`, `action-fusion`, and `fusion`.
       - Fit `fusion` and `action-fusion` on `fusion_tr_mask = valid_j_tr & valid_act_tr & valid_face_tr` and score on `fusion_te_mask = valid_j_te & valid_act_te & valid_face_te` per `05_wave_b_reflex.md` §5.
       - Save the fitted `react-face` probe (`mean`, `scale`, `coef`, `intercept`, `dataset`) to `DATA_ROOT/transfer/{dataset}_face_probe.npz`.

5. **`tests/test_reflex_encoders_and_probes.py`:**
   - Fast synthetic unit tests covering:
     - `ReflexFaceEncoder`: exact 153-d output shape/dtype, whitespace-padded OpenFace header handling, 1-indexed vs. 0-indexed frame alignment, and raising `RuntimeError` when `confidence < 0.7` or `success == 0` on all frames in the window;
     - `ReflexSpeechEncoder`: exact 96-d output shape/dtype on overlapping participant utterances, ignoring `"robot"`-only rows, and raising `RuntimeError` when no participant speech overlaps the window;
     - `run_probes` on synthetic `reflex` and `reflex_resolve` caches: produces all expected conditions (`judge`, `judge-large`, `judge-frontier`, `action-probe`, `react-face`, `react-nonverbal`, `react-nonverbal|spoke=1`, `react-spoke`, `react-full`, `step-prior`, `fusion`, `action-fusion`, `fusion_minus_action_fusion`, `fusion_minus_action_best`, plus the 6 `:shuffled` controls), handles sparse speech without crashing, and writes `{dataset}_face_probe.npz`.

**Validation (must all pass before committing `B2`):**
- Run `pytest -q tests/test_reflex_encoders_and_probes.py`.
- Smoke-test `reflex-face-v1` and `reflex-speech-v1` extraction on `--limit 10` real items from `reflex` and `reflex_resolve`, confirming shapes `(153,)` and `(96,)` and finite `float32` values.
- Run `scripts/battery.sh` bare (with G2 3 times in a row).
- Commit as `feat(b2): add reflex-face-v1 and reflex-speech-v1 encoders and REFLEX probe conditions` and push to `main`.

---

### Item B3 — REFLEX feature extraction and VLM judge evaluation (`reflex` and `reflex_resolve`)

**Read first:** [`05_wave_b_reflex.md`](05_wave_b_reflex.md) §4–§5, [`03_eval_harness.md`](03_eval_harness.md) §8, §9, §12.

**Deliverables (data & cache artefacts under `DATA_ROOT`; no code changes required unless a bug is found):**

1. **Extract CSV reaction features (`reflex-face-v1` and `reflex-speech-v1`) on all items:**
   - `PYTHONPATH=src ./venv/bin/python -m features.extract --dataset reflex --encoder reflex-face-v1 --split all`
   - `PYTHONPATH=src ./venv/bin/python -m features.extract --dataset reflex --encoder reflex-speech-v1 --split all`
   - `PYTHONPATH=src ./venv/bin/python -m features.extract --dataset reflex_resolve --encoder reflex-face-v1 --split all`
   - `PYTHONPATH=src ./venv/bin/python -m features.extract --dataset reflex_resolve --encoder reflex-speech-v1 --split all`

2. **Extract action visual features (`siglip-b16-224-u9`) sequentially under `memguard`:**
   - Record `PYTHONPATH=src ./venv/bin/python -m shared.memguard --status` first.
   - `PYTHONPATH=src ./venv/bin/python -m features.extract --dataset reflex --encoder siglip-b16-224-u9 --split all`
   - `PYTHONPATH=src ./venv/bin/python -m features.extract --dataset reflex_resolve --encoder siglip-b16-224-u9 --split all`

3. **Run VLM judges sequentially under `memguard` (use detached supervision via `tools/daemonize.py` + `tools/run_supervised.sh` for any run > 30 min; confirm `PPID 1`):**
   - Local judge (`qwen2.5vl:7b`, `--backend ollama`) on `train` and `test` for `reflex` and `reflex_resolve`:
     - `PYTHONPATH=src ./venv/bin/python -m judge.vlm_judge --dataset reflex --split train --backend ollama`
     - `PYTHONPATH=src ./venv/bin/python -m judge.vlm_judge --dataset reflex --split test --backend ollama`
     - `PYTHONPATH=src ./venv/bin/python -m judge.vlm_judge --dataset reflex_resolve --split train --backend ollama`
     - `PYTHONPATH=src ./venv/bin/python -m judge.vlm_judge --dataset reflex_resolve --split test --backend ollama`
   - Large local judge (`gemma4:26b`, `--backend ollama-large`, test split only; **never unload `gemma4:26b`**) on `reflex` and `reflex_resolve`:
     - `PYTHONPATH=src ./venv/bin/python -m judge.vlm_judge --dataset reflex --split test --backend ollama-large`
     - `PYTHONPATH=src ./venv/bin/python -m judge.vlm_judge --dataset reflex_resolve --split test --backend ollama-large`

**Validation (must all pass before committing `B3`):**
- Verify coverage on both `reflex` and `reflex_resolve`:
  - `siglip-b16-224-u9`: $\ge 99\%$ of items cached;
  - `reflex-face-v1`: $\ge 95\%$ of items cached (any exclusions logged in `errors.jsonl` with honest `no valid OpenFace/Facetorch/Body frames` reasons);
  - `qwen2.5vl_7b` (`train` and `test`) and `gemma4_26b` (`test`): $\ge 98\%$ of items scored with non-null `judge_prob`.
- Run `scripts/battery.sh --slow` bare (with G2 3 times in a row).
- Record the B3 resolution line (exact counts, coverage %, and wall-clock times) in `docs/ongoing_general_errors.md` §3, commit as `feat(b3): extract REFLEX features and evaluate local and large VLM judges` and push to `main`.

---

### Item B4 — Wave B probe evaluation, scorecard rows, `tools/report_reflex.py`, MicroDuck probe verification, and Wave B close-out

**Read first:** [`05_wave_b_reflex.md`](05_wave_b_reflex.md) §5–§6, [`03_eval_harness.md`](03_eval_harness.md) §11.8–§11.9.

**Deliverables:**

1. **Run probes and append scorecard rows for `reflex` and `reflex_resolve`:**
   - `PYTHONPATH=src ./venv/bin/python -m harness.probes --dataset reflex`
   - `PYTHONPATH=src ./venv/bin/python -m harness.probes --dataset reflex_resolve`
   - Verify `PYTHONPATH=src ./venv/bin/python -m harness.scorecard` displays all rows for `reflex` and `reflex_resolve`.
   - Verify `DATA_ROOT/transfer/reflex_face_probe.npz` and `DATA_ROOT/transfer/reflex_resolve_face_probe.npz` exist on disk, contain finite `mean`, `scale`, `coef`, and `intercept` arrays (`coef.shape == (1, 153)`), and reproduce the scorecard `react-face` test AUROC to 4 decimal places when applied directly to the test split's cached `reflex-face-v1` features.

2. **`tools/report_reflex.py` and `tests/test_report_reflex.py`:**
   - Implement `prediction_check_reflex(rows, dataset)` following `03_eval_harness.md` §11.8 and `05_wave_b_reflex.md` §6:
     - Clause 1, `"reaction-only beats action_best"`: `react-face`'s `ci_low > action_best` AUROC (`held` / `did not hold` / `not tested`).
     - Clause 2, `"reactions add to the action-only model"`: `fusion_minus_action_fusion`'s `ci_low > 0` (`held` / `did not hold` / `not tested`).
     - H1 pass bar: `react-face >= 0.65` and `fusion_minus_action_fusion >= +0.03` with `ci_low > 0` (`met` / `not met` / `not testable`).
     - Context line (no verdict): `fusion − action_best`, `judge`, `judge-large`, `action-probe`, `step-prior`, `action-fusion`, `react-nonverbal`, `react-spoke`, `react-full`.
   - Generate `docs/evals/<date>_reflex_h1.md` covering both `reflex` and `reflex_resolve` in one comprehensive Wave B report:
     - Executive summary, prediction checks for `reflex` and `reflex_resolve`, and the pre-registered Wave B kill-rule evaluation (`05_wave_b_reflex.md` §6);
     - Full conditions tables and one-sided leak-check tables (`leak_check_label` from `src/harness/reporting.py`; fails iff `ci_low > 0.5`);
     - Judge diagnostics and answer distributions (`qwen2.5vl:7b` and `gemma4:26b`);
     - Compute and wall-clock times computed from the caches (`index.jsonl` and judge `.jsonl`);
     - Stage A MicroDuck artifact summary (`reflex_face_probe.npz` and `reflex_resolve_face_probe.npz`, plus top-5 positive and top-5 negative standardized feature weights by name from the 153-d `reflex-face-v1` vector);
     - 10 sampled test items per dataset (5 `label = 1`, 5 `label = 0`, `default_rng(0)`) with verified visual descriptions of the frames in `video_cam1.mp4` (never template text; inspect the extracted frames first per Lesson L8/A7b), plus TSV exports `docs/evals/<date>_reflex_examples.tsv` and `docs/evals/<date>_reflex_resolve_examples.tsv`.
   - Add fast unit tests in `tests/test_report_reflex.py` testing `prediction_check_reflex` on synthetic row dicts (all clauses held/met, did not hold/not met, and missing/null rows).

3. **Close out Wave B in the docs:**
   - Update `docs/agent_execution_guide.md`:
     - Update §1.4 (latest bare battery numbers) and §1.5 (add `reflex` and `reflex_resolve` headline results with row timestamps);
     - Move `B1`–`B4` to §5.1 (`Already delivered`);
     - Set the guide header and status to `Queue Complete: Wave B delivered; awaiting maintainer review for Wave C (H2 transfer) / DW6 (Stage A MicroDuck)`.
   - Record `B4` in `docs/ongoing_general_errors.md` §3.

**Validation (must all pass before committing `B4`):**
- Confirm all leak checks pass (`ci_low <= 0.5` on every `:shuffled` row across `reflex` and `reflex_resolve`; if any has `ci_low > 0.5`, STOP and investigate).
- Open and read `docs/evals/<date>_reflex_h1.md` and both `.tsv` files, verifying that every number matches `results/scorecard.jsonl` and no template placeholder text exists.
- Run `scripts/battery.sh --slow` bare (with G2 3 times in a row).
- Commit as `feat(b4): evaluate Wave B REFLEX probes, export MicroDuck face probes, and generate Wave B report` and push to `main`.

---

## 4. Deferred: do NOT start

| Work | Trigger | Who writes the spec |
|---|---|---|
| **Wave C: H2 cross-dataset transfer** (`DW3`) | Wave B (`B1`–`B4`) complete and verified by the designer | The designer |
| **Stage A live MicroDuck demo** (`DW6`) | Wave B (`B1`–`B4`) complete, MicroDuck hardware ready, and maintainer opts in | The designer |
| **Robot transfer, revisited** (`DW7`, offline H3 v2 with REFLEX) | Wave B shows a reaction gain (`ci_low > 0`) **and** the designer writes the pre-registered spec | The designer |
| **Wave D v2** (HoloAssist → RoboReward with v1's two flaws fixed) | **Not selected** (Issue 10 = A). Do not run it | — |
| **Ego4D false-positive set** (`DW4`) | Wave C+ | The designer |

---

## 5. Do NOT change

### 5.1 Already delivered (verified by the designer)

| Item | Commit | What it delivered |
|---|---|---|
| A1–A6, A6b, A6c | `651039b`–`0f16799`, `76e71cc`, `8c6f04e` | the harness, the encoders, the judges, the memory guard |
| A7, A7b, A7c, A8 | `6d72492`, `d77584c`, `1346e0b`, `36a7769` | the Oops! and HoloAssist pipelines; `judge-large` |
| A9, A9b, A10 | `3f0b78d`, `6744719`, `492bd79` | the Wave A summary (corrected); `shared.fsutil.data_glob` |
| A11–A14 | `abbc273`, `608c729`, `eea8933`, `d6d9c54` | `step-prior`, `action-fusion`, the ablation Δ; per-session audibility; `holoassist_audible`; the Wave A completion |
| A15 | `347f242` | the computed caveat; the one-sided leak label in the reports |
| D1 | `f3f982c` | `holoassist_pool`, the reaction labeler, the pseudo-labels (the refactor is byte-identical) |
| D2 | `d18dbbf` | `transfer.prior_lr`/`heads`/`curve`, encoder `siglip-b16-224-u9`, `sources.roboreward`, `tools/download_roboreward.py`, `h3_verdict` |
| D3 | `03ea7eb` | the full, verified RoboReward download; items (0 label-rule violations), split, u9 features, `judge-large` reference |
| D4 | `315f3a6` | the Wave D v1 curve, its 44 rows, the report and the pre-registered verdict |

### 5.2 Accepted equivalents (do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure.** `extract_frame` retries a seek up to 0.3 s earlier.
- **`GeminiJudge`:** 32 output tokens, thinking off, and a 429 is not cached. `guard()` holds the heavy lock through the model load only.
- **Scorecard rows carry the parent commit's `git_sha` with `dirty: true`.**
- **The HoloAssist group partition is computed over all 1,758 sessions,** and `holoassist_audible` reuses it. Its `not_audible` count includes the 85 sessions without video.
- **`shared/memguard.py` logs `waited_ms = 0` iff admitted on the first check** (D2; previously, under 50 ms). Admission behavior is unchanged.
- **`DATA_ROOT/judge/roboreward/gemma4_26b/scores.jsonl`** is a stray duplicate of that judge cache, written in D3. No code reads it. Leave it.
- **Wave D v1's pool has 0.52% mistakes, and its true-label head is overfit.** These are part of v1's record (`04` §8), **not** bugs to patch. v1 is accepted as the result (Issue 10 = A).

### 5.3 Maintainer decisions

- **October 8–10:** outcome supervision only, with no human rating; "just ask an LLM" is the control; hidden-outcome data first; Mac Studio only; commit to `main`; Issue 5 → B (`judge-large` `gemma4:26b`).
- **October 20:** Issue 6 → A (`holoassist_audible`); Issue 7 → A (`step-prior` and ablation gain `fusion_minus_action_fusion`).
- **October 21–22:** Issue 8 → A (Wave D spec); Issue 9 → B (full RoboReward download); Issue 10 → A (Wave D v1 accepted).
- **November 1:** **Issue 1 & Issue 4 → REFLEX (`10.5281/zenodo.14590226`, CC BY 4.0).** After Option E (`M4` bounced, `M5` no reply) did not yield access, Wave B pivots to REFLEX (`05_wave_b_reflex.md`, guide items `B1`–`B4`), evaluating `react-face` (`reflex-face-v1`, `DW2`) and `react-nonverbal` (`reflex-speech-v1`) on human reactions to a physical robot and exporting face probes for the Pollen Robotics MicroDuck (`DW6`).

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **Grouped splits and CIs. Split files are written once. The scorecard is append-only.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels. The leak check is one-sided.**
- **Existing `oops`, `holoassist`, `holoassist_audible`, and `roboreward` caches, splits, and scorecard rows are never modified.**

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, H1–H3, the decision log (entries 1–28) | `00_thesis.md` |
| Wave B: the REFLEX contract (download, alignment, `reflex` & `reflex_resolve` items, `reflex-face-v1` & `reflex-speech-v1`, probes, pass/kill rules) | `05_wave_b_reflex.md` |
| The scorecard, conditions, metrics, judges, guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| Wave D: the contract (§1–§7) and the v1 result with the designer's notes (§8) | `04_wave_d_robot_check.md` |
| Datasets: REFLEX, Oops!, HoloAssist (`holoassist_audible`, `holoassist_pool`), RoboReward | `02_data_sources.md` |
| Issues 1–10 (resolved); deferred work DW1–DW7; lessons L1–L17 | `ongoing_general_errors.md` |
| Reports | `docs/evals/` |
| Operations | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard (for every item)

- **Red first, then green, then falsify.**
- **Reproduce the stated reference values.** Ensure existing datasets (`oops`, `holoassist`, `holoassist_audible`) are unaffected.
- **A flaky gate is a failing gate.** Run G2 three times with `TMPDIR` on the SSD.
- **Report every missed validation bar in the commit body** (L14).
- **Never loosen a bar or change a pre-registered rule to pass it.** File it as **Issue 11**.

---

## 8. THE LOOP

```
For each item in §2 (B1 -> B2 -> B3 -> B4):
  (1) Read the item spec in §3 and the referenced sections of 05_wave_b_reflex.md,
      02_data_sources.md, and 03_eval_harness.md.
  (2) Write the tests first (red), implement the item (green), and falsify each gate.
  (3) For any job > 30 min (B1 download, B3 judges), launch detached via
      tools/daemonize.py + tools/run_supervised.sh and confirm PPID 1.
      Check shared.memguard --status before every model job.
  (4) Run scripts/battery.sh bare (--slow when the item ran models; G2 3 times with
      TMPDIR="/Volumes/Extreme SSD/tmp").
  (5) Record the resolution in docs/ongoing_general_errors.md §3, stage explicit paths,
      commit `feat(b<n>): ...` on `main`, and push with `/usr/bin/git push origin main`.
```

---

## 9. Definition of Done for Wave B

- [x] A1–A15 and D1–D4 delivered and verified (§5.1).
- [x] Wave B REFLEX contract pre-registered (`05_wave_b_reflex.md`, November 1, 2026) and baseline battery re-verified bare (§1.4).
- [x] **B1:** REFLEX downloaded (`37,219,808,401` bytes, MD5 verified, archive deleted), extracted (55 participants × 10 files), and `reflex` + `reflex_resolve` items and splits built and validated.
- [x] **B2:** `ReflexFaceEncoder` (`reflex-face-v1`, 153-d) and `ReflexSpeechEncoder` (`reflex-speech-v1`, 96-d) implemented, wired into `features.extract` and `harness.probes`, and verified by unit tests.
- [x] **B3:** Features (`reflex-face-v1`, `reflex-speech-v1`, `siglip-b16-224-u9`) and VLM judges (`qwen2.5vl:7b` train+test, `gemma4:26b` test) run on `reflex` and `reflex_resolve` under `memguard`.
- [ ] **B4:** Scorecard rows appended for `reflex` and `reflex_resolve`, MicroDuck face probes (`{reflex,reflex_resolve}_face_probe.npz`) exported and verified, `docs/evals/<date>_reflex_h1.md` generated, and guide closed out to `Queue Complete`.
