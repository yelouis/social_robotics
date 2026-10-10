# Agent Execution Guide: Active Build: Wave A (evaluation harness + first H1 numbers, 9 items), October 8, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:** on October 8, 2026 the project was reoriented. The v0 pipeline (six hand-built affect layers over Ego4D, plus a human-rated benchmark) is archived at git tag `v0-saf-final` and removed from the tree, because it could not be validated ([`LESSONS_v0.md`](LESSONS_v0.md)).
- **What exists:** the grounding docs, `src/shared/vlm_client.py`, `src/models_config.py`, `src/config.py`, `tools/` and two tests.
- **What you build:** **Wave A**. It is the evaluation harness that tells us automatically whether human reactions carry reward information, and its first H1 numbers on two licensed datasets (Oops! and HoloAssist).

**Read before anything else, in this order:**
1. [`00_thesis.md`](00_thesis.md): the thesis, H1–H3, the pass and kill criteria, and the decision log.
2. [`03_eval_harness.md`](03_eval_harness.md): **the contract** for every path, schema, constant and prompt you implement.
3. [`02_data_sources.md`](02_data_sources.md): the item definitions for Oops! and HoloAssist.
4. [`LESSONS_v0.md`](LESSONS_v0.md) and [`ongoing_general_errors.md`](ongoing_general_errors.md) §2: the traps.

**The maintainer's words:**
- *"The fundamental core of this project is to use human emotion as a policy for robot learning in addition to the traditional RL stack for robots. We want to add this additional reward signal and investigate if this additional reward signal helped the robot learn a task better and perhaps allow robots to train on much more data like vlogs or any POV youtube video."*
- *"Ideally we should avoid human in the loop ratings because that is not scalable and we want to be able to somehow measure whether our thesis is working often."*
- *"How is this any different from asking an LLM model what is socially appropriate…?"* This is why every H1 result is measured against action-only controls.
- *"No need to open up a new branch, just push to the repo."*

**Status:** **Active Build: Wave A** (A1–A9), in the §2 order.
- One maintainer decision is pending, and it does not block you: **Issue 1** (web video) gates only Wave B, which is not in this guide.
- **Issue 3** (SSD space) was resolved on October 8: the v0 videos were deleted and 1.5 TiB is free.
- A4 may end by filing under **Issue 2**.

**Every path, number, field name and literal string in this guide and in `03_eval_harness.md` is a decision, not a suggestion. Implement as written; do not substitute your own.**

---

## 0. Standing constraints (apply to every item)

1. **The battery is the regression bar.** After A1, run `scripts/battery.sh` bare after every item and update §1.4. Read every exit code bare; never pipe a gate through something that swallows its code.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`.
   - **Python 3.9 syntax only:** no `match`; no `X | Y` unions evaluated at runtime (use `typing.Optional`/`Union`, or `from __future__ import annotations`).
   - **Before any run that loads models or touches video:**
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies are frozen.**
   - **Install no new pip packages.** Everything Wave A needs is in `venv` (§1.1).
   - **The only model downloads allowed:** `google/siglip-base-patch16-224` (Hugging Face) and `iic/emotion2vec_plus_large` (via `funasr`; it may already be cached).
   - **Pull no Ollama models.** `qwen2.5vl:7b` is installed.
4. **Data placement and disk.**
   - All data lives under `DATA_ROOT` (`src/config.py`, default `/Volumes/Extreme SSD/social_robotics`): `raw/<dataset>/`, `items/`, `features/`, `judge/`, `runs/`. **No video or audio on the internal disk, ever.**
   - **Before any download, check the disk rule:** free space *after* the download and any extraction must stay **≥ 50 GiB**. If it would not, STOP and file it as a new issue with the measurement.
5. **Authorized downloads, and nothing else:**
   - the Oops! videos + annotations bundle (45 GB) from `https://oops.cs.columbia.edu/data`;
   - HoloAssist **labels** (111 MB) from the official data links (`https://holoassist.github.io/`);
   - HoloAssist **pitch-shifted videos** (184.20 GB), **only** after A4 resolves Issue 2 as "independent" (and the disk rule holds).
   - **No YouTube or other web-video downloading in Wave A.**
   - **Never circumvent a login wall, CAPTCHA or bot check** (no cookies, tokens, proxies or client spoofing).
6. **Never delete anything you did not create.** The remaining v0 artifacts, caches and environments stay. You may delete a downloaded archive *after* you have verified its extraction.
7. **Long runs (> 30 min) run detached:** `./venv/bin/python tools/daemonize.py <log> bash tools/run_supervised.sh <progress.json> <runner…>`, then confirm `PPID 1` (`tools/README.md`). Never use your own background-task feature: it is reaped after ~1–2 h.
   - **Every long runner is resumable.** It skips finished `item_id`s.
   - **It maintains `<out_dir>/progress.json`**: a JSON **list** of finished `item_id`s, rewritten atomically (temp file + `os.replace`). That is what the supervisor counts.
8. **Determinism.** Seeds: `numpy.random.default_rng(0)` and `torch.manual_seed(0)`. Iterate inputs in sorted `item_id` order.
9. **No outcome leakage.**
   - The judge never receives audio, the reaction window outside the action window, labels, Oops! descriptions, or HoloAssist mistake/purpose labels.
   - Probes never see test labels at fit time.
   - **Nothing is tuned on the test split.** The hyperparameters in `03_eval_harness.md` §6 are fixed.
10. **The scorecard is append-only.** Never edit or delete a row in `results/scorecard.jsonl`. A wrong row is superseded by a new row whose `notes` say why.
11. **Honest nulls (L3).** A failed feature, judge answer or condition is `null` plus a reason, counted in `n_excluded` or in a `not run: <reason>` row. Never zeros.
12. **Every stage logs its counts (L2).** Every CLI ends by printing one line: `items_in=<a> items_out=<b> excluded=<c> elapsed_s=<t>`. Exclusions are broken down by reason in the run's `errors.jsonl` or stats JSON.
13. **Commits.**
    - One item = one Conventional Commit on `main`, scope = item id: `feat(a2): …`, `fix(a1): …`.
    - The body states the WHY, the red run and the green run.
    - Push after every item with **`/usr/bin/git push origin main`**. (`~/.local/bin/git` shadows the system git and lacks the https helper.)
    - **No branches, no PRs. Never amend a pushed commit.**
14. **Record the resolution in the same commit:** one line under **"Wave A"** in `ongoing_general_errors.md` §3: `A<n> — <title> — git log --grep "(a<n>)" — <measured result>`.
15. **When this guide and a contract doc disagree, STOP and file it** in `ongoing_general_errors.md` as a new issue (the next number is **Issue 5**), with options.
16. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer.

---

## 1. Verified baseline (October 8, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1 (`/opt/homebrew/bin/ffmpeg`).
- **Ollama:** `qwen2.5vl:7b` (also as `:latest`), `gemma4:26b`, `gemma4:latest`, `glm4:latest`, `moondream:latest`, `tinyllama:latest`.
- **`venv`, Python 3.9.6:**
  - ML: torch 2.8.0 · torchaudio 2.8.0 · transformers 4.46.3 · funasr 1.3.1 · scikit-learn 1.6.1 · scipy 1.13.1;
  - data: numpy 2.0.2 · pandas 2.3.3 · pyarrow 21.0.0;
  - clients: google-genai 1.47.0 · httpx 0.28.1 · ollama 0.6.1 · huggingface_hub 0.36.2;
  - tooling: ruff 0.15.11 · pytest 8.4.2;
  - other: opencv-python 4.10.0.84 · librosa 0.11.0 · mediapipe 0.10.35 · hsemotion-onnx 0.3.1.
- **Not installed and not needed in Wave A:** yt-dlp, mlx-whisper.
- **`.env`** defines `GOOGLE_API_KEY` and `HF_TOKEN` (names checked, values not read), plus unused v0 keys.

### 1.2 Storage (Extreme SSD)

- 1.8 TiB total, **1.5 TiB free** (after the October 8 cleanup).
- **Deleted on October 8 (Issue 3):** the v0 videos: 1,083 Ego4D clips (1.2 TB) and the Charades-Ego videos and archives. The manifest, with every path and all Ego4D clip ids for re-download, is `DATA_ROOT/DELETED_2026-10-08.json`.
- **Still in place; do not touch:** `huggingface_cache/` 106 GB · `ego4d_data/` 82 GB (annotations + precomputed features, no videos) · `saf_env/` 19 GB · `social_robotics/full_run_2026_06_18/` 13 GB · `social_robotics/bench_v0/` 4.5 GB.

### 1.3 Repository

- `main` contains the reorientation (`454b40d`) and this spec.
- v0 is at tag `v0-saf-final` (`886bd71`).
- `src/` holds `config.py`, `models_config.py` (one key: `vlm_judge` → `qwen2.5vl:7b` on this host) and `shared/vlm_client.py`.

### 1.4 Gates (run bare October 8, 2026; the regression bar)

| # | Gate | Command | Result |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · clean |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **11 passed** |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · **4 passed** |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | **no slow tests yet** (A5, A6) |

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| A1 | Battery green and scripted | Every later item is validated against it |
| A2 | Metrics, scorecard, self-test | The ruler. Everything downstream writes scorecard rows through it |
| A3 | Items and grouped splits | The schema every adapter writes and every metric resamples by |
| A4 | HoloAssist labels + independence check | Cheap (111 MB, no video). **Its verdict gates A8's 184 GB download,** so start that clock early |
| A5 | Encoders + feature cache | Needs A3's items. Feeds the probes in A7/A8 |
| A6 | VLM judge (local + frontier) | Needs A3's items. Feeds the `judge` and `fusion` conditions |
| A7 | Oops!: end-to-end H1 | The smaller dataset (45 GB). It proves the whole pipeline before the big download, and is the visible-outcome contrast |
| A8 | HoloAssist: end-to-end H1 | Needs A4 = independent and A7's proven pipeline |
| A9 | Re-measure; close out Wave A | Measures the finished system and writes the summary the designer uses to spec Wave B |

---

## 3. The items

### A1: Battery green and scripted

**What this means for the maintainer:** one command answers "is the code healthy?", so a regression is visible the moment it lands.

**The gap:** G1 exits 1 with the 3 errors listed in §1.4. There is no battery script, no pytest config, and no `slow` marker.

**Implementation:**
1. Fix the 3 lint errors **without behavior change**:
   - delete the unused `auto = _auto_tier()` in `_format_banner` (`src/models_config.py:179`);
   - split the semicolon statement (`tests/test_vlm_timeout.py:25`) onto two lines;
   - split `import os, sys` (`tools/daemonize.py:6`) into two imports.
2. Create `pytest.ini`:
   ```ini
   [pytest]
   testpaths = tests
   markers =
       slow: loads models, needs network, or needs data on the SSD
   ```
3. Create `scripts/battery.sh` (bash, `set -u`, **not** `set -e`), executable. It runs, in order, each gate with its exit code captured bare, and prints exactly `G<n> <name>: exit <code>`:
   - **G1:** `./venv/bin/ruff check src tests tools`
   - **G2:** `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/`
   - **G3:** `PYTHONPATH=src SR_NO_MODEL_BANNER=1 ./venv/bin/python -m harness.scorecard --selftest`. Until A2 lands, this prints `G3 selftest: skipped (harness not built)` and contributes 0.
   - **G4**, only when called with `--slow`: `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m slow tests/`

   The script exits with the **maximum** of the gate codes.
4. In `README.md` → Setup, replace the Tests line with `- **Battery:** \`scripts/battery.sh\` (add \`--slow\` for model/data tests)`.

**Validation:**
- `scripts/battery.sh` → `G1 lint: exit 0`, `G2 tests: exit 0` (2 passed), `G3 selftest: skipped…`; overall exit 0.
- **Falsifying check:** temporarily add `import json` (unused) at the top of `src/config.py` → G1 exits 1 **and** `battery.sh` exits 1 → revert → both 0. Record both runs in the commit body.

**Blast radius:** `README.md` (Setup), §1.4 of this guide.

---

### A2: Metrics, scorecard and self-test

**What this means for the maintainer:** this is the ruler. If it is wrong, every later number is wrong. The self-test proves it can tell signal from noise *and* that its confidence intervals respect groups.

**The gap:** nothing exists. The contract is `03_eval_harness.md` §4 (row schema) and §7 (metrics).

**Implementation:**
1. `src/harness/__init__.py` (empty) and `src/harness/metrics.py`:
   - `auroc_ci(y, score, groups, n_boot=1000, seed=0) -> Tuple[float, float, float]`: point AUROC (`sklearn.metrics.roc_auc_score`) plus a group-bootstrap percentile CI, exactly per §7: resample `n_groups` groups with replacement, take all their items, redraw single-class resamples, and raise `RuntimeError("bootstrap: >10000 draws needed for 1000 two-class resamples")` past 10,000 draws.
   - `delta_auroc_ci(y, score_a, score_b, groups, n_boot=1000, seed=0)`: `AUROC(b) − AUROC(a)`, with **paired** resamples (the same groups for both scores in each draw).
   - `spearman_ci(x, y, groups, n_boot=1000, seed=0)`: `scipy.stats.spearmanr` with the same group bootstrap.
   - Items whose score is `None`/NaN are dropped before computing, and their count is returned to the caller (the row's `n_excluded`).
2. `src/harness/scorecard.py`:
   - `ScorecardRow`: a dataclass with **exactly** the §4 fields, in that order.
   - `make_row(...)` fills `ts` (UTC ISO-8601, `Z`), `git_sha` (`/usr/bin/git rev-parse --short HEAD`) and `dirty` (`/usr/bin/git status --porcelain` non-empty).
   - `append_rows(rows, path="results/scorecard.jsonl")`: append-only, one JSON object per line, `f.flush()` + `os.fsync()`. It never opens the file in a mode that truncates.
   - `config_hash(config: dict) -> str`: SHA-256 of `json.dumps(config, sort_keys=True, separators=(",", ":"))`, first 12 hex.
   - **CLI** (`python -m harness.scorecard`):
     - no arguments → a table of the latest row per `(hypothesis, dataset, split, condition, metric)`, newest first;
     - `--history <dataset>` → every row for that dataset in time order;
     - `--selftest` → step 3.
3. **`--selftest`** builds synthetic data with `default_rng(0)` and writes rows only to a temp file, **never** to `results/scorecard.jsonl`. It prints `PASS`/`FAIL` per check, and exits 0 only if all pass:
   - **(a) Planted signal:** 60 groups × 20 items, labels Bernoulli(0.5), `score = label + 0.8·N(0,1)` → AUROC ≥ 0.75 **and** `ci_low > 0.5`.
   - **(b) Null:** same groups, `score ~ N(0,1)` independent of the label → `ci_low ≤ 0.5 ≤ ci_high`.
   - **(c) Grouping is real:**
     - Data: 40 groups × 25 items; a label constant within each group (Bernoulli(0.5) per group); `score = group_offset + 0.1·N(0,1)`, where `group_offset = label + N(0,1)` per group.
     - Requirement: the grouped CI width must be **≥ 1.5×** the width from an *item*-level bootstrap of the same data. Implement the item bootstrap as a private helper used only here.
   - **(d) Paired Δ:**
     - `score_b = score_a + 1.5·label` → Δ `ci_low > 0`;
     - `score_b = score_a` → Δ == 0 exactly, and the CI is (0, 0).
4. `tests/test_harness_metrics.py`:
   - AUROC equals `roc_auc_score` on 3 fixed arrays;
   - each self-test check as a unit test;
   - NaN scores are excluded and counted;
   - `append_rows` appends (write 2 rows, then 1, read back 3);
   - `config_hash` is order-independent across dict key orders.

**Validation:**
- G3 → `exit 0` with 4 PASS lines.
- **Falsifying check:** temporarily change the bootstrap to resample *items* instead of groups → **(c) must FAIL** and G3 exit 1 → revert. This is the assertion that proves the CIs are grouped.
- Also run with `score_b = score_a` and confirm the Δ row is exactly 0.

**Blast radius:** `scripts/battery.sh` (G3 becomes live), this guide's §1.4.

---

### A3: Items and grouped splits

**What this means for the maintainer:** a person who appears in both training and test makes the numbers look better than they are. This item makes that impossible by construction.

**The gap:** nothing exists. Contract: `03_eval_harness.md` §3 (items) and §5 (splits).

**Implementation:**
1. `src/harness/items.py`:
   - an `Item` dataclass with exactly the §3 fields;
   - `read_items(path)`;
   - `write_items(items, path)`: atomic, temp file + `os.replace`, items sorted by `item_id`;
   - `validate_items(items, check_paths=True) -> List[str]`: returns error strings, empty when valid. The rejections are exactly those in §3, with messages **verbatim**:
     - `duplicate item_id: <id>`
     - `bad window <field> [<s>, <e>] in <id>`
     - `label must be 0 or 1 in <id>`
     - `empty group_id in <id>`
     - `missing video_path for <id>: <path>`
2. `src/harness/splits.py`:
   - **`make_group_split(items, dataset, official=None, force=False, path_dir="splits")`:**
     - If `official` is given (a mapping `item_id → "train"|"test"`), first verify that no `group_id` maps to both. If one does, raise `ValueError("official split is not group-disjoint: <n> groups straddle")` verbatim. Otherwise use it.
     - Else: a grouped 70/30 split. Take the sorted unique `group_id`s, shuffle them with `default_rng(0)`, and assign the first `round(0.7·n)` to train.
     - Write `splits/<dataset>.json` with keys `dataset, seed, source, train, test, sha256`. `sha256` is over `json.dumps({"train": sorted(train), "test": sorted(test)}, sort_keys=True)`.
     - If the file exists and `force` is false, raise `FileExistsError("split exists: splits/<dataset>.json (use force=True and say why in the commit)")` verbatim.
   - **`load_split(dataset)`:** re-computes `sha256` and raises if it differs from the stored one.
3. `tests/test_harness_items.py`:
   - each rejection, with its exact message;
   - group-disjointness on 500 random items in 37 groups;
   - two runs produce byte-identical split files;
   - the overwrite refusal;
   - the official-split straddle error;
   - tamper detection (edit the file → `load_split` raises).

**Validation:**
- G2 green.
- **Falsifying check:** temporarily assign train/test per *item* instead of per group → the disjointness test must fail → revert.

**Blast radius:** none outside `src/harness/` and `tests/`.

---

### A4: HoloAssist labels and the independence check (no video)

**What this means for the maintainer:** if HoloAssist's mistake labels were written *from* the instructor's corrections, then "the instructor's reaction predicts mistakes" would be circular, and 184 GB of downloading would buy a meaningless number. This item finds out for 111 MB.

**The gap:** we do not know HoloAssist's exact annotation schema, how mistakes were labeled, or whether its official splits are participant-disjoint (`02_data_sources.md`, HoloAssist "Unknown until downloaded"). Issue 2 is open.

**Implementation:**
1. Check the disk rule (§0.4). Download **only** the labels archive (111 MB) from the official data links page (`https://holoassist.github.io/`) into `DATA_ROOT/raw/holoassist/labels/`. Write `DATA_ROOT/raw/holoassist/DOWNLOAD.json` with `{url, bytes, sha256, downloaded_at}` for each file.
2. Read the annotation sections of the paper (arXiv 2309.17024) and the dataset README. In `02_data_sources.md` → HoloAssist, add a subsection **"Schema (as downloaded, <date>)"** listing:
   - the exact file names;
   - the JSON fields for fine-grained actions (start, end, verb, noun, the mistake/correct attribute, and its exact field name and values);
   - the utterance fields (start, end, speaker role, purpose label, and whether a transcript is present);
   - task names;
   - performer and instructor identifiers;
   - the official split files.
3. `src/sources/holoassist.py --stats` prints, and you record in that subsection:
   - the number of sessions, fine-grained actions with a mistake/correct attribute, and mistakes (count and %);
   - the number of instructor utterances;
   - **performer ids present in more than one official split** (0 means participant-disjoint);
   - **the share of mistake actions vs. correct actions with an instructor utterance overlapping `[start, end + 5.0]`** (the label-level `react-spoke` signal; no video needed).
4. **Decide independence from the protocol text** and quote it verbatim in the subsection:
   - **(a)** The protocol says mistakes were annotated from the performer's video and actions, not from instructor speech → mark **Issue 2 "Resolved: independent"** with the quote and numbers, and add a line to §3 of the tracking doc.
   - **(b)** The protocol says mistakes were derived from, or marked using, instructor interventions → **STOP.** File under Issue 2 with these options and a `Your selection: _____` line, and **do not start A8**:
     - A: HoloAssist as an H2 target only, with the caveat stated;
     - B: drop HoloAssist;
     - C: restrict to mistakes with no instructor utterance in the window (no-reaction items).
   - **(c)** The protocol is silent or ambiguous → treat it as **(b)**. Do not decide it yourself.

**Validation:**
- Re-hash the downloaded files: they match `DOWNLOAD.json`.
- `--stats` run twice gives identical output.
- The two `react-spoke` rates are reported with their denominators.
- **The falsifying check for your reading:** the quote you cite must contain the words describing *how* mistakes were labeled. A quote that only says that mistakes *exist* does not resolve the issue.

**Blast radius:** `02_data_sources.md` (the new schema subsection); `ongoing_general_errors.md` (Issue 2 and §3).

---

### A5: Encoders and the feature cache

**What this means for the maintainer:** these turn a few seconds of video or audio into numbers a probe can learn from. They are computed once and reused by every later experiment.

**The gap:** nothing exists. Contract: `03_eval_harness.md` §9 (encoders, cache).

**Implementation:**
1. `src/features/cache.py`: `FeatureCache(dataset, encoder_id)` with `path(item_id)`, `has(item_id)`, `save(item_id, array, window, elapsed_ms)` and `load(item_id)`.
   - Files go to `DATA_ROOT/features/<dataset>/<encoder_id>/`. The sanitized `item_id` replaces every character outside `[A-Za-z0-9._-]` with `_`.
   - `index.jsonl` is append-only. `save` writes the `.npy` atomically, temp + `os.replace`.
2. `src/features/visual.py`: `FrameEncoder` (`encoder_id = "siglip-b16-224"`), exactly per §9.
   - **Frames:** at `t = s, s+1, s+2, …` strictly below `e`, plus `e` itself (so at least 2 frames). Each is extracted with `ffmpeg -ss <t> -i <video> -frames:v 1` to an in-memory PNG.
   - **Model:** `SiglipModel` + `AutoProcessor`, `get_image_features`; each frame L2-normalized, then the mean → `(768,) float32`. Device: `mps` if available, else `cpu`.
3. `src/features/audio.py`: `NonverbalAudioEncoder` (`encoder_id = "e2v-plus-large"`), exactly per §9.
   - **Audio:** `ffmpeg -ss <s> -to <e> -i <video> -vn -ac 1 -ar 16000 -f wav` to a temp file under `TMPDIR`.
   - **Model:** `funasr.AutoModel(model="iic/emotion2vec_plus_large")`, `generate(..., granularity="utterance", extract_embedding=True)` → `(1024,) float32`.
   - **Never read or store its emotion-category outputs.**
4. `src/features/extract.py`: CLI `python -m features.extract --dataset <d> --encoder <id> [--split train|test|all] [--force] [--limit N]`.
   - Reads `DATA_ROOT/items/<d>/items.jsonl` and the split file.
   - Skips cached items, appends failures to `errors.jsonl` (`item_id`, error, traceback), and **never writes a feature for a failed item**.
   - Maintains `progress.json` (§0.7) and ends with the §0.12 count line.
5. `tests/test_features.py`:
   - **Fast:** cache round-trip; sanitization; a failed extraction writes no `.npy`; resumability (an injected exception on the 3rd of 6 items → the rerun extracts only the remaining items).
   - **`@pytest.mark.slow`:**
     - shapes and dtypes are exactly `(768,) float32` and `(1024,) float32`, all finite;
     - **determinism:** the same item twice → max abs diff ≤ 1e-5;
     - **the encoders carry information (falsifying):**
       - Make 3 s of silence and two 3 s speech clips of the same sentence in different voices, offline: `say -v Samantha -o a.aiff "<sentence>"` and `say -v Daniel -o b.aiff …`, converted with ffmpeg.
       - Assert `cos(speechA, speechB) > cos(speechA, silence)` for the audio encoder.
       - Assert `cos(frames of a real video window, frames of a black-video window) < cos(same real window, itself shifted by +0.5 s)` for the visual encoder. Build the test videos with ffmpeg `testsrc`/`color=black`.

**Validation:**
- G2 green; G4 (`battery.sh --slow`) green.
- **Falsifying check:** temporarily make the audio encoder return `np.zeros(1024)` → the information test must FAIL → revert.

**Blast radius:** none.

---

### A6: The VLM judge (local + frontier)

**What this means for the maintainer:** this is the "just ask an LLM" control the maintainer asked for. If reactions cannot beat it, the thesis does not hold. It has to be implemented exactly as specified, or the comparison is not fair.

**The gap:** nothing exists. Contract: `03_eval_harness.md` §8 (frames, prompt **verbatim**, questions **verbatim**, parse rule, retries, cache, the frontier anchor).

**Implementation:**
1. `src/judge/vlm_judge.py`:
   - `build_prompt(context_text, question, n=4) -> str` renders the §8 template character-for-character.
   - `prompt_hash(question)` per §8.
   - `parse_prob(text) -> Optional[float]` uses the §8 regex.
   - `sample_frames(video, window) -> List[Path]`: the 4 frames per §8, longer side 768 px, JPEG quality 90, under `TMPDIR`.
2. **`OllamaJudge`** calls `shared.vlm_client.ollama_chat(model=get_model("vlm_judge"), prompt=…, image_paths=…, options={"temperature": 0, "num_ctx": 8192, "num_predict": 16}, timeout=180)`. Retries 2 and 3 use `temperature` 0.3. Calls are serial (concurrency 1).
3. **`GeminiJudge`** uses `google.genai.Client()` (reads `GOOGLE_API_KEY`), model `"gemini-3.6-flash"`, with the same 4 JPEGs as inline image parts plus the same prompt text, and `temperature=0, max_output_tokens=16`. On HTTP 429 or 5xx it sleeps 30 s and retries, up to 5 tries; then `null`. **Test split only, capped at 2,000 items** (`default_rng(0)` sample, sorted).
4. **The cache,** exactly per §8, is checked before any call. CLI: `python -m judge.vlm_judge --dataset <d> --split <train|test> --backend ollama|gemini [--limit N]`. It maintains `progress.json` and ends with the §0.12 count line, plus `parse_failures=<n> api_errors=<m>`.
5. `tests/test_judge.py`:
   - **Fast:**
     - `parse_prob`: `"73"`→0.73, `"I think 85."`→0.85, `"100"`→1.0, `"0"`→0.0, `"probability: 7%"`→0.07, `"none"`→None, `"250"`→None;
     - the rendered prompt for an Oops! item equals a literal expected string (copied from §8 with `context_text = "A short clip from a home video."`);
     - `prompt_hash` changes when the question changes;
     - with `ollama_chat` monkeypatched, the payload contains **only** the prompt string and 4 image paths, with no audio file and no label;
     - a second call on a cached item makes **0** requests;
     - 3 unparsable replies → `null` and attempts = 3.
   - **`@pytest.mark.slow`:** the live local judge on 4 synthetic `ffmpeg testsrc` items returns non-null probabilities. That is a liveness check, not an accuracy bar.

**Validation:**
- G2 and G4 green.
- **Falsifying check:** temporarily append the item's `label` to `context_text` inside the judge → the payload test must FAIL → revert.

**Blast radius:** `src/models_config.py` only if the `vlm_judge` key is missing (it is not; do not change its values).

---

### A7: Oops!, end-to-end H1 (the visible-outcome contrast)

**What this means for the maintainer:** the first real number. Failures in fail videos are *visible*, so we predict the judge does well and reactions add little. If the pipeline cannot reproduce that unglamorous prediction, nothing it says about hidden outcomes can be trusted.

**The gap:** no data, no adapter, no probe code. Contracts:
- `02_data_sources.md` → Oops! (item definition);
- `03_eval_harness.md` §5–§7 (splits, conditions, metrics).

**Implementation:**
1. **Download.**
   - Check the disk rule (bundle + extraction; ≈ 90 GB transient). Download the videos + annotations bundle from `https://oops.cs.columbia.edu/data` into `DATA_ROOT/raw/oops/`, and write `DOWNLOAD.json` (`url, bytes, sha256, downloaded_at`).
   - Extract. Verify that the number of video files matches the clip count implied by the annotation files. **Only then** delete the archive, and note it in `DOWNLOAD.json`.
   - Read the CC BY-NC-SA 4.0 terms on the page. Commit no Oops! pixels or frames to git, ever.
2. **Schema.** Add **"Schema (as downloaded, <date>)"** under Oops! in `02_data_sources.md`: the annotation file names, the field holding the per-worker failure onsets, the train/val membership, and whether a source-compilation id can be derived from filenames (and how).
3. **The adapter.** `src/sources/oops.py` builds items exactly per `02_data_sources.md` → Oops! (median onset `t`; the skip rules; `pre`/`post` windows; labels pre = 1, post = 0; the constant `context_text`; `group_id`; caps). It writes `DATA_ROOT/items/oops/items.jsonl` and `DATA_ROOT/items/oops/stats.json`: clips seen, kept, and skipped **per reason**, plus items per split.
4. **The split.** `make_group_split(items, "oops", official=<train→train, val→test>)` → `splits/oops.json`.
5. **Features:** `siglip-b16-224` and `e2v-plus-large` over both splits (detached run, §0.7).
6. **The judge:** `ollama` on **both** splits (fusion needs train-split judge scores); `gemini` on test (≤ 2,000).
7. **The probes.** `src/harness/probes.py` → `python -m harness.probes --dataset oops`:
   - fits and scores every condition in `03_eval_harness.md` §6 with the fixed hyperparameters;
   - appends rows for `judge`, `judge-frontier`, `action-probe`, `react-nonverbal` and `fusion`, plus a `not run` row for `react-spoke` and for `react-full` (`notes: "not run: no annotations of reactor speech in Oops!"`);
   - appends the `fusion_minus_action_best` Δ row.
   - **It also runs the shuffled-label control:** train labels permuted with `default_rng(0)`, `action-probe`, `react-nonverbal` and `fusion` refit, and rows written with condition names suffixed `:shuffled` and `notes: "shuffled-label control"`.
8. **The report,** `docs/evals/<YYYY-MM-DD>_oops_h1.md`:
   - the conditions table (value, CI, n, `n_excluded`) and the Δ row;
   - the shuffled controls;
   - judge parse failures and API errors;
   - wall-clock time per stage;
   - **the prediction check**, stated plainly whichever way it came out: *predicted: judge strong, Δ ≈ 0*;
   - 10 example items (5 pre, 5 post, `default_rng(0)`), **described in words**, with their judge and `react-nonverbal` probabilities. No images.

**Validation:**
- `validate_items` returns no errors. The skip counts in `stats.json` sum to the clips seen.
- **Leakage check:** every item's `context_text` equals the constant string.
- **The falsifying control:** every `:shuffled` row's CI contains 0.5. If any `:shuffled` row's `ci_low` is > 0.5, something leaks. STOP and find it before anything else.
- Every §6 condition has a row (real or `not run`), and the Δ row exists.
- G1–G3 are green.

**Blast radius:** `02_data_sources.md` (schema subsection); `results/scorecard.jsonl`; `splits/oops.json`; `docs/evals/`.

---

### A8: HoloAssist, end-to-end H1 (first-person, hidden outcome)

**What this means for the maintainer:** the first test where the outcome can be partly hidden from the camera, and where the reactor (the instructor) is watching the actor's first-person view. That is the closest Wave A gets to "a person watching a robot work".

**Start condition:** both of:
- Issue 2 is resolved "independent" (A4);
- A7 is closed.

The disk rule (§0.4) still applies to the 184.20 GB download.

If any is missing, skip to A9 and record A8 as "blocked on <issue>".

**The gap:** no videos, no adapter. Contract: `02_data_sources.md` → HoloAssist (item definition, sampling) and A4's schema subsection.

**Implementation:**
1. **Download** the pitch-shifted videos (184.20 GB) from the official links into `DATA_ROOT/raw/holoassist/videos/`, with `DOWNLOAD.json` entries. Extract and verify as in A7.1.
2. **The instructor-audio presence check** (automatic; no listening):
   - For 20 sessions (`default_rng(0)`), compute the mean RMS (dBFS) of the audio inside instructor-utterance spans, and inside spans of ≥ 2 s with **no** annotated utterance from anyone.
   - **Requirement:** the median per-session difference must be **≥ +3.5 dB**.
   - If it fails, STOP and file under Issue 2 (*instructor not audible in the video audio*), with these options and a `Your selection: _____` line:
     - A: `react-spoke`/`react-full` from annotations only;
     - B: drop HoloAssist's audio channel.
3. **The adapter.** `src/sources/holoassist.py` builds items exactly per `02_data_sources.md` → HoloAssist: one per fine-grained action with an attribute; the windows; `context_text` from task + verb + noun only; `group_id`; sampling caps. It also records `meta.spoke` (0/1) and `meta.transcript` (the overlapping instructor text, or null).
4. **The split.** Use the official split if A4 found it participant-disjoint (0 straddling performers); otherwise a grouped 70/30 split.
5. **Features, judge and probes** exactly as A7.5–A7.7, plus:
   - `react-spoke`: the `meta.spoke` value as the score;
   - `react-full`: TF-IDF + logistic regression on `meta.transcript` (an empty string when no speech), **only if** the annotations carry transcripts; otherwise a `not run` row.
   - The shuffled-label controls as in A7.
6. **The report,** `docs/evals/<YYYY-MM-DD>_holoassist_h1.md`: the same sections as A7's, plus:
   - **the pitch-shift caveat**, stated at the top;
   - the `react-spoke` vs. `react-nonverbal` comparison, answering *"is the voice signal more than 'the instructor said something'?"* in one plain paragraph;
   - the audio-presence measurement from step 2.

**Validation:**
- **Leakage check (falsifying):** no `context_text` contains `mistake`, `correct`, `wrong`, `error`, `fix` or `instead` (case-insensitive). Assert it in a test over the real `items.jsonl`, then demonstrate it bites by injecting `"mistake"` into one fixture item.
- The shuffled-control CIs contain 0.5.
- Group disjointness holds in `splits/holoassist.json`.
- Every §6 condition has a row; the Δ row exists. G1–G3 are green.

**Blast radius:** `results/scorecard.jsonl`; `splits/holoassist.json`; `docs/evals/`; `ongoing_general_errors.md` (Issue 2, if filed).

---

### A9: Re-measure; close out Wave A

**What this means for the maintainer:** one page that says whether Wave A found reaction signal beyond the action-only controls, and what the designer needs to know to write Wave B.

**Implementation:**
1. Run `scripts/battery.sh --slow` bare and update §1.4 with the real numbers.
2. Write `docs/evals/<YYYY-MM-DD>_wave_a_summary.md`:
   - the headline table (dataset × condition, with CIs and n), the Δ rows, and the shuffled controls;
   - which predictions held and which did not, stated plainly;
   - **anything suspicious** (an excluded share > 5%, a judge parse-failure rate > 2%, a condition that could not run);
   - compute times per stage;
   - **observations for Wave B:** encoder failures, the judge's behavior, and the observed speed of data handling;
   - A8's status if it was blocked.
3. Add a line per item to `ongoing_general_errors.md` §3 (if not already done), and add Wave A to §5.1 below.
4. **Rewrite this guide's title and status** to **"Queue Complete: waiting on Issue 1 and the Wave B spec"**, or **"Queue Complete: waiting on Issue <n>"** if A8 is blocked. **Then stop. Do not invent work.**

**Validation:** every number in the summary matches a row in `results/scorecard.jsonl` (cite its `ts`); G1–G4 green.

---

## 4. Deferred: do NOT start

- **Wave B (hidden-outcome verdict data), DW1.** Needs Issue 1's selection **and** a Wave B spec from the designer. Do not write it yourself.
- **The face encoder** (`react-face`), DW2: part of Wave B.
- **H2 transfer** (HoloAssist / AM-FED+, BAD only if granted), DW3: needs Wave B.
- **The Ego4D false-positive set,** DW4.
- **Wave D, the robot check (offline H3; RoboReward)**, DW5: the designer writes its spec after Wave A's results. Do not start it, and do not download RoboReward in Wave A.
- **Stage A live** (the microduck), DW6.
- **Any web or YouTube video acquisition.**
- **Re-downloading any Ego4D or Charades-Ego video.**
- **Anything from tag `v0-saf-final`** (layers, bench, visualizer): read it for reference, never restore it.

---

## 5. Do NOT change

### 5.1 Already delivered

- **R0, the reorientation** (`454b40d`): v0 removed from the tree, the grounding docs, and the kept utilities (`vlm_client.py`, `models_config.py`, `config.py`, `tools/`).
- **R1, this spec:** the guide, `03_eval_harness.md` (the contract), `02_data_sources.md` (item definitions), the tracking doc and `AGENTS.md`.
- `shared/vlm_client.ollama_chat`'s enforced timeout is load-bearing (`LESSONS_v0.md`, "Operations"). Use it; do not replace it with the `ollama` Python client.

### 5.2 Accepted equivalents (checked; do not "fix" these back)

- None yet. The designer adds entries here after verifying Wave A.

### 5.3 Maintainer decisions

**October 8, 2026:**
- **The thesis and its framing:** human reactions as an additional reward signal for robot learning, learnable from web-scale video. **No human-in-the-loop rating.** Measure whether the thesis works, often and automatically.
- **SAF v0 retired.** Tag, then remove from main.
- **Both deployment stages,** in order: web reactions as free labels (B) first, then live reactions (A).
- **Any footage for training;** results reported on both robot-reaction (BAD) and first-person (HoloAssist) data.
- **"Just ask an LLM" is the control to beat** (the maintainer's challenge).
- **Hidden-outcome data first.**
- **Three-month goal: a paper-grade H1 + H2. Compute: the Mac Studio only.**
- **The designer does not code; an implementing agent builds from this guide.**
- **Commit straight to `main`, with no branches.**
- **Issue 3 → delete the v0 videos:** *"clean up any videos you want from the v0 leftovers. Feel free to delete Ego4D if you think that is the right choice."* The designer deleted the Ego4D raw videos and the Charades-Ego videos (manifest in §1.2).

### 5.4 Invariants and intentional design decisions

- **Supervise on outcomes, never on emotion.** No emotion categories as features or targets, from any model (including HSEmotion classes and emotion2vec categories). Embeddings are allowed.
- **`label = 1` always means a good outcome.**
- **The judge never hears audio** and never sees anything outside the action window.
- **Two action-only controls:** the Δ is taken against `action_best`.
- **Grouped splits and grouped CIs only.** Split files are written once.
- **The scorecard is append-only.**
- **H2 targets are never training data.**
- **Fixed probe hyperparameters;** nothing is tuned on test.
- **Nulls with reasons, never zeros.**
- **Pixels from licensed datasets are never committed to git.**

### 5.5 Assessed and rejected: do NOT re-propose

- **Human rating rounds, golden labels, rater UIs, pre-seeded review.** The v0 benchmark failed this way (0/349 rated).
- **Hand-built affect channels** (gaze scores, proxemics, nod/flinch detectors, categorical emotion) as the representation.
- **Self-recorded data collection or any human-subjects study** (taste tests with friends; people watching robot clips on a webcam). The maintainer declined both (October 9, 2026).
- **Partnering with an academic institution.** The maintainer declined it (October 9, 2026).
- **Ego4D bystander footage as H1 data.** It has no outcome labels and rarely contains an evaluative reaction.
- **A first-person-only data restriction.** It was considered and rejected (`00_thesis.md`, "Why first-person footage isn't required").
- **Using Oops! descriptions or HoloAssist mistake/purpose labels as model inputs.**
- **Tuning prompts, hyperparameters, windows or caps on test results.** If a number looks wrong, file it.
- **Circumventing YouTube or any site's bot checks.**

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, H1–H3, pass/kill, the decision log | `00_thesis.md` |
| Items, splits, the scorecard schema, conditions, metrics, the judge prompt, encoders | `03_eval_harness.md` §3–§9 |
| Oops! and HoloAssist item definitions, facts, leakage rules | `02_data_sources.md` |
| Prior work and baselines | `01_prior_work.md` |
| Issues 1–3, maintainer actions, deferred work, lessons, the resolved index | `ongoing_general_errors.md` |
| Operations (detached runs, ollama, decoding, storage) | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard

- **Red first.** Before building an item, run its falsifying check against the current code (or against an absent module) and record the failure.
- **Every gate must be able to fail.** Each item names the change that turns its key test red. Show it red, then green.
- **Falsify the pipeline, not just the code:** the shuffled-label controls must sit at chance.
- **Report denominators** (`n_items`, `n_groups`, `n_excluded`) with every number.
- **Never loosen a bar or a threshold to pass it.** File it with the measurement and options.
- **Read your own outputs.** Open `stats.json`, the reports and a sample of judge `raw` answers, and describe what you saw in the commit body.

---

## 8. THE LOOP

```
(1) Is there an approved item? A1–A9, in §2 order. If all are done or
    blocked, STOP. Never start §4 work. Never fill in a `Your selection:`
    line.
(2) Read the item and EVERY contract section it names. Copy paths,
    constants, prompts and error strings VERBATIM.
(3) RED FIRST: run the item's falsifying check; record the failure.
(4) Build only what the item says. Nothing from §5.5.
(5) GREEN; then falsify (break, see red, restore, see green).
(6) Open every artefact you produced and describe it.
(7) scripts/battery.sh bare (add --slow when the item touched models).
    Update §1.4.
(8) ONE commit on main, scope = item id (`feat(a3): …`). WHY + red/green
    in the body. ONE line under "Wave A" in ongoing_general_errors.md §3.
    Never amend after pushing.
(9) /usr/bin/git push origin main
(10) Next item. A failed bar, an impossible rule or a missing input:
     file it, then continue only with items that do not depend on it.
```

---

## 9. Definition of Done: Wave A

- [ ] A1–A9 each landed as one pushed commit on `main`, scoped to its id, with red and green runs recorded.
- [ ] `scripts/battery.sh` exits 0, and `--slow` exits 0.
- [ ] G3's self-test passes all 4 checks; check (c) was shown to fail with item-level resampling.
- [ ] Split files exist for each dataset run, are group-disjoint, and are hash-verified.
- [ ] Issue 2 is resolved with a verbatim protocol quote, or filed with options.
- [ ] Oops!: every §6 condition has a row (real or `not run`), the Δ row exists, the shuffled controls sit at chance, and the report is written.
- [ ] HoloAssist: the same as Oops!, **or** A8 is recorded as blocked with its issue named.
- [ ] `docs/evals/<date>_wave_a_summary.md` is written, and every number in it is traceable to a scorecard row.
- [ ] §1.4 is re-measured bare.
- [ ] This guide is rewritten to **Queue Complete** (naming what it waits on). **Then stop. Do not invent work.**
