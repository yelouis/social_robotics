# Agent Execution Guide: Active Build: A15 report fix, then Wave D, the offline robot check (D1 → D2 → D3 → D4), October 21, 2026 (Issue 9 = B)

**You are an engineering agent with no memory of this project.**
- **What happened:** on October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
- **What exists: Wave A is complete** (A1–A14; `651039b` … `d6d9c54`; verified by the designer October 21):
  - **On `holoassist_audible`,** the instructor's non-verbal voice adds **+0.021 [0.006, 0.034]** to the action-only model (the ablation gain). That met Wave D's start condition, and the maintainer chose to start Wave D (Issue 8 = A).
  - **That gain is mostly whether the instructor spoke** (`00_thesis.md` entry 21).
- **What you build,** in the §2 order:
  - **A15:** a small report fix carried over from the last guide;
  - **D1:** a HoloAssist pretraining pool of 20,000 actions, labeled by the reaction model;
  - **D2:** the Wave D code (the prior-regularized learner, the robot encoder, the RoboReward adapter and downloader, and the verdict and report), tested on synthetic data only;
  - **D3:** download **all** of RoboReward (Issue 9 = B: 43.2 GB) and build its items, features and judge reference;
  - **D4:** run the pre-registered curve, write the report and close out.
- **The contract for D1–D4 is [`04_wave_d_robot_check.md`](04_wave_d_robot_check.md).** It is pre-registered: its pass rule and interpretations are fixed before any robot data exists. **Do not change them, and do not re-run with other settings after seeing a result.**

**Read before anything else, in this order:**
1. [`04_wave_d_robot_check.md`](04_wave_d_robot_check.md), **in full.**
2. [`00_thesis.md`](00_thesis.md): H3 and decision-log entries 20–23.
3. [`03_eval_harness.md`](03_eval_harness.md): §3, §8 (judges, including the `roboreward` question and the shared `gemma4:26b` rules), §11.8–§11.9 and §12 (memory).
4. [`02_data_sources.md`](02_data_sources.md): HoloAssist (`holoassist_audible`, `holoassist_pool`) and **RoboReward**.
5. [`ongoing_general_errors.md`](ongoing_general_errors.md): §1, **Issue 9** (decided: B), and lessons L10–L14.

**The maintainer's words:**
- *"The fundamental core of this project is to use human emotion as a policy for robot learning in addition to the traditional RL stack for robots. We want to add this additional reward signal and investigate if this additional reward signal helped the robot learn a task better and perhaps allow robots to train on much more data like vlogs or any POV youtube video."*
- *"Ideally we should avoid human in the loop ratings because that is not scalable and we want to be able to somehow measure whether our thesis is working often."*
- *"How is this any different from asking an LLM model what is socially appropriate…?"*
- *"No need to open up a new branch, just push to the repo."*
- **October 21, 2026:** *"For issue 8, lets go with Option A."* and *"For issue 9, lets go with Option B."*

**Status:** **Active Build.** Next is **D1**.
- **Nothing in this queue waits on the maintainer.** Issue 9 is decided (B: the full download).

**Every path, number, field name, formula and literal string in this guide, in `03_eval_harness.md` and in `04_wave_d_robot_check.md` is a decision, not a suggestion. Implement as written; do not substitute your own.**

---

## 0. Standing constraints (apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare after every item (`--slow` when the item ran models). G2 must pass 3 times in a row. Update §1.4.
   - Read every exit code bare.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only.
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies are frozen.**
   - No new pip packages. `scipy` 1.13.1, `scikit-learn` 1.6.1 and `huggingface_hub` 0.36.2 are already installed.
   - Pull no Ollama models, and download no models.
4. **Downloads:** the **only** authorized download is **all of** RoboReward, in D3: `teetone/RoboReward` (dataset), revision `469b9af76e8539e8d2ac553b081307738fd92ca5`, into `DATA_ROOT/raw/roboreward/`. That is 54,140 files and 43,204,474,744 bytes.
   - The free space after the download must stay ≥ 50 GiB.
   - Never circumvent a login, gate or rate limit.
5. **Data placement.** All data lives under `DATA_ROOT`. No video or audio on the internal disk. Never delete anything you did not create.
6. **Long runs (> 30 min) run detached:**
   ```bash
   ./venv/bin/python tools/daemonize.py <log> bash tools/run_supervised.sh <progress.json> <runner…>
   ```
   - Confirm `PPID 1`.
   - Every long runner is resumable, maintains `progress.json`, and ends with the count line `items_in=<a> items_out=<b> excluded=<c> elapsed_s=<t>`.
7. **Memory (`03_eval_harness.md` §12).**
   - Never run two of this project's model-loading jobs at once.
   - Run `python -m shared.memguard --status` before each model job, and record it.
   - Exit 75 is never a pass.
   - **`gemma4:26b` is shared; never unload it.**
8. **Determinism.** Use the seeds exactly as written in `04_wave_d_robot_check.md`, with inputs in sorted `item_id` order.
9. **No leakage.**
   - The robot **test** split is used only for scoring: it is never used to fit the scaler, a head, a detector or the labeler.
   - The labeler never sees robot data or the pool's labels.
   - Shuffled controls never see the true labels.
10. **The scorecard is append-only.** Nulls carry derived reasons.
11. **Reports are generated from the rows** (L10). The verdict text comes from `h3_verdict` only.
12. **Globs over the SSD use `shared.fsutil.data_glob`** (L13).
13. **Commits.**
    - One item = one Conventional Commit on `main`, scope = item id (`fix(a15): …`, `feat(d1): …`).
    - The body states the WHY, the red run, the green run, **and any validation bar that was missed** (L14).
    - Push with **`/usr/bin/git push origin main`**. No branches, no PRs, never amend.
    - Stage by explicit path only. If `git status` shows a change you did not make, STOP and ask.
14. **Record the resolution in the same commit:** one line in `ongoing_general_errors.md` §3 (under "Wave A" for A15, under a new heading **"Wave D"** for D1–D4).
15. **When this guide and a contract disagree, or a bar cannot be met, STOP and file it** as a new issue (the next number is **Issue 10**), with options.
16. **Never fill in a `Your selection: _____` line.** Issues 1, 4 and 9 belong to the maintainer.

---

## 1. Verified baseline (October 21, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1 (with `ffprobe`).
- **Ollama:** `qwen2.5vl:7b` (ours) and `gemma4:26b` (shared). No model was loaded.
- **`venv`, Python 3.9.6:** torch 2.8.0, transformers 4.46.3, funasr 1.3.1, scikit-learn 1.6.1, scipy 1.13.1, numpy 2.0.2, huggingface_hub 0.36.2, soundfile, httpx 0.28.1, ruff 0.15.11, pytest 8.4.2.
- **The SSD is exFAT** (L13), with about 1.3 TiB free.

### 1.2 Repository

- `HEAD` includes `d6d9c54` (`feat(a14)`) plus the designer's October 21 docs commits. The working tree is clean.

### 1.3 Data in place (what D1–D4 read)

| What | Where | State |
|---|---|---|
| `holoassist_audible` items, split, features (SigLIP, e2v), judge caches | `DATA_ROOT/items/holoassist_audible/`, `splits/holoassist_audible.json`, `features/holoassist_audible/` | complete (A13) |
| Per-session audibility | `DATA_ROOT/runs/holoassist_session_audibility.jsonl` | 1,673 sessions; 548 `audible: true` |
| HoloAssist videos and labels | `DATA_ROOT/raw/holoassist/` | 2,111 videos |
| RoboReward | `DATA_ROOT/raw/roboreward/` | **not downloaded yet.** D3 downloads all of it (Issue 9 = B). The SSD has 1.3 TiB free |

### 1.4 Gates (run bare October 21, 2026, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result on `d6d9c54` |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · clean |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **104 passed**, 3 runs out of 3 |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **9 passed** (121 s); `memguard --status`: 53.1 GB, pressure 1, lock free |

### 1.5 Reference values to reproduce

| Where | Value | Used by |
|---|---|---|
| `holoassist_audible` `judge-large` AUROC on test items whose verb is not `drop` | **0.498 [0.481, 0.514]**, n = 1,361 | A15 |
| `holoassist` (full) `judge-large` on test items whose verb is not `drop` | **0.489**, n = 1,818 | A15 |
| `holoassist_audible` `react-nonverbal` (row `2026-10-21T16:50:22Z`) | **0.6539 [0.6257, 0.6749]** | D1: the refit labeler must reproduce it to 4 dp on the `holoassist_audible` test split |

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| A15 | The report generators: the computed `judge-large` caveat and the one-sided leak label | Carried over. Small and independent; it fixes Wave A's record |
| D1 | The `holoassist_pool` (20,000 actions), its features, the reaction labeler and the pseudo-labels | It needs no download. Its ~4 h extraction runs while D2 is coded |
| D2 | The Wave D code: `transfer.prior_lr`/`heads`/`curve`, encoder `siglip-b16-224-u9`, `sources.roboreward`, `tools/download_roboreward.py`, `h3_verdict` and the report generator. Synthetic tests only | It needs no data, so it can land before Issue 9 is decided. D3 and D4 run it |
| D3 | Download all of RoboReward (Issue 9 = B); build its items and split; extract `siglip-b16-224-u9`; the `judge-large` reference | It needs D2's code (the download tool, the adapter and the u9 encoder) |
| D4 | Fit the scaler and the heads; run the pre-registered curve; rows, report and verdict; close out | It needs D1 (pseudo-labels) and D3 (robot features) |

---

## 3. The items

### A15: Report-generator accuracy: the computed `judge-large` caveat and the one-sided leak check

**What this means for the maintainer:** the audible-session report states a number from a different dataset (0.489, where its own value is 0.498). Two reports also label a harmless below-chance shuffled result as `FAIL`. Both are fixed, and no result changes.

**The gap** (read at `d6d9c54`):
- **`tools/report_holoassist.py:134` and `:139` hard-code `judge-large scores 0.489 on test items whose verb is not "drop".`**
  - That is the full `holoassist` value, and it was printed verbatim into `docs/evals/2026-10-21_holoassist_audible_h1.md:60` and the summary's §8.2.
  - On `holoassist_audible` the value is 0.498 [0.481, 0.514] (n = 1,361).
  - The literal came from the A9b spec; the generator must compute it.
- **The shuffled tables apply a two-sided rule:**
  - `tools/report_holoassist.py:615–633` (header `Covers 0.5?`, value `FAIL` when the CI misses 0.5);
  - `tools/report_oops.py:435–460` (the same, `FAIL (No)`).
  - So `docs/evals/2026-10-15_holoassist_h1.md:61` shows `step-prior:shuffled … FAIL` for 0.464 [0.440, 0.489]. By `03_eval_harness.md` §11.9 that is `below chance (one permutation; not a leak)`.

**Implementation:**
1. **New `src/harness/reporting.py`** (Python 3.9):
   - `leak_check_label(value, ci_low, ci_high) -> str` returns exactly one of:
     - `— (not run)` when `value is None`;
     - `FAIL: above chance (leak)` when `ci_low > 0.5`;
     - `below chance (one permutation; not a leak)` when `ci_high < 0.5`;
     - `pass` otherwise.
   - `judge_nondrop_auroc(dataset, model_tag, data_root, splits_dir) -> Optional[Tuple[float, float, float, int]]`:
     - it loads `items/<dataset>/items.jsonl`, `splits/<dataset>.json` and `judge/<dataset>/<model_tag>/<prompt_hash>.jsonl`, the latter found with `data_glob` and excluding `errors.jsonl`;
     - it keeps the test items whose `meta.verb != "drop"` and have a non-null judge probability;
     - it returns `auroc_ci(labels, probs, group_ids)` as `(point, ci_low, ci_high, n)`, or `None` if the cache is absent.
2. **`tools/report_holoassist.py`:**
   - the caveat (`:134`, `:139`) becomes `judge-large scores <v> [<lo>, <hi>] on the <n> test items whose verb is not "drop" (computed from judge/<dataset>/gemma4_26b)`. If `judge_nondrop_auroc` returns `None`, it reads `judge-large not run on <dataset>`;
   - the line at `:615` becomes `Leak check (03_eval_harness.md §11.9): a shuffled control fails only if its ci_low > 0.5; one permutation below chance is not a leak.`;
   - the header at `:617` becomes `| Condition | Metric | Value | 95% CI | Leak check | Notes |`;
   - the values at `:629–:633` come from `leak_check_label`.
3. **`tools/report_oops.py:435–460`:** the same three changes to its shuffled table.
4. **Regenerate the three reports:**
   - `docs/evals/2026-10-10_oops_h1.md` (`tools/report_oops.py`);
   - `docs/evals/2026-10-15_holoassist_h1.md` (`tools/report_holoassist.py --dataset holoassist`);
   - `docs/evals/2026-10-21_holoassist_audible_h1.md` (`--dataset holoassist_audible`).

   Add to each the line `*Updated <YYYY-MM-DD> (A15): judge-large caveat computed per dataset; shuffled controls labeled by the one-sided leak check (03_eval_harness.md §11.9). No result changed.*`
5. **The summary** (`docs/evals/2026-10-15_wave_a_summary.md` §8.2): replace the two `**Caveat:**` lines (`holoassist` and `holoassist_audible`) with the regenerated reports' caveat lines, verbatim. Leave the designer note in §8.4 as it is.

**Validation:**
- **Red first:** new tests, which fail on `d6d9c54`.
  - **`tests/test_report_holoassist.py`, a caveat computed from data:**
    - build a `tmp_path` DATA_ROOT holding 8 test items (4 `drop`, 4 not), a split file and a judge-large cache where the 4 non-`drop` items are perfectly separated;
    - the caveat line contains `1.000` and `on the 4 test items`;
    - change one non-`drop` probability so that AUROC = 0.75, and the line contains `0.750`.
  - **`tests/test_report_holoassist.py` and `tests/test_report_oops.py`, the leak label:**
    - `(0.51, 0.505, 0.53)` → `FAIL: above chance (leak)`;
    - `(0.464, 0.440, 0.489)` → `below chance (one permutation; not a leak)`;
    - `(0.49, 0.46, 0.52)` → `pass`;
    - `(None, None, None)` → `— (not run)`.
- **Falsify:** put the literal `0.489` back into the caveat → the computed-caveat test goes red → revert.
- **Reproduce the §1.5 references:**
  - the regenerated audible report contains `judge-large scores 0.498 [0.481, 0.514] on the 1361 test items`;
  - the regenerated `holoassist` report's value rounds to `0.489`.
- **The labels in the regenerated reports:**
  - no report contains `FAIL`;
  - the `holoassist` `step-prior:shuffled` row reads `below chance (one permutation; not a leak)`;
  - every other `:shuffled` row reads `pass` or `— (not run)`.
- `git diff results/scorecard.jsonl` is empty.
- G1–G3 green (G2 three times).

**Blast radius:** `src/harness/reporting.py` (new), `tools/report_holoassist.py`, `tools/report_oops.py`, `tests/test_report_holoassist.py`, `tests/test_report_oops.py`, the three reports above, `docs/evals/2026-10-15_wave_a_summary.md` §8.2; `ongoing_general_errors.md` §3; this guide's §1.4.

**Then:** continue with D1. Do not rewrite the guide's title.

---

### D1: The HoloAssist pretraining pool, the reaction labeler and the pseudo-labels

**What this means for the maintainer:** it turns 20,000 HoloAssist moments into "free" success labels taken **from the instructor's voice alone**. That is the thesis's claim that reactions can label video at scale, and Wave D's human input.

**The gap** (read at `d6d9c54`; contract `04_wave_d_robot_check.md` §2.2 and §3):
- **`src/sources/holoassist.py` cannot build the pool:**
  - `build_holoassist_items` (`:181`) always balances classes and applies the caps;
  - it has no way to exclude a given item set;
  - there is no `--build-pool` CLI (args at `:417–:423`).
- **There is no `src/transfer/` package,** and no labeler or pseudo-labels.

**Implementation:**
1. **`src/sources/holoassist.py`:**
   - factor the per-action candidate construction (windows, `context_text`, label, `meta`) out of `build_holoassist_items` into a helper that both functions use. **Do not duplicate the item definition**, and `holoassist` and `holoassist_audible` must rebuild byte-identically (see Validation);
   - add **`build_holoassist_pool(session_allowlist, exclude_keys, n_total=20000, seed=0)`**. The candidates are the allowlisted sessions' actions minus `exclude_keys`, a set of `(video_name, event index)`; sample `n_total` with `default_rng(seed)` in the natural class proportion; then sort by `item_id`;
   - **CLI:**
     ```bash
     python -m sources.holoassist --build-pool --audible-from "$DATA_ROOT/runs/holoassist_session_audibility.jsonl" --exclude-from "$DATA_ROOT/items/holoassist_audible/items.jsonl" --n 20000
     ```
     It writes `items/holoassist_pool/{items.jsonl,stats.json}` (with `dataset = holoassist_pool` and the class counts). There is no split file.
2. **Extract features** (detached; one model job at a time; about 3 h, then about 1 h):
   ```bash
   python -m features.extract --dataset holoassist_pool --encoder siglip-b16-224
   python -m features.extract --dataset holoassist_pool --encoder e2v-plus-large
   ```
3. **New `src/transfer/__init__.py` and `src/transfer/pseudolabel.py`,** run as `python -m transfer.pseudolabel`:
   - **refit the labeler exactly per §3** on the `holoassist_audible` train split, and save `DATA_ROOT/transfer/labeler.npz`;
   - **score every pool item** and write `DATA_ROOT/transfer/holoassist_pool_pseudolabels.jsonl` (`item_id`, `q`, `label`, `group_id`, `labeler_unseen_group`);
   - **append the two rows** `pseudo-label-quality` and `pseudo-label-quality:all` (hypothesis `H3`, dataset `holoassist_pool`, split `pool`; grouped bootstrap CIs; `n_excluded` = pool items without an e2v feature).

**Validation:**
- **Red first:** the new tests fail on `d6d9c54`.
  - **`tests/test_sources_holoassist.py`, the pool builder on the 3-session synthetic fixture:**
    - excluded keys never appear;
    - only allowlisted sessions appear;
    - the class proportion follows the candidates (no balancing);
    - `n_total` is respected;
    - the result is deterministic.
  - **`tests/test_transfer_pseudolabel.py`:** the labeler refit equals a direct `sklearn` fit on synthetic features, and the `labeler_unseen_group` flag is correct.
- **The refactor changed nothing:** rebuilding `holoassist` and `holoassist_audible` into a temp dir gives byte-identical `items.jsonl` files (same sha256 as the files on the SSD).
- **Reproduce §1.5:** the refit labeler scores the `holoassist_audible` test split at **0.6539 [0.6257, 0.6749]** (4 dp).
- **The pool:** 20,000 items; no `holoassist_audible` item; every session audible; the class counts are recorded. A mistake share of roughly 5% is expected; record the real one.
- **Feature exclusions** are ≤ 5% for each encoder, with reasons.
- G1–G4 green.

**Blast radius:** `src/sources/holoassist.py`, `src/transfer/{__init__,pseudolabel}.py` (new), `tests/test_sources_holoassist.py`, `tests/test_transfer_pseudolabel.py` (new), `results/scorecard.jsonl` (2 rows appended); `ongoing_general_errors.md` §3 ("Wave D"); this guide's §1.4.

---

### D2: The Wave D code (synthetic tests only; no network, no data)

**What this means for the maintainer:** everything needed to run the robot check is built and proven on toy data first, so the real run cannot quietly do something other than the pre-registered plan.

**The gap** (read at `d6d9c54`; contract `04_wave_d_robot_check.md` §2.1, §4–§6):
- **`src/features/visual.py`:** `get_sample_times` (`:14`) is 1-fps only, and `FrameEncoder.encoder_id` (`:44`) is fixed. `src/features/extract.py` knows only two encoders (`:37–:39`, `:77`, `:97`, `:150`).
- **`src/judge/vlm_judge.py:33` `DATASET_QUESTIONS`** has no `roboreward`.
- **None of these exist:** `transfer.prior_lr`, `transfer.heads`, `transfer.curve`, `sources.roboreward`, `tools/download_roboreward.py`, `tools/report_wave_d.py`.

**Implementation:**
1. **`src/features/visual.py`:**
   - `get_sample_times_u9(s, e)` returns `[s + k·(e − s)/8 for k in 0..8]`;
   - `FrameEncoder(sampling="1fps"|"u9")`, where `encoder_id` is `siglip-b16-224` or `siglip-b16-224-u9`;
   - **`features/extract.py`** accepts `--encoder siglip-b16-224-u9`, under the memory-guard step `sr_siglip`.
2. **`DATASET_QUESTIONS["roboreward"]`** = `Did the robot complete the task successfully by the end of the video?` (verbatim, from `03_eval_harness.md` §8).
3. **`src/transfer/prior_lr.py`:**
   - `fit_prior_lr(X, y, w0=None, C=1.0, sample_weight=None) -> Tuple[np.ndarray, float]` and `predict_proba(X, w, b)`, exactly per §4;
   - `soft_label_rows(X, q) -> (X2, y2, sw2)`, per §4.
4. **`src/transfer/heads.py`:**
   - `fit_scaler(X_pool, X_robot_train)` → saved to `DATA_ROOT/transfer/scaler.npz`;
   - `fit_heads(Xs_pool, q, y_true) -> {"react", "shuffled", "truth"}`, per §4 (shuffled via `default_rng(0)`) → saved to `heads.npz`.
5. **`src/transfer/curve.py`** (`python -m transfer.curve`):
   - it loads the pool features and pseudo-labels, the robot features and `splits/roboreward.json`;
   - it fits the scaler and the heads, then runs §5's draws (`N` × 20 × 4 priors);
   - it computes the `mean20` AUROCs, gains, `react_minus_shuffled`, `robot-full` and the leak check with the paired group bootstrap;
   - it writes `curve_scores.npz` (every draw's test scores) and appends the §5 rows. The `judge-large` row comes from the judge cache, through the existing `build_test_only_judge_row` logic. `roboreward-8b` is `not run`, with the reason pinned in `04` §5;
   - it ends with the count line.
6. **`src/sources/roboreward.py`:**
   - **the pure rule functions:** `classify(split, source, file_stem, reward) -> ("success"|"failure"|"excluded", reason)`, `group_of(source, file_stem)` and `parse_row(row) -> (split, source, stem, task, reward)`. Make the file-path field name a module constant, verified against the real schema in D3;
   - **`sample_train_pool(eligible_train, seed=0) -> (successes, failures)`** (2,000 + 2,000, per §2.1);
   - **`build_items(...)`** (durations via `ffprobe`) → `items/roboreward/{items.jsonl,stats.json}`;
   - **the split file through `make_group_split(official=…)`**, with the §2.1 straddle rule applied **before** sampling;
   - **CLI:** `python -m sources.roboreward --build`.
7. **`tools/download_roboreward.py`** (Issue 9 = B; there is no selective mode):
   - **First, check the disk rule:** free space must be ≥ 43,204,474,744 bytes + 50 GiB, else STOP.
   - Then `huggingface_hub.snapshot_download(repo_id="teetone/RoboReward", repo_type="dataset", revision="469b9af76e8539e8d2ac553b081307738fd92ca5", local_dir=DATA_ROOT/raw/roboreward)`. It is resumable: rerunning it skips completed files.
   - **Then verify against the Hub's metadata at that revision** (`HfApi().dataset_info(…, files_metadata=True)`): every one of the 54,140 files exists locally with the Hub's byte size. Use `data_glob` to count the local `.mp4` files, and expect 54,135.
   - **Write `DATA_ROOT/raw/roboreward/DOWNLOAD.json`:** `{repo_id, revision, n_files, bytes, n_mp4, metadata_sha256: {train, val, test}, downloaded_at, free_gib_after}`.
   - It ends with the count line. **D2 never runs it against the network.**
8. **`tools/report_wave_d.py`:**
   - **`h3_verdict(rows) -> List[str]`**, exactly per `04` §6: the supported/not-supported line plus exactly one interpretation line, chosen in table order;
   - **the report** `docs/evals/<YYYY-MM-DD>_wave_d_robot_check.md`:
     - setup and counts;
     - pseudo-label quality;
     - the curve table (prior × N, with CIs);
     - the gains and `react_minus_shuffled`;
     - the references;
     - the leak check (§11.9 label);
     - the `h3_verdict` lines verbatim;
     - the caveats: Wave A's signal is mostly speech presence; the human–robot domain gap; the vision-only label rule; pitch-shifted audio;
     - compute time.

**Validation** (all synthetic; **red first** for each test):
- **`tests/test_transfer_prior_lr.py`:**
  - **(a) Equivalence:** with `w0 = None` it equals `sklearn` `LogisticRegression(C=1.0, class_weight="balanced")` on a 200×20 synthetic set (test AUROC within 1e-4; coefficients `rtol=1e-3`);
  - **(b) The prior helps when it is right:** with `N = 20` and `w0` = the true generating direction, the mean AUROC over 20 draws beats `w0 = 0`;
  - **(c) Falsifying:** with `w0` = a random direction, it does not beat `w0 = 0` by more than 0.01;
  - **(d) Soft labels:** `q ∈ {0, 1}` reproduces the hard-label fit.
- **`tests/test_sources_roboreward.py`:**
  - each `classify` case of §2.1 (OXE original r5 → success; OXE original r3 → excluded `counterfactual`; attempt r2 → failure; attempt r4 → excluded `partial`; `robo_arena` r5/r1/r3 → success/failure/excluded);
  - `group_of` strips `_attempt_<k>_score_<s>` and keeps the `robo_arena` uuid;
  - `sample_train_pool` is deterministic and balanced;
  - the straddle rule removes the train-side items.
- **`tests/test_features.py`:** `get_sample_times_u9(0, 8)` = `[0, 1, …, 8]` and `get_sample_times_u9(0, 4)` = `[0, 0.5, …, 4]`.
- **`tests/test_report_wave_d.py`:** `h3_verdict` on five fixtures (supported, plus one per interpretation row), each printing exactly the expected interpretation.
- **`tests/test_transfer_curve.py`:** on a synthetic robot set and pool:
  - the rows have the §5 names and metrics;
  - the draws are paired (the same indices for all priors);
  - the leak check is at chance;
  - the bootstrap is deterministic.
- **`tools/download_roboreward.py`:** with `snapshot_download` and `dataset_info` monkeypatched to a local fixture of 5 files:
  - a complete fixture passes;
  - a fixture with one file missing **fails** the verification;
  - a fixture with one file truncated **fails** it;
  - a fake free-space reading below the disk rule makes it refuse to start.
- G1–G3 green.

**Blast radius:** `src/features/{visual,extract}.py`, `src/judge/vlm_judge.py`, `src/transfer/{prior_lr,heads,curve}.py` (new), `src/sources/roboreward.py` (new), `tools/download_roboreward.py` (new), `tools/report_wave_d.py` (new), and the new tests; `ongoing_general_errors.md` §3; this guide's §1.4.

---

### D3: RoboReward: the full download, items, split, robot features and the `judge-large` reference

**What this means for the maintainer:** it puts all of RoboReward on disk (your Issue 9 choice: 43.2 GB) and turns the part the spec uses into the labeled set the robot check is scored on.

**Start condition:** D2 is pushed.

**Implementation:**
1. `memguard --status` (record it).
2. **Download** (run it detached; about 20 min at the ~37 MB/s seen for HoloAssist, possibly longer):
   ```bash
   ./venv/bin/python tools/daemonize.py "$DATA_ROOT/runs/download_roboreward.log" bash tools/run_supervised.sh "$DATA_ROOT/raw/roboreward/progress.json" ./venv/bin/python tools/download_roboreward.py
   ```
   - Confirm `PPID 1`.
   - Check that `DOWNLOAD.json` shows 54,140 files, 43,204,474,744 bytes and 54,135 `.mp4` files, and that `free_gib_after ≥ 50`.
3. **Record the schema** in `02_data_sources.md` → RoboReward, as **"Schema (as downloaded, <date>)"**:
   - the `metadata.jsonl` field names, including the file-path field (fix `sources.roboreward`'s constant if it differs, and say so in the commit body);
   - the counts per split;
   - the duration range.
4. **`python -m sources.roboreward --build`:**
   - `items/roboreward/`;
   - `splits/roboreward.json`, a first write;
   - `stats.json` with counts per source × rule outcome × reason, the straddle removals and any pool shortfall.
5. **Extract** (detached): `python -m features.extract --dataset roboreward --encoder siglip-b16-224-u9` (all items).
6. **The `judge-large` reference** (detached): `python -m judge.vlm_judge --dataset roboreward --split test --backend ollama-large`.

**Validation:**
- **No OXE original with `reward < 5` and no `reward` 3 or 4** appears among the items. Assert it in a slow test over the real `items.jsonl`.
- **Every `_attempt_` item is a failure, and every success is an original or a `robo_arena` file.**
- **The groups are correct:** for 20 random OXE items, the original and its attempts share a `group_id`, and each `robo_arena` rollout's cameras share one.
- **The split** is hash-verified, with straddle removals recorded.
- **The sizes:** the train pool has 2,000 + 2,000 items (or the recorded shortfall), and the test has 1,200–2,000 items. **Fewer than 600 test items: STOP** and file Issue 10.
- **Coverage:** u9 feature exclusions ≤ 5%; the judge covers ≥ 90% of the test items.
- G1–G4 green.

**Blast radius:** `src/sources/roboreward.py` (the field constant, if needed), `splits/roboreward.json` (new), `docs/02_data_sources.md`; `ongoing_general_errors.md` (§3, and **Issue 9 → 🧪 Resolved** with the `DOWNLOAD.json` numbers); this guide's §1.4.

---

### D4: The pre-registered curve, the report, the H3 verdict and the close-out

**What this means for the maintainer:** the answer to "do reaction-labeled human videos help a robot learn what success looks like from fewer labels?", judged by the rule written before the data existed.

**Start condition:** D1 and D3 are pushed.

**Implementation:**
1. **`python -m transfer.curve`.** It takes minutes: there are no models, only the cached features.
2. **`python tools/report_wave_d.py`** → `docs/evals/<YYYY-MM-DD>_wave_d_robot_check.md`.
3. **One line under "Wave D"** in `ongoing_general_errors.md` §3: the `h3_verdict` lines verbatim, with the row `ts`.
4. **Rewrite this guide's title and status to "Queue Complete: waiting on the designer's verification of Wave D, and Issue 1 (Wave B)". Then stop. Do not invent work.**

**Validation:**
- **The leak check** `detector:none:N=1000:shuffled` passes (`ci_low ≤ 0.5`).
- **The real-data equivalence spot check:** for one draw (N = 1000, r = 0), `prior_lr` with `w0 = None` matches `sklearn` on the real robot features (test AUROC within 1e-4).
- **Every §5 row exists:** 20 + 15 + 5 + 4, plus the two pseudo-label rows from D1.
- **The report's verdict lines equal `h3_verdict(rows)`,** and every number in the report matches a row (a one-off script; output in the commit body).
- **No re-run with changed settings** (`04` §6). If something is wrong, file Issue 10; do not "fix" the numbers.
- G1–G4 green.

**Blast radius:** `results/scorecard.jsonl` (appended), `DATA_ROOT/transfer/*`, `docs/evals/<date>_wave_d_robot_check.md`; `ongoing_general_errors.md` §3; this guide (close-out).

---

## 4. Deferred: do NOT start

| Work | Trigger | Who writes the spec |
|---|---|---|
| **RoboReward-8B reference** | Not selected (Issue 9 = B). It would need a new maintainer decision to unfreeze `transformers`, plus a designer spec | The designer |
| **Wave B: verdict-video H1 pilot**, DW1 | **Issue 1** selected **and** a Wave B spec (which restates the kill rule in ablation-gain terms) | The designer |
| **Online H3** (training a policy with the reward) | After the offline result, and a cloud CUDA GPU | The designer |
| **Face encoder**, DW2 · **H2 transfer**, DW3 · **Ego4D false-positive set**, DW4 | Wave B | The designer |
| **Stage A live (microduck)**, DW6 | Hardware, H2 done, and the maintainer opts in | The designer |

Also not to do:
- other N budgets, `C` values, frame rules, pool sizes, labelers or priors (`04` §6);
- the designer's exploratory strata as rows;
- any probe re-run on Wave A datasets;
- any judge prompt change;
- anything from `v0-saf-final`.

---

## 5. Do NOT change

### 5.1 Already delivered (verified by the designer)

| Item | Commit | What it delivered |
|---|---|---|
| A1–A6, A6b, A6c | `651039b`–`0f16799`, `76e71cc`, `8c6f04e` | the harness, the encoders, the judges, the memory guard |
| A7, A7b, A7c, A8 | `6d72492`, `d77584c`, `1346e0b`, `36a7769` | the Oops! and HoloAssist pipelines; `judge-large` |
| A9, A9b | `3f0b78d`, `6744719` | the Wave A summary, corrected |
| A10 | `492bd79` | `shared.fsutil.data_glob` |
| A11 | `abbc273` | `step-prior`, `action-fusion` and the ablation Δ |
| A12, A13 | `608c729`, `eea8933` | per-session audibility; `holoassist_audible` (ablation +0.021 [0.006, 0.034]) |
| A14 | `d6d9c54` | the Wave A completion; the Wave D trigger met by the rule |

### 5.2 Accepted equivalents (do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure.**
- **`extract_frame` retries a seek up to 0.3 s earlier.**
- **`GeminiJudge`:** 32 output tokens, thinking off, and a 429 is not cached.
- **`guard()` holds the heavy lock through the model load only.**
- **Scorecard rows carry the parent commit's `git_sha` with `dirty: true`.**
- **The HoloAssist group partition is computed over all 1,758 sessions,** and `holoassist_audible` reuses it.
- **`holoassist_audible`'s `not_audible` count includes the 85 sessions without video,** and it has 1,420 test items (above the estimate).

### 5.3 Maintainer decisions

- **October 8–10:**
  - outcome supervision only, with no human rating;
  - "just ask an LLM" is the control;
  - hidden-outcome data first;
  - Mac Studio only;
  - commit to `main`;
  - Issue 5 → B.
- **October 20:** Issue 6 → A (a fresh sample); Issue 7 → A; the pass bar and the Wave D trigger use the ablation gain.
- **October 21:** **Issue 8 → A** (write and run Wave D); **Issue 9 → B** (the full RoboReward download; RoboReward-8B not run).

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good** (for RoboReward, success). **The judge never hears audio.**
- **Grouped splits and CIs. Split files are written once. The scorecard is append-only.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels. The leak check is one-sided.**
- **The Wave D detector never reads the instruction.** Counterfactual-instruction negatives are excluded.
- **The robot test split is used only for scoring.**

### 5.5 Assessed and rejected: do NOT re-propose

- human rating, rater UIs, golden labels;
- hand-built affect channels;
- self-recorded data, studies, academic partners;
- tuning anything on test results;
- changing a pre-registered rule after seeing its result;
- fine-tuning encoders end to end (Mac Studio only);
- circumventing download gates or rate limits.

---

## 6. Where the contracts live

| What | Where |
|---|---|
| **Wave D: data, labels, learner, curve, rows, the pre-registered verdict** | **`04_wave_d_robot_check.md`** |
| The thesis, H1–H3, the decision log (entries 19–23) | `00_thesis.md` |
| The scorecard, conditions, metrics, judges (§8, incl. `roboreward`), guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| Datasets: Oops!, HoloAssist (`holoassist_audible`, `holoassist_pool`), RoboReward | `02_data_sources.md` |
| Issues 1, 4, 9 (open); 2, 3, 5–8 (resolved); lessons L1–L14 | `ongoing_general_errors.md` |
| Operations | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard

- **Red first, then green, then falsify.**
- **Reproduce the reference values** (§1.5). A mismatch is a finding.
- **A flaky gate is a failing gate.** Run G2 three times.
- **Report every missed validation bar in the commit body** (L14).
- **Never loosen a bar or change a pre-registered rule to pass it.** File it.

---

## 8. THE LOOP

```
(1) Next item in §2 order: A15, D1, D2, D3, D4. If all are done: STOP
    (D4's close-out). Never start §4 work. Never fill in a
    `Your selection:` line.
(2) Read the item and every contract section it names (04 in full for D1–D4).
(3) RED FIRST: run the item's new tests; record the failure.
(4) Build only what the item says.
(5) GREEN; then falsify (break, see red, restore, see green).
(6) Open every artefact you produced and read it.
(7) scripts/battery.sh bare with TMPDIR on the SSD (G2 three times;
    --slow if the item ran models). Update §1.4.
(8) ONE commit on main, scope = item id, explicit paths only. WHY +
    red/green + any missed bar in the body. ONE line in
    ongoing_general_errors.md §3.
(9) /usr/bin/git push origin main
```

---

## 9. Definition of Done

- [ ] A15, D1, D2, D3 and D4 each landed as one pushed commit, with red and green runs recorded.
- [ ] The A15 report fixes are in: no report hard-codes 0.489, and no below-chance shuffled control is shown as `FAIL`.
- [ ] D1: the pool has 20,000 items; the labeler reproduces 0.6539; the pseudo-label rows are written; `holoassist` and `holoassist_audible` rebuild byte-identically.
- [ ] D2: every synthetic test passes, including the `sklearn` equivalence and the falsifying random-prior test.
- [ ] D3: all of RoboReward is on disk and verified (54,140 files, 43,204,474,744 bytes); the label rule is enforced on the real items; the groups and split are verified; Issue 9 is moved to 🧪 Resolved.
- [ ] D4: every §5 row exists; the leak check passes; the report's verdict equals `h3_verdict`; there are no re-runs with changed settings.
- [ ] This guide is rewritten to **Queue Complete** (naming what it waits on). **Then stop. Do not invent work.**
