# Agent Execution Guide: Active Build: A16 → B4b → E1 → E2 (fixture timeout; Wave B report correction; the write-up), November 5, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:**
  - On October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
  - **Waves A, D and B are complete and verified by the designer:** A1–A15, D1–D4 and B1–B4, commits `651039b` … `dcb44be` (§5.1).
  - **Wave B (REFLEX) recorded the pre-registered verdict PROCEED,** through one clause: `reflex` `react-face` 0.609 [0.569, 0.644].
    - The designer's verification found that clause to be a **window-length artifact** (`05_wave_b_reflex.md` §7).
    - On November 5 the maintainer chose **Issue 12 = A**: Wave B is recorded as negative in substance, and the project moves to **the write-up** (Wave E, [`06_writeup.md`](06_writeup.md); `00_thesis.md` entry 30).
- **What is approved now, four items, in this order:**
  1. **A16:** bound the `say` audio fixture with a timeout, so the slow tests cannot hang.
  2. **B4b:** make the Wave B report generator compute what it states, regenerate the report, and reproduce the designer's post-hoc diagnostics in code.
  3. **E1:** the paper's fact base, generated tables and figures, and a checker that rejects any number in the prose that is not in the fact base.
  4. **E2:** the first full draft of the paper, `docs/paper/draft.md`.
- **What you must not touch:**
  - any scorecard row, cache, split or item file;
  - the pre-registered verdicts;
  - the decisions reserved for the maintainer at draft review (`06_writeup.md` §8).

**Every path, number, field name and literal string in this guide is a decision, not a suggestion. Implement as written; do not substitute your own values.**

---

## 0. Standing constraints (they apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare (`--slow` when the item touches the slow tests). G2 must pass **3 times in a row** with the `TMPDIR` below.
   - Read every exit code bare. A gate that did not run is reported as *not run*, never as passed.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only (no `match`, no runtime `X | Y`).
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies, models and downloads are frozen.** No new pip packages, no new model downloads, no Ollama pulls, and no data downloads.
4. **Data lives under `DATA_ROOT`** (`/Volumes/Extreme SSD/social_robotics`; exFAT, 1.0 TiB free).
   - **No item writes anything under `DATA_ROOT`.**
   - Never delete anything you did not create.
   - Every glob over the SSD uses `shared.fsutil.data_glob` (L13).
5. **Memory (`03_eval_harness.md` §12).** No item loads a model. Do not run an item while another of this project's model jobs runs. **`gemma4:26b` is shared with another project; never unload it.**
6. **The scorecard is append-only, and no item appends to it.** The post-hoc diagnostics go to `docs/evals/` and the paper's facts to `docs/paper/`, never to `results/scorecard.jsonl`.
7. **Pre-registered rules and recorded verdicts are never changed.** That covers `05_wave_b_reflex.md` §1–§6 and the PROCEED verdict. The designer's notes sit beside the verdict; they do not replace it.
8. **Commits:**
   - one item = one Conventional Commit on `main` (`test(a16): …`, `fix(b4b): …`, `feat(e1): …`, `docs(e2): …`), with the WHY, the red and green runs, and any missed validation bar in the body (L14);
   - stage explicit paths only;
   - add the item's line to `docs/ongoing_general_errors.md` §3;
   - push with `/usr/bin/git push origin main`;
   - no branches, no PRs, never amend.
9. **When a contract and this guide disagree, or a bar cannot be met, STOP and file a new issue** (the next number is **Issue 13**) in `docs/ongoing_general_errors.md`, with options.
10. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer. No issue is open now.

---

## 1. Verified baseline (November 4, 2026; the designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1.
- **SSD:** 1.0 TiB free.
- **`memguard --status`:** 48.6 GB available, pressure 1, heavy lock free, `gemma4:26b` resident (shared).
- **`venv`:** Python 3.9.6, torch 2.8.0, transformers 4.46.3, scikit-learn 1.6.1, scipy 1.13.1, numpy 2.0.2, huggingface_hub 0.36.2, ruff 0.15.11, pytest 8.4.2.

### 1.2 Repository

- `HEAD` is `dcb44be` (`feat(b4)`), plus the designer's November 4 docs commit `3540ee3` and the November 5 docs commit (Issue 12 = A; `06_writeup.md`). The working tree is clean.

### 1.3 Data on the SSD (complete; read-only for both items)

| What | Where |
|---|---|
| Oops!, HoloAssist (`holoassist`, `holoassist_audible`, `holoassist_pool`) | `raw/`, `items/`, `features/`, `judge/` |
| RoboReward and the Wave D v1 artefacts | `raw/roboreward/`, `items/roboreward/`, `transfer/{labeler,scaler,heads,curve_scores}.npz` |
| REFLEX | `raw/reflex/Dataset/` (55 participants; `DOWNLOAD.json` `verified: true`); `items/{reflex,reflex_resolve}/`; `features/{reflex,reflex_resolve}/{reflex-face-v1,reflex-speech-v1,siglip-b16-224-u9}/`; `judge/{reflex,reflex_resolve}/{qwen2.5vl_7b,gemma4_26b}/`; `transfer/{reflex,reflex_resolve}_face_probe.npz` |

### 1.4 Gates (run bare November 4, 2026 on `dcb44be`, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · `All checks passed!` |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **122 passed, 11 deselected**, 3 runs out of 3 (68.55 s, 67.28 s, 66.44 s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS (`planted signal auroc=0.830 [0.807, 0.852]`, `null ci=[0.459, 0.523]`, `ratio=4.66`, `paired delta ci_low=0.138`) |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | **First run: hung 9.6 min** in the `say` fixture (`tests/test_features.py:146`). The designer stopped the hung `say`, giving 3 setup errors and exit 1. **Rerun: exit 0, 11 passed, 122 deselected (75.35 s).** A16 fixes the hang |

### 1.5 Headline results (the scorecard is the source)

| Dataset | Result | Row `ts` |
|---|---|---|
| `oops` | ablation gain +0.015 [−0.001, 0.029]; `action-probe` 0.779; `judge-large` 0.708 | `2026-10-21T04:12:36–42Z` |
| `holoassist` | `step-prior` 0.751; `action-fusion` 0.682; no reaction Δ (the audio was inaudible in most sessions) | `2026-10-21T04:12:57Z–04:13:02Z` |
| `holoassist_audible` | `react-nonverbal` 0.654; **ablation gain +0.021 [0.006, 0.034]**, mostly *whether the instructor spoke*; fusion − `step-prior` −0.065 | `2026-10-21T16:50:20–30Z` |
| `roboreward` (Wave D v1) | **NOT SUPPORTED**: `gain:react:N=100` +0.0009 [−0.0006, 0.0024] | `2026-10-22T14:22:28Z–14:24:54Z` |
| `reflex` (Wave B) | `react-face` 0.609 [0.569, 0.644], the **PROCEED** clause, **found to be a window-length artifact** (§4 note); ablation Δ +0.023 [−0.004, 0.045]; `action-fusion` 0.794; `judge` 0.720; `judge-large` 0.500 (it answered 100 on every item) | `2026-11-05T05:49:35Z–42Z` |
| `reflex_resolve` (Wave B) | `react-face` 0.552 [0.440, 0.663]; `react-nonverbal` 0.660 [0.545, 0.766]; ablation Δ +0.036 [−0.002, 0.078]; **`step-prior` 0.938** (the outcome follows the step); fusion − `step-prior` −0.192 | `2026-11-05T05:49:45Z–51Z` |

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| 1 | **A16**: a timeout and retries for the `say` fixture | B4b's validation runs the battery. A16 removes the hang that stalled G4 on November 4, so B4b's gates can be trusted to finish |
| 2 | **B4b**: Wave B report generator computes what it states; post-hoc diagnostics in code | It needs a battery that cannot hang. Its JSON of diagnostics is an input to E1 |
| 3 | **E1**: fact base, tables, figures, number checker | It reads B4b's JSON and the corrected report. E2 cannot be checked without it |
| 4 | **E2**: the draft | It pastes E1's tables and figures and must pass E1's checker |

After E2, the queue is empty (§9).

---

## 3. The items

### Item A16: bound the `say` audio fixture with a timeout and retries

**What this means for the user:** today, one stalled text-to-speech call can hang the slow tests forever, so a validation run never finishes and nobody knows whether it passed.

**The gap:**
- `tests/test_features.py:146–147` (fixture `audio_fixtures`) runs `subprocess.run(["say", "-v", "Samantha", ...], check=True)` and the same for `"Daniel"`, with **no `timeout`**.
- On November 4, 2026 one call sat in state `S` at 0% CPU for 9.6 minutes. The same command, run seconds later, returned in 1 s.

**Implementation:**
1. In `tests/test_features.py`, add a module-level helper:
   ```python
   def _say_to_file(voice: str, out_path: Path, text: str, attempts: int = 3, timeout_s: float = 60.0) -> None:
   ```
   - Each attempt runs `subprocess.run(["say", "-v", voice, "-o", str(out_path), text], check=True, timeout=timeout_s)`.
   - On `subprocess.TimeoutExpired`, delete `out_path` if it exists, then try again.
   - After `attempts` timeouts, raise `RuntimeError(f"say timed out {attempts} times ({timeout_s:.0f} s each) for voice {voice}")`.
   - **Do not `pytest.skip`.** A fixture that cannot build its input is a failure, not a pass.
2. Replace the two `say` calls in `audio_fixtures` with `_say_to_file("Samantha", sam_aiff, sentence)` and `_say_to_file("Daniel", dan_aiff, sentence)`.
3. Leave the `ffmpeg` calls alone. They finish in milliseconds on generated input.

**Validation:**
- New **fast** tests (not marked `slow`), in `tests/test_features.py`:
  - **(falsifying)** `test_say_helper_passes_timeout`: monkeypatch `subprocess.run` with a recorder and call `_say_to_file`. Assert the recorded call has `timeout == 60.0` and `check is True`. *Against today's code, which passes no `timeout`, this fails.*
  - `test_say_helper_retries_then_succeeds`: the fake raises `subprocess.TimeoutExpired` twice, then returns. Assert exactly 3 calls and no exception.
  - `test_say_helper_raises_after_three_timeouts`: the fake always raises `TimeoutExpired`. Assert `RuntimeError` matching `"say timed out 3 times"`, and that a partial file created by the fake was deleted.
- Run `scripts/battery.sh --slow` bare. G4 must show **11 passed**, and G2 runs 3 times in a row with **125 passed** (122 + 3 new).
- Commit `test(a16): bound the say audio fixture with a timeout and retries`.

**Blast radius:** `tests/test_features.py` only. The `docs/ongoing_general_errors.md` §3 line goes in the same commit.

---

### Item B4b: the Wave B report generator computes what it states; the post-hoc diagnostics in code

**What this means for the user:**
- **Today:** the Wave B report states resolution counts that do not exist in the data, and the finding that PROCEED is a window-length artifact exists only as the designer's scratch computation.
- **After B4b:** every number in the report is computed, and anyone can rerun the diagnostics behind Issue 12.

**The gap:**
- **`tools/report_reflex.py:711–730` appends three hand-written paragraphs** (three `md.append(...)` calls at lines 711, 720 and 726, after the section header at line 710) (section "3.3 Shuffled-Control & Methodological Notes"), with literal numbers in the prose:
  1. **The leak-check paragraph:** its numbers (`0.5100`, `0.0626`, `[0.3302, 0.6526]`, `0.6025`, "92nd percentile") are correct, but they are typed in.
  2. **The step-prior paragraph is false.**
     - It quotes `475 / 478 = 99.4%` and `13 / 126 = 10.3%`, which are not in the data, and names an "`Explain` column" (the column is `Explanation Level`).
     - **Actual resolution by explanation level:** high 165/187, medium 115/143, low 106/143, zero 102/131.
     - **By step:** `pick` 219/219, `carry` 198/220, `place` 71/165.
  3. **The `judge-large` paragraph:** its counts are typed in.
- **This broke L10 and B4's own validation** ("every number matches"). The designer's note at the end of `docs/evals/2026-11-04_reflex_h1.md` records the error.

**Implementation:**
1. **`tools/report_reflex.py`: delete the three hard-coded `md.append(...)` paragraphs at lines 711–730, and generate section 3.3 from data:**
   - **(a) Resolution by step and by explanation level** for `reflex_resolve`, read from `DATA_ROOT/items/reflex_resolve/items.jsonl`:
     - two Markdown tables, one by `meta.verb` and one by `meta.noun`;
     - columns `level | resolved | total | rate`, where `rate` is a percentage with one decimal place and no `%` sign;
     - one set over all items, and one over the test split.
     - **The heading must name the column `Explanation Level`.**
   - **(b) For every `:shuffled` row whose `leak_check_label` starts with `FAIL`:** compute a 50-seed permutation null in code.
     - Seeds `0 … 49`; for each, `np.random.default_rng(seed).shuffle` a copy of `y_train`.
     - Use the same feature cache, and the same `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")` fit on the train items with valid features, as `harness.probes`.
     - Score the test items with valid features.
     - Print mean, SD (`ddof=0`), min, max, the seed-0 value, the seed-0 percentile among the 50, and how many of the 50 the true-label condition's test AUROC beats.
     - Supported for `react-face`, `react-nonverbal` and `action-probe`. For any other condition, print `permutation null not computed for <condition>`.
     - **Expected for `reflex_resolve` `react-nonverbal:shuffled`:** mean 0.5100, SD 0.0626, min 0.3302, max 0.6526, seed 0 = 0.6025; the true label beats 50 of 50.
   - **(c) The constant-judge note:** for each judge cache and dataset, if one `judge_prob` value covers ≥ 95% of the test items, print `"<model> answered prob=<v> on <k> of <n> test items (AUROC <value from the row>)"`. Expected: `gemma4_26b` on `reflex`, prob 1.0 on 374 of 374.
   - **(d) Preserve the designer's note.** When the output file already exists and contains a line starting with `## Designer note`, copy everything from that line to the end of the file verbatim to the end of the regenerated file, preceded by a line `---`.
2. **New `tools/diag_reflex_posthoc.py`:** reproduces the designer's post-hoc diagnostics and writes `docs/evals/2026-11-04_reflex_posthoc.json`, an object keyed by the names below, each value `{"value", "ci_low", "ci_high", "n_test"}`.
   - **Read-only.** No cache writes, no scorecard rows.
   - **Probes:** `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")`, fit on train items with valid features.
   - **CIs:** `harness.metrics.auroc_ci` (grouped, 1,000 resamples, seed 0). Deltas use `harness.metrics.delta_auroc_ci(y, score_base, score_new, groups)`.
   - **On new windows,** call `features.face.ReflexFaceEncoder().encode(participant_dir, a, b)` or `ReflexSpeechEncoder().encode(participant_dir, wr_s, wr_e)` directly. A `RuntimeError` means that item has no features.

   | Key | Definition | Expected (3 dp) |
   |---|---|---|
   | `reflex.react_face` | cached `reflex-face-v1` | 0.608 [0.569, 0.644] |
   | `reflex.window_length` | score = s·L, L = `reaction_frames[1] − reaction_frames[0] + 1`, s = +1 if train AUROC(L) ≥ 0.5 else −1; test items with valid face features | 0.735 [0.675, 0.796]; train AUROC 0.775 with s = −1 |
   | `reflex.face_means_only` | the 50 dims whose name (`get_reflex_face_feature_names()`) ends in `_mean` | 0.515 [0.471, 0.564] |
   | `reflex.length_plus_face_minus_length` | LR on `[L]` vs LR on `[L, 153 face dims]`, Δ = second − first | −0.069 [−0.103, −0.030] |
   | `reflex.face_equal_length_start` | per episode, K = min(pre length, fail length) of the two reaction windows; window `[rsf, rsf + K − 1]` for both items | 0.525 [0.473, 0.572], n_test 342 |
   | `reflex.face_equal_length_end` | as above, window `[ref − K + 1, ref]` | 0.511 [0.457, 0.561], n_test 343 |
   | `reflex_resolve.react_face` | cached `reflex-face-v1` | 0.552 [0.440, 0.663] |
   | `reflex_resolve.face_before_resolution` | window = `meta.action_frames` | 0.545 [0.461, 0.634], n_test 173 |
   | `reflex_resolve.nonverbal_before_resolution` | speech window = `meta.wall_action_window_sec` | 0.643 [0.480, 0.843], n_test 104 (16 negatives) |
   | `reflex_resolve.step_prior` | `harness.probes.step_prior_scores` on test items with valid face features | 0.941 [0.886, 0.980] |
   | `reflex_resolve.step_prior_plus_face_minus_step_prior` | LR on `[logit(step-prior LOO train), 153 face dims]` vs the step-prior test scores | −0.121 [−0.193, −0.059] |
   | `reflex_resolve.nonverbal_shuffled_null` | the 50-seed permutation null of step 1(b) for `react-nonverbal`. **This key alone has the shape** `{"mean", "sd", "min", "max", "seed0", "true_value", "true_beats"}` | mean 0.510, sd 0.063, min 0.330, max 0.653, seed0 0.603; true 0.660 beats 50 |

3. **`tests/test_report_reflex.py`: add fast tests** (synthetic fixtures; no `DATA_ROOT`):
   - **(falsifying)** `test_resolution_tables_are_computed`: build 9 synthetic `reflex_resolve` items (verbs `pick` 3/3, `carry` 2/3, `place` 1/3 resolved; two nouns). Assert that the generated section 3.3 contains `| pick | 3 | 3 | 100.0 |`, `| carry | 2 | 3 | 66.7 |` and `| place | 1 | 3 | 33.3 |`, and that it contains neither `475` nor `Explain column`. *Against today's code this fails: it prints the hard-coded paragraph.*
   - `test_designer_note_preserved`: write an output file ending in `## Designer note\nkeep me\n`, regenerate, and assert the file ends with `## Designer note\nkeep me\n`.
   - `test_permutation_null_only_for_fail_rows`: rows with one `FAIL` and one `pass` shuffled row. Assert that exactly one null block is printed, for the `FAIL` condition.
4. **Regenerate the report in place:**
   ```bash
   PYTHONPATH=src ./venv/bin/python tools/report_reflex.py --date 2026-11-04
   ```

**Validation (all must pass before committing):**
- `PYTHONPATH=src ./venv/bin/python tools/diag_reflex_posthoc.py` exits 0.
- **Every value in the JSON matches §3's table to 3 decimal places** (CI bounds within ±0.001), with `n_test` exact. **A mismatch is a STOP:** file Issue 13 with both values. Do not edit the table.
- **`git diff docs/evals/2026-11-04_reflex_h1.md` changes only section 3.3.**
  - The designer's note is byte-identical.
  - §1, §2 and §4–§7, including the 20 example descriptions, are unchanged. The two `.tsv` files are byte-identical; check with `git diff --stat`.
- `grep -rn "475 / 478\|13 / 126\|Explain column" tools/ docs/evals/` returns nothing, except the designer's note in the report, which quotes the old error.
- `git diff results/scorecard.jsonl` is empty, and no file under `DATA_ROOT` has an mtime after you started.
- `scripts/battery.sh` bare, with G2 3 times in a row.
- Commit `fix(b4b): compute Wave B report notes from data; reproduce designer post-hoc diagnostics`.

**Blast radius, all in one commit:**
- `tools/report_reflex.py`;
- `tools/diag_reflex_posthoc.py` (new);
- `tests/test_report_reflex.py`;
- `docs/evals/2026-11-04_reflex_h1.md` (regenerated);
- `docs/evals/2026-11-04_reflex_posthoc.json` (new);
- the `docs/ongoing_general_errors.md` §3 line.

---

### Item E1: the paper's fact base, generated tables and figures, and a number checker

**What this means for the user:** every number in the paper comes from one file built from the data, and a checker refuses any number in the prose that is not in it. Two earlier reports (A7, B4) shipped numbers nobody had computed (L10). The paper must not.

**The gap:** there is no `docs/paper/`. The numbers live in scorecard rows, reports and design docs, and nothing checks prose against them.

**Read first:** [`06_writeup.md`](06_writeup.md) §4 and §5. Every key, table and figure is defined there.

**Implementation:**
1. **`tools/build_paper_facts.py`**
   - **CLI:** `PYTHONPATH=src ./venv/bin/python tools/build_paper_facts.py`. It reads `DATA_ROOT` and `results/scorecard.jsonl` but never writes to them.
   - **Writes:**
     - `docs/paper/facts.json`, with exactly two top-level keys, `computed` and `recorded` (`06` §4);
     - the tables `docs/paper/tables/T1.md` … `T6.md`;
     - the figures `docs/paper/figures/f1_ablation.png`, `f2_window_length.png`, `f3_wave_d.png` (`06` §5).
   - **`computed`:**
     - scorecard rows: for each (dataset, condition), the row with the latest `ts`;
     - every key of B4b's `docs/evals/2026-11-04_reflex_posthoc.json`;
     - the dataset and named facts of `06` §4.1 items 3–4;
     - the verdict strings from `prediction_check`, `prediction_check_hidden`, `prediction_check_reflex` and `evaluate_kill_rule`, imported from the existing `tools/report_*.py`;
     - the constants of `06` §4.1 item 6.
   - **`recorded`:** the six entries of `06` §4.2, held in a dict inside the script, each `{"value", "source", "quote"}`.
     - At run time the script asserts that each `quote` occurs **verbatim** in its source file, so a recorded number cannot drift from its document.
     - Copy each quote exactly as it appears in the file today.
2. **`tools/check_paper_numbers.py <draft.md>`**
   - **(a) Tables first.** For every `<!-- table:Tn -->` … `<!-- /table:Tn -->` block, assert the text between the markers equals `docs/paper/tables/Tn.md` byte for byte. Then drop the block.
   - **(b) Then drop** fenced code blocks, inline code spans, link targets `(...)` after `]`, image lines, arXiv identifiers matching `\d{4}\.\d{4,5}`, ISO dates `\d{4}-\d{2}-\d{2}`, and written dates matching `(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(, \d{4})?`.
   - **(c) Extract numeric tokens** with `(?<![\w.§])[+−-]?\d[\d,]*(?:\.\d+)?%?(?![\w])`. A fraction like `219/219` yields both numbers.
     - Exempt only bare four-digit integers from 1990 to 2030 (years).
   - **(d) A token matches** if its absolute value equals the absolute value of some number in `facts.json`, rounded to the token's decimal places.
     - Percent tokens are compared both as written and divided by 100.
     - Commas are removed before comparing.
   - **(e) Output.** Print `checked=<n> matched=<n> unmatched=<n>`. Exit 1 if any token is unmatched, listing each with its line number and the line's text; otherwise exit 0.
3. **`tests/test_paper_tools.py`** (fast, synthetic, no `DATA_ROOT`):
   - **(falsifying)** `test_checker_rejects_invented_number`:
     - facts `{0.7352, 1208, 0.954}`;
     - draft `"On November 1, 2026, window length scored 0.735 on 1,208 items (95.4%); see §11.10 and L18, as in 2020; the face scored 0.736."`;
     - assert exit 1 with exactly one unmatched token, `0.736`.
   - `test_checker_table_drift`: one character changed inside a table block → exit 1.
   - `test_recorded_quote_must_exist`: a recorded entry whose quote is absent from its (temporary) source file → `ValueError`.
   - `test_latest_row_wins`: two scorecard rows for one (dataset, condition) → the facts use the later `ts`.

**Validation:**
- **The builder exits 0** and writes `facts.json`, T1–T6 and F1–F3.
- **Expected values (a mismatch is a STOP; file Issue 13):**
  - `computed` reproduces every value in `06` §4.1 items 3–4 exactly;
  - `scorecard/reflex/react-face/value` rounds to 0.6085 at 4 dp;
  - `scorecard/holoassist_audible/fusion_minus_action_fusion` is 0.021 [0.006, 0.034] at 3 dp;
  - `posthoc/reflex.window_length/value` is 0.735 at 3 dp.
- **T3's verdict strings match** the corresponding lines in the generated reports in `docs/evals/` (`2026-10-10_oops_h1.md`, `2026-10-15_holoassist_h1.md`, `2026-10-21_holoassist_audible_h1.md`, `2026-11-04_reflex_h1.md`).
- **Open each figure and confirm:**
  - **F1:** five intervals; only `holoassist_audible`'s lies entirely right of 0, and none reaches +0.03 with `ci_low > 0`;
  - **F2:** the original-window interval sits above 0.5, both equal-length intervals cover 0.5, and window length alone is the highest point;
  - **F3:** four overlapping curves.
- **The four new tests pass,** and the falsifying one fails against a checker stub that always exits 0.
- **Battery:** `scripts/battery.sh` bare, with G2 3 times in a row.
- **Commit:** `feat(e1): paper fact base, generated tables and figures, number checker`.

**Blast radius, all in one commit:**
- `tools/build_paper_facts.py`, `tools/check_paper_numbers.py`, `tests/test_paper_tools.py`;
- `docs/paper/facts.json`, `docs/paper/tables/T1.md`–`T6.md`, `docs/paper/figures/*.png`;
- the `docs/ongoing_general_errors.md` §3 line.

---

### Item E2: the first full draft of the paper

**What this means for the user:** a complete, reviewable draft in which every number is checked, ready for the decisions only the maintainer can make (title, author line, venue, what to release).

**The gap:** no draft exists.

**Read first:** all of [`06_writeup.md`](06_writeup.md). The claims (§2), structure (§3), citations (§6) and style (§7) are decisions, not suggestions.

**Implementation:**
1. **Write `docs/paper/draft.md`:**
   - **Structure:** the section order and required content of `06` §3, using only claims C1–C5 and never anything on `06` §2's "must not claim" list.
   - **Tables:** paste T1–T6 verbatim between their markers.
   - **Figures:** include F1–F3 with relative paths (`figures/f1_ablation.png`), each with a one-sentence caption that states what the figure shows and its sample sizes.
   - **Author line:** `[AUTHOR NAME], Independent Researcher`, exactly.
   - **References:** a list at the end with only the allowed citations (`06` §6).
     - Use what `01_prior_work.md` gives: authors, venue and year where listed, plus the arXiv identifier.
     - REFLEX is "Naoum et al., HRI 2025, arXiv:2502.14185".
     - Where a field is not in the repository, write `[CITATION NEEDED: full reference]` instead of guessing.
2. **Write `docs/paper/README.md`:** the commands, in order, that rebuild everything the paper cites. These are B4b's diagnostics, E1's builder and checker, and each generator under `tools/report_*.py` with the arguments that produced the dated reports.

**Validation:**
- **The checker passes:** `PYTHONPATH=src ./venv/bin/python tools/check_paper_numbers.py docs/paper/draft.md` exits 0. Put its `checked=… matched=… unmatched=0` line in the commit body.
- **The Wave B check passes:** a scripted check that every paragraph containing `PROCEED` also contains `window` (`06` §2). Also check that no sentence calls Wave B "passed" or "supported".
- **The style check passes:** `grep -niw "novel\|groundbreaking\|remarkabl[ey]"` returns nothing.
- **Citation markers are counted:** list every `[CITATION NEEDED: …]` marker in the commit body.
- **Length:** the main text (sections 1–9) is 5,000–7,000 words by `wc -w` on those sections. Report the number.
- **The rendered draft is checked:** open it and confirm the three figures load and every table renders as a table.
- **Battery:** `scripts/battery.sh` bare.
- **Commit:** `docs(e2): first full draft of the write-up`.

**Blast radius:** `docs/paper/draft.md`, `docs/paper/README.md`, and the `docs/ongoing_general_errors.md` §3 line.

---

## 4. Deferred: do NOT start

**Issue 12 was decided November 5 (A: write up).** Nothing below starts without its trigger.

| Work | Status / trigger | Who writes the spec |
|---|---|---|
| **The maintainer's draft review** (title, author line, venue, release, citations; `06_writeup.md` §8) | After E2 and the designer's verification of it | The maintainer decides; the designer turns it into items |
| **A decisive hidden-outcome test with faces** (DW8; for example CANDOR, Issue 12's option D) | **Maintainer opt-in only.** Any request is drafted by the designer and approved by the maintainer before it is sent | The designer |
| **Wave B v2** (REFLEX with the flaws fixed) | **Not selected** (Issue 12 = A) | — |
| **Wave C: H2 transfer** (DW3) | **Not selected** (Issue 12 = A): no reaction model worth transferring | — |
| **Robot transfer, revisited** (DW7) | **Not selected** (Issue 12 = A); its trigger, a Wave B reaction gain, was not met | — |
| **Stage A live MicroDuck demo** (DW6) | **Paused:** needs a validated reaction reader (the REFLEX face probes read window length, L18), hardware, and the maintainer's opt-in | The designer |
| **Wave D v2** | **Not selected** (Issue 10 = A) | — |
| **Ego4D false-positive set** (DW4) | **Not selected** (it served Wave C) | — |

**The note behind Issue 12, so you do not "fix" it yourself:**
- A `fail` item's reaction window is longer than a `pre` item's (45.7 vs 35.5 frames), and `reflex-face-v1`'s `max`/`std` features grow with length.
- Window length alone scores 0.735. With equal-length windows, `react-face` falls to chance.
- The verdict stays as recorded. The paper reports both facts together (`06_writeup.md` §2).

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
| D1–D4 | `f3f982c`, `d18dbbf`, `03ea7eb`, `315f3a6` | the Wave D v1 pool, transfer code, the RoboReward download and its curve, report and verdict |
| B1 | `f4c9939` | the REFLEX download (`37,219,808,401` bytes, MD5 verified, archive deleted); 1,208 `reflex` and 604 `reflex_resolve` items; the 38 / 17 participant split. The designer recomputed every item's windows: 0 mismatches |
| B2 | `5f0631b` | `reflex-face-v1` (153-d), `reflex-speech-v1` (96-d), the judge questions, and the 20 REFLEX probe conditions. Checked line by line against `05_wave_b_reflex.md` §4–§5 |
| B3 | `0d146dd` | REFLEX features and judges at the coverage claimed (face 97.1% / 97.7%, SigLIP 99.9% / 100%, judges 100% on test) |
| B4 | `dcb44be` | 40 scorecard rows, the exported face probes (reproduce 0.6085), the report, the TSVs, the 20 example descriptions (two checked against the video). **Its section 3.3 text is corrected by B4b** |

### 5.2 Accepted equivalents (do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure.** `extract_frame` retries a seek up to 0.3 s earlier.
- **`GeminiJudge`:** 32 output tokens, thinking off, and a 429 is not cached. `guard()` holds the heavy lock through the model load only.
- **Scorecard rows carry the parent commit's `git_sha` with `dirty: true`.**
- **The HoloAssist group partition is computed over all 1,758 sessions,** and `holoassist_audible` reuses it.
- **`shared/memguard.py` logs `waited_ms = 0` iff admitted on the first check.**
- **`DATA_ROOT/judge/roboreward/gemma4_26b/scores.jsonl`** is a stray duplicate from D3. Leave it.
- **Wave D v1's pool and its overfit true-label head** are part of v1's record (`04` §8), not bugs.
- **B1 normalized `C3-2`'s misspelled header `Resloved?`.**
  - This gives 604 `reflex_resolve` items (488 / 116), not the spec's expected 593.
  - The 11 extra episodes are all `TRUE` and are exactly the ones the spec's count missed.
- **The exported `{dataset}_face_probe.npz` files have no `dataset` key.** The file name carries it.
- **`react-nonverbal|spoke=1` equals `react-nonverbal` on REFLEX by construction:** speech features exist only where the participant spoke.
- **`meta.transcript` is empty and `react-full` did not run,** because the spec named `Transcript` and the column is `Text`. v1 keeps this. Only a v2 changes it.
- **The `reflex_resolve` `react-nonverbal:shuffled` row stays labeled `FAIL`** in the report, with its permutation null beside it. Never relabel a pinned check.

### 5.3 Maintainer decisions

- **October 8–10:**
  - outcome supervision only, with no human rating;
  - "just ask an LLM" is the control;
  - Mac Studio only;
  - commit to `main`;
  - Issue 5 → B.
- **October 20–22:** Issue 6 → A; Issue 7 → A; Issue 8 → A; Issue 9 → B; Issue 10 → A.
- **October 23:** Issue 1 → E (the two requests were sent; M4 bounced and M5 got no reply).
- **November 1:** Wave B moves to **REFLEX** (`00_thesis.md` entry 28). Issues 1 and 4 are resolved by it.
- **November 2:** Issue 11, wait for the external ComfyUI job rather than stop it.
- **November 5:** **Issue 12 → A.** Wave B is recorded as negative in substance. The project moves to the write-up (`06_writeup.md`, items E1–E2). Wave B v2, Wave C (DW3) and DW7 are not selected; DW6 is paused.

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **Grouped splits and CIs. Split files are written once. The scorecard is append-only.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels. The leak check is one-sided.**
- **Existing caches, items, splits and scorecard rows are never modified.** Post-hoc diagnostics never become scorecard rows.

### 5.5 Assessed and rejected: do NOT re-propose

- human rating, rater UIs, golden labels;
- hand-built affect channels;
- self-recorded data, studies, academic partners;
- tuning anything on test results;
- changing a pre-registered rule or a recorded verdict after seeing its result;
- fine-tuning encoders end to end (Mac Studio only);
- circumventing download gates or rate limits.

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, H1–H3, the decision log (entries 1–30) | `00_thesis.md` |
| Wave B: the REFLEX contract (§1–§6), plus the v1 result and the designer's notes (§7) | `05_wave_b_reflex.md` |
| **Wave E: the write-up contract** (claims, structure, fact base, tables, figures, citations, style, the maintainer's decisions) | `06_writeup.md` |
| Scorecard, conditions, judges, guardrails (§11, including §11.10 window-length control and §11.11 action-only inputs), memory guard (§12) | `03_eval_harness.md` |
| Wave D: the contract and the v1 result (§8) | `04_wave_d_robot_check.md` |
| Datasets, including REFLEX "As built" | `02_data_sources.md` |
| Issues 1–12 (all resolved); deferred work DW1–DW8; lessons L1–L19 | `ongoing_general_errors.md` |
| Reports | `docs/evals/` |
| Operations | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard (for every item)

- **Red first, then green, then falsify.** Each item names its falsifying test; run it against today's code before your change and see it fail.
- **Reproduce the stated reference values exactly.** A mismatch is a STOP, never a silent update.
- **A flaky gate is a failing gate.** Run G2 three times with `TMPDIR` on the SSD.
- **Report every missed validation bar in the commit body** (L14).
- **Never loosen a bar or change a pre-registered rule to pass it.** File it as **Issue 13**.

---

## 8. THE LOOP

```
For each item in §2 (A16 -> B4b -> E1 -> E2):
  (1) Read the item in §3, and the sections it names.
  (2) Write the falsifying test first and watch it fail (red); implement (green); falsify each gate.
  (3) Run scripts/battery.sh bare (--slow for A16; G2 3 times with TMPDIR="/Volumes/Extreme SSD/tmp").
      For E1 and E2, also run tools/check_paper_numbers.py as the item says.
  (4) Add the item's line to docs/ongoing_general_errors.md §3, stage explicit paths,
      commit on main, and push with /usr/bin/git push origin main.
```

---

## 9. Definition of Done for this state

- [x] A1–A15, D1–D4 and B1–B4 delivered and verified (§5.1).
- [x] Battery measured bare on November 4 (§1.4), including G4's hang and clean rerun.
- [x] Wave B's verdict is recorded, with the designer's notes beside it (`05` §7; the report's designer note).
- [x] Issue 12 decided (A, November 5): the write-up. Its contract is `06_writeup.md`.
- [ ] **A16:** the `say` fixture has a 60 s timeout and 3 attempts; G4 shows 11 passed; G2 shows 125 passed 3 times in a row.
- [ ] **B4b:** section 3.3 is computed; the posthoc JSON matches §3's table; the designer's note is preserved; nothing outside the blast radius changed.
- [ ] **E1:** `facts.json`, T1–T6 and F1–F3 are built and match the expected values; the checker's falsifying test passes.
- [ ] **E2:** `docs/paper/draft.md` passes the checker with `unmatched=0`, follows `06` §2–§7, and its `[CITATION NEEDED]` markers are listed in the commit body.
- [ ] **Then the queue is empty.** The only legitimate triggers for new work are the designer's verification of E1–E2, the maintainer's draft-review decisions (`06_writeup.md` §8), or a new issue the maintainer raises. **Do not invent work.**
