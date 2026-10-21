# Social Robotics — Agent Execution Guide (Queue Complete: waiting on Issue 1 (Wave B) and the Wave D spec)

**You are an engineering agent with no memory of this project.**
- **What happened:** on October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
- **What exists:** Wave A's first pass has landed (A1–A9, A6b, A6c, A7b, A7c; `651039b` … `3f0b78d`):
  - the harness, the encoders, three judges and the shared memory guard;
  - H1 runs on Oops! and HoloAssist;
  - a summary.

  The designer verified it on October 20 (§1).
- **What you build:** six items, in the §2 order.
  - **A10:** the external SSD is exFAT, so macOS writes hidden `._*` twin files there. Make every glob that touches the SSD ignore them.
  - **A9b:** correct the Wave A summary and the HoloAssist report.
  - **A11:** add the `step-prior` and `action-fusion` controls and the **ablation Δ**, then re-score Oops! and HoloAssist (Issue 7, maintainer decision).
  - **A12:** measure in every HoloAssist session whether the instructor is audible (Issue 6).
  - **A13:** build the audible-session evaluation set `holoassist_audible` as a fresh sample, and run it end to end (Issue 6).
  - **A14:** close out Wave A.

**Read before anything else, in this order:**
1. [`00_thesis.md`](00_thesis.md): H1 and its pass bar (**amended October 20: the ablation gain**), H3's start condition (amended likewise), and decision-log entries 19–20.
2. [`03_eval_harness.md`](03_eval_harness.md): **the contract.**
   - §3 (`holoassist_audible`);
   - §4 (**the ablation Δ row**);
   - §6 (**`step-prior` with its exact formula, and `action-fusion`**);
   - §7 (metrics);
   - §11 (guardrails: §11.2, `react-full` is never a headline; §11.8, the prediction-check rules);
   - §12 (the memory guard).
3. [`02_data_sources.md`](02_data_sources.md) → HoloAssist: "As built (A8)" and **"Audible-session evaluation set: `holoassist_audible`"**.
4. [`ongoing_general_errors.md`](ongoing_general_errors.md): §1, Issues 6 and 7 (decided), and lessons L10–L13.

**The maintainer's words:**
- *"The fundamental core of this project is to use human emotion as a policy for robot learning in addition to the traditional RL stack for robots. We want to add this additional reward signal and investigate if this additional reward signal helped the robot learn a task better and perhaps allow robots to train on much more data like vlogs or any POV youtube video."*
- *"Ideally we should avoid human in the loop ratings because that is not scalable and we want to be able to somehow measure whether our thesis is working often."*
- *"How is this any different from asking an LLM model what is socially appropriate…?"* This is why every H1 result is measured against action-only controls.
- *"No need to open up a new branch, just push to the repo."*
- **October 20, 2026:** Issue 6 → **A**, with a **fresh sample**; Issue 7 → **A**; the pass bar uses **the ablation gain** (`00_thesis.md` entry 20).

**Status:** **Queue Complete: waiting on Issue 1 (Wave B) and the Wave D spec.**

**Every path, number, field name, formula and literal string in this guide and in `03_eval_harness.md` is a decision, not a suggestion. Implement as written; do not substitute your own.**

---

## 0. Standing constraints (apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare after every item (`--slow` when the item ran models), and update §1.4.
   - Read every exit code bare.
   - **After A10, G2 must pass 3 times in a row** with the `TMPDIR` below.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only (no `match`, no runtime `X | Y`).
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies and downloads are frozen.**
   - No new pip packages. No new model downloads; every model needed is cached or installed.
   - Pull no Ollama models. **Download nothing.**
4. **Data placement.**
   - All data lives under `DATA_ROOT` (`/Volumes/Extreme SSD/social_robotics`). **No video or audio on the internal disk.**
   - Never delete anything you did not create.
5. **Long runs (> 30 min) run detached:**
   ```bash
   ./venv/bin/python tools/daemonize.py <log> bash tools/run_supervised.sh <progress.json> <runner…>
   ```
   - Confirm `PPID 1`. Never use your own background-task feature: it is reaped after ~1–2 h.
   - Every long runner is resumable, maintains `progress.json`, and ends with the count line `items_in=<a> items_out=<b> excluded=<c> elapsed_s=<t>`.
6. **Memory (`03_eval_harness.md` §12).**
   - **Never run two of this project's model-loading jobs at once.**
   - Run `PYTHONPATH=src ./venv/bin/python -m shared.memguard --status` before each model job, and record it.
   - **A memory deferral (exit 75) is never a pass.** Never raise `FLOOR`, a declared peak or a wait limit.
   - **`gemma4:26b` is shared with `animated_infographics`; never unload it.** Never stop anything this project did not start.
7. **Determinism.**
   - Use `numpy.random.default_rng(0)` and `torch.manual_seed(0)`, in sorted `item_id` order.
   - **The probe hyperparameters are fixed:** `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")` on a train-only `StandardScaler`. **Nothing is tuned on test.**
8. **No outcome leakage.**
   - The judge never hears audio. Probes never see test labels at fit time.
   - **A shuffled-label control never sees a feature derived from the true labels.** For example, `step-prior` is recomputed from the shuffled labels.
9. **The scorecard is append-only.** A re-score appends a full row set, with `--notes` saying why. Never edit or delete a row.
10. **Honest nulls (L3).** A condition that cannot run is a row with `value: null` and a reason derived from the evidence. A partial run is a `partial` row.
11. **Reports are generated from the rows, clause by clause (L10; `03_eval_harness.md` §11.8).**
    - Verdicts follow the pinned rules only.
    - `react-full` is never a headline. `react-spoke` is a diagnostic.
    - A number that cannot be recounted from a named source is `not recorded`.
12. **Globs over the SSD go through `shared.fsutil.data_glob`** (from A10 onward; L13).
13. **Commits.**
    - One item = one Conventional Commit on `main`, scope = item id (`fix(a10): …`, `feat(a11): …`).
    - The body states the WHY, the red run and the green run.
    - Push with **`/usr/bin/git push origin main`**. No branches, no PRs. Never amend a pushed commit.
    - **Stage by explicit path only.** If `git status` shows a change you did not make, STOP and ask the maintainer.
14. **Record the resolution in the same commit:** one line under "Wave A" in `ongoing_general_errors.md` §3: `A<n> — <title> — git log --grep "(a<n>)" — <measured result>`.
15. **When this guide and a contract doc disagree, or a bar cannot be met, STOP and file it** in `ongoing_general_errors.md` as a new issue (the next number is **Issue 8**), with options. Continue only with items that do not depend on it.
16. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer.

---

## 1. Verified baseline (October 20, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1; `/usr/bin/footprint`.
- **Ollama:** `qwen2.5vl:7b` (`vlm_judge`) and `gemma4:26b` (`vlm_judge_large`, shared).
- **`venv`, Python 3.9.6:** torch 2.8.0, transformers 4.46.3, funasr 1.3.1, scikit-learn 1.6.1, numpy 2.0.2, soundfile, httpx 0.28.1, ruff 0.15.11, pytest 8.4.2.

### 1.2 Storage

- **The Extreme SSD is exFAT**, with about 1.3 TiB free. macOS tags new files with `com.apple.provenance`, so exFAT writes hidden `._<name>` twins (L13).
- **In place:**
  - Oops!: 29,940 videos, items, features, and the caches of both local judges plus Gemini (50 answers);
  - HoloAssist: 2,111 videos (`raw/holoassist/videos/<session>/Export_py/Video_pitchshift.mp4`), items, SigLIP features, and both local-judge caches.
- **Not present:** `e2v-plus-large` features for HoloAssist.

### 1.3 Repository

- `HEAD` includes `3f0b78d` (`feat(a9)`) plus the designer's October 20 docs commits. The working tree is clean.

### 1.4 Gates (run bare October 20, 2026, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result on `3f0b78d` | Result on A10 | Result on A9b | Result on A11 | Result on A12 | Result on A13 | Result on A14 |
|---|---|---|---|---|---|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · clean | exit 0 · clean | exit 0 · clean | exit 0 · clean | exit 0 · clean | exit 0 · clean | exit 0 · clean |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | **Flaky:** 1 failed / 61 passed in 1 run of 3 (`test_gemini_judge_429_exhaustion_raises` globbed `._errors.jsonl`); 62 passed otherwise. **A10's target** | exit 0 · **64 passed** (3/3 runs passed: 35.4s, 35.5s, 35.3s) | exit 0 · **67 passed** (3/3 runs passed: 37.1s, 35.4s, 35.4s) | exit 0 · **71 passed** (3/3 runs passed: 56.7s, 54.2s, 55.3s) | exit 0 · **73 passed** (3/3 runs passed: 54.8s, 54.1s, 53.9s) | exit 0 · **75 passed** (3/3 runs passed: 55.4s, 56.3s, 53.9s) | exit 0 · **75 passed** (3/3 runs passed: 54.8s, 54.1s, 53.6s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS | exit 0 · 4 PASS | exit 0 · 4 PASS | exit 0 · 4 PASS | exit 0 · 4 PASS | exit 0 · 4 PASS | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **9 passed** (134 s); `memguard --status`: 57.0 GB, pressure 1, lock free | exit 0 · **9 passed** (120 s); `memguard --status`: 53.1 GB, pressure 1, lock free | — (fast battery only) | — (fast battery only) | — (fast battery only) | exit 0 · **9 passed** (90 s); `memguard --status`: 51.2 GB, pressure 1, lock free | exit 0 · **9 passed** (69 s); `memguard --status`: 30.1 GB, pressure 1, lock free |

### 1.5 Results so far, and the reference values A11 must reproduce

| Dataset | Condition | AUROC [95% CI] | Source |
|---|---|---|---|
| Oops! | `judge` · `judge-large` · `action-probe` · `react-nonverbal` · `fusion` | 0.472 · 0.708 · 0.779 · 0.711 · 0.793 | scorecard `2026-10-15T06:23:28–30Z` |
| Oops! | `fusion_minus_action_best` | +0.015 [−0.002, 0.029] | scorecard `…06:23:31Z` |
| HoloAssist | `judge` · `judge-large` · `action-probe` · `react-spoke` · `react-full` | 0.488 · 0.544 · 0.569 · 0.676 · 0.739 | scorecard `2026-10-15T20:07:56–58Z` |
| HoloAssist | `react-nonverbal`, `fusion`, Δ | not run (Issue 6) | scorecard |
| **Reference (A11 must match to 3 dp)** | HoloAssist `step-prior` (§6 formula) | **0.751 [0.726, 0.774]** | designer, October 20 |
| **Reference** | HoloAssist `action-fusion` (judge + step-prior + SigLIP; fusion not runnable) | **0.682 [0.658, 0.706]** | designer, October 20 |
| **Reference** | Oops! `action-fusion` (judge + SigLIP; fusion's train items) | **0.779**; ablation Δ **+0.015 [−0.001, 0.029]** | designer, October 20 |

### 1.6 Processes

- None of this project's jobs were running on October 20, and no Ollama model was loaded.

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| A10 | Ignore exFAT `._*` files in every glob over the SSD | G2 is flaky under the required `TMPDIR`, so no later item can trust a green run until it lands. The report tools that A9b and A11 regenerate with use these globs |
| A9b | Correct the Wave A summary and the HoloAssist report | It introduces `prediction_check_hidden`, which A11 and A13 reuse. It fixes statements on `main` that the scorecard does not support |
| A11 | `step-prior`, `action-fusion`, the ablation Δ; re-score Oops! and HoloAssist | Issue 7. It needs only cached features and judges (minutes). A13's audible run depends on this probe code |
| A12 | Per-session audibility over all 1,673 sessions | Issue 6, step 1. About 1–2 h of ffmpeg, with no models. A13's session allowlist comes from its output |
| A13 | Build `holoassist_audible` (a fresh sample) and run it end to end | Issue 6, step 2. It needs A11 (probes) and A12 (allowlist). This is the long run: about 10 h of model time, detached |
| A14 | Close out Wave A | It measures and summarizes everything above, and evaluates Wave D's start condition by the amended rule |

---

## 3. The items

### A10: Ignore exFAT AppleDouble (`._*`) files in every glob over the SSD

**What this means for the maintainer:** a hidden bookkeeping file that macOS writes on the external drive can make a test fail at random. The report tools could also read it as if it were judge data.

**The gap** (read at `3f0b78d`):
- **The flaky test:** `tests/test_judge.py:329` keeps every `*.jsonl` except `errors.jsonl` and `progress.json`. On October 20 it picked up `._errors.jsonl` (4,096 bytes), 1 run in 3.
- **The same exposure in production:**
  - `tools/report_oops.py:133`, `:196` and `:285`;
  - `tools/report_holoassist.py:198`;
  - `tools/download_oops.py:114`, `:115` and `:145`;
  - `tools/download_holoassist.py:93` and `:138`.

**Implementation:**
1. **New `src/shared/fsutil.py`:**
   - `is_appledouble(path) -> bool`: `Path(path).name.startswith("._")`.
   - `data_glob(directory, pattern: str, recursive: bool = False) -> List[Path]`: the sorted `glob`/`rglob` matches, excluding `is_appledouble`.
2. Replace every call site above with `data_glob`, keeping each one's extra filters.
3. Leave the temp-dir cleanup loops at `src/judge/vlm_judge.py:197` and `:312` as they are.
4. **`tests/test_judge.py:329`:** use `data_glob`. The test also **writes its own `._errors.jsonl`** (`b"\x00\x05\x16\x07"` + 4,092 zero bytes) before asserting.

**Validation:**
- **Red first:** a new `tests/test_fsutil.py`.
  - `data_glob` on a dir holding `a.jsonl`, `errors.jsonl` and `._a.jsonl` returns exactly `[a.jsonl, errors.jsonl]`.
  - The Oops! report's judge-diagnostic counts (the function containing `tools/report_oops.py:133`) are unchanged by a planted `._2409b2876016.jsonl`.
- **Falsify:** remove the exclusion → both red → restore.
- **G2 three times in a row** with the SSD `TMPDIR`: all green, recorded.
- G1–G4 green.

**Blast radius:** `src/shared/fsutil.py` (new), the four `tools/` files above, `tests/test_judge.py`, `tests/test_fsutil.py` (new); `ongoing_general_errors.md` §3; this guide's §1.4.

---

### A9b: Correct the Wave A summary and the HoloAssist report

**What this means for the maintainer:** the Wave A summary is how you and the designer judge the thesis. Today it claims that reactions "overcome" the action-only controls on HoloAssist, where no such comparison was ever computed.

**The gap** (read at `3f0b78d`):
- **`docs/evals/2026-10-15_wave_a_summary.md`:**
  - `:13` says "10,000 cluster-bootstrap draws" (it is 1,000 resamples);
  - `:26` is a row with no scorecard row behind it;
  - `:52–55` say `judge-large` "held" and `judge-frontier` "held on sample". By §11.8 both **did not hold**;
  - `:64–66` are an invented HoloAssist clause;
  - `:96` says "zero API errors across 7,708 VLM queries" (there were 10,780 local answers, 2 ffmpeg errors and 1,024 Gemini 429 items);
  - `:108` puts the HoloAssist 7B judge at 56,986.1 s (the cache sums to **40,656.9 s**), and `:112` repeats it in the totals;
  - `:118–119` headline `react-full` and "overcomes", with no Δ;
  - `:126` says the 26B judge "resolved the judge failure";
  - there is no Wave D start-condition section.
- **`tools/report_holoassist.py:17` `prediction_check`:**
  - its clauses are not the thesis's;
  - its pass-bar line (`:83–:101`) takes `max(react-spoke, react-full, react-nonverbal)`, so it names `react-full` 0.739 the "best reaction".

**Implementation:**
1. **`tools/report_holoassist.py`:** replace `prediction_check` with **`prediction_check_hidden(rows: Dict[str, dict]) -> List[str]`**, implementing `03_eval_harness.md` §11.8 (hidden outcome, as amended October 20). It returns exactly these lines, in order:
   - `- **Clause 1, "reaction-only beats action_best":** react-nonverbal <v> [<lo>, <hi>] vs <action_best name> <v> → <held|did not hold|not tested: <react-nonverbal notes>>`. It held iff `react-nonverbal ci_low > action_best's value`. Take the `action_best` name from the `fusion_minus_action_best` row's notes (`action_best=<name>`).
   - `- **Clause 2, "reactions add to the action-only model":** ablation Δ <v> [<lo>, <hi>] → <held|did not hold|not tested: <notes, or "no fusion_minus_action_fusion row">>`. It held iff `ci_low > 0`.
   - `- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → <met|not met|not testable: <which row did not run>>`.
   - `- **Context, no verdict:** fusion − action_best <v> [<lo>, <hi>] or <notes>; judge …; judge-large …; action-probe …; step-prior … (if present); action-fusion … (if present); react-spoke … (diagnostic: did the instructor speak); react-full … (leakage condition: the instructor's words; never a headline, §11.2).`
   - `- **Caveat:** HoloAssist step text carries outcome information (02_data_sources.md → HoloAssist → As built (A8)); judge-large scores 0.489 on test items whose verb is not "drop".`

   Each row reference prints `<value> [<lo>, <hi>]`, or the row's `notes` when the value is null.
2. **Regenerate `docs/evals/2026-10-15_holoassist_h1.md`.** Add as its second line: `*Corrected <YYYY-MM-DD> (A9b): prediction check per 03_eval_harness.md §11.8; react-full is no longer a "best reaction". The numbers are unchanged.*`
3. **Correct the summary.** Add as its second line: `*Corrected <YYYY-MM-DD> (A9b): see the commit body. No scorecard row changed.*`
   - **`:13`** → `95% CIs are percentile intervals over 1,000 group-bootstrap resamples (groups: compilation for Oops!, session prefix for HoloAssist; numpy default_rng(0); single-class resamples redrawn, at most 10,000 draws), per 03_eval_harness.md §7.`
   - **Delete row `:26`.** In the HoloAssist `react-full` row, add to Notes: `leakage condition (the instructor's words); never a headline (§11.2)`.
   - **§2.1:** replace the Clause 1 bullets with the lines that `tools/report_oops.prediction_check` prints for the latest Oops! rows, verbatim.
   - **§2.2:** replace the body with the lines that `prediction_check_hidden` prints for the latest HoloAssist rows, verbatim.
   - **§4 item 3** → *Judge answers:*
     - local: Oops! 2,708 (7B; 0 parse failures; 2 ffmpeg errors on one undecodable train clip) + 1,072 (26B; 0/0); HoloAssist 5,000 (7B; 0/0) + 2,000 (26B; 0/0) = 10,780 answers;
     - frontier (Oops! test only): 50 scored, 1,024 items with HTTP 429 rows, 1 other error;
     - every test-split condition that ran excluded 0 items.

     Recount each figure, with the commands in the commit body.
   - **§4, add item 4:** `HoloAssist step text leaks the outcome: the step verb "drop" is a mistake 95% of the time, and judge-large's 0.544 comes from it (0.489 on the other test items). A step-prior control is added in A11 (Issue 7).`
   - **§5:** recompute every figure from its source:
     - features: the sum of `elapsed_ms` over the last `index.jsonl` entry per item;
     - judges: the sum of `elapsed_ms` per `<prompt_hash>.jsonl`;
     - download: `DOWNLOAD.json`/`download.log`;
     - otherwise `not recorded`.

     The HoloAssist 7B judge is **40,656.9 s**. Recompute the totals.
   - **§6 Observation 1** → `On HoloAssist the action-only controls in this run are weak (action-probe 0.569; judges 0.488 and 0.544) and react-spoke reaches 0.676, but no Δ was computed (Issue 6), and the step text carries outcome information (Issue 7). Whether reactions add information beyond action-only cues on hidden outcomes is untested in this pass; A11–A13 test it.`
   - **§6 Observation 2:** prefix it with `Hypothesis, from the presence check (not tested):`.
   - **§6 Observation 3** → `The 26B judge raised Oops! from 0.472 to 0.708, still below the action-probe (0.779) and short of the "strong" bar; on HoloAssist its 0.544 comes from the verb "drop" (0.489 otherwise).`
   - **§6 Observation 4** → `memguard.log through October 15: 10 shared-model stops (sr_judge_large_load) and 77 unloads, all of our own 7B judge (sr_judge_load); no unload of gemma4:26b; no jetsam report names a process of ours.`
   - **Add §7 "Wave D start condition":** `Not met in this pass. Oops!'s Δ CI includes 0; HoloAssist's Δ was not run. Re-evaluated in A14 by the amended rule (ablation gain, 00_thesis.md entry 20).`
4. **New `tests/test_report_holoassist.py`:**
   - **(a)** the real HoloAssist rows → Clause 1 `not tested`, Clause 2 `not tested`, pass bar `not testable`. No line containing `held` or `met` also contains `react-full`.
   - **(b)** `react-nonverbal 0.80 [0.76, 0.84]`, `action_best = action-probe 0.57`, ablation Δ `0.05 [0.02, 0.08]` → held, held, met.
   - **(c, falsifying)** `react-full 0.90`, `react-nonverbal` null → pass bar `not testable`.

**Validation:**
- **Red first:** (a)–(c) fail against the current generator.
- **The corrected summary:**
  - contains none of `10,000 cluster`, `held on sample`, `7,708`, `56,986`, `overcomes`, `Overcomes`, `resolved the judge failure`;
  - contains `1,000 group-bootstrap` and `Wave D start condition`;
  - has three `did not hold` lines in §2.1.
- **Every number in its tables equals its scorecard row** (a one-off script; its output goes in the commit body).
- `git diff results/scorecard.jsonl` is empty.
- G1–G3 green (G2 three times).

**Blast radius:** `tools/report_holoassist.py`, `tests/test_report_holoassist.py` (new), `docs/evals/2026-10-15_holoassist_h1.md`, `docs/evals/2026-10-15_wave_a_summary.md`; `ongoing_general_errors.md` §3; this guide's §1.4.

---

### A11: The `step-prior` and `action-fusion` controls and the ablation Δ; re-score Oops! and HoloAssist (Issue 7 A)

**What this means for the maintainer:** on HoloAssist, knowing only which step was attempted predicts mistakes better than the video does. This item makes the reactions beat that. It also makes the main comparison measure exactly what the reactions add: the same model with and without them.

**The gap** (read at `3f0b78d`; contract `03_eval_harness.md` §4 and §6, amended October 20):
- **`src/harness/probes.py`** has none of `step-prior`, `action-fusion` or `fusion_minus_action_fusion`.
- **The HoloAssist branch is keyed to one name:** `dataset == "holoassist"` (`:388`), with the else-branch of `if dataset == "oops"` (`:431`, `:449`).
- **Fusion** (`:496–:511`) has no `logit(step-prior)`.
- **`action_candidates`** (`:552–:563`) has no `step-prior` or `action-fusion`.
- **The shuffled controls** (`:592–:647`) have neither.
- **`tools/report_oops.py:17` `prediction_check`'s Clause 2** reads `fusion_minus_action_best`. §11.8 now requires the ablation Δ.

**Implementation** (`src/harness/probes.py` unless stated):
1. **`HOLOASSIST_DATASETS = ("holoassist", "holoassist_audible")`.** Every HoloAssist-only branch tests `dataset in HOLOASSIST_DATASETS`, and Oops! is an explicit `dataset == "oops"` branch (no catch-all `else`).
2. **`step_prior_scores(train_items, target_items, train_labels, loo: bool) -> np.ndarray`**, exactly per §6.
   - The levels are base rate → verb → (verb, noun), with `m = 5`.
   - With `loo=True`, each train item's own label is removed at all three levels. Train features use `loo=True`; test scores use `loo=False`.
   - It returns P(label = 1).
3. **The `step-prior` row** (HoloAssist datasets) uses `auroc_ci` on the test scores, with notes `text-only: train-split P(correct | verb, noun), hierarchical smoothing m=5`.
   - On Oops! write `value: null` with notes `not run: context_text is constant on Oops! (one string for every item)`.
4. **Fusion inputs:**
   - HoloAssist datasets: `[logit(judge), logit(step-prior), action features, react features]`;
   - Oops!: unchanged.
5. **`action-fusion`:** fusion's inputs minus the reaction features.
   - **When fusion can run,** fit it on exactly `fusion_tr_mask` and score it on fusion's test mask.
   - **When fusion cannot run,** fit it on `valid_j & valid_act`.
   - It writes an `action-fusion` row and enters `action_candidates`.
6. **`action_candidates`** = `judge`, `judge-large`, `judge-frontier` (at ≥ 90% coverage), `action-probe`, `step-prior` (HoloAssist datasets) and `action-fusion`.
7. **The ablation row `fusion_minus_action_fusion`:** `delta_auroc_ci(y_test, action_fusion_probs, fusion_probs, groups_test)`, with notes `ablation: reaction features removed`.
   - When fusion did not run: `value: null` and `not run: fusion not run (<fusion notes>)`.
   - Keep `fusion_minus_action_best` exactly as it is.
8. **Shuffled controls:**
   - add `step-prior:shuffled` (scores recomputed from `y_train_shuffled`) and `action-fusion:shuffled`;
   - in `fusion:shuffled` and `action-fusion:shuffled`, the `step-prior` features (LOO train and test) are **recomputed from `y_train_shuffled`**.
9. **`cfg`** gains `"step_prior": {"levels": ["base", "verb", "verb+noun"], "m": 5, "loo_train": true}` and `"ablation": "fusion_minus_action_fusion"`.
10. **`tools/report_oops.py` `prediction_check`:**
    - Clause 2 reads `fusion_minus_action_fusion` (held iff the CI contains 0).
    - Add a context line `- **Context, no verdict:** fusion − action_best <v> [<lo>, <hi>]`.
    - The A7b tests still pass, updated to the new row.
11. **`tools/report_holoassist.py` `prediction_check_hidden`:** replace the static caveat's first clause with the `step-prior` row (`<v> [<lo>, <hi>]`) where it exists.
12. **Re-score both datasets** (minutes; no models):
    ```bash
    PYTHONPATH=src ./venv/bin/python -m harness.probes --dataset oops --notes "rerun: action-fusion ablation added (Issue 7 A; 00_thesis entry 20)"
    PYTHONPATH=src ./venv/bin/python -m harness.probes --dataset holoassist --notes "rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20)"
    ```
13. **Regenerate both reports.** Add the line `*Updated <YYYY-MM-DD> (A11): step-prior, action-fusion and the ablation Δ added (Issue 7 A).*`
14. **Move Issue 7 to 🧪 Resolved** in `ongoing_general_errors.md`, with the measured rows.

**Validation:**
- **Red first:** new tests in `tests/test_probes.py`, which fail on the current code.
  - **(a) The formula:** a 6-item synthetic train set with hand-computed `p0`, `p_v` and `p_vn` (put the arithmetic in the test docstring). Scores match to 1e-12, and the LOO train feature of an item differs from its non-LOO value exactly as hand-computed.
  - **(b) Shuffled controls see no true labels (falsifying):** a synthetic dataset where the verb fully determines the label. `step-prior` AUROC is 1.0, but `step-prior:shuffled` and `fusion:shuffled` have CIs containing 0.5. Temporarily compute the shuffled `step-prior` from the true labels → (b) goes red → revert.
  - **(c)** `action-fusion` is fit on exactly `fusion_tr_mask` when fusion runs: assert the train index sets are equal.
  - **(d)** The Oops! branch writes the `step-prior` `not run` row. The `holoassist_audible` name takes the HoloAssist branch.
- **Reproduce the designer's reference values (§1.5) to 3 dp from the new rows:**
  - HoloAssist `step-prior` **0.751 [0.726, 0.774]**;
  - HoloAssist `action-fusion` **0.682 [0.658, 0.706]**;
  - Oops! `action-fusion` **0.779**, and Oops! `fusion_minus_action_fusion` **+0.015 [−0.001, 0.029]**.

  A mismatch means the formula or the masks differ from §6: **find out why before going on.**
- **The re-score changed nothing else:** every previously existing condition in the new row sets equals its October 15 value to 4 dp (`judge`, `action-probe`, `fusion`, the shuffled controls, …).
- Every `:shuffled` CI contains 0.5.
- G1–G3 green.

**Blast radius:** `src/harness/probes.py`, `tools/report_oops.py`, `tools/report_holoassist.py`, `tests/test_probes.py`, `tests/test_report_oops.py`, `tests/test_report_holoassist.py`, `results/scorecard.jsonl` (appended), `docs/evals/2026-10-10_oops_h1.md`, `docs/evals/2026-10-15_holoassist_h1.md`; `ongoing_general_errors.md` (§3; Issue 7 → Resolved); this guide's §1.4.

---

### A12: Per-session instructor audibility over every HoloAssist session (Issue 6 A, step 1)

**What this means for the maintainer:** it finds the HoloAssist sessions where the instructor's voice can actually be heard over the work noise. Only those can test whether the voice carries information.

**The gap** (read at `3f0b78d`): `tools/check_audio_presence.py` measures only a 20-session random sample (`run_audio_presence_check`, `:184`; the candidate filter at `:194–:206`). Nothing records audibility per session for all sessions.

**Implementation:**
1. **Add `--all-sessions` to `tools/check_audio_presence.py`.** For every annotated session whose video exists (sorted by `video_name`), write one JSON line to `DATA_ROOT/runs/holoassist_session_audibility.jsonl`:
   ```json
   {"video_name": "...", "task": "...", "diff_db": 4.53, "audible": true,
    "n_instructor_spans": 16, "n_silent_gaps": 6, "reason": null}
   ```
   - Use the existing `measure_session_audio_presence` (`:152`).
   - `audible` = `diff_db ≥ 3.5`, as the constant **`AUDIBLE_DB = 3.5`**.
   - A session with no instructor utterance, or no no-utterance gap of ≥ 2 s, gets `diff_db: null`, `audible: null` and `reason`: `"no instructor utterance"` or `"no silent gap >= 2 s"`.
   - A decode error gets `diff_db: null`, `audible: null` and `reason: "error: <first 120 chars>"`.
2. **Resumable:** keep `DATA_ROOT/runs/holoassist_session_audibility.progress.json` (the list of finished `video_name`s, rewritten atomically), append each line with fsync, skip finished sessions, and use `data_glob` for any glob (A10).
3. **End with** the count line plus `audible=<a> inaudible=<i> undefined=<u>`.
4. **Run it detached** (§0.5). It is ffmpeg only, with no models, about 1–2 h.
5. **Record in `02_data_sources.md`** → HoloAssist → "Audible-session evaluation set" a subsection **"Audibility (as measured, <date>)"**: the counts audible/inaudible/undefined, overall and by `task`.

**Validation:**
- **Red first:** a new test in `tests/test_audio_presence.py`.
  - A synthetic session with no instructor utterance → `audible: null`, `reason: "no instructor utterance"`.
  - A synthetic +10 dB session → `audible: true`; a flat session → `audible: false`.
- **Determinism against A8:** the 20 sessions in `DATA_ROOT/runs/holoassist_audio_presence.json` reproduce their `diff_db` to **0.01 dB**. Six of them are audible (`z129`, `z094`, `z195`, `z167`, `z124 marius`, `z035`).
- **Every annotated session with video appears exactly once:** 1,673 lines, unique `video_name`s.
- G1–G3 green.

**Blast radius:** `tools/check_audio_presence.py`, `tests/test_audio_presence.py`, `docs/02_data_sources.md`; `ongoing_general_errors.md` §3; this guide's §1.4.

---

### A13: The audible-session evaluation set `holoassist_audible`, a fresh sample, run end to end (Issue 6 A, step 2)

**What this means for the maintainer:** this is Wave A's real hidden-outcome test of the thesis's channel. On sessions where the instructor can be heard, does their non-verbal voice tell us about the performer's mistake beyond what the action, the step name and a judge already tell us?

**Start condition:** A11 and A12 are pushed, and `holoassist_session_audibility.jsonl` has 1,673 lines.

**The gap** (read at `3f0b78d`; contract `02_data_sources.md` → "Audible-session evaluation set"):
- **`src/sources/holoassist.py` is single-dataset:**
  - `build_holoassist_items` (`:181`) hard-codes `dataset="holoassist"` (`:285`) and the `holoassist:` id prefix;
  - `main` writes `items/holoassist/` (`:427`) and `splits/holoassist.json` (`:440`);
  - there is no session allowlist.
- **`src/judge/vlm_judge.py:33` `DATASET_QUESTIONS`** has no `holoassist_audible` key.
- **`tools/report_holoassist.py` hard-codes `holoassist`** at `:169`, `:193`, `:290`, `:317–:318`, `:349–:351`, `:502` and `:513`.
- **`tools/pipeline_holoassist.sh` waits on a hard-coded PID (`78507`).** Do not copy that.

**Implementation:**
1. **`build_holoassist_items(…, dataset: str = "holoassist", session_allowlist: Optional[Set[str]] = None)`:**
   - `Item.dataset = dataset`, and `item_id = f"{dataset}:{vname}:{idx}"`;
   - with an allowlist, also set `meta.source_item_id = f"holoassist:{vname}:{idx}"`;
   - sessions outside the allowlist are skipped with the reason `not_audible` (the actions counted in `stats.json`);
   - **the group partition stays computed over all 1,758 annotated sessions**, unchanged, so it is identical to `holoassist`'s;
   - sampling is unchanged (all mistakes plus an equal number of correct items, caps 1,500/1,000 per class, `default_rng(0)`), applied to the allowed candidates.
2. **CLI:** `python -m sources.holoassist --build --dataset holoassist_audible --audible-from "$DATA_ROOT/runs/holoassist_session_audibility.jsonl"`.
   - The allowlist is the sessions with `audible: true`.
   - It writes `items/holoassist_audible/{items.jsonl,stats.json}` and `splits/holoassist_audible.json`. This is a **first write: no `--force-split`.**
3. **`DATASET_QUESTIONS["holoassist_audible"] = DATASET_QUESTIONS["holoassist"]`** (the same string, so the same `prompt_hash`).
4. **`tools/report_holoassist.py --dataset <name>`** (default `holoassist`) drives every path, and writes `docs/evals/<YYYY-MM-DD>_<dataset>_h1.md`.
   - For `holoassist_audible`, add a section **"The subset"**: the audible-session rule, the session and item counts, and the task mix next to `holoassist`'s (from both `stats.json` files).
5. **New `tools/pipeline_holoassist_audible.sh`** (`set -euo pipefail`, no PIDs) runs these in sequence, each under `run_supervised.sh` with its own progress file:
   1. `python -m shared.memguard --status`;
   2. `features.extract --dataset holoassist_audible --encoder siglip-b16-224`;
   3. `--encoder e2v-plus-large`;
   4. `judge.vlm_judge --dataset holoassist_audible --split train --backend ollama`;
   5. `--split test --backend ollama`;
   6. `--split test --backend ollama-large`;
   7. `harness.probes --dataset holoassist_audible`;
   8. `tools/report_holoassist.py --dataset holoassist_audible`.

   Launch it detached (§0.5). Expect about 10 h, of which about 8 h is the 7B judge at ~8 s per item.
6. **Write the example TSV** (10 items: 5 correct, 5 mistakes; `default_rng(0)`), with sentences you write after looking at each item's 4 judge frames. **Describe only what is visible; do not use the label or the transcript.** Then regenerate the report.
7. **Move Issue 6 to 🧪 Resolved** with the measured rows.

**Validation:**
- **Before the long run** (fast, after the build):
  - `validate_items` returns no errors;
  - every `item_id` starts with `holoassist_audible:`;
  - **every item's session has `audible: true`** in the jsonl. Falsify: add a non-audible session to the allowlist in a test fixture → the builder test fails;
  - **every train group of `holoassist_audible` is a train group of `holoassist`, and every test group a test group;**
  - `straddling_groups` = 0;
  - the leakage regex finds nothing in `context_text`.
- **The sanity range** (`02_data_sources.md`): 3,000–4,000 items, 1,000–1,300 of them test. **If there are fewer than 400 test items or fewer than 20 test groups, STOP and file Issue 8** with the counts. The maintainer chose a fresh sample to get a tight CI.
- **After the run:**
  - every judge covers ≥ 90% of its items;
  - `e2v-plus-large` and `siglip-b16-224` exclusions are ≤ 5%, with reasons;
  - every §6 condition has a row, including `step-prior`, `action-fusion`, `fusion_minus_action_fusion`, `fusion_minus_action_best` and `react-nonverbal|spoke=1`;
  - **every `:shuffled` CI contains 0.5.** If any `ci_low` is > 0.5, STOP and find the leak;
  - the report prints `prediction_check_hidden`'s lines, and `react-full` carries no verdict.
- G1–G4 green.

**Blast radius:** `src/sources/holoassist.py`, `src/judge/vlm_judge.py`, `tools/report_holoassist.py`, `tools/pipeline_holoassist_audible.sh` (new), `tests/test_sources_holoassist.py`, `splits/holoassist_audible.json` (new), `results/scorecard.jsonl` (appended), `docs/evals/<date>_holoassist_audible_h1.md` and its TSV; `ongoing_general_errors.md` (§3; Issue 6 → Resolved); this guide's §1.4.

---

### A14: Close out Wave A

**What this means for the maintainer:** one page that says, by the amended pass bar, whether reactions added information beyond the action-only model on any Wave A dataset, and whether the robot check (Wave D) may start.

**Implementation:**
1. Run `scripts/battery.sh --slow` bare (G2 three times) and update §1.4.
2. **Append a dated section to `docs/evals/2026-10-15_wave_a_summary.md`:** **"Completion (<YYYY-MM-DD>): step-prior, ablation gain, audible-session HoloAssist"**, containing:
   - **the headline table:** dataset × condition for `oops`, `holoassist` and `holoassist_audible`, with value, CI, n, `n_excluded` and the row `ts`, using the latest rows;
   - **the prediction-check lines,** verbatim from `prediction_check` (Oops!) and `prediction_check_hidden` (`holoassist`, `holoassist_audible`);
   - **the subset caveat** (quiet device tasks over-represented; the rule chosen after the 20-session check failed, but before any outcome was seen), the pitch-shift caveat and the group (pair, not person) caveat;
   - **"Wave D start condition":** **met** iff some dataset's `fusion_minus_action_fusion` has `ci_low > 0`; name the dataset and the row `ts`. Otherwise **not met**;
   - compute time per stage for A13, recounted from the caches.
3. Make sure each item has its §3 line, and that Issues 6 and 7 are under 🧪 Resolved.
4. **Rewrite this guide's title and status to "Queue Complete: waiting on Issue 1 (Wave B)",** adding **"and the Wave D spec"** if the start condition is met. **Then stop. Do not invent work.**

**Validation:** every number in the new section matches a scorecard row (cite its `ts`); G1–G4 green.

---

## 4. Deferred: do NOT start

| Work | Trigger | Who writes the spec |
|---|---|---|
| **Wave B: verdict-video H1 pilot**, DW1 | **Issue 1** selected **and** A14 done **and** a Wave B spec | The designer |
| **Wave D: offline robot check (RoboReward)**, DW5 | A14 finds a CI-excluding-0 ablation gain on ≥ 1 dataset (`00_thesis.md` H3, amended) **and** a Wave D spec | The designer |
| **Face encoder**, DW2 · **H2 transfer**, DW3 · **Ego4D false-positive set**, DW4 | Wave B | The designer |
| **Stage A live (microduck)**, DW6 | Hardware, H2 done, and the maintainer opts in | The designer |

Also not to do:
- a `fusion-spoke` row (Issue 6 option B was not selected);
- any change to the fusion combiner (the ablation was chosen instead);
- paid frontier-judge runs;
- any judge prompt change;
- any download;
- anything from `v0-saf-final`.

---

## 5. Do NOT change

### 5.1 Already delivered (verified by the designer)

| Item | Commit | What it delivered |
|---|---|---|
| A1–A6 | `651039b`–`0f16799` | the battery; metrics and scorecard; items and splits; HoloAssist labels and independence; encoders and cache; the judges |
| A6b + A6c | `76e71cc`, `8c6f04e` | the memory guard: re-entrant `guard()`, fail-closed reads, supervisor accounting, `check_shared`, fast tests isolated from live HTTP |
| A7 + A7b | `6d72492`, `d77584c` | the Oops! pipeline and its generated report |
| A7c | `1346e0b` | `judge-large` (`gemma4:26b`, declared peak 33 GB) |
| A8 | `36a7769` | the HoloAssist pipeline (`holoassist`: 3,000/2,000; partitioned over all sessions); the audio check (Issue 6) |
| A9 | `3f0b78d` | the Wave A summary (A9b corrects it; A14 completes it) |

### 5.2 Accepted equivalents (do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure.**
- **`extract_frame` retries a seek up to 0.3 s earlier** at a clip's end.
- **`GeminiJudge`:** `max_output_tokens=32`, `thinking_budget=0`, and a 429 is not cached after 5 tries.
- **`guard()` holds the heavy lock through the model load only.**
- **Scorecard rows carry the parent commit's `git_sha` with `dirty: true`.** The code of record is the item's own commit.
- **HoloAssist groups are partitioned over all 1,758 annotated sessions** (238/102). `holoassist_audible` reuses that partition.
- **The 7B judge checks `/api/ps` before every call.**
- **`download_holoassist.py` checked for 50 GiB free before the download, not after extraction.** The rule held.

### 5.3 Maintainer decisions

- **October 8:**
  - outcome supervision only, with no human rating;
  - "just ask an LLM" is the control;
  - hidden-outcome data first;
  - Mac Studio only;
  - commit to `main`.
- **October 9:** no self-recorded data, studies or academic partner; Wave D in scope.
- **October 10:** Issue 5 → B (a larger local judge).
- **October 20:**
  - **Issue 6 → A, with a fresh sample;**
  - **Issue 7 → A;**
  - **the pass bar and Wave D's trigger use the ablation gain** (fusion − `action-fusion`).

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **The pass-bar Δ is the ablation gain.** `fusion_minus_action_best` is always reported beside it.
- **A judge enters `action_best` only at ≥ 90% coverage.** `step-prior` and `action-fusion` are action-only controls.
- **Grouped splits and grouped CIs only. Split files are written once.**
- **The scorecard is append-only. Nulls carry reasons.**
- **`react-full` is never a headline.**
- **`gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels, directly or through a derived feature.**

### 5.5 Assessed and rejected: do NOT re-propose

- human rating, rater UIs, golden labels;
- hand-built affect channels;
- self-recorded data, studies, academic partners;
- Ego4D bystander footage as H1 data;
- tuning prompts, hyperparameters, windows, caps, subsets or the combiner on test results;
- circumventing bot checks;
- changing `AUDIBLE_DB`, the sampling caps or the step-prior's `m` after seeing results.

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, H1 pass bar (ablation gain), H3 trigger, the decision log (entries 19–20) | `00_thesis.md` |
| Items, splits, the scorecard (§4, incl. the ablation Δ), conditions (§6, incl. `step-prior`, `action-fusion`), metrics (§7), judges (§8), guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| `holoassist_audible`'s definition; the HoloAssist and Oops! as-built caveats | `02_data_sources.md` |
| Issues 1 and 4 (open), 6 and 7 (decided), 2, 3, 5 (resolved); lessons L1–L13 | `ongoing_general_errors.md` |
| Operations (detached runs, ollama, decoding, storage) | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard

- **Red first.** Show each new test failing against the current code.
- **Every gate must be able to fail.** Break, see red, restore, see green.
- **Reproduce the designer's reference values** (§1.5). A mismatch is a finding, never a rounding question.
- **A flaky gate is a failing gate.** Run G2 three times.
- **Report text is checked against the rows.** Verdicts follow §11.8 only.
- **Never loosen a bar to pass it.** File it, with the measurement and options.

---

## 8. THE LOOP

```
(1) Next approved item, in §2 order: A10, A9b, A11, A12, A13, A14. If all
    are done, STOP (A14's close-out). Never start §4 work. Never fill in a
    `Your selection:` line.
(2) Read the item and every contract section it names.
(3) RED FIRST: run its new tests; record the failure.
(4) Build only what the item says.
(5) GREEN; then falsify (break, see red, restore, see green).
(6) Open every artefact you produced and read it.
(7) scripts/battery.sh bare, with TMPDIR on the SSD (G2 three times;
    --slow if the item ran models). Update §1.4.
(8) ONE commit on main, scope = item id, explicit paths only. WHY +
    red/green in the body. ONE line under "Wave A" in
    ongoing_general_errors.md §3.
(9) /usr/bin/git push origin main
```

---

## 9. Definition of Done: Wave A completion

- [ ] A10, A9b, A11, A12, A13 and A14 each landed as one pushed commit, with red and green runs recorded.
- [ ] No glob over the SSD returns a `._*` file. G2 passed 3 times in a row with `TMPDIR` on the SSD.
- [ ] The summary and the reports carry only §11.8 verdicts, and `react-full` is never a headline.
- [ ] `step-prior`, `action-fusion` and the ablation Δ reproduce the §1.5 reference values. The shuffled controls (including `step-prior:shuffled`) sit at chance.
- [ ] Every annotated session with video has an audibility line.
- [ ] `holoassist_audible` is built as a fresh sample, with the shared partition, and run end to end; every §6 row exists.
- [ ] Issues 6 and 7 are under 🧪 Resolved, with their measured rows.
- [ ] The Wave D start condition is evaluated by the ablation rule and stated.
- [ ] This guide is rewritten to **Queue Complete** (naming what it waits on). **Then stop. Do not invent work.**
