# Agent Execution Guide: Active Build: Wave A corrections (A10 → A9b), October 20, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:** on October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
- **What exists: all of Wave A has landed** (A1–A9, A6b, A6c, A7b, A7c; `651039b` … `3f0b78d`, October 9–15):
  - the harness, the encoders, three judges and the shared memory guard;
  - H1 runs on Oops! and HoloAssist;
  - the close-out summary.

  The designer verified it on October 20 by reading each item's code and outputs against its contract and re-running the battery (§1).
- **What you build:** two small corrections, in the §2 order:
  - **A10:** make every glob that touches the external SSD ignore macOS's hidden `._*` files. Today they make a test fail at random, and they could be read as data;
  - **A9b:** correct the Wave A summary and the HoloAssist report, which state things the scorecard does not support.
- **Then stop.** The rest of Wave A's HoloAssist analysis waits on the maintainer's **Issues 6 and 7** (§4).

**Read before anything else, in this order:**
1. [`00_thesis.md`](00_thesis.md): H1, the pass and kill criteria, and the decision log. Entry 19 is the state of Wave A's results.
2. [`ongoing_general_errors.md`](ongoing_general_errors.md):
   - §1, the October 20 verification;
   - Issues 6 and 7, which are not yours to decide;
   - lessons L10–L13.
3. [`03_eval_harness.md`](03_eval_harness.md) §7 (metrics) and §11 (guardrails): §11.2, `react-full` is never reported alone; **§11.8, the prediction-check rules A9b implements**.
4. [`02_data_sources.md`](02_data_sources.md) → HoloAssist → "As built (A8)".

**The maintainer's words:**
- *"The fundamental core of this project is to use human emotion as a policy for robot learning in addition to the traditional RL stack for robots. We want to add this additional reward signal and investigate if this additional reward signal helped the robot learn a task better and perhaps allow robots to train on much more data like vlogs or any POV youtube video."*
- *"Ideally we should avoid human in the loop ratings because that is not scalable and we want to be able to somehow measure whether our thesis is working often."*
- *"How is this any different from asking an LLM model what is socially appropriate…?"* This is why every H1 result is measured against action-only controls, and why a summary that overstates a result is a defect, not a style issue.
- *"No need to open up a new branch, just push to the repo."*

**Status:** **Active Build: Wave A corrections.** Next is **A10**.
- **Open maintainer decisions:**
  - **Issue 6:** which reaction channel HoloAssist's Δ uses;
  - **Issue 7:** what action-only control the reactions must beat on HoloAssist;
  - **Issue 1:** Wave B's data source.
- **None of them blocks A10 or A9b.**

**Every path, number, field name and literal string in this guide is a decision, not a suggestion. Implement as written; do not substitute your own.**

---

## 0. Standing constraints (apply to every item)

1. **The battery is the regression bar.** Run `scripts/battery.sh` bare after every item and update §1.4. Read every exit code bare.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only.
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
   - **Run the battery with this `TMPDIR`.** It is the environment A10 must make reliable.
3. **No new dependencies, no downloads, no model runs.** Neither item loads a model or touches video, except G4 in the battery.
4. **Never delete anything you did not create.** No video or audio on the internal disk.
5. **Never edit or delete a row in `results/scorecard.jsonl`.** Neither item writes rows.
6. **Reports are generated from the rows, clause by clause (L10).**
   - A sentence that states a result is computed from the scorecard row it describes.
   - **`react-full` is never a headline or a "best reaction"** (§11.2). `react-spoke` is a diagnostic.
   - A number that cannot be recounted from a named source is written `not recorded`.
7. **Commits.**
   - One item = one Conventional Commit on `main`, scope = item id (`fix(a10): …`, `fix(a9b): …`).
   - The body states the WHY, the red run and the green run.
   - Push with **`/usr/bin/git push origin main`**. No branches, no PRs. Never amend a pushed commit.
   - **Stage by explicit path only.** If `git status` shows a change you did not make, STOP and ask the maintainer.
8. **Record the resolution in the same commit:** one line under "Wave A" in `ongoing_general_errors.md` §3: `A<n> — <title> — git log --grep "(a<n>)" — <measured result>`.
9. **When this guide and a contract doc disagree, STOP and file it** in `ongoing_general_errors.md` as a new issue (the next number is **Issue 8**), with options.
10. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer.
11. **Memory.** Run `PYTHONPATH=src ./venv/bin/python -m shared.memguard --status` before G4, and record it. Never unload `gemma4:26b` or anything this project did not start.

---

## 1. Verified baseline (October 20, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1; `/usr/bin/footprint`.
- **Ollama:** `qwen2.5vl:7b` (`vlm_judge`) and `gemma4:26b` (`vlm_judge_large`, shared with `animated_infographics`). No model was loaded at verification.
- **`venv`, Python 3.9.6:** torch 2.8.0, transformers 4.46.3, funasr 1.3.1, scikit-learn 1.6.1, numpy 2.0.2, google-genai 1.47.0, httpx 0.28.1, ruff 0.15.11, pytest 8.4.2.

### 1.2 Storage

- **The Extreme SSD is exFAT.** 1.8 TiB total, about 1.3 TiB free.
  - macOS adds a `com.apple.provenance` attribute to new files, so exFAT writes hidden `._<name>` twins: `DATA_ROOT/runs/._memguard.log` exists, and pytest temp dirs on the SSD hold dozens.
  - There are none today under `judge/`, `features/` or `items/`.
- **Data in place:**
  - Oops!: 29,940 videos, items, features and both local-judge caches;
  - HoloAssist: 2,111 videos, items, SigLIP features and both local-judge caches;
  - no `e2v-plus-large` features for HoloAssist (Issue 6).

### 1.3 Repository

- `HEAD` = `3f0b78d` (`feat(a9)`), pushed. The working tree is clean.
- The designer's October 20 docs commit follows it.

### 1.4 Gates (run bare October 20, 2026, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result on `3f0b78d` |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · clean |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | **Flaky:** 1 failed / 61 passed in 1 of 3 runs (`tests/test_judge.py::test_gemini_judge_429_exhaustion_raises`, which globbed a 4,096-byte `._errors.jsonl`); 62 passed in the other 2 runs and without the SSD `TMPDIR`. **A10's target** |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **9 passed** (134 s); `memguard --status`: 57.0 GB available, pressure 1, lock free |

### 1.5 What the results are (scorecard rows; do not recompute them)

| Dataset | Condition | AUROC [95% CI] | Row `ts` |
|---|---|---|---|
| Oops! | `judge` / `judge-large` / `judge-frontier` (50 of 1,072) | 0.472 [0.440, 0.498] / 0.708 [0.684, 0.731] / 0.704 [0.544, 0.821] | `2026-10-15T06:23:28Z` |
| Oops! | `action-probe` · `react-nonverbal` · `fusion` | 0.779 · 0.711 · 0.793 | `…06:23:29–30Z` |
| Oops! | `fusion_minus_action_best` (action_best = action-probe) | **+0.015 [−0.002, 0.029]** | `…06:23:31Z` |
| HoloAssist | `judge` / `judge-large` | 0.488 [0.466, 0.512] / 0.544 [0.528, 0.562] | `2026-10-15T20:07:56Z` |
| HoloAssist | `action-probe` · `react-spoke` · `react-full` | 0.569 · 0.676 · 0.739 | `…20:07:57–58Z` |
| HoloAssist | `react-nonverbal`, `fusion`, Δ | **not run** (Issue 6) | `…20:07:57–58Z` |

**Designer measurements, not in the scorecard** (`02_data_sources.md` → HoloAssist → As built (A8)):
- a train-split step-verb prior scores **0.721** on the HoloAssist test split;
- `judge-large` scores **0.489** on the test items whose verb is not `drop`.

### 1.6 Processes

- None of this project's jobs were running on October 20. The Gemini job has ended; its Oops! cache holds 50 answers.

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| A10 | Ignore exFAT `._*` files in every glob over the SSD | G2 is flaky under the required `TMPDIR`, so no later item can trust a green run until this lands. A9b also regenerates a report through one of the affected globs (`tools/report_holoassist.py:198`) |
| A9b | Correct the Wave A summary and the HoloAssist report | It is the document the designer specs Wave B from, and the one the maintainer reads to judge the thesis. It needs a reliable battery (A10) |

---

## 3. The items

### A10: Ignore exFAT AppleDouble (`._*`) files in every glob over the SSD

**What this means for the maintainer:** a hidden bookkeeping file that macOS writes on the external drive can make a test fail at random. The same file could also be read by the report tools as if it were judge data.

**The gap** (read at `3f0b78d`):
- **Fast-suite flake:** `tests/test_judge.py:329` keeps every `*.jsonl` except `errors.jsonl` and `progress.json`. On October 20 it picked up `._errors.jsonl` (4,096 bytes) and failed `assert cf.stat().st_size == 0`. That was 1 run in 3 with `TMPDIR` on the SSD.
- **Production globs with the same exposure:**
  - `tools/report_oops.py:133`, `:196` and `:285` (judge caches);
  - `tools/report_holoassist.py:198` (judge caches);
  - `tools/download_oops.py:114`, `:115` and `:145`, and `tools/download_holoassist.py:93` and `:138` (the verification counts of `.mp4`/`.json`).
- **Why it happens:** the SSD is exFAT. macOS tags new files with `com.apple.provenance`, so a hidden `._<name>` twin appears beside some files, at times we cannot predict.

**Implementation:**
1. **New `src/shared/fsutil.py`** (Python 3.9):
   ```python
   def is_appledouble(path) -> bool: ...   # Path(path).name.startswith("._")
   def data_glob(directory, pattern: str, recursive: bool = False) -> List[Path]: ...
   # sorted matches of Path(directory).glob(pattern) (rglob if recursive), excluding is_appledouble
   ```
2. **Replace every call site listed in the gap with `data_glob`.** Keep each one's existing extra filters, such as `!= "errors.jsonl"`. The `tools/` scripts already run with `PYTHONPATH=src`.
3. **Leave the temp-dir cleanup loops in `src/judge/vlm_judge.py:197` and `:312` alone.** They should delete `._*` files too.
4. **`tests/test_judge.py:329`:** use `data_glob`. The test must also **write a `._errors.jsonl` itself** (the bytes `b"\x00\x05\x16\x07"` + 4,092 zero bytes, the AppleDouble header) before asserting. That makes the condition deterministic instead of depending on macOS.

**Validation:**
- **Red first:** a new `tests/test_fsutil.py`.
  - It builds a dir holding `a.jsonl`, `errors.jsonl` and `._a.jsonl` (the AppleDouble bytes above), and asserts that `data_glob(dir, "*.jsonl")` returns exactly `[a.jsonl, errors.jsonl]`. It fails before `fsutil` exists.
  - It also puts a `._2409b2876016.jsonl` beside a 3-line judge cache in `tmp_path` and asserts that the Oops! report's judge-diagnostic counts (the function containing `tools/report_oops.py:133`) equal the counts without it.
- **Falsify:** remove the `is_appledouble` exclusion → both tests red → restore → green.
- **The modified 429 test passes** with its planted `._errors.jsonl`.
- **The flake is gone:** run G2 **three times in a row** with `TMPDIR="/Volumes/Extreme SSD/tmp"`. All 3 are green; record all 3 in the body.
- G1–G4 green.

**Blast radius:** `src/shared/fsutil.py` (new), `tools/report_oops.py`, `tools/report_holoassist.py`, `tools/download_oops.py`, `tools/download_holoassist.py`, `tests/test_judge.py`, `tests/test_fsutil.py` (new); `ongoing_general_errors.md` §3; this guide's §1.4.

---

### A9b: Correct the Wave A summary and the HoloAssist report

**What this means for the maintainer:** the one-page Wave A summary is how you and the designer judge the thesis and plan Wave B. Today it says things the scorecard does not support. The worst is that "reaction signal overcomes action probes" on HoloAssist, when no HoloAssist Δ was ever computed and a step-name prior beats every action-only control.

**The gap** (read at `3f0b78d`):
- **`docs/evals/2026-10-15_wave_a_summary.md`:**
  - **`:13`:** "10,000 cluster-bootstrap draws". The code draws **1,000** group resamples; 10,000 is only the draw ceiling (`03_eval_harness.md` §7).
  - **`:26`:** a row for an Oops! `react-nonverbal|spoke=1` condition that has no scorecard row.
  - **`:52–55`:** `judge-large` "held" and `judge-frontier` "held on sample". By the pinned rule (`ci_low > 0.5` **and** AUROC ≥ action-probe − 0.05 = 0.729) **both did not hold**, which is what the generated Oops! report says.
  - **`:64–66`:** an invented HoloAssist clause ("VLM judge baseline is weak → held"). The thesis's hidden-outcome prediction has two clauses, reaction-only > `action_best` and fusion > `action_best`, and **both are untested**.
  - **`:96`:** "zero API errors across 7,708 VLM queries". There were **10,780** local answers (Oops! 2,708 + 1,072; HoloAssist 5,000 + 2,000) with 2 ffmpeg errors, plus the Gemini cache's 1,024 items with HTTP 429 rows.
  - **`:108`:** the HoloAssist 7B judge at 56,986.1 s. The cache sums to **40,656.9 s**, and the totals at `:112` inherit the error.
  - **`:118–119`:** "Reaction signal exists and overcomes action probes". It headlines `react-full` (the leakage condition, §11.2) and compares unpaired AUROCs, with no Δ and no mention of the 0.721 step prior (Issue 7).
  - **`:126`:** it says the 26B judge "resolved the judge failure on gross physical accidents". It did not reach the "strong" bar, and its HoloAssist 0.544 is the verb `drop`.
  - **Missing:** the state of Wave D's start condition.
- **`tools/report_holoassist.py:17–105`:**
  - its "Clause 1, VLM judge baseline → at chance / above chance" is not the thesis's prediction;
  - its pass-bar line takes `max(react-spoke, react-full, react-nonverbal)` (`:83–:101`), so it names **`react-full` 0.739 as the "best reaction"**;
  - `docs/evals/2026-10-15_holoassist_h1.md` §2 shows both.

**Implementation:**
1. **`tools/report_holoassist.py`:** replace the prediction-check function (`:17`, `prediction_check`) with `prediction_check_hidden(rows: Dict[str, dict]) -> List[str]`, implementing `03_eval_harness.md` §11.8. It returns exactly these lines, in order, filled from the rows:
   - `- **Clause 1, "reaction-only beats action_best":** react-nonverbal <v> [<lo>, <hi>] vs <action_best name> <v> → <held|did not hold|not tested: <react-nonverbal notes>>`. It **held** iff `react-nonverbal ci_low > action_best's value`.
   - `- **Clause 2, "fusion beats action_best":** Δ <v> [<lo>, <hi>] → <held|did not hold|not tested: <Δ notes>>`. It **held** iff `ci_low > 0`.
   - `- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and Δ ≥ +0.03 with ci_low > 0 → <met|not met|not testable: react-nonverbal or Δ not run>`.
   - `- **Context, no verdict:** judge <v> [<lo>, <hi>]; judge-large <v> [<lo>, <hi>]; action-probe <v> [<lo>, <hi>]; react-spoke <v> [<lo>, <hi>] (diagnostic: did the instructor speak); react-full <v> [<lo>, <hi>] (leakage condition: the instructor's words; never a headline, §11.2).`
   - `- **Caveat:** a train-split step-verb prior scores 0.721 on this test split, above every action-only control; judge-large scores 0.489 without the verb "drop" (02_data_sources.md → HoloAssist → As built (A8); Issue 7).`
2. **Regenerate `docs/evals/2026-10-15_holoassist_h1.md`.** Add as its second line: `*Corrected <YYYY-MM-DD> (A9b): prediction check uses the thesis's hidden-outcome clauses; react-full is no longer a "best reaction"; step-prior caveat added. The numbers are unchanged.*`
3. **Correct the summary** (`docs/evals/2026-10-15_wave_a_summary.md`). Add as its second line: `*Corrected <YYYY-MM-DD> (A9b): see the commit body for each correction. No scorecard row changed.*`
   - **`:13`** → `95% CIs are percentile intervals over 1,000 group-bootstrap resamples (groups: compilation for Oops!, session prefix for HoloAssist; numpy default_rng(0); single-class resamples redrawn, at most 10,000 draws), per 03_eval_harness.md §7.`
   - **Delete row `:26`.** In the HoloAssist `react-full` row, add to Notes: `leakage condition (the instructor's words); never a headline (§11.2)`.
   - **§2.1:** replace the Clause 1 bullets with the three lines `tools/report_oops.prediction_check` prints for the latest Oops! rows, **verbatim**. Keep Clause 2.
   - **§2.2:** replace the section body with the lines `prediction_check_hidden` prints for the latest HoloAssist rows, **verbatim**.
   - **§4 item 3** → *Judge answers:*
     - local: Oops! 2,708 (7B; 0 parse failures; 2 ffmpeg errors on one undecodable train clip) + 1,072 (26B; 0/0); HoloAssist 5,000 (7B; 0/0) + 2,000 (26B; 0/0) = 10,780 answers;
     - frontier (Oops! test only): 50 scored, 1,024 items with HTTP 429 rows, 1 other error;
     - every test-split condition that ran excluded 0 items.

     Recount each figure from the caches and `errors.jsonl` files, and put the recount commands in the commit body.
   - **§4, add item 4:** `HoloAssist step text leaks the outcome: a train-split verb prior scores 0.721 on test, above every action-only control, and judge-large's 0.544 comes from the verb "drop" (0.489 on the other test items). Issue 7.`
   - **§5:** recompute every figure from its source:
     - features: the sum of `elapsed_ms` over the last `index.jsonl` entry per item;
     - judges: the sum of `elapsed_ms` in each `<prompt_hash>.jsonl`;
     - download: `DOWNLOAD.json`/`download.log`;
     - anything without a recountable source: `not recorded`.

     The HoloAssist 7B judge is **40,656.9 s**. Recompute the totals.
   - **§6 Observation 1** → `On HoloAssist the action-only controls are weak (action-probe 0.569; judges 0.488 and 0.544) and react-spoke reaches 0.676, but no Δ was computed (Issue 6) and a step-name prior reaches 0.721 (Issue 7). Whether reactions add information beyond action-only cues on hidden outcomes is untested in Wave A.`
   - **§6 Observation 2:** prefix it with `Hypothesis, from the presence check (not tested):`.
   - **§6 Observation 3** → `The 26B judge raised Oops! from 0.472 to 0.708, still below the action-probe (0.779) and short of the "strong" bar; on HoloAssist its 0.544 comes from the verb "drop" (0.489 otherwise). Neither local judge recognizes procedural mistakes from frames.`
   - **§6 Observation 4:** state the facts only: `memguard.log through October 15: 10 shared-model stops (sr_judge_large_load) and 77 unloads, all of our own 7B judge (sr_judge_load); no unload of gemma4:26b; no jetsam report names a process of ours.`
   - **Add §7 "Wave D start condition":** `Not met. Oops!'s Δ CI includes 0 (+0.015 [−0.002, 0.029]); HoloAssist's Δ was not run (Issues 6, 7).`
4. **New `tests/test_report_holoassist.py`:**
   - **(a)** the real HoloAssist rows (as fixtures) → Clause 1 `not tested`, Clause 2 `not tested`, pass bar `not testable`, and **no line containing a verdict word (`held`, `met`) also contains `react-full`**;
   - **(b)** `react-nonverbal 0.70 [0.66, 0.74]`, `action-probe 0.57`, Δ `0.05 [0.02, 0.08]` → held, held, met;
   - **(c, falsifying)** `react-full 0.90`, `react-nonverbal` null, Δ null → pass bar `not testable`. It proves `react-full` cannot stand in.

**Validation:**
- **Red first:** tests (a)–(c) fail against the current generator. Record it.
- **Grep the corrected summary:**
  - it contains **none** of `10,000 cluster`, `held on sample`, `7,708`, `56,986`, `Overcomes`, `overcomes`, `resolved the judge failure`;
  - it **contains** `1,000 group-bootstrap`, `Wave D start condition` and `Not met`, and three `did not hold` lines in §2.1.
- **Every number in the summary's tables equals its scorecard row** (latest row per dataset and condition, to 3 decimals). Check it with a one-off script whose output goes in the commit body.
- `git diff results/scorecard.jsonl` is empty.
- G1–G3 green (G2 three times, as in A10).

**Blast radius:** `tools/report_holoassist.py`, `tests/test_report_holoassist.py` (new), `docs/evals/2026-10-15_holoassist_h1.md`, `docs/evals/2026-10-15_wave_a_summary.md`; `ongoing_general_errors.md` §3; this guide's §1.4. **Not** Issues 6 and 7, which the maintainer decides.

**Then close out:** rewrite this guide's title and status to **"Queue Complete: waiting on Issues 6 and 7 (HoloAssist's H1 test) and Issue 1 (Wave B)"**. **Then stop. Do not invent work.**

---

## 4. Deferred: do NOT start

| Work | Trigger | Who writes the spec |
|---|---|---|
| **HoloAssist's H1 Δ** (per-session audibility, `react-nonverbal`/`fusion` on the audible subset, a `fusion-spoke` row, a `step-prior` control, or none of these) | The maintainer selects **Issue 6** and **Issue 7** | The designer, after both selections |
| **Wave B: verdict-video H1 pilot**, DW1 | **Issue 1** selected **and** A9b/A10 landed **and** the designer's Wave B spec | The designer |
| **Face encoder** `react-face`, DW2 · **H2 transfer**, DW3 · **Ego4D false-positive set**, DW4 | Wave B | The designer |
| **Wave D: offline robot check (RoboReward)**, DW5 | Wave A shows a positive, CI-excluding-0 Δ on ≥ 1 dataset (**not met**: `00_thesis.md` entry 19) **and** a Wave D spec | The designer |
| **Stage A live (microduck)**, DW6 | Hardware, H2 done, and the maintainer opts in | The designer |

Also not to do:
- paid frontier-judge runs (Issue 5 chose B);
- any change to a judge prompt or question;
- any new download;
- any rerun of probes, features or judges;
- anything from tag `v0-saf-final`.

---

## 5. Do NOT change

### 5.1 Already delivered (verified by the designer)

| Item | Commit | What it delivered |
|---|---|---|
| A1–A6 | `651039b`–`0f16799` | the battery; metrics and scorecard; items and splits; HoloAssist labels and independence; encoders and cache; the judges |
| A6b + A6c | `76e71cc`, `8c6f04e` | the memory guard: re-entrant `guard()`, fail-closed reads, supervisor accounting, `check_shared`, fast tests isolated from live HTTP |
| A7 + A7b | `6d72492`, `d77584c` | the Oops! pipeline and its corrected, generated report |
| A7c | `1346e0b` | `judge-large` (`gemma4:26b`): the declared peak of 33 GB, measured at 28.10 GB; the Oops! re-score, with every earlier value identical |
| A8 | `36a7769` | HoloAssist: `holoassist:` item ids; the split (3,000/2,000; 238/102 groups, partitioned over all sessions); the audio check (FAIL, Issue 6); SigLIP; both local judges; probes; the report |
| A9 | `3f0b78d` | the Wave A summary (A9b corrects it) |

### 5.2 Accepted equivalents (do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure.** It is stricter than the median of the rest; the caps were never binding.
- **`extract_frame` retries a seek up to 0.3 s earlier** when ffmpeg returns no frame at a clip's final timestamp.
- **`GeminiJudge`:** `max_output_tokens=32`, `thinking_budget=0`, and a 429 that survives its 5 tries is not cached (`03_eval_harness.md` §8).
- **`guard()` holds the heavy lock through the model load only,** not through the first item.
- **An item's scorecard rows carry its parent commit's `git_sha` with `dirty: true`.** The code of record is the item's own commit.
- **HoloAssist groups are partitioned over all 1,758 annotated sessions,** not only those with video. The partition (238/102) does not depend on which videos exist; 91 test groups end up with sampled items.
- **`judge` (7B) now checks `/api/ps` before every call,** including retries, not only the first. The behavior is the same, with one extra cheap request.
- **`download_holoassist.py` checked for 50 GiB free before the download, not after extraction.** The rule held: 1.3 TiB stayed free, and all 2,111 videos were verified.

### 5.3 Maintainer decisions

- **October 8:**
  - outcome supervision only, with no human rating;
  - "just ask an LLM" is the control to beat;
  - hidden-outcome data first;
  - Mac Studio only;
  - commit straight to `main`;
  - the v0 videos deleted.
- **October 9:**
  - no self-recorded data, no human-subjects study, no academic partner;
  - Wave D (offline robot check) in scope; the microduck deferred.
- **October 10:**
  - **Issue 5 → B**, a larger local judge, not paid frontier calls;
  - the A8 draft committed as found.

### 5.4 Invariants

- **Supervise on outcomes, never on emotion; `label = 1` is a good outcome.**
- **The judge never hears audio.**
- **The Δ is taken against the best action-only control; a judge enters `action_best` only at ≥ 90% coverage.**
- **Grouped splits and grouped CIs only; split files are written once.**
- **The scorecard is append-only; nulls carry reasons.**
- **`react-full` is never reported alone or as a headline.**
- **`gemma4:26b` is shared and never unloaded by us. `guard()` is re-entrant, and reads fail closed.**

### 5.5 Assessed and rejected: do NOT re-propose

- human rating rounds, rater UIs, golden labels;
- hand-built affect channels;
- self-recorded data, human-subjects studies, an academic partner;
- Ego4D bystander footage as H1 data;
- tuning prompts, hyperparameters, windows, caps or subsets on test results;
- circumventing any site's bot checks;
- choosing HoloAssist's reaction channel or action-only control yourself. That is Issues 6 and 7.

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, H1–H3, pass/kill, the decision log (entry 19 = Wave A's state) | `00_thesis.md` |
| Items, splits, scorecard, conditions, metrics (§7), judges (§8), guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| Item definitions, schemas, as-built caveats (the Oops! audio; the HoloAssist step prior and audibility) | `02_data_sources.md` |
| Issues 1, 4, 6, 7 (open); 2, 3, 5 (resolved); lessons L1–L13; the resolved index | `ongoing_general_errors.md` |
| Operations (detached runs, ollama, decoding, storage) | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard

- **Red first.** Show each new test failing against the current code before the fix.
- **Every gate must be able to fail.** Break, see red, restore, see green.
- **A flaky gate is a failing gate.** Repeat G2 three times under the required environment.
- **Report text is checked against the rows.** Any sentence with a number names its source, and a verdict follows the pinned rule.
- **Never loosen a bar to pass it.** File it, with the measurement and options.

---

## 8. THE LOOP

```
(1) Is there an approved item? A10, then A9b. If both are done, STOP:
    rewrite the title to Queue Complete (A9b's close-out). Never start §4
    work. Never fill in a `Your selection:` line.
(2) Read the item and every contract section it names.
(3) RED FIRST: run the item's new tests; record the failure.
(4) Build only what the item says.
(5) GREEN; then falsify (break, see red, restore, see green).
(6) Open every artefact you changed and read it.
(7) scripts/battery.sh bare, with TMPDIR on the SSD (G2 three times).
    Update §1.4.
(8) ONE commit on main, scope = item id, explicit paths only. WHY +
    red/green in the body. ONE line under "Wave A" in
    ongoing_general_errors.md §3.
(9) /usr/bin/git push origin main
```

---

## 9. Definition of Done: Wave A corrections

- [ ] A10 and A9b each landed as one pushed commit, with red and green runs recorded.
- [ ] No glob over the SSD returns a `._*` file. G2 passed 3 times in a row with `TMPDIR` on the SSD.
- [ ] The summary and the HoloAssist report carry only rule-based verdicts; `react-full` is never a headline; every table number matches its scorecard row; Wave D's start condition is stated as not met.
- [ ] `results/scorecard.jsonl` is unchanged.
- [ ] This guide is rewritten to **Queue Complete: waiting on Issues 6 and 7 (HoloAssist's H1 test) and Issue 1 (Wave B)**. **Then stop. Do not invent work.**

The only legitimate triggers for new work after that are:
- the maintainer's selection on Issue 6, 7 or 1, followed by the designer's spec;
- a new issue the maintainer raises.
