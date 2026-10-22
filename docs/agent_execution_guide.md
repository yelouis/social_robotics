# Agent Execution Guide: Active Build: Wave A report fix (A15), October 21, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:** on October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
- **What exists: Wave A is complete** (A1–A14 and their fixes; `651039b` … `d6d9c54`, October 9–21):
  - the harness, the encoders and three judges;
  - the shared memory guard;
  - the `step-prior` and `action-fusion` controls and the ablation Δ;
  - H1 runs on `oops`, `holoassist` and `holoassist_audible`;
  - the summary (`docs/evals/2026-10-15_wave_a_summary.md`).

  The designer verified it on October 21 (§1).
- **What you build:** **one small item, A15.** It fixes a number the HoloAssist report generator hard-codes, and the label it puts on shuffled-control results.
- **Then stop.** What comes next is the maintainer's **Issue 8** (whether to start Wave D) and **Issue 1** (Wave B's data). Neither is yours.

**Read before anything else, in this order:**
1. [`ongoing_general_errors.md`](ongoing_general_errors.md): §1 (the October 21 verification), **Issue 8** (not yours to decide), and lessons L10–L14.
2. [`03_eval_harness.md`](03_eval_harness.md): **§11.9** (the one-sided shuffled-control rule A15 implements) and §11.8 (prediction checks).
3. [`02_data_sources.md`](02_data_sources.md) → HoloAssist → "Audible-session evaluation set" → "As run (A13)".

**The maintainer's words:**
- *"The fundamental core of this project is to use human emotion as a policy for robot learning in addition to the traditional RL stack for robots. We want to add this additional reward signal and investigate if this additional reward signal helped the robot learn a task better and perhaps allow robots to train on much more data like vlogs or any POV youtube video."*
- *"Ideally we should avoid human in the loop ratings because that is not scalable and we want to be able to somehow measure whether our thesis is working often."*
- *"How is this any different from asking an LLM model what is socially appropriate…?"* This is why every H1 result is measured against action-only controls, and why a wrong number in a report is a defect.
- *"No need to open up a new branch, just push to the repo."*

**Status:** **Active Build: Wave A report fix.** Next is **A15**. It needs no maintainer decision, no models and no downloads.

**Every path, number, field name and literal string in this guide is a decision, not a suggestion. Implement as written; do not substitute your own.**

---

## 0. Standing constraints (apply to every item)

1. **The battery is the regression bar.** Run `scripts/battery.sh` bare after the item, with G2 three times, and update §1.4. Read every exit code bare.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only.
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **No new dependencies, no downloads, no model runs, no probe re-runs.** A15 reads existing caches and scorecard rows only.
4. **Globs over the SSD go through `shared.fsutil.data_glob`** (L13).
5. **The scorecard is append-only, and A15 writes no rows.** `git diff results/scorecard.jsonl` must be empty.
6. **Reports are generated from data (L10).** Every number in a report line is computed from a named source (a scorecard row or a cache) at generation time, never typed in as a literal.
7. **Commits.**
   - One Conventional Commit, `fix(a15): …`, on `main`, with the WHY, the red run and the green run in the body.
   - Stage explicit paths only. Push with **`/usr/bin/git push origin main`**. No branches, no PRs, never amend.
   - If `git status` shows a change you did not make, STOP and ask the maintainer.
8. **Record the resolution in the same commit:** one line under "Wave A" in `ongoing_general_errors.md` §3.
9. **When this guide and a contract doc disagree, STOP and file it** as a new issue (the next number is **Issue 9**), with options.
10. **Never fill in a `Your selection: _____` line.** Issues 1, 4 and 8 belong to the maintainer.

---

## 1. Verified baseline (October 21, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1.
- **Ollama:** `qwen2.5vl:7b` (ours) and `gemma4:26b` (shared with `animated_infographics`; never unload it). No model was loaded at verification.
- **`venv`, Python 3.9.6:** torch 2.8.0, transformers 4.46.3, funasr 1.3.1, scikit-learn 1.6.1, numpy 2.0.2, soundfile, httpx 0.28.1, ruff 0.15.11, pytest 8.4.2.
- **The SSD is exFAT** (L13), with about 1.3 TiB free.

### 1.2 Repository

- `HEAD` includes `d6d9c54` (`feat(a14)`) plus the designer's October 21 docs commit. The working tree is clean.

### 1.3 Data and caches (what A15 reads)

| Dataset | Items | Test | Judge-large cache | Scorecard rows (latest) |
|---|---|---|---|---|
| `oops` | 2,710 | 1,072 | `judge/oops/gemma4_26b/2409b2876016.jsonl` | `2026-10-21T04:12:36–42Z` |
| `holoassist` | 5,000 | 2,000 | `judge/holoassist/gemma4_26b/4f0760a8e441.jsonl` | `2026-10-21T04:12:57Z–04:13:02Z` |
| `holoassist_audible` | 4,420 | 1,420 (53 groups) | `judge/holoassist_audible/gemma4_26b/4f0760a8e441.jsonl` | `2026-10-21T16:50:20–30Z` |

### 1.4 Gates (run bare October 21, 2026, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result on `d6d9c54` |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · clean |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **75 passed**, 3 runs out of 3 (56 s, 54 s, 54 s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **9 passed** (123 s); `memguard --status`: 53.8 GB, pressure 1, lock free |

### 1.5 Reference values A15 must reproduce (designer, October 21, from the caches)

| Dataset | `judge-large` AUROC on test items whose verb is not `drop` | n |
|---|---|---|
| `holoassist_audible` | **0.498 [0.481, 0.514]** | 1,361 |
| `holoassist` | **0.489**: its CI is computed by A15 and must contain 0.489 | 1,818 |

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| A15 | Make the HoloAssist report compute its `judge-large` caveat, and make both report generators apply the one-sided shuffled-control rule | The only queued work. It needs nothing else, and after it Wave A's record is correct |

---

## 3. The item

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

**Then close out:** rewrite this guide's title and status to **"Queue Complete: waiting on Issue 8 (Wave D go/no-go) and Issue 1 (Wave B)"**. **Then stop. Do not invent work.**

---

## 4. Deferred: do NOT start

| Work | Trigger | Who writes the spec |
|---|---|---|
| **Wave D: offline robot check (RoboReward)**, DW5 | The trigger is **met** by the rule. The maintainer selects **Issue 8 option A**, then the designer writes and pre-registers the Wave D spec, including the download size for approval | The designer |
| **Wave B: verdict-video H1 pilot**, DW1 | **Issue 1** selected **and** a Wave B spec (which also restates the kill rule in ablation-gain terms) | The designer |
| **Face encoder**, DW2 · **H2 transfer**, DW3 · **Ego4D false-positive set**, DW4 | Wave B | The designer |
| **Stage A live (microduck)**, DW6 | Hardware, H2 done, and the maintainer opts in | The designer |

Also not to do:
- pre-declaring or adding the designer's exploratory strata (instructor spoke / silent / no `drop`) as scorecard rows;
- any probe, feature or judge re-run;
- any download, including RoboReward;
- any judge prompt change;
- anything from `v0-saf-final`.

---

## 5. Do NOT change

### 5.1 Already delivered (verified by the designer)

| Item | Commit | What it delivered |
|---|---|---|
| A1–A6 | `651039b`–`0f16799` | the battery; metrics and scorecard; items and splits; HoloAssist labels and independence; encoders and cache; the judges |
| A6b + A6c | `76e71cc`, `8c6f04e` | the memory guard (re-entrant, fail-closed, shared lock, `check_shared`) |
| A7, A7b, A7c | `6d72492`, `d77584c`, `1346e0b` | the Oops! pipeline and report; `judge-large` |
| A8 | `36a7769` | the `holoassist` pipeline; the audio check |
| A9, A9b | `3f0b78d`, `6744719` | the Wave A summary, corrected |
| A10 | `492bd79` | `shared.fsutil.data_glob`: the exFAT `._*` files are ignored |
| A11 | `abbc273` | `step-prior`, `action-fusion` and the ablation Δ; re-scores matching the designer's references exactly |
| A12 | `608c729` | per-session audibility (1,673 sessions; 548 audible) |
| A13 | `eea8933` | `holoassist_audible` (4,420 items; ablation Δ +0.021 [0.006, 0.034]) |
| A14 | `d6d9c54` | the completion section; the Wave D start condition met by the rule |

### 5.2 Accepted equivalents (do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure.**
- **`extract_frame` retries a seek up to 0.3 s earlier.**
- **`GeminiJudge`:** 32 output tokens, thinking off, and a 429 is not cached after 5 tries.
- **`guard()` holds the heavy lock through the model load only.**
- **Scorecard rows carry the parent commit's `git_sha` with `dirty: true`.**
- **The HoloAssist group partition is computed over all 1,758 annotated sessions,** and `holoassist_audible` reuses it.
- **`holoassist_audible`'s `not_audible` skip count includes the 85 sessions without video** (`02_data_sources.md` → As run (A13)).
- **`holoassist_audible` has 1,420 test items,** above the 1,000–1,300 estimate. Only the lower bound was a stop rule.
- **The 7B judge checks `/api/ps` before every call.**

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
  - Issue 6 → A, with a fresh sample;
  - Issue 7 → A;
  - the pass bar and Wave D's trigger use the ablation gain.

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **The pass-bar Δ is the ablation gain,** and `fusion_minus_action_best` is always reported beside it.
- **Grouped splits and CIs; split files are written once. The scorecard is append-only; nulls carry reasons.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **A shuffled control never sees the true labels. Its check is one-sided** (§11.9).

### 5.5 Assessed and rejected: do NOT re-propose

- human rating, rater UIs, golden labels;
- hand-built affect channels;
- self-recorded data, studies, academic partners;
- Ego4D bystander footage as H1 data;
- tuning prompts, hyperparameters, windows, caps, subsets or the combiner on test results;
- circumventing bot checks;
- changing a rule (a trigger, a pass bar, `AUDIBLE_DB`) after seeing the result it would decide. That is why Issue 8's option C is marked as such.

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, the H1 pass bar, the H3 trigger, the decision log (entries 19–21) | `00_thesis.md` |
| The scorecard (§4), conditions (§6), metrics (§7), judges (§8), guardrails (§11, incl. §11.8–§11.9), memory guard (§12) | `03_eval_harness.md` |
| Dataset definitions and as-built facts (Oops!, `holoassist`, `holoassist_audible`) | `02_data_sources.md` |
| Issues 1, 4, 8 (open); 2, 3, 5, 6, 7 (resolved); lessons L1–L14 | `ongoing_general_errors.md` |
| Operations | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard

- **Red first, then green, then falsify.**
- **Reproduce the designer's reference values** (§1.5). A mismatch is a finding.
- **A flaky gate is a failing gate.** Run G2 three times.
- **A result that misses a stated validation bar is reported in the commit body**, even if you believe it is benign (L14).
- **Never loosen a bar to pass it.** File it.

---

## 8. THE LOOP

```
(1) The only approved item is A15. When it is done, STOP (A15's close-out).
    Never start §4 work. Never fill in a `Your selection:` line.
(2) Read the item and §11.9 of 03_eval_harness.md.
(3) RED FIRST: run the new tests; record the failure.
(4) Build only what the item says.
(5) GREEN; then falsify (put 0.489 back, see red, revert).
(6) Open the three regenerated reports and the summary §8.2, and read them.
(7) scripts/battery.sh bare with TMPDIR on the SSD (G2 three times).
    Update §1.4.
(8) ONE commit `fix(a15): …`, explicit paths, WHY + red/green in the body,
    ONE line under "Wave A" in ongoing_general_errors.md §3.
(9) /usr/bin/git push origin main
```

---

## 9. Definition of Done

- [ ] A15 landed as one pushed commit, with red and green runs recorded.
- [ ] No report hard-codes the 0.489. The audible report shows `0.498 [0.481, 0.514]` on 1,361 items.
- [ ] No report shows `FAIL` for a below-chance shuffled control. The leak check follows §11.9 in both generators.
- [ ] `results/scorecard.jsonl` is unchanged. G1–G3 are green, with G2 passing 3 times in a row.
- [ ] This guide is rewritten to **Queue Complete: waiting on Issue 8 (Wave D go/no-go) and Issue 1 (Wave B)**. **Then stop. Do not invent work.**

The only legitimate triggers for new work after that are:
- the maintainer's selection on Issue 8 or Issue 1, followed by the designer's spec;
- a new issue the maintainer raises.
