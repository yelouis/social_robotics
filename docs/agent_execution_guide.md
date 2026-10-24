# Agent Execution Guide: Queue Complete: waiting on Wave B's data access (Issue 1 = E), October 23, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:** on October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
- **What exists:** Waves A and D are complete and verified by the designer (A1–A15, D1–D4; `651039b` … `315f3a6`):
  - **Wave A (H1):**
    - Oops!: the ablation gain is +0.015 [−0.001, 0.029] (≈ 0, as predicted).
    - HoloAssist, audible sessions: the instructor's non-verbal voice adds **+0.021 [0.006, 0.034]** to the action-only model. That gain is mostly *whether the instructor spoke*.
  - **Wave D (H3 offline, pre-registered):** the verdict is **NOT SUPPORTED**.
    - Reaction-labeled HoloAssist video gave no gain to a RoboReward success detector (+0.0009 [−0.0006, 0.0024] at N = 100).
    - The designer found two flaws in the spec, and diagnostics showing that every human head is at chance on robot videos (`04_wave_d_robot_check.md` §8).
- **Wave D's result is accepted** (Issue 10 = A, October 22). No v2 is run, and robot transfer is revisited only after Wave B, through a new spec (DW7).
- **There is no approved work.** The next work is **Wave B**. Its data source was decided October 23 (**Issue 1 = E**: licensed datasets on request). Wave B waits on the replies to those requests (maintainer actions M4 and M5; M4's address bounced on October 23), then on the designer's Wave B spec.

**Status:** **Queue Complete. Do not invent work.** The only legitimate triggers for new work are:
1. **a yes to an E request (M4 or M5),** then the designer's Wave B spec and new guide items;
2. **a new issue the maintainer raises.**

If you were started on this guide with no new spec in it, read §1, run the battery to confirm the baseline (§1.4), report the result, and **stop**.

**Every path, number, field name and literal string in this guide and in the contract docs is a decision, not a suggestion.**

---

## 0. Standing constraints (they apply to any future item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare (`--slow` when the work touches models). G2 must pass 3 times in a row with the `TMPDIR` below.
   - Read every exit code bare.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only.
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies, models and downloads are frozen.** No new pip packages, no new model downloads, no Ollama pulls, and no data downloads without a maintainer decision.
4. **Data lives under `DATA_ROOT`** (`/Volumes/Extreme SSD/social_robotics`; exFAT, about 1.1 TiB free). No video or audio on the internal disk. Never delete anything you did not create.
5. **Long runs (> 30 min) run detached** (`tools/daemonize.py` + `tools/run_supervised.sh`; confirm `PPID 1`). Runners are resumable and end with the count line.
6. **Memory (`03_eval_harness.md` §12).**
   - Never run two of this project's model-loading jobs at once.
   - Run `python -m shared.memguard --status` first. Exit 75 is never a pass.
   - **`gemma4:26b` is shared; never unload it.**
7. **The scorecard is append-only.** Nulls carry derived reasons. Reports are generated from rows. Verdicts follow the pinned rules (`03_eval_harness.md` §11.8; `04_wave_d_robot_check.md` §6).
8. **Pre-registered rules are never changed after their result is seen.** That includes `04_wave_d_robot_check.md` §1–§7, the pass bars and the triggers. A follow-up is a new, dated spec.
9. **Commits:**
   - one item = one Conventional Commit on `main`, with the WHY, the red and green runs, and any missed validation bar in the body (L14);
   - explicit paths only;
   - `/usr/bin/git push origin main`;
   - no branches, no PRs, never amend.
10. **When a contract and this guide disagree, or a bar cannot be met, STOP and file a new issue** (the next number is **Issue 11**), with options.
11. **Never fill in a `Your selection: _____` line.** Issues 1 and 4 belong to the maintainer.

---

## 1. Verified baseline (October 22, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1.
- **Ollama:** `qwen2.5vl:7b` (ours) and `gemma4:26b` (shared). No model was loaded at verification.
- **`venv`, Python 3.9.6:** torch 2.8.0, transformers 4.46.3, funasr 1.3.1, scikit-learn 1.6.1, scipy 1.13.1, numpy 2.0.2, huggingface_hub 0.36.2, ruff 0.15.11, pytest 8.4.2.

### 1.2 Repository

- `HEAD` includes `315f3a6` (`feat(d4)`) plus the designer's October 22 docs commit. The working tree is clean.

### 1.3 Data on the SSD

| What | Where | State |
|---|---|---|
| Oops! | `raw/oops/`, `items/oops/`, `features/oops/`, `judge/oops/` | complete |
| HoloAssist (`holoassist`, `holoassist_audible`, `holoassist_pool`) | `raw/holoassist/`, `items/…`, `features/…`, `judge/…` | complete; per-session audibility in `runs/holoassist_session_audibility.jsonl` |
| RoboReward (all of it) | `raw/roboreward/` (`DOWNLOAD.json`: 54,140 files, 43,204,474,744 bytes) | complete; `items/roboreward/` (4,000 train pool + 1,517 test); `features/roboreward/siglip-b16-224-u9/`; `judge/roboreward/gemma4_26b/` |
| Wave D artefacts | `transfer/{labeler,scaler,heads,curve_scores}.npz`, `transfer/holoassist_pool_pseudolabels.jsonl` | complete (v1) |

### 1.4 Gates (run bare October 22, 2026, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result on `315f3a6` |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · clean |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **106 passed**, 3 runs out of 3 (63 s, 60 s, 59 s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **11 passed** (124 s); `memguard --status`: 53.1 GB, pressure 1, lock free |

### 1.5 Headline results (the scorecard is the source)

| Dataset | Result | Row `ts` |
|---|---|---|
| `oops` | ablation gain +0.015 [−0.001, 0.029]; `action-probe` 0.779; `judge-large` 0.708 | `2026-10-21T04:12:36–42Z` |
| `holoassist` | `step-prior` 0.751; `action-fusion` 0.682; no reaction Δ (the audio was inaudible in most sessions) | `2026-10-21T04:12:57Z–04:13:02Z` |
| `holoassist_audible` | `react-nonverbal` 0.654; **ablation gain +0.021 [0.006, 0.034]**; fusion − `step-prior` −0.065 | `2026-10-21T16:50:20–30Z` |
| `roboreward` (Wave D v1) | **NOT SUPPORTED**: `gain:react:N=100` +0.0009 [−0.0006, 0.0024]; `gain:truth:N=100` −0.0507; `robot-full` 0.657; `judge-large` 0.747 | `2026-10-22T14:22:28Z–14:24:54Z` |

---

## 2. Execution order

**Empty.** Nothing is approved. See the Status triggers above.

---

## 3. The items

None. The last queue (A15, D1–D4) was delivered and verified on October 22 (§5.1).

---

## 4. Deferred: do NOT start

| Work | Trigger | Who writes the spec |
|---|---|---|
| **Robot transfer, revisited** (DW7): a new pre-registered spec with Wave B's reaction model and a human source closer to robot manipulation. It applies L15 and L16 | Wave B shows a reaction gain **and** a closer human source is identified **and** the designer has written the spec (Issue 10 = A) | The designer |
| **Wave D v2** (the same data, with the two flaws fixed) | **Not selected** (Issue 10 = A). Do not run it | — |
| **Wave B: verdict-video H1 pilot**, DW1 | Issue 1 = E (decided October 23) **and** an E dataset granted (M4 or M5) **and** a Wave B spec (which restates the kill rule in ablation-gain terms) | The designer |
| **RoboReward-8B reference** | A maintainer decision to unfreeze `transformers` (it is Qwen3-VL-based) **and** a spec | The designer |
| **Face encoder**, DW2 · **H2 transfer**, DW3 · **Ego4D false-positive set**, DW4 | Wave B | The designer |
| **Online H3** (training a policy with the reward) | After offline H3 shows a gain, and a cloud CUDA GPU | The designer |
| **Stage A live (microduck)**, DW6 | Hardware, H2 done, and the maintainer opts in | The designer |

Also not to do:
- re-running Wave D with any changed setting;
- deleting or rewriting any v1 artefact or row;
- adding the designer's exploratory diagnostics as scorecard rows;
- any download;
- anything from `v0-saf-final`.

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
- **Wave D v1's pool has 0.52% mistakes, and its true-label head is overfit.** These are part of v1's record (`04` §8), **not** bugs to patch. v1 is accepted as the result (Issue 10 = A), and any future robot-transfer test is a new spec (DW7).

### 5.3 Maintainer decisions

- **October 8–10:**
  - outcome supervision only, with no human rating;
  - "just ask an LLM" is the control;
  - hidden-outcome data first;
  - Mac Studio only;
  - commit to `main`;
  - Issue 5 → B.
- **October 20:** Issue 6 → A (a fresh sample); Issue 7 → A; the pass bar and the Wave D trigger use the ablation gain.
- **October 21:** Issue 8 → A (run Wave D); Issue 9 → B (the full RoboReward download; RoboReward-8B not run).
- **October 22:** **Issue 10 → A.** Wave D v1 is the offline-H3 result; robot transfer is revisited only after Wave B (DW7).
- **October 23:** **Issue 1 → E.** Wave B's verdict data is requested from Affectiva (AM-FED / AM-FED+) and from the beverage-tasting database's author. If neither says yes, Issue 1 is reopened.

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **Grouped splits and CIs. Split files are written once. The scorecard is append-only.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels. The leak check is one-sided.**
- **The Wave D detector never reads the instruction. The robot test split is used only for scoring.**

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
| The thesis, H1–H3, the decision log (entries 19–26) | `00_thesis.md` |
| The scorecard, conditions, metrics, judges, guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| Wave D: the contract (§1–§7) and the v1 result with the designer's notes (§8) | `04_wave_d_robot_check.md` |
| Datasets: Oops!, HoloAssist (`holoassist_audible`, `holoassist_pool`), RoboReward | `02_data_sources.md` |
| Issues 1 and 4 (open); 2, 3 and 5–10 (resolved); deferred work DW1–DW7; lessons L1–L16 | `ongoing_general_errors.md` |
| Reports | `docs/evals/` (Oops!, HoloAssist, HoloAssist-audible, the Wave A summary, Wave D) |
| Operations | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard (for any future item)

- **Red first, then green, then falsify.**
- **Reproduce the stated reference values.** A mismatch is a finding.
- **A flaky gate is a failing gate.** Run G2 three times.
- **Report every missed validation bar in the commit body** (L14).
- **Check a derived pool's class counts before using it** (L15), **and a head's zero-shot transfer before building a prior on it** (L16).
- **Never loosen a bar or change a pre-registered rule to pass it.** File it.

---

## 8. THE LOOP

```
(1) Is there an approved item in §2? It is empty: STOP. Do not invent work.
    Never start §4 work. Never fill in a `Your selection:` line.
(2) If asked to confirm health: run scripts/battery.sh bare with TMPDIR on
    the SSD (G2 three times), compare with §1.4, report, and stop.
```

---

## 9. Definition of Done for this state

- [x] A1–A15 and D1–D4 delivered and verified (§5.1).
- [x] The battery is green and re-measured bare on October 22 (§1.4).
- [x] Wave D v1's verdict is recorded, with the designer's post-hoc notes beside it (`04` §8; the report's designer note).
- [x] Issue 10 decided (A): Wave D v1 is accepted as the result.
- [x] Issue 1 decided (E, October 23).
- [ ] **Waiting:** the replies to the E requests (M4, M5), then the Wave B spec. **Queue empty: do not invent work.**
