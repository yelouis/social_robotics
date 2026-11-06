# 06: Wave E, the write-up: the contract

**What this is:** the specification for a paper-style write-up of everything Waves A, B and D measured. The maintainer chose it on November 5, 2026 (Issue 12 = A; `00_thesis.md` entry 30). Wave B is recorded as negative in substance.

**The build spec** is [`agent_execution_guide.md`](agent_execution_guide.md), items **E1** (fact base, tables, figures, number checker) and **E2** (the draft). The designer verifies both. The maintainer reviews the draft and makes the decisions in §8.

---

## 1. What the paper is

- **The question:** do an observer's spontaneous reactions (voice, face, posture) add information about an outcome beyond what an action-only model sees, especially when the outcome is hidden? If they do, they could serve as a reward signal for robots.
- **The answer the data supports:** almost nothing, on every dataset we could obtain.
  - The one surviving gain is small: HoloAssist's instructor voice, +0.021 [0.006, 0.034].
  - It is mostly *whether the instructor spoke*.
- **The second contribution: the traps.**
  - Four ways a reaction study can produce a false positive, each one met in this project, with the numbers.
  - A fifth, a fusion model losing to its own strongest input.
- **Format:**
  - a Markdown draft, `docs/paper/draft.md`, with generated tables and figures;
  - main text about 5,000–7,000 words, plus appendices;
  - converting it to LaTeX or a venue template is a later maintainer decision (§8).

## 2. Claims: what the paper may and may not say

**It may claim exactly these.** Each claim is backed by the named facts (§4 keys).

| Id | Claim | Facts |
|---|---|---|
| **C1** | On a visible-outcome dataset (Oops!), reactions add nothing measurable to an action-only model | `oops` ablation Δ and its CI; `action-probe`; `judge-large` |
| **C2** | On a hidden-outcome dataset (HoloAssist), the instructor's voice adds a small amount, only where the instructor is audible. Most of it is whether they spoke | `holoassist_audible` ablation Δ; `react-nonverbal`; `react-spoke`; the recorded audible strata |
| **C3** | With people reacting to a physical robot (REFLEX), no reaction channel adds to action-only models. The face signal that passed the pre-registered rule is explained by window length | `reflex` and `reflex_resolve` ablation Δ; `react-face`; the B4b post-hoc diagnostics |
| **C4** | Reaction-labeled human video did not make a robot success detector more accurate with fewer labels (RoboReward) | the `roboreward` gain rows at every N; `react_minus_shuffled`; the recorded Wave D diagnostics |
| **C5** | Five traps, each with evidence: annotation text leaks labels; window length mimics reactions; privacy masks leak posture; single-seed shuffled controls give false alarms; a fusion model can lose to its own strongest input | the HoloAssist `step-prior` and verb `drop` rate; the REFLEX diagnostics; the leak-check rows and the 50-seed nulls; `fusion_minus_action_best` |

**It must not claim:**
- that reactions carry no information in general, or in live interaction (only these datasets, channels and encoders were tested);
- anything about verdict videos or taste tests (never obtained);
- that any result is "significant" except by its stated CI rule;
- that Wave B "passed" or was "supported." Wave B is reported as *PROCEED by the pre-registered rule, explained by a window-length artifact found in verification*, with both facts in the same paragraph;
- any capability of the MicroDuck or any robot (nothing ran on a robot).

## 3. Structure

| § | Section | Required content |
|---|---|---|
| — | Title, author line, abstract | Title: §8 lists the options; the draft uses option 1. Author line: `[AUTHOR NAME], Independent Researcher` (a placeholder for the maintainer). Abstract ≤ 200 words: question, data, pre-registration, result (C1–C4 in one sentence each, or fewer), the traps, the implication |
| 1 | Introduction | Robots learning from human reactions (EMPATHIC; error detection from reactions: BAD, ERR@HRI); the competing baseline, an action-only model or VLM judge; the question; four contributions (a pre-registered multi-dataset test; a harness with action-only controls and an ablation; a negative result; the traps) |
| 2 | Related work | Only the allowed citations (§6) |
| 3 | Hypotheses and pre-registration | H1's visible vs. hidden prediction; the ablation gain (`fusion − action-fusion`); the pass bar; the kill rule as restated November 1 (decision 28); H3's rule (Wave D). Every rule was written before its data was touched (dates from `00_thesis.md`). Post-hoc notes sit beside verdicts and never replace them |
| 4 | Data | Table T1. One paragraph per dataset: what an item is, its two windows, the label source, and the group unit. REFLEX's privacy mask (silhouette visible) is stated here |
| 5 | Methods | The conditions (`03_eval_harness.md` §6), encoders (§9), judges (§8), metrics (AUROC; grouped bootstrap, 1,000 resamples; paired Δ), shuffled controls (§11.9), and compute (one Mac Studio, no fine-tuning) |
| 6 | Results | T2, T3 and F1 for H1; T5 and F3 for Wave D. Text states each claim (C1–C4) with its numbers |
| 7 | What fooled us, and could fool you | 7.1 annotation text leaks labels (HoloAssist); 7.2 window length mimics reactions (REFLEX; T4, F2); 7.3 privacy masks leak posture (REFLEX; two frames described in words, no images of participants); 7.4 single-seed shuffled controls (T6); 7.5 a fusion model losing to its own prior |
| 8 | Discussion and limitations | Why reactions added little here (strong action-only models; visible events; outcomes set by task structure; speech presence). The data limits: no hidden-outcome verdict data with faces was obtainable; 17 test participants on REFLEX; one machine. What a decisive test needs: a hidden outcome, faces, equal windows, and an action channel with no person in it |
| 9 | Conclusion | One paragraph |
| A | Reproduction | Commands to regenerate the facts, tables, figures and every report (`docs/paper/README.md`) |
| B | Full results | Per-dataset scorecard tables (generated) |
| C | Recorded facts | Every `recorded` fact (§4.2) with its source path, section and verbatim quote |
| D | Data and ethics statement | Licenses as stated in `02_data_sources.md`; no new human data was collected; no human rating was used |

## 4. The fact base (`docs/paper/facts.json`, built by E1)

**Every number in the draft's prose must be a value in this file or in a generated table** (§5). The checker (E1) enforces this.

### 4.1 `computed` (built from the repository and `DATA_ROOT` by `tools/build_paper_facts.py`)

1. **Scorecard rows.** For each (dataset, condition), take the row with the latest `ts` and store `value`, `ci_low`, `ci_high`, `n_items` and `n_groups` under `scorecard/<dataset>/<condition>/<field>`.
   - **H1 datasets:** `oops`, `holoassist`, `holoassist_audible`, `reflex`, `reflex_resolve`; all their conditions.
   - **`roboreward`:** all 44 rows. **`holoassist_pool`:** both rows.
2. **The B4b post-hoc diagnostics.** Every key of `docs/evals/2026-11-04_reflex_posthoc.json`, under `posthoc/<key>/<field>`.
3. **Dataset facts,** from `items.jsonl` and the split files. Expected values:

   | Dataset | Items | Train (groups, positives) | Test (groups, positives) |
   |---|---|---|---|
   | `oops` | 2,710 | 1,638 (320, 819) | 1,072 (90, 536) |
   | `holoassist` | 5,000 | 3,000 (218, 1,500) | 2,000 (91, 1,000) |
   | `holoassist_audible` | 4,420 | 3,000 (131, 1,500) | 1,420 (53, 710) |
   | `reflex` | 1,208 | 834 (38, 417) | 374 (17, 187) |
   | `reflex_resolve` | 604 | 417 (38, 338) | 187 (17, 150) |
   | `roboreward` | train pool 4,000 (2,000 success); test 1,517 (527 success), from `items/roboreward/stats.json` | — | — |

4. **Named facts:**
   - `holoassist/verb_drop`: 435 items with `meta.verb == "drop"`, 415 of them mistakes (95.4%);
   - `reflex/reaction_frames_mean`: 45.7 for `fail` and 35.5 for `pre`, over all items;
   - `reflex_resolve/resolved_by_verb`: `pick` 219/219, `carry` 198/220, `place` 71/165;
   - `reflex_resolve/resolved_by_level`: high 165/187, medium 115/143, low 106/143, zero 102/131.
5. **Prediction checks.** The verdict strings that `tools/report_oops.py` `prediction_check`, `tools/report_holoassist.py` `prediction_check_hidden` and `tools/report_reflex.py` `prediction_check_reflex` / `evaluate_kill_rule` return for the latest rows. **Import them; never re-implement them.**
6. **Constants** (fixed by the contracts, not measured), under `constants/`:
   - chance AUROC 0.5;
   - the pass bars 0.65 (reaction-only AUROC) and 0.03 (ablation gain);
   - 1,000 bootstrap resamples;
   - 50 permutation seeds;
   - the 22-frame `fail` window extension;
   - REFLEX's 4.4 fps;
   - Wave D's budgets 50, 100, 200, 500 and 1000;
   - 55 REFLEX participants.

### 4.2 `recorded` (numbers that the designer measured and wrote into a design doc, but no script in the repository reproduces)

Each entry is `{"value", "source": "<path> <section>", "quote": "<verbatim sentence containing the number>"}`. These six, and no others:

| Key | Value(s) | Source |
|---|---|---|
| `wave_a/audible_strata` | +0.012 [−0.004, 0.030] where the instructor spoke; −0.026 where silent; +0.025 without `drop` | `ongoing_general_errors.md`, Resolved entry 6 (Issue 8) |
| `wave_d/head_zero_shot` | `react` 0.508 [0.475, 0.540]; `shuffled` 0.521 [0.487, 0.554]; `truth` 0.491 [0.457, 0.522] | `04_wave_d_robot_check.md` §8 |
| `wave_d/well_posed_head` | 0.550 [0.515, 0.585] | `04_wave_d_robot_check.md` §8 |
| `wave_d/head_norms` | 0.99, 1.03, 26.9 | `04_wave_d_robot_check.md` §8 |
| `wave_d/pool_mistakes` | 104 of 20,000 (0.52%) | `04_wave_d_robot_check.md` §8 |
| `holoassist/step_prior_shuffled_null` | 50-seed mean 0.496 (SD 0.030) | `03_eval_harness.md` §11.9 |

Appendix C lists all of them, so a reader can see which numbers no script in this repository reproduces.

## 5. Tables and figures (generated by E1; the draft includes them verbatim)

Each generated table is a Markdown file in `docs/paper/tables/`. The draft pastes it between the markers `<!-- table:T1 -->` and `<!-- /table:T1 -->`. The checker skips text between markers, but verifies that it matches the file byte for byte. Cells are written `0.xxx [0.xxx, 0.xxx]`, and a missing row is `n/a`.

| Id | Content |
|---|---|
| **T1** | Datasets. Columns: name, outcome visibility (`02_data_sources.md` table), reaction channel(s), items, train / test items, test groups, test positive rate |
| **T2** | H1 results. Columns: `oops`, `holoassist`, `holoassist_audible`, `reflex`, `reflex_resolve`. Rows: `judge`, `judge-large`, `action-probe`, `step-prior`, `action-fusion`, `react-nonverbal`, `react-face`, `react-spoke`, `fusion`, `fusion_minus_action_fusion` (ablation Δ), `fusion_minus_action_best` |
| **T3** | Pre-registered checks per H1 dataset: clause 1, clause 2, pass bar, and for `reflex` / `reflex_resolve` the kill-rule verdict. Verdict strings come from §4.1 item 5 |
| **T4** | REFLEX diagnostics: every `posthoc/*` key with value, CI and `n_test` |
| **T5** | Wave D: rows N ∈ {50, 100, 200, 500, 1000}. Columns: `detector:none`, `gain:react`, `gain:shuffled`, `gain:truth`, `react_minus_shuffled`. Then `robot-full` and `judge-large` |
| **T6** | Shuffled controls: per H1 dataset, the count passing (`ci_low ≤ 0.5`) out of the count run, plus every `FAIL` row with its 50-seed null (from the B4b JSON) |
| **F1** | Forest plot: the ablation Δ with its CI for the five H1 datasets, with vertical lines at 0 and at +0.03 (the pass bar). `docs/paper/figures/f1_ablation.png` |
| **F2** | REFLEX `react-face` AUROC with CI under: the original windows, means only, equal-length start, equal-length end; plus window length alone. Horizontal line at 0.5. `f2_window_length.png` |
| **F3** | Wave D: test AUROC against N (log axis) for the `none`, `react`, `shuffled` and `truth` detectors, with CI bands. `f3_wave_d.png` |

**Figure rules:**
- matplotlib only (already installed); PNG at 150 dpi;
- axis labels with units;
- markers distinguishable in grayscale;
- no images of any participant.

## 6. Citations

- **Allowed:** the works listed under "Core", "Data sources with outcome labels" and "Baselines we must beat or complement" in [`01_prior_work.md`](01_prior_work.md), plus REFLEX (arXiv 2502.14185). Use the identifiers given there.
- **Anything else** (encoders, judges, statistics, CANDOR, …) is written `[CITATION NEEDED: <what>]` for the maintainer. **Never invent an author, title, venue, year or identifier.**
- The "To read" list in `01_prior_work.md` is not allowed: its relevance is unverified.

## 7. Style

- Plain, precise sentences. Every result carries its CI.
- No hype words ("novel", "groundbreaking", "remarkably").
- First person plural ("we"), as is conventional.
- Small whole numbers in prose are spelled out ("three datasets"), so the checker sees only measured numbers.
- Section and lesson references (`§11.10`, `L18`) are allowed.

## 8. Decisions for the maintainer at draft review (not before)

1. **Title.**
   - Option 1 (used in the draft): *"Do Observer Reactions Add Information Beyond the Action? A Pre-registered Negative Result for Reaction-Based Robot Reward"*.
   - Option 2: *"Reactions as Reward, Revisited: Little Signal Beyond Action-Only Models, and Five Ways to Fool Yourself"*.
2. **The author line.**
3. **Where to publish:** for example an arXiv preprint (cs.RO), a workshop, or nowhere yet.
4. **What to release:** code, the scorecard, the generated tables. Releasing anything derived from HoloAssist, Oops! or REFLEX follows each dataset's license (`02_data_sources.md`).
5. **Filling the `[CITATION NEEDED]` markers.**
