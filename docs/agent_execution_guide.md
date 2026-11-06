# Agent Execution Guide: Active Build: A17b → E1c → E2c (A17's tests as specified; constants named, a license exemption, the pre-amendment fact, the task mix, the F1 legend; the draft corrected, round three), November 6, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:**
  - On October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
  - **Waves A, D and B are complete.** Wave B's pre-registered verdict, PROCEED, came through one clause that the designer found to be a window-length artifact. The maintainer recorded Wave B as negative in substance (Issue 12 = A), and the project is writing it up (Wave E, [`06_writeup.md`](06_writeup.md)).
  - **A18, A17, E1b and E2b were delivered** (`d6718bc`, `a2dcec4`, `ee06239`, `8b5aef0`). The designer verified them on November 6 (§5.1):
    - **The tooling is sound.** Everything rebuilds byte for byte, the paper's whole reproduction leaves `git status` clean, and the battery passes.
    - **The draft passes every mechanical check, but it is still wrong in places no check can see.**
      - It names a judge setting that does not exist and misnames the Oops! license.
      - It credits prior work, REFLEX included, with a result `01_prior_work.md` does not record.
      - It leaves out that `reflex_resolve`'s reaction window contains the outcome.
      - It gives the scorecard's later number as the one measured before the October 20 amendment.
      - It still overstates the pre-registration. Part of that came from the designer's own contract, now corrected (`06_writeup.md` §3).
    - **The checker** still lets any number equal to a contract constant pass in any paragraph.
    - **A17's tests** were fitted to its code rather than written as specified.
- **What is approved now, three items, in this order:**
  1. **A17b:** A17's tests, as specified.
  2. **E1c:**
     - constants count only when a paragraph names them;
     - a license-name exemption;
     - the recorded pre-amendment numbers;
     - the HoloAssist task-mix table B8;
     - F1's legend moved off the data.
  3. **E2c:** the draft corrected against its sources, round three.
- **What you must not touch:**
  - any scorecard row, cache, split or item file;
  - the pre-registered verdicts;
  - the maintainer's draft-review decisions (`06_writeup.md` §8).

**Every path, number, field name and literal string in this guide is a decision, not a suggestion. Implement as written; do not substitute your own values.**

---

## 0. Standing constraints (they apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare. G2 must pass **3 times in a row** with the `TMPDIR` below.
   - Read every exit code bare. A gate that did not run is reported as *not run*, never as passed.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only (no `match`, no runtime `X | Y`).
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies, models and downloads are frozen.** No new pip packages, no new model downloads, no Ollama pulls, no data downloads, and no network calls. Every citation string you need is already in `01_prior_work.md`.
4. **Data lives under `DATA_ROOT`** (`/Volumes/Extreme SSD/social_robotics`; exFAT).
   - **No item writes anything under `DATA_ROOT`.**
   - `touch` a marker file in your scratch directory before you start. When you finish, `find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f` must print nothing.
   - Never delete anything you did not create. Every glob over the SSD uses `shared.fsutil.data_glob` (L13).
5. **Memory (`03_eval_harness.md` §12).** No item loads a model. **`gemma4:26b` is shared with another project; never unload it.**
6. **The scorecard is append-only, and no item appends to it.**
7. **Pre-registered rules and recorded verdicts are never changed.** That covers `05_wave_b_reflex.md` §1–§6 and the PROCEED verdict. The designer's notes sit beside the verdict; they do not replace it.
8. **A test this guide specifies is part of the spec (L23).**
   - Write it first, with the specified inputs and expected values, and watch it fail.
   - If the code cannot meet it, STOP and file Issue 13. Never edit a specified expectation to fit the code.
   - Name every deviation in the commit body, however small.
9. **A general rule applies to every sentence, not only to the rows of a table (L24).** E2c lists every error the designer found. If you find another instance of the same error class, fix it the same way and list it in the commit body. The error classes:
   - a definition not restated from its contract;
   - a cited work described beyond `01`;
   - a fact key in prose;
   - a bare `§` reference to a project document.
10. **Commits:**
    - one item = one Conventional Commit on `main` (`test(a17b): …`, `feat(e1c): …`, `docs(e2c): …`);
    - the body holds the WHY, the red and green runs, and **every** validation bar's result, including any bar you missed (L14);
    - stage explicit paths only;
    - add the item's line to `docs/ongoing_general_errors.md` §3;
    - push with `/usr/bin/git push origin main`;
    - no branches, no PRs, never amend.
11. **When a contract and this guide disagree, or a bar cannot be met, STOP and file a new issue** (the next number is **Issue 13**) in `docs/ongoing_general_errors.md`, with options.
12. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer. No issue is open now.

---

## 1. Verified baseline (November 6, 2026; the designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6).
- **`memguard --status`:** 57.0 GB available, pressure 1, heavy lock free, no Ollama models loaded.
- **`venv`:** Python 3.9.6, torch 2.8.0, transformers 4.46.3, scikit-learn 1.6.1, scipy 1.13.1, numpy 2.0.2, matplotlib 3.9.4, ruff 0.15.11, pytest 8.4.2.

### 1.2 Repository

- `HEAD` is `8b5aef0` (`docs(e2b)`), plus the designer's verification commit for this guide. The working tree is clean.

### 1.3 Gates (run bare November 6, 2026 on `8b5aef0`, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · `All checks passed!` |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **143 passed, 11 deselected**, 3 runs out of 3 (71.69 s, 72.51 s, 70.89 s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **11 passed** (91.33 s) |
| — | `DATA_ROOT` marker | `find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f` after the battery | **prints nothing** (A18 holds) |

### 1.4 Reproduction, measured

| Command | Result |
|---|---|
| `tools/build_paper_facts.py --paper-dir <scratch>` | `facts.json`, T1–T6, B1–B7 and F1–F3 byte-identical to the committed files |
| `docs/paper/README.md` steps 1–4, run in place | every command exits 0; `git status --short` prints nothing; the Wave D note survives |
| `tools/check_paper_numbers.py docs/paper/draft.md` | exit 0 · `checked=821 matched=821 unmatched=0 blocks=44/44` |
| The same draft with constants counting only when named (E1c's rule, prototyped by the designer) | **41 tokens unmatched**: every one is a constant the paragraph uses without naming it |

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| 1 | **A17b**: A17's tests as specified | Independent and small. It restores the test that guards the paper's reproduction step before the paper's last round |
| 2 | **E1c**: constants named; license exemption; the pre-amendment fact; B8; the F1 legend | E2c cites the new recorded fact, pastes B8, writes `CC BY-NC-SA 4.0` and tags its constants. **After E1c the committed draft fails the checker (41 untagged constants) until E2c.** That is expected; record it in E1c's commit body |
| 3 | **E2c**: the draft corrected | It needs everything E1c builds |

After E2c, the queue is empty (§9).

---

## 3. The items

### Item A17b: A17's tests, as specified

**What this means for the user:** the test that guards the paper's reproduction step would still pass if the Wave D report generator stopped writing its report. Nobody would find out until a reader's reproduction silently produced nothing new.

**The gap:**
- **`tests/test_report_wave_d.py:68–80`, `test_wave_d_report_keeps_designer_note`,** asserts only two things: that the file ends with the note, and that it contains the note once.
  - If `generate_wave_d_report` wrote nothing at all, the old file would satisfy both.
  - A17's spec also asserted that the file starts with the regenerated report's first heading. That assertion is missing.
- **It asserts `endswith("\n---\n\n## Designer note …")`.** The spec's `"\n\n---\n\n…"` also checks the blank line before `---`.
- **`tests/test_report_reflex.py:479`, `test_ordinal`,** checks 10 of the 14 specified values. 22nd, 101st, 111th and 112th are missing. The function is right for all four (checked by the designer).
- **`test_designer_note_suffix_cases` asserts the helper as implemented.** The helper prepends `"\n---\n\n"` and adds a `---` before a note that has none. That behavior is **accepted** (§5.2); leave the helper and this test as they are.

**Implementation:**
1. In `test_wave_d_report_keeps_designer_note`, after `text = out.read_text(...)`, add:
   ```python
   assert text.startswith("# Wave D Robot Check Evaluation Report (2026-10-22)\n")
   assert text.endswith("\n\n---\n\n## Designer note (October 22, 2026; post hoc)\nkeep me\n")
   ```
   Keep the two existing asserts.
2. In `test_ordinal`, add:
   ```python
   assert ordinal(22) == "22nd"
   assert ordinal(101) == "101st"
   assert ordinal(111) == "111th"
   assert ordinal(112) == "112th"
   ```
3. Change no production code.

**Validation:**
- **Falsify the heading assertion.**
  - Temporarily replace the `target_path.write_text(...)` call in `tools/report_wave_d.py` (lines 193–196) with `pass`.
  - Run the test: it must fail on the `startswith` line. Today's version of the test passes against that mutation.
  - Restore the file, and the test passes. Report both runs.
- **Falsify the blank-line assertion.**
  - Temporarily make `with_designer_note` in `src/harness/reporting.py` return `generated.rstrip("\n") + suffix`.
  - The new `endswith` must fail; today's `endswith` passes against that mutation.
  - Restore the file.
- `git diff` after the restores shows only the two test files.
- `scripts/battery.sh` bare: G2 **143 passed** (no new test functions), 3 runs out of 3. The `DATA_ROOT` marker check prints nothing.
- Commit `test(a17b): assert the regenerated Wave D heading and the specified ordinals`.

**Blast radius:** `tests/test_report_wave_d.py`, `tests/test_report_reflex.py`, and the `docs/ongoing_general_errors.md` §3 line.

---

### Item E1c: constants count only when named; license exemption; the pre-amendment fact; the task mix; the F1 legend

**What this means for the user:**
- **The checker has a loophole.** Any number equal to one of the contract constants (0.5, 20, 50, 100, 1,000 and the rest) passes in any paragraph, whatever that paragraph claims. Once the maintainer starts editing the draft, a wrong "100" or "20" would pass unnoticed.
- **The paper also lacks things it needs:**
  - the numbers the designer actually saw before the October 20 amendment;
  - the HoloAssist task mix that `02_data_sources.md` requires next to every `holoassist_audible` result;
  - a checker that accepts the Oops! license's real name;
  - an F1 whose legend does not hide data.

**The gap:**
1. **Constants match everywhere.** `tools/check_paper_numbers.py:164–166` builds `const_numbers` from every `computed/constants/` value, and line 242 seeds every block's allowed set with them.
   - In today's draft, 41 tokens match nothing but a constant.
   - "We evaluate up to 20 conditions" (draft line 98) and "Inspecting 20 test items" (line 239) both matched `wave_d_draws_per_budget`.
2. **`CC_BY_RE` (line 35) is `CC\s+BY(?:-NC)?\s+…`.** E1b added the `-NC` without a spec line. The corrected Appendix D must write the Oops! license as `CC BY-NC-SA 4.0` (`02_data_sources.md`), and today its `4.0` would be checked.
3. **`HEADING_NUM_RE` and `LIST_NUM_RE` (lines 37–38)** were added by E1b without a spec line. They are accepted (§5.2), but no test pins them.
4. **The pre-amendment numbers are not in `facts.json`.** They are `step-prior` 0.751 [0.726, 0.774] and the combined action-only model 0.682 [0.658, 0.706], from `03_eval_harness.md` §6, "Designer reference values".
   - The draft quotes the scorecard's later `action-fusion` row (0.681 [0.657, 0.706]) in their place.
5. **No generated table holds the HoloAssist task mix.**
6. **F1's legend hides data.** `tools/build_paper_facts.py:613` sets `ax.legend(loc="lower right", …)`, which covers the upper end of `reflex_resolve`'s interval (0.078) and its end cap. This is the designer's own E1b spec error.
7. **The paper will need one more constant.** E2c states the grouped split as "70% of groups in train" (`03_eval_harness.md` §5), and no constant holds 0.7.

**Implementation:**
1. **Checker** (`tools/check_paper_numbers.py`):
   - Delete `const_numbers` (lines 164–166), and seed `allowed_numbers` with an empty set (line 242).
   - A constant is then allowed only when a block's tag names it, for example `computed/constants/chance_auroc`. Every other rule is unchanged.
   - Set `CC_BY_RE = re.compile(r"CC\s+BY(?:-(?:NC|SA|ND))*\s+\d+(?:\.\d+)?")`.
   - Keep `HEADING_NUM_RE` and `LIST_NUM_RE` exactly as they are.
2. **Recorded fact** (`tools/build_paper_facts.py`, `RECORDED_FACTS`): add this key after `holoassist/step_prior_shuffled_null` (line 106), exactly as written:
   ```python
   "holoassist/pre_amendment_reference": {
       "value": {
           "step_prior": {"value": 0.751, "ci_low": 0.726, "ci_high": 0.774},
           "action_fusion": {"value": 0.682, "ci_low": 0.658, "ci_high": 0.706},
       },
       "source": "docs/03_eval_harness.md §6",
       "quotes": [
           "`step-prior` **0.751 [0.726, 0.774]**",
           "`action-fusion` (judge + step-prior + SigLIP) **0.682 [0.658, 0.706]**",
       ],
   },
   ```
   `validate_recorded_facts` must accept it unchanged; both quotes occur verbatim in `03_eval_harness.md` §6.
3. **One constant** (the `constants` dict): add exactly `"grouped_split_train_share": 0.7,  # 03_eval_harness.md §5`. **Add no others;** a number that needs another constant is a STOP (Issue 13).
4. **B8** (`tools/build_paper_facts.py`):
   - Import `compute_task_mix_comparison` from `tools.report_holoassist`, beside `prediction_check_hidden` (line 30).
   - In `build_all_paper_artifacts`, after `_build_tables(...)` (line 825):
     - call `lines = compute_task_mix_comparison(data_root)`;
     - if `lines` is empty, raise `RuntimeError(f"B8: HoloAssist items not found under {data_root}")`;
     - write `paper_dir / "tables" / "B8.md"` as `"\n".join(lines) + "\n"`.
   - **Never re-implement the counting.**
5. **F1** (line 613): `ax.legend(loc="upper left", fontsize=8.5, frameon=True)`. Nothing else in F1 changes. F3's legend (line 718) stays where it is.
6. **`docs/paper/README.md`:** steps 2 and 3 list `B1.md`–`B8.md` where they now say `B1`–`B7`.
7. **Tests** (`tests/test_paper_tools.py`; fast):
   - **(falsifying)** `test_checker_constants_need_a_tag`:
     - facts: `{"computed": {"constants": {"wave_d_draws_per_budget": 20}, "scorecard": {"reflex": {"judge": {"value": 0.72}}}}, "recorded": {}}`;
     - draft: `"We evaluate up to 20 conditions; the judge scored 0.720.\n<!-- facts: computed/scorecard/reflex/judge -->\n"`;
     - assert exit 1 and `unmatched == ["20"]`;
     - then rewrite the draft with the tag `computed/scorecard/reflex/judge; computed/constants/wave_d_draws_per_budget` and assert exit 0.
     - *Today the first case exits 0.*
   - **Update `test_checker_scopes_numbers_to_paragraph_tags` (line 111).** Its tag becomes `computed/scorecard/oops; computed/constants/chance_auroc`, and it must still give exactly `unmatched == ["0.021"]`.
   - **Extend `test_checker_exempts_reference_numbers` (line 219):**
     - insert `CC BY-NC-SA 4.0; ` after `CC BY 4.0; ` in its draft;
     - append the text `"\n## 4 Data\n\n1. First item.\n"`;
     - it must still exit 0 with `checked=0`.
   - **`test_repo_recorded_facts_are_quoted`:** `validate_recorded_facts(RECORDED_FACTS)` against the real repository returns without error, and `"holoassist/pre_amendment_reference" in RECORDED_FACTS`.
8. **Rebuild** with `PYTHONPATH=src ./venv/bin/python tools/build_paper_facts.py`.

**Validation:**
- **Red first:** `test_checker_constants_need_a_tag` fails against today's checker.
- **The builder exits 0.** `git status --short docs/paper/` then lists only these files:
  - `facts.json`: the new recorded key and `grouped_split_train_share`. Check with a script that every other `computed` value is unchanged;
  - `tables/B8.md`: new;
  - `figures/f1_ablation.png`;
  - `README.md`: the B8 mention.
- **B8 equals the audible report's table byte for byte.** This must print nothing:
  ```bash
  sed -n '/^| Task | HoloAssist Items/,/^| \*\*Total/p' docs/evals/2026-10-21_holoassist_audible_h1.md | diff - docs/paper/tables/B8.md
  ```
- **Open F1.** The legend is at the upper left, and every marker, interval and end cap is visible, including `reflex_resolve`'s upper cap at 0.078.
- **Run the checker on the committed draft.** Expected first line: `checked=821 matched=780 unmatched=41 blocks=44/44`. Every unmatched token is a constant its block does not name. Put the first line in the commit body, and report any other count.
- **Battery:** `scripts/battery.sh` bare; G2 **145 passed**, 3 runs out of 3; the `DATA_ROOT` marker check prints nothing.
- **Commit:** `feat(e1c): constants count only when named; license exemption; pre-amendment fact; HoloAssist task mix; F1 legend`.

**Blast radius:**
- `tools/check_paper_numbers.py`, `tools/build_paper_facts.py`, `tests/test_paper_tools.py`;
- `docs/paper/facts.json`, `docs/paper/tables/B8.md`, `docs/paper/figures/f1_ablation.png`, `docs/paper/README.md`;
- the `docs/ongoing_general_errors.md` §3 line.

---

### Item E2c: the draft, corrected against its sources (round three)

**What this means for the user:**
- **Today:** the maintainer is about to review and sign this draft. It names a judge setting that does not exist, misnames a dataset's license, credits prior work with a result nobody recorded, hides that one dataset's reaction window contains the outcome, and describes the pre-registration as cleaner than it was.
- **After E2c:** every definition, license, identifier and description of prior work is the one its source states, and the pre-registration is described exactly.

**The gap.** Line numbers are in today's `docs/paper/draft.md`; each row names its source of truth.

| # | Line | Wrong today | Source of truth |
|---|---|---|---|
| 1 | 99 | "with strict audio isolation (`SR_NO_AUDIO`)". Nothing by that name exists: `grep -rn SR_NO_AUDIO src tools tests docs/0*.md` prints nothing | `03_eval_harness.md` §8: "No audio is ever sent." |
| 2 | 99 | "(`temperature = 0`)" is attached to `judge-large` alone | `03` §8: all three judges use temperature 0 |
| 3 | 69 | "All splits are group-disjoint (`default_rng(0)`) with locked file digests (Table `T1`)" | `03` §5. Oops! uses its official train/val split (`02`); RoboReward its own train/test split (`04` §2.1); HoloAssist and REFLEX a grouped 70/30 split (`05` §3.1). T1 shows no digests |
| 4 | 85 | Labels given as `"Correct"` / `"Wrong"` | `02`, HoloAssist schema: `Action Correctness` is `"Correct Action"` or one of three `"Wrong Action, …"` values; `otherwise` counts as a mistake |
| 5 | 52, 58 | The pre-amendment disclosure quotes the scorecard's later `action-fusion` 0.681 [0.657, 0.706] | `03` §6: the designer measured 0.682 [0.658, 0.706] and 0.751 [0.726, 0.774] before any implementation (E1c's recorded key) |
| 6 | 52 | The timeline omits what the amendment decided, and says each later evaluation "ran under rules fixed before its data was touched" | `06_writeup.md` §3 (corrected November 6) |
| 7 | 58 | "(amended October 20, 2026, Decision 20; November 1, 2026, Decision 28)" reads as two amendments | `00_thesis.md` H1; `05` §6: decision 28 only named REFLEX's headline probe |
| 8 | 55 | Clauses stated with the October 20 Δ under "(October 8, 2026)" | `00` H1 (the prediction); `03` §11.8 (the clause checks, amended October 20) |
| 9 | 85 | `holoassist_audible` is not reported as a subset | `02`: "Report it as a subset … Give its task mix next to `holoassist`'s." |
| 10 | 88, 169, 234, 277 | `reflex_resolve`'s reaction window, which contains the outcome, is never mentioned; `react-nonverbal` 0.660 is reported without it | `05_wave_b_reflex.md` §7, flaw 3 |
| 11 | 14 | "without explicit button presses" (EMPATHIC); all four prior works, REFLEX included, "showed that classifiers can predict … above chance (0.5)"; TAMER called "subsequent" to EMPATHIC, which it predates | `01_prior_work.md`'s tables; `02`, REFLEX |
| 12 | 39 | "Whereas prior studies test whether reactions beat chance (0.5) alone" | `01` records what each study did, not that they tested against chance alone |
| 13 | 42 | HoloAssist: "a HoloLens performer … while a remote instructor intervenes verbally on mistakes" | `02`: "An egocentric performer (HoloLens camera) completes physical tasks while a remote instructor watches the live feed and talks them through it." |
| 14 | 45 | DVD learned "from human and robot task videos" | `01`: "The established 'reward from human video' route, without reactions" |
| 15 | 22, 42 | v0 retired "because human raters could not reliably label emotions"; v0 "showed that reaction models must be trained on objective outcomes" | `00_thesis.md`, "What changed (October 2026) and why"; `06` §7 (no "establish"/"prove") |
| 16 | 29, 288 | "with frozen splits, models, and decision rules"; "five pre-registered human-reaction datasets"; "was exposed in verification as a window-length artifact" | §3's timeline; `06` §2 ("explained by a window-length artifact found in verification") |
| 17 | 271 | "on visible failures (`oops`, `reflex`)", though T1 calls `reflex` partly visible; "step identity strongly is associated" | T1; `02` |
| 18 | 277 | "a distant camera where `gemma4:26b` outputs 100 on all 374 items": the cause is not established | the Wave B report's constant-judge note |
| 19 | 22, 166, 256 | Fact keys in prose ("the `recorded` `wave_a/audible_strata` check"; "(`holoassist/step_prior_shuffled_null`)"); the audible strata are not called exploratory | `06` §7 (corrected November 6); `ongoing_general_errors.md`, Resolved entry 6: "exploratory strata" |
| 20 | 25, 103, 204, 237, 239, 242, 281 | Bare `§11.x` references to `03_eval_harness.md` read as sections of this paper, which uses bare `§` for its own sections | `06` §7 (corrected November 6) |
| 21 | 98, 239 | "up to 20 conditions" and "Inspecting 20 test items" are not measured numbers. They passed by matching the 20 draws per budget | `06` §7: small whole counts are spelled out |
| 22 | 295–300 | Appendix A names an item id ("before A17"); `export HF_HOME="$HF_HOME"` and `export TMPDIR="$TMPDIR"` do nothing; `DATA_ROOT` is never set | `06` §3 row A and §7; `src/config.py:13` reads `DATA_ROOT` from the environment |
| 23 | 545 | The Oops! license is printed as "`CC BY 4.0` non-commercial share-alike" | `02`: CC BY-NC-SA 4.0 |
| 24 | 546 | "Performer faces are blurred" | not in `02` |
| 25 | 520–538 | Appendix C lists six recorded keys | `06` §4.2: seven, since E1c |
| 26 | 324–420 | Appendix B lacks the task mix | `06` §5, B8 |
| 27 | all | Constants are used without being named in the tag | `06` §4 (changed November 6); E1c |

**Implementation:**
1. **Correct each row.** Where wording matters, it is given here; use it exactly.
   - **(1)** Replace "with strict audio isolation (`SR_NO_AUDIO`)" with "and are never sent audio".
   - **(2)** Delete "(`temperature = 0`)" and add, as the bullet's last sentence: "All three judges use temperature zero."
   - **(3)** Replace the sentence with: "All splits are group-disjoint and written once, with a checksum of their item lists. Oops! and RoboReward use their official splits; HoloAssist and REFLEX use a grouped split with 70% of groups in train (`default_rng(0)`)." Tag the paragraph with `computed/constants/grouped_split_train_share`.
   - **(4)** Replace "labeled `"Correct"` (`label = 1`) or `"Wrong"` (`label = 0`)" with "whose `Action Correctness` attribute is `"Correct Action"` (`label = 1`) or a mistake (`label = 0`: any of the three `"Wrong Action, …"` values, or `otherwise`)".
   - **(5, 6)** Replace §3's opening paragraph (line 52) with this paragraph and its tag:
     > **Pre-registration timeline.** The H1 prediction of October 8, 2026 preceded all data. The H1 pass bar was proposed the same day and amended once, on October 20, 2026, after the Oops! and HoloAssist results: its Δ became the ablation gain (`fusion_minus_action_fusion`), and a `step-prior` control was added. As disclosed in `00_thesis.md` Decision 20, the designer had measured the two numbers behind that change on the HoloAssist test labels before the amendment: the combined action-only model (judge, `step-prior` and SigLIP features) at 0.682 [0.658, 0.706], against `step-prior` alone at 0.751 [0.726, 0.774]. The amendment decided two later outcomes: under the original Δ (`fusion_minus_action_best`), `holoassist_audible`'s clause two would not have held (-0.065 [-0.113, -0.025]), and Wave D's start condition, amended the same day, would not have been met. Every later evaluation ran under rules fixed before any of its reaction features were extracted or any of its conditions was scored: `holoassist_audible` (rules of October 20, run October 21; it shares HoloAssist's sessions and group split, whose test labels the amendment used), Wave D (specified October 21, after we read test-row metadata, rewards included, through the Hugging Face viewer API, but before any robot video was downloaded), and Wave B (specified November 1, after we inspected the archive's file layout and annotation tables, label counts included, but before any feature was extracted). Post-hoc notes sit beside pre-registered verdicts and never replace them.
     > `<!-- facts: recorded/holoassist/pre_amendment_reference; computed/scorecard/holoassist_audible/fusion_minus_action_best -->`
   - **(5, 7)** Replace the pass-bar bullet (line 58) with:
     > - **H1 pass bar (proposed October 8, 2026; amended once, October 20, 2026, Decision 20; REFLEX's headline probe, `react-face`, set November 1, 2026, before any REFLEX feature was extracted, Decision 28):** headline reaction-only AUROC >= 0.65 **and** paired ablation gain `fusion_minus_action_fusion` >= +0.03 with `ci_low > 0.0`. The amendment replaced `fusion_minus_action_best` with `fusion_minus_action_fusion` because, on `holoassist`, concatenating SigLIP features with `step-prior` and `judge` (0.682 [0.658, 0.706] in the designer's pre-amendment measurement) diluted `step-prior` (0.751 [0.726, 0.774]); holding the combiner fixed isolates the marginal contribution of reactions.
     - The block's tag names `recorded/holoassist/pre_amendment_reference` and every constant it uses.
   - **(8)** Replace "H1 predicts contrasting behavior for visible versus hidden outcomes (October 8, 2026):" with "H1 predicted on October 8, 2026 that reactions add little where the outcome is visible and add information where it is hidden. The clause checks below are those of `03_eval_harness.md` §11.8, as amended on October 20, 2026:"
   - **(9)** End §4's HoloAssist paragraph with: "`holoassist_audible` is reported as a subset: device-setup tasks, where the instructor is most often audible, are over-represented, and the audibility rule was chosen after the twenty-session check failed but before any `holoassist_audible` result existed (task mix: Table `B8`, Appendix B.8)."
   - **(10)** Four edits:
     - in §4's REFLEX paragraph, after the `reflex_resolve` windows, add: "Its reaction window runs through `Resolution`, while its action window stops before it, so the reaction channels watch the resolution happen and the action-only channels do not (a flaw of our specification, found in verification). Cut at the end of `Explanation`, before the outcome, `react-face` scores 0.545 [0.461, 0.634] and `react-nonverbal` 0.643 [0.480, 0.843] (104 items)." Add the two `computed/posthoc/reflex_resolve.*_before_resolution` prefixes to the tag;
     - in C3, after "`react-nonverbal` at 0.660 [0.545, 0.766]", insert " (its reaction window includes the resolution itself, §4)";
     - begin §7.2's last bullet with "**Pre-resolution truncation on `reflex_resolve` (removing the outcome from the reaction window, §4):**";
     - in §8's REFLEX bullet, add "and a `reflex_resolve` reaction window that includes the outcome".
   - **(11)** Replace the two sentences from "In the laboratory, EMPATHIC" through "above chance (0.5)." with:
     > In the laboratory, EMPATHIC recorded the faces of people watching an agent act sub-optimally and learned a mapping from their implicit feedback to task statistics (Cui et al., CoRL 2020; arXiv:2009.13649). Related studies predicted trainers' explicit feedback from their facial expressions (Li et al., arXiv 2020; arXiv:2001.08703), and BADNet predicted failure from bystanders' webcam reactions to videos of human and robot task failures (Bremers et al., IROS 2023; arXiv:2303.04835). The ERR@HRI challenges study multimodal detection of robot errors from human reactions (Spitale et al., arXiv 2024; arXiv:2407.06094; Parreira et al., `ERR@HRI 3.0` at ICMI 2026; arXiv:2607.11570), and REFLEX recorded people's reactions to a physical robot's failures and explanations (Khanna et al., HRI 2025; arXiv:2502.14185; dataset by Naoum and Khanna, Zenodo 2025; doi:10.5281/zenodo.14590226).
   - **(12)** Replace the sentence with: "Whereas these studies predict failures or feedback from reactions, we test whether reactions add information *above* action-only probes, step priors, and VLM judges."
   - **(13)** Replace "where a HoloLens performer executes physical tasks while a remote instructor intervenes verbally on mistakes" with "where an egocentric performer completes physical tasks while a remote instructor watches the live feed and talks them through it".
   - **(14)** Replace "learned generalizable robot reward functions from human and robot task videos without bystander reactions" with "learned generalizable robot reward functions from human videos, without reactions".
   - **(15)** Two replacements:
     - line 22's opening clause becomes: "After retiring an earlier prototype (`v0-saf-final`, October 8, 2026), whose categorical-emotion labels could not be validated because it had no outcome variable and the human ratings it relied on had no answer for most moments, we redesigned our study around outcome supervision, pre-registered pass bars, and falsification gates.";
     - line 42's first sentence becomes: "Our retired prototype (`v0-saf-final`) is why we supervise on objective outcomes rather than on ratings of emotion."
   - **(16)** Three replacements:
     - "with frozen splits, models, and decision rules" → "with splits, models and decision rules fixed before each evaluation and one disclosed amendment (§3)";
     - "Across five pre-registered human-reaction datasets" → "Across five human-reaction datasets";
     - "was exposed in verification as a window-length artifact" → "is explained by a window-length artifact found in verification".
   - **(17)** Two replacements:
     - "First, on visible failures (`oops`, `reflex`)," → "First, where the failure is visible (`oops`) or partly visible (`reflex`) in the action video,";
     - "step identity strongly is associated" → "step identity is strongly associated".
   - **(18)** "a distant camera where `gemma4:26b` outputs 100 on all 374 items" → "`gemma4:26b` answering 100 on all 374 items".
   - **(19)** Three replacements:
     - line 166: "and the `recorded` `wave_a/audible_strata` check shows the ablation gain is" → "and exploratory strata, computed after the result (Appendix C), put the ablation gain at";
     - line 256: "(`holoassist/step_prior_shuffled_null`)" → "(Appendix C)";
     - line 22: after "+0.012 [-0.004, 0.030] within spoken items", insert ", an exploratory stratum".
   - **(20)** Every bare `§11.x` names its document: "`03_eval_harness.md` §11.10". In the three headings, write "(L18; `03_eval_harness.md` §11.10)", and likewise for §11.11 and §11.9.
   - **(21)** "up to 20 conditions" → "up to twenty conditions"; "Inspecting 20 test items" → "Inspecting twenty test items".
   - **(22)** Appendix A: replace the paragraph before the code block with:
     > All facts (`docs/paper/facts.json`), tables (`docs/paper/tables/T1.md`–`T6.md`, `B1.md`–`B8.md`), figures (`docs/paper/figures/`), and evaluation reports (`docs/evals/*.md`) cited in this paper can be rebuilt deterministically from the data directory and `results/scorecard.jsonl` with the commands below. Set `DATA_ROOT` to the directory that holds the datasets' items, features and judge caches, and `HF_HOME` and `TMPDIR` as for any run; `docs/paper/README.md` gives this machine's values. Step four regenerates the five dated reports in `docs/evals/` in place, and each report generator keeps the designer's note at the end of its output file. Afterwards `git status --short docs/evals docs/paper` prints nothing:
     - Replace the block's first three lines with:
       ```bash
       export DATA_ROOT=/path/to/social_robotics_data
       export HF_HOME=/path/to/huggingface_cache
       export TMPDIR=/path/to/tmp
       ```
   - **(23)** "under its non-commercial research license (`CC BY 4.0` non-commercial share-alike)" → "under its license, CC BY-NC-SA 4.0 (non-commercial research and educational use)".
   - **(24)** "Performer faces are blurred and audio tracks are pitch-shifted by the dataset creators for participant privacy." → "Its audio tracks are pitch-shifted by the dataset's creators for anonymization."
   - **(25)** In Appendix C, "six diagnostic quantities" becomes "seven recorded entries". After the `holoassist/step_prior_shuffled_null` entry, add this entry and its tag:
     > **`holoassist/pre_amendment_reference`:** `step-prior` 0.751 [0.726, 0.774]; the combined action-only model (judge + `step-prior` + SigLIP) 0.682 [0.658, 0.706]; both measured by the designer on the `holoassist` test labels before the October 20, 2026 amendment. Source: `docs/03_eval_harness.md` §6. Verbatim quotes: *"`step-prior` 0.751 [0.726, 0.774]"* *"`action-fusion` (judge + step-prior + SigLIP) 0.682 [0.658, 0.706]"* (bold markers omitted).
     > `<!-- facts: recorded/holoassist/pre_amendment_reference -->`
   - **(26)** After Appendix B.7, add "### Appendix B.8: HoloAssist Task Mix (`holoassist` vs. `holoassist_audible`)", then `<!-- table:B8 -->`, the content of `docs/paper/tables/B8.md` verbatim, and `<!-- /table:B8 -->`. Appendix B's opening sentence adds "and the HoloAssist task mix (B8)".
   - **(27)** Every block that uses a constant names it in its tag (`computed/constants/<key>`). E1c's checker finds every one; today there are 41 tokens.
2. **Apply the error classes to the whole draft** (§0.9). List every further instance you fix in the commit body. If you find none, say so.
3. **Change nothing else.** The structure, the claims, the tables T1–T6 and B1–B7, and the remaining text stay as they are.
4. **`docs/paper/README.md`:**
   - step 0 adds `export DATA_ROOT="/Volumes/Extreme SSD/social_robotics"` (this machine's value);
   - step 4's "preserving each file's trailing `*Designer note*` block" becomes "keeping each file's `## Designer note` section".

**Validation:**
- **The checker:** `PYTHONPATH=src ./venv/bin/python tools/check_paper_numbers.py docs/paper/draft.md` exits 0 with `unmatched=0`, and `blocks=` shows every numeric block tagged. Put its first line in the commit body.
- **E2b's checks still hold:**
  - none of its 24 banned strings occurs;
  - every one of its required strings occurs, scoped as before. The §3 timeline block now starts with `**Pre-registration timeline.**`;
  - every prose block that contains `PROCEED` also contains `window`;
  - no sentence calls Wave B "passed" or "supported";
  - `grep -niw "novel\|groundbreaking\|remarkabl[ey]"` returns nothing.
- **New banned strings.** A scripted check that none of these occurs in `draft.md`:
  - `SR_NO_AUDIO`, `non-commercial share-alike`, `faces are blurred`, `"Correct"` and `"Wrong"` (each with its double quotes);
  - `above chance (0.5)`, `explicit button presses`, `beat chance (0.5) alone`, `intervenes verbally`, `human and robot task videos`;
  - `must be trained on objective outcomes`, `could not reliably label`, `frozen splits`, `five pre-registered`, `exposed in verification`, `distant camera`, ``visible failures (`oops`, `reflex`)``, `strongly is associated`;
  - `A17`, `HF_HOME="$HF_HOME"`, `All splits are group-disjoint (`, `up to 20 conditions`, `Inspecting 20`.
- **Scoped checks.** The same script checks each of these:
  - **no line containing `§11.`** lacks `03_eval_harness.md`;
  - **`wave_a/audible_strata` and `holoassist/step_prior_shuffled_null`** occur only in tags and after the `## Appendix C` heading;
  - **the §3 block starting `**Pre-registration timeline.**`** contains `0.682 [0.658, 0.706]`, `0.751 [0.726, 0.774]`, `-0.065 [-0.113, -0.025]`, `Wave D's start condition`, `October 20`, `test labels`, `viewer API`, `label counts` and `before any feature was extracted`, and does **not** contain `0.681`;
  - **the §4 block starting `**HoloAssist`** contains `[start, end + 5.0]`, `pitch-shifted`, `both sides of the split`, `"Correct Action"`, `otherwise`, `over-represented` and `B8`;
  - **the §4 block starting `**REFLEX`** contains `watch the resolution happen`, `0.545 [0.461, 0.634]` and `0.643 [0.480, 0.843]`;
  - **the block starting `**Claim `C2``** contains `exploratory`;
  - **the block starting `**Claim `C3``** contains `includes the resolution itself`;
  - **the block starting `**Claim `C4``** contains `reaction labels cannot be judged here` and `exploratory`;
  - **anywhere:** `CC BY-NC-SA 4.0`, `export DATA_ROOT=`, `<!-- table:B8 -->` and `holoassist/pre_amendment_reference`.
- **Citation markers:** list every remaining `[CITATION NEEDED: …]` marker in the commit body. The list should match E2b's 13.
- **Length:**
  - the abstract is ≤ 200 words;
  - sections 1–9, by `wc -w` after deleting every `<!-- … -->` comment, with tables, are between 5,000 and 7,000;
  - report the prose-only count too.
- **Reproduction:** run README steps 1–4 in place; `git status --short docs/evals docs/paper` prints nothing.
- **The rendered draft:** open it and confirm that the three figures load, F1's legend covers no data, and every table, B8 included, renders as a table.
- **Battery:** `scripts/battery.sh` bare; G2 **145 passed**, 3 runs out of 3; the `DATA_ROOT` marker check prints nothing.
- **Commit:** `docs(e2c): correct the draft's definitions, license, prior-work claims and pre-registration timeline against their sources`.

**Blast radius:** `docs/paper/draft.md`, `docs/paper/README.md`, and the `docs/ongoing_general_errors.md` §3 line.

---

## 4. Deferred: do NOT start

| Work | Status / trigger | Who writes the spec |
|---|---|---|
| **The maintainer's draft review** (title, author line, venue, release, remaining citations; `06_writeup.md` §8) | After E2c **and** the designer's verification of it | The maintainer decides; the designer turns it into items |
| **A decisive hidden-outcome test with faces** (DW8; for example CANDOR) | **Maintainer opt-in only.** The designer drafts any request, and the maintainer approves it before it is sent | The designer |
| **Wave B v2** (REFLEX with the flaws fixed) | **Not selected** (Issue 12 = A) | — |
| **Wave C: H2 transfer** (DW3) | **Not selected** (Issue 12 = A) | — |
| **Robot transfer, revisited** (DW7) | **Not selected** (Issue 12 = A); its trigger, a Wave B reaction gain, was not met | — |
| **Stage A live MicroDuck demo** (DW6) | **Paused:** it needs a validated reaction reader, hardware, and the maintainer's opt-in | The designer |
| **Wave D v2** | **Not selected** (Issue 10 = A) | — |
| **Ego4D false-positive set** (DW4) | **Not selected** | — |

**The note behind Issue 12, so you do not "fix" it yourself:**
- A `fail` item's reaction window is longer than a `pre` item's (45.7 vs. 35.5 frames), and `reflex-face-v1`'s `max`/`std` features grow with length.
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
| A11–A14 | `abbc273`, `608c729`, `eea8933`, `d6d9c54` | `step-prior`, `action-fusion` and the ablation Δ; per-session audibility; `holoassist_audible`; the Wave A completion |
| A15 | `347f242` | the computed caveat; the one-sided leak label in the reports |
| D1–D4 | `f3f982c`, `d18dbbf`, `03ea7eb`, `315f3a6` | the Wave D v1 pool, the transfer code, the RoboReward download and its curve, the report and the verdict |
| B1–B4 | `f4c9939`, `5f0631b`, `0d146dd`, `dcb44be` | REFLEX: the download, items and split; the encoders and conditions; features and judges; 40 scorecard rows, the exported face probes (reproducing 0.6085), the report and the TSVs |
| A16 | `1551391` | `_say_to_file` with a 60 s timeout, 3 attempts, partial-file cleanup and a `RuntimeError`; three fast tests |
| B4b | `5f2d855` | §3.3 of the Wave B report computed from data; `tools/diag_reflex_posthoc.py` and its JSON (all 12 values exact) |
| E1 | `7154531` | `tools/build_paper_facts.py`, `tools/check_paper_numbers.py`, `facts.json`, T1–T6 and F1–F3 |
| E2 | `e470487` | the first full draft and `docs/paper/README.md` |
| A18 | `d6718bc` | the autouse fixture that keeps the tests' memory-guard events in a temporary directory; `test_log_event_goes_to_tmp_under_pytest`. The real log no longer changes |
| A17 | `a2dcec4` | `designer_note_suffix`, `with_designer_note` and `ordinal` in `src/harness/reporting.py`, used by both report generators. All five reports regenerate in place with their notes. **A17b completes its tests** |
| E1b | `ee06239` | the paragraph-scoped checker; recorded quotes covering every recorded number; T3, T4 and F1 fixed; F2 and F3 sizes read from the data; B1–B7. **E1c closes the constants loophole and moves F1's legend** |
| E2b | `8b5aef0` | the 27 listed corrections; every numeric block tagged; B1–B7 pasted; the README's in-place reproduction. **E2c corrects what the list missed** |

### 5.2 Accepted equivalents (do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure.** `extract_frame` retries a seek up to 0.3 s earlier.
- **`GeminiJudge`:** 32 output tokens, thinking off, and a 429 is not cached. `guard()` holds the heavy lock through the model load only.
- **Scorecard rows carry the parent commit's `git_sha` with `dirty: true`.**
- **The HoloAssist group partition is computed over all 1,758 sessions,** and `holoassist_audible` reuses it.
- **`shared/memguard.py` logs `waited_ms = 0` iff admitted on the first check.**
- **`DATA_ROOT/judge/roboreward/gemma4_26b/scores.jsonl`** is a stray duplicate from D3. Leave it.
- **Wave D v1's pool and its overfit true-label head** are part of v1's record (`04` §8), not bugs.
- **B1 normalized `C3-2`'s misspelled header `Resloved?`:** 604 `reflex_resolve` items (488 / 116).
- **The exported `{dataset}_face_probe.npz` files have no `dataset` key.** The file name carries it.
- **`react-nonverbal|spoke=1` equals `react-nonverbal` on REFLEX by construction.**
- **`meta.transcript` is empty and `react-full` did not run on REFLEX** (`Transcript` vs. `Text`). Only a v2 changes it.
- **The `reflex_resolve` `react-nonverbal:shuffled` row stays labeled `FAIL`,** with its permutation null beside it.
- **E1's two extra constants:** `h3_pass_bar_gain` 0.02 (`00_thesis.md` H3) and `wave_d_draws_per_budget` 20 (`04` §4).
- **`facts.json` keeps both nested and flat keys in `computed`.** They flatten to the same paths.
- **F3 shows the `truth` curve below the other three.** The figure is right.
- **The oops `judge-frontier` row has `n_items` 50 and `n_groups` 90.** The harness records the test split's groups. It is shown as stored.
- **A17's `designer_note_suffix` returns the note with a leading `"\n---\n\n"`, and inserts a `---` when the source has none.** No current report lacks the separator, and the output is byte-identical on every real report. `test_designer_note_suffix_cases` tests that behavior.
- **E1b's three unlisted exemptions:** a heading's leading section number (`HEADING_NUM_RE`), an ordered-list marker (`LIST_NUM_RE`), and `CC BY` variants (E1c generalizes the last).
- **E2b's length was reported both with the tags (6,775) and without them (6,441).** From E2c on, the count is taken without the `<!-- … -->` comments.
- **F2's x tick labels carry `n` as well as its point labels.**

### 5.3 Maintainer decisions

- **October 8–10:**
  - outcome supervision only, with no human rating;
  - "just ask an LLM" is the control;
  - Mac Studio only;
  - commit to `main`;
  - Issue 5 → B.
- **October 20–22:** Issue 6 → A; Issue 7 → A; Issue 8 → A; Issue 9 → B; Issue 10 → A.
- **October 23:** Issue 1 → E (the two requests were sent; M4 bounced and M5 got no reply).
- **November 1:** Wave B moves to REFLEX (`00_thesis.md` entry 28).
- **November 2:** Issue 11: wait for the external ComfyUI job rather than stop it.
- **November 5:** **Issue 12 → A.** Wave B is recorded as negative in substance, and the project writes up (`06_writeup.md`). Wave B v2, Wave C (DW3) and DW7 are not selected; DW6 is paused.

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **Grouped splits and CIs. Split files are written once. The scorecard is append-only.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels. The leak check is one-sided.**
- **Existing caches, items, splits and scorecard rows are never modified.** Post-hoc diagnostics never become scorecard rows.
- **No number in a generated table, figure or report is typed in by hand** (L10). **No citation detail is written from memory** (L21).
- **A specified test is written as specified** (L23).

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
| The thesis, H1–H3, the decision log (entries 1–32) | `00_thesis.md` |
| Prior work, and **the verified bibliographic records the paper cites** | `01_prior_work.md` |
| Datasets: the HoloAssist item definition, schema and caveats; the `holoassist_audible` subset rule; REFLEX "As built"; the licenses | `02_data_sources.md` |
| Splits (§5), conditions and the pre-amendment reference values (§6), judges (§8), encoders (§9), guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| Wave D: the contract and the v1 result (§8) | `04_wave_d_robot_check.md` |
| Wave B: the REFLEX contract (§1–§6), plus the v1 result, its flaws and the designer's notes (§7) | `05_wave_b_reflex.md` |
| **Wave E, the write-up contract:** claims, structure, the pre-registration timeline (§3, corrected November 6), the fact base and tags (§4), tables and figures (§5), citations (§6), style (§7), the maintainer's decisions (§8) | `06_writeup.md` |
| Issues 1–12 (all resolved); deferred work DW1–DW8; lessons L1–L24 | `ongoing_general_errors.md` |
| Reports | `docs/evals/` |
| The paper, its facts, tables and figures | `docs/paper/` |

---

## 7. Validation standard (for every item)

- **Red first, then green, then falsify.**
  - Each item names its falsifying test or mutation. Run it against today's code before your change, and see it fail.
- **Reproduce the stated reference values exactly.** A mismatch is a STOP, never a silent update.
- **A flaky gate is a failing gate.** Run G2 three times with `TMPDIR` on the SSD.
- **Open every artefact you produce:** the figures, the regenerated reports and the rendered draft.
- **Report every validation bar in the commit body, met or missed** (L14), and every deviation from a specified test (L23).
- **Never loosen a bar or change a pre-registered rule to pass it.** File it as **Issue 13**.

---

## 8. THE LOOP

```
For each item in §2 (A17b -> E1c -> E2c):
  (0) touch a marker file in your scratch directory.
  (1) Read the item in §3, and the sections and files it names.
  (2) Write the specified (falsifying) test first, exactly as specified, and watch it fail (red);
      implement (green); run every specified mutation and watch it fail.
  (3) Run scripts/battery.sh bare (G2 3 times with TMPDIR="/Volumes/Extreme SSD/tmp").
      Run the item's own checks (the builder and checker for E1c; every scripted check and the
      in-place reproduction for E2c).
      find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f  must print nothing.
  (4) Add the item's line to docs/ongoing_general_errors.md §3, stage explicit paths,
      commit on main, and push with /usr/bin/git push origin main.
```

---

## 9. Definition of Done for this state

- [x] A1–A15, D1–D4, B1–B4, A16, B4b, E1, E2, A18, A17, E1b and E2b delivered and verified (§5.1).
- [x] Battery measured bare on November 6 (§1.3), with reproduction measured (§1.4).
- [x] The designer's corrections recorded:
  - `06_writeup.md` §3–§8 (the timeline, the data rows, Appendix A, constants by tag, the seventh recorded key, B8, F1's legend, the reference and exploratory rules, the title note);
  - lessons L23–L24;
  - `00_thesis.md` entry 32.
- [ ] **A17b:** the Wave D test asserts the regenerated heading and the blank line; `test_ordinal` has all 14 values; both mutations fail; G2 shows 143 passed.
- [ ] **E1c:** constants count only when named; `CC BY-NC-SA` exempt; `holoassist/pre_amendment_reference` recorded; `grouped_split_train_share` added; B8 equals the audible report's table; F1's legend covers no data; G2 shows 145 passed.
- [ ] **E2c:**
  - the draft passes the checker, E2b's string checks and the new banned and scoped checks;
  - every further instance of an error class is fixed and listed;
  - the reproduction leaves `git status` clean.
- [ ] **Then the queue is empty.** The only legitimate triggers for new work are:
  - the designer's verification of E2c;
  - the maintainer's draft-review decisions (`06_writeup.md` §8);
  - a new issue the maintainer raises.
  - **Do not invent work.**
