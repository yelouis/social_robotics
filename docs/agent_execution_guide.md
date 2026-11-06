# Agent Execution Guide: Active Build: A18 → A17 → E1b → E2b (test hygiene; reports keep their notes; a checker that ties numbers to facts; the draft corrected), November 5, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:**
  - On October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
  - **Waves A, D and B are complete.** Wave B's pre-registered verdict, PROCEED, came through one clause that the designer found to be a window-length artifact. The maintainer recorded Wave B as negative in substance (Issue 12 = A), and the project is writing it up (Wave E, [`06_writeup.md`](06_writeup.md)).
  - **A16, B4b, E1 and E2 were delivered** (commits `1551391`, `5f2d855`, `7154531`, `e470487`), and the designer verified them the same day (§5.1).
    - **The tooling is sound:** everything reproduces byte for byte.
    - **The first draft is not.** Its HoloAssist windows are invented, "reactions add at most +0.021" is false, its pre-registration claim is wrong, the HoloAssist caveats are missing, and three reference titles and the REFLEX paper's first author are wrong.
    - **The number checker could not see any of it.** It matched each number against all 474 numbers in the fact base, and it skipped inline code (lessons L20–L22 in `ongoing_general_errors.md`).
- **What is approved now, four items, in this order:**
  1. **A18:** the fast tests stop writing the real memory-guard log on the SSD.
  2. **A17:** the report generators keep the designer's notes when they regenerate, and print correct ordinals.
  3. **E1b:** the checker ties every number to the facts its paragraph names. Recorded quotes cover every recorded number. T3, T4 and F1 are fixed, and Appendix B is generated.
  4. **E2b:** the draft, corrected point by point and tagged.
- **What you must not touch:**
  - any scorecard row, cache, split or item file;
  - the pre-registered verdicts;
  - the maintainer's draft-review decisions (`06_writeup.md` §8).

**Every path, number, field name and literal string in this guide is a decision, not a suggestion. Implement as written; do not substitute your own values.**

---

## 0. Standing constraints (they apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare (`--slow` for A18). G2 must pass **3 times in a row** with the `TMPDIR` below.
   - Read every exit code bare. A gate that did not run is reported as *not run*, never as passed.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only (no `match`, no runtime `X | Y`).
   - Before running anything:
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies, models and downloads are frozen.** No new pip packages, no new model downloads, no Ollama pulls, no data downloads, and no network calls (the citation strings you need are already in `01_prior_work.md`).
4. **Data lives under `DATA_ROOT`** (`/Volumes/Extreme SSD/social_robotics`; exFAT).
   - **No item writes anything under `DATA_ROOT`.**
   - **From A18 on this is checked:** `touch` a marker file in your scratch directory before you start; when you finish, `find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f` must print nothing.
   - Never delete anything you did not create.
   - Every glob over the SSD uses `shared.fsutil.data_glob` (L13).
5. **Memory (`03_eval_harness.md` §12).** No item loads a model except G4's existing slow tests. **`gemma4:26b` is shared with another project; never unload it.**
6. **The scorecard is append-only, and no item appends to it.**
7. **Pre-registered rules and recorded verdicts are never changed.** That covers `05_wave_b_reflex.md` §1–§6 and the PROCEED verdict. The designer's notes sit beside the verdict; they do not replace it.
8. **Commits:**
   - one item = one Conventional Commit on `main` (`test(a18): …`, `fix(a17): …`, `feat(e1b): …`, `docs(e2b): …`), with the WHY, the red and green runs, and **every** validation bar's result in the body, including any bar you missed (L14);
   - stage explicit paths only;
   - add the item's line to `docs/ongoing_general_errors.md` §3;
   - push with `/usr/bin/git push origin main`;
   - no branches, no PRs, never amend.
9. **When a contract and this guide disagree, or a bar cannot be met, STOP and file a new issue** (the next number is **Issue 13**) in `docs/ongoing_general_errors.md`, with options.
10. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer. No issue is open now.

---

## 1. Verified baseline (November 5, 2026, evening; the designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6).
- **`memguard --status`:** 57.0 GB available, pressure 1, heavy lock free, no Ollama models loaded.
- **`venv`:** Python 3.9.6, torch 2.8.0, transformers 4.46.3, scikit-learn 1.6.1, scipy 1.13.1, numpy 2.0.2, matplotlib 3.9.4, ruff 0.15.11, pytest 8.4.2.

### 1.2 Repository

- `HEAD` is `e470487` (`docs(e2)`), plus the designer's verification commit for this guide. The working tree is clean.

### 1.3 Gates (run bare November 5, 2026 on `e470487`, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · `All checks passed!` |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **132 passed, 11 deselected**, 3 runs out of 3 (69.01 s, 69.52 s, 69.25 s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **11 passed** (79.17 s). A16's timeout holds: no hang |
| — | Side effect | `find "$DATA_ROOT" -newer <marker> -type f` after the battery | **`runs/memguard.log` changed**: each G2 run appended 8 lines from `tests/test_judge.py`. A18 fixes this |

### 1.4 Reproduction, measured (each generator was run into a scratch directory and compared with `cmp`)

| Command | Result |
|---|---|
| `tools/diag_reflex_posthoc.py` | JSON byte-identical |
| `tools/report_reflex.py --date 2026-11-04` (existing report copied in first) | report and both TSVs byte-identical |
| `tools/build_paper_facts.py --paper-dir <scratch>` | `facts.json`, T1–T6, F1–F3 byte-identical |
| `tools/check_paper_numbers.py docs/paper/draft.md` | exit 0, `checked=743 matched=743 unmatched=0` |
| `tools/report_oops.py`, `report_holoassist.py` ×2 (README arguments) | byte-identical |
| `tools/report_wave_d.py --date 2026-10-22` | **2,695 bytes vs 3,620 committed: the designer's note is deleted.** A17 fixes this |

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| 1 | **A18**: fast tests stop writing the real memguard log | Every later item checks that nothing under `DATA_ROOT` changed. Today that bar fails after any G2 run, so nothing after it could be validated |
| 2 | **A17**: report generators keep designer notes; ordinals | E2b's README must prove reproduction by regenerating every report **in place**. Today that deletes the Wave D note |
| 3 | **E1b**: paragraph-scoped checker; facts, tables and figures fixed | E2b tags every paragraph and pastes the new T3, T4 and B1–B7. **After E1b, the committed draft fails the checker until E2b.** That is expected; record it in E1b's commit body |
| 4 | **E2b**: the draft corrected | It needs E1b's checker, tables and figure, and A17's safe reproduction |

After E2b, the queue is empty (§9).

---

## 3. The items

### Item A18: the fast tests stop writing the real memory-guard log

**What this means for the user:** every test run writes 8 fake events into the log that diagnoses out-of-memory crashes on this shared Mac. Each item's "nothing on the SSD changed" check also fails for a reason that has nothing to do with the item.

**The gap:**
- `src/shared/memguard.py:125` `log_event(…, data_root=None)` appends to `Path(data_root or DATA_ROOT) / "runs" / "memguard.log"`. `DATA_ROOT` is the module global imported at line 17 from `config` (`/Volumes/Extreme SSD/social_robotics`).
- **`tests/test_judge.py` drives the judges' guard path with no `data_root`.** Each G2 run therefore appends 8 lines to the real log: 5 `step=sr_judge_load action=admit`, 2 `action=unload` and 1 `step=sr_judge_large_load action=stop`.
  - Measured November 5: pids 93638, 94533 and 95258, one per G2 run.
  - With `DATA_ROOT` pointed at a scratch directory, the same 8 lines land there instead.

**Implementation:**
1. In `tests/conftest.py`, add an autouse fixture:
   ```python
   @pytest.fixture(autouse=True)
   def _memguard_log_in_tmp(tmp_path_factory, monkeypatch):
       import shared.memguard as memguard
       monkeypatch.setattr(memguard, "DATA_ROOT", tmp_path_factory.mktemp("memguard_data_root"))
   ```
   It applies to slow tests too. Tests that pass `data_root=` explicitly are unaffected.
2. In `tests/test_memguard.py`, add the fast test `test_log_event_goes_to_tmp_under_pytest`:
   ```python
   import config
   import shared.memguard as memguard
   assert Path(memguard.DATA_ROOT) != Path(config.DATA_ROOT)   # first, so the red run writes nothing
   memguard.log_event("sr_test_probe", "admit", 0, 1.0, 1)
   log = Path(memguard.DATA_ROOT) / "runs" / "memguard.log"
   assert "step=sr_test_probe action=admit" in log.read_text(encoding="utf-8")
   ```

**Validation:**
- **(falsifying)** Run the new test against today's `conftest.py`: it fails on the first assertion, before writing anything.
- Touch a marker, then run `scripts/battery.sh --slow` bare, then G2 twice more. Results: G2 **133 passed**, 3 runs out of 3; G4 **11 passed**; G1 and G3 exit 0.
- `find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f` prints nothing.
- Commit `test(a18): keep the fast tests' memory-guard events out of the real log`.

**Blast radius:** `tests/conftest.py`, `tests/test_memguard.py`, and the `docs/ongoing_general_errors.md` §3 line.

---

### Item A17: report generators keep the designer's notes; correct ordinals

**What this means for the user:**
- **Today:** following the paper's own reproduction steps silently deletes the note that warns readers not to over-read Wave D's result.
- **After A17:** regenerating every report leaves the repository unchanged.

**The gap:**
- **`tools/report_wave_d.py:192` writes `"\n".join(lines) + "\n"` over the whole file.**
  - The committed `docs/evals/2026-10-22_wave_d_robot_check.md` ends with a `---` line, a blank line and a `## Designer note (October 22, 2026; …)` section: lines 47–55, added by the designer in `6a0ba13`.
  - Regenerating with `--date 2026-10-22`, the command in `docs/paper/README.md` step 4, deletes those lines (measured: 2,695 bytes vs 3,620).
- **`tools/report_reflex.py:999–1014` keeps its note with its own inline code** (from B4b). The rule should be shared.
- **`tools/report_reflex.py:620` prints `({pct0}th percentile among the 50)`,** which gives "92th" in the report's §3.3.

**Measured layout (both committed reports):**
- the last generated line;
- one blank line;
- `---`;
- one blank line;
- `## Designer note …`, through the end of the file.

**Implementation:**
1. **In `src/harness/reporting.py`, add:**
   ```python
   def designer_note_suffix(existing_text: str) -> str:
   ```
   - Find the first line that starts with `## Designer note`. If there is none, return `""`.
   - Otherwise, walk upward from that line over blank lines. If the next line up is exactly `---`, the suffix starts at that `---` line; otherwise it starts at the heading line.
   - Return the text from the start of that line to the end, unchanged.
   ```python
   def with_designer_note(generated: str, existing_text: str) -> str:
   ```
   - Let `s = designer_note_suffix(existing_text)`.
   - Return `generated` unchanged if `s == ""`.
   - Otherwise return `generated.rstrip("\n") + "\n\n" + s`.
   ```python
   def ordinal(n: int) -> str:
   ```
   - Return `f"{n}th"` if `n % 100` is 11, 12 or 13.
   - Otherwise use `st`, `nd` or `rd` for a last digit of 1, 2 or 3, and `th` for any other.
2. **`tools/report_wave_d.py`:** before writing, read the target if it exists, and write `with_designer_note(generated, existing)`.
3. **`tools/report_reflex.py`:**
   - replace lines 999–1014 with the same two helpers;
   - replace `{pct0}th` with `{ordinal(pct0)}`.
4. **Tests (fast):**
   - **(falsifying)** `tests/test_report_wave_d.py::test_wave_d_report_keeps_designer_note`:
     - build the synthetic rows the file already uses;
     - write a target file `"old\n\n---\n\n## Designer note (test)\nkeep me\n"`;
     - regenerate into it;
     - assert the file ends with `"\n\n---\n\n## Designer note (test)\nkeep me\n"` and contains the generated report's first heading.
     - *Today this fails: the note is overwritten.*
   - `tests/test_report_reflex.py::test_designer_note_suffix_cases`, three cases:
     - no heading → `""`;
     - `"x\n\n---\n\n## Designer note a\nb\n"` → `"---\n\n## Designer note a\nb\n"`;
     - `"x\n## Designer note a\n"` → `"## Designer note a\n"`.
   - `tests/test_report_reflex.py::test_ordinal`: 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 22nd, 92nd, 100th, 101st, 111th, 112th.
   - The existing `test_designer_note_preserved` must still pass unchanged.

**Validation:**
- **Regenerate all five dated reports in place,** with the exact five commands in `docs/paper/README.md` step 4.
  - `git diff --stat docs/evals/` then lists **exactly one file**, `docs/evals/2026-11-04_reflex_h1.md`, with **one line changed**: `92th` → `92nd`.
  - Every other report and every TSV is byte-identical, including the Wave D note.
- The `DATA_ROOT` marker check prints nothing.
- `scripts/battery.sh` bare: G2 **136 passed**, 3 runs out of 3.
- Commit `fix(a17): keep designer notes when regenerating reports; correct ordinals`. The regenerated reflex report goes in the same commit.

**Blast radius:**
- `src/harness/reporting.py`, `tools/report_wave_d.py`, `tools/report_reflex.py`;
- `tests/test_report_wave_d.py`, `tests/test_report_reflex.py`;
- `docs/evals/2026-11-04_reflex_h1.md` (one line);
- the `docs/ongoing_general_errors.md` §3 line.

---

### Item E1b: a checker that ties numbers to their facts; facts, tables and figures corrected

**What this means for the user:**
- **Today the checker cannot tell a right number from a coincidental one.** It accepts about a third of invented three-decimal AUROCs.
- **It never looked inside code spans or figure captions.** That is where the draft's invented windows sat.
- **After E1b,** each paragraph declares where its numbers come from, and the checker holds it to that.

**The gap:**
1. **`tools/check_paper_numbers.py:30` `_collect_numbers` pools every number in `facts.json` (474 distinct),** and `_token_matches_facts` (line 45) accepts a token that matches any of them.
   - Measured: 211 of the 601 three-decimal values 0.400–1.000 match some fact, and 71 of the 101 two-decimal values do.
   - The draft's "92%" (a percentile not in the facts) matched only `recorded/wave_d/pool_mistakes` 92.
2. **Line 19 `INLINE_CODE_RE` deletes every inline code span** (the invented `[max(a_start, a_end - 2.0), a_end]` was never checked), and **line 21 `IMAGE_LINE_RE` deletes figure captions.**
3. **Line 17 `TABLE_BLOCK_RE` accepts only `T\d+`.**
4. **`tools/build_paper_facts.py:35–115` `RECORDED_FACTS` has single quotes that miss stored numbers.**
   - `wave_d/head_norms` stores `move_react_shuffled` 0.015, `move_truth` 0.33 and `truth_in_sample_auroc` 0.997.
   - `wave_d/head_zero_shot` stores `cosine` 0.0.
   - None of these is in its quote. They are right, per `04_wave_d_robot_check.md` §8, but unprotected.
5. **T3 (lines 371–389) pastes the imported verdict strings whole.** That includes the path `/Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large`, twice per `holoassist` row, and the internal ids in "PROCEED to DW3 and DW6".
6. **T4 (line 402) types `| 140 |`** for the null row's `n_test`.
7. **F1 (lines 466–509) is wrong in four ways:**
   - it plots a sixth, non-ablation row (`reflex: [length+face] - [length]`), added to reach a "five intervals" count that the old guide wrongly expected;
   - it types `n=1,072`, `n=1,420`, `n=345` and `n=173`;
   - it types the annotation "audio check < +3.5 dB";
   - its x tick labels overlap (`−0.100−0.075−0.050−0.025`).
8. **F2 (lines 511–555) and F3 (lines 557–584) type their `n` values** into labels and the title.
9. **The draft's Appendix B is hand-copied into fenced code**, which the checker skips. All 139 rows are correct today, but nothing keeps them so.

**Implementation:**
1. **Fact paths (`tools/check_paper_numbers.py`):** add `flatten_facts(obj) -> Dict[str, float]`.
   - It maps every numeric leaf (an `int` or `float`, not a `bool`) to its path.
   - A path is the keys from the top (`computed` or `recorded`) joined with `/`; list items get their index.
   - Examples: `computed/scorecard/oops/judge/value`, `computed/posthoc/reflex.window_length/ci_low`, `computed/datasets/oops/test_items`, `computed/holoassist/verb_drop/mistakes`, `computed/constants/pass_bar_ablation_delta`, `recorded/wave_d/head_norms/value/truth`.
   - The flat duplicate keys already in `computed` (for example `"scorecard/oops/judge/value"`) produce the same paths as the nested ones.
2. **Blocks and tags.** Process the draft in this order:
   - **(a) Tables first.** For every `<!-- table:Xn -->` … `<!-- /table:Xn -->` with `X` in `{T, B}`, assert byte equality with `docs/paper/tables/Xn.md`, then blank the block (keep its line count).
   - **(b) Blank fenced code blocks.**
   - **(c) Split the rest into blocks:** maximal runs of non-blank lines.
   - **(d) Read each block's tags:** every `<!-- facts: p1; p2; … -->` in it. The prefixes are split on `;` and stripped.
   - **(e) Remove all HTML comments.** Then, in this order:
     - delete link targets `(...)` after `]` (this also removes image paths), arXiv ids, ISO dates and written dates (unchanged rules);
     - delete every match of these exemption patterns:
       - `\[CITATION NEEDED:[^\]]*\]`;
       - `\b(?:Issues?|Decisions?|[Ee]ntr(?:y|ies))\s+\d+(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*[–-]\s*)\d+)*`;
       - `\b(?:Figure|Table)\s+\d+`;
       - `\bdoi:\S+`;
       - `ERR@HRI\s+\d+(?:\.\d+)?`;
       - `CC\s+BY\s+\d+(?:\.\d+)?`;
       - `` `?\blabel\s*=\s*[01]\b`? ``;
     - for inline code spans: a span with no whitespace is deleted (an identifier: `qwen2.5vl:7b`, `siglip-b16-224-u9`, `N=100`, `reflex:D2-1:10:pre`); a span with whitespace loses only its backticks, so its content is checked.
   - **(f) Extract tokens** with the existing `NUM_TOKEN_RE`; the 1990–2030 year exemption is unchanged.
   - **(g) Allowed numbers.**
     - Each block that has at least one token must have at least one tag.
     - Every prefix must have **at least three segments** and must match at least one path, either exactly or as `prefix + "/"`.
     - The block's allowed numbers are the values under its prefixes, plus every value under `computed/constants/`.
     - A token matches by the existing `_token_matches_facts` rule, against the allowed set only.
   - **(h) Output.**
     - `checked=<n> matched=<n> unmatched=<n> blocks=<tagged>/<blocks with numbers>`;
     - then one line per error: `line <n>: no facts tag`, `line <n>: prefix too short: <p>`, `line <n>: prefix matches nothing: <p>`, `line <n>: unmatched token '<t>' in: <original line>`, `table <Xn>: …`.
     - Exit 1 on any error.
3. **Constants** (`build_paper_facts.py`, `constants` dict at line 623): add exactly these, each with a source comment. **Add no others;** a number that needs another constant is a STOP (Issue 13).

   | Key | Value | Source |
   |---|---|---|
   | `probe_C` | 1.0 | `03_eval_harness.md` line 134 |
   | `probe_max_iter` | 2000 | same |
   | `ablation_ci_low_bar` | 0.0 | `00_thesis.md` H1 pass ("the paired CI excluding 0") |
   | `permutation_seed_range` | [0, 49] | `03` §11.9; the B4b null |
   | `holoassist_reaction_extension_s` | 5.0 | `02_data_sources.md`, HoloAssist item definition (`[start, end + 5.0]`) |
   | `oops_pre_window_s` | [-4.0, -1.0] | `src/sources/oops.py:170–178` |
   | `oops_post_window_s` | [0.0, 3.0] | `src/sources/oops.py:194–202` |
   | `audible_threshold_db` | 3.5 | `02`, `holoassist_audible` |
   | `judge_scale_max` | 100 | `03` §8 ("an integer from 0 to 100") |
   | `frontier_min_coverage` | 0.9 | `00_thesis.md` entry 16 |
   | `reflex_face_dims` | 153 | `05_wave_b_reflex.md` §4 |
   | `reflex_speech_dims` | 96 | `05` §4 |
   | `face_summary_columns` | 50 | `05` §4.1 (the `_mean` dims) |
   | `hume_prosody_columns` | 48 | `05` §4.2 |
   | `siglip_dims` | 768 | `03` §9 |
4. **Recorded facts:**
   - Each entry becomes `{"value", "source", "quotes": [ … ]}`.
   - **`validate_recorded_facts`** asserts that each quote occurs verbatim in the source file, **and** that every numeric leaf of `value` matches, by `_token_matches_facts(t, {leaf})`, at least one token that `NUM_TOKEN_RE` extracts from the entry's quotes. Otherwise it raises `ValueError` naming the key and the number.
   - The quotes, copied from the files as they are today:
     - **`wave_d/head_norms`**, three quotes, all from `docs/04_wave_d_robot_check.md` §8:
       - ``The norms are `react` 0.99, `shuffled` 1.03 and `truth` 26.9.``
       - ``The `truth` head, fit on 104 mistakes, is overfit: in-sample AUROC 0.997, norm 26.9.``
       - ``At N = 100, the `react` and `shuffled` priors moved test predictions by 0.015 on average; `truth` moved them by 0.33.``
     - **`wave_d/head_zero_shot`**, one quote, the existing sentence plus the sentence that follows it in the file:
       - ``**Each human head on its own, scored on the robot test set:** `react` 0.508 [0.475, 0.540], `shuffled` 0.521 [0.487, 0.554], `truth` 0.491 [0.457, 0.522]. All are at chance, and every head's direction is orthogonal to the no-prior robot detector's (cosine ≈ 0.00).``
     - **The other four entries:** their current single quote becomes a one-item list.
5. **T3:** columns `| Dataset | Check | Measured | Verdict |`. Build each row from the imported line exactly as follows:
   - Skip the same lines as today: those starting with `**Context`, `**Caveat` or `**judge-frontier`.
   - `check` is the bold label, as today.
   - `rest` is the text after `:**`.
   - Split `rest` at its **last** ` → `; with no ` → `, `measured = rest` and `verdict = ""`.
   - If `measured` contains ` not run`, cut it at its first `:`. This gives "react-nonverbal not run" and "ablation Δ not run".
   - Cut `verdict` at its first `:`. This gives "not tested" and "not testable".
   - **The kill-rule row:**
     - Check = `Wave B kill rule (pre-registered)`;
     - Measured = the text inside the **last** pair of parentheses of the message;
     - Verdict = the status (`PROCEED`).
   - Put the cell-cutting in a pure function, `t3_cells(line: str) -> Tuple[str, str, str]`.
   - The full strings stay in `facts.json` unchanged.
6. **T4:** the null row's `n_test` = `scorecard["reflex_resolve"]["react-nonverbal:shuffled"]["n_items"]`.
7. **F1:**
   - **Rows,** top to bottom: `oops`, `holoassist`, `holoassist_audible`, `reflex`, `reflex_resolve`. **No other row.**
   - **Intervals** (marker plus CI) for the four rows whose `fusion_minus_action_fusion` value is not null.
   - **Labels:** `f"{ds} (n={row['n_items']:,})"`. The `holoassist` label is `holoassist`, and its row carries the text `no Δ: reaction features not extracted (audio check failed)` at x = 0.002.
   - **Axis:** `set_xlim(-0.06, 0.10)` and `set_xticks([-0.05, 0.0, 0.03, 0.05, 0.10])`, with `FormatStrFormatter("%.2f")`.
   - **Reference lines:** a solid line at 0 (legend "Δ = 0") and a dashed line at 0.03 (legend "pass bar +0.03"); legend at `loc="lower right"`.
   - **Size and title:** `figsize=(7.5, 3.6)`; title `H1 ablation gain (fusion − action-fusion), 95% grouped-bootstrap CI`.
8. **F2 and F3:** sample sizes come from the data.
   - F2's labels use each posthoc key's `n_test`.
   - F3's title uses `n_items` and `n_groups` of the `roboreward` row `detector:none:N=50`, formatted `{:,}`.
9. **Appendix B tables:** write `docs/paper/tables/B1.md` … `B7.md`, one each for `oops`, `holoassist`, `holoassist_audible`, `reflex`, `reflex_resolve`, `holoassist_pool` and `roboreward`.
   - Header: `| Condition | Value [95% CI] | n_items | n_groups | Row ts |`.
   - One row per condition, in the order `load_latest_scorecard_rows` returns them (first appearance in the scorecard).
   - Condition in backticks; value via `_fmt_cell`, so a null value is `n/a`; `n_items`, `n_groups` and `ts` as stored.
10. **Tests** (`tests/test_paper_tools.py`; fast, synthetic):
    - **(falsifying)** `test_checker_scopes_numbers_to_paragraph_tags`:
      - facts `{"computed": {"scorecard": {"oops": {"fusion_minus_action_fusion": {"value": 0.0149}}, "holoassist_audible": {"fusion_minus_action_fusion": {"value": 0.0206}}}, "constants": {"chance_auroc": 0.5}}, "recorded": {}}`;
      - draft `"Oops gained 0.015, not 0.021, over chance 0.5.\n<!-- facts: computed/scorecard/oops -->\n"`;
      - assert exit 1 and `unmatched == ["0.021"]`.
      - *Today's checker exits 0, because 0.021 matches the other dataset.*
    - `test_checker_requires_a_tag`: `"The face scored 0.735.\n"`, with a fact 0.7352 → exit 1, and the output contains `no facts tag`.
    - `test_checker_rejects_short_prefix`: tag `computed/scorecard` → exit 1, `prefix too short`.
    - `test_checker_reads_code_spans_and_captions`:
      - a block ``"The window is `[a_end - 0.5, a_end + 2.0]`.\n<!-- facts: computed/scorecard/oops -->"`` → `2.0` unmatched;
      - a block `"![Figure 1: n_test = 999](figures/f.png)\n<!-- facts: computed/scorecard/oops -->"` → `999` unmatched;
      - a block ``"Judged by `qwen2.5vl:7b`."`` with no tag → no error.
    - `test_checker_exempts_reference_numbers`: ``"Decisions 2, 12, and 13; Issue 12 = A; Figure 2; doi:10.5281/zenodo.14590226; ERR@HRI 3.0; CC BY 4.0; `label = 1`; [CITATION NEEDED: Gemma 4 reference]."`` with no tag → exit 0, `checked=0`.
    - `test_recorded_values_must_be_quoted`:
      - source file text `"The norms are 0.99 and 26.9.\n"`;
      - entry value `{"a": 0.99, "b": 0.33}`, with that sentence as its only quote;
      - expect `ValueError` whose message contains `0.33`.
    - `test_t3_cells_drop_paths`, two cases:
      - `"- **Clause 1, \"x\":** react-nonverbal not run: 0 of 3000 items cached in /Volumes/X; rerun: y vs step-prior 0.751 → not tested: not run: 0 of 3000 items cached in /Volumes/X"` → `("Clause 1, \"x\"", "react-nonverbal not run", "not tested")`;
      - `"- **Clause 2, \"Δ ≈ 0\":** ablation Δ 0.015 [-0.001, 0.029] → held"` → `("Clause 2, \"Δ ≈ 0\"", "ablation Δ 0.015 [-0.001, 0.029]", "held")`.
    - **Update** `test_checker_rejects_invented_number`:
      - nest its facts as `{"computed": {"a": {"b": {"x": 0.7352, "n": 1208}}}, "recorded": {"c": {"value": 0.954, "source": "s", "quotes": ["q"]}}}`;
      - end its draft with `\n<!-- facts: computed/a/b; recorded/c/value -->\n`;
      - it must still give exactly one unmatched token, `0.736`.
11. **Rebuild** with `PYTHONPATH=src ./venv/bin/python tools/build_paper_facts.py`.

**Validation:**
- **Red first:** the falsifying test fails against today's checker.
- **The builder exits 0.** `git diff docs/paper/` then shows these changes only:
  - `facts.json`: the recorded `quotes` lists and the 15 new constants. The `computed` scorecard, posthoc, datasets and named values are unchanged;
  - `T3.md` and `T4.md`: the new formats;
  - `B1.md`–`B7.md`: new;
  - `f1_ablation.png`, `f2_window_length.png` and `f3_wave_d.png`: regenerated. **F2 and F3 must look unchanged except possibly their label text.**
  - T1, T2, T5 and T6 are byte-identical.
- **T3:** `grep -c "/Volumes\|DW[0-9]" docs/paper/tables/T3.md` prints `0`. Every Measured and Verdict cell occurs verbatim, with `**` removed, in its dated report.
- **B1–B7:** together they hold all 139 rows, and every value matches the scorecard.
- **Open F1** and confirm five labels, four intervals, no overlapping tick labels, and that only `holoassist_audible`'s interval lies right of 0.
- **Run the checker on the committed draft:** it must now exit 1 with `no facts tag` errors and T3/T4 drift. **That is expected;** put the first line of its output in the commit body. E2b fixes the draft.
- `scripts/battery.sh` bare: G2 **143 passed**, 3 runs out of 3. The `DATA_ROOT` marker check prints nothing.
- Commit `feat(e1b): scope paper numbers to their facts; quote every recorded number; fix T3, T4, F1; generate Appendix B`.

**Blast radius:**
- `tools/check_paper_numbers.py`, `tools/build_paper_facts.py`, `tests/test_paper_tools.py`;
- `docs/paper/facts.json`, `docs/paper/tables/` (T3, T4, B1–B7), `docs/paper/figures/*.png`;
- the `docs/ongoing_general_errors.md` §3 line.

---

### Item E2b: the draft, corrected and tagged

**What this means for the user:** the maintainer is about to put their name on this draft. Today it states windows nobody used, a gain bound that its own table contradicts, a pre-registration claim that the decision log contradicts, and three wrong paper titles. After E2b, every sentence is traceable to a source.

**The gap** (line numbers in today's `docs/paper/draft.md`; each item names its source of truth):

| # | Line | Wrong today | Source of truth |
|---|---|---|---|
| 1 | 72 | HoloAssist windows `[max(a_start, a_end - 2.0), a_end]` and `[a_end - 0.5, a_end + 3.0]` (invented) | `02_data_sources.md` HoloAssist "Item definition": action `[start, end]`; reaction `[start, end + 5.0]`, clipped to the video's duration |
| 2 | 72 | The two HoloAssist caveats are missing | `02`: the audio is pitch-shifted; groups are session-name prefixes (most likely pairs), so a person in more than one pair can appear on both sides of the split |
| 3 | 26 | "reactions add at most +0.021 [0.006, 0.034]" | T2: REFLEX's ablation gains are +0.023 and +0.036; only +0.021 has `ci_low > 0` |
| 4 | 7 | "neither dataset achieves a positive ablation gain" | both REFLEX ablation CIs include 0 (T2) |
| 5 | 43 | "Every hypothesis … was committed … before test evaluation" | `00_thesis.md` decision 20 (with its disclosure); `06_writeup.md` §3, as corrected |
| 6 | 49 | "The PROCEED verdict stands in the scorecard" | verdicts are not scorecard rows; it stands as recorded in the Wave B report |
| 7 | 140 | "reflects failure impact audio as much as human vocal reaction" | `00_thesis.md` entry 17: Oops! `react-nonverbal` is not evidence of reaction signal, because its window contains the failure's own sound |
| 8 | 142 | `react-spoke` "beats" `react-nonverbal` (CIs overlap) | `06` §7 |
| 9 | 144 | "a pure window-length artifact" | `06` §2: "explained by" |
| 10 | 160 | the pre-registered interpretation is missing; the diagnostics "show why"; "cosine similarity 0.0" has no referent | `facts.json` `computed/prediction_checks/roboreward_h3`; `04` §8: "Exploratory diagnostics (measured after the result was seen…)"; "orthogonal to the no-prior robot detector's (cosine ≈ 0.00)"; "truth hurts mostly measures an oversized prior from an overfit head" |
| 11 | 172 | "An initial B4 report note hallucinated…"; "dictated by the physical task verb" | not one of C5's five traps; `06` §7 |
| 12 | 176 | "we would have … exported `reflex_face_probe.npz`" | B4 did export both probes (§5.1); the robot demonstration is paused |
| 13 | 197 | `fail` windows "extend 22 frames past `Failure.End`" | `05` §3: up to 22 frames, at most to `Explanation.End` |
| 14 | 222 | "seed 0 at the 92% mark"; the null listed twice (4 dp and 3 dp) | the percentile is not in `facts.json` |
| 15 | 226 | the fusion trap cites L12 | L12 is about annotation vocabulary; this trap is decision 20 |
| 16 | 238 | "AM-FED+ and taste-liking contacts were unreachable" | `00_thesis.md` entry 27: the AM-FED+ address bounced; the taste-liking author did not reply |
| 17 | 176, 240, 242 | internal ids `DW6`, `DW8` | `06` §7 |
| 18 | 248 | "establish a stricter baseline" | `06` §7 |
| 19 | 33 | TAMER "assist policy learning when combined with explicit rewards"; ERR@HRI "across face, head, and body features" | `01_prior_work.md`'s descriptions only |
| 20 | 13, 33, 74, 510, 527 | "Naoum et al." for the HRI 2025 paper | `01_prior_work.md` "Verified bibliographic records": the paper is Khanna et al.; the dataset is Naoum and Khanna |
| 21 | 517–530 | Three wrong titles (REFLEX, REACT, TOPReward), recalled titles and years, and `[CITATION NEEDED: full reference]` on works that are now verified | the same verified records |
| 22 | 116–136 | No caption after T3 explains its PROCEED row | `06` §2–§3 |
| 23 | 138 | F1's caption describes the removed post-hoc row | E1b's F1 |
| 24 | 284–470 | Appendix B in fenced code (unchecked) | `06` §3, §5: B1–B7 between markers |
| 25 | all | No paragraph names its facts | `06` §4; E1b |
| 26 | 254–275 | Appendix A prints this machine's paths (`/Volumes/Extreme SSD/…`) | The paper is for outside readers. Appendix A gives the commands with `$DATA_ROOT`, `$HF_HOME` and `$TMPDIR`, and points to `docs/paper/README.md`, which keeps this machine's values |
| 27 | 19 | "yields zero gain" for +0.0009 [−0.0006, 0.0024] | `06` §7: say "no measurable gain" |

**Implementation:**
1. **Correct each row of the gap table.** The required wording, where wording matters:
   - **(1)** "The action window is the annotated action's `[start, end]`, and the reaction window is `[start, end + 5.0]` s, clipped to the video's duration."
   - **(2)** Two sentences in §4's HoloAssist paragraph, and one bullet in §8 Limitations:
     - the audio is pitch-shifted by the dataset's creators for anonymization, so prosody is measured on altered voices;
     - groups are session-name prefixes, most likely instructor–performer pairs, so a person who took part in more than one pair can appear on both sides of the split, which the data cannot rule out.
   - **(3)** "The only ablation gain whose CI excludes zero is +0.021 [0.006, 0.034] (`holoassist_audible`); REFLEX's +0.023 [−0.004, 0.045] and +0.036 [−0.002, 0.078] are indistinguishable from zero."
   - **(4)** In the abstract, state both REFLEX ablation gains with their CIs, or say "neither REFLEX ablation gain excludes zero". The abstract stays ≤ 200 words.
   - **(5)** §3's opening states the timeline of `06_writeup.md` §3:
     - the prediction of October 8 preceded all data;
     - the pass bar was proposed then and amended once, on October 20, after the Oops! and HoloAssist results, when its Δ became the ablation gain and a `step-prior` control was added;
     - the designer had measured the two numbers behind the change on the HoloAssist test labels;
     - every later evaluation (`holoassist_audible`, Wave D and Wave B) ran under rules fixed before its data was touched.
     - Use the words "October 20".
   - **(10)** C4 quotes the interpretation verbatim: "Human→robot transfer does not help in this setup even with true human labels; reaction labels cannot be judged here."
     - It introduces the diagnostics as "exploratory diagnostics, measured after the result was seen (`04_wave_d_robot_check.md` §8)".
     - It says each head's direction is orthogonal to the robot detector's (cosine 0.00).
     - It says the true-label prior's harm mostly reflects an oversized prior from an overfit head.
   - **(11)** Delete the "initial B4 report note" sentence. "is instead dictated by" becomes "varies mostly with".
   - **(12)** "The face probes had already been exported for a planned live robot demonstration; that demonstration is paused until a reaction reader passes a window-length control."
   - **(14)** Keep the null as mean 0.510 (SD 0.063, range [0.330, 0.653]) with seed 0 = 0.603 and true 0.660 beating 50 of 50. Drop the percentile and the 4-dp repeats.
   - **(17)** No `DW` id anywhere. Describe the work in words.
   - **(19)** Describe a cited work only with what `01_prior_work.md` says it did.
   - **(20, 21)** In-text citations and the reference list, exactly per `06_writeup.md` §6, from the verified records:
     - REFLEX appears as **two references**, the paper and the dataset; each mention cites the one it means;
     - Oops! gains arXiv:1911.11206;
     - every `[CITATION NEEDED: full reference]` that the records now fill is removed;
     - keep `[CITATION NEEDED: …]` for SigLIP, Qwen2.5-VL, Gemma 4, Gemini, emotion2vec+, OpenFace, Facetorch, Hume prosody, scikit-learn, AM-FED+, CANDOR, Ego4D and Charades-Ego.
   - **(22)** Immediately after the T3 block, add a caption paragraph. It must contain "PROCEED" and "window": the kill rule returned PROCEED through `reflex` `react-face` alone, and verification found that clause explained by window length (§7.2).
   - **(23)** F1's caption describes five rows and four intervals. It gives each `n_test` and says `holoassist` has no Δ.
   - **(24)** Replace each fenced Appendix B block with its `<!-- table:Bn -->` block, pasted verbatim from `docs/paper/tables/Bn.md`.
   - **(25)** Every block with a checked number, captions and appendices included, ends with its `<!-- facts: … -->` line. Use the narrowest prefixes that cover the paragraph: name the specific scorecard datasets, posthoc keys, `computed/datasets/<ds>` and `recorded/<key>`. Never use a whole family when one dataset is meant.
2. **Re-paste** T3 and T4 from E1b's files; T1, T2, T5 and T6 are unchanged. Paste B1–B7.
3. **Change nothing else:** the structure, the claims and the remaining text stay.
4. **`docs/paper/README.md` step 4:**
   - state that the five commands regenerate the reports **in place**;
   - add a final command, `git status --short docs/evals docs/paper`, which must print nothing.

**Validation:**
- **The checker:** `PYTHONPATH=src ./venv/bin/python tools/check_paper_numbers.py docs/paper/draft.md` exits 0 with `unmatched=0`, and `blocks=` shows every numeric block tagged. Put its first line in the commit body.
- **Banned strings.** A scripted check that **none** of these occurs in `draft.md`:
  - `a_end`, `at most +0.021`, `pure window`, `dictated`, `would have attributed`, `92% mark`;
  - `Naoum et al.`, `Studying Human Reactions`, `as Zero-Shot Rewards`, `REFLEX: A Dataset`;
  - `establish`, `unreachable`, `hallucinat`, `DW3`, `DW6`, `DW8`, `/Volumes`, `full reference]`;
  - `before test evaluation`, `stands in the scorecard`, `assist policy learning`, `as much as human vocal`, `face, head, and body features`, `yields zero gain`.
- **Required strings.** The same script checks that **each** of these occurs, scoped where it says so. Today `pitch-shifted`, `October 20` and `exploratory` already occur elsewhere in the draft, so only the scoped check proves the fix:
  - **in the §4 block that starts `**HoloAssist`:** `[start, end + 5.0]`, `pitch-shifted`, `both sides of the split`;
  - **in the §3 block that holds the timeline:** `October 20` and `test labels`;
  - **in the block that starts `**Claim `C4``:** `reaction labels cannot be judged here` and `exploratory`;
  - **anywhere:**
    - `Khanna`, `Hidden Zero-Shot Rewards for Robotics`, `REFLEX Dataset: A Multimodal Dataset of Human Reactions to Robot Failures and Explanations`, `Analyzing Both Human Reactions and Evaluative Feedback`;
    - `arXiv:1911.11206`, `doi:10.5281/zenodo.14590226`;
    - `<!-- table:B1 -->` through `<!-- table:B7 -->`.
- **The Wave B check:** every prose block containing `PROCEED` also contains `window`, the T3 caption included. No sentence calls Wave B "passed" or "supported".
- **The style check:** `grep -niw "novel\|groundbreaking\|remarkabl[ey]"` returns nothing.
- **Citation markers:** list every remaining `[CITATION NEEDED: …]` marker in the commit body.
- **Length:**
  - the abstract is ≤ 200 words;
  - sections 1–9 by `wc -w` are between 5,000 and 7,000, counted as E2 counted them, with tables. Report the prose-only count too.
- **Reproduction:** run README step 4 in place, then `git status --short docs/evals docs/paper` prints nothing.
- **The rendered draft:** open it and confirm that the three figures load and every table, B1–B7 included, renders as a table.
- **Battery:** `scripts/battery.sh` bare; G2 **143 passed**, 3 runs out of 3; the `DATA_ROOT` marker check prints nothing.
- **Commit:** `docs(e2b): correct the draft against its sources; tag every number with its facts`.

**Blast radius:** `docs/paper/draft.md`, `docs/paper/README.md`, and the `docs/ongoing_general_errors.md` §3 line.

---

## 4. Deferred: do NOT start

| Work | Status / trigger | Who writes the spec |
|---|---|---|
| **The maintainer's draft review** (title, author line, venue, release, remaining citations; `06_writeup.md` §8) | After E2b **and** the designer's verification of it | The maintainer decides; the designer turns it into items |
| **A decisive hidden-outcome test with faces** (DW8; for example CANDOR) | **Maintainer opt-in only.** Any request is drafted by the designer and approved by the maintainer before it is sent | The designer |
| **Wave B v2** (REFLEX with the flaws fixed) | **Not selected** (Issue 12 = A) | — |
| **Wave C: H2 transfer** (DW3) | **Not selected** (Issue 12 = A) | — |
| **Robot transfer, revisited** (DW7) | **Not selected** (Issue 12 = A); its trigger, a Wave B reaction gain, was not met | — |
| **Stage A live MicroDuck demo** (DW6) | **Paused:** needs a validated reaction reader, hardware, and the maintainer's opt-in | The designer |
| **Wave D v2** | **Not selected** (Issue 10 = A) | — |
| **Ego4D false-positive set** (DW4) | **Not selected** | — |

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
| B1–B4 | `f4c9939`, `5f0631b`, `0d146dd`, `dcb44be` | REFLEX: the download, items and split; the encoders and conditions; features and judges; 40 scorecard rows, the exported face probes (reproducing 0.6085), the report and the TSVs |
| A16 | `1551391` | `_say_to_file` with a 60 s timeout, 3 attempts, partial-file cleanup and a `RuntimeError`; three fast tests. G4 ran without a hang |
| B4b | `5f2d855` | §3.3 of the Wave B report computed from data; `tools/diag_reflex_posthoc.py` and its JSON (all 12 values exact). Everything reproduces byte for byte |
| E1 | `7154531` | `tools/build_paper_facts.py`, `tools/check_paper_numbers.py`, `facts.json`, T1–T6, F1–F3; byte-reproducible. **E1b corrects the checker's scope, the recorded quotes, T3, T4, F1 and Appendix B** |
| E2 | `e470487` | the first full draft and `docs/paper/README.md`. **E2b corrects it** |

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
- **`meta.transcript` is empty and `react-full` did not run on REFLEX** (`Transcript` vs `Text`). Only a v2 changes it.
- **The `reflex_resolve` `react-nonverbal:shuffled` row stays labeled `FAIL`,** with its permutation null beside it.
- **E1's two extra constants:** `h3_pass_bar_gain` 0.02 (`00_thesis.md` H3) and `wave_d_draws_per_budget` 20 (`04` §4).
- **`facts.json` keeps both nested and flat keys in `computed`.** They flatten to the same paths.
- **E2's length was counted with the pasted tables** (5,938; prose alone 4,624), as the guide literally said. E2b keeps that count.
- **F3 shows the `truth` curve below the other three.** The old guide's "four overlapping curves" was loose wording. The figure is right.
- **The oops `judge-frontier` row has `n_items` 50 and `n_groups` 90.** The harness records the test split's groups. It is shown as stored.

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
- **November 2:** Issue 11, wait for the external ComfyUI job rather than stop it.
- **November 5:** **Issue 12 → A.** Wave B is recorded as negative in substance; the project writes up (`06_writeup.md`). Wave B v2, Wave C (DW3) and DW7 are not selected; DW6 is paused.

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **Grouped splits and CIs. Split files are written once. The scorecard is append-only.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels. The leak check is one-sided.**
- **Existing caches, items, splits and scorecard rows are never modified.** Post-hoc diagnostics never become scorecard rows.
- **No number in a generated table, figure or report is typed in by hand** (L10). **No citation detail is written from memory** (L21).

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
| The thesis, H1–H3, the decision log (entries 1–31) | `00_thesis.md` |
| Prior work, and **the verified bibliographic records the paper cites** | `01_prior_work.md` |
| Datasets, including the HoloAssist item definition and caveats, and REFLEX "As built" | `02_data_sources.md` |
| Scorecard, conditions, judges, guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| Wave D: the contract and the v1 result (§8) | `04_wave_d_robot_check.md` |
| Wave B: the REFLEX contract (§1–§6), plus the v1 result and the designer's notes (§7) | `05_wave_b_reflex.md` |
| **Wave E: the write-up contract** (claims, structure, the pre-registration timeline, fact base and tags, tables, figures, citations, style, the maintainer's decisions) | `06_writeup.md` |
| Issues 1–12 (all resolved); deferred work DW1–DW8; lessons L1–L22 | `ongoing_general_errors.md` |
| Reports | `docs/evals/` |
| The paper, its facts, tables and figures | `docs/paper/` |

---

## 7. Validation standard (for every item)

- **Red first, then green, then falsify.** Each item names its falsifying test; run it against today's code before your change and see it fail.
- **Reproduce the stated reference values exactly.** A mismatch is a STOP, never a silent update.
- **A flaky gate is a failing gate.** Run G2 three times with `TMPDIR` on the SSD.
- **Open every artefact you produce:** the figures, the regenerated reports and the rendered draft.
- **Report every validation bar in the commit body, met or missed** (L14).
- **Never loosen a bar or change a pre-registered rule to pass it.** File it as **Issue 13**.

---

## 8. THE LOOP

```
For each item in §2 (A18 -> A17 -> E1b -> E2b):
  (0) touch a marker file in your scratch directory.
  (1) Read the item in §3, and the sections and files it names.
  (2) Write the falsifying test first and watch it fail (red); implement (green); falsify each gate.
  (3) Run scripts/battery.sh bare (--slow for A18; G2 3 times with TMPDIR="/Volumes/Extreme SSD/tmp").
      Run the item's own checks (the reports for A17; the builder and checker for E1b and E2b).
      find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f  must print nothing.
  (4) Add the item's line to docs/ongoing_general_errors.md §3, stage explicit paths,
      commit on main, and push with /usr/bin/git push origin main.
```

---

## 9. Definition of Done for this state

- [x] A1–A15, D1–D4, B1–B4, A16, B4b, E1 and E2 delivered and verified (§5.1).
- [x] Battery measured bare on November 5 (§1.3), with reproduction measured (§1.4).
- [x] The designer's corrections recorded: `06_writeup.md` §3–§8, `01_prior_work.md` (verified records), REFLEX author order in `02` and `05`, and lessons L20–L22.
- [ ] **A18:** no test writes under `DATA_ROOT`; G2 shows 133 passed 3 times in a row; G4 shows 11 passed.
- [ ] **A17:** all five reports regenerate in place with only the `92th` → `92nd` line changing; G2 shows 136 passed.
- [ ] **E1b:** the paragraph-scoped checker, quoted recorded numbers, the T3, T4, F1, F2 and F3 fixes, and B1–B7; G2 shows 143 passed.
- [ ] **E2b:** the draft passes the new checker and the banned and required string checks, and its reproduction leaves `git status` clean.
- [ ] **Then the queue is empty.** The only legitimate triggers for new work are the designer's verification of E2b, the maintainer's draft-review decisions (`06_writeup.md` §8), or a new issue the maintainer raises. **Do not invent work.**
