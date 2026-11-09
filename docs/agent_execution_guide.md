# Agent Execution Guide: Active Build: B6 → E1e → E2e (Pre-registered confirmatory OpenFace same-phase gaze test on REFLEX; load OpenFace gaze results into the fact base; report confirmatory result in the draft and Wave B contract), November 9, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:**
  - On October 8, 2026 the project was reoriented to "human reactions as a reward signal, supervised on outcomes". The v0 pipeline is archived at git tag `v0-saf-final` ([`LESSONS_v0.md`](LESSONS_v0.md)).
  - **Waves A, D and B are complete.** Wave B's pre-registered verdict, PROCEED, came through one clause that the designer found to be a window-length artifact. The maintainer recorded Wave B as negative in substance (Issue 12 = A), and the project wrote it up (Wave E, [`06_writeup.md`](06_writeup.md)).
  - **A17b, E1c, E2c, B5, E1d, E2d and X1 were delivered** (`f77a0da`, `43be1b9`, `8fa05ef`, `7a13f43`, `f8dcc3b`, `5f74488`) and verified by the designer on November 9, 2026 (§5.1):
    - `tools/diag_reflex_v2.py` records the exploratory November 7–8 REFLEX v2 (`ReflexFaceV2Encoder`, 47 features) and the independent review's falsification checks in `docs/evals/2026-11-08_reflex_v2_posthoc.json`.
    - `docs/paper/draft.md` discloses that 622 of the 1,093 `speech.csv` rows matched by v1's §3.4 rule are `robot/participant` segments containing the robot's voice (`79%` of `reflex_resolve` speech items; `participant`-only `react-nonverbal` 0.501 [0.309, 0.734] on 66 items) and passes the number checker (`checked=849 matched=849 unmatched=0 blocks=45/45`).
  - **Issue 13 = C (decided by the maintainer November 9, 2026):**
    - Keep Issue 12 = A (v2 stays an exploratory diagnostic in `docs/evals/2026-11-08_reflex_v2_posthoc.json` and `05_wave_b_reflex.md` §7), and run **one pre-registered confirmatory test** (`05_wave_b_reflex.md` §8) comparing successful vs. failed object handlings at the **exact same physical phase and equal duration** in REFLEX (`275` pairs = `550` items across all 55 participants: `380` train, `170` test) using automatic camera gaze from `openface.csv` (`ReflexOpenFaceGazeEncoder`, 26-d), evaluated once.
- **What is approved now, three items, in this order:**
  1. **B6:** Implement `tools/eval_reflex_openface_gaze.py` and `tests/test_eval_reflex_openface_gaze.py` per `05_wave_b_reflex.md` §8, run the pre-registered confirmatory OpenFace same-phase gaze test once on the test split, and write `docs/evals/2026-11-09_reflex_openface_gaze.json`.
  2. **E1e:** Load `docs/evals/2026-11-09_reflex_openface_gaze.json` into `docs/paper/facts.json` under `computed/openface_gaze/<key>/<field>`.
  3. **E2e:** Report the confirmatory OpenFace same-phase gaze result in `docs/05_wave_b_reflex.md` §8.4 and `docs/paper/draft.md` (§3, §6 Claim C3, §7.2, and §8), tagged with `computed/openface_gaze/<key>`.
- **What you must not touch:**
  - any scorecard row, cache, split or item file under `DATA_ROOT` or `results/scorecard.jsonl`;
  - `05_wave_b_reflex.md` §1–§7 or Wave B's recorded v1 `PROCEED` verdict;
  - the maintainer's draft-review decisions (`06_writeup.md` §8).

**Every path, number, field name and literal string in this guide is a decision, not a suggestion. Implement as written; do not substitute your own values.**

---

## 0. Standing constraints (they apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare (`--slow` when the item adds or touches a slow test). G2 must pass **3 times in a row** with the `TMPDIR` below.
   - Read every exit code bare. A gate that did not run is reported as *not run*, never as passed.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`. Python 3.9 syntax only (no `match`, no runtime `X | Y`).
   - Before running anything:
     ```bash
     export DATA_ROOT="/Volumes/Extreme SSD/social_robotics"
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies, models and downloads are frozen.** No new pip packages, no new model downloads, no Ollama pulls, no data downloads, and no network calls.
4. **Data lives under `DATA_ROOT`** (`/Volumes/Extreme SSD/social_robotics`; exFAT).
   - **No item writes anything under `DATA_ROOT`.**
   - `touch` a marker file in your scratch directory before you start. When you finish, `find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f` must print nothing.
   - Never delete anything you did not create. Every glob over the SSD uses `shared.fsutil.data_glob` (L13).
5. **Memory (`03_eval_harness.md` §12).** No item loads a model. **`gemma4:26b` is shared with another project; never unload it.**
6. **The scorecard is append-only, and no item appends to it.**
7. **Pre-registered rules and recorded verdicts are never changed.** That covers `05_wave_b_reflex.md` §1–§8 and the v1 `PROCEED` verdict.
8. **A test this guide specifies is part of the spec (L23).**
   - Write it first, with the specified inputs and expected values, and watch it fail.
   - If the code cannot meet it, STOP and file Issue 14. Never edit a specified expectation to fit the code.
   - Name every deviation in the commit body, however small.
9. **Do not tune or iterate on the test split.** `B6` evaluates `05_wave_b_reflex.md` §8 once with the exact 26 OpenFace gaze features, windows, probe (`StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")`), and decision rule specified there. Never try alternative feature subsets, confidence thresholds, windows, or hyperparameters.
10. **Commits:**
    - one item = one Conventional Commit on `main` (`feat(b6): …`, `feat(e1e): …`, `docs(e2e): …`);
    - the body holds the WHY, the red and green runs, and **every** validation bar's result, including any bar you missed (L14);
    - stage explicit paths only;
    - add the item's line to `docs/ongoing_general_errors.md` §3;
    - push with `/usr/bin/git push origin main`;
    - no branches, no PRs, never amend.
11. **When a contract and this guide disagree, or a bar cannot be met, STOP and file a new issue** (the next number is **Issue 14**) in `docs/ongoing_general_errors.md`, with options.
12. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer. No issue is open now.

---

## 1. Verified baseline (November 9, 2026; the designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6).
- **`memguard --status`:** 57.0 GB available, pressure 1, heavy lock free, no Ollama models loaded.
- **`venv`:** Python 3.9.6, torch 2.8.0, transformers 4.46.3, scikit-learn 1.6.1, scipy 1.13.1, numpy 2.0.2, matplotlib 3.9.4, ruff 0.15.11, pytest 8.4.2.

### 1.2 Repository

- `HEAD` is `5f74488` (`docs(e2d)`), plus the designer's verification and pre-registration commit (`docs(r13)`). The working tree is clean.

### 1.3 Gates (run bare November 9, 2026 on `5f74488`, with `TMPDIR` on the SSD; the regression bar)

| # | Gate | Command | Result |
|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · `All checks passed!` |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **149 passed, 12 deselected**, 3 runs out of 3 (72.42 s, 71.85 s, 74.23 s) |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | exit 0 · **12 passed, 149 deselected** (166.35 s) |
| — | `DATA_ROOT` marker | `find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f` after the battery | **prints nothing** (A18 holds) |

### 1.4 Reproduction, measured

| Command | Result |
|---|---|
| `tools/diag_reflex_v2.py --out <scratch>` | byte-identical to `docs/evals/2026-11-08_reflex_v2_posthoc.json` (75 keys) |
| `tools/build_paper_facts.py --paper-dir <scratch>` | `facts.json`, T1–T6, B1–B8 and F1–F3 byte-identical to the committed files |
| `docs/paper/README.md` steps 1–4, run in place | every command exits 0; `git status --short` prints nothing; the Wave D note survives |
| `tools/check_paper_numbers.py docs/paper/draft.md` | exit 0 · `checked=849 matched=849 unmatched=0 blocks=45/45` |

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| 1 | **B6**: Evaluate the pre-registered confirmatory OpenFace same-phase gaze test on REFLEX (`05_wave_b_reflex.md` §8) | Produces `docs/evals/2026-11-09_reflex_openface_gaze.json` under the locked pre-registered rules |
| 2 | **E1e**: Load `2026-11-09_reflex_openface_gaze.json` into `docs/paper/facts.json` under `computed/openface_gaze/` | E2e cites the confirmatory numbers from `facts.json` so no number is typed without verification (L10) |
| 3 | **E2e**: Report the confirmatory OpenFace same-phase gaze result in `docs/05_wave_b_reflex.md` §8.4 and `docs/paper/draft.md` | Completes Option C of Issue 13 and prepares the draft for maintainer review |

After E2e, the queue is empty (§9).

---

## 3. The items

### Item B6: evaluate the pre-registered OpenFace same-phase gaze test on REFLEX

**What this means for the user:** answers Option C of Issue 13 with a single pre-registered evaluation (`05_wave_b_reflex.md` §8): whether automatic camera gaze from `openface.csv` distinguishes successful from failed robot object handlings at the same physical phase and equal duration, and how automatic `openface.csv` gaze compares against the dataset authors' semi-manual `gaze.csv` labels on both same-phase and within-episode contrasts.

**Implementation:**
1. **Write `tools/eval_reflex_openface_gaze.py`**, modeled on `tools/diag_reflex_v2.py`:
   - Read-only over `DATA_ROOT` (`/Volumes/Extreme SSD/social_robotics`) and `results/scorecard.jsonl`.
   - Top-level constants:
     ```python
     STATUS_TEXT = (
         "pre-registered confirmatory test (same_phase.openface_gaze and "
         "same_phase.openface_fusion_minus_action_fusion, specified November 9, 2026 in "
         "docs/05_wave_b_reflex.md §8 before any OpenFace feature was extracted, evaluated once) "
         "plus secondary diagnostic comparisons; never scorecard rows"
     )
     OPENFACE_GAZE_COLS: Tuple[str, ...] = (
         "gaze_0_x",
         "gaze_0_y",
         "gaze_0_z",
         "gaze_1_x",
         "gaze_1_y",
         "gaze_1_z",
         "gaze_angle_x",
         "gaze_angle_y",
     )
     SAME_PHASE_PAIRS: Tuple[Tuple[str, str, Tuple[str, str, str, str], Tuple[str, str, str, str]], ...] = (
         ("pick_1", "pick", ("2", "A", "Carry", "Pre"), ("1", "B", "Pick", "Failure")),
         ("pick_2", "pick", ("3", "B", "Carry", "Pre"), ("4", "A", "Pick", "Failure")),
         ("carry_1", "carry", ("1", "D", "Place", "Pre"), ("1", "B", "Carry", "Failure")),
         ("carry_2", "carry", ("3", "A", "Place", "Pre"), ("2", "A", "Carry", "Failure")),
         ("carry_3", "carry", ("4", "C", "Place", "Pre"), ("3", "B", "Carry", "Failure")),
     )
     ```
   - `get_openface_gaze_feature_names() -> List[str]`: returns the 26 feature names in exact order:
     - `[f"of_{c}_mean" for c in OPENFACE_GAZE_COLS]` (8),
     - `[f"of_{c}_std" for c in OPENFACE_GAZE_COLS]` (8),
     - `[f"of_{c}_delta" for c in OPENFACE_GAZE_COLS]` (8),
     - `["of_gaze_angle_speed_mean", "of_gaze_angle_speed_delta"]` (2).
   - `ReflexOpenFaceGazeEncoder` (`encoder_id = "reflex-openface-gaze-v1"`):
     - `_load_participant(participant_dir: Path)`: reads `participant_dir / "openface.csv"`, strips whitespace from column names, filters rows where `int(float(success)) == 1` and `float(confidence) >= 0.7`, and caches `(frames_int_np, vals_float32_np_shape_N_8)` sorted by frame.
     - `encode(participant_dir: Path, rsf: int, ref: int) -> np.ndarray`: selects valid rows in `[rsf, ref]`. If `n_valid == 0`, raises `RuntimeError(f"no valid OpenFace frames in [{rsf}, {ref}]")`. Computes the 26 features per `05_wave_b_reflex.md` §8.2 (`mean`, `std` if `n_valid >= 2` else `0`, `delta = mean(sub[half:]) - mean(sub[:half])` with `half = n_valid // 2` if `n_valid >= 2` else `0`, and `step_speed = np.sqrt(np.diff(gx)**2 + np.diff(gy)**2)` mean and delta) and returns a 26-d `np.float32` vector with `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`.
   - `build_same_phase_items_and_windows(reflex_items: Sequence[Item], split_data: Dict[str, Any], root: Path)`:
     - Maps each of the 5 `SAME_PHASE_PAIRS` per participant via `analysis.csv` `(Round No., Object, Action, State)` `(Start Frame, End Frame)` to the corresponding `Item` in `reflex_items`.
     - Clones each matched `Item` with `item_id = f"reflex_same_phase:{pid}:{pair_id}:{'pos' if label == 1 else 'neg'}"`, `meta = dict(orig_item.meta, orig_item_id=orig_item.item_id, verb=phase, noun="object", pair_id=pair_id, pair_key=f"{pid}:{pair_id}")`, and equal-length end-anchored window `(ef - K + 1, ef)` where `K = min(pef - psf + 1, nef - nsf + 1)`.
     - Returns `(all_items, tr_items, te_items, windows_by_id, orig_id_by_new_id)`.
   - `evaluate_confirmatory_verdict(standalone: MetricResult, ablation_delta: MetricResult) -> Dict[str, Any]`:
     - Implements the pre-registered rule in `05_wave_b_reflex.md` §8.3:
       - `standalone_above_chance = bool(standalone.ci_low is not None and float(standalone.ci_low) > 0.5)`
       - `ablation_gain_positive = bool(ablation_delta.ci_low is not None and float(ablation_delta.ci_low) > 0.0)`
       - `confirmatory_supported = bool(standalone_above_chance and ablation_gain_positive)`
       - `h1_pass_bar_met = bool(standalone.point is not None and float(standalone.point) >= 0.65 and ablation_delta.point is not None and float(ablation_delta.point) >= 0.03 and ablation_gain_positive)`
       - `verdict = "SUPPORTED" if confirmatory_supported else "NOT SUPPORTED"`
   - Compute and store in `docs/evals/2026-11-09_reflex_openface_gaze.json`:
     - `"_status": STATUS_TEXT` and `"_verdict": evaluate_confirmatory_verdict(...)`;
     - **Confirmatory & diagnostic `same_phase.*` keys** (`380` train items, `170` test items across `17` test participants):
       - `same_phase.n_train_items` (`{"value": 380}`), `same_phase.n_test_items` (`{"value": 170}`), `same_phase.n_train_valid_openface_items` (`{"value": 376}`), `same_phase.n_test_valid_openface_items` (`{"value": 157}`), `same_phase.n_train_valid_openface_pairs` (`{"value": 186}`), `same_phase.n_test_valid_openface_pairs` (`{"value": 77}`);
       - Action controls on `same_phase` (using each item's `orig_item_id` to look up cached `siglip-b16-224-u9` and judge scores, and `step_prior_scores` on the physical phase `verb` in `{"pick", "carry"}`):
         `same_phase.action_probe`, `same_phase.judge`, `same_phase.judge_large`, `same_phase.step_prior`, `same_phase.action_fusion`, `same_phase.window_length` (confirming `0.500` on equalized windows);
       - Primary confirmatory OpenFace gaze checks on `same_phase`:
         `same_phase.openface_gaze`, `same_phase.openface_fusion`, `same_phase.openface_fusion_minus_action_fusion`, `same_phase.openface_fusion_minus_action_best`;
       - Pair-complete and phase-stratum checks on `same_phase`:
         `same_phase.openface_gaze_pair_complete`, `same_phase.openface_fusion_minus_action_fusion_pair_complete`, `same_phase.openface_gaze_pick`, `same_phase.openface_gaze_carry`;
       - Semi-manual `gaze.csv` (`GAZE_IDX`), automatic `face_only` (`FACE_IDX`), and `v2_all` (`ALL_IDX`) on the exact same `same_phase` items and windows:
         `same_phase.dataset_gaze`, `same_phase.dataset_gaze_fusion`, `same_phase.dataset_gaze_fusion_minus_action_fusion`, `same_phase.dataset_gaze_fusion_minus_action_best`, `same_phase.dataset_gaze_pick`, `same_phase.dataset_gaze_carry`, `same_phase.face_only`, `same_phase.face_only_fusion`, `same_phase.face_only_fusion_minus_action_fusion`, `same_phase.v2_all`, `same_phase.v2_all_fusion`, `same_phase.v2_all_fusion_minus_action_fusion`;
     - **Within-episode `reflex.*` comparison keys** (`1,208` items: `834` train, `374` test):
       - `reflex.openface_gaze_action_equal_end`, `reflex.openface_fusion_action_equal_end`, `reflex.openface_fusion_minus_action_fusion_action_equal_end`, `reflex.openface_fusion_minus_action_best_action_equal_end`;
       - `reflex.openface_gaze_equal_start`, `reflex.openface_fusion_equal_start`, `reflex.openface_fusion_minus_action_fusion_equal_start`, `reflex.openface_fusion_minus_action_best_equal_start`;
       - `reflex.placebo_early_vs_late_pre_openface_gaze` and `reflex.onset_straddle_openface_gaze`;
     - **Uncorrupted full-object handlings (`uncorrupted_gap.*`)** (`Gap 2` Round 1 Object C and `Gap 4` Round 2 Object B vs. matched-duration `Place.Failure` windows, per `05_wave_b_reflex.md` §8.3 item 4):
       - `uncorrupted_gap.openface_gaze` and `uncorrupted_gap.openface_gaze_pair_complete`.
2. **Write `tests/test_eval_reflex_openface_gaze.py` (red first):**
   - **fast (`test_openface_gaze_encoder_synthetic_and_honest_null`):** writes a synthetic `openface.csv` in `tmp_path` with valid (`success=1, confidence=0.95`) and invalid (`success=0` or `confidence=0.4`) frames; asserts `ReflexOpenFaceGazeEncoder().encode(tmp_path, 1, 10)` returns a 26-d finite `np.float32` vector whose mean matches only the valid rows, and raises `RuntimeError` on a window `[20, 30]` containing only invalid or missing frames.
   - **fast (`test_same_phase_windows_equal_length_and_balanced_step_prior`):** builds synthetic `analysis.csv` and `Item` pairs in `tmp_path`; asserts `build_same_phase_items_and_windows` gives both items of every pair the exact same window length `K`, assigns identical physical `meta["verb"]` (`"pick"` or `"carry"`) to positive and negative items of each pair, and yields a constant `0.5` `step_prior_scores` on balanced train pairs (falsify by checking that leaving `orig_item.meta["verb"]` unnormalized leaks the label with AUROC `> 0.9`).
   - **fast (`test_confirmatory_verdict_helper`):** tests `evaluate_confirmatory_verdict` on synthetic `MetricResult` objects, confirming `"SUPPORTED"` iff both `standalone.ci_low > 0.5` and `ablation_delta.ci_low > 0.0`, and `"NOT SUPPORTED"` when either fails.
   - **slow (`test_eval_reflex_openface_gaze_reproduces_json`):** runs `evaluate_reflex_openface_gaze()` and asserts it matches `docs/evals/2026-11-09_reflex_openface_gaze.json` to 6 decimals and verifies the exact item/pair counts (`380`, `170`, `376`, `157`, `186`, `77`) and `same_phase.window_length["value"] == 0.5`.

**Validation:**
- Two runs of `tools/eval_reflex_openface_gaze.py` produce byte-identical JSON.
- `git status --short` lists only `tools/eval_reflex_openface_gaze.py`, `tests/test_eval_reflex_openface_gaze.py`, `docs/evals/2026-11-09_reflex_openface_gaze.json`, and `docs/ongoing_general_errors.md`.
- `scripts/battery.sh --slow` passes bare, with G2 passing 3 runs in a row (`152 passed`) and G4 passing (`13 passed`).
- The `DATA_ROOT` marker check prints nothing.
- Commit: `feat(b6): evaluate pre-registered OpenFace same-phase gaze test on REFLEX`.

---

### Item E1e: load the confirmatory OpenFace gaze results into the paper fact base

**What this means for the user:** makes every number in `docs/evals/2026-11-09_reflex_openface_gaze.json` addressable in `docs/paper/facts.json` under `computed/openface_gaze/<key>/<field>` so `tools/check_paper_numbers.py` verifies every number reported in `docs/paper/draft.md`.

**Implementation:**
1. **`tools/build_paper_facts.py`:**
   - Add `DEFAULT_OPENFACE_GAZE_JSON = REPO_ROOT / "docs" / "evals" / "2026-11-09_reflex_openface_gaze.json"`.
   - Add `load_openface_gaze_into_computed(computed: Dict[str, Any], openface_gaze_json_path: Path) -> None`, storing every non-`_` dictionary key of `2026-11-09_reflex_openface_gaze.json` under `computed["openface_gaze"][k]` and flat `computed[f"openface_gaze/{k}/{sub_k}"]`, exactly as `load_posthoc_v2_into_computed` works today.
   - Wire `openface_gaze_json_path: Path = DEFAULT_OPENFACE_GAZE_JSON` into `build_computed_facts`, `build_all_paper_artifacts`, and `parse_args` (`--openface-gaze`).
   - Change nothing else.
2. **Fast test (`tests/test_paper_tools.py`):**
   - `test_openface_gaze_lands_under_computed_openface_gaze`: writes a two-key JSON in `tmp_path`, runs `load_openface_gaze_into_computed`, and asserts both nested `computed["openface_gaze"]` and flat `computed["openface_gaze/<key>/<field>"]` entries match. Watch it fail before the implementation.
3. **Rebuild:** `PYTHONPATH=src ./venv/bin/python tools/build_paper_facts.py`.

**Validation:**
- `git status --short docs/paper` lists only `facts.json`.
- A script confirms every pre-existing `computed` and `recorded` value in `facts.json` is unchanged.
- `tools/check_paper_numbers.py docs/paper/draft.md` exits 0 with `unmatched=0`.
- `scripts/battery.sh` passes bare (G2 `153 passed` 3/3); the `DATA_ROOT` marker check prints nothing.
- Commit: `feat(e1e): load the confirmatory OpenFace gaze results into the paper fact base`.

---

### Item E2e: report the confirmatory OpenFace same-phase gaze test in the draft and Wave B contract

**What this means for the user:** records the pre-registered confirmatory OpenFace same-phase gaze result (`05_wave_b_reflex.md` §8) in `docs/05_wave_b_reflex.md` and `docs/paper/draft.md` (with every number checked by `tools/check_paper_numbers.py`), and updates the git-ignored explainer PDF (`scratch/Social_Robotics_REFLEX_v2_Explainer_CORRECTED.pdf`) so the maintainer has the complete picture in one place.

**Implementation:**
1. **`docs/05_wave_b_reflex.md` §8:**
   - Add `### 8.4 Result of the pre-registered confirmatory test (November 9, 2026)` recording the measured numbers from `docs/evals/2026-11-09_reflex_openface_gaze.json`:
     - the primary confirmatory result (`same_phase.openface_gaze`, `same_phase.action_fusion`, `same_phase.openface_fusion`, `same_phase.openface_fusion_minus_action_fusion`, `same_phase.openface_fusion_minus_action_best`, and `_verdict`);
     - the pair-complete (`same_phase.openface_gaze_pair_complete`) and phase-stratum (`pick` and `carry`) results;
     - the same-phase comparison with semi-manual `gaze.csv` (`same_phase.dataset_gaze`, `same_phase.dataset_gaze_fusion_minus_action_fusion`) and `face_only` (`same_phase.face_only`);
     - the within-episode `reflex` comparison (`reflex.openface_gaze_action_equal_end`, `reflex.openface_fusion_minus_action_fusion_action_equal_end`, `reflex.openface_gaze_equal_start`, `reflex.placebo_early_vs_late_pre_openface_gaze`, `reflex.onset_straddle_openface_gaze`) and the uncorrupted gap check (`uncorrupted_gap.openface_gaze`).
2. **`docs/paper/draft.md`:**
   - Update these four places, extending each block's `<!-- facts: … -->` tag with the exact `computed/openface_gaze/<key>` and `computed/posthoc_v2/<key>` prefixes used:
     - **§3 (`Pre-registration timeline`):** add one sentence at the end of the timeline paragraph noting that after the November 7–8 exploratory v2 diagnostics, we pre-registered and ran one confirmatory test (`05_wave_b_reflex.md` §8, November 9, 2026) comparing successful versus failed object handlings at the same physical phase and equal duration in REFLEX using automatic OpenFace gaze angles (`openface.csv`, 26 features) after label-blind inspection of episode schedules and valid-frame counts.
     - **§6 (`Claim C3`):** after the exploratory equal-length `react-face` sentence, add the confirmatory same-phase OpenFace gaze result (`same_phase.openface_gaze` and its ablation gain `same_phase.openface_fusion_minus_action_fusion` on the `157` valid test items / `85` test pairs) alongside the exploratory within-episode `gaze.csv` vs. `openface.csv` and same-phase `gaze.csv` diagnostic numbers, stating the pre-registered verdict (`SUPPORTED` or `NOT SUPPORTED`) plainly without overclaiming.
     - **§7.2 (`Window length mimics reactions`):** add one concise bullet on phase-matched vs. within-episode gaze on REFLEX, reporting the exploratory within-episode `gaze.csv` gain (`0.720 [0.689, 0.752]`, gain `+0.053 [0.025, 0.084]`), the early-vs-late `Pre` placebo (`0.679 [0.631, 0.732]`), and what happens under the pre-registered same-phase contrast (`same_phase.openface_gaze` and `same_phase.dataset_gaze`).
     - **§8 (`Discussion and limitations`, REFLEX bullet):** mention the confirmatory same-phase OpenFace gaze test alongside the window-length, silhouette, resolution-window, and speaker-mix findings.
3. **Update the git-ignored explainer (`scratch/reflex_v2_review/explainer/build_explainer.py` → `scratch/Social_Robotics_REFLEX_v2_Explainer_CORRECTED.pdf`):**
   - Add a concise section to `build_explainer.py` summarizing the November 9 confirmatory OpenFace same-phase test (`2026-11-09_reflex_openface_gaze.json`), verify that every decimal in `explainer.md` matches `facts.json` (`0` unmatched), and rebuild `scratch/Social_Robotics_REFLEX_v2_Explainer_CORRECTED.pdf`.

**Validation:**
- `PYTHONPATH=src ./venv/bin/python tools/check_paper_numbers.py docs/paper/draft.md` exits 0 with `unmatched=0`.
- All E2b, E2c, and E2d banned-string and required-string checks pass.
- Abstract is ≤ 200 words; sections 1–9 (without HTML comments, with tables) are between 5,000 and 7,000 words by `wc -w`.
- `docs/paper/README.md` steps 1–4 run in place leave `git status --short docs/evals docs/paper` clean (after staging `draft.md`).
- `scripts/battery.sh` passes bare (G2 `153 passed` 3/3); the `DATA_ROOT` marker check prints nothing.
- Commit: `docs(e2e): report the confirmatory OpenFace same-phase gaze test in the draft and Wave B contract`.

---

## 4. Deferred: do NOT start

| Work | Status / trigger | Who writes the spec |
|---|---|---|
| **The maintainer's draft review** (title, author line, venue, release, remaining citations; `06_writeup.md` §8) | After E2e | The maintainer decides; the designer turns it into items |
| **A decisive hidden-outcome test with faces** (DW8; for example CANDOR) | **Maintainer opt-in only.** The designer drafts any request, and the maintainer approves it before it is sent | The designer |
| **Wave C: H2 transfer** (DW3) | **Not selected** (Issue 12 = A, Issue 13 = C) | — |
| **Robot transfer, revisited** (DW7) | **Not selected** (Issue 12 = A, Issue 13 = C) | — |
| **Stage A live MicroDuck demo** (DW6) | **Paused:** needs a validated reaction reader, hardware, and the maintainer's opt-in | The designer |
| **Wave D v2** | **Not selected** (Issue 10 = A) | — |
| **Ego4D false-positive set** (DW4) | **Not selected** | — |

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
| E1, E2, A18, A17, E1b, E2b | `7154531`, `e470487`, `d6718bc`, `a2dcec4`, `ee06239`, `8b5aef0` | the fact base, tables, figures, number checker, draft, and memguard/report-note fixes |
| A17b, E1c, E2c | `f77a0da`, `43be1b9`, `8fa05ef` | A17's specified assertions and ordinals; tagged constants, `CC BY-NC-SA` exemption, `holoassist/pre_amendment_reference`, `B8.md`, F1 legend at upper left; round-three draft corrections |
| B5, E1d, E2d | `7a13f43`, `f8dcc3b`, `5f74488` | `tools/diag_reflex_v2.py` and `docs/evals/2026-11-08_reflex_v2_posthoc.json` (75 keys); `--posthoc-v2` in `build_paper_facts.py`; speech speaker-mix disclosure in `draft.md`, `05_wave_b_reflex.md` §7, and `02_data_sources.md` |

### 5.2 Accepted equivalents (do not "fix" these back)

- All accepted equivalents from §5.2 of the November 6 guide remain in force.
- `docs/evals/2026-11-08_reflex_v2_posthoc.json` and `docs/evals/2026-11-09_reflex_openface_gaze.json` are standalone JSON files in `docs/evals/` loaded into `facts.json` under `computed/posthoc_v2/` and `computed/openface_gaze/`; neither writes rows to `results/scorecard.jsonl`.

### 5.3 Maintainer decisions

- **October 8–November 5:** Decisions 1–30 (`00_thesis.md`), including Issue 12 → A.
- **November 9:** **Issue 13 → C.** Keep Issue 12 = A (v2 stays an exploratory diagnostic), and run one pre-registered confirmatory test (`05_wave_b_reflex.md` §8) of automatic OpenFace gaze on phase-matched REFLEX object handlings (`B6` → `E1e` → `E2e`).

### 5.4 Invariants

- **Supervise on outcomes; `label = 1` is good. The judge never hears audio.**
- **Grouped splits and CIs. Split files are written once. The scorecard is append-only.**
- **`react-full` is never a headline. `gemma4:26b` is never unloaded by us.**
- **Shuffled controls never see the true labels. The leak check is one-sided.**
- **Existing caches, items, splits and scorecard rows are never modified.**
- **No number in a generated table, figure or report is typed in by hand** (L10). **No citation detail is written from memory** (L21).
- **A specified test is written as specified** (L23).

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, H1–H3, the decision log (entries 1–33) | `00_thesis.md` |
| Prior work, and **the verified bibliographic records the paper cites** | `01_prior_work.md` |
| Datasets: the HoloAssist item definition, schema and caveats; the `holoassist_audible` subset rule; REFLEX "As built"; the licenses | `02_data_sources.md` |
| Splits (§5), conditions and the pre-amendment reference values (§6), judges (§8), encoders (§9), guardrails (§11), memory guard (§12) | `03_eval_harness.md` |
| Wave D: the contract and the v1 result (§8) | `04_wave_d_robot_check.md` |
| Wave B: the REFLEX contract (§1–§6), the v1 result and exploratory v2 notes (§7), and **the pre-registered confirmatory OpenFace same-phase gaze contract (§8)** | `05_wave_b_reflex.md` |
| Wave E, the write-up contract | `06_writeup.md` |
| Issues 1–13 (all resolved); deferred work DW1–DW8; lessons L1–L24 | `ongoing_general_errors.md` |
| Reports and evaluation JSONs | `docs/evals/` |
| The paper, its facts, tables and figures | `docs/paper/` |

---

## 7. Validation standard (for every item)

- **Red first, then green, then falsify.**
- **Reproduce the stated reference values and counts exactly.** A mismatch is a STOP, never a silent update.
- **A flaky gate is a failing gate.** Run G2 three times with `TMPDIR` on the SSD.
- **Open every artefact you produce.**
- **Report every validation bar in the commit body, met or missed** (L14), and every deviation from a specified test (L23).
- **Never loosen a bar or change a pre-registered rule to pass it.** File it as **Issue 14**.

---

## 8. THE LOOP

```
For each item in §2 (B6 -> E1e -> E2e):
  (0) touch a marker file in your scratch directory.
  (1) Read the item in §3, and the sections and files it names.
  (2) Write the specified (falsifying) test first, exactly as specified, and watch it fail (red);
      implement (green); run every specified check and watch it pass.
  (3) Run scripts/battery.sh bare (--slow for B6; G2 3 times with TMPDIR="/Volumes/Extreme SSD/tmp").
      Run the item's own checks.
      find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f  must print nothing.
  (4) Add the item's line to docs/ongoing_general_errors.md §3, stage explicit paths,
      commit on main, and push with /usr/bin/git push origin main.
```

---

## 9. Definition of Done for this state

- [x] A1–A18, D1–D4, B1–B5, E1–E1d, and E2–E2d delivered and verified (§5.1).
- [x] Battery measured bare on November 9 (§1.3), with reproduction measured (§1.4).
- [x] Issue 13 = C recorded (`00_thesis.md` entry 33, `ongoing_general_errors.md` Resolved entry 11) and `05_wave_b_reflex.md` §8 pre-registered before any OpenFace gaze feature extraction.
- [ ] **B6:** `tools/eval_reflex_openface_gaze.py` and `tests/test_eval_reflex_openface_gaze.py` written; `docs/evals/2026-11-09_reflex_openface_gaze.json` generated deterministically; G1–G4 exit 0 (G2 152 passed 3/3, G4 13 passed).
- [ ] **E1e:** `2026-11-09_reflex_openface_gaze.json` loaded into `docs/paper/facts.json` under `computed/openface_gaze/`; `test_openface_gaze_lands_under_computed_openface_gaze` passes; G1–G3 exit 0 (G2 153 passed 3/3).
- [ ] **E2e:** confirmatory OpenFace same-phase gaze result reported in `docs/05_wave_b_reflex.md` §8.4 and `docs/paper/draft.md`; `check_paper_numbers.py` exits 0 (`unmatched=0`); reproduction leaves `git status` clean; G1–G3 exit 0 (G2 153 passed 3/3).
- [ ] **Then the queue is empty.** Do not invent work.
