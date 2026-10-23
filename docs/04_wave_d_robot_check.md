# 04: Wave D, the offline robot check (H3): the contract

**What this answers:** does pre-training a success detector on **reaction-labeled human task video** make a **robot** success detector more accurate when only a few robot labels exist? This is the "robots can learn from vlogs" promise of `00_thesis.md` H3, tested offline. No robot is involved.

**Status:** pre-registered October 21, 2026 (the maintainer chose Issue 8 option A). **Every rule below, including the pass and null rules and their interpretations, is fixed before any robot data is downloaded.** It changes only through a dated decision-log entry in `00_thesis.md`. The build spec is [`agent_execution_guide.md`](agent_execution_guide.md) (items D1–D4). The download is **Issue 9: option B** (the full dataset).

**Honest expectation, written before the run:**
- Wave A's reaction signal is modest (`holoassist_audible` ablation gain +0.021 [0.006, 0.034]) and is mostly *whether the instructor spoke* (`00_thesis.md` entry 21).
- The human data (egocentric assembly steps) and the robot data (third-person and wrist-camera manipulation episodes) are far apart.
- **A null result is the most likely outcome,** and it will be reported plainly.

---

## 1. The design in one paragraph

1. A **reaction labeler**, the `react-nonverbal` probe fit on `holoassist_audible`'s train split, gives 20,000 *other* HoloAssist actions a soft label **from the instructor's voice alone**. That is the stage-B "free labels" story.
2. A linear **human success head** is fit on those actions' SigLIP features with those soft labels.
3. On RoboReward, a linear **robot success detector** is fit on N robot labels with the human head as its **prior** (L2 regularization toward the human weights, instead of toward 0).
4. It is compared with the identical detector without a prior, at N = 50 … 1,000.
5. **Two controls separate the effects:**
   - a head fit on **shuffled** soft labels: is it "any human-video prior" rather than reactions?
   - a head fit on HoloAssist's **true** labels: can anything transfer from this human data at all?

## 2. Data

### 2.1 Robot: RoboReward (`dataset = roboreward`)

- **Source:** `teetone/RoboReward` (Hugging Face; CC-BY-4.0), **revision `469b9af76e8539e8d2ac553b081307738fd92ca5`**, stored under `DATA_ROOT/raw/roboreward/`. **The whole dataset is downloaded** (Issue 9 option B, October 21): 54,140 files and 43,204,474,744 bytes, through one `huggingface_hub.snapshot_download(repo_id="teetone/RoboReward", repo_type="dataset", revision=<above>, local_dir=DATA_ROOT/raw/roboreward)`. Videos this spec does not select stay on disk, unused.
- **Splits used:**
  - RoboReward `train` → the **robot train pool**;
  - RoboReward `test` → **evaluation**;
  - `val` is not used.
- **The label is vision-only: the detector never reads the instruction.**
  - **Success (`label = 1`):**
    - OXE sources: an *original* episode file (no `_attempt_` in its name) with `reward = 5`;
    - `robo_arena`: `reward = 5`.
  - **Failure (`label = 0`):**
    - OXE sources: an `_attempt_` (clipped) file with `reward ≤ 2`;
    - `robo_arena`: `reward ≤ 2`.
  - **Excluded, counted per reason in `stats.json`:**
    - OXE *original* files with `reward < 5`. These are counterfactual-instruction relabels: the video shows another task completed, so a detector that reads no instruction cannot judge them. This exclusion is the thesis's.
    - Any `reward` of 3 or 4 (partial progress; the middle is excluded, as with verdicts).
- **`group_id`:**
  - OXE: `<source>:<original episode stem>`, i.e. the file stem with any `_attempt_<k>_score_<s>` suffix removed. An original and its clipped attempts share a group.
  - `robo_arena`: `robo_arena:<rollout uuid>`, the part of the stem before the first `_`. The camera views of one rollout share a group.
- **`item_id`** = `roboreward:<split>:<file stem>`.
- **Windows:** `action_window_sec` = `reaction_window_sec` = `[0, D]`, where `D` is the video duration from `ffprobe`.
- **`context_text`** = `"Task: <task>."`. It is used **only** by the judge reference (§5).
- **The split file `splits/roboreward.json`** comes from `make_group_split(official=…)`, mapping RoboReward train → `train` and test → `test`.
  - **If any group straddles the two splits,** remove that group's train-side items from the robot train pool, before any sampling, and record the count. This rule is label-free and decided now.
- **The robot train pool:** from the filtered train items, sample **2,000 successes and 2,000 failures** with `default_rng(0)`, in sorted `item_id` order per class. If a class has fewer than 2,000, take all of it and record the shortfall.
- **The test set:** every filtered test item. An estimated ~1,600 (~480 successes, ~1,120 failures), from 800 test rows read through the Hugging Face viewer API on October 21.

### 2.2 Human: the HoloAssist pretraining pool (`dataset = holoassist_pool`)

- **Candidates:** every fine-grained action with an `Action Correctness` attribute in the **548 audible sessions** (`DATA_ROOT/runs/holoassist_session_audibility.jsonl`, `audible: true`), built with the `holoassist` item definition.
  - **Exclude** every action that is a `holoassist_audible` item, train or test (matched on `(video_name, event index)`).
- **Sample 20,000** with `default_rng(0)`, in the **natural class proportion** (no balancing: free labels come without knowing the outcome).
- **`item_id`** = `holoassist_pool:<video_name>:<event index>`, with `meta.source_item_id` = `holoassist:<video_name>:<event index>`.
- **This pool is a pretraining set, not an evaluation set.** It has no split file.
- **Features:** `siglip-b16-224` (action window) and `e2v-plus-large` (reaction window), through `features.extract`.

## 3. The reaction labeler and the pseudo-labels

- **The labeler** is exactly A13's `react-nonverbal` probe, refit deterministically: `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")` on the `e2v-plus-large` features and true labels of the **`holoassist_audible` train split** (3,000 items).
  - Its parameters (scaler mean and scale, `coef`, `intercept`) are saved to `DATA_ROOT/transfer/labeler.npz`.
- **The pseudo-label** `q_i` = the labeler's P(good) for each pool item, from its `e2v-plus-large` features. It is written to `DATA_ROOT/transfer/holoassist_pool_pseudolabels.jsonl`, one line per item: `{item_id, q, label, group_id, labeler_unseen_group}`.
  - `labeler_unseen_group` is true iff the item's group is a `holoassist_audible` **test** group, i.e. a session the labeler never saw.
- **Pseudo-label quality** is reported, never a gate. Two rows on dataset `holoassist_pool`, split `pool`:
  - `pseudo-label-quality` = AUROC(`q`, `label`) on the `labeler_unseen_group` items;
  - `pseudo-label-quality:all` = the same on all 20,000.

## 4. Features, heads and the detector

- **Robot features:** a new encoder id **`siglip-b16-224-u9`**: the same SigLIP model and device as `siglip-b16-224`.
  - It takes **9 frames at `t_k = k·D/8`, `k = 0…8`** (from the first frame to the last), using `extract_frame`'s existing seek fallback for `t = D`.
  - Each frame is L2-normalized, then mean-pooled → 768-d float32.
  - **Why a new id:** episodes last seconds to minutes, so 1-fps sampling would vary widely in cost and frame count. Nine fixed frames always include the end state, where success shows.
- **One feature space:** a `StandardScaler` `S`, fit **label-free** on the concatenation of the pool's `siglip-b16-224` features and the robot train pool's `siglip-b16-224-u9` features.
  - It is **never** fit on test. Every human head and robot detector uses `S`.
- **Prior-regularized logistic regression** (`transfer.prior_lr`), the only learner in Wave D:
  - **Objective:** minimize `½‖w − w0‖² + C · Σ_i s_i · logloss(y_i, w·x_i + b)`, with `C = 1`, `b` unpenalized and `x` = `S(features)`.
  - **Class weights:** `s_i = n / (2 · n_{y_i})` (balanced) for hard labels.
  - **Solver:** `scipy.optimize.minimize(method="L-BFGS-B", jac=True, options={"maxiter": 2000, "gtol": 1e-6})`, initialized at `w = w0`, `b = 0`.
  - **With `w0 = 0` this is exactly** `sklearn` `LogisticRegression(C=1.0, class_weight="balanced")` on the same features. A test pins this.
- **Soft labels** (the human heads): each item contributes two rows, `(x_i, 1)` with weight `s1 · q_i` and `(x_i, 0)` with weight `s0 · (1 − q_i)`, where `s1 = n / (2 Σ q)` and `s0 = n / (2 Σ (1 − q))`.
- **The human heads** are all fit with `w0 = 0` on the 20,000 pool items:
  - **`w_react`:** the soft labels `q`;
  - **`w_shuffled`:** `q` permuted with `default_rng(0)`, the same videos with the reaction information destroyed;
  - **`w_truth`:** the true labels, hard and balanced. It is a reference: what *labeled* human data would give.
- **Priors for the robot detector:** `none` (`w0 = 0`), `react` (`w_react`), `shuffled` (`w_shuffled`) and `truth` (`w_truth`).

## 5. The curve, the reference lines and the rows

- **The robot label budgets:** `N ∈ {50, 100, 200, 500, 1000}`.
  - For each `N` and each draw `r ∈ {0…19}`, `rng = default_rng(1000·N + r)` draws `N/2` successes and `N/2` failures without replacement from the robot train pool.
  - **All four priors are fit on the same draw** (paired), and each scores the test set with P(success).
- **Metrics,** on the test set:
  - **The `mean20` AUROC of a (prior, N):** the mean over the 20 draws of the test AUROC.
  - **Its CI:** 1,000 group-bootstrap resamples of the test set (`default_rng(0)`, the groups of §2.1), recomputing the 20-draw mean in each, with the 2.5th and 97.5th percentiles.
  - **A gain** `gain:<prior>:N=<n>` = the mean over draws of `AUROC(prior) − AUROC(none)`, with the same paired bootstrap.
  - **`react_minus_shuffled:N=<n>`** = the mean over draws of `AUROC(react) − AUROC(shuffled)`, with the same bootstrap.
- **Reference lines** (context, with no verdict):
  - **`robot-full`:** `w0 = 0` fit on the whole 4,000-item robot train pool;
  - **`judge-large`:** `gemma4:26b`, zero-shot on the test set, under the §8 rules of `03_eval_harness.md` (4 frames over `[0, D]`, the shared-model options, never unloaded by us). The `context_text` is the instruction, and the **question, verbatim**, is `Did the robot complete the task successfully by the end of the video?`;
  - **`roboreward-8b`:** a `not run` row with the notes `not run: RoboReward-8B is Qwen3-VL-based and needs a transformers newer than the frozen 4.46.3 (Issue 9 chose B)`.
- **The leak check:** `detector:none:N=1000:shuffled`. Each draw's robot labels are permuted (`default_rng(1000·N + r + 7)`) before fitting. It fails iff `ci_low > 0.5` (`03_eval_harness.md` §11.9).
- **Scorecard rows** (`hypothesis = "H3"`, `dataset = "roboreward"`, `split = "test"`):

  | Condition | Metric | Count |
  |---|---|---|
  | `detector:<prior>:N=<n>` | `auroc_mean20` | 20 |
  | `gain:<prior>:N=<n>` for `react`, `shuffled`, `truth` | `delta_auroc_mean20` | 15 |
  | `react_minus_shuffled:N=<n>` | `delta_auroc_mean20` | 5 |
  | `robot-full` | `auroc` | 1 |
  | `judge-large` | `auroc` | 1 |
  | `roboreward-8b` | `auroc` | 1 |
  | `detector:none:N=1000:shuffled` | `auroc_mean20` | 1 |

  Plus the two `pseudo-label-quality` rows of §3.

## 6. Pre-registered verdict (H3, offline)

- **Supported** iff **both:**
  - `gain:react:N=100` ≥ **+0.02**, with `ci_low > 0`;
  - **and** `react_minus_shuffled:N=100` has `ci_low > 0`.
- **Not supported** otherwise. The report prints exactly one interpretation, chosen by this table in order (first match):

  | If | Then print |
  |---|---|
  | `gain:truth:N=100` `ci_low ≤ 0` | "Human→robot transfer does not help in this setup even with true human labels; reaction labels cannot be judged here." |
  | `gain:react:N=100` `ci_low ≤ 0` | "Labeled human video transfers, but reaction-derived labels are too weak to." |
  | `react_minus_shuffled:N=100` `ci_low ≤ 0` | "A human-video prior helps regardless of its labels; the gain is not a reaction effect." |
  | otherwise | "The gain is real but below the +0.02 bar." |

- **No re-runs with other budgets, `C`, frame rules, pool sizes, labelers or priors after the result is seen.** A follow-up is a new, dated spec.
- **The thesis's kill rule is unaffected:** it is decided on Wave B (verdict videos) only.

## 7. Where things live

| What | Where |
|---|---|
| RoboReward raw data and `DOWNLOAD.json` | `DATA_ROOT/raw/roboreward/` |
| Items and stats | `DATA_ROOT/items/{roboreward,holoassist_pool}/` |
| The labeler, pseudo-labels, human heads (`heads.npz`), the scaler (`scaler.npz`), the curve's raw per-draw scores (`curve_scores.npz`) | `DATA_ROOT/transfer/` |
| Code | `src/sources/roboreward.py`, `src/transfer/{prior_lr,heads,curve}.py`, `tools/download_roboreward.py`, `tools/report_wave_d.py` |
| The report | `docs/evals/<date>_wave_d_robot_check.md` |

---

## 8. Result of v1 (October 22, 2026), and the designer's post-hoc notes

**The pre-registered verdict is NOT SUPPORTED** (`docs/evals/2026-10-22_wave_d_robot_check.md`; rows `2026-10-22T14:22:28Z`–`14:24:54Z`).
- `gain:react:N=100` = +0.0009 [−0.0006, 0.0024]; `react_minus_shuffled:N=100` = +0.0008 [−0.0013, 0.0029]; `gain:truth:N=100` = −0.0507 [−0.0779, −0.0240].
- The interpretation printed by §6 is row 1, the true-label gain's `ci_low ≤ 0`.
- **References:** `robot-full` 0.657 [0.624, 0.690]; `judge-large` 0.747 [0.720, 0.774]; the leak check passes.
- **The verdict stands. These notes do not change it.** They record what the designer found when verifying the run, so that no one reads more into v1 than it supports.

**Two flaws in this spec** (the implementation followed it faithfully):
1. **§2.2's exclusion depleted the pool's mistakes.**
   - Wave A's balanced `holoassist_audible` sample had taken about 92% of the audible sessions' mistakes, so the pool has 104 mistakes in 20,000 (0.52%), and none in the sessions the labeler never saw. §3's primary quality row is therefore undefined.
   - The `truth` head, fit on 104 mistakes, is overfit: in-sample AUROC 0.997, norm 26.9.
2. **§4's prior strength is the head's norm, which nothing controlled.** The norms are `react` 0.99, `shuffled` 1.03 and `truth` 26.9.
   - At N = 100, the `react` and `shuffled` priors moved test predictions by 0.015 on average; `truth` moved them by 0.33.
   - So "truth hurts" mostly measures an oversized prior from an overfit head, not transfer itself.

**Exploratory diagnostics** (measured after the result was seen, so they are not evidence for any verdict):
- **Each human head on its own, scored on the robot test set:** `react` 0.508 [0.475, 0.540], `shuffled` 0.521 [0.487, 0.554], `truth` 0.491 [0.457, 0.522]. All are at chance, and every head's direction is orthogonal to the no-prior robot detector's (cosine ≈ 0.00).
- **Well-posed true-label heads:** one fit on `holoassist_audible` train (1,500 + 1,500) scores 0.550 [0.515, 0.585] on robot test; one fit on full `holoassist` train scores 0.472 [0.437, 0.509].
- **Reading:** HoloAssist's "correct step" carries little robot-success information in SigLIP space, so the binding constraint is the human data's domain. A version with both flaws fixed would most likely also be null.

**Decision (October 22, Issue 10 = A, the maintainer):**
- **v1 is Wave D's result.** No v2 is run.
- **Robot transfer is revisited only after Wave B,** through a new pre-registered spec with Wave B's reaction model and a human source closer to robot manipulation (DW7, `ongoing_general_errors.md` §4).
- **That spec must apply L15 and L16:** check the pool's class counts per stratum; set the priors at equal norm; check each head's zero-shot transfer first.
