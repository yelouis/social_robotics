# Do Observer Reactions Add Information Beyond the Action? A Pre-registered Negative Result for Reaction-Based Robot Reward

[AUTHOR NAME], Independent Researcher

## Abstract

Can a robot learn which actions succeeded or failed from spontaneous human reactions—facial expressions, posture, vocal prosody, or intervention timing—without explicit human ratings? We test whether observer reactions add outcome information beyond what action-only visual probes, task-step priors, and zero-shot vision-language model (VLM) judges already infer from the action video. Under pre-registered hypotheses, grouped bootstrap confidence intervals, and paired ablations, we evaluate five human-reaction datasets across Oops!, HoloAssist, and REFLEX plus an offline human-to-robot transfer test on RoboReward. Across every dataset we could obtain, spontaneous reactions add almost nothing beyond action-only models. On visible failures (`oops`), reactions add nothing measurable (0.015 [-0.001, 0.029]). On egocentric task assistance (`holoassist_audible`), instructor voice yields a small ablation gain (0.021 [0.006, 0.034]) that misses our +0.03 pass bar and mostly reflects whether the instructor spoke. On physical robot failures (`reflex`, `reflex_resolve`), neither dataset achieves a positive ablation gain, and the sole `PROCEED` clause (`react-face` 0.609 [0.569, 0.644]) is explained by a window-length artifact (0.735 [0.675, 0.796] from window length alone; chance on equal-length windows). Reaction-labeled human video also fails to improve robot success detection on `roboreward` (+0.0009 [-0.0006, 0.0024]). Finally, we document five traps that manufacture false positives in reaction-reward studies.

---

## §1. Introduction

When people watch an agent or another person perform a physical task, they react spontaneously to how the action unfolds: a bystander winces when an object drops, an instructor speaks up when an assembly step goes wrong, or a collaborator shifts posture when a robot places a block in the wrong bin. Human-robot interaction and reinforcement learning studies propose treating those implicit human reactions as a free reward or error signal. In the laboratory, EMPATHIC mapped spectator facial reactions to task reward and optimality statistics, ranking trajectory events and improving policies without explicit button presses (Cui, Zhang, Allievi, Stone, Niekum, Knox, CoRL 2020; arXiv:2009.13649). Subsequent benchmarks—including the BAD dataset of webcam reactions to failure videos (Bremers et al., IROS 2023; arXiv:2303.04835), the ERR@HRI multimodal error-detection challenges (2024; arXiv:2407.06094; `ERR@HRI 3.0` at ICMI 2026; arXiv:2607.11570), TAMER facial-feedback studies (Li et al., 2020; arXiv:2001.08703), and the physical human-robot collaboration dataset REFLEX (Naoum et al., HRI 2025; arXiv:2502.14185)—showed that classifiers can predict task failures or evaluative feedback from human facial, postural, or vocal cues above chance (0.5).

Yet beating chance (0.5) from reactions alone is weaker than showing that reactions provide *new* information for robot learning. In practice, the system also observes the action itself—the robot gripper, tabletop objects, egocentric manipulation frames, and task step. Pretrained vision encoders `[CITATION NEEDED: SigLIP reference]` and zero-shot vision-language model (VLM) judges `[CITATION NEEDED: Qwen2.5-VL reference]` `[CITATION NEEDED: Gemma 4 reference]` can inspect action frames directly, while action-only reward models such as DVD (RSS 2021; arXiv:2103.16817), RoboReward (2026; arXiv:2601.00675), and TOPReward (2026; arXiv:2602.19313) score trajectories without human reactions. If an action-only probe, a task-step prior, or a VLM judge already captures everything the observer reacts to, then facial or vocal reaction sensors add complexity and privacy risk without improving reward estimation. Conversely, observer reactions should help where the outcome is *hidden* from the action camera or depends on human domain knowledge.

This paper asks: **do spontaneous observer reactions (voice, face, gaze, and posture) add outcome information beyond what an action-only model already sees, especially when the outcome is hidden, and does reaction-labeled human video improve sample efficiency for robot success detection?**

After retiring an earlier categorical-emotion prototype (`v0-saf-final`, retired October 8, 2026) because human raters could not reliably label emotions without an objective outcome, we redesigned our study around outcome supervision, pre-registered pass bars, and falsification gates. Across three waves—Wave A on internet failure clips (`oops`) and first-person guidance (`holoassist`, `holoassist_audible`), Wave B on humans reacting to a physical Baxter robot (`reflex`, `reflex_resolve`), and Wave D on human-to-robot transfer (`roboreward`)—the empirical answer across every dataset we could obtain is **almost nothing**. On visible failures (`oops`), reactions add nothing measurable (0.015 [-0.001, 0.029]). On hidden-outcome guidance (`holoassist_audible`), instructor voice yields a small ablation gain (0.021 [0.006, 0.034]) that misses our +0.03 pass bar and is mostly whether the instructor spoke (`react-spoke` 0.665 [0.636, 0.691]; +0.012 [-0.004, 0.030] within spoken items). On physical robot failures (`reflex`, `reflex_resolve`), neither dataset achieves a positive ablation gain (0.023 [-0.004, 0.045] and 0.036 [-0.002, 0.078]), and the sole `PROCEED` clause (`reflex` `react-face` 0.609 [0.569, 0.644]) is explained by a window-length artifact found in verification (0.735 [0.675, 0.796] from window length alone; 0.525 [0.473, 0.572] and 0.511 [0.457, 0.561] on equal-length windows). Pre-training on 20,000 reaction-labeled human clips also yields zero gain on `roboreward` (+0.0009 [-0.0006, 0.0024] at 100 robot labels).

Verification also exposed **five methodological traps** that produce false positives in reaction-reward studies: (i) step annotation text leaking labels (`drop` steps in HoloAssist fail 95.4% of the time, 415 of 435 items); (ii) `max` and `std` summaries over variable-duration windows reading window length rather than facial expression (§11.10, L18); (iii) GDPR privacy masks hiding facial pixels while leaking body silhouettes into "action-only" video (§11.11, L19); (iv) single-seed shuffled controls triggering false alarms (0.603 [0.507, 0.691] at seed 0 vs. a 50-seed null mean of 0.510, SD 0.063; and 0.464 [0.440, 0.489] vs. 0.496, SD 0.030; §11.9, L14); and (v) naive feature concatenation causing fusion to lose to its strongest input (-0.065 [-0.113, -0.025] on `holoassist_audible` and -0.192 [-0.315, -0.037] on `reflex_resolve`, L12).

We make four contributions:
- **A pre-registered, multi-dataset test:** five H1 datasets (`oops`, 2,710 items; `holoassist`, 5,000; `holoassist_audible`, 4,420; `reflex`, 1,208; `reflex_resolve`, 604) plus offline robot transfer (`roboreward`, 5,517), with frozen splits, models, and decision rules.
- **An incremental-value harness:** pairing frozen encoders with zero-shot VLM judges, action probes, `step-prior`, and a fixed-combiner ablation (`fusion_minus_action_fusion`).
- **A transparent negative result:** reactions add at most +0.021 [0.006, 0.034] AUROC over action-only models and +0.0009 [-0.0006, 0.0024] in robot transfer.
- **Five empirical traps and guardrails:** exact mechanisms, post-hoc diagnostics, and design rules for future benchmarks.

---

## §2. Related Work

**Implicit human reactions as reward and error signals.** Implicit facial, vocal, and postural cues have been studied in laboratory reinforcement learning to reduce reliance on explicit feedback. Li et al. (2020; arXiv:2001.08703) analyzed trainer facial expressions in TAMER, showing they predict evaluative feedback and assist policy learning when combined with explicit rewards. Cui, Zhang, Allievi, Stone, Niekum, and Knox (CoRL 2020; arXiv:2009.13649) introduced EMPATHIC, mapping spectator facial reactions during sub-optimal agent play to reward and optimality statistics to rank events, improve policies, and score robot trajectories. Subsequent benchmarks targeted robot error detection from bystander reactions: Bremers et al. (IROS 2023; arXiv:2303.04835) collected the BAD dataset of webcam reactions to human and robot failure videos, the ERR@HRI challenges (2024; arXiv:2407.06094; `ERR@HRI 3.0` at ICMI 2026; arXiv:2607.11570) standardized multimodal error detection across face, head, and body features, and REACT (2024; arXiv:2402.00190) studied reactions and evaluative feedback over time. Most recently, Naoum et al. (HRI 2025; arXiv:2502.14185) released REFLEX, recording 55 participants sorting objects with a Baxter robot across 604 scripted pick, carry, and place failures and four explanation levels (`zero`, `low`, `medium`, `high`). Whereas prior studies test whether reactions beat chance (0.5) alone, we test whether they add information *above* action-only probes, step priors, and VLM judges.

**Video datasets with objective action outcomes.** Our retired prototype (`v0-saf-final`) showed that reaction models must be trained on objective outcomes rather than rater guesses about emotion. For visible failures, Epstein, Chen, and Vondrick (CVPR 2020) introduced Oops!, annotating the timestamp where intentional action becomes unintentional failure (2,710 windows across 1,355 clips here). For hidden or partially visible outcomes in egocentric manipulation, Wang et al. (ICCV 2023; arXiv:2309.17024) released HoloAssist, where a HoloLens performer executes physical tasks while a remote instructor intervenes verbally on mistakes (5,000 items in `holoassist` and 4,420 in the audible subset `holoassist_audible`). Paired with REFLEX (`reflex`, 1,208 items; `reflex_resolve`, 604 items), these datasets span internet video, egocentric guidance, and collaborative robotics.

**Action-only visual reward models and zero-shot VLM judges.** Reaction rewards must compete with models that score action video directly. DVD (RSS 2021; arXiv:2103.16817) learned generalizable robot rewards from human and robot task videos without bystander reactions. RoboReward (2026; arXiv:2601.00675) trained a vision-language reward model over Open X-Embodiment and RoboArena rollouts, while TOPReward (2026; arXiv:2602.19313) used pretrained VLM token probabilities as zero-shot rewards. Our harness includes zero-shot VLM judges (`qwen2.5vl:7b` `[CITATION NEEDED: Qwen2.5-VL reference]`, `gemma4:26b` `[CITATION NEEDED: Gemma 4 reference]`, and `gemini-3.6-flash` `[CITATION NEEDED: Gemini reference]`) and linear probes over frozen SigLIP embeddings `[CITATION NEEDED: SigLIP reference]`, and tests human-to-robot transfer on RoboReward (4,000 train and 1,517 test rollouts).

---

## §3. Hypotheses and Pre-registration

Every hypothesis, condition, split, probe hyperparameter, and pass/kill rule was committed to version-controlled contracts before test evaluation (`00_thesis.md`, `03_eval_harness.md`, `04_wave_d_robot_check.md`, `05_wave_b_reflex.md`). Post-hoc notes sit beside pre-registered verdicts and never overwrite scorecard rows.

**Hypothesis H1 (Outcome Signal Beyond the Action).** On tasks with objective outcomes (`label = 1` for normal/correct or resolved execution; `label = 0` for failure, mistake, or unresolved outcome), H1 predicts contrasting behavior for visible versus hidden outcomes (October 8, 2026):
- **Visible outcomes (`oops`):** The zero-shot VLM judge is strong (**Clause one**, `ci_low > 0.5` and within 0.05 of `action-probe`), and adding reactions yields near-zero gain (**Clause two**, `fusion_minus_action_fusion` CI covering 0.0).
- **Hidden outcomes (`holoassist`, `holoassist_audible`, `reflex`, `reflex_resolve`):** The headline reaction-only probe (`react-nonverbal` on HoloAssist; `react-face` on REFLEX) beats `action_best` (**Clause one**, `ci_low > action_best`), and multimodal fusion beats the action-only combiner (**Clause two**, `fusion_minus_action_fusion` `ci_low > 0.0`).
- **Proposed H1 pass bar (amended October 20, 2026, `Decision 20`; November 1, 2026, `Decision 28`):** Headline reaction-only AUROC >= 0.65 **and** paired ablation gain `fusion_minus_action_fusion` >= +0.03 with `ci_low > 0.0`. We amended the delta from `fusion_minus_action_best` to `fusion_minus_action_fusion` on October 20, 2026 after observing on `holoassist` that concatenating SigLIP features with `step-prior` and `judge` (`action-fusion` 0.681 [0.657, 0.706]) diluted `step-prior` (0.751 [0.726, 0.774]); holding the combiner fixed isolates the marginal contribution of reactions.
- **Restated Wave B kill rule (November 1, 2026, `Decision 28`):** On `reflex` and `reflex_resolve`, if **neither** dataset achieves `fusion_minus_action_fusion` `ci_low > 0.0` **and** `react-face` has `ci_low <= 0.5` on both, the project stops; if either holds on at least one dataset, the pre-registered verdict is `PROCEED`. Wave B recorded `PROCEED` on November 4, 2026 solely because `reflex` `react-face` scored 0.609 [0.569, 0.644] (`ci_low > 0.5`), which verification showed is a window-length artifact (0.735 [0.675, 0.796] from window length alone). The `PROCEED` verdict stands in the scorecard alongside the maintainer's November 5, 2026 decision (`Issue 12 = A`) recording Wave B as negative in substance.

**Hypothesis H3 (Offline Robot Data Efficiency, Wave D).** Pre-registered on October 21, 2026 (`04_wave_d_robot_check.md`) after `holoassist_audible` met its trigger (0.021 [0.006, 0.034], `ci_low > 0.0`), H3 tests whether regularizing a robot success probe toward a human reaction-pseudo-label head trained on 20,000 `holoassist_pool` clips improves `roboreward` AUROC across budgets `N` in {50, 100, 200, 500, 1,000} (20 draws per budget). H3 is **supported** iff at `N=100`, `gain:react:N=100` is >= +0.02 with `ci_low > 0.0` **and** `react_minus_shuffled:N=100` has `ci_low > 0.0`; otherwise H3 is **not supported**.

---

## §4. Data

Every dataset pairs an action window with an observer reaction window (except `roboreward`, which contains robot rollouts for H3) and an objective outcome label. All splits are group-disjoint (`default_rng(0)`) with locked file digests (Table `T1`).

<!-- table:T1 -->
| Name | Outcome visibility | Reaction channel(s) | Items | Train / test items | Test groups | Test positive rate |
|---|---|---|---|---|---|---|
| `oops` | Visible | Non-verbal audio (`e2v-plus-large`) | 2,710 | 1,638 / 1,072 | 90 | 0.500 (536/1072) |
| `holoassist` | Hidden / partly visible | Instructor voice (`e2v-plus-large`), speech presence (`react-spoke`) | 5,000 | 3,000 / 2,000 | 91 | 0.500 (1000/2000) |
| `holoassist_audible` | Hidden / partly visible | Instructor voice (`e2v-plus-large`), speech presence (`react-spoke`) | 4,420 | 3,000 / 1,420 | 53 | 0.500 (710/1420) |
| `reflex` | Partly visible | Face/gaze/posture (`reflex-face-v1`), speech prosody (`reflex-speech-v1`) | 1,208 | 834 / 374 | 17 | 0.500 (187/374) |
| `reflex_resolve` | Hidden | Face/gaze/posture (`reflex-face-v1`), speech prosody (`reflex-speech-v1`) | 604 | 417 / 187 | 17 | 0.802 (150/187) |
| `roboreward` | Visible | None (robot rollouts; H3 transfer target) | 5,517 | 4,000 / 1,517 | 888 | 0.347 (527/1517) |
<!-- /table:T1 -->

**Oops! (`oops`).** From the validated failure-onset timestamps of Epstein, Chen, and Vondrick (CVPR 2020), each clip yields a pre-failure item (`label = 1`, `[t - 4, t - 1]` s) and a post-failure item (`label = 0`, `[t, t + 3]` s), totaling 2,710 items across 1,355 clips (1,638 train across 320 groups with 819 positives; 1,072 test across 90 groups with 536 positives, 50.0% positive rate). The reaction window equals the action window and contains both filmer vocalizations and failure impact sounds.

**HoloAssist (`holoassist` and `holoassist_audible`).** From Wang et al. (ICCV 2023; arXiv:2309.17024), each item is a fine-grained action segment labeled `"Correct"` (`label = 1`) or `"Wrong"` (`label = 0`), grouped by performer-instructor session prefix. The action window is `[max(a_start, a_end - 2.0), a_end]` s and the reaction window is `[a_end - 0.5, a_end + 3.0]` s. On `holoassist` (5,000 items: 3,000 train across 218 groups with 1,500 positives; 2,000 test across 91 groups with 1,000 positives), our 20-session audio gate failed due to loud tool noise on the headset microphone. Per `Issue 6 = A`, a label-blind audit selected audible sessions to form `holoassist_audible` (4,420 items: 3,000 train across 131 groups with 1,500 positives; 1,420 test across 53 groups with 710 positives, 50.0% positive rate).

**REFLEX (`reflex` and `reflex_resolve`).** From Naoum et al. (HRI 2025; arXiv:2502.14185; 55 participants, 604 Baxter sorting episodes at 4.4 fps in `video_cam1.mp4`, split into 38 train and 17 test participants), `reflex` (1,208 items: 834 train with 417 positives; 374 test with 187 positives) pairs each episode's pre-failure state (`label = 1`, `[Pre.Start, Pre.End]`) with its failure state (`label = 0`, action `[Failure.Start, Failure.End]`, reaction extended 22 frames to `min(Failure.End + 22, Explanation.End)`). Second, `reflex_resolve` (604 items after normalizing `C3-2`'s `'Resloved?'` header: 488 `TRUE`, 116 `FALSE`; 417 train with 338 positives; 187 test with 150 positives, 80.2% positive rate) pairs action `[Failure.Start, Explanation.End]` with reaction `[Failure.Start, Resolution.End]` to predict `Resolved? == TRUE`. Although `video_cam1.mp4` applies a black privacy mask over the participant, the mask preserves the participant's full-body silhouette, head tilts, and gestures (L19).

**RoboReward and HoloAssist Pool (`roboreward`, `holoassist_pool`).** From RoboReward (2026; arXiv:2601.00675), excluding counterfactual-instruction negatives yields 4,000 balanced training rollouts (2,000 successes) and 1,517 test rollouts (527 successes, 34.7% positive rate, across 888 groups; 5,517 total). The pre-training pool `holoassist_pool` has 20,000 non-overlapping HoloAssist clips across 185 groups (5,682 in the 53 audible test groups) scored by the `holoassist_audible` `react-nonverbal` probe.

---

## §5. Methods

**Evaluation conditions (`03_eval_harness.md` §6).** We evaluate up to 20 conditions per dataset:
- **Action-only visual judges (`judge`, `judge-large`, `judge-frontier`):** Zero-shot VLMs score action-window frames and `context_text` on a 0 to 100 scale divided by 100, with strict audio isolation (`SR_NO_AUDIO`). `judge` uses `qwen2.5vl:7b`, `judge-large` uses `gemma4:26b` (`temperature = 0`), and `judge-frontier` uses `gemini-3.6-flash` (entering `action_best` only at >= 90% coverage).
- **Action-only visual probe (`action-probe`):** Linear probe on frozen SigLIP embeddings (`siglip-b16-224` on Wave A; `siglip-b16-224-u9` on Waves B and D).
- **Task-step prior (`step-prior`):** Train-split `P(label = 1 | verb, noun)` with hierarchical smoothing and leave-one-out (`LOO`) train counts (`03_eval_harness.md` §6).
- **Action combiner (`action-fusion`) and `action_best`:** Logistic regression on `[logit(judge), logit(step-prior), X_action]`; `action_best` is the highest-AUROC action-only condition.
- **Reaction-only conditions (`react-nonverbal`, `react-face`, `react-spoke`, `react-full`):** Linear probes on `e2v-plus-large` or `reflex-speech-v1` (`react-nonverbal` and `react-nonverbal|spoke=1`), `reflex-face-v1` (`react-face`), binary silence `1.0 - float(spoke)` (`react-spoke`), and transcript TF-IDF (`react-full`, a leakage context row only, §11.2).
- **Multimodal fusion (`fusion`) and ablation (`fusion_minus_action_fusion`):** `fusion` concatenates `action-fusion` inputs with standardized reaction features; `fusion_minus_action_fusion` is the paired test AUROC difference on identical items.

**Encoders and probes (`03_eval_harness.md` §9).** All encoders are frozen: `google/siglip-base-patch16-224` `[CITATION NEEDED: SigLIP reference]`, `iic/emotion2vec_plus_large` `[CITATION NEEDED: emotion2vec+ reference]`, `ReflexFaceEncoder` (`mean`, `std`, `max` over 50 OpenFace `[CITATION NEEDED: OpenFace reference]`, Facetorch `[CITATION NEEDED: Facetorch reference]`, and body columns plus head angular-velocity `std`), and `ReflexSpeechEncoder` (duration-weighted `mean` and `max` over Hume prosody `[CITATION NEEDED: Hume prosody reference]`). Every probe uses a train-only `StandardScaler` and `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")` `[CITATION NEEDED: scikit-learn reference]`.

**Metrics and leak checks (`03_eval_harness.md` §5, §11.9).** We report AUROC with confidence intervals from 1,000 group-bootstrap resamples (`default_rng(0)`), using paired group draws for deltas. Each learned condition has a `:shuffled` control permuting `y_train` (`default_rng(0)`) and recomputing `step-prior` tables. A `:shuffled` row fails iff `ci_low > 0.5`, which triggers a 50-seed permutation null (`seeds 0..49`, L14). All runs used one Apple Mac Studio with no fine-tuning under `shared.memguard`.

---

## §6. Results

We present H1 results across the five human-reaction datasets (Table `T2`, Table `T3`, Figure `F1`) and H3 transfer results on `roboreward` (Table `T5`, Figure `F3`), organized by claims **`C1`–`C4`**.

<!-- table:T2 -->
| Condition | `oops` | `holoassist` | `holoassist_audible` | `reflex` | `reflex_resolve` |
|---|---|---|---|---|---|
| `judge` | 0.472 [0.440, 0.498] | 0.488 [0.466, 0.512] | 0.528 [0.499, 0.560] | 0.720 [0.680, 0.761] | 0.473 [0.422, 0.523] |
| `judge-large` | 0.708 [0.684, 0.731] | 0.544 [0.528, 0.562] | 0.516 [0.499, 0.533] | 0.500 [0.500, 0.500] | 0.474 [0.409, 0.548] |
| `action-probe` | 0.779 [0.754, 0.805] | 0.569 [0.542, 0.597] | 0.576 [0.526, 0.623] | 0.730 [0.705, 0.767] | 0.721 [0.604, 0.853] |
| `step-prior` | n/a | 0.751 [0.726, 0.774] | 0.737 [0.712, 0.765] | 0.500 [0.500, 0.500] | 0.938 [0.883, 0.980] |
| `action-fusion` | 0.778 [0.754, 0.805] | 0.681 [0.657, 0.706] | 0.650 [0.599, 0.694] | 0.794 [0.744, 0.841] | 0.713 [0.595, 0.859] |
| `react-nonverbal` | 0.711 [0.684, 0.740] | n/a | 0.654 [0.626, 0.675] | 0.526 [0.493, 0.568] | 0.660 [0.545, 0.766] |
| `react-face` | n/a | n/a | n/a | 0.609 [0.569, 0.644] | 0.552 [0.440, 0.663] |
| `react-spoke` | n/a | 0.676 [0.655, 0.695] | 0.665 [0.636, 0.691] | 0.519 [0.503, 0.537] | 0.505 [0.439, 0.566] |
| `fusion` | 0.793 [0.771, 0.816] | n/a | 0.671 [0.618, 0.713] | 0.817 [0.772, 0.857] | 0.749 [0.645, 0.877] |
| `fusion_minus_action_fusion` | 0.015 [-0.001, 0.029] | n/a | 0.021 [0.006, 0.034] | 0.023 [-0.004, 0.045] | 0.036 [-0.002, 0.078] |
| `fusion_minus_action_best` | 0.015 [-0.002, 0.029] | n/a | -0.065 [-0.113, -0.025] | 0.023 [-0.004, 0.045] | -0.192 [-0.315, -0.037] |
<!-- /table:T2 -->

<!-- table:T3 -->
| Dataset | Check | Verdict string |
|---|---|---|
| `oops` | Clause 1, "the judge is strong" (judge) | AUROC 0.472 [0.440, 0.498] (1072 scored) → did not hold |
| `oops` | Clause 1, "the judge is strong" (judge-large) | AUROC 0.708 [0.684, 0.731] (1072 scored) → did not hold |
| `oops` | Clause 1, "the judge is strong" (judge-frontier) | AUROC 0.704 [0.544, 0.821] (50 scored) → did not hold |
| `oops` | Clause 2, "Δ ≈ 0" | ablation Δ 0.015 [-0.001, 0.029] → held |
| `holoassist` | Clause 1, "reaction-only beats action_best" | react-nonverbal not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large; rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20) vs step-prior 0.751 → not tested: not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large; rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20) |
| `holoassist` | Clause 2, "reactions add to the action-only model" | ablation Δ not run: fusion not run (not run: react-nonverbal not run on train); rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20) → not tested: not run: fusion not run (not run: react-nonverbal not run on train); rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20) |
| `holoassist` | H1 pass bar (proposed, 00_thesis.md) | react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not testable: react-nonverbal, ablation Δ |
| `holoassist_audible` | Clause 1, "reaction-only beats action_best" | react-nonverbal 0.654 [0.626, 0.675] vs step-prior 0.737 → did not hold |
| `holoassist_audible` | Clause 2, "reactions add to the action-only model" | ablation Δ 0.021 [0.006, 0.034] → held |
| `holoassist_audible` | H1 pass bar (proposed, 00_thesis.md) | react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not met |
| `reflex` | Clause 1, "reaction-only beats action_best" | react-face 0.609 [0.569, 0.644] vs action-fusion 0.794 → did not hold |
| `reflex` | Clause 2, "reactions add to the action-only model" | ablation Δ 0.023 [-0.004, 0.045] → did not hold |
| `reflex` | H1 pass bar (proposed, 00_thesis.md) | react-face ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not met |
| `reflex_resolve` | Clause 1, "reaction-only beats action_best" | react-face 0.552 [0.440, 0.663] vs step-prior 0.938 → did not hold |
| `reflex_resolve` | Clause 2, "reactions add to the action-only model" | ablation Δ 0.036 [-0.002, 0.078] → did not hold |
| `reflex_resolve` | H1 pass bar (proposed, 00_thesis.md) | react-face ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not met |
| `reflex` / `reflex_resolve` | Wave B Kill Rule | **PROCEED** — PROCEED to DW3 and DW6: at least one dataset demonstrated react-face beats chance (ci_low > 0.5). (reflex: react-face ci_low=0.569, ablation ci_low=-0.004; reflex_resolve: react-face ci_low=0.440, ablation ci_low=-0.002) |
<!-- /table:T3 -->

![Figure 1: Paired ablation gain (fusion minus action-fusion) with 95% group-bootstrap confidence intervals across the five H1 datasets (n_test = 1,072 on oops, not run on holoassist, n_test = 1,420 on holoassist_audible, n_test = 345 on reflex, and n_test = 173 on reflex_resolve) plus the length-controlled REFLEX diagnostic (n_test = 345).](figures/f1_ablation.png)

**Claim `C1` (`oops`, visible outcomes): Reactions add nothing measurable to an action-only model.** On `oops` (1,072 test items across 90 groups), where failures are plainly visible, `action-probe` scores 0.779 [0.754, 0.805] and `judge-large` (`gemma4:26b`) scores 0.708 [0.684, 0.731] (`judge` scores 0.472 [0.440, 0.498]; the 50-item `judge-frontier` sample scores 0.704 [0.544, 0.821]). Clause one ("the judge is strong", within 0.05 of `action-probe`) did not hold because 0.708 [0.684, 0.731] trails 0.779 [0.754, 0.805] by more than 0.05, though `judge-large` is well above chance zero-shot. Clause two held: adding `e2v-plus-large` audio to `action-fusion` (0.778 [0.754, 0.805]) yields `fusion` 0.793 [0.771, 0.816], a paired ablation gain `fusion_minus_action_fusion` of 0.015 [-0.001, 0.029] (`fusion_minus_action_best` 0.015 [-0.002, 0.029]), whose CI covers 0.0. Even standalone `react-nonverbal` (0.711 [0.684, 0.740]) reflects failure impact audio as much as human vocal reaction.

**Claim `C2` (`holoassist` and `holoassist_audible`, hidden outcomes): Instructor voice adds a small gain only where audible, and most of it is whether the instructor spoke.** On `holoassist` (2,000 test items across 91 groups), acoustic extraction was halted after failing the 20-session audio gate; visual models scored modestly (`action-probe` 0.569 [0.542, 0.597], `judge` 0.488 [0.466, 0.512], `judge-large` 0.544 [0.528, 0.562]) compared to `react-spoke` (0.676 [0.655, 0.695]) and `step-prior` (0.751 [0.726, 0.774]). On the audible subset `holoassist_audible` (1,420 test items across 53 groups), `react-nonverbal` scores 0.654 [0.626, 0.675], raising `action-fusion` (0.650 [0.599, 0.694]) to `fusion` 0.671 [0.618, 0.713]. The paired ablation gain `fusion_minus_action_fusion` is 0.021 [0.006, 0.034] (Clause two held, `ci_low > 0.0`), the sole positive ablation interval in our study (Figure `F1`). However, 0.021 [0.006, 0.034] misses the +0.03 H1 pass bar (`not met`), Clause one (`react-nonverbal` 0.654 [0.626, 0.675] vs. `step-prior` 0.737 [0.712, 0.765]) did not hold, and `fusion` loses to `step-prior` (`fusion_minus_action_best` -0.065 [-0.113, -0.025]). Moreover, binary speech presence (`react-spoke` 0.665 [0.636, 0.691]) beats `react-nonverbal` (0.654 [0.626, 0.675]), `react-nonverbal|spoke=1` on the 837 spoken test items drops to 0.582 [0.548, 0.612], and the `recorded` `wave_a/audible_strata` check shows the ablation gain is +0.012 [-0.004, 0.030] where the instructor spoke, -0.026 where silent, and +0.025 without `drop`.

**Claim `C3` (`reflex` and `reflex_resolve`, physical human-robot collaboration): No reaction channel adds to action-only models, and the face signal that triggered `PROCEED` is a window-length artifact.** On `reflex` (374 test items across 17 participants; 345 face-valid, 109 spoken), action models detect failures (`action-probe` 0.730 [0.705, 0.767], `judge` 0.720 [0.680, 0.761], `action-fusion` 0.794 [0.744, 0.841]; `judge-large` outputs 100 on all 374 items for 0.500 [0.500, 0.500]). Adding `react-face` gives `fusion` 0.817 [0.772, 0.857] and an ablation gain `fusion_minus_action_fusion` of 0.023 [-0.004, 0.045] covering 0.0 (Clause two did not hold; pass bar not met). On `reflex_resolve` (187 test items across 17 participants; 173 face-valid, 140 spoken), `step-prior` dominates at 0.938 [0.883, 0.980] (0.941 [0.886, 0.980] on the 173 face-valid items), with `action-probe` at 0.721 [0.604, 0.853], `react-nonverbal` at 0.660 [0.545, 0.766], `react-face` at 0.552 [0.440, 0.663], and `fusion_minus_action_fusion` at 0.036 [-0.002, 0.078] (`ci_low = -0.002 <= 0.0`, Clause two did not hold; `fusion` 0.749 [0.645, 0.877] trails `step-prior` by -0.192 [-0.315, -0.037]). Wave B recorded `PROCEED` solely because `reflex` `react-face` scored 0.609 [0.569, 0.644] (0.6085 at four decimal places, `ci_low > 0.5`), which §7.2 shows is a pure window-length artifact (45.7 vs. 35.5 frames; window length alone scores 0.735 [0.675, 0.796]; equal-length windows drop `react-face` to 0.525 [0.473, 0.572] and 0.511 [0.457, 0.561]).

<!-- table:T5 -->
| Budget `N` / Reference | `detector:none` | `gain:react` | `gain:shuffled` | `gain:truth` | `react_minus_shuffled` |
|---|---|---|---|---|---|
| `N=50` | 0.562 [0.547, 0.577] | 0.001 [-0.001, 0.003] | 0.000 [-0.002, 0.002] | -0.049 [-0.077, -0.022] | 0.001 [-0.002, 0.004] |
| `N=100` | 0.579 [0.559, 0.598] | 0.001 [-0.001, 0.002] | 0.000 [-0.001, 0.002] | -0.051 [-0.078, -0.024] | 0.001 [-0.001, 0.003] |
| `N=200` | 0.596 [0.575, 0.617] | 0.000 [-0.001, 0.002] | -0.000 [-0.001, 0.001] | -0.042 [-0.068, -0.018] | 0.000 [-0.001, 0.002] |
| `N=500` | 0.611 [0.588, 0.630] | 0.000 [-0.001, 0.001] | 0.000 [-0.000, 0.001] | -0.026 [-0.047, -0.007] | -0.000 [-0.002, 0.001] |
| `N=1000` | 0.625 [0.602, 0.647] | -0.000 [-0.001, 0.000] | 0.000 [-0.000, 0.001] | -0.006 [-0.022, 0.008] | -0.000 [-0.001, 0.000] |
| `robot-full` | 0.657 [0.624, 0.690] | n/a | n/a | n/a | n/a |
| `judge-large` | 0.747 [0.720, 0.774] | n/a | n/a | n/a | n/a |
<!-- /table:T5 -->

![Figure 3: Wave D offline human-to-robot transfer curve on RoboReward (n_test = 1,517 across 888 groups), showing 20-draw mean test AUROC and 95% group-bootstrap confidence bands across robot label budgets N in {50, 100, 200, 500, 1000} for detectors regularized toward no human prior, the reaction pseudo-label prior, the shuffled pseudo-label prior, and the true human-label prior.](figures/f3_wave_d.png)

**Claim `C4` (`roboreward`, Wave D offline H3): Reaction-labeled human video did not improve robot success detection at any label budget.** On `roboreward` (1,517 test clips across 888 groups; Table `T5`, Figure `F3`), `detector:none` rises from 0.562 [0.547, 0.577] at `N=50` to 0.579 [0.559, 0.598] at `N=100`, 0.596 [0.575, 0.617] at `N=200`, 0.611 [0.588, 0.630] at `N=500`, 0.625 [0.602, 0.647] at `N=1000`, and 0.657 [0.624, 0.690] on all 4,000 clips (`robot-full`), while zero-shot `judge-large` scores 0.747 [0.720, 0.774]. Regularizing toward the `holoassist_pool` reaction head (`pseudo-label-quality:all` 0.621 [0.568, 0.674]) yields `gain:react` of +0.0008 [-0.0012, 0.0028] at `N=50`, +0.0009 [-0.0006, 0.0024] at `N=100` (0.001 [-0.001, 0.002] in Table `T5`), +0.0004 [-0.0006, 0.0015] at `N=200`, 0.0000 [-0.0007, 0.0007] at `N=500`, and -0.0001 [-0.0005, 0.0004] at `N=1000`, with `react_minus_shuffled:N=100` at +0.0008 [-0.0013, 0.0029] (0.001 [-0.001, 0.003]). Regularizing toward the true human-label head (`gain:truth`) hurts (-0.0507 [-0.0779, -0.0240] at `N=100`, or -0.051 [-0.078, -0.024]). Offline H3 is **NOT SUPPORTED**. The `recorded` Wave D diagnostics (`04_wave_d_robot_check.md` §8) show why: `holoassist_audible` had taken 92% of audible mistakes, leaving 104 mistakes in 20,000 pool items (0.52%) so `truth` overfit (0.997 in-sample AUROC, weight norm 26.9 vs. 0.99 for `react` and 1.03 for `shuffled`, shifting predictions by 0.33 vs. 0.015), yet every human head is near chance zero-shot on robot test clips (`react` 0.508 [0.475, 0.540], `shuffled` 0.521 [0.487, 0.554], `truth` 0.491 [0.457, 0.522], and well-posed heads trained on `holoassist_audible`'s 1,500 + 1,500 split or full `holoassist` train score only 0.550 [0.515, 0.585] and 0.472 [0.437, 0.509]; cosine similarity 0.0).

---

## §7. What Fooled Us, and Could Fool You

Our second main contribution (**`C5`**) documents five methodological traps encountered in Waves A, B, and D, each backed by exact diagnostics.

### §7.1 Trap One: Annotation Text and Task Structure Leak Outcome Labels (L12)

Action segments in human-in-the-loop datasets carry step labels (`verb`, `noun`) that are routinely passed in `context_text`. On `holoassist`, auditing step labels (L12) revealed that annotators used `"drop"` for unintentional drops: 435 of 5,000 items have `meta.verb == "drop"`, and 415 of those 435 (95.4%) are mistakes (`label = 0`). Reading `"Step: drop ..."` allowed `judge-large` (0.544 [0.528, 0.562]) to infer mistakes from text, and our `step-prior` baseline (`P(correct | verb, noun)`) scored 0.751 [0.726, 0.774] on `holoassist` and 0.737 [0.712, 0.765] on `holoassist_audible`—beating every visual, audio, and judge model.

Similarly, on `reflex_resolve`, `step-prior` scored 0.938 [0.883, 0.980] (0.941 [0.886, 0.980] on the 173 face-valid test items). An initial B4 report note hallucinated uncomputed counts claiming `Explanation Level` (`meta.noun`) drove resolution (L10). Recomputing directly from `reflex_resolve/items.jsonl` (604 items) showed resolution is high across all explanation levels (`high` 165/187 = 88.2%, `medium` 115/143 = 80.4%, `low` 106/143 = 74.1%, `zero` 102/131 = 77.9%) and is instead dictated by the physical task verb (`meta.verb`): `pick` resolves 219/219 (100.0%), `carry` 198/220 (90.0%), and `place` 71/165 (43.0%).

### §7.2 Trap Two: Window Length Mimics Facial Reactions (L18, §11.10)

Wave B recorded `PROCEED` because `reflex` `react-face` scored 0.609 [0.569, 0.644] (0.608 [0.569, 0.644] in Table `T4`, 0.6085 at four decimal places, on 345 test items), with `ci_low = 0.569 > 0.5`. Without window-length verification, we would have attributed this signal to facial expressions and exported `reflex_face_probe.npz` for robot deployment (`DW6`).

<!-- table:T4 -->
| Diagnostic key | Value [95% CI] | `n_test` |
|---|---|---|
| `reflex.react_face` | 0.608 [0.569, 0.644] | 345 |
| `reflex.window_length` | 0.735 [0.675, 0.796] | 345 |
| `reflex.face_means_only` | 0.515 [0.471, 0.564] | 345 |
| `reflex.length_plus_face_minus_length` | -0.069 [-0.103, -0.030] | 345 |
| `reflex.face_equal_length_start` | 0.525 [0.473, 0.572] | 342 |
| `reflex.face_equal_length_end` | 0.511 [0.457, 0.561] | 343 |
| `reflex_resolve.react_face` | 0.552 [0.440, 0.663] | 173 |
| `reflex_resolve.face_before_resolution` | 0.545 [0.461, 0.634] | 173 |
| `reflex_resolve.nonverbal_before_resolution` | 0.643 [0.480, 0.843] | 104 |
| `reflex_resolve.step_prior` | 0.941 [0.886, 0.980] | 173 |
| `reflex_resolve.step_prior_plus_face_minus_step_prior` | -0.121 [-0.193, -0.059] | 173 |
| `reflex_resolve.nonverbal_shuffled_null` | mean 0.510 (SD 0.063, range [0.330, 0.653], seed 0 = 0.603; true 0.660 beats 50/50) | 140 |
<!-- /table:T4 -->

![Figure 2: Post-hoc window-length diagnostics on REFLEX immediate failure detection (reflex), comparing test AUROC and 95% group-bootstrap confidence intervals for react-face on the original windows (n_test = 345), react-face restricted to its 50 mean dimensions (n_test = 345), react-face on equal-length start-anchored windows (n_test = 342), react-face on equal-length end-anchored windows (n_test = 343), and window length alone (-L, n_test = 345).](figures/f2_window_length.png)

In `05_wave_b_reflex.md`, `Pre` items (`label = 1`) use `[Pre.Start, Pre.End]` (averaging 35.5 frames across 604 items), whereas `Failure` items (`label = 0`) extend 22 frames past `Failure.End` (averaging 45.7 frames). Because `ReflexFaceEncoder` takes `[mean, std, max]` over 50 columns and sample `max`/`std` grow with frame count `L`, `reflex-face-v1` encodes window duration:
- **Window length alone beats `react-face`:** Scoring test items by `-L` yields **0.735 [0.675, 0.796]** (345 items) vs. 0.608 [0.569, 0.644].
- **Means only are at chance:** Keeping only the 50 `_mean` dimensions drops AUROC to **0.515 [0.471, 0.564]** (345 items).
- **Equal-length windows erase the signal:** Truncating each episode's `pre` and `fail` windows to `K = min(L_pre, L_fail)` reduces `react-face` to **0.525 [0.473, 0.572]** start-anchored (342 items) and **0.511 [0.457, 0.561]** end-anchored (343 items).
- **Adding face to length hurts:** `[L, X_face]` trails `[L]` alone by **-0.069 [-0.103, -0.030]** (345 items).
- **Pre-resolution truncation on `reflex_resolve`:** Restricting reactions to `[Failure.Start, Explanation.End]` gives 0.545 [0.461, 0.634] (173 items) for `react-face` and 0.643 [0.480, 0.843] (104 items) for `react-nonverbal`.

### §7.3 Trap Three: Privacy Masks Leak Body Posture into "Action-Only" Controls (L19, §11.11)

We initially treated REFLEX's `video_cam1.mp4` as an action-only view because a black GDPR segmentation mask covers the participant. Inspecting 20 test items (L19) showed that the black mask's contour preserves the participant's upper-body silhouette, head tilts, and arm gestures. In `reflex:D2-1:10:pre` (`label = 1`), while Baxter holds a dark box above a blue tray, the black-masked participant tilts their head, turns left, and raises their right hand to their head. In `reflex:D1-1:4:fail` (`label = 0`), while Baxter's gripper stalls over a blue block, the masked silhouette leans forward watching the stuck gripper. Thus `judge` (0.720 [0.680, 0.761]) and `action-probe` (0.730 [0.705, 0.767] on `reflex`; 0.721 [0.604, 0.853] on `reflex_resolve`) also observe partner posture (§11.11).

### §7.4 Trap Four: Single-Seed Shuffled Controls Give False Alarms in Both Directions (L14, §11.9)

A single-seed `:shuffled` CI resamples test groups conditional on one random training permutation (`default_rng(0)`), ignoring variance across training permutations (L14; Table `T6`).

<!-- table:T6 -->
| Dataset | Passing (`ci_low <= 0.5`) / Run | Failing `:shuffled` row(s) & 50-seed permutation null |
|---|---|---|
| `oops` | 4 / 4 | none |
| `holoassist` | 3 / 3 | none |
| `holoassist_audible` | 5 / 5 | none |
| `reflex` | 6 / 6 | none |
| `reflex_resolve` | 5 / 6 | `react-nonverbal:shuffled` 0.603 [0.507, 0.691] (`FAIL: above chance (leak)`); 50-seed null mean 0.510, SD 0.063, range [0.330, 0.653], seed 0 = 0.603, true 0.660 beats 50/50 |
<!-- /table:T6 -->

On `holoassist`, `step-prior:shuffled` at seed 0 scored 0.464 [0.440, 0.489] (below 0.5), whereas a 50-seed permutation null (`holoassist/step_prior_shuffled_null`) averaged 0.496 (SD 0.030, 20% of seeds below 0.47). On `reflex_resolve`, `react-nonverbal:shuffled` (140 test items across 17 groups) scored **0.603 [0.507, 0.691]** at seed 0 (`FAIL: above chance (leak)`), whereas across 50 training permutations (`seeds 0..49`) it averaged **0.5100** (SD **0.0626**, range **[0.3302, 0.6526]**; 0.510, 0.063, [0.330, 0.653]), with seed 0 at the 92% mark by chance and true `react-nonverbal` (0.660 [0.545, 0.766]) beating 50/50 seeds.

### §7.5 Trap Five: Naive Feature Concatenation Makes Fusion Lose to Its Own Strongest Input (L12)

Under uniform regularization (`LogisticRegression(C=1.0, class_weight="balanced")`), concatenating a scalar feature (`logit(step-prior)`) with high-dimensional visual or facial vectors dilutes the scalar (L12):
- On `holoassist`, `step-prior` is 0.751 [0.726, 0.774], but `action-fusion` drops to 0.681 [0.657, 0.706].
- On `holoassist_audible`, `step-prior` is 0.737 [0.712, 0.765], `action-fusion` drops to 0.650 [0.599, 0.694], and `fusion` (0.671 [0.618, 0.713]) trails `step-prior` (`action_best`) by **-0.065 [-0.113, -0.025]**.
- On `reflex_resolve`, `step-prior` is 0.938 [0.883, 0.980] (0.941 [0.886, 0.980] on 173 face-valid items), `[logit(step-prior), X_face]` loses **-0.121 [-0.193, -0.059]**, and `fusion` (0.749 [0.645, 0.877]) trails `step-prior` by **-0.192 [-0.315, -0.037]**.

---

## §8. Discussion and Limitations

**Why spontaneous reactions added so little here.** Three factors explain the near-zero gains across `oops`, `holoassist`, `holoassist_audible`, `reflex`, and `reflex_resolve`. First, on visible failures (`oops`, `reflex`), visual probes (0.779 [0.754, 0.805], 0.730 [0.705, 0.767]) and VLM judges (0.708 [0.684, 0.731], 0.720 [0.680, 0.761]) already detect errors from workspace video. Second, in structured tasks (`holoassist`, `holoassist_audible`, `reflex_resolve`), step identity strongly determines outcomes (`drop` failing 95.4%; `pick` resolving 100.0% vs. `place` 43.0%), so `step-prior` (0.737 [0.712, 0.765] to 0.941 [0.886, 0.980]) dominates. Third, when an instructor reacts to a hidden mistake (`holoassist_audible`), the signal is mostly binary speech intervention (`react-spoke` 0.665 [0.636, 0.691]) rather than vocal prosody (`react-nonverbal|spoke=1` 0.582 [0.548, 0.612]).

**Limitations.** We do not claim reactions carry no information in general or in live interaction (`06_writeup.md` §2):
- **No hidden-outcome facial verdict dataset was obtainable:** YouTube prohibits video downloads (`Issue 1`), `AM-FED+` `[CITATION NEEDED: AM-FED+ reference]` and taste-liking contacts were unreachable, and `BAD` (Bremers et al., IROS 2023; arXiv:2303.04835) requires an institutional IRB (`Issue 4`).
- **REFLEX sample size and camera view:** Only 17 test participants (374 `reflex` and 187 `reflex_resolve` items; 140 spoken test items), wide CIs ([-0.004, 0.045], [-0.002, 0.078]), a distant camera where `gemma4:26b` predicts 1.0 on all 374 items, and silhouette leakage through the privacy mask.
- **Frozen encoders on one machine:** Linear probes on frozen features on one Mac Studio, with no video fine-tuning or physical robot rollouts (`DW6`).

**What a decisive test needs.** Future evaluations (e.g., `DW8` on CANDOR `[CITATION NEEDED: CANDOR reference]`) require: (i) a genuinely hidden outcome with `step-prior` near 0.5; (ii) unmasked facial and vocal recordings; (iii) strictly equal-length reaction windows verified against a `window-length` control (§11.10); and (iv) an action video containing zero pixels or silhouettes of the reactor (§11.11).

---

## §9. Conclusion

Across five pre-registered human-reaction datasets—`oops` (2,710 items), `holoassist` (5,000), `holoassist_audible` (4,420), `reflex` (1,208), and `reflex_resolve` (604)—plus 5,517 robot trajectories in `roboreward`, spontaneous observer reactions added almost nothing beyond action-only visual probes, task-step priors, and zero-shot VLM judges. No dataset met our H1 pass bar (>= 0.65 reaction AUROC and >= +0.03 ablation gain) or supported offline H3 human-to-robot transfer (+0.0009 [-0.0006, 0.0024] at 100 robot labels). The sole positive ablation interval (`holoassist_audible`, +0.021 [0.006, 0.034]) mostly reflects whether the instructor spoke (`react-spoke` 0.665 [0.636, 0.691]), and the sole Wave B `PROCEED` clause (`reflex` `react-face`, 0.609 [0.569, 0.644]) was exposed in verification as a window-length artifact (0.735 [0.675, 0.796] from window length alone; 0.525 [0.473, 0.572] and 0.511 [0.457, 0.561] on equal-length windows). Together with exact diagnostics for five methodological traps, these findings establish a stricter baseline for evaluating reaction-based rewards.

---

## Appendix A. Reproduction

All facts (`docs/paper/facts.json`), tables (`docs/paper/tables/T1.md`–`T6.md`), figures (`docs/paper/figures/f1_ablation.png`, `f2_window_length.png`, `f3_wave_d.png`), and evaluation reports (`docs/evals/*.md`) cited in this paper can be rebuilt deterministically from `DATA_ROOT` (`/Volumes/Extreme SSD/social_robotics`) and `results/scorecard.jsonl` using the commands documented in `docs/paper/README.md`:

```bash
export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
export TMPDIR="/Volumes/Extreme SSD/tmp"
export SR_NO_MODEL_BANNER=1

# 1. Recompute Wave B post-hoc diagnostics (docs/evals/2026-11-04_reflex_posthoc.json)
PYTHONPATH=src ./venv/bin/python tools/diag_reflex_posthoc.py

# 2. Rebuild paper facts.json, tables T1..T6, and figures F1..F3
PYTHONPATH=src ./venv/bin/python tools/build_paper_facts.py

# 3. Run the mechanical number and table-parity checker on this draft
PYTHONPATH=src ./venv/bin/python tools/check_paper_numbers.py docs/paper/draft.md

# 4. Regenerate the five dated evaluation reports in docs/evals/
PYTHONPATH=src ./venv/bin/python tools/report_oops.py --date 2026-10-10 --updated-date 2026-10-21 --updated-a15-date 2026-10-21
PYTHONPATH=src ./venv/bin/python tools/report_holoassist.py --dataset holoassist --date 2026-10-15 --corrected-date 2026-10-20 --updated-date 2026-10-21 --updated-a15-date 2026-10-21
PYTHONPATH=src ./venv/bin/python tools/report_holoassist.py --dataset holoassist_audible --date 2026-10-21 --updated-a15-date 2026-10-21
PYTHONPATH=src ./venv/bin/python tools/report_wave_d.py --date 2026-10-22
PYTHONPATH=src ./venv/bin/python tools/report_reflex.py --date 2026-11-04
```

---

## Appendix B. Full Results

Below is the complete inventory of latest scorecard rows per `(dataset, condition)` from `results/scorecard.jsonl` (as stored in `docs/paper/facts.json` under `computed/scorecard`).

### Appendix B: Oops! (`dataset = oops`, Test Split)

```text
Condition                        Value      95% CI               N_items  N_groups Timestamp
---------------------------------------------------------------------------------------------------------
judge                            0.472      [0.440, 0.498]       1072     90       2026-10-21T04:12:36Z
judge-frontier                   0.704      [0.544, 0.821]       50       90       2026-10-21T04:12:36Z
action-probe                     0.779      [0.754, 0.805]       1072     90       2026-10-21T04:12:37Z
react-nonverbal                  0.711      [0.684, 0.740]       1072     90       2026-10-21T04:12:37Z
react-spoke                      null       n/a                  1072     90       2026-10-21T04:12:37Z
react-full                       null       n/a                  1072     90       2026-10-21T04:12:37Z
fusion                           0.793      [0.771, 0.816]       1072     90       2026-10-21T04:12:38Z
fusion_minus_action_best         0.015      [-0.002, 0.029]      1072     90       2026-10-21T04:12:40Z
action-probe:shuffled            0.510      [0.484, 0.534]       1072     90       2026-10-21T04:12:41Z
react-nonverbal:shuffled         0.468      [0.433, 0.501]       1072     90       2026-10-21T04:12:41Z
fusion:shuffled                  0.513      [0.488, 0.537]       1072     90       2026-10-21T04:12:42Z
judge-large                      0.708      [0.684, 0.731]       1072     90       2026-10-21T04:12:36Z
step-prior                       null       n/a                  1072     90       2026-10-21T04:12:37Z
action-fusion                    0.778      [0.754, 0.805]       1072     90       2026-10-21T04:12:38Z
fusion_minus_action_fusion       0.015      [-0.001, 0.029]      1072     90       2026-10-21T04:12:39Z
step-prior:shuffled              null       n/a                  1072     90       2026-10-21T04:12:41Z
action-fusion:shuffled           0.510      [0.484, 0.534]       1072     90       2026-10-21T04:12:42Z
```

### Appendix B: HoloAssist Full (`dataset = holoassist`, Test Split)

```text
Condition                        Value      95% CI               N_items  N_groups Timestamp
---------------------------------------------------------------------------------------------------------
judge                            0.488      [0.466, 0.512]       2000     91       2026-10-21T04:12:57Z
judge-large                      0.544      [0.528, 0.562]       2000     91       2026-10-21T04:12:58Z
judge-frontier                   null       n/a                  2000     91       2026-10-21T04:12:58Z
action-probe                     0.569      [0.542, 0.597]       2000     91       2026-10-21T04:12:58Z
react-nonverbal                  null       n/a                  2000     91       2026-10-21T04:12:58Z
react-nonverbal|spoke=1          null       n/a                  955      87       2026-10-21T04:12:58Z
react-spoke                      0.676      [0.655, 0.695]       2000     91       2026-10-21T04:12:59Z
react-full                       0.739      [0.716, 0.761]       2000     91       2026-10-21T04:12:59Z
fusion                           null       n/a                  2000     91       2026-10-21T04:13:00Z
fusion_minus_action_best         null       n/a                  2000     91       2026-10-21T04:13:01Z
action-probe:shuffled            0.492      [0.461, 0.525]       2000     91       2026-10-21T04:13:01Z
react-nonverbal:shuffled         null       n/a                  2000     91       2026-10-21T04:13:01Z
fusion:shuffled                  null       n/a                  2000     91       2026-10-21T04:13:02Z
step-prior                       0.751      [0.726, 0.774]       2000     91       2026-10-21T04:13:00Z
action-fusion                    0.681      [0.657, 0.706]       2000     91       2026-10-21T04:13:00Z
fusion_minus_action_fusion       null       n/a                  2000     91       2026-10-21T04:13:00Z
step-prior:shuffled              0.464      [0.440, 0.489]       2000     91       2026-10-21T04:13:02Z
action-fusion:shuffled           0.493      [0.461, 0.526]       2000     91       2026-10-21T04:13:02Z
```

### Appendix B: HoloAssist Audible (`dataset = holoassist_audible`, Test Split)

```text
Condition                        Value      95% CI               N_items  N_groups Timestamp
---------------------------------------------------------------------------------------------------------
judge                            0.528      [0.499, 0.560]       1420     53       2026-10-21T16:50:20Z
judge-large                      0.516      [0.499, 0.533]       1420     53       2026-10-21T16:50:20Z
judge-frontier                   null       n/a                  1420     53       2026-10-21T16:50:21Z
action-probe                     0.576      [0.526, 0.623]       1420     53       2026-10-21T16:50:21Z
react-nonverbal                  0.654      [0.626, 0.675]       1420     53       2026-10-21T16:50:22Z
react-nonverbal|spoke=1          0.582      [0.548, 0.612]       837      52       2026-10-21T16:50:22Z
react-spoke                      0.665      [0.636, 0.691]       1420     53       2026-10-21T16:50:23Z
react-full                       0.759      [0.728, 0.791]       1420     53       2026-10-21T16:50:23Z
step-prior                       0.737      [0.712, 0.765]       1420     53       2026-10-21T16:50:24Z
fusion                           0.671      [0.618, 0.713]       1420     53       2026-10-21T16:50:25Z
action-fusion                    0.650      [0.599, 0.694]       1420     53       2026-10-21T16:50:25Z
fusion_minus_action_fusion       0.021      [0.006, 0.034]       1420     53       2026-10-21T16:50:26Z
fusion_minus_action_best         -0.065     [-0.113, -0.025]     1420     53       2026-10-21T16:50:27Z
action-probe:shuffled            0.490      [0.455, 0.529]       1420     53       2026-10-21T16:50:27Z
react-nonverbal:shuffled         0.496      [0.465, 0.526]       1420     53       2026-10-21T16:50:28Z
step-prior:shuffled              0.483      [0.449, 0.510]       1420     53       2026-10-21T16:50:29Z
action-fusion:shuffled           0.491      [0.455, 0.530]       1420     53       2026-10-21T16:50:29Z
fusion:shuffled                  0.495      [0.465, 0.528]       1420     53       2026-10-21T16:50:30Z
```

### Appendix B: REFLEX Immediate Failure (`dataset = reflex`, Test Split)

```text
Condition                        Value      95% CI               N_items  N_groups Timestamp
---------------------------------------------------------------------------------------------------------
judge                            0.720      [0.680, 0.761]       374      17       2026-11-05T05:49:35Z
judge-large                      0.500      [0.500, 0.500]       374      17       2026-11-05T05:49:36Z
judge-frontier                   null       n/a                  374      17       2026-11-05T05:49:36Z
action-probe                     0.730      [0.705, 0.767]       374      17       2026-11-05T05:49:36Z
react-face                       0.609      [0.569, 0.644]       345      17       2026-11-05T05:49:37Z
react-nonverbal                  0.526      [0.493, 0.568]       109      17       2026-11-05T05:49:37Z
react-nonverbal|spoke=1          0.526      [0.493, 0.568]       109      15       2026-11-05T05:49:37Z
react-spoke                      0.519      [0.503, 0.537]       374      17       2026-11-05T05:49:38Z
react-full                       null       n/a                  374      17       2026-11-05T05:49:38Z
step-prior                       0.500      [0.500, 0.500]       374      17       2026-11-05T05:49:38Z
fusion                           0.817      [0.772, 0.857]       345      17       2026-11-05T05:49:39Z
action-fusion                    0.794      [0.744, 0.841]       345      17       2026-11-05T05:49:39Z
fusion_minus_action_fusion       0.023      [-0.004, 0.045]      345      17       2026-11-05T05:49:39Z
fusion_minus_action_best         0.023      [-0.004, 0.045]      345      17       2026-11-05T05:49:40Z
action-probe:shuffled            0.491      [0.468, 0.511]       374      17       2026-11-05T05:49:40Z
react-face:shuffled              0.542      [0.494, 0.599]       345      17       2026-11-05T05:49:41Z
react-nonverbal:shuffled         0.483      [0.461, 0.528]       109      17       2026-11-05T05:49:41Z
step-prior:shuffled              0.500      [0.500, 0.500]       374      17       2026-11-05T05:49:41Z
action-fusion:shuffled           0.500      [0.477, 0.520]       345      17       2026-11-05T05:49:42Z
fusion:shuffled                  0.527      [0.489, 0.568]       345      17       2026-11-05T05:49:42Z
```

### Appendix B: REFLEX Post-Explanation Resolution (`dataset = reflex_resolve`, Test Split)

```text
Condition                        Value      95% CI               N_items  N_groups Timestamp
---------------------------------------------------------------------------------------------------------
judge                            0.473      [0.422, 0.523]       187      17       2026-11-05T05:49:45Z
judge-large                      0.474      [0.409, 0.548]       187      17       2026-11-05T05:49:45Z
judge-frontier                   null       n/a                  187      17       2026-11-05T05:49:45Z
action-probe                     0.721      [0.604, 0.853]       187      17       2026-11-05T05:49:45Z
react-face                       0.552      [0.440, 0.663]       173      17       2026-11-05T05:49:46Z
react-nonverbal                  0.660      [0.545, 0.766]       140      17       2026-11-05T05:49:46Z
react-nonverbal|spoke=1          0.660      [0.545, 0.766]       140      17       2026-11-05T05:49:47Z
react-spoke                      0.505      [0.439, 0.566]       187      17       2026-11-05T05:49:47Z
react-full                       null       n/a                  187      17       2026-11-05T05:49:47Z
step-prior                       0.938      [0.883, 0.980]       187      17       2026-11-05T05:49:47Z
fusion                           0.749      [0.645, 0.877]       173      17       2026-11-05T05:49:48Z
action-fusion                    0.713      [0.595, 0.859]       173      17       2026-11-05T05:49:48Z
fusion_minus_action_fusion       0.036      [-0.002, 0.078]      173      17       2026-11-05T05:49:49Z
fusion_minus_action_best         -0.192     [-0.315, -0.037]     173      17       2026-11-05T05:49:49Z
action-probe:shuffled            0.432      [0.329, 0.541]       187      17       2026-11-05T05:49:50Z
react-face:shuffled              0.578      [0.459, 0.679]       173      17       2026-11-05T05:49:50Z
react-nonverbal:shuffled         0.603      [0.507, 0.691]       140      17       2026-11-05T05:49:50Z
step-prior:shuffled              0.440      [0.278, 0.593]       187      17       2026-11-05T05:49:51Z
action-fusion:shuffled           0.476      [0.387, 0.574]       173      17       2026-11-05T05:49:51Z
fusion:shuffled                  0.432      [0.302, 0.566]       173      17       2026-11-05T05:49:51Z
```

### Appendix B: Wave D HoloAssist Pseudo-Label Pool (`dataset = holoassist_pool`)

```text
Condition                        Value      95% CI               N_items  N_groups Timestamp
---------------------------------------------------------------------------------------------------------
pseudo-label-quality             null       n/a                  5682     53       2026-10-22T08:37:06Z
pseudo-label-quality:all         0.621      [0.568, 0.674]       20000    185      2026-10-22T08:37:06Z
```

### Appendix B: Wave D RoboReward Transfer Curve (`dataset = roboreward`, Test Split)

```text
Condition                        Value      95% CI               N_items  N_groups Timestamp
---------------------------------------------------------------------------------------------------------
detector:none:N=50               0.562      [0.547, 0.577]       1517     888      2026-10-22T14:22:28Z
detector:react:N=50              0.563      [0.548, 0.577]       1517     888      2026-10-22T14:22:28Z
detector:shuffled:N=50           0.562      [0.547, 0.578]       1517     888      2026-10-22T14:22:28Z
detector:truth:N=50              0.512      [0.484, 0.540]       1517     888      2026-10-22T14:22:28Z
gain:react:N=50                  0.001      [-0.001, 0.003]      1517     888      2026-10-22T14:22:28Z
gain:shuffled:N=50               0.000      [-0.002, 0.002]      1517     888      2026-10-22T14:22:29Z
gain:truth:N=50                  -0.049     [-0.077, -0.022]     1517     888      2026-10-22T14:22:29Z
react_minus_shuffled:N=50        0.001      [-0.002, 0.004]      1517     888      2026-10-22T14:22:29Z
detector:none:N=100              0.579      [0.559, 0.598]       1517     888      2026-10-22T14:23:01Z
detector:react:N=100             0.580      [0.559, 0.598]       1517     888      2026-10-22T14:23:01Z
detector:shuffled:N=100          0.579      [0.558, 0.598]       1517     888      2026-10-22T14:23:01Z
detector:truth:N=100             0.528      [0.500, 0.555]       1517     888      2026-10-22T14:23:01Z
gain:react:N=100                 0.001      [-0.001, 0.002]      1517     888      2026-10-22T14:23:01Z
gain:shuffled:N=100              0.000      [-0.001, 0.002]      1517     888      2026-10-22T14:23:01Z
gain:truth:N=100                 -0.051     [-0.078, -0.024]     1517     888      2026-10-22T14:23:01Z
react_minus_shuffled:N=100       0.001      [-0.001, 0.003]      1517     888      2026-10-22T14:23:01Z
detector:none:N=200              0.596      [0.575, 0.617]       1517     888      2026-10-22T14:23:33Z
detector:react:N=200             0.597      [0.575, 0.617]       1517     888      2026-10-22T14:23:33Z
detector:shuffled:N=200          0.596      [0.574, 0.617]       1517     888      2026-10-22T14:23:33Z
detector:truth:N=200             0.554      [0.528, 0.580]       1517     888      2026-10-22T14:23:33Z
gain:react:N=200                 0.000      [-0.001, 0.002]      1517     888      2026-10-22T14:23:33Z
gain:shuffled:N=200              -0.000     [-0.001, 0.001]      1517     888      2026-10-22T14:23:33Z
gain:truth:N=200                 -0.042     [-0.068, -0.018]     1517     888      2026-10-22T14:23:34Z
react_minus_shuffled:N=200       0.000      [-0.001, 0.002]      1517     888      2026-10-22T14:23:34Z
detector:none:N=500              0.611      [0.588, 0.630]       1517     888      2026-10-22T14:24:07Z
detector:react:N=500             0.611      [0.588, 0.631]       1517     888      2026-10-22T14:24:07Z
detector:shuffled:N=500          0.611      [0.588, 0.631]       1517     888      2026-10-22T14:24:07Z
detector:truth:N=500             0.584      [0.560, 0.607]       1517     888      2026-10-22T14:24:07Z
gain:react:N=500                 0.000      [-0.001, 0.001]      1517     888      2026-10-22T14:24:07Z
gain:shuffled:N=500              0.000      [-0.000, 0.001]      1517     888      2026-10-22T14:24:07Z
gain:truth:N=500                 -0.026     [-0.047, -0.007]     1517     888      2026-10-22T14:24:07Z
react_minus_shuffled:N=500       -0.000     [-0.002, 0.001]      1517     888      2026-10-22T14:24:07Z
detector:none:N=1000             0.625      [0.602, 0.647]       1517     888      2026-10-22T14:24:44Z
detector:react:N=1000            0.625      [0.602, 0.647]       1517     888      2026-10-22T14:24:44Z
detector:shuffled:N=1000         0.625      [0.603, 0.647]       1517     888      2026-10-22T14:24:45Z
detector:truth:N=1000            0.618      [0.593, 0.642]       1517     888      2026-10-22T14:24:45Z
gain:react:N=1000                -0.000     [-0.001, 0.000]      1517     888      2026-10-22T14:24:45Z
gain:shuffled:N=1000             0.000      [-0.000, 0.001]      1517     888      2026-10-22T14:24:45Z
gain:truth:N=1000                -0.006     [-0.022, 0.008]      1517     888      2026-10-22T14:24:45Z
react_minus_shuffled:N=1000      -0.000     [-0.001, 0.000]      1517     888      2026-10-22T14:24:45Z
detector:none:N=1000:shuffled    0.498      [0.491, 0.505]       1517     888      2026-10-22T14:24:53Z
robot-full                       0.657      [0.624, 0.690]       1517     888      2026-10-22T14:24:54Z
judge-large                      0.747      [0.720, 0.774]       1517     888      2026-10-22T14:24:54Z
roboreward-8b                    null       n/a                  1517     888      2026-10-22T14:24:54Z
```

---

## Appendix C. Recorded Facts

Per `06_writeup.md` §4.2, six diagnostic quantities cited in this paper were measured by the designer during verification passes and recorded in project design documents rather than recomputed by a standalone script in `tools/`. Each entry below lists its key in `docs/paper/facts.json` (`recorded`), its values, its source document and section, and the verbatim quote verified by `tools/build_paper_facts.py` at build time:

- **`wave_a/audible_strata`**
  - **Values:** +0.012 [-0.004, 0.030] where the instructor spoke; -0.026 where silent; +0.025 without `drop`.
  - **Source:** `docs/ongoing_general_errors.md` §2 Resolved entry (`Issue 8`).
  - **Verbatim quote:** *"The designer's exploratory strata showed the gain is mostly whether the instructor spoke: +0.012 [−0.004, 0.030] where they spoke, −0.026 where they were silent, and +0.025 without `drop`."*
- **`wave_d/head_zero_shot`**
  - **Values:** `react` 0.508 [0.475, 0.540]; `shuffled` 0.521 [0.487, 0.554]; `truth` 0.491 [0.457, 0.522]; cosine similarity 0.0.
  - **Source:** `docs/04_wave_d_robot_check.md` §8.
  - **Verbatim quote:** *"**Each human head on its own, scored on the robot test set:** `react` 0.508 [0.475, 0.540], `shuffled` 0.521 [0.487, 0.554], `truth` 0.491 [0.457, 0.522]."*
- **`wave_d/well_posed_head`**
  - **Values:** `audible_train` 0.550 [0.515, 0.585]; `full_train` 0.472 [0.437, 0.509].
  - **Source:** `docs/04_wave_d_robot_check.md` §8.
  - **Verbatim quote:** *"**Well-posed true-label heads:** one fit on `holoassist_audible` train (1,500 + 1,500) scores 0.550 [0.515, 0.585] on robot test; one fit on full `holoassist` train scores 0.472 [0.437, 0.509]."*
- **`wave_d/head_norms`**
  - **Values:** `react` 0.99; `shuffled` 1.03; `truth` 26.9; prediction shift 0.015 vs. 0.33; in-sample AUROC 0.997.
  - **Source:** `docs/04_wave_d_robot_check.md` §8.
  - **Verbatim quote:** *"The norms are `react` 0.99, `shuffled` 1.03 and `truth` 26.9."*
- **`wave_d/pool_mistakes`**
  - **Values:** 104 mistakes in 20,000 items (0.52%); 92% of audible mistakes taken by `holoassist_audible`.
  - **Source:** `docs/04_wave_d_robot_check.md` §8.
  - **Verbatim quote:** *"Wave A's balanced `holoassist_audible` sample had taken about 92% of the audible sessions' mistakes, so the pool has 104 mistakes in 20,000 (0.52%), and none in the sessions the labeler never saw."*
- **`holoassist/step_prior_shuffled_null`**
  - **Values:** 50-seed mean 0.496, SD 0.030, with 20% of seeds below 0.47.
  - **Source:** `docs/03_eval_harness.md` §11.9.
  - **Verbatim quote:** *"Over 50 permutation seeds the same control averages 0.496 (SD 0.030), and 20% of seeds fall below 0.47 (designer, October 21)."*

---

## Appendix D. Data and Ethics Statement

No new human-subjects data was collected, recorded, or rated during this project (`00_thesis.md` `Decisions 2, 12, and 13`). All experiments were conducted on publicly released research datasets obtained from their official repositories under their stated licenses (`02_data_sources.md`):
- **Oops!** (Epstein, Chen, Vondrick, CVPR 2020): downloaded from the official Columbia University dataset site (`oops.cs.columbia.edu`).
- **HoloAssist** (Wang et al., ICCV 2023; arXiv:2309.17024): downloaded from the official project site (`holoassist.github.io`) under the Community Data License Agreement (`CDLAv2`). Performer faces are blurred and audio tracks are pitch-shifted by the dataset creators for participant privacy.
- **REFLEX** (Naoum et al., HRI 2025; arXiv:2502.14185): downloaded from Zenodo (`10.5281/zenodo.14590226`) under the Creative Commons Attribution license (`CC BY 4.0`). Scene videos (`video_cam1.mp4`) are anonymized with a black participant mask by the dataset creators, and no participant images are reproduced in this paper.
- **RoboReward** (2026; arXiv:2601.00675): downloaded from Hugging Face (`datasets/teetone/RoboReward`) under the `CC BY 4.0` license; contains robot manipulation trajectories only.
- **Retired prototype data:** All raw Ego4D `[CITATION NEEDED: Ego4D reference]` and Charades-Ego `[CITATION NEEDED: Charades-Ego reference]` video files from the retired `v0-saf-final` prototype were permanently deleted from disk on October 8, 2026 (`Issue 3`, `DELETED_2026-10-08.json`).
- **Respect for access controls:** We never scraped or bypassed download restrictions (`Issue 1` and `Issue 4`). Controlled-access datasets requiring institutional IRB review (`BAD`) were not accessed, and no face or voice embeddings of identifiable individuals are distributed.

---

## References

- **EMPATHIC:** Cui, Zhang, Allievi, Stone, Niekum, Knox. *The EMPATHIC Framework for Task Learning from Implicit Human Feedback*. CoRL 2020; arXiv:2009.13649.
- **BAD dataset:** Bremers et al. *The Bystander Affect Detection (BAD) Dataset for Failure Detection in HRI*. IROS 2023; arXiv:2303.04835. `[CITATION NEEDED: full reference]`
- **ERR@HRI 2024:** *ERR@HRI 2024 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Interactions*. 2024; arXiv:2407.06094. `[CITATION NEEDED: full reference]`
- **`ERR@HRI 3.0`:** *`ERR@HRI 3.0` Challenge*. ICMI 2026; arXiv:2607.11570. `[CITATION NEEDED: full reference]`
- **Facial Feedback for RL (TAMER case study):** Li et al. *Facial Feedback for Reinforcement Learning: A Case Study and Offline Analysis Using the TAMER Framework*. 2020; arXiv:2001.08703. `[CITATION NEEDED: full reference]`
- **HoloAssist:** Wang et al. *HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World*. ICCV 2023; arXiv:2309.17024. `[CITATION NEEDED: full reference]`
- **Oops!:** Epstein, Chen, Vondrick. *Oops! Predicting Unintentional Action in Video*. CVPR 2020. `[CITATION NEEDED: full reference]`
- **REACT:** *REACT: Two Datasets for Studying Human Reactions and Evaluative Feedback to Robots Over Time*. 2024; arXiv:2402.00190. `[CITATION NEEDED: full reference]`
- **REFLEX:** Naoum et al. *REFLEX: A Dataset of Human Reactions to Robot Failures and Explanations*. HRI 2025; arXiv:2502.14185. `[CITATION NEEDED: full reference]`
- **RoboReward:** *RoboReward*. 2026; arXiv:2601.00675. `[CITATION NEEDED: full reference]`
- **TOPReward:** *TOPReward: Token Probabilities as Zero-Shot Rewards for Robotics*. 2026; arXiv:2602.19313. `[CITATION NEEDED: full reference]`
- **DVD:** *Learning Generalizable Robotic Reward Functions from "In-The-Wild" Human Videos*. RSS 2021; arXiv:2103.16817. `[CITATION NEEDED: full reference]`
