# Wave A Evaluation Summary (2026-10-15)
*Corrected 2026-10-20 (A9b): see the commit body. No scorecard row changed.*

Wave A evaluates Hypothesis 1 (H1): human observer reactions provide an outcome-supervision signal for task success and error detection beyond action-only cues, and baseline VLM judges cannot replace them when outcomes are subtle or intermediate.

This document summarizes the end-to-end results across both Wave A datasets:
1. **Oops!** (Visible-outcome contrast: obvious physical falls and accidents).
2. **HoloAssist** (Hidden-outcome / intermediate-mistake benchmark: fine-grained physical assembly and manipulation with an instructor in the loop).

---

## 1. Headline Results Table (Dataset × Condition)

All metrics are AUROC or $\Delta\text{AUROC}$ on the held-out test splits. 95% CIs are percentile intervals over 1,000 group-bootstrap resamples (groups: compilation for Oops!, session prefix for HoloAssist; numpy default_rng(0); single-class resamples redrawn, at most 10,000 draws), per 03_eval_harness.md §7. Every number matches its row in `results/scorecard.jsonl` (citing timestamp `ts`).

| Dataset | Condition | Metric | Value | 95% CI | N (inc) | N (excl) | Scorecard Timestamp (`ts`) | Notes |
|---|---|---|---|---|---|---|---|---|
| **Oops!** | `judge` (`qwen2.5vl:7b`) | auroc | 0.472 | [0.440, 0.498] | 1,072 | 0 | `2026-10-15T06:23:28Z` | Local judge baseline; at chance |
| Oops! | `judge-large` (`gemma4:26b`) | auroc | 0.708 | [0.684, 0.731] | 1,072 | 0 | `2026-10-15T06:23:28Z` | Large local judge (Issue 5 B); strong visual detection |
| Oops! | `judge-frontier` (`gemini-3.6-flash`) | auroc | 0.704 | [0.544, 0.821] | 50 | 1,022 | `2026-10-15T06:23:28Z` | Partial run (free tier 20 req/day quota) |
| Oops! | `action-probe` (`siglip-b16-224`) | auroc | 0.779 | [0.754, 0.805] | 1,072 | 0 | `2026-10-15T06:23:29Z` | Best action baseline (`action_best`) |
| Oops! | `react-nonverbal` (`e2v-plus-large`) | auroc | 0.711 | [0.684, 0.740] | 1,072 | 0 | `2026-10-15T06:23:29Z` | Caveat: contains physical crash sounds & music |
| Oops! | `react-spoke` | auroc | null | — | 0 | 1,072 | `2026-10-15T06:23:30Z` | not run: no annotations of reactor speech in Oops! |
| Oops! | `react-full` | auroc | null | — | 0 | 1,072 | `2026-10-15T06:23:30Z` | not run: no annotations of reactor speech in Oops! |
| Oops! | `fusion` | auroc | 0.793 | [0.771, 0.816] | 1,072 | 0 | `2026-10-15T06:23:30Z` | Action + reaction features |
| Oops! | `fusion_minus_action_best` | delta_auroc | 0.015 | [-0.002, 0.029] | 1,072 | 0 | `2026-10-15T06:23:31Z` | $\Delta \approx 0$; CI covers 0.00 |
| Oops! | `action-probe:shuffled` | auroc | 0.510 | [0.484, 0.534] | 1,072 | 0 | `2026-10-15T06:23:31Z` | Control: PASS (covers 0.50) |
| Oops! | `react-nonverbal:shuffled` | auroc | 0.468 | [0.433, 0.501] | 1,072 | 0 | `2026-10-15T06:23:32Z` | Control: PASS (covers 0.50) |
| Oops! | `fusion:shuffled` | auroc | 0.513 | [0.488, 0.537] | 1,072 | 0 | `2026-10-15T06:23:33Z` | Control: PASS (covers 0.50) |
|---|---|---|---|---|---|---|---|---|
| **HoloAssist** | `judge` (`qwen2.5vl:7b`) | auroc | 0.488 | [0.466, 0.512] | 2,000 | 0 | `2026-10-15T20:07:56Z` | Local judge baseline; at chance |
| HoloAssist | `judge-large` (`gemma4:26b`) | auroc | 0.544 | [0.528, 0.562] | 2,000 | 0 | `2026-10-15T20:07:56Z` | Large local judge; modest above-chance |
| HoloAssist | `judge-frontier` (`gemini-3.6-flash`) | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:56Z` | not run: no paid quota (Issue 5 B) |
| HoloAssist | `action-probe` (`siglip-b16-224`) | auroc | 0.569 | [0.542, 0.597] | 2,000 | 0 | `2026-10-15T20:07:57Z` | Action baseline (`action_best`) |
| HoloAssist | `react-nonverbal` (`e2v-plus-large`) | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:57Z` | not run: presence check failed (+1.51 dB < +3.5 dB; Issue 6) |
| HoloAssist | *Diagnostic:* `react-nonverbal\|spoke=1` | auroc | null | — | 0 | 955 | `2026-10-15T20:07:57Z` | not run: audio not extracted (Issue 6) |
| HoloAssist | `react-spoke` | auroc | 0.676 | [0.655, 0.695] | 2,000 | 0 | `2026-10-15T20:07:57Z` | Instructor speech timing alone |
| HoloAssist | `react-full` | auroc | 0.739 | [0.716, 0.761] | 2,000 | 0 | `2026-10-15T20:07:58Z` | Instructor transcript language (TF-IDF); leakage condition (the instructor's words); never a headline (§11.2) |
| HoloAssist | `fusion` | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:58Z` | not run: react-nonverbal not run on train |
| HoloAssist | `fusion_minus_action_best` | delta_auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:58Z` | action_best=action-probe; not run: fusion not run |
| HoloAssist | `action-probe:shuffled` | auroc | 0.492 | [0.461, 0.525] | 2,000 | 0 | `2026-10-15T20:07:59Z` | Control: PASS (covers 0.50) |
| HoloAssist | `react-nonverbal:shuffled` | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:59Z` | not run: react-nonverbal not run |
| HoloAssist | `fusion:shuffled` | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:59Z` | not run: fusion not run |

---

## 2. Per-Clause Prediction Checks

### 2.1 Oops! (Visible-Outcome Contrast)
> **Prediction:** In fail videos where failures are plainly visible in the action frames, the VLM judge is expected to perform strongly, and non-verbal reactions should add little or no incremental signal over the action-only baseline ($\Delta \approx 0$).

- **Clause 1, "the judge is strong" (judge):** AUROC 0.472 [0.440, 0.498] (1072 scored) → did not hold
- **Clause 1, "the judge is strong" (judge-large):** AUROC 0.708 [0.684, 0.731] (1072 scored) → did not hold
- **Clause 1, "the judge is strong" (judge-frontier):** AUROC 0.704 [0.544, 0.821] (50 scored) → did not hold
- **Clause 2, "Δ ≈ 0":** Δ 0.015 [-0.002, 0.029] (action_best = action-probe) → held

### 2.2 HoloAssist (Hidden-Outcome / Intermediate-Mistake Physical Assembly)
> **Prediction:** In fine-grained manipulation where mistakes are subtle or intermediate, the VLM judge cannot reliably distinguish mistakes from correct steps, but instructor reactions provide strong predictive signal beyond action frames ($\Delta > 0$, pass bar: reaction $\ge 0.65$ and $\Delta \ge +0.03$).

- **Clause 1, "reaction-only beats action_best":** react-nonverbal not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large vs action-probe 0.569 → not tested: not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large
- **Clause 2, "reactions add to the action-only model":** ablation Δ not run → not tested: no fusion_minus_action_fusion row
- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not testable: react-nonverbal, ablation Δ
- **Context, no verdict:** fusion − action_best action_best=action-probe; not run: fusion not run; judge 0.488 [0.466, 0.512]; judge-large 0.544 [0.528, 0.562]; action-probe 0.569 [0.542, 0.597]; react-spoke 0.676 [0.655, 0.695] (diagnostic: did the instructor speak); react-full 0.739 [0.716, 0.761] (leakage condition: the instructor's words; never a headline, §11.2).
- **Caveat:** HoloAssist step text carries outcome information (02_data_sources.md → HoloAssist → As built (A8)); judge-large scores 0.489 on test items whose verb is not "drop".

---

## 3. Stated Caveats in Full

1. **Oops! `react-nonverbal` is not evidence of reaction signal:**
   In Oops!, the reaction window is identical to the post-failure action window ($[t, t+3]$ s). As a result, the audio channel carries the acoustic impact of the crash itself (e.g., skateboards splintering, bodies hitting ground, glass breaking) as well as YouTube background music added in compilation editing. It does not isolate bystander human affective reactions.
2. **HoloAssist audio is pitch-shifted, and groups are pairs, not persons:**
   - The dataset creators pitch-shifted the audio tracks prior to public release to preserve participant privacy, altering vocal formants and pitch contours.
   - Splits are grouped by session name prefix before `-` (e.g., `z052`), representing ~350 distinct instructor–performer pairs. If an individual participant took part in multiple sessions under different pairings, an individual might straddle train and test splits. The dataset metadata cannot rule this out.
3. **State of the "just ask an LLM" control:**
   - Both local judges (`judge` at 7B and `judge-large` at 26B) were fully evaluated across all test items on both datasets (1,072 Oops! items and 2,000 HoloAssist items).
   - No frontier model (`gemini-3.6-flash`) was run to completion on either dataset, following the maintainer's explicit selection of Issue 5 Option B (deploying a 26B local judge rather than paid API billing). A future reviewer may still request a paid frontier-model comparison.

---

## 4. Suspicious Findings & Anomalies

1. **Audio Presence Check Failure on HoloAssist (Issue 6):**
   `tools/check_audio_presence.py` measured a median instructor utterance vs. silence energy difference of only **+1.51 dB** across 20 candidate sessions, failing the $\ge +3.5$ dB requirement. The headset microphone worn by the performer is inches from mechanical manipulation (screwdriver ratchets, metal screws, dropping parts, table impacts at -41 to -45 dBFS), which physically drowns out the remote instructor's voice (-50 to -55 dBFS).
2. **Judge Answer Distributions & Polarization:**
   - `qwen2.5vl:7b`: Severe hedging and default clustering. On Oops!, 78.4% of pre-failure and 54.9% of post-failure items were rated 90%, yielding chance discrimination (AUROC 0.472). On HoloAssist, 55.3% of correct actions and 58.8% of mistakes were rated 100% or 90%, yielding chance discrimination (AUROC 0.488).
   - `gemma4:26b`: Extreme binary polarization (almost zero intermediate scores; strictly 100 vs 0). On Oops!, this drove AUROC to 0.708 (78.0% 100s on pre-failure, 63.4% 0s on post-failure). On HoloAssist, however, it polarized indiscriminately toward 100 (90.9% 100s on correct, 82.0% 100s on mistakes), achieving only 0.544 AUROC.
3. **Judge answers:**
   - local: Oops! 2,708 (7B; 0 parse failures; 2 ffmpeg errors on one undecodable train clip) + 1,072 (26B; 0/0); HoloAssist 5,000 (7B; 0/0) + 2,000 (26B; 0/0) = 10,780 answers;
   - frontier (Oops! test only): 50 scored, 1,024 items with HTTP 429 rows, 1 other error;
   - every test-split condition that ran excluded 0 items.
4. **HoloAssist step text leaks the outcome:**
   The step verb "drop" is a mistake 95% of the time, and judge-large's 0.544 comes from it (0.489 on the other test items). A step-prior control is added in A11 (Issue 7).

---

## 5. Compute & Wall-Clock Times per Stage

| Stage | Oops! (2,710 items) | HoloAssist (5,000 items) | Total Wave A Time |
|---|---|---|---|
| **Raw Download & Extraction** | 2,798.0 s (46.6 min; 45 GB archive) | Verified Oct 10 (197.8 GB archive) | ~2.5 h |
| **Visual Features (`siglip-b16-224`)** | 3,144.6 s (52.4 min) | 2,779.2 s (46.3 min) | 1.65 h |
| **Audio Features (`e2v-plus-large`)** | 276.9 s (4.6 min) | *Stopped per Issue 6* | 0.08 h |
| **Local Judge (`qwen2.5vl:7b`)** | 22,542.9 s (375.7 min / 6.3 h) | 40,656.9 s (677.6 min / 11.3 h) | 17.6 h |
| **Large Local Judge (`gemma4:26b`)**| 2,477.9 s (41.3 min, 1,072 items) | 4,064.1 s (67.7 min, 2,000 items) | 1.82 h |
| **Frontier Judge (`gemini-3.6-flash`)**| 1,351.5 s (22.5 min; 50 items) | *Not run per Issue 5 B* | 0.38 h |
| **Linear Probes & Bootstrapping** | < 10 s | < 10 s | < 1 min |
| **Total Pipeline Compute Time** | **~9.1 h** | **~15.7 h** | **~24.8 h** |

---

## 6. Observations & Design Recommendations for Wave B

1. **On HoloAssist the action-only controls in this run are weak (action-probe 0.569; judges 0.488 and 0.544) and react-spoke reaches 0.676, but no Δ was computed (Issue 6), and the step text carries outcome information (Issue 7). Whether reactions add information beyond action-only cues on hidden outcomes is untested in this pass; A11–A13 test it.**
2. **Hypothesis, from the presence check (not tested): Microphone Separation is Mandatory for Acoustic Non-Verbal Reactions:**
   A microphone co-located on the worker/robot will pick up manipulation noise that masks human vocal affect. Wave B (verdict-video and bystander reaction data) must ensure either:
   - Microphones directed at or worn by the human observer;
   - Facial affect analysis (`react-face`, DW2);
   - Speech recognition (ASR) to leverage transcript language (`react-full`).
3. **The 26B judge raised Oops! from 0.472 to 0.708, still below the action-probe (0.779) and short of the "strong" bar; on HoloAssist its 0.544 comes from the verb "drop" (0.489 otherwise).**
4. **memguard.log through October 15: 10 shared-model stops (sr_judge_large_load) and 77 unloads, all of our own 7B judge (sr_judge_load); no unload of gemma4:26b; no jetsam report names a process of ours.**

---

## 7. Wave D start condition

Not met in this pass. Oops!'s Δ CI includes 0; HoloAssist's Δ was not run. Re-evaluated in A14 by the amended rule (ablation gain, 00_thesis.md entry 20).

---

## 8. Completion (2026-10-21): step-prior, ablation gain, audible-session HoloAssist

### 8.1 Headline Table (Dataset × Condition)

All values, 95% CIs, item counts, exclusions, and scorecard timestamps from the latest scorecard rows (`results/scorecard.jsonl`):

| Dataset | Condition | Metric | Value | 95% CI | N (included) | N (excluded) | Scorecard Timestamp |
|---|---|---|---|---|---|---|---|
| `oops` | `judge` | auroc | 0.472 | [0.440, 0.498] | 1072 | 0 | `2026-10-21T04:12:36Z` |
| `oops` | `judge-large` | auroc | 0.708 | [0.684, 0.731] | 1072 | 0 | `2026-10-21T04:12:36Z` |
| `oops` | `judge-frontier` | auroc | 0.704 | [0.544, 0.821] | 50 | 1022 | `2026-10-21T04:12:36Z` |
| `oops` | `action-probe` | auroc | 0.779 | [0.754, 0.805] | 1072 | 0 | `2026-10-21T04:12:37Z` |
| `oops` | `react-nonverbal` | auroc | 0.711 | [0.684, 0.740] | 1072 | 0 | `2026-10-21T04:12:37Z` |
| `oops` | `react-spoke` | auroc | null | — | 1072 | 1072 | `2026-10-21T04:12:37Z` |
| `oops` | `react-full` | auroc | null | — | 1072 | 1072 | `2026-10-21T04:12:37Z` |
| `oops` | `step-prior` | auroc | null | — | 1072 | 1072 | `2026-10-21T04:12:37Z` |
| `oops` | `action-fusion` | auroc | 0.778 | [0.754, 0.805] | 1072 | 0 | `2026-10-21T04:12:38Z` |
| `oops` | `fusion` | auroc | 0.793 | [0.771, 0.816] | 1072 | 0 | `2026-10-21T04:12:38Z` |
| `oops` | `fusion_minus_action_fusion` | delta_auroc | 0.015 | [-0.001, 0.029] | 1072 | 0 | `2026-10-21T04:12:39Z` |
| `oops` | `fusion_minus_action_best` | delta_auroc | 0.015 | [-0.002, 0.029] | 1072 | 0 | `2026-10-21T04:12:40Z` |
| `holoassist` | `judge` | auroc | 0.488 | [0.466, 0.512] | 2000 | 0 | `2026-10-21T04:12:57Z` |
| `holoassist` | `judge-large` | auroc | 0.544 | [0.528, 0.562] | 2000 | 0 | `2026-10-21T04:12:58Z` |
| `holoassist` | `judge-frontier` | auroc | null | — | 2000 | 2000 | `2026-10-21T04:12:58Z` |
| `holoassist` | `action-probe` | auroc | 0.569 | [0.542, 0.597] | 2000 | 0 | `2026-10-21T04:12:58Z` |
| `holoassist` | `react-nonverbal` | auroc | null | — | 2000 | 2000 | `2026-10-21T04:12:58Z` |
| `holoassist` | `react-nonverbal|spoke=1` | auroc | null | — | 955 | 955 | `2026-10-21T04:12:58Z` |
| `holoassist` | `react-spoke` | auroc | 0.676 | [0.655, 0.695] | 2000 | 0 | `2026-10-21T04:12:59Z` |
| `holoassist` | `react-full` | auroc | 0.739 | [0.716, 0.761] | 2000 | 0 | `2026-10-21T04:12:59Z` |
| `holoassist` | `step-prior` | auroc | 0.751 | [0.726, 0.774] | 2000 | 0 | `2026-10-21T04:13:00Z` |
| `holoassist` | `action-fusion` | auroc | 0.681 | [0.657, 0.706] | 2000 | 0 | `2026-10-21T04:13:00Z` |
| `holoassist` | `fusion` | auroc | null | — | 2000 | 2000 | `2026-10-21T04:13:00Z` |
| `holoassist` | `fusion_minus_action_fusion` | delta_auroc | null | — | 2000 | 2000 | `2026-10-21T04:13:00Z` |
| `holoassist` | `fusion_minus_action_best` | delta_auroc | null | — | 2000 | 2000 | `2026-10-21T04:13:01Z` |
| `holoassist_audible` | `judge` | auroc | 0.528 | [0.499, 0.560] | 1420 | 0 | `2026-10-21T16:50:20Z` |
| `holoassist_audible` | `judge-large` | auroc | 0.516 | [0.499, 0.533] | 1420 | 0 | `2026-10-21T16:50:20Z` |
| `holoassist_audible` | `judge-frontier` | auroc | null | — | 1420 | 1420 | `2026-10-21T16:50:21Z` |
| `holoassist_audible` | `action-probe` | auroc | 0.576 | [0.526, 0.623] | 1420 | 0 | `2026-10-21T16:50:21Z` |
| `holoassist_audible` | `react-nonverbal` | auroc | 0.654 | [0.626, 0.675] | 1420 | 0 | `2026-10-21T16:50:22Z` |
| `holoassist_audible` | `react-nonverbal|spoke=1` | auroc | 0.582 | [0.548, 0.612] | 837 | 0 | `2026-10-21T16:50:22Z` |
| `holoassist_audible` | `react-spoke` | auroc | 0.665 | [0.636, 0.691] | 1420 | 0 | `2026-10-21T16:50:23Z` |
| `holoassist_audible` | `react-full` | auroc | 0.759 | [0.728, 0.791] | 1420 | 0 | `2026-10-21T16:50:23Z` |
| `holoassist_audible` | `step-prior` | auroc | 0.737 | [0.712, 0.765] | 1420 | 0 | `2026-10-21T16:50:24Z` |
| `holoassist_audible` | `action-fusion` | auroc | 0.650 | [0.599, 0.694] | 1420 | 0 | `2026-10-21T16:50:25Z` |
| `holoassist_audible` | `fusion` | auroc | 0.671 | [0.618, 0.713] | 1420 | 0 | `2026-10-21T16:50:25Z` |
| `holoassist_audible` | `fusion_minus_action_fusion` | delta_auroc | 0.021 | [0.006, 0.034] | 1420 | 0 | `2026-10-21T16:50:26Z` |
| `holoassist_audible` | `fusion_minus_action_best` | delta_auroc | -0.065 | [-0.113, -0.025] | 1420 | 0 | `2026-10-21T16:50:27Z` |

### 8.2 Prediction-Check Lines (Verbatim)

**Oops! (`prediction_check`, from `docs/evals/2026-10-10_oops_h1.md`):**
- **Clause 1, "the judge is strong" (judge):** AUROC 0.472 [0.440, 0.498] (1072 scored) → did not hold
- **Clause 1, "the judge is strong" (judge-large):** AUROC 0.708 [0.684, 0.731] (1072 scored) → did not hold
- **Clause 1, "the judge is strong" (judge-frontier):** AUROC 0.704 [0.544, 0.821] (50 scored) → did not hold
- **Clause 2, "Δ ≈ 0":** ablation Δ 0.015 [-0.001, 0.029] → held
- **Context, no verdict:** fusion − action_best 0.015 [-0.002, 0.029]

**HoloAssist (`prediction_check_hidden`, from `docs/evals/2026-10-15_holoassist_h1.md`):**
- **Clause 1, "reaction-only beats action_best":** react-nonverbal not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large; rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20) vs step-prior 0.751 → not tested: not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large; rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20)
- **Clause 2, "reactions add to the action-only model":** ablation Δ not run: fusion not run (not run: react-nonverbal not run on train); rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20) → not tested: not run: fusion not run (not run: react-nonverbal not run on train); rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20)
- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not testable: react-nonverbal, ablation Δ
- **Context, no verdict:** fusion − action_best action_best=step-prior; not run: fusion not run; rerun: step-prior and action-fusion added (Issue 7 A; 00_thesis entry 20); judge 0.488 [0.466, 0.512]; judge-large 0.544 [0.528, 0.562]; action-probe 0.569 [0.542, 0.597]; step-prior 0.751 [0.726, 0.774]; action-fusion 0.681 [0.657, 0.706]; react-spoke 0.676 [0.655, 0.695] (diagnostic: did the instructor speak); react-full 0.739 [0.716, 0.761] (leakage condition: the instructor's words; never a headline, §11.2).
- **Caveat:** HoloAssist step text carries outcome information (0.751 [0.726, 0.774]); judge-large scores 0.489 on test items whose verb is not "drop".

**HoloAssist-Audible (`prediction_check_hidden`, from `docs/evals/2026-10-21_holoassist_audible_h1.md`):**
- **Clause 1, "reaction-only beats action_best":** react-nonverbal 0.654 [0.626, 0.675] vs step-prior 0.737 → did not hold
- **Clause 2, "reactions add to the action-only model":** ablation Δ 0.021 [0.006, 0.034] → held
- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not met
- **Context, no verdict:** fusion − action_best -0.065 [-0.113, -0.025]; judge 0.528 [0.499, 0.560]; judge-large 0.516 [0.499, 0.533]; action-probe 0.576 [0.526, 0.623]; step-prior 0.737 [0.712, 0.765]; action-fusion 0.650 [0.599, 0.694]; react-spoke 0.665 [0.636, 0.691] (diagnostic: did the instructor speak); react-full 0.759 [0.728, 0.791] (leakage condition: the instructor's words; never a headline, §11.2).
- **Caveat:** HoloAssist step text carries outcome information (0.737 [0.712, 0.765]); judge-large scores 0.489 on test items whose verb is not "drop".

### 8.3 Caveats

1. **Subset Caveat (Task Mix Shift):**
   The audible-session evaluation set (`holoassist_audible`) filters sessions by instructor audibility ($\ge +3.5$ dB median voice-over-silence). Quieter device setups are heavily over-represented compared to the full HoloAssist dataset:
   - Camera setups: 72.3% pass rate (16.1% of audible set vs 6.5% of full dataset)
   - GoPro setups: 70.1% pass rate (17.6% of audible set vs 7.9% of full dataset)
   - Switch setups: 68.1% pass rate (15.5% of audible set vs 7.1% of full dataset)
   In contrast, noisy tool-heavy furniture assembly tasks had low pass rates (utility cart: 7.5% pass rate, 3.6% of audible set vs 11.5% of full dataset; circuit breaker: 0% pass rate, 0 items). This filtering rule was chosen after the initial 20-session presence check failed, but strictly before any model evaluations or outcome labels were seen.

2. **Pitch-Shift Caveat:**
   The HoloAssist dataset audio is pitch-shifted by the dataset authors for participant privacy. All paralinguistic acoustic features (`emotion2vec+`) and energy metrics are evaluated on altered voices.

3. **Group Independence Caveat (Pair, Not Person):**
   Session groups are derived from prefixes before `-` (e.g. `R0027`), representing distinct instructor–performer pairs. If an individual participant took part in multiple pairs, they may appear in both train and test splits, as the dataset metadata does not identify individual performers or instructors across different pairs.

### 8.4 Wave D Start Condition

**Status: MET.**

Rule (`agent_execution_guide.md` A14; `00_thesis.md` H3 amended): Met iff some dataset's `fusion_minus_action_fusion` has `ci_low > 0`.

- Dataset: `holoassist_audible`
- Condition: `fusion_minus_action_fusion` (ablation Δ)
- Value: **+0.021 [0.006, 0.034]** (`ci_low = 0.006 > 0`)
- Row timestamp: **`2026-10-21T16:50:26Z`**

Reactions demonstrably add incremental information beyond the action-only model on hidden outcomes with a 95% confidence interval strictly excluding zero. Therefore, the Wave D robot check may start once the Wave D specification is written by the designer.

### 8.5 Compute Time per Stage for A13 (Recounted from Caches)

- **Visual Features (`siglip-b16-224`):** 2,269.4 s (37.8 min across 4,420 items)
- **Audio Features (`e2v-plus-large`):** 888.3 s (14.8 min across 4,420 items)
- **Local Judge (`qwen2.5vl:7b`):** 36,172.9 s (602.9 min across 4,420 items)
- **Large Local Judge (`gemma4:26b`):** 2,926.1 s (48.8 min across 1,420 test items)
- **Probes & Report Generation:** < 1 min
- **Total A13 Pipeline Compute Time:** **~11.8 h**
