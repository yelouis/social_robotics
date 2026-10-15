# Wave A Evaluation Summary (2026-10-15)

Wave A evaluates Hypothesis 1 (H1): human observer reactions provide an outcome-supervision signal for task success and error detection beyond action-only cues, and baseline VLM judges cannot replace them when outcomes are subtle or intermediate.

This document summarizes the end-to-end results across both Wave A datasets:
1. **Oops!** (Visible-outcome contrast: obvious physical falls and accidents).
2. **HoloAssist** (Hidden-outcome / intermediate-mistake benchmark: fine-grained physical assembly and manipulation with an instructor in the loop).

---

## 1. Headline Results Table (Dataset × Condition)

All metrics are AUROC or $\Delta\text{AUROC}$ on the held-out test splits. 95% confidence intervals are computed with 10,000 cluster-bootstrap draws grouped by session/video (`harness.metrics`). Every number matches its row in `results/scorecard.jsonl` (citing timestamp `ts`).

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
| Oops! | *Diagnostic:* `react-nonverbal\|spoke=1` | auroc | null | — | 0 | 1,072 | — | Not applicable to Oops! |
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
| HoloAssist | `react-full` | auroc | 0.739 | [0.716, 0.761] | 2,000 | 0 | `2026-10-15T20:07:58Z` | Instructor transcript language (TF-IDF) |
| HoloAssist | `fusion` | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:58Z` | not run: react-nonverbal not run on train |
| HoloAssist | `fusion_minus_action_best` | delta_auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:58Z` | action_best=action-probe; not run: fusion not run |
| HoloAssist | `action-probe:shuffled` | auroc | 0.492 | [0.461, 0.525] | 2,000 | 0 | `2026-10-15T20:07:59Z` | Control: PASS (covers 0.50) |
| HoloAssist | `react-nonverbal:shuffled` | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:59Z` | not run: react-nonverbal not run |
| HoloAssist | `fusion:shuffled` | auroc | null | — | 0 | 2,000 | `2026-10-15T20:07:59Z` | not run: fusion not run |

---

## 2. Per-Clause Prediction Checks

### 2.1 Oops! (Visible-Outcome Contrast)
> **Prediction:** In fail videos where failures are plainly visible in the action frames, the VLM judge is expected to perform strongly, and non-verbal reactions should add little or no incremental signal over the action-only baseline ($\Delta \approx 0$).

- **Clause 1, "VLM judge baseline is strong":**
  - `judge` (`qwen2.5vl:7b`): AUROC **0.472 [0.440, 0.498]** (1,072 scored) → **did not hold** (at chance / below 0.50).
  - `judge-large` (`gemma4:26b`): AUROC **0.708 [0.684, 0.731]** (1,072 scored) → **held** (reliably detects visible accidents).
  - `judge-frontier` (`gemini-3.6-flash`): AUROC **0.704 [0.544, 0.821]** (50 scored) → **held on sample** (quota-limited).
- **Clause 2, "Reactions add little or no incremental signal ($\Delta \approx 0$)":**
  - Best action baseline: `action-probe` = 0.779 [0.754, 0.805].
  - Multimodal fusion: `fusion` = 0.793 [0.771, 0.816].
  - $\Delta\text{AUROC}$ (`fusion_minus_action_best`): **+0.015 [-0.002, 0.029]** → **held** (CI lower bound is $\le 0.00$; no statistically significant incremental value over action video).

### 2.2 HoloAssist (Hidden-Outcome / Intermediate-Mistake Physical Assembly)
> **Prediction:** In fine-grained manipulation where mistakes are subtle or intermediate, the VLM judge cannot reliably distinguish mistakes from correct steps, but instructor reactions provide strong predictive signal beyond action frames ($\Delta > 0$, pass bar: reaction $\ge 0.65$ and $\Delta \ge +0.03$).

- **Clause 1, "VLM judge baseline is weak / near chance":**
  - `judge` (`qwen2.5vl:7b`): AUROC **0.488 [0.466, 0.512]** (2,000 scored) → **held** (indistinguishable from chance).
  - `judge-large` (`gemma4:26b`): AUROC **0.544 [0.528, 0.562]** (2,000 scored) → **held** (barely above chance; fails to recognize physical assembly errors despite strong performance on Oops!).
- **Clause 2, "Reactions provide strong signal beyond action frames":**
  - `action-probe` (SigLIP on action frames): AUROC **0.569 [0.542, 0.597]** (modest action signal).
  - `react-spoke` (timing of instructor utterance): AUROC **0.676 [0.655, 0.695]** (exceeds the 0.65 reaction bar).
  - `react-full` (TF-IDF on instructor spoken words): AUROC **0.739 [0.716, 0.761]** (strong reaction signal).
  - Multimodal acoustic audio (`e2v-plus-large`) was **not run** because the audio presence check failed (+1.51 dB vs. $\ge +3.5$ dB threshold; Issue 6). Consequently, acoustic `react-nonverbal` and `fusion` remain null.

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
3. **Excluded Shares & Parse Failures:**
   - All evaluated conditions had **0 items excluded (0.0%)**, with zero parse failures and zero API errors across 7,708 VLM queries.
   - All unrun conditions carry honest, explicit reasons derived from data configuration and contracts.

---

## 5. Compute & Wall-Clock Times per Stage

| Stage | Oops! (2,710 items) | HoloAssist (5,000 items) | Total Wave A Time |
|---|---|---|---|
| **Raw Download & Extraction** | 2,798.0 s (46.6 min; 45 GB archive) | Verified Oct 10 (197.8 GB archive) | ~2.5 h |
| **Visual Features (`siglip-b16-224`)** | 3,144.6 s (52.4 min) | 2,779.2 s (46.3 min) | 1.65 h |
| **Audio Features (`e2v-plus-large`)** | 276.9 s (4.6 min) | *Stopped per Issue 6* | 0.08 h |
| **Local Judge (`qwen2.5vl:7b`)** | 22,542.9 s (375.7 min / 6.3 h) | 56,986.1 s (949.8 min / 15.8 h) | 22.1 h |
| **Large Local Judge (`gemma4:26b`)**| 2,477.9 s (41.3 min, 1,072 items) | 4,114.6 s (68.6 min, 2,000 items) | 1.83 h |
| **Frontier Judge (`gemini-3.6-flash`)**| 1,351.5 s (22.5 min; 50 items) | *Not run per Issue 5 B* | 0.38 h |
| **Linear Probes & 10k Bootstrapping** | < 10 s | < 10 s | < 1 min |
| **Total Pipeline Compute Time** | **~9.1 h** | **~19.4 h** | **~28.5 h** |

---

## 6. Observations & Design Recommendations for Wave B

1. **Reaction Signal Exists and Overcomes Action Probes in Complex Tasks:**
   In HoloAssist, visual action representations (`siglip-b16-224`) achieve only 0.569 AUROC on subtle procedural errors, and VLMs cannot resolve them (0.488–0.544). In contrast, human reaction timing alone (`react-spoke`) reaches 0.676 AUROC, and reaction transcripts (`react-full`) reach 0.739 AUROC. Observer reactions provide rich supervisory signal precisely where action-only computer vision struggles.
2. **Microphone Separation is Mandatory for Acoustic Non-Verbal Reactions:**
   A microphone co-located on the worker/robot will pick up manipulation noise that masks human vocal affect. Wave B (verdict-video and bystander reaction data) must ensure either:
   - Microphones directed at or worn by the human observer;
   - Facial affect analysis (`react-face`, DW2);
   - Speech recognition (ASR) to leverage transcript language (`react-full`).
3. **VLM Judges Need Scale for Gross Failures, but Scale Alone Does Not Solve Task Grounding:**
   Scaling from 7B to 26B (`gemma4:26b`) resolved the judge failure on gross physical accidents (0.472 $\to$ 0.708 on Oops!), but barely moved the needle on procedural task mistakes (0.488 $\to$ 0.544 on HoloAssist). Zero-shot generic VLMs are not sufficient evaluators for fine-grained robotic manipulation without explicit task grounding.
4. **Memory Guard and Infrastructure Robustness:**
   The `shared.memguard` subsystem successfully prevented out-of-memory crashes throughout 28+ hours of intensive inference across 64 GB host memory, sharing resources seamlessly with concurrent host processes. This architecture is validated and ready for Wave B.
