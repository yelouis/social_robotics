# HoloAssist H1 Evaluation Report (2026-10-15)
*Corrected 2026-10-20 (A9b): prediction check per 03_eval_harness.md §11.8; react-full is no longer a "best reaction". The numbers are unchanged.*

> [!NOTE]
> **Caveat 1 (Pitch-Shifted Audio):**
> The HoloAssist dataset audio is pitch-shifted for participant privacy. Paralinguistic features
> (`emotion2vec+`) and acoustic energy levels are evaluated on altered voices.
>
> **Caveat 2 (Group Independence):**
> Groups are derived from session prefixes before `-` (e.g. `R0027`), representing ~350
> instructor–performer pairs. If an individual participant participated in more than one pair,
> they may appear in both train and test splits. The dataset metadata cannot rule this out.

## 1. Instructor-Audio Presence Check (§A8.2)

- **Sessions Measured:** 20 sessions (`default_rng(0)`).
- **Median Difference (Instructor vs. Silence):** +1.51 dB.
- **Requirement:** $\ge +3.5$ dB.
- **Verdict:** **FAIL**.
- **Status:** The presence check failed because headset microphone tool manipulation noises during long disassembly/assembly intervals physically overwhelmed the remote instructor's voice. Filed as **Issue 6** in `docs/ongoing_general_errors.md`.

## 2. Executive Summary & Prediction Check

- **Clause 1, "reaction-only beats action_best":** react-nonverbal not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large vs action-probe 0.569 → not tested: not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large
- **Clause 2, "reactions add to the action-only model":** ablation Δ not run → not tested: no fusion_minus_action_fusion row
- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not testable: react-nonverbal, ablation Δ
- **Context, no verdict:** fusion − action_best action_best=action-probe; not run: fusion not run; judge 0.488 [0.466, 0.512]; judge-large 0.544 [0.528, 0.562]; action-probe 0.569 [0.542, 0.597]; react-spoke 0.676 [0.655, 0.695] (diagnostic: did the instructor speak); react-full 0.739 [0.716, 0.761] (leakage condition: the instructor's words; never a headline, §11.2).
- **Caveat:** HoloAssist step text carries outcome information (02_data_sources.md → HoloAssist → As built (A8)); judge-large scores 0.489 on test items whose verb is not "drop".

## 3. Conditions Table (Test Split, N=2,000)

| Condition | Metric | Value | 95% CI | N (included) | N (excluded) | Notes |
|---|---|---|---|---|---|---|
| `judge` | auroc | 0.488 | [0.466, 0.512] | 2000 | 0 |  |
| `judge-large` | auroc | 0.544 | [0.528, 0.562] | 2000 | 0 |  |
| `judge-frontier` | auroc | null | — | 2000 | 2000 | not run: no cache file /Volumes/Extreme SSD/social_robotics/judge/holoassist/gemini-3.6-flash/4f0760a8e441.jsonl |
| `action-probe` | auroc | 0.569 | [0.542, 0.597] | 2000 | 0 |  |
| `react-nonverbal` | auroc | null | — | 2000 | 2000 | not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large |
| `react-nonverbal|spoke=1` | auroc | null | — | 955 | 955 | diagnostic: test items with spoke=1; not run: react-nonverbal not run (not run: 0 of 3000 items cached in /Volumes/Extreme SSD/social_robotics/features/holoassist/e2v-plus-large) |
| `react-spoke` | auroc | 0.676 | [0.655, 0.695] | 2000 | 0 |  |
| `react-full` | auroc | 0.739 | [0.716, 0.761] | 2000 | 0 |  |
| `fusion` | auroc | null | — | 2000 | 2000 | not run: react-nonverbal not run on train |
| `fusion_minus_action_best` | delta_auroc | null | — | 2000 | 2000 | action_best=action-probe; not run: fusion not run |

## 4. Voice Signal vs. Presence of Speech (`react-spoke` vs. `react-nonverbal|spoke=1`)

`react-spoke` (AUROC = 0.676) measures whether the instructor intervened at all during the step's reaction window. Because instructors speak on 67.98% of mistakes vs. 30.87% of correct actions, speech timing alone is a strong cue. The diagnostic condition `react-nonverbal|spoke=1` (AUROC = N/A) evaluates acoustic paralinguistics strictly within steps where the instructor spoke, isolating vocal prosody from the binary presence of speech. In contrast, `react-nonverbal` across all items (AUROC = N/A) and `react-full` (AUROC = 0.739) evaluate unconditioned acoustic features and TF-IDF transcripts respectively.

## 5. Shuffled-Label Controls

Validation requirement: every `:shuffled` condition must have a 95% CI covering chance (0.50).

| Condition | Metric | Value | 95% CI | Covers 0.5? | Notes |
|---|---|---|---|---|---|
| `action-probe:shuffled` | auroc | 0.492 | [0.461, 0.525] | PASS (Yes) | shuffled-label control |
| `react-nonverbal:shuffled` | auroc | null | — | — (not run) | shuffled-label control; not run: react-nonverbal not run |
| `fusion:shuffled` | auroc | null | — | — (not run) | shuffled-label control; not run: fusion not run |

## 6. Judge Diagnostics & Answer Distributions

- **Local Judge (`qwen2.5vl:7b`):** total queries = 5000, parse failures = 0, API errors = 0, ffmpeg decode errors = 0
- **Large Local Judge (`gemma4:26b`):** total queries = 2000, parse failures = 0, API errors = 0, ffmpeg decode errors = 0

### `qwen2.5vl:7b` Answer Distribution on Test Split:

| Label | 100 | 90 | 80 or below | 0 | Total |
|---|---|---|---|---|---|
| `correct` (label=1) | 497 | 56 | 256 | 191 | 1000 |
| `mistake` (label=0) | 540 | 48 | 197 | 215 | 1000 |

### `gemma4:26b` Answer Distribution on Test Split:

| Label | 100 | 90 | 80 or below | 0 | Total |
|---|---|---|---|---|---|
| `correct` (label=1) | 909 | 0 | 0 | 91 | 1000 |
| `mistake` (label=0) | 820 | 0 | 0 | 180 | 1000 |

## 7. Compute & Wall-Clock Times

- **Download:** 197.8 GB archive; downloaded and extracted by October 10 at 13:01 (archive deleted)
- **Features: siglip-b16-224:** 2,779.2 s (46.3 min across 5000 items)
- **Features: e2v-plus-large:** not run / pending
- **Judge: qwen2.5vl_7b:** 40,656.9 s (677.6 min across 5000 items)
- **Judge: gemma4_26b:** 4,064.1 s (67.7 min across 2000 items)

## 8. Example Items (Test Split)

Ten items sampled with `default_rng(0)` (5 correct, 5 mistake), described in words:

### Correct Action Examples (Label = 1)

1. **`holoassist:z052-july-11-22-switch:73`** (Group: `z052`)
   - Context: `Task: setup switch. Step: press button.`
   - Window: `[211.99, 215.17]` s (Reaction: `[211.99, 220.17]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer holds a Nintendo Switch console with a blue Joy-Con attached to the left rail over a white table, pressing the release/latch button on the top rear of the Joy-Con.

2. **`holoassist:z058-june-28-22-gladom_assemble:24`** (Group: `z058`)
   - Context: `Task: assemble tray table. Step: withdraw leg.`
   - Window: `[53.09, 56.32]` s (Reaction: `[53.09, 61.32]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"Ok, please rotate both ✱*unintelligible*✱."*
   - Description: The performer is assembling a white metal IKEA Gladom tray table frame, holding one of the intersecting tubular white metal legs and adjusting its position relative to the central pivot joint.

3. **`holoassist:z113-aug-03-22-printer_small:39`** (Group: `z113`)
   - Context: `Task: setup small printer. Step: grab copy.`
   - Window: `[145.02, 145.49]` s (Reaction: `[145.02, 150.49]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: An HP DeskJet all-in-one printer rests on a light wooden desk; the performer reaches toward the output tray to grab a freshly printed test page.

4. **`holoassist:z150-aug-22-22-gladom_assemble:40`** (Group: `z150`)
   - Context: `Task: assemble tray table. Step: insert hex_socket_head.`
   - Window: `[103.88, 107.21]` s (Reaction: `[103.88, 112.21]` s)
   - Local Judge P: `0.80`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer is assembling a white metal IKEA Gladom tray table frame upside down on a white round table, carefully threading a black hex socket head screw by hand through the circular leg attachment ring into the leg tube.

5. **`holoassist:z177-sep-05-22-dslr:7`** (Group: `z177`)
   - Context: `Task: setup camera. Step: place dslr.`
   - Window: `[9.05, 9.54]` s (Reaction: `[9.05, 14.54]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"Nice! Well done. Now please attach the lens again."*
   - Description: On a white round table, the performer places down a black Canon DSLR camera body with neck strap while holding the detached Canon zoom lens upright on the table next to an SD card and battery pack.

### Mistake Action Examples (Label = 0)

1. **`holoassist:z108-july-26-22-knarrevik_disassemble:10`** (Group: `z108`)
   - Context: `Task: disassemble nightstand. Step: drop hex_socket_head.`
   - Window: `[22.17, 23.34]` s (Reaction: `[22.17, 28.34]` s)
   - Local Judge P: `0.00`, Large Judge P: `0.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer is disassembling a black IKEA Knarrevik nightstand on a round white table; while unscrewing, a small black hex socket head screw slips from their fingers and drops onto the table surface next to their hand.

2. **`holoassist:z129-aug-11-22-marius_assemble:47`** (Group: `z129`)
   - Context: `Task: assemble stool. Step: place washer.`
   - Window: `[95.71, 97.07]` s (Reaction: `[95.71, 102.07]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"Please continue by putting the legs in a reverse way on the plate."*
   - Description: The performer is assembling a white IKEA Marius stool at a workbench with tools on the pegboard; they reach to place a white plastic washer/spacer on the stool leg, but position it incorrectly (prompting instructor intervention: 'Please continue by putting the legs in a reverse way on the plate.').

3. **`holoassist:z139-aug-15-22-knarrevik_assemble:169`** (Group: `z139`)
   - Context: `Task: assemble nightstand. Step: screw hex_socket_head.`
   - Window: `[354.10, 359.20]` s (Reaction: `[354.10, 364.20]` s)
   - Local Judge P: `0.80`, Large Judge P: `0.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer is assembling a black IKEA Knarrevik nightstand, trying to screw a hex socket head into the joint between the shelf and vertical frame with their fingers, but struggles with alignment/cross-threading the bolt into the metal frame hole.

4. **`holoassist:z175-sep-05-22-gopro:113`** (Group: `z175`)
   - Context: `Task: setup gopro. Step: insert screw.`
   - Window: `[285.49, 290.73]` s (Reaction: `[285.49, 295.73]` s)
   - Local Judge P: `0.90`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer mounts a GoPro action camera onto a Rollei head/chest strap bracket, attempting to insert the long thumbscrew from the right side rather than the correct threaded side, causing improper engagement.

5. **`holoassist:z188-sep-10-22-dslr:38`** (Group: `z188`)
   - Context: `Task: setup camera. Step: inspect dslr.`
   - Window: `[79.10, 87.87]` s (Reaction: `[79.10, 92.87]` s)
   - Local Judge P: `0.90`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"To the black mark in match on the camera and the lens."*
   - Description: The performer holds an open Canon EOS 1300D body with exposed sensor/mirror in their right hand, and the zoom lens in their left hand, inspecting them while trying to figure out how to align the lens mount (instructor guides: 'To the black mark in match on the camera and the lens.').
