# HoloAssist (Audible Sessions) H1 Evaluation Report (2026-10-21)

> [!NOTE]
> **Caveat 1 (Pitch-Shifted Audio):**
> The HoloAssist dataset audio is pitch-shifted for participant privacy. Paralinguistic features
> (`emotion2vec+`) and acoustic energy levels are evaluated on altered voices.
>
> **Caveat 2 (Group Independence):**
> Groups are derived from session prefixes before `-` (e.g. `R0027`), representing ~350
> instructor–performer pairs. If an individual participant participated in more than one pair,
> they may appear in both train and test splits. The dataset metadata cannot rule this out.

## The subset: Audible-Session Evaluation Set

- **Audible-session rule:** Sessions with median instructor speech dBFS minus silent gap (>= 2 s) dBFS >= +3.5 dB (measured label-blind in A12; `tools/check_audio_presence.py --all-sessions`).
- **Session and item yield:**
  - All HoloAssist: 1,758 sessions (1,673 with video); 5,000 items sampled (3,000 train / 2,000 test).
  - Audible HoloAssist: 548 audible sessions; 4,420 items sampled (3,000 train / 1,420 test across 53 test groups).
- **Task mix comparison:**
  Quieter device setups (camera, GoPro, Switch, printers) are over-represented in the audible set compared to the full dataset, while noisy furniture assembly tasks are reduced:

| Task | HoloAssist Items | % | HoloAssist-Audible Items | % |
|---|---|---|---|---|
| `assemble computer` | 102 | 2.0% | 26 | 0.6% |
| `assemble laser scanner` | 47 | 0.9% | 0 | 0.0% |
| `assemble nightstand` | 621 | 12.4% | 455 | 10.3% |
| `assemble stool` | 390 | 7.8% | 354 | 8.0% |
| `assemble tray table` | 346 | 6.9% | 136 | 3.1% |
| `assemble utility cart` | 573 | 11.5% | 157 | 3.6% |
| `change belt` | 15 | 0.3% | 9 | 0.2% |
| `change circuit breaker` | 25 | 0.5% | 0 | 0.0% |
| `disassemble nightstand` | 426 | 8.5% | 258 | 5.8% |
| `disassemble stool` | 185 | 3.7% | 89 | 2.0% |
| `disassemble tray table` | 164 | 3.3% | 92 | 2.1% |
| `disassemble utility cart` | 338 | 6.8% | 84 | 1.9% |
| `make coffee with espresso machine` | 91 | 1.8% | 38 | 0.9% |
| `make coffee with nespresso machine` | 207 | 4.1% | 89 | 2.0% |
| `repair motorcycle` | 60 | 1.2% | 5 | 0.1% |
| `setup big printer` | 161 | 3.2% | 267 | 6.0% |
| `setup camera` | 324 | 6.5% | 711 | 16.1% |
| `setup gopro` | 394 | 7.9% | 776 | 17.6% |
| `setup small printer` | 175 | 3.5% | 187 | 4.2% |
| `setup switch` | 356 | 7.1% | 687 | 15.5% |
| **Total** | **5000** | **100.0%** | **4420** | **100.0%** |

## 1. Instructor-Audio Presence Check (§A8.2)

- **Sessions Measured:** 20 sessions (`default_rng(0)`).
- **Median Difference (Instructor vs. Silence):** +1.51 dB.
- **Requirement:** $\ge +3.5$ dB.
- **Verdict:** **FAIL**.
- **Status:** The presence check failed because headset microphone tool manipulation noises during long disassembly/assembly intervals physically overwhelmed the remote instructor's voice. Filed as **Issue 6** in `docs/ongoing_general_errors.md`.

## 2. Executive Summary & Prediction Check

- **Clause 1, "reaction-only beats action_best":** react-nonverbal 0.654 [0.626, 0.675] vs step-prior 0.737 → did not hold
- **Clause 2, "reactions add to the action-only model":** ablation Δ 0.021 [0.006, 0.034] → held
- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → not met
- **Context, no verdict:** fusion − action_best -0.065 [-0.113, -0.025]; judge 0.528 [0.499, 0.560]; judge-large 0.516 [0.499, 0.533]; action-probe 0.576 [0.526, 0.623]; step-prior 0.737 [0.712, 0.765]; action-fusion 0.650 [0.599, 0.694]; react-spoke 0.665 [0.636, 0.691] (diagnostic: did the instructor speak); react-full 0.759 [0.728, 0.791] (leakage condition: the instructor's words; never a headline, §11.2).
- **Caveat:** HoloAssist step text carries outcome information (0.737 [0.712, 0.765]); judge-large scores 0.489 on test items whose verb is not "drop".

## 3. Conditions Table (Test Split, N=1,420)

| Condition | Metric | Value | 95% CI | N (included) | N (excluded) | Notes |
|---|---|---|---|---|---|---|
| `judge` | auroc | 0.528 | [0.499, 0.560] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `judge-large` | auroc | 0.516 | [0.499, 0.533] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `judge-frontier` | auroc | null | — | 1420 | 1420 | not run: no cache file /Volumes/Extreme SSD/social_robotics/judge/holoassist_audible/gemini-3.6-flash/4f0760a8e441.jsonl; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `action-probe` | auroc | 0.576 | [0.526, 0.623] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `react-nonverbal` | auroc | 0.654 | [0.626, 0.675] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `react-nonverbal|spoke=1` | auroc | 0.582 | [0.548, 0.612] | 837 | 0 | diagnostic: test items with spoke=1; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `react-spoke` | auroc | 0.665 | [0.636, 0.691] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `react-full` | auroc | 0.759 | [0.728, 0.791] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `step-prior` | auroc | 0.737 | [0.712, 0.765] | 1420 | 0 | text-only: train-split P(correct | verb, noun), hierarchical smoothing m=5; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `fusion` | auroc | 0.671 | [0.618, 0.713] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `action-fusion` | auroc | 0.650 | [0.599, 0.694] | 1420 | 0 | Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `fusion_minus_action_fusion` | delta_auroc | 0.021 | [0.006, 0.034] | 1420 | 0 | ablation: reaction features removed; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `fusion_minus_action_best` | delta_auroc | -0.065 | [-0.113, -0.025] | 1420 | 0 | action_best=step-prior; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |

## 4. Voice Signal vs. Presence of Speech (`react-spoke` vs. `react-nonverbal|spoke=1`)

`react-spoke` (AUROC = 0.665) measures whether the instructor intervened at all during the step's reaction window. Because instructors speak on 67.98% of mistakes vs. 30.87% of correct actions, speech timing alone is a strong cue. The diagnostic condition `react-nonverbal|spoke=1` (AUROC = 0.582) evaluates acoustic paralinguistics strictly within steps where the instructor spoke, isolating vocal prosody from the binary presence of speech. In contrast, `react-nonverbal` across all items (AUROC = 0.654) and `react-full` (AUROC = 0.759) evaluate unconditioned acoustic features and TF-IDF transcripts respectively.

## 5. Shuffled-Label Controls

Validation requirement: every `:shuffled` condition must have a 95% CI covering chance (0.50).

| Condition | Metric | Value | 95% CI | Covers 0.5? | Notes |
|---|---|---|---|---|---|
| `action-probe:shuffled` | auroc | 0.490 | [0.455, 0.529] | PASS (Yes) | shuffled-label control; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `react-nonverbal:shuffled` | auroc | 0.496 | [0.465, 0.526] | PASS (Yes) | shuffled-label control; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `step-prior:shuffled` | auroc | 0.483 | [0.449, 0.510] | PASS (Yes) | shuffled-label control; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `action-fusion:shuffled` | auroc | 0.491 | [0.455, 0.530] | PASS (Yes) | shuffled-label control; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |
| `fusion:shuffled` | auroc | 0.495 | [0.465, 0.528] | PASS (Yes) | shuffled-label control; Issue 6 A: audible sessions (>= +3.5 dB; 02_data_sources.md) |

## 6. Judge Diagnostics & Answer Distributions

- **Local Judge (`qwen2.5vl:7b`):** total queries = 4420, parse failures = 0, API errors = 0, ffmpeg decode errors = 0
- **Large Local Judge (`gemma4:26b`):** total queries = 1420, parse failures = 0, API errors = 0, ffmpeg decode errors = 0

### `qwen2.5vl:7b` Answer Distribution on Test Split:

| Label | 100 | 90 | 80 or below | 0 | Total |
|---|---|---|---|---|---|
| `correct` (label=1) | 432 | 41 | 145 | 92 | 710 |
| `mistake` (label=0) | 397 | 60 | 126 | 127 | 710 |

### `gemma4:26b` Answer Distribution on Test Split:

| Label | 100 | 90 | 80 or below | 0 | Total |
|---|---|---|---|---|---|
| `correct` (label=1) | 657 | 0 | 0 | 53 | 710 |
| `mistake` (label=0) | 634 | 0 | 0 | 76 | 710 |

## 7. Compute & Wall-Clock Times

- **Download:** 197.8 GB archive; downloaded and extracted by October 10 at 13:01 (archive deleted)
- **Features: siglip-b16-224:** 2,269.4 s (37.8 min across 4420 items)
- **Features: e2v-plus-large:** 888.3 s (14.8 min across 4420 items)
- **Judge: qwen2.5vl_7b:** 36,172.9 s (602.9 min across 4420 items)
- **Judge: gemma4_26b:** 2,926.1 s (48.8 min across 1420 items)

## 8. Example Items (Test Split)

Ten items sampled with `default_rng(0)` (5 correct, 5 mistake), described in words:

### Correct Action Examples (Label = 1)

1. **`holoassist_audible:z056-june-28-22-knarrevik_assemble:91`** (Group: `z056`)
   - Context: `Task: assemble nightstand. Step: place allen_wrench.`
   - Window: `[349.10, 350.50]` s (Reaction: `[349.10, 355.50]` s)
   - Local Judge P: `0.80`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer assembles a black metal frame on a white table, holding a silver L-shaped Allen wrench in their right hand and inserting the short end into a black hex screw head on a vertical strut while steadying the strut with their left hand.

2. **`holoassist_audible:z073-july-01-22-dslr:5`** (Group: `z073`)
   - Context: `Task: setup camera. Step: lift dslr.`
   - Window: `[13.29, 14.17]` s (Reaction: `[13.29, 19.17]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: A hand in a red sleeve grasps a black Nikon DSLR camera with an attached lens cap and neck strap, lifting it upright off a white tabletop into the center of the frame.

3. **`holoassist_audible:z099-july-13-22-switch:71`** (Group: `z099`)
   - Context: `Task: setup switch. Step: stand nintendo_switch.`
   - Window: `[155.19, 157.70]` s (Reaction: `[155.19, 162.70]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"Nice. The next step is to change the game card."*
   - Description: Both hands hold a black Nintendo Switch console with blue and red Joy-Cons attached, resting it upright on its bottom edge on a white desk next to a detached grip controller and game cards.

4. **`holoassist_audible:z129-aug-11-22-rashult_assemble:157`** (Group: `z129`)
   - Context: `Task: assemble utility cart. Step: grab screwdriver.`
   - Window: `[317.94, 319.03]` s (Reaction: `[317.94, 324.03]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: While assembling a black curved metal cart frame on a white table scattered with screws and caster wheels, the performer steadies a joint with their left hand and brings forward a blue-handled screwdriver with their right hand toward the frame.

5. **`holoassist_audible:z171-sep-03-22-knarrevik_assemble:14`** (Group: `z171`)
   - Context: `Task: assemble nightstand. Step: screw hex_socket_head.`
   - Window: `[43.20, 49.90]` s (Reaction: `[43.20, 54.90]` s)
   - Local Judge P: `0.80`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer works over a black metal nightstand assembly on a white desk, using their fingers to thread a black screw by hand into a pre-drilled corner hole of the metal frame next to loose screws and an Allen key.

### Mistake Action Examples (Label = 0)

1. **`holoassist_audible:z092-july-11-22-gopro:64`** (Group: `z092`)
   - Context: `Task: setup gopro. Step: press sd_card.`
   - Window: `[137.94, 139.90]` s (Reaction: `[137.94, 144.90]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `0`, Transcript: *"None"*
   - Description: The performer holds a small black GoPro camera body upright on a table with their left hand, using their right thumb and index finger to press and fidget with the narrow edge of an open compartment at the top.

2. **`holoassist_audible:z109-july-27-22-switch:27`** (Group: `z109`)
   - Context: `Task: setup switch. Step: insert joy_con_controller.`
   - Window: `[62.51, 64.45]` s (Reaction: `[62.51, 69.45]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"That goes to your right side, yes."*
   - Description: The performer holds the black Nintendo Switch console body facing away in their left hand, using their right hand to slide a bright red Joy-Con downward onto the console's right-side mounting rail.

3. **`holoassist_audible:z114-aug-03-22-printer_big:33`** (Group: `z114`)
   - Context: `Task: setup big printer. Step: pull cartridge_gate.`
   - Window: `[61.15, 62.81]` s (Reaction: `[61.15, 67.81]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"You can open it from the top. The black ink is the one on the left."*
   - Description: In front of a large office multifunction printer, the performer grips the upper front white panel door near the top corners with both hands and pulls it forward to hinge it open.

4. **`holoassist_audible:z157-aug-27-22-rashult_disassemble:88`** (Group: `z157`)
   - Context: `Task: disassemble utility cart. Step: drop screw.`
   - Window: `[338.20, 340.19]` s (Reaction: `[338.20, 345.19]` s)
   - Local Judge P: `0.00`, Large Judge P: `0.00`, Spoke: `0`, Transcript: *"None"*
   - Description: While holding a black-handled screwdriver near the vertical metal tubes of a cart frame on a desk, a small metallic screw slips from the hand and drops downward onto the desk surface near an open wrench and caster parts.

5. **`holoassist_audible:z185-sep-09-22-dslr:18`** (Group: `z185`)
   - Context: `Task: setup camera. Step: press release_button.`
   - Window: `[30.09, 30.52]` s (Reaction: `[30.09, 35.52]` s)
   - Local Judge P: `1.00`, Large Judge P: `1.00`, Spoke: `1`, Transcript: *"The button that's on the camera. It's on the bottom of the left side."*
   - Description: The performer holds a black Canon EOS 1300D DSLR camera fitted with a large zoom lens on a white table next to a detached battery pack, with their left thumb searching across the camera body's side panel near the lens mount.
