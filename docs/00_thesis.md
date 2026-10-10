# 00: Thesis, Hypotheses & Decision Log

**Read this first.** This is the grounding document for any person or coding agent working in this repo. If something here conflicts with code or another doc, this file wins until it is updated.

---

## The thesis

People react to what others do: a wince, a laugh, a "yes!", a face at the first bite. Those reactions are a **reward signal**. We want to show three things:

1. The signal carries information about **whether an action went well** that the action video alone does not.
2. It can be learned **at internet scale from ordinary video** (vlogs, POV and reaction footage) with **no human labeling**.
3. Adding it to the conventional RL reward makes **robot learning better**.

## What changed (October 2026) and why

The v0 project, the Social-Affective Filter (SAF), is archived at git tag `v0-saf-final`. It built a six-layer perception pipeline over 991 Ego4D clips (gaze, categorical emotion, prosody, proxemics, nods and flinches). It then tried to validate the pipeline with human-rated "which emotion, about which task" labels. That failed.

- **The maintainer could not rate the moments.** 0 of 349 were rated, because the question had no answer for most of them.
- **A frontier model labeled ordinary engagement as approval.** Gemini pre-seeds came out 130 "approving", 67 "neutral" and 0 "disapproving", with rationales like "picks up card carefully".
- **The pipeline itself barely produced signal.** Emotion was readable on 5.1% of segments and non-zero on 1.1%.

The root causes are in [`LESSONS_v0.md`](LESSONS_v0.md). In short:

- the project never had an outcome variable, so the only possible validation was human judgment;
- Ego4D social footage rarely contains an evaluative reaction to the wearer's action;
- the label question was ill-posed;
- categorical emotion was the wrong intermediate representation.

## Core design principle: supervise on outcome, never on emotion

Emotion is a **latent variable**. We never label it, never sort it into categories, and never ask a human what someone felt. Every model is trained and evaluated against an **outcome label** that comes from the data itself:

- Was the food rated well? (a spoken verdict)
- Was a mistake made? (a dataset annotation)
- Did the attempt fail? (a failure timestamp)

Consequences:
- **No human rating in the loop.** Every metric is automatic and cheap enough to run on every change ([`03_eval_harness.md`](03_eval_harness.md)). Human spot-checks of *our code* (e.g. "did the verdict parser read '7 out of 10' correctly?") are fine; they test code, not label meaning.
- The only question a model has to answer is *"does this reaction predict the outcome?"*, never *"which emotion is this?"*
- Representations come from pretrained encoders, not hand-built emotion, gaze or nod channels.

## Two-stage deployment story

| Stage | Where the reaction comes from | What we build | Human present at robot runtime? |
|---|---|---|---|
| **B: reactions as free labels** | Reactions in web video | A reward model: the reactions label the actor's action, so the model learns which outcomes are good | No |
| **A: live reactions** | A person watching the robot | A reaction reader pre-trained on web video that reads *this* person's face and voice live | Yes |

Order: H1 and H2 serve both stages. Stage B comes first because that is where the scale is. Stage A comes later, with a small robot (possibly a microduck) as the first live testbed.

**Why first-person footage isn't required.** The policy (VLA) sees through the robot's cameras. The reward model is a separate critic that outputs a score, and robot setups usually include side-view cameras anyway. The domain gap that matters is human hands vs. robot gripper, not camera angle. The *live* reaction reader (stage A) does match Ego4D's geometry (robot camera → person in front of it), which is why first-person data is part of evaluation (HoloAssist) even though training uses any footage.

**Why this is not "just ask an LLM what's socially appropriate".** An LLM already knows the general norms, so a model that only learned those would be an expensive copy of it. What a reaction carries that an LLM cannot have:

1. **Hidden outcomes.** Whether the soup is too salty, whether a handover was uncomfortable, whether the instructor saw a wrong screw. Nothing in the action frames shows this.
2. **Execution, not category.** *This* handover at *this* speed made *this* person flinch.
3. **This person, now** (stage A). Preferences differ, and only observing the person reveals theirs.

That is why the VLM judge is the **central control** in H1: the thesis must beat it where it should and tie it where it should.

---

## Hypotheses

### H1: Signal
On data with known outcomes, observer reactions predict the outcome **and add information beyond what a VLM judge infers from the action alone**.

- **Conditions** (exact definitions: [`03_eval_harness.md`](03_eval_harness.md) §6):
  - (a) **Action only, two controls.** A zero-shot **VLM judge** (reactions masked, audio muted) is the "just ask an LLM" control. An **action probe** trained on the same labels is the fair learned control. `action_best` is whichever scores higher.
  - (b) **Reaction only.** Face, non-verbal audio and full audio are reported separately.
  - (c) **Fusion** of (a) and (b).
- **Metric:** AUROC with 95% bootstrap CI, on group splits (no person, channel or session on both sides).
- **Prediction:**
  - On **visible-outcome** data (Oops! failures), (a) is strong and (c) ≈ (a).
  - On **hidden-outcome** data (verdict videos, HoloAssist mistakes), (b) > (a) and (c) > (a).
- **Pass (proposed):** on hidden-outcome data, reaction-only AUROC ≥ 0.65 **and** fusion − `action_best` ≥ +0.03 with the paired CI excluding 0.
- **Kill:** if reactions add nothing over `action_best` on the **verdict data** (Wave B, the purest hidden-outcome test), the strong thesis is false. Stop, and write up the negative result. Wave A's HoloAssist and Oops! results are reported, but never trigger the kill rule on their own.

### H2: Scale and transfer
A reaction model trained on web video transfers to held-out datasets it never saw, including **reactions to robots** (BAD, ERR@HRI) and **first-person** data (HoloAssist), and improves with more web data.

- **Metrics:**
  - zero-shot AUROC on each target;
  - a scaling curve (AUROC vs. hours of web training data, log-x);
  - comparison to a model trained on the target's own small train split.
- **Pass (proposed):** zero-shot ≥ the in-domain small-data baseline on at least one held-out target (HoloAssist, AM-FED+, or BAD if granted), plus a monotone scaling curve.

### H3: Robot data efficiency (offline in the 3-month scope; online later)
Reaction-labeled human task video makes a robot **success detector** (the reward model) more accurate **with fewer robot labels**. The thesis's practical promise, "robots can learn from vlogs", is exactly this.

- **Offline (in scope, Wave D):**
  - **Data:** [RoboReward](https://huggingface.co/datasets/teetone/RoboReward) (CC-BY-4.0; 54,135 robot rollouts scored 1–5; a 2,831-item human-verified test split). Only real successes and failures are used; the counterfactual-instruction negatives are excluded, because our detector reads no instructions.
  - **Measure:** a data-efficiency curve, AUROC (success vs. failure) against the number of robot training labels (e.g. 50–1,000), **with vs. without** pre-training on reaction-labeled human task video (HoloAssist-style). Reference lines: RoboReward-8B and the VLM judge. **Never framed as beating RoboReward-8B.**
  - **Start condition:** Wave A shows a positive, CI-excluding-0 Δ (reactions over `action_best`) on at least one dataset. Otherwise there is no signal to transfer.
- **Online (later, not in scope):** a policy trained with the improved reward learns faster in simulation (needs a cloud CUDA GPU; `microduck_rl`/mjlab need CUDA) or on hardware. This is the final link; the 3-month paper stops one step before it.

---

## Scope & resources (next ~3 months)

- **Goal:** paper-grade evidence for H1 and H2, plus **offline H3** (robot data efficiency, Wave D, gated on Wave A). Online H3 (actually training a robot) is later.
- **Compute:** Mac Studio (M4 Max, 64 GB) only. That means **frozen pretrained encoders + small probes**; no end-to-end video-model fine-tuning.
- **Robot:** none yet. A microduck (Pollen Robotics) may come later for stage A.

## Roadmap

The build is run by an implementing agent from [`agent_execution_guide.md`](agent_execution_guide.md). Issues and decisions live in [`ongoing_general_errors.md`](ongoing_general_errors.md).

| Wave | Deliverable | Gate |
|---|---|---|
| **A** (≈ weeks 1–4) | Harness (items, splits, metrics, scorecard), encoders, VLM judge. First H1 numbers on **Oops!** (visible-outcome contrast) and **HoloAssist** (first-person, hidden outcome) | HoloAssist independence check (A4) |
| **B** (≈ weeks 4–7) | **Hidden-outcome H1 pilot**: a reaction followed by a verdict (AM-FED+, creator-permitted or CC taste-test videos, per Issue 1), with face + non-verbal audio. If no verdict source is granted, HoloAssist carries the hidden-outcome test | **Issue 1** selected; Wave A closed |
| **C** (≈ weeks 7–10) | **H2**: transfer to HoloAssist / AM-FED+ (plus BAD only if the access request is granted); scaling curves | Wave B closed |
| **D** (≈ weeks 7–10, alongside C) | **Robot check (offline H3)**: on RoboReward, does reaction-labeled human task video make a robot success detector more accurate with fewer robot labels? | Wave A shows reaction signal on ≥ 1 dataset; spec written by the designer after Wave A |
| **E** (≈ weeks 11–12) | Paper write-up | — |

---

## Decision log

**2026-10-08** (maintainer interview, after the v0 rating attempt failed):
1. SAF v0 retired: tagged `v0-saf-final`, code and docs removed from main, lessons kept in [`LESSONS_v0.md`](LESSONS_v0.md).
2. **No human rating in the loop.** Outcome supervision only; automatic metrics on every change.
3. Deployment: **both stages**, with B (web labels) first and A (live reactions) later.
4. Data: **any footage** for training, not first-person only. Evaluation reports both side-view robot-reaction data (BAD) and first-person data (HoloAssist).
5. The **"just ask an LLM/VLM" baseline** is H1's central control (raised by the maintainer as the key novelty challenge).
6. **Hidden-outcome data first** (verdict videos, HoloAssist mistakes), with visible-outcome data (Oops!) as the contrast.
7. Three-month goal: **paper-grade H1 + H2**. Compute: Mac Studio only.

**2026-10-08** (designer, writing the Wave A spec):
8. **Two action-only controls, not one.** A zero-shot judge can be weak for reasons unrelated to reactions, so a probe trained on the same labels is added and the Δ is taken against the better of the two (`action_best`). This strengthens the "just ask an LLM" control the maintainer asked for.
9. **The kill rule is decided on verdict videos only.** HoloAssist's instructor audio is pitch-shifted and Oops! is a localization task, so neither is a clean hidden-outcome test on its own.
10. **Implementation is done by a separate agent from the execution guide.** The designer writes specs and validation and does not code (maintainer: *"Don't perform any coding … Create an agent_execution guide with clear guidelines and validation and let another agent implement"*). **Commits go straight to `main`, with no feature branches** (maintainer: *"No need to open up a new branch, just push to the repo"*).
11. **Issue 3 → the v0 videos were deleted** (maintainer: *"clean up any videos you want from the v0 leftovers. Feel free to delete Ego4D if you think that is the right choice"*). That was the 1,083 Ego4D clips and the Charades-Ego videos, 1.336 TB in total. The manifest is `DATA_ROOT/DELETED_2026-10-08.json`.
12. **No self-recorded data.** The maintainer declined recording their own taste tests (*"I will not do this"*, October 9, 2026). Hidden-outcome verdict data must come from licensed datasets (AM-FED+) or creator-permitted and CC video (Issue 1).
13. **No human-subjects studies and no academic partner** (October 9, 2026; Issue 4). A robot-reaction target exists only if BAD access is granted to an independent researcher. Otherwise H2 transfer is shown on HoloAssist and AM-FED+.
14. **The offline robot check is in scope** (maintainer, October 9, 2026, after asking whether it is recommended). It is the only robot result feasible with no people, partner or hardware, and it is framed as data efficiency, not as beating RoboReward. **The live microduck demo stays deferred** (DW6).

**2026-10-10** (designer, verification of A1–A7):
15. **`react-spoke` scores `1 − spoke`.** Silence means P(good) is higher, because `label = 1` is a correct action. The direction is fixed a priori from A4's label-level rates and is never chosen on test (`03_eval_harness.md` §6).
16. **A partial frontier run is reported as partial.** It enters `action_best` only at ≥ 90% coverage of the test split (`03_eval_harness.md` §6, §8).
17. **First Oops! result** (`docs/evals/2026-10-10_oops_h1.md`):
    - **Δ ≈ 0, as predicted:** Δ = +0.015 [−0.002, 0.029].
    - **The local judge, predicted to be strong, was at chance:** 0.472.
    - **Oops! `react-nonverbal` (0.711) is not evidence of reaction signal,** because its audio window contains the failure's own sound (`02_data_sources.md` → Oops! → "As built").
    - **Whether a frontier judge is run is Issue 5.**

**2026-10-10** (maintainer):
18. **Issue 5 → option B: a larger local judge instead of paid frontier calls.** `gemma4:26b` (already installed, vision-capable) is added as the condition `judge-large`, on the test split only. It enters `action_best` at ≥ 90% coverage. It is shared with `animated_infographics`, so it runs with that project's runner options and is never unloaded by us (`03_eval_harness.md` §8). The frontier judge stays on the free tier.

## Open questions

Tracked as issues in [`ongoing_general_errors.md`](ongoing_general_errors.md): **Issue 1** web-video sourcing & licensing (needs your selection before Wave B), **Issue 4** (robot-reaction target) is decided: the BAD request, else none. **Issue 5** (the frontier judge) is decided: option B, a larger local judge (`judge-large`). **Issue 2** (resolved: independent) and **Issue 3** (SSD capacity) are resolved.
