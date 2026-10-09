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

- **Conditions:**
  - (a) **VLM judge, action only.** Reactions are masked or muted. This is the "just ask an LLM" control.
  - (b) **Reaction only.** Face, non-verbal audio and full audio are reported separately.
  - (c) **Fusion** of (a) and (b).
- **Metric:** AUROC with 95% bootstrap CI, on group splits (no person, channel or session on both sides).
- **Prediction:**
  - On **visible-outcome** data (Oops! failures), (a) is strong and (c) ≈ (a).
  - On **hidden-outcome** data (verdict videos, HoloAssist mistakes), (b) > (a) and (c) > (a).
- **Pass (proposed):** on hidden-outcome data, reaction-only AUROC ≥ 0.65 **and** fusion − judge ≥ +0.03 with the CI excluding 0.
- **Kill:** if reactions add nothing over the judge even on hidden outcomes, the strong thesis is false. Stop, and write up the negative result.

### H2: Scale and transfer
A reaction model trained on web video transfers to held-out datasets it never saw, including **reactions to robots** (BAD, ERR@HRI) and **first-person** data (HoloAssist), and improves with more web data.

- **Metrics:**
  - zero-shot AUROC on each target;
  - a scaling curve (AUROC vs. hours of web training data, log-x);
  - comparison to a model trained on the target's own small train split.
- **Pass (proposed):** zero-shot ≥ the in-domain small-data baseline on at least one robot-reaction target, plus a monotone scaling curve.

### H3: Learning (stretch goal, after the H1/H2 paper)
Adding reaction-derived reward to task reward improves robot learning.

- **Offline first:** Kendall τ between reward-model scores and ground-truth success on labeled robot trajectories, against a VLM reward baseline (RoboReward/TOPReward-style).
- **Online later:** success vs. samples in MuJoCo (Mac Studio), and live reactions to a small robot for stage A.

---

## Scope & resources (next ~3 months)

- **Goal:** paper-grade evidence for H1 and H2. H3 is a stretch goal.
- **Compute:** Mac Studio (M4 Max, 64 GB) only. That means **frozen pretrained encoders + small probes**; no end-to-end video-model fine-tuning.
- **Robot:** none yet. A microduck may come later for stage A.

## Roadmap

| Phase | Weeks | Deliverable | Gate |
|---|---|---|---|
| 0 | 1 | Eval harness skeleton (`results/scorecard.jsonl`); data-access requests sent (BAD, ERR@HRI); HoloAssist download started | — |
| 1 | 1–3 | **H1 pilot**: ~200 verdict videos end-to-end, all three conditions | Is there *any* reaction signal beyond the judge? If none, stop or rethink |
| 2 | 3–7 | Scale the verdict corpus; HoloAssist (first-person, hidden outcome); Oops! (visible-outcome contrast) | H1 pass/kill |
| 3 | 7–10 | **H2**: transfer to BAD / ERR@HRI / HoloAssist; scaling curves | H2 pass |
| 4 | 10–12 | Paper write-up; decide on H3 | — |

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

## Open questions

- **Web-video sourcing & licensing.** Option one: download public videos for research and release them dehydrated (IDs + timestamps + labels + features, as v0 did). Option two: restrict to CC-BY via the YouTube Data API. **Needs a decision before the Phase-1 harvest.**
- **HoloAssist label independence.** If mistake annotations were derived from instructor interventions, the instructor's reaction and the label are not independent. Check before using it for H1.
- **Oops! contains only failures.** That makes it a failure-*localization* task. A visible success/failure set (e.g. sports shots) may be needed for a clean contrast.
- **BAD / ERR@HRI need an access request or data-use agreement.** There is lead time, so request early.
