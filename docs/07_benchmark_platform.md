# 07: SocialRobotics-Bench — Benchmark Platform ("SRB")

**Status: v0 IMPLEMENTED (July 14) — all design decisions resolved. The `bench/` toolchain is built and the 196-moment pilot kit set is cut; the maintainer rating round is the pending step (§7 v0 step 4).** The public name is **SocialRobotics-Bench** (Resolved #3); "SRB" survives as the internal shorthand and id prefix. This document is the grounding spec for the benchmark effort and is deliberately **separate from the engine pipeline** (docs/00–06). The engine extracts social features at scale; the benchmark measures whether *other people's models* can read social feedback as a reward signal. The two share tooling but must never share ground truth (§2).

---

## 0. Separation Contract: Benchmark vs. Engine

The engine (Nodes 01–05, Layers 02b/03x) keeps exactly **two roles** in the benchmark, both demoted from its role in training-data curation:

1. **Moment proposer** — the light front half (Node 02 social-presence detection + Layer 02b bystander-anchored climax segments) cuts *candidate moments* out of raw video. Humans do not scan raw timelines; MimeQA-style full manual scanning (annotators watched all 8 h of corpus for 806 QA pairs) does not scale to sparse naturalistic POV footage.
2. **Track-B signal serializer** — the 03x layer outputs are rendered into the *input text* of Track-B items (§1.3). They appear in the item's **question**, never in its **answer**.

**The engine's labels are never benchmark ground truth.** Golden labels come only from the human rating instrument (§4). Engine outputs are stored as a hidden column for the pipeline-vs-human agreement statistic (§6.4) and are **not shown to raters during test-set labeling** (anti-anchoring, §4.6).

### Interface boundary
The benchmark consumes one artifact from the engine and nothing else:

```
candidate_moments.jsonl        # one line per candidate moment
{
  "moment_id":      "srb-<corpus>-<clip_id>-<t_ms>",
  "corpus":         "ego4d | charades_ego | epic_kitchens | cc_web | upload",
  "clip_id":        "<source-native id>",
  "t_climax_sec":   214.5,
  "window_sec":     [211.5, 218.5],          # engine reaction window, advisory
  "source":         "engine | human_flag | upload_selfselect",
  "engine_prefill": { ... 03x channel values + confidences ... },   # HIDDEN from test raters
  "task_label_hint": "plays cards"           # advisory only, marked as machine-generated
}
```

Any future engine change only needs to keep emitting this schema. Benchmark code lives in a new top-level `bench/` directory (v0; split into its own repository at v1 when public artifacts appear) and never imports from `src/layer_*` except `shared/` video-cutting utilities.

---

## 1. What the Benchmark Measures

The framing: **the model is the camera wearer, mid-task. Social feedback arrives from bystanders. Can the model read it as a reward signal and let it steer behavior?** Four task families, in order of capability depth:

| Family | Capability | Question shape | Gold source |
|---|---|---|---|
| **F1 — Feedback reading** | Infer the valence of social feedback just received | 4-way MCQ | Rating Q-B5 (+ Q-B3/B4 for "no feedback") |
| **F2 — Credit assignment** | Decide whether an observed reaction is *about the wearer's action* at all | 3-way MCQ | Rating Q-B4 |
| **F3 — Preference ranking** | Given two (context, action) moments, pick the one that received the better social response | 2-way pairwise | Paired Q-B5 verdicts |
| **F4 — Feedback-conditioned behavior** | Given the feedback, predict what the wearer *actually did next* | 3-way MCQ | Rating Q-B7 |

Design notes carried over from the July planning discussions (docs/06):
- F2's hard negatives are exactly ambiguity-taxonomy case 3 (reaction attribution confound) — the class no automated layer can label, which is *why* the gold must be human.
- F3 is deliberately **relative**: pairwise preference is robust to per-channel label noise (noise partially cancels) and directly evaluates the reward-model use case. This mirrors Robometer/RoboReward's trajectory-ranking metrics, but for social response instead of task progress.
- F4 is **descriptive, not normative**: "what did the wearer do next" (observable, verifiable from the after-roll) — never "what *should* the agent do" (contestable). The causal claim ("because of the feedback") belongs in the paper's analysis, not the metric.

### 1.1 Item example (F1, Track A)
> *[video: 7 s clip]* The camera wearer points at the signboard while a companion stands nearby.
> **Q: Based on the visible people's response, what social feedback did the wearer's action receive?**
> (a) Approving / positive (b) Disapproving / negative (c) Neutral acknowledgment — noticed, no evaluative response (d) No reaction directed at the wearer

### 1.2 Item example (F2)
> *[video: 7 s clip]* A person in the clip reacts visibly (e.g., steps back, changes expression).
> **Q: Is this reaction a response to the camera wearer's action?**
> (a) Yes — responding to the wearer (b) No — responding to something/someone else (phone, third party, separate event) (c) Not enough evidence to tell

(c) is a **legitimate gold class** (from consensus "Unsure", §4.5) — models that never abstain lose points here. This is the calibration mechanism: over-reading social signal is the failure mode that matters most for social robotics.

### 1.3 Two input tracks (every item exists in both where possible)
- **Track A (pixels)**: the clip itself (shipped directly for CC/upload rings; via rehydration pointers for Ego4D, §3).
- **Track B (dehydrated signals)**: no pixels — the serialized 03x signals are the observation. Tests whether a model can *reason over an explicit social-reward channel* even when perception is solved for it. Track B has **no license constraints on any ring, including Ego4D** (it is dehydrated by construction, docs/04 rule), so it is always fully publishable. Example Track-B observation block:

```
[social signals, window 211.5–218.5 s]
bystander P3: attention 0.31→0.74 rising (conf 0.80); head gesture: nod ×2 (conf 0.71);
  emotion: neutral→smile (conf 0.55); proxemics: approach 1.9 m→1.1 m (conf 0.44)
ambient prosody: valence +0.31 (scope: ambient, conf 0.62)
wearer: 4 hand detections in action span; egomotion mean 2.1 / peak 6.8
wearer action: "points at the signboard"
```

A model failing Track A but passing Track B has a **perception** deficit; failing both is a **social-reasoning** deficit. Reporting the A/B delta per family is a headline result of the benchmark. (Precedent: SIV-Bench's subtitle-on/off variants.)

---

## 2. Contamination & Integrity Policy

Three distinct leaks, three distinct rules (all binding on every platform version):

| Leak | Description | Rule |
|---|---|---|
| **Pixel leakage** | Test video was in a VLM's pretraining crawl | Test items come from license-gated corpora (Ring 1), post-cutoff CC content (Ring 2), or unpublished uploads (Ring 3). Every wave records a **harvest date**; evaluators compare against model cutoffs. |
| **Label leakage** | Test items drawn from the same corpus whose annotations we publish for training | **Corpus-disjoint or split-stamped**: no clip in any published training artifact (HF datasets, qa_pairs.jsonl) may source a test item. Ego4D test clips require the docs/06 Issue-6 wearer-grouped split stamped *first*. Test answers are never published (sealed scorer, §7). |
| **Pipeline circularity** | Same automated annotator generates training labels and test answers → benchmark measures "agreement with our pipeline" | Test gold is **human-only** (§4). Engine outputs hidden from test raters. Pipeline-vs-human agreement is *reported*, never *assumed*. |

Additional integrity gates (automated, §5.4): blind-baseline filter (kills items answerable from text priors), valence balancing within action categories (kills "action caption predicts reaction" shortcuts), and control-item quota (kills "always predict a reaction").

**Wave lifecycle** (LiveBench-style): items are released in dated waves (`socialrobotics-bench-2026.4`, `socialrobotics-bench-2027.1`, …). A wave is *live* for a fixed period, then *retired*. Retired waves get their answers published and become ordinary training/analysis data — contamination-by-time turned into a feature. Leaderboard entries are pinned to the wave they ran on.

---

## 3. Data Source Rings

| Ring | Source | Pixel-leak risk | Can we ship pixels? | Effort | Version |
|---|---|---|---|---|---|
| **1 — Held-out licensed corpora** | Ego4D held-out split; Charades-Ego; EPIC-KITCHENS; (later Ego-Exo4D, HoloAssist) | Low (Ego4D gated) → medium (EPIC is widely trained on) | **No** — rehydration pointers only (Track A gated behind evaluator's own license); Track B fully public | Low — engine already ingests these by design (docs/00) | v0 |
| **2 — Fresh CC web video** | Newly-uploaded CC-BY egocentric video (GoPro POV vlogs, cooking-with-family, market walks, group activities), harvested by dated keyword sweeps (the SIV-Bench collection recipe) | **Zero at harvest time** (post-cutoff by construction) | **Yes** (CC-BY) — no rehydration wall | Medium — harvest adapter + license verifier | v1 |
| **3 — Community / lab uploads** | HRI labs and contributors upload consented clips, or staged scenarios (deliberately negative reactions, which naturalistic footage under-samples) | Zero (unpublished) | **Yes** (upload license grant, §8) | High — portal + consent + moderation | v1 (manual intake) / v2 (self-serve) |

Ring 1 doubles as a **cross-domain generalization** result (train on Ego4D, test on other corpora' scenes/cultures/rigs). Ring 2 is the contamination killer *and* the license liberator. Ring 3 is the only ring that yields deliberately staged hard cases.

---

## 4. The Human Rating Instrument (the centerpiece)

### 4.1 What needs human rating — summary

| Human task | Applied to | Per-item time | Raters | Stage |
|---|---|---|---|---|
| **Moment triage** (Q-A*) | Every candidate moment | ~10–15 s | 1 | A |
| **Golden labeling** (Q-B*) | Every triage-accepted moment | ~60–90 s | **v0: 1 (the maintainer — Resolved #1)**; ≥2 independent from the first public wave; 1 (train-split verification) | B |
| **Missed-moment flagging** (Q-C1) | Whole clips, opportunistically during B | ~0 (side effect) | — | B |
| **Upload moderation** (Q-M*) | Every Ring-2/3 clip before it enters the queue | ~30–60 s | 1 + escalation | pre-A |
| **Item QA spot-check** | 10 % random sample of templated items | ~20 s | 1 | post-templating |
| **Human baseline** | Final released items | full item time | 1 held-out rater (never saw the items in stages A/B) — first available at v1; impossible under the v0 single-rater regime | pre-release |

Everything else is automated: candidate mining, blind-baseline gate, valence balancing, dedup, pairing (F3), scoring.

### 4.2 The rating kit (what a rater sees per moment)

Generated by `bench/make_rating_kit.py` from `candidate_moments.jsonl` + local video access:

1. **Clip S** — the moment: `[t_climax − 4 s, t_climax + 3 s]`, **with audio**, ≤720p. (Static contact-sheet strips are insufficient for gold: nods, flinches, and prosody are invisible in stills. Strips remain the format for *triage* and for the Issue-4 train-split verification, where keep/drop granularity suffices.)
2. **Clip S+** — the after-roll: `[t_climax + 3 s, t_climax + 12 s]` — required for Q-B7 (what the wearer did next).
3. **Strip** — 8-frame contact sheet across S (fast triage; thumbnail in CSV/Label-Studio).
4. Context line: corpus, clip id, moment index within clip. The engine's `task_label_hint` is shown *grayed and marked machine-generated*; `engine_prefill` is **absent from test kits** (§4.6).

Ego4D licensing: kits contain source pixels → **internal-only**, exactly like Node-05 renders. Each Ring-1 rater either holds their own (free) Ego4D license and runs `make_rating_kit.py` locally, or works on the shared machine. Ring-2/3 kits carry no such restriction.

### 4.3 Stage A — Triage questions (verbatim)

> **A1.** Besides the camera wearer, can you see at least one person in this moment? — `Yes / No / Unsure`
> **A2.** Is the moment watchable? (not too dark, blurred, or occluded to judge people's behavior) — `Yes / No`
> **A3.** Flag any of the following: `minor is a focal subject / nudity or private context / identifiable sensitive info (screens, documents, plates) / person in distress / other (note)`

Routing: A1=No → drop (or → control-candidate pool if the engine proposed it as a `no_audience` control). A2=No → drop. Any A3 flag → quarantine for maintainer review (Ring 1: excluded from any published artifact; Ring 2/3: excluded from the benchmark entirely).

### 4.4 Stage B — Golden labeling questions (verbatim, with skip logic)

Shown beneath Clip S (+ Clip S+ for B7). One form per moment.

> **B1 — Wearer action.** *In this moment, what does the camera wearer do?* One short phrase, **verb first** (e.g., "points at the signboard", "hands over a card"). If the wearer is only talking/listening, enter `conversation`. If you cannot tell, enter `unclear`.
>
> **B2 — Audience.** *Is at least one visible person positioned to notice the wearer's action* (facing them, close by, or clearly within view)? — `Yes / No / Unsure`
>
> **B3 — Reaction occurrence.** *During this clip, does any visible person noticeably react* (change expression, gesture, move, vocalize)? — `Yes / No / Unsure`
> ↳ **If No or Unsure: skip B4–B6.** (B3=No with B2=Yes is a gold **"no reaction"** item — the control class.)
>
> **B4 — Directedness (credit assignment).** *What is the reaction most likely responding to?* — `The wearer's action / Something or someone else (phone, third party, separate event) / Can't tell`
> ↳ **If not "the wearer's action": skip B5.** ("Something else" is gold for F2 option (b); "Can't tell" is gold for F2 option (c).)
>
> **B5 — Valence.** *How would the wearer most reasonably read this reaction?* — `Approving / positive · Disapproving / negative · Neutral acknowledgment (noticed, no evaluative response) · Mixed or ambiguous`
>
> **B6 — Evidence channels** (select all that apply): `facial expression / head gesture (nod, shake, tilt) / hand or body gesture / movement toward or away (proxemics) / voice or sound / gaze or attention shift`
>
> **B7 — What happened next** *(shown only when Clip S+ exists and B3=Yes).* *Immediately after this reaction, does the wearer noticeably change what they are doing?* — `Continues the same activity / Adjusts or corrects the action / Stops or disengages / Can't tell`
>
> **B8 — Confidence.** *Overall, how confident are you in your answers for this moment?* — `Confident / Somewhat confident / Guessing`
>
> **B9 — Notes / flags** (optional free text): anything odd — wrong window, multiple simultaneous reactors, suspected staged content, etc.

> **C1 — Missed moment** (per clip, optional): *While reviewing this clip's moments, did you notice a clear social reaction the system did NOT propose?* If yes, note the approximate timestamp. → becomes a `source: human_flag` line in `candidate_moments.jsonl` (the escape valve for engine selection bias — these flagged misses are a valuable hard-case pool *and* a free engine-recall measurement).

Mapping to the docs/06 ambiguity taxonomy: B1 sentinels reproduce cases 1–2 (`conversation` → F1/Track-B eligible, excluded from F3/F4; `unclear` → excluded from all action-conditioned families); B4 operationalizes case 3; B3=No + B2=Yes covers the control classes of case 6; case 8 (caption describes a bystander, not the wearer) is prevented by B1's "camera wearer" phrasing plus the labeling-guide worked example.

### 4.5 Adjudication & agreement (test split)

Two regimes (Resolved #1, July 7): the v0 pilot runs single-rater; every public wave from v1 runs the full multi-rater protocol.

**v0 single-rater regime** (the maintainer is the sole rater — κ is unmeasurable with one rater, so reliability is measured as test–retest self-consistency):
- After a **≥ 3-day washout**, the rater re-rates a **random 20 % sample** of accepted moments blind (fresh kit order, original answers hidden).
- **Self-consistency gate**: ≥ 80 % exact categorical match on B3/B4/B5 between the two passes. Below the gate, the instrument (question wording, clip length) gets revised and the pass repeats — not the gate.
- Any moment answered `Guessing` (B8), and any moment whose two passes disagree on B3/B4/B5, is **dropped from test candidacy** (it may still serve the train split).
- Every pilot artifact (datasheet, HF card) must state plainly: *single-annotator golden labels (project author)* — pilot items are dev-grade, not citable gold.

**v1+ multi-rater regime** (binding for anything public and citable):
- Every test-split moment is labeled by **2 independent raters**.
- **Categorical fields (B2–B5, B7)**: exact match required. Disagreement → third rater adjudicates majority; no majority → item dropped from test (may still serve the train split).
- **B1 caption**: semantic match judged by the adjudicator (same action, different words = match). Sentinel/non-sentinel disagreement = hard disagreement → third rater.
- **B6 channels**: union of both raters (it feeds per-channel *reporting*, not item gold).
- Any rater answering `Guessing` (B8) → automatic third rater.
- Publish **Cohen's κ per field**; target κ ≥ 0.6 on B3/B4/B5 before any wave releases. If a wave misses the target, the instrument gets revised — not the target.

### 4.6 Anti-anchoring policy

**Test-split raters never see engine outputs** (no prefill, no channel suggestions) — prefilled forms anchor humans toward the model's answer, which would silently reintroduce pipeline circularity (§2 leak 3). Engine prefill **is** used in the train-split verification workflow (docs/06 Issue 4), where speed matters more than independence and gold status is not claimed. The engine-vs-human comparison happens *after* labeling, offline, and is published as the agreement statistic.

### 4.7 Rater onboarding

- **Labeling guide** (`bench/LABELING_GUIDE.md`): one worked example per ambiguity-taxonomy case (the docs/06 table maps 1:1 onto guide sections), plus counter-examples for the two sentinels.
- **Qualification set**: 15 moments with maintainer-established gold; ≥ 80 % categorical agreement to qualify. Re-qualification after any guide revision.
- Rater pool (Resolved #1/#4, July 7): v0 = the maintainer as sole rater; v1+ = **trusted collaborators only, manually invited** (GitHub-collaborator model). Paid annotation (Prolific) is out of scope unless a future budget decision reopens it.

### 4.8 Stage M — Upload moderation questions (Ring 2/3 only, verbatim)

> **M1.** Does the video plausibly match its claimed license / does the uploader plausibly hold rights? (CC-BY source link verifiable, or uploader attests they recorded it) — `Yes / No / Escalate`
> **M2.** Egocentric or near-egocentric POV with ≥ 1 visible non-wearer person? — `Yes / No`
> **M3.** Consent attestation present for identifiable people? (Ring 3 requirement, §8) — `Yes / No / N-A (Ring 2 public CC)`
> **M4.** Content flags (same list as A3) — `…`
> **M5.** Obvious spam / duplicate / synthetic-generated video? — `Yes / No`

Any `No/Escalate` on M1–M3 or flag on M4/M5 → rejected before entering the candidate queue.

---

## 5. From Golden Labels to Benchmark Items

All templating is deterministic (`bench/build_items.py`); no LLM writes items in v0/v1 (this keeps scoring objective and the item distribution auditable — LiveBench's "objective ground truth" principle).

### 5.1 Item templates (verbatim)

**F1 (valence, 4-way).** Gold from B5; the "(d) no reaction" class from moments with B2=Yes ∧ B3=No (agreed) or B4="something else".
> *The camera wearer {B1 caption, gerundized}. Based on the visible people's response in the clip, what social feedback did the wearer's action receive?*
> (a) Approving / positive (b) Disapproving / negative (c) Neutral acknowledgment (d) No reaction directed at the wearer

Eligibility: B1 not `unclear`; B1 `conversation` allowed (steady-state social reading is real signal — taxonomy case 1 routes it to perception-style items). B5="Mixed/ambiguous" items are excluded from F1 (unresolvable 4-way gold) but retained for F2.

**F2 (credit assignment, 3-way).** Gold from B4 (including "Can't tell" → option c).
> *A person in the clip reacts during this moment. Is their reaction a response to the camera wearer?*
> (a) Yes — it responds to the wearer's action (b) No — it responds to something or someone else (c) Not enough evidence to tell

**F3 (pairwise preference, 2-way).** Built by `bench/pair_items.py` from gold-confirmed B5 verdicts (per whichever §4.5 regime is in force):
> *Clips 1 and 2 each show the camera wearer performing a similar action. In which clip did the wearer's action receive the more positive social response?* — `Clip 1 / Clip 2`

Pairing rules: same action category (verb-cluster match on B1), different clips (never two moments of one clip — same bystander leaks), valences strictly ordered (approving > neutral > no-reaction > disapproving; pair only non-adjacent ranks in v0 to keep gold unambiguous), presentation order randomized, each moment reused in ≤ 2 pairs.

**F4 (what happened next, 3-way).** Gold from confirmed B7 (per the §4.5 regime); only moments with B7 ≠ "Can't tell".
> *[Clip S only — the after-roll is withheld.] The wearer has just received the reaction shown. What did the wearer actually do next?*
> (a) Continued the same activity (b) Adjusted or corrected the action (c) Stopped or disengaged

### 5.2 Track-B rendering
Same items with the observation block (§1.3) replacing/augmenting the clip. Serializer: `bench/render_track_b.py` reading `engine_prefill` (this is the one sanctioned use of engine output in items — it is the *question*, and its honesty is itself part of what Track B measures: the signals carry confidences, including wrong/low-confidence channels).

### 5.3 Class balance quotas (per wave, enforced by `build_items.py`)
- F1: each valence class ≥ 15 % of items; "no reaction" class 20–30 % (kills "always predict a reaction").
- Within each action category (verb cluster): at least 2 distinct gold classes present, or the category's items are capped at 3 (kills caption→answer priors).
- F2: "something else" + "not enough evidence" jointly ≥ 35 %.

### 5.4 Automated gates (run before human item-QA)
1. **Blind-baseline gate**: a text-only LLM answers each item from `{caption + question + options}`, 5 samples; drop items with blind accuracy ≥ 60 % (4-way) / ≥ 70 % (3-way) / ≥ 75 % (pairwise). Report the kill rate per wave.
2. **Dedup**: no two items from the same `(clip, bystander)` pair within a family.
3. **Near-duplicate video check** (Ring 2): perceptual-hash against previous waves.
4. *(v2 only)* **Model-in-the-loop hardening**: drop items every frontier model already answers correctly — but always retain a random un-hardened 30 % slice so the wave's difficulty distribution stays interpretable (SIV-Bench's consensus-filtering, minus its distribution-skew failure mode).

---

## 6. Scoring, Metrics & Reporting

1. **Per-family accuracy** (F1–F4), Track A and Track B separately; the **A/B delta** per family (perception vs. reasoning deficit, §1.3).
2. **Macro-averaged over gold classes** within F1/F2 (class quotas make micro ≈ macro, but report both).
3. **Social Hallucination Rate (SHR)** — headline safety metric: on gold-"no reaction"/"not directed" items, the fraction of model answers asserting evaluative feedback (F1 a/b, F2 a). The failure mode that matters for social robotics is *over*-reading approval.
4. **Pipeline-vs-human agreement** — engine prefill vs. human gold per channel (published honesty number; also the engine's free evaluation).
5. **Human baseline** — held-out rater on the released items (MimeQA's 86 %-vs-31 % gap is the model for headroom reporting). First available at v1: the v0 single-rater regime (Resolved #1) has no untainted rater.
6. **Per-channel breakdown** — model accuracy conditioned on B6 evidence tags ("models read proxemics but miss prosody" is the citable finding).
7. Scoring is deterministic string-match on option letters; no LLM judge in v0/v1.

---

## 7. Platform Versions

### v0 — "Contact-sheet benchmark" (extends docs/06 Issue 4; goal: validate the instrument)

**Scope**: Ring 1 only — held-out Ego4D clips (Charades-Ego declined for v0; Resolved #2). Maintainer as sole rater (Resolved #1). Output: a pilot wave (`socialrobotics-bench-pilot`) of ~100–150 golden-labeled items across F1/F2 (+F3 pairs if valence spread allows), Track B public on HF, Track A as sealed questions + rehydration pointers. **The go/no-go gate for everything after: test–retest self-consistency ≥ 80 % on B3/B4/B5 (§4.5 single-rater regime) and a blind-gate survival rate ≥ 50 %.**

Deliverables & steps (≈ 1–2 weeks of tooling + ~6–8 h of maintainer rating, incl. triage and the 20 % re-rate pass):
1. `bench/` scaffold + `candidate_moments.jsonl` exporter from an annotated manifest (thin shim over `expand_task_segments`; **depends on docs/06 Issue 6 wearer-grouped split stamping for the Ego4D held-out selection — blocking prerequisite**).
2. `bench/make_rating_kit.py` — cuts Clip S / Clip S+ / strip per moment (reuses Node-05's decode utilities; kits are internal-only for Ego4D, per §4.2). Kept remote-runnable for v1's trusted raters: a rater without local videos runs it against their own Ego4D download (the Issue-4 "run the rehydrater" selection).
3. `bench/LABELING_GUIDE.md` + 15-item qualification set (authored now; it gates future trusted raters from v1 — the v0 sole rater is its author, so it cannot gate them).
4. **Rating round**: Stage A triage → Stage B labeling, maintainer as sole rater, via **CSV** (`ratings_maintainer.csv`, columns: `moment_id, A1..A3, B1..B9, C1`; one row per moment; the strip/clips referenced by relative path). After a ≥ 3-day washout, blind re-rate of a random 20 % sample (`ratings_maintainer_retest.csv`) for the §4.5 self-consistency gate. No UI built.
5. `bench/adjudicate.py` — validates the CSV (enums, skip logic), applies the §4.5 single-rater rules (drops `Guessing` and retest-disagreement moments), emits `golden_labels.jsonl` + a self-consistency report. (The multi-rater merge/κ path is added at v1.)
6. `bench/build_items.py` + `bench/pair_items.py` + `bench/render_track_b.py` + blind gate (`bench/blind_gate.py`, local LLM) → `items_{family}_{track}.jsonl`.
7. `bench/score.py` — deterministic scorer; run 2–3 open VLMs (qwen2.5-VL via the existing registry client) + 1 frontier API model as anchor numbers. (Human-baseline pass deferred to v1 — no untainted rater exists under the single-rater regime.)
8. Write-up of instrument metrics (self-consistency, time-per-item, blind kill rate, engine agreement) → go/no-go on v1. **Time-per-item is measured deliberately: it is the deciding input for ⚠️ Issue 3's tooling choice.**

### v1 — "Hosted labeling + public waves" (goal: first citable public wave)

**Scope**: adds Ring 2 (CC web harvest) and manual Ring-3 intake; rating tooling per ⚠️ Issue 3; public leaderboard; `socialrobotics-bench-2026.x` wave with shipped pixels for Ring-2/3 items. **The ≥2-rater protocol resumes here** (Resolved #1 bounds the single-rater regime to v0). Benchmark code splits into its own repo.

Steps:
1. **Ring-2 harvest adapter**: dated keyword sweeps (LLM-generated queries per scenario, the SIV-Bench recipe) → CC-BY filter (API-verified license + archived snapshot of the license page) → download → Node-02/02b pre-pass → `candidate_moments.jsonl`. Harvest date recorded per clip (§2 pixel-leak rule).
2. **Rating tooling per ⚠️ Issue 3** (refreshed July 7 under the trusted-rater constraint — the original Label-Studio-vs-custom-UI framing assumed a managed rater pool that Resolved #1/#4 eliminated): git-native CSV scaled up, a local single-page rater, or a minimal Label Studio instance. Whichever is chosen must emit `golden_labels.jsonl` so **every v0 script downstream of adjudication is reused unchanged**; `bench/adjudicate.py` gains the multi-rater merge/κ path either way.
3. **Upload intake (manual)**: a simple form (HF Space or static form + object storage) collecting the clip, license grant, and consent attestation (§8); Stage-M moderation queue in Label Studio.
4. **Sealed scorer**: HF Space holding private gold; accepts predictions JSONL; per-model daily submission cap + aggregate-only feedback (answer-extraction resistance); leaderboard pinned per wave.
5. **lmms-eval / VLMEvalKit adapter** (`--tasks socialrobotics_bench`) — the accessibility step that decides whether anyone runs it.
6. **Wave assembly** (`bench/export_wave.py`): balance quotas, gates, datasheet (harvest dates, κ, blind kill rate, SHR definitions), HF dataset (Ring-2/3 pixels + all Track B), retirement schedule.
7. **Trusted-rater onboarding** (Resolved #4): raters are manually invited collaborators (GitHub-collaborator model), qualified via the §4.7 set; wave size and cadence scale to the trusted pool's available hours — there is no paid-annotation budget.

### v2 — "Community platform" (goal: self-sustaining refresh)

Build **only if** v1 produces external demand (leaderboard submissions from groups we don't know, unsolicited upload interest). Adds:
1. Self-serve upload portal with accounts, contributor credits on the dataset card, and an upload-status tracker.
2. **Staged-scenario program**: partner HRI labs record scripted interactions (deliberately disapproving reactions are the under-sampled class in all naturalistic rings); a scenario playbook doc specifies coverage targets by valence × channel × setting.
3. Model-in-the-loop hardening (§5.4.4) with the 30 % random slice.
4. Refresh cadence sized to trusted-rater capacity (target 1–2 waves/year; Resolved #4 rules out paid bulk labeling), retirement automation (answers auto-publish on retirement; retired waves merge into the public training corpus — closing the loop with the engine's HF surface).
5. Governance: takedown SLA, wave patch releases (scorer re-versions when an item is removed).

### Version comparison

| | v0 pilot | v1 public | v2 community |
|---|---|---|---|
| Rings | 1 | 1 + 2 (+3 manual) | 1 + 2 + 3 self-serve |
| Labeling | maintainer-only CSV kits (single rater) | ⚠️ Issue 3 tooling, 2-rater overlap (trusted pool) | + contributor program (trusted, manually added) |
| Items | ~100–150, F1/F2 (±F3) | ~800–1,500, F1–F4, dated wave | waves ×2/yr, hardened |
| Pixels shipped | No (Track B public; Track A via rehydration) | Yes for Ring 2/3 | Yes |
| Eval access | scripts in repo | sealed scorer + lmms-eval adapter | + leaderboard archive |
| Gate to next | self-consistency ≥ 80 %, blind-survival ≥ 50 % | external submissions exist | — |

---

## 8. Consent, Licensing & Governance

- **No face blurring, ever**: facial reaction *is* the measured signal, so anonymization-by-blur is impossible. Consequence: consent must be real, not cosmetic.
- **Ring 1**: source-dataset licenses govern; pixels never shipped (Track A rehydration-only), consistent with the docs/04 dehydration rule.
- **Ring 2**: CC-BY only, license snapshot archived at harvest; attribution file shipped with each wave; takedown honored regardless of license.
- **Ring 3 upload grant** (click-through, drafted at v1): uploader (a) affirms they recorded the video or hold redistribution rights, (b) grants a CC-BY (or bespoke research-redistribution) license, (c) attests every identifiable person consented to research use and redistribution, (d) confirms no focal minors. Stage-M moderation independently screens all four.
- **Takedown**: published contact + removal within a stated SLA; scorer re-versioned (wave patch) so leaderboard comparability is preserved.
- **Audio**: prosody is a measured channel → audio ships with Ring-2/3 clips; consent language covers voice explicitly.

---

## 9. Build Order & Cross-Doc Dependencies

```
[engine, done]      02b segments + controls + captions ──┐
[engine, done]      03x layers + segment_dataset join ───┼─→ candidate_moments exporter (v0.1)
[docs/06 Issue 6]   wearer-grouped split stamping ───────┘        │ (blocking for Ring-1 test selection)
[docs/06 Issue 4]   contact-sheet verification ──(train split only; shares kit tooling)
                                                                  ↓
                    v0 instrument pilot ──self-consistency gate──→ v1 public wave ──demand gate──→ v2
```

The full-991 run (in flight, docs/06) feeds the *training* surface and the engine-agreement statistic — it is **not** on the benchmark's critical path except as the corpus from which Ring-1 held-out clips are excluded.

---

## 🧪 Resolved Issues & Implementation Refinements

These are **design decisions**, not code fixes — "Solution" below records the selection, the design changes propagated through this document, and the consequences the selection binds. Original issue numbering is preserved in each title (docs/06 convention).

1. **Issue 1: v0 Rater Pool — Single-Rater Golden Labels (Resolved - July 7)**:
   - **Problem**: The rating instrument (§4.5) mandated two independent raters per test moment with a Cohen's-κ ≥ 0.6 release gate, which made the pilot's schedule hostage to rater recruitment; the drafted options were two project members (correlated errors inflate κ) vs. one member plus external collaborators (Ego4D license + download latency per rater).
   - **Solution** (user-directed — neither drafted option): *"I will be doing most of the rating myself so lets just say one rater is enough to provide the golden labels for now."* The maintainer is the **sole v0 rater**. Design consequences propagated: (1) κ is mathematically unmeasurable with one rater, so the v0 reliability measure and go/no-go gate were **replaced with test–retest self-consistency** — blind re-rate of a random 20 % sample after a ≥ 3-day washout, ≥ 80 % exact categorical match on B3/B4/B5 (§4.5 "v0 single-rater regime"; §7 v0 gate; §9 diagram). (2) Moments answered `Guessing` or disagreeing between the two passes are dropped from test candidacy. (3) Every pilot artifact must be labeled *single-annotator golden labels (project author)* — pilot items are dev-grade, not citable gold. (4) The v0 human-baseline pass is deferred to v1 (no untainted rater exists). (5) The *"for now"* in the selection is encoded as a **v0-only regime**: the ≥ 2-independent-rater protocol resumes at the first public v1 wave (§4.5 "v1+ multi-rater regime") — a citable public wave never ships single-rater gold.

2. **Issue 2: v0 Corpus Scope — Ego4D Held-Out Only (Resolved - July 7)**:
   - **Problem**: Undecided whether the pilot validates only the rating instrument (Ego4D held-out clips) or also attempts the first cross-corpus claim by adding Charades-Ego — whose ~30 s scripted clips structurally break Layer 02b's clustering thresholds (e.g. the ≥ 30 s bystander-free gap a `no_audience` control requires can never occur), meaning engine re-tuning smuggled into an instrument-validation milestone.
   - **Solution** (Option A, selected July 7): **Ego4D held-out clips only** for v0. The `candidate_moments.jsonl` exporter scope is fixed accordingly (§7 v0 step 1), and docs/06 Issue-6 wearer-grouped split stamping remains the single blocking engine prerequisite. Charades-Ego (and any Ring-1 sibling corpus) is revisited at v1 alongside the Ring-2 harvest work, with 02b threshold re-tuning recorded as its prerequisite.

3. **Issue 4: Public Benchmark Name — SocialRobotics-Bench (Resolved - July 7)**:
   - **Problem**: The permanent public identifier (HF slugs, eval-harness task name, paper title) was a placeholder ("SRB / Social-Reward-Bench"), and it blocks the *first* public artifact — v0's Track-B HF publish — because HF dataset renames break every `load_dataset()` call and fragment citations.
   - **Solution** (user counter-proposal, selected July 7): the benchmark is named **SocialRobotics-Bench**. Slugs pinned now: HF dataset `<org>/socialrobotics-bench`, eval-harness task `socialrobotics_bench`, waves `socialrobotics-bench-<year>.<quarter>`, pilot wave `socialrobotics-bench-pilot`. "**SRB**" is retained as the internal shorthand and `moment_id` prefix (the acronym still reads as **S**ocial**R**obotics-**B**ench). Names updated throughout this document (title, §2 wave lifecycle, §7). One caveat recorded: the name shares its surface with the generic field term "social robotics", so discoverability rides on the hyphenated compound — verify the HF namespace is free at org-creation time before the first upload.

5. **Issue 3: Rating Tooling — Local Single-Page Rater (Resolved - July 14)**:
   - **Problem**: The v0 rating round shipped as a CSV-only workflow (§7 v0 step 4): the rater juggles 196 video files against spreadsheet rows, enum typos surface only at batch validation, and the §4.4 skip logic is enforced after the fact rather than at entry. The refreshed options (A.1 git-native CSV / A.2 local single-page rater / A.3 minimal Label Studio) carried an **adoption rule** deferring A.2 until v0 measured time-per-item > ~75 s.
   - **Solution** (Option A.2, user-directed July 14 — *"Can you create a UI for me to perform these ratings on? … My clicks can write to a spreadsheet if necessary"*): built now, ahead of the adoption rule; the user's direct request supersedes the measurement trigger, and the measurement it was waiting on (time-per-item) is now captured automatically by the tool itself. Implemented as `bench/rater.html` + `bench/rate_server.py` (stdlib-only, binds 127.0.0.1): video and Stage-A/B form in one view, keyboard-first answering, §4.4 skip logic enforced at entry (skipped fields are written **blank**, never stale), autosave per answer, resume-on-first-unrated, and automatic `seconds_spent` capture (first completion preserved across re-edits). `--retest` serves the washout kit and writes the retest CSV **without loading round-one answers** (blindness is structural, not procedural). Two deviations from the drafted option, both forced by browser reality: (1) a local HTTP server rather than bare `file://`, because pages cannot write local files and Safari refuses video without HTTP range support — the server implements 206 ranges; (2) it therefore ships as two files, not one. **The workflow is unchanged**: the UI is a front-end over the *same* `ratings_maintainer.csv` (identical `RATING_COLUMNS`), so `adjudicate.py --validate/--make-retest/--finalize` and every downstream script consume its output unmodified, spreadsheet editing still works, and the git-native audit trail of A.1 is preserved. Anti-anchoring (§4.6) is enforced by construction: `engine_prefill` is never serialized to the page (verified — the state payload carries no prefill key); only the greyed machine-generated caption appears, per §4.2. Verified end-to-end in-browser on the real 196-moment kit set: skip-logic cascade (B3→B4→B5), multi-select, save → CSV → `adjudicate.py --validate` clean, auto-advance to next unrated. Tests: `tests/test_bench_v0.py` gains 2 UI-contract tests (every UI-shaped row passes `validate()`; store round-trip preserves first timing) — suite 246/246. Consequence for v1: A.3 (Label Studio) is now moot for the trusted-pool scale, and the same tool serves v1 by pointing it at a per-rater CSV.

4. **Issue 5: Annotator Sourcing — Trusted Collaborators, Manually Added (Resolved - July 7)**:
   - **Problem**: A v1 wave of ~800–1,500 items × 2 raters × 60–90 s/item ≈ 40–75 h of Stage-B labor, forcing a decision between unpaid collaborators (stall risk), Prolific-paid raters (~$1.1–1.5 k/wave, web-servable kits required), or a hybrid.
   - **Solution** (Option A variant, selected July 7): *"I will manually add raters that I trust as collaborators."* The rater pool is **trusted collaborators only, manually invited** (GitHub-collaborator model); there is **no paid-annotation budget**. Consequences propagated: (1) wave size and cadence scale to the trusted pool's available hours — a wave shrinks or slips rather than lowering the 2-rater bar (§7 v1 step 7; v2 cadence target softened to capacity-scaled 1–2 waves/year). (2) All Prolific/paid-pool language removed (§4.7, §7, version table). (3) The web-servable-kit requirement that paid raters would have forced is dropped, and — because every rater is now trusted — account management, task queueing, and rater-pool tooling fall out of scope entirely, which collapsed the premise of the original Issue-3 options (see refreshed ⚠️ Issue 3 below).

## ⚠️ Unresolved Issues & Suggestions

None open. Issues 1, 2, 4 and 5 were resolved July 7; Issue 3 (refreshed July 7) was resolved July 14 — see 🧪 above.

---

## Cross-references
- docs/00 §Pipeline (the engine this platform deliberately does *not* depend on for gold)
- docs/02b (moment proposer), docs/03 §Multi-Window Segments (trajectory basis for F4)
- docs/04 (dehydration rule → Track B's unconditional publishability)
- docs/06 Issue 4 (train-split verification workflow; shares kit tooling with §4.2), Issue 5 taxonomy (→ §4.4 routing, §5.1 eligibility), Issue 6 (split stamping — blocking prerequisite for Ring-1 test selection)
- External precedents: SIV-Bench (collection recipe, subtitle ablation), MimeQA (human-verified small-N credibility, CC sourcing), RoboReward/Robometer (reward-model metric suite), LiveBench/Dynabench (wave refresh, objective scoring, model-in-the-loop), MMToM-QA/EgoSocialArena (simulation route — considered and not taken: sim-real gap in social nuance defeats the purpose).
