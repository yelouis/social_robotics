# 06: VLA Fine-Tuning & Benchmark Curation Roadmap

## Objective
The corpus carries **no action tokens**, so it cannot feed a VLA policy via behavior cloning directly. It can still contribute through three routes, all of which the July 2 planning discussion identified as viable with the data we already produce:

1. **Social reward / outcome model** — each reaction segment is a *(context → wearer action → bystander response)* triplet; a model mapping (context, action) → the measured layer signals is a reward model usable for RL fine-tuning, trajectory ranking, or success detection.
2. **VQA co-training corpus** — RT-2/π0/GR00T-style co-training of the VLM backbone on non-action vision-language data; our per-segment labels render mechanically into QA pairs ("Is the bystander attending?" → 03a; "How did they react?" → 03b/03e).
3. **Latent / pseudo-action route** — extract the wearer's own actions (hands, ego-motion) as pseudo-action tokens, giving (obs, action, social outcome) tuples for offline RL later.

Plus a two-tier **benchmark**: Tier 1 = social *perception* (held-out segments → questions scored against layer labels); Tier 2 = social *outcome prediction* (context + action shown, model predicts the bystander's reaction before the window is revealed).

The issues below are the pipeline changes needed to curate the right data for those goals. **Issues 1–3 are decided and sequenced** (July 2): implement #1 first, then #2, then #3 — ideally all three before the full-991 Layer-02b re-annotation so one pass over the corpus produces captioned, control-balanced, joinable segments. Issues 4–6 are pending review.

---

## 🧪 Resolved Issues & Implementation Refinements

1. **Per-segment action captions (Resolved - July 2)**:
   - **Problem**: `task_label` is Ego4D scenario metadata at whole-video granularity, and the June 30 spot-check showed how coarse that is ("Hiking" on a street-market scene, "Eating at a restaurant" on a creek crossing). No field recorded **what the wearer actually did at a given climax**, and every VLA use above (QA pairs, outcome prediction, reward modeling) needs the action grounded per segment, not per video.
   - **Solution** (Option A, selected July 2): VLM captioning pass in Layer 02b (`_caption_segment`, `src/layer_02b_task_climax/pipeline.py`). Two frames around each climax (`[−1 s, +0.5 s]`, downscaled ≤1024 px) go to the registry `filtering_vlm` (qwen2.5-VL via `shared/vlm_client.ollama_chat` with its enforced timeout) with a constrained prompt (deterministic `temperature 0`, `num_predict 48` — the smoke test saw an uncapped decode spend 25 s explaining its way to "unclear"); result recorded as `segment_action_caption`. Two sentinel escapes prevent hallucination — `conversation` (talking/listening only, the Issue-5 brainstorm case) and `unclear` — with salvage for verbose answers ending on a sentinel. The retired flow-era `vlm_model`/`skip_vlm` kwargs are repurposed: every pre-existing caller passes `skip_vlm=True`, so the lazy back-compat path **never** makes VLM calls; only the explicit pipeline/CLI captions (`--no-captions` / `SR_02B_ACTION_CAPTIONS=0` to disable). Failure is per-segment isolated (field absent, everything else intact). Validated on real clips: "bends down towards another person", "points at the signboard", "shuffles cards", correct `conversation`/`unclear` sentinels. **Measured cost: ~12 s/segment cold** (7B VLM, single GPU, serialized; supersedes the pre-implementation "seconds per clip" estimate) → ~4–5 min/clip dense, ~2–3 GPU-days for the full 991 — resumable, and reducible if needed via the `small` tier (qwen2.5vl:3b), 1-frame sampling, or capping captioned segments per task; that trim decision belongs to the 991 run. Suite 222/222.

## ⚠️ Unresolved Issues & Suggestions

---

### Issue 2: Unified per-segment export + QA-pair emitter (PRIORITY 2 — selected July 2)
**Status**: ⚠️ Confirmed Unresolved — Layer 04 exports **per-layer** datasets only. A fine-tuning row or benchmark item is one record per `(video_id, task_id, segment_index, person_id)` **joined across 03a–03f**, carrying each channel's confidence (`segment_face_px`, `proxemic_confidence`, `interpolated_fraction`, `audio_present`) so curation can filter per channel. Today every consumer would have to re-implement that join, including the subtle parts (distinct-window dedup, person alignment, null-vs-unmeasured).

**Option A (recommended)**: **New export mode in Layer 04** — `build_segment_dataset`: performs the cross-layer join once, emits one row per (clip, task, segment, person) with per-channel values + confidences + explicit null-reasons, plus a **QA-pair renderer** that mechanically converts joined rows into co-training text (question templates per channel; answers from the labels).
  - *Pros*: The join logic (dedup, re-anchor window matching, negative-id filtering) lives once, next to the code that already knows those rules; QA pairs become a free second output; benchmark items are reproducible from a pinned export.
  - *Cons*: A second export path to maintain alongside the per-layer one; schema versioning needed from day one (additive-only, like the manifest contract).

**Option B**: **Document the join keys only** and leave joining to consumers.
  - *Pros*: No new code.
  - *Cons*: The dedup/re-anchor subtleties (docs/03 guardrails) get re-implemented — and gotten wrong — per consumer; benchmark reproducibility suffers.

Your selection: **Option A — implement SECOND.**

---

### Issue 3: Control (negative) segments (PRIORITY 3 — selected July 2)
**Status**: ⚠️ Confirmed Unresolved — Layer 02b samples **only bystander-dense moments** by design, so the dataset is all-positive. A reward model or benchmark built on it degenerates to "always predict a social reaction." Verified structurally: every emitted segment comes from a qualifying bystander cluster.

**Option A (recommended)**: **02b emits flagged control segments** (`is_control: true`, capped at ~2–3/task): (a) *present-but-unreactive* — windows inside a bystander cluster but far from the chosen climax; (b) *action-without-audience* — windows sampled from the spans 02b currently discards (no qualifying cluster). Controls are **measured by the 03x layers like any segment** — that measurement IS the negative label ("bystander present, no reaction").
  - *Pros*: Reuses the existing clustering machinery; additive schema; gives the benchmark discriminative power and the reward model calibrated negatives.
  - *Cons*: ~30 % more 03x compute (controls must be measured to be labeled); consumption helpers need an explicit contract for whether `expand_task_segments` yields controls (must default in a way that doesn't silently change existing layers).

**Option B**: **Mine negatives post-hoc at training time** from raw detections.
  - *Pros*: No pipeline change.
  - *Cons*: Negatives would carry no measured layer signals (unverified "nothing happened"); benchmark items not reproducible from the export.

Your selection: **Option A — implement THIRD.**

---

### Issue 4: Human-verified eval split (pending review)
**In one sentence**: our labels are model-generated, which is fine for training bulk but not for a citable public benchmark — the test split needs a human to confirm each item.

**What it looks like**: after Issues 1–3, export candidate eval items (a few hundred), review them in one sitting, store a `human_verified: true/false` verdict that Layer 04 carries into the benchmark export.

**Option A (recommended)**: **Contact-sheet + CSV workflow** — render each candidate segment as a frame strip (exactly like the June 30 spot-check strips), review as images, record verdicts in a CSV the export joins on.
  - *Pros*: Zero UI work; a few hundred items ≈ one focused session.
  - *Cons*: Coarse verdicts (keep/drop), no per-channel correction.

**Option B**: **Interactive Layer-05 extension** — click-through verdicts per channel in the visualizer.
  - *Pros*: Finer-grained labels; reusable QA tool.
  - *Cons*: Real UI work; overkill if verdicts are mostly keep/drop.

Your selection: Yes, lets build this but I currently do not have the time to review this. Ideally this should be optional for now. However, when we get to this stage, I would like to see the Contact-sheet + CSV workflow. Ideally someone working on this project with me who pulls from github would be able to perform this review easily. So anything required for this should not require the locally downloaded videos. Maybe this will have to force someone else to run the reehyrdater script when reviewing?

---

### Issue 5: Wearer pseudo-action features (pending review)
**In one sentence**: for the latent-action VLA route, each segment should also record what the **wearer** did — hands and ego-motion — so (observation, action, outcome) tuples can be built later without re-decoding the corpus.

**What it looks like**: three additive per-segment fields — wearer hand trajectories (Node 02 already emits `hand_detections`; just re-cut to the window), an ego-motion summary over `[climax−2 s, climax+1 s]`, and 03f's ego-kinetic features re-cut to segment windows.

**Option A**: **Small additive fields now** (in 02b/03f) so the 991 pass captures them.
  - *Pros*: Cheap (data mostly exists); avoids a second full-corpus decode later.
  - *Cons*: Speculative — schema chosen before the latent-action model that will consume it.

**Option B**: **Defer** until a latent-action approach is actually selected.
  - *Pros*: No speculative schema.
  - *Cons*: A future 991-scale re-decode just to add these fields.

Your selection: Proceed with Option A. However, brainstorm the cases where we are unsure what the action in that clip was doing and what the outcome was. Maybe the person was just chatting in that clip.

---

### Issue 6: Split hygiene + richness metadata (pending review)
**In one sentence**: decide train/val/test membership **per clip (ideally per wearer) now, before more data is published** — segment-level splits would leak (the same wearer/scene on both sides) — and score each segment's "richness" so training mixes can be balanced.

**What it looks like**: a one-shot tool stamps each manifest entry with `benchmark_split: train|val|test` (grouped by Ego4D wearer/scenario metadata so no wearer straddles splits), and Layer 04 computes a per-segment `richness` count (#channels with confident signal) used for balanced sampling and for targeting future acquisition (current labels skew heavily to cards/screens/walking).

**Option A**: **Stamp splits in the manifest now** + richness at export.
  - *Pros*: Leakage impossible by construction for everything published afterward; richness is a cheap by-product of Issue 2's join.
  - *Cons*: Split policy locked early; changing it later invalidates comparisons.

**Option B**: **Assign splits at publish time** per dataset.
  - *Pros*: Flexible.
  - *Cons*: Every publish re-derives splits; drift between datasets makes cross-dataset comparisons leak-prone.

Your selection: We don't have to implement this now until the 991 videos have completed. Once it completes, maybe we will discover something else. Maybe note that I am leaning towards Option B.
