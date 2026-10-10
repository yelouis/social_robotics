# 02: Data Sources

Every source must provide three things: an **action**, a **reaction** from someone watching (or tasting, or supervising), and an **outcome label that does not come from a human rating the reaction**. Without an outcome label a source is unusable here, however many reactions it contains. That is the v0 lesson ([`LESSONS_v0.md`](LESSONS_v0.md)).

| Source | Outcome visible in the action? | Reaction channel | Outcome label | Role | Access |
|---|---|---|---|---|---|
| **Verdict videos** (web) | **Hidden** | Taster's face + voice before the verdict | Spoken verdict ("7 out of 10"), parsed from ASR | H1 core; H2 training corpus and scaling curve | Web; sourcing decision pending (Issue 1) |
| **HoloAssist** | **Hidden / partly visible** | Remote instructor's voice | Mistake annotations per action segment | H1 first-person; H2 target | Public download ([site](https://holoassist.github.io)) |
| **Oops!** | **Visible** | Filmer's audio (laughs, gasps), visible spectators | Failure-onset timestamp | H1 visible-outcome contrast | Public download ([site](https://oops.cs.columbia.edu/data)) |
| **BAD** / **ERR@HRI 3.0** | Visible (stimulus video) | Webcam face of the viewer | Failure vs. control stimulus | H2 target: reactions *to robots* | Controlled access: an IRB-reviewed protocol from an affiliated institution (Issue 4) |
| **REACT** | — | Reactions to robots + explicit feedback | Explicit evaluative feedback | H2 target candidate | Check availability |
| **Ego4D v0 corpus** (archived) | — | Bystanders | **None** | Optional: false-positive rate on steady-state co-activity | Derived data on the SSD; raw videos deleted October 8 (re-download by id) |

---

## Verdict videos: the core hidden-outcome source

**What they are:** taste tests and reviews where a person tries something, reacts, and *then* states a verdict: food reviews that score dishes out of 10, "trying X for the first time", drink and snack rankings. The reaction comes before the verdict, and the verdict is the label. This format is extremely common, which is what makes the H2 scaling curve possible.

**Why it is the right first test:** the outcome (how good the food was) is **not visible** in the action frames. A VLM judge can see the dish and its name and guess from priors, but it cannot taste. If reactions carry reward information anywhere, they carry it here. It is also the robotics pitch: a robot serving food cannot taste it.

**Labeling pipeline (fully automatic):**
1. **ASR** with word timestamps (Whisper; `mlx-whisper` on Apple Silicon).
2. **Verdict extraction**: a regex for `N/10`, `N out of 10`, `N stars`, with an LLM fallback for phrasings like "solid eight". Map each verdict to its item when one video reviews several (segment by item mentions in the transcript).
3. **Reaction window**: from the item's first taste to the start of the verdict utterance, minus a safety margin. First-taste detection starts as a simple transcript and visual heuristic, tuned in the Wave-B pilot.
4. **Label**: binary high/low (e.g. ≥ 8 vs ≤ 5 out of 10, middle excluded) for AUROC, plus the raw score for rank correlation.
5. **Code QA**: check verdict parsing and windowing on a random sample of transcripts. This tests our parser, not the meaning of any label.

**Leakage and confounds (each is a reported condition or control, not an afterthought):**
- **Words inside the reaction window** ("oh this is amazing") partly state the verdict. Report three conditions: face only, non-verbal audio (paralinguistic embeddings, no transcript) and full audio + transcript. The claim we care about most is face plus non-verbal audio.
- **Reviewer identity.** Some reviewers rate everything high. Use group splits by channel, and add a within-reviewer analysis that ranks items for the same person (the offline analogue of stage A personalization).
- **Item priors.** Some foods are polarizing. That is exactly what the VLM-judge control absorbs: it sees the item and its name.

## HoloAssist: first-person, hidden outcome

An egocentric performer (HoloLens camera) completes physical tasks while a remote instructor watches the live feed and talks them through it.

**Facts (checked October 8, 2026; re-verify on download):**
- **Scale:** 166 h, 350 instructor–performer pairs. The homepage now says 169 h.
- **License:** "CDLAv2", described on the homepage as permissive. Confirm the exact variant on download.
- **Annotations:** the paper says each **fine-grained action carries a mistake/correct attribute**, and each **utterance carries a purpose label** (the type of verbal intervention). Raw and processed annotation formats are released, with train/val/test splits.
- **Official download sizes:**

  | Component | Size |
  |---|---|
  | Labels | 111 MB |
  | Pitch-shifted videos | 184.20 GB |
  | Compressed videos (width 256) | 144.62 GB |
  | Depth, hand pose, gaze, IMU, calibration | ≈ 800 GB (not needed) |

- **The audio is pitch-shifted** for anonymization. Paralinguistic features are therefore measured on altered voices; record this as a caveat in every HoloAssist result.
- **Unknown until downloaded:** the exact JSON field names; whether the instructor's voice is audible in the video's audio track; whether the official splits are participant-disjoint.

**Item definition (Wave A):**
- One item per annotated fine-grained action that carries a mistake/correct attribute.
- `label` = 1 if correct, 0 if mistake.
- `action_window_sec` = the action's `[start, end]`.
- `reaction_window_sec` = `[start, end + 5.0]`, clipped to the video duration.
- `context_text` = `"Task: <task name>. Step: <verb> <noun>."` from the annotation. **Never the mistake attribute or any utterance purpose label.**
- `group_id` = the performer identifier if the metadata has one, else the session id.
- `react-spoke` = 1 if any instructor utterance overlaps the reaction window. `react-full` = the concatenated text of those utterances, if the annotations carry transcripts.
- **Sampling:** all mistake items plus an equal number of correct items (`default_rng(0)`), capped at 1,500 per class in train and 1,000 per class in test.

**Independence check (Wave A item A4, before any video download):** if the annotation protocol says mistake labels were assigned *from* the instructor's interventions, reaction and label are not independent and H1 on HoloAssist would be circular. If so, stop and file it. The intervention correlating with mistakes is expected and is the signal. *Deriving the label from it* is the problem.

## Oops!: visible-outcome contrast

Web "fail" videos with a marked failure onset. The reaction is mostly the filmer's audio: a laugh, a gasp, "oh no". Some clips carry compilation music instead.

**Facts (checked October 8, 2026; re-verify on download):**
- **Scale:** 20,723 clips from YouTube fail compilations, 50+ h.
- **Labels:** failure onsets marked by 3 Mechanical Turk workers each (median standard deviation ≈ 0.5 s).
- **Download:** a single videos + annotations bundle of **45 GB**. Optical-flow frames (1,019 GB) are not needed.
- **License:** non-commercial research/educational use, **CC BY-NC-SA 4.0**.

**Item definition (Wave A):** every clip is a failure, so the task is **localization**. Each usable clip yields two items:
- **The failure time** `t` is the median of the annotated onsets. Skip the clip if the onsets are missing, if their standard deviation is > 1.0 s, if `t − 4.0 < 0`, or if `t + 3.0 >` the clip duration.
- **`pre`:** window `[t − 4.0, t − 1.0]`, `label` = 1 (still going as intended).
- **`post`:** window `[t, t + 3.0]`, `label` = 0.
- `action_window_sec` = `reaction_window_sec` = that window.
- `context_text` = `"A short clip from a home video."` for every item. **The dataset's natural-language descriptions are never used**; they describe the failure.
- `group_id` = the source compilation id if it can be derived from the clip filename, else the clip id.
- **Split:** the official train split for fitting and the official val split as `test`.
- **Caps:** 2,000 clips (4,000 items) in train and 1,000 clips (2,000 items) in test, sampled `default_rng(0)`.

The VLM judge should be strong here, because the failure is visible. That is the point: it is the condition where we *predict* reactions add little. Spectators visible in frame cannot be masked from the judge. That biases the comparison toward the judge, which is conservative for our claim.

## BAD / ERR@HRI / REACT: reactions to robots

**If BAD access is granted, these commitments from the access request bind the project:**
- BAD is a held-out evaluation set only, with no training on it;
- it is stored only in an encrypted volume on the maintainer's personal hardware (never employer devices, never cloud sync);
- it is processed **locally only**. Frames are never sent to any cloud service or API, **including the Gemini frontier judge**;
- only aggregate metrics are published; never frames, face crops, embeddings or per-participant results;
- the data and everything derived from it are destroyed at study end, or 12 months after download, whichever comes first, and QDR is notified.

These are the only sources where the reactions are *to a robot*. They are H2 targets, never training data, so that "trained on web video, transferred to robots" stays a clean zero-shot claim. **BAD requires an IRB-reviewed protocol from an affiliated institution**, which the maintainer (unaffiliated) lacks. The access request asks whether an exception is possible, and **Issue 4** holds the fallback: our own consented BAD-style reaction study.

## Ego4D v0 corpus (archived; optional reuse)

991 clips, 23,378 segments, already on the SSD. It has no outcome labels, so it cannot test H1 or H2. It is still useful as a **false-positive set**: mostly steady-state co-activity (cleaning, cards, groceries) where no evaluative reaction occurs. A reward model that fires confidently here is hallucinating social feedback. Do not extend it.

## Licensing policy

- Each dataset's own license governs it.
- Web video follows the v0 **dehydration rule**: never redistribute pixels we do not own. Releases contain video IDs, timestamps, labels and derived features only.
- How web video may be *downloaded* for research is **Issue 1** in [`ongoing_general_errors.md`](ongoing_general_errors.md).
- Raw video lives only on the external SSD under `DATA_ROOT` (`src/config.py`).
