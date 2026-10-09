# 02: Data Sources

Every source must provide three things: an **action**, a **reaction** from someone watching (or tasting, or supervising), and an **outcome label that does not come from a human rating the reaction**. Without an outcome label a source is unusable here, however many reactions it contains. That is the v0 lesson ([`LESSONS_v0.md`](LESSONS_v0.md)).

| Source | Outcome visible in the action? | Reaction channel | Outcome label | Role | Access |
|---|---|---|---|---|---|
| **Verdict videos** (web) | **Hidden** | Taster's face + voice before the verdict | Spoken verdict ("7 out of 10"), parsed from ASR | H1 core; H2 training corpus and scaling curve | Web; licensing decision pending ([00](00_thesis.md) open questions) |
| **HoloAssist** | **Hidden / partly visible** | Remote instructor's voice | Mistake annotations per action segment | H1 first-person; H2 target | Public download ([site](https://holoassist.github.io)) |
| **Oops!** | **Visible** | Filmer's audio (laughs, gasps), visible spectators | Failure-onset timestamp | H1 visible-outcome contrast | Public download ([site](https://oops.cs.columbia.edu/data)) |
| **BAD** / **ERR@HRI 3.0** | Visible (stimulus video) | Webcam face of the viewer | Failure vs. control stimulus | H2 target: reactions *to robots* | Request + data-use agreement |
| **REACT** | — | Reactions to robots + explicit feedback | Explicit evaluative feedback | H2 target candidate | Check availability |
| **Ego4D v0 corpus** (archived) | — | Bystanders | **None** | Optional: false-positive rate on steady-state co-activity | On SSD (`full_run_2026_06_18/segment_dataset_991/`) |

---

## Verdict videos: the core hidden-outcome source

**What they are:** taste tests and reviews where a person tries something, reacts, and *then* states a verdict: food reviews that score dishes out of 10, "trying X for the first time", drink and snack rankings. The reaction comes before the verdict, and the verdict is the label. This format is extremely common, which is what makes the H2 scaling curve possible.

**Why it is the right first test:** the outcome (how good the food was) is **not visible** in the action frames. A VLM judge can see the dish and its name and guess from priors, but it cannot taste. If reactions carry reward information anywhere, they carry it here. It is also the robotics pitch: a robot serving food cannot taste it.

**Labeling pipeline (fully automatic):**
1. **ASR** with word timestamps (Whisper; `mlx-whisper` on Apple Silicon).
2. **Verdict extraction**: a regex for `N/10`, `N out of 10`, `N stars`, with an LLM fallback for phrasings like "solid eight". Map each verdict to its item when one video reviews several (segment by item mentions in the transcript).
3. **Reaction window**: from the item's first taste to the start of the verdict utterance, minus a safety margin. First-taste detection starts as a simple transcript and visual heuristic, tuned in the Phase-1 pilot.
4. **Label**: binary high/low (e.g. ≥ 8 vs ≤ 5 out of 10, middle excluded) for AUROC, plus the raw score for rank correlation.
5. **Code QA**: check verdict parsing and windowing on a random sample of transcripts. This tests our parser, not the meaning of any label.

**Leakage and confounds (each is a reported condition or control, not an afterthought):**
- **Words inside the reaction window** ("oh this is amazing") partly state the verdict. Report three conditions: face only, non-verbal audio (paralinguistic embeddings, no transcript) and full audio + transcript. The claim we care about most is face plus non-verbal audio.
- **Reviewer identity.** Some reviewers rate everything high. Use group splits by channel, and add a within-reviewer analysis that ranks items for the same person (the offline analogue of stage A personalization).
- **Item priors.** Some foods are polarizing. That is exactly what the VLM-judge control absorbs: it sees the item and its name.

## HoloAssist: first-person, hidden outcome

An egocentric performer (headset camera) completes physical tasks while a remote instructor watches the live feed and talks them through it. Annotations include **mistakes**, **intervention types** and action segments. The reaction is the instructor's voice; the outcome is the mistake label for the segment.

**Check before use:** if mistake labels were derived *from* the instructor's interventions, reaction and label are not independent and the H1 number would be circular. Read the annotation protocol in the paper and repo first.

## Oops!: visible-outcome contrast

Web "fail" videos with a marked failure onset. The reaction is mostly the filmer's audio: a laugh, a gasp, "oh no". Because every clip is a failure, the task is **localization**: does the reaction signal mark the moment things went wrong? The VLM judge should be strong here. That is the point: it is the condition where we *predict* reactions add little.

Limitation: spectators visible in frame cannot be fully masked from the judge. That biases the comparison toward the judge, so it is conservative for our claim.

## BAD / ERR@HRI / REACT: reactions to robots

These are the only sources where the reactions are *to a robot*. They are H2 targets, never training data, so that "trained on web video, transferred to robots" stays a clean zero-shot claim. **Request access in Phase 0**, because data-use agreements take time.

## Ego4D v0 corpus (archived; optional reuse)

991 clips, 23,378 segments, already on the SSD. It has no outcome labels, so it cannot test H1 or H2. It is still useful as a **false-positive set**: mostly steady-state co-activity (cleaning, cards, groceries) where no evaluative reaction occurs. A reward model that fires confidently here is hallucinating social feedback. Do not extend it.

## Licensing policy

- Each dataset's own license governs it.
- Web video follows the v0 **dehydration rule**: never redistribute pixels we do not own. Releases contain video IDs, timestamps, labels and derived features only.
- How web video may be *downloaded* for research is an open decision ([00](00_thesis.md)).
- Raw video lives only on the external SSD under `DATA_ROOT` (`src/config.py`).
