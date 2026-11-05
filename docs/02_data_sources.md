# 02: Data Sources

Every source must provide three things: an **action**, a **reaction** from someone watching (or tasting, or supervising), and an **outcome label that does not come from a human rating the reaction**. Without an outcome label a source is unusable here, however many reactions it contains. That is the v0 lesson ([`LESSONS_v0.md`](LESSONS_v0.md)).

| Source | Outcome visible in the action? | Reaction channel | Outcome label | Role | Access |
|---|---|---|---|---|---|
| **REFLEX** (`reflex`, `reflex_resolve`) | **Partly visible** (`reflex`: `Pre` vs `Failure`) / **Hidden** (`reflex_resolve`: `Resolved?` after `Failure + Explanation`) | Human partner's facial AUs, gaze, head rotation/velocity, valence/arousal, arm posture (`react-face`) + vocal prosody (`react-nonverbal`) | Robot failure (`Pre` vs `Failure`) and post-explanation collaborative resolution (`Resolved?` `TRUE` vs `FALSE`) | **Wave B H1 core (`05_wave_b_reflex.md`)** + robot-reaction benchmark + Stage A MicroDuck pretraining (`DW2`, `DW6`) | **Public download ([Zenodo 14590226](https://zenodo.org/records/14590226), CC BY 4.0, 37.22 GB)** |
| **Verdict videos** (web / AM-FED+ / taste-liking) | **Hidden** | Taster's face + voice before the verdict | Spoken verdict ("7 out of 10") or self-reported liking | Superseded for Wave B by REFLEX after Option E (`M4` bounced, `M5` no reply) did not yield access (`00_thesis.md` entry 28) | Superseded (Issue 1) |
| **HoloAssist** | **Hidden / partly visible** | Remote instructor's voice | Mistake annotations per action segment | H1 first-person; H2 target | Public download ([site](https://holoassist.github.io)) |
| **Oops!** | **Visible** | Filmer's audio (laughs, gasps), visible spectators | Failure-onset timestamp | H1 visible-outcome contrast | Public download ([site](https://oops.cs.columbia.edu/data)) |
| **RoboReward** | **Visible** (vision-only subset) | None (robot rollouts) | Task completion reward (`5` vs `≤ 2`) | Wave D offline H3 (`04_wave_d_robot_check.md`) | Public download ([Hugging Face](https://huggingface.co/datasets/teetone/RoboReward), CC BY 4.0) |
| **BAD** / **ERR@HRI 3.0** | Visible (stimulus video) | Webcam face of the viewer | Failure vs. control stimulus | Replaced by REFLEX (Issue 4, November 1, 2026) | Controlled access: IRB required |
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
- **Unknown until downloaded** *(status October 10, 2026)*:
  - the exact JSON field names: **answered** by A4 (below);
  - whether the official splits are participant-disjoint: **answered** by A4: they are not;
  - whether the instructor's voice is audible in the video's audio track: **still open**. A8 step 2 measures it.

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

### Schema (as downloaded, October 9, 2026)

- **Exact file names:**
  - `data-annotation-trainval-v1_1.json` (117,011,015 bytes, sha256: `cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c`): train and validation annotation events for 1,758 sessions.
  - `data-splits-v1_2.zip` (10,554 bytes, sha256: `e10674e7ac32957386d5e88afa60acf3335251b92ba2f8cfb97a736dc85e1621`): official split session lists: `train-v1_2.txt` (1,466 sessions), `val-v1_2.txt` (207 sessions), and `test-v1_2.txt` (438 sessions).
- **Fine-grained action fields:**
  - `label`: `"Fine grained action"`
  - `start`, `end`: floats in seconds (duration typically 1–2 s)
  - `attributes["Verb"]`: action verb string (e.g. `"grab"`, `"insert"`, `"screw"`)
  - `attributes["Noun"]`: action object string (e.g. `"handheld_grip"`, `"screw"`, `"battery"`)
  - `attributes["Adjective"]`: adjective or `"none"`
  - `attributes["adverbial"]`: adverbial or `"none"`
  - Mistake/correct attribute: exact field name is `attributes["Action Correctness"]`, with values:
    - `"Correct Action"`: 141,691
    - `"Wrong Action, corrected by instructor verbally"`: 4,052
    - `"Wrong Action, corrected by student"`: 3,061
    - `"Wrong Action, not corrected"`: 92
    - `"otherwise"`: 357 (annotator free-form mistake explanations, e.g. accidental battery door closure)
  - `attributes["Incorrect Action Explanation"]`: human explanation text of why the action is incorrect, or `"none"`
  - `attributes["Incorrect Action Corrected by"]`: `"none"`, `"instructor"`, or `"student"`
- **Utterance fields:**
  - `label`: `"Conversation"`
  - `start`, `end`: floats in seconds
  - Speaker role: encoded as the prefix in `attributes["Conversation Purpose"]` (`instructor-start-conversation_...`, `instructor-reply-to-student_...`, `student-start-conversation_...`)
  - Purpose label: `attributes["Conversation Purpose"]` (e.g. `"instructor-start-conversation_correct the wrong action"`, `"instructor-start-conversation_follow-up instruction"`, `"instructor-start-conversation_confirming the previous action"`, etc.)
  - Transcript: `attributes["Transcription"]` (text string, e.g. `"Now disassemble it."`, `"*unintelligible* it."`)
  - Transcript confidence: `attributes["Transcription Confidence"]` (`"high-confidence-transcription"`, `"low-confidence-transcription"`)
- **Task names:**
  - Session-level `taskType` string (20 manipulation tasks across 16 objects, e.g. `"setup gopro"`, `"setup nintendo switch"`, `"assemble ikea stool"`, `"change dslr battery"`, etc.)
- **Performer and instructor identifiers:**
  - No explicit participant ID field exists in the session metadata (`batch`, `videoMetadata`, `events`, `taskId`, `taskType`, `video_name`). Session names (`video_name`) begin with recording/pair prefixes (e.g. `z114`, `R005`, `z108`).
  - Across the 2,111 sessions in the official splits, there are 340 unique prefixes.
- **Official split files and disjointness:**
  - `train-v1_2.txt` (1,466 sessions), `val-v1_2.txt` (207 sessions), `test-v1_2.txt` (438 sessions).
  - **Performer prefix straddle:** 236 out of 340 prefixes appear in more than one official split (e.g. `z108` appears in train, val, and test). The official splits are partitioned randomly per-task, **not participant-disjoint**. Therefore, A8 must use a grouped 70/30 split (`splits/holoassist.json`) rather than the official splits.
- **Measured stats (`src/sources/holoassist.py --stats`):**
  - Sessions: 1,758
  - Fine-grained actions with mistake/correct attribute: 148,896 (149,253 including `"otherwise"`)
  - Mistakes: 7,205 (**4.84%**) (or 7,562 [**5.07%**] including `"otherwise"`)
  - Instructor utterances: 29,222
  - Performer IDs in official splits: 340 unique prefixes, **236 present in >1 official split** (0 means participant-disjoint)
  - `react-spoke` signal in reaction window `[start, end + 5.0]`:
    - Mistake actions: 4,898 / 7,205 (**67.98%**) (or 5,053 / 7,562 [**66.82%**] including `"otherwise"`)
    - Correct actions: 43,736 / 141,691 (**30.87%**)
- **Independence verdict:**
  - Protocol quote verbatim:
    > *"Action Correctness: Indicate whether the action is correct or a mistake to achieve the task. The options are: Correct action, Wrong action, corrected by instructor verbally, Wrong action, corrected by performer, Wrong action, not corrected, Others"* (HoloAssist README)
    > *"Incorrect Action Explanation: Provided by the human annotators to explain why they believe the action is wrong."* (HoloAssist README)
    > *"Mistakes include the ones that are 'self-corrected by the task performers', are 'verbally corrected by the instructors', and 'are not corrected labeled'. Our human annotators annotate all three mistake types separately, but for benchmark evaluation, we will consolidate them into one mistake class. We defer the detailed study of differentiating whether and how the mistakes are corrected to future work. To ensure the annotation quality, we additionally ask the third-person annotators to explain why the action is a mistake and also assign a mapping to every mistake that is corrected by an instructor verbally to the conversation sentence whose type is 'instructor correcting mistakes'."* (Paper arXiv:2309.17024 §3.2)
  - Analysis:
    Mistakes were evaluated and annotated from third-person review of the performer's movements and whether the action achieves the task ("Indicate whether the action is correct or a mistake to achieve the task"), with annotators writing explicit explanations ("Incorrect Action Explanation"). 43.8% of mistakes (3,153 / 7,205) were corrected by the student or uncorrected, with no verbal correction by the instructor. In 32.0% of mistakes (2,307 / 7,205), no instructor utterance overlapped `[start, end + 5.0]`. The mapping from mistakes to verbal corrections was an additional linking step, not the source from which the mistake label was derived.
  - Verdict: **Issue 2 Resolved: independent**.

### Item construction: pinned details (designer, October 10, 2026)

These were left open by the item definition above. The A8 draft adapter chose them; they are now decisions.
- **`item_id` = `"holoassist:<video_name>:<event index>"`**, where `<event index>` is the action's position in that session's `events` list. This is the §3 format; the draft's `<video_name>_<id>_<idx>` lacked the dataset prefix.
- **`"otherwise"` is a mistake** (`label = 0`). A4 found these 357 values carry annotator explanations of wrong actions. With it, a session's mistakes are 7,562 (5.07%).
- **`group_id` = the session-name prefix before the first `-`** (e.g. `R0027` from `R0027-12-GoPro`).
  - There is no performer id in the metadata. The 340 prefixes match the paper's ~350 instructor–performer pairs, so a prefix is most likely a **pair**, not a person.
  - **Caveat for every HoloAssist result:** if a person (performer or instructor) took part in more than one pair, they can appear on both sides of the split. The data cannot rule this out. State it next to the pitch-shift caveat.
- **Task name `fix motorcycle` is rendered `repair motorcycle`** in `context_text`. It is a task name, not an outcome, but the leakage test bans the word `fix`. Verbs and nouns are not rewritten; the real items contain no banned word (checked October 10: 0 of 5,000).
- **`reaction_window_sec` end = `min(duration, end + 5.0)`**, where `duration` is `videoMetadata.duration.seconds`. If that field is missing, skip the action with reason `missing_duration`. The draft defaulted to 10⁹ s, which silently disables the clip.

### Videos (as extracted, October 15, 2026)

- **Total sessions extracted on Extreme SSD:** 2,111 session directories under `raw/holoassist/videos/`, matching every session in the official split lists (`train-v1_2.txt` [1,466], `val-v1_2.txt` [207], `test-v1_2.txt` [438]).
- **Directory layout:** `raw/holoassist/videos/<session>/Export_py/Video_pitchshift.mp4` (all 2,111 sessions have this exact path). The video archive (`1f4f68b9...`, 184 GB) was deleted after extraction per SSD space invariants.
- **Annotated session coverage:** Of the 1,758 sessions in `data-annotation-trainval-v1_1.json`, exactly **1,673 sessions** have an extracted video. The remaining **85 sessions** are not present in the official video release archive, **and none of them is in any official split list** (designer, October 20). They appear to be annotated sessions that were withheld from the release, so skipping them is correct.
- **Handling of missing videos:** Missing videos are skipped honestly by the item builder with reason `missing_video` (6,589 candidate action instances skipped across the 85 sessions; recorded in `DATA_ROOT/items/holoassist/stats.json`), never silently ignored.

### As built (A8): what the results showed about the item definition (designer, October 20, 2026)

- **The step text carries outcome information.** `context_text` includes the step verb and noun.
  - **One verb is an outcome:** `drop`, 435 of the 5,000 items, is a mistake 95% of the time.
  - **A step prior beats every action-only control:** the per-verb mistake rate learned on train scores **0.721** on test (verb + noun 0.727). Without the `drop` items it still scores 0.666, because the class-balanced sample has different step mixes per class.
  - **`judge-large` is reading the verb:** its 0.544 is entirely the `drop` items (0.489 on the rest). The 7B judge does not use it (0.468 on the rest).
  - What the H1 bar should be is **Issue 7**. Until it is decided, every HoloAssist result is reported with this caveat.
- **The instructor is audible in only about a third of sessions** (audio presence check: median +1.51 dB; 6 of 20 sessions ≥ +3.5 dB, mostly quiet device setups). That is **Issue 6**.

### Audible-session evaluation set: `holoassist_audible` (maintainer decision, October 20, 2026; Issue 6 option A, fresh sample)

The question it answers: on the sessions where the instructor can actually be heard, does their non-verbal voice add information beyond the action-only model?
- **Session audibility, measured once, label-blind:** `tools/check_audio_presence.py --all-sessions` (guide item A12).
  - It runs on every annotated session that has a video (1,673), with the same measurement as the A8 check: the mean dBFS inside instructor-utterance spans minus the mean dBFS inside no-utterance gaps of ≥ 2 s.
  - A session is **audible** iff that difference is **≥ +3.5 dB**.
  - A session with no instructor utterance, or no qualifying gap, is **undefined** and excluded, with its reason recorded.
  - The measurement reads the conversation annotations and the audio only, never `Action Correctness`.
  - Output: `DATA_ROOT/runs/holoassist_session_audibility.jsonl` (one line per session).
- **Items:** the same item definition as `holoassist` above (windows, `context_text`, labels with `otherwise` as a mistake, `meta.spoke`, `meta.transcript`), restricted to audible sessions.
  - `item_id` = `"holoassist_audible:<video_name>:<event index>"`, and `meta.source_item_id` = `"holoassist:<video_name>:<event index>"`.
  - `dataset` = `holoassist_audible`.
- **Groups and split:** exactly the `holoassist` group partition: the session prefix, shuffled over all 1,758 annotated sessions with `default_rng(0)`, the first `round(0.7·n)` to train. Restricting to audible sessions never moves a group across the split. Written once to `splits/holoassist_audible.json`.
- **A fresh sample:** within audible sessions, all mistakes plus an equal number of correct items (`default_rng(0)`), capped at 1,500 per class in train and 1,000 per class in test. It is drawn fresh, not as a subset of the 5,000 `holoassist` items.
- **Expected size** (from the 20-session sample; a sanity range, not a target): 3,000–4,000 items, of which 1,000–1,300 are test.
- **Conditions:** every §6 condition, including `step-prior`, `action-fusion`, the ablation Δ and `react-nonverbal|spoke=1`.
  - The judges use the `holoassist` question verbatim, so the `prompt_hash` is identical. The caches are its own, under `judge/holoassist_audible/` and `features/holoassist_audible/`.
- **Report it as a subset:** quieter tasks (device setups) are over-represented, and the rule was chosen after the 20-session check failed, though before any outcome was seen. Give its task mix next to `holoassist`'s.

#### As run (A13, October 21, 2026; verified by the designer)

- **Size:** 4,420 items: 3,000 train (the 1,500-per-class cap was reached) and 1,420 test (710 per class; 53 test groups). That is above the 1,000–1,300 sanity range estimated from 20 sessions; only the lower bound was a stop rule.
- **Groups:** every group is on the same side of the split as in `holoassist`; `straddling_groups` is 0.
- **`stats.json` skip reasons:** `not_audible` (107,363 actions) also counts the 6,589 actions of the 85 annotated sessions without video. Those were never measured, so they cannot be on the allowlist. This is accepted; the item set is unaffected.
- **Reproduced by the designer from the caches, to 4 dp:** fusion 0.6711, action-fusion 0.6505, ablation Δ +0.0206 [0.0063, 0.0343].

#### Audibility (as measured, October 20, 2026)

Measured over all 1,673 annotated sessions with video (`tools/check_audio_presence.py --all-sessions`; A12):
- **Overall:** 1,673 sessions; **548 audible** (32.8%), **1,125 inaudible** (67.2%), **0 undefined**.
- **By task:**

| Task | Total | Audible | Inaudible | % Audible |
|---|---|---|---|---|
| setup camera | 129 | 93 | 36 | 72.1% |
| setup gopro | 129 | 90 | 39 | 69.8% |
| setup switch | 128 | 87 | 41 | 68.0% |
| setup big printer | 106 | 60 | 46 | 56.6% |
| setup small printer | 89 | 32 | 57 | 36.0% |
| assemble stool | 96 | 29 | 67 | 30.2% |
| assemble nightstand | 103 | 27 | 76 | 26.2% |
| disassemble tray table | 121 | 26 | 95 | 21.5% |
| disassemble nightstand | 111 | 23 | 88 | 20.7% |
| disassemble stool | 116 | 20 | 96 | 17.2% |
| assemble tray table | 109 | 18 | 91 | 16.5% |
| change belt | 17 | 3 | 14 | 17.6% |
| make coffee with nespresso machine | 97 | 15 | 82 | 15.5% |
| make coffee with espresso machine | 41 | 6 | 35 | 14.6% |
| disassemble utility cart | 81 | 8 | 73 | 9.9% |
| assemble utility cart | 67 | 5 | 62 | 7.5% |
| assemble computer | 71 | 5 | 66 | 7.0% |
| fix motorcycle | 21 | 1 | 20 | 4.8% |
| assemble laser scanner | 19 | 0 | 19 | 0.0% |
| change circuit breaker | 22 | 0 | 22 | 0.0% |
| **Total** | **1,673** | **548** | **1,125** | **32.8%** |

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

### Schema (as downloaded, October 10, 2026)

- **Exact file names:**
  - `annotations.tar.gz` (1,113,937 bytes, packaged inside `video_and_anns.tar.gz`):
    - `transition_times.json` (2,804,116 bytes): timing annotations for 10,961 clips.
    - `heldout_transition_times.json` (601,411 bytes): 4,791 challenge test evaluation clips (`t: [-1.0]`).
    - `train.txt` (1,400,823 bytes, 23,201 clips) and `val.txt` (400,835 bytes, 6,739 clips): official train/validation split partition.
    - `train_filtered.txt` (967,790 bytes, 16,012 clips) and `val_filtered.txt` (281,327 bytes, 4,711 clips): subset filtered for consensus.
    - `filtered_vids.txt` (1,249,118 bytes, 20,724 clips): total compilation video clip list.
- **Per-worker failure onset fields:**
  - Key in `transition_times.json` is the clip name (without `.mp4` extension).
  - `t`: list of 3 float timestamps in seconds marked by MTurk annotators. `-1.0` denotes that the annotator found no failure transition.
  - `len`: float duration of the video clip in seconds.
  - `stdev`: standard deviation across worker timestamps.
  - `rel_t`: relative transition timestamps (`t / len`).
  - `rel_stdev`: relative standard deviation (`stdev / len`).
  - `n_notfound`: count of workers (0 to 3) who marked `-1.0`.
- **Train / validation membership:**
  - `train.txt` and `val.txt` are mutually disjoint (0 shared clips).
  - Of the 10,961 clips in `transition_times.json`, exactly 6,170 belong to `train.txt` and 4,791 belong to `val.txt` (100% split coverage).
- **Source compilation ID derivation:**
  - Derived from clip filename: clip names are constructed from YouTube compilation titles followed by a clip index (e.g., `"FailFactory - No Pain, No Gain (Workout Fails)12"`).
  - Derivation rule: `re.sub(r"\d+$", "", clip_id).strip()`. If the stripped string is non-empty, use it; otherwise use `clip_id`.
  - Result: 469 unique compilation groups across the 10,961 annotated clips.

### As built (A7; verified October 10, 2026)

- **"Onsets missing" means any worker said "no failure".** A clip is skipped if `n_notfound > 0`, fewer than 3 onsets exist, or any onset is negative. This is stricter than taking the median of the remaining onsets. It is accepted, because a worker who saw no failure makes the onset ambiguous, and the caps were never binding.
- **Yield** (`DATA_ROOT/items/oops/stats.json`):
  - 10,961 clips seen; **1,355 kept**.
  - Skipped: onsets missing 4,134 · stdev > 1.0 s 1,126 · `t − 4.0 < 0` 3,571 · `t + 3.0 > len` 775 · video missing 0.
  - Items: **1,638 train / 1,072 test** (819 / 536 clips; caps 2,000 / 1,000 not reached). Test has 90 compilation groups.
  - The `pre` window needs `t ≥ 4 s`, which by itself removed a third of the clips.
- **One clip is undecodable** (`Grass Is Always Greener - Get Off My Lawn! (July 2018) _ FailArmy29`, train). It has no features and no judge score, and is honestly excluded from the train fits.
- **The "reaction" audio is not only a reaction.** `reaction_window_sec` = `action_window_sec`, so the `post` window's audio contains the failure's own sound (the crash, splash or thud) and any compilation music, as well as the filmer's laugh or gasp. **`react-nonverbal` on Oops! (AUROC 0.711) is therefore not evidence of reaction signal.** It measures how much the audio track differs before and after a failure. State this wherever the number is reported.

## RoboReward: robot episodes for Wave D (added October 21, 2026)

- **Source:** `teetone/RoboReward` (Hugging Face; CC-BY-4.0), revision `469b9af76e8539e8d2ac553b081307738fd92ca5`.
  - 43.2 GB: train 45,072 videos (38.64 GB), val 6,232 (2.91 GB), test 2,831 (1.63 GB), plus one `metadata.jsonl` per split.
  - The test split has 1,000 `robo_arena` rollouts and about 100 episodes from each of 22 Open X-Embodiment sources.
- **Fields** (read through the Hugging Face viewer API, October 21): `video`, `task` (the instruction), `reward` (1–5) and `gpt5_mini_check`.
  - OXE files named `…_index_<n>.mp4` are original episodes; `…_attempt_<k>_score_<s>.mp4` are clipped versions of them.
  - `robo_arena` files are named `<rollout uuid>_<policy>_<camera>.mp4`, with several camera views per rollout.
  - In 800 sampled test rows: OXE originals had rewards 1–5 (rewards below 5 are counterfactual-instruction relabels); attempts 1–4; `robo_arena` 1–5.
- **Wave D's vision-only labels, groups, filters and sampling are pinned in `04_wave_d_robot_check.md` §2.1.** The whole dataset is downloaded (Issue 9 option B: 54,140 files, 43,204,474,744 bytes).

### Schema (as downloaded, October 22, 2026)

- **Files and bytes:** 54,140 files total, 43,204,474,744 bytes, 54,135 `.mp4` video files, revision `469b9af76e8539e8d2ac553b081307738fd92ca5`.
- **Counts per split:**
  - `train`: 45,072 videos, 1 `metadata.jsonl` (sha256 `f0a21f02dce886446c28a92e5203c49de068bacd81df9011fbd824e567ab2312`)
  - `val`: 6,232 videos, 1 `metadata.jsonl` (sha256 `73710f1ad522050b46e0b7f6e0ca35d6e7c0c61221b20f3f239b2dee878fdaec`)
  - `test`: 2,831 videos, 1 `metadata.jsonl` (sha256 `3901cea9a7981d2e2d9fde4225a4b2ff0ebf2f4af69fc96972a2cfbe557db0cd`)
- **Metadata field names:**
  - `file_name`: relative path to video file under split directory (e.g. `austin_sirius_dataset_converted_externally_to_rlds/...mp4`). Note: the viewer API previewed this as `video`, but the actual on-disk metadata key is `file_name`. `src/sources/roboreward.py` resolves `file_name` (falling back to `video`).
  - `task`: language instruction string.
  - `reward`: integer rubric score (1 to 5).
  - `gpt5_mini_check`: verification text or null.
- **Durations of the 5,517 built items** (designer correction, October 22: the line written in D3 said 0.9–82.7 s):
  - all items: 0.1–134.6 s (median 7.7 s, mean 16.1 s); 519 are under 1 s, mostly clipped `_attempt_` failures;
  - train pool: median 6.6 s;
  - test: median 16.3 s. The test split is 45% `robo_arena`, against 19% in the train pool, which follows RoboReward's own split composition.

## The HoloAssist pretraining pool (`holoassist_pool`, Wave D)

- These are 20,000 actions from the 548 audible sessions that are not `holoassist_audible` items, in the natural class proportion. They are a pretraining set for Wave D's human heads, not an evaluation set.
- The definition is pinned in `04_wave_d_robot_check.md` §2.2.

## REFLEX: multimodal human reactions to robot failures and explanations (Wave B, added November 1, 2026)

Fifty-five human participants collaborate with a physical Baxter robot on a tabletop object-sorting task across five robot explanation strategies while the robot commits programmed manipulation failures (`Pick`, `Carry`, `Place`) and explains them (Naoum et al., KTH, HRI 2025; arXiv:2502.14185).

**Facts & Archive Schema (verified November 1, 2026 against Zenodo API and `Dataset.zip` central directory):**
- **Source:** [Zenodo 14590226](https://zenodo.org/records/14590226) (`DOI: 10.5281/zenodo.14590226`, version 1.1.0, March 2, 2025; **CC BY 4.0**).
- **Archive file:** `Dataset.zip` (`https://zenodo.org/api/records/14590226/files/Dataset.zip/content`), **37,219,808,401 bytes** (37.22 GB), MD5 **`334caac04f3e607ac91182dea7a03929`**.
- **Directory layout under `DATA_ROOT/raw/reflex/Dataset/`:**
  - 5 strategy folders, 11 participants each (**55 participants total**):
    - `C1-Fixed-Low/C1-1` … `C1-11`
    - `C2-Fixed-Medium/C2-1` … `C2-11`
    - `C3-Fixed-High/C3-1` … `C3-11`
    - `D1-Decay-Smooth/D1-1` … `D1-11` *(named `Decay-Slow` in the paper's `DatasetGuide.md`, but the directory in `Dataset.zip` is `D1-Decay-Smooth`)*
    - `D2-Decay-Rapid/D2-1` … `D2-11`
  - Every participant directory contains 10 required files: `analysis.csv`, `body.csv`, `facetorch.csv`, `gaze.csv`, `hume.csv`, `openface.csv`, `questionnaire.csv`, `speech.csv`, `time.csv`, and `video_cam1.mp4` (`video_cam2.mp4` is missing in `C1-Fixed-Low/C1-10` and `D1-Decay-Smooth/D1-3` and is not used).
  - The zip contains `__MACOSX/` and `._*` AppleDouble entries; extraction and any glob over `DATA_ROOT` must ignore them (`shared.fsutil.data_glob`).
- **Video anonymization & why `video_cam1.mp4` is reaction-masked by construction:**
  - `video_cam1.mp4` is encoded at $\approx 4.4444\text{ Hz}$ (`40/9` fps via `cv2.VideoWriter`) with **no audio stream**. *(**As built:** `ffprobe` reports 4.4 fps (`r_frame_rate` = `avg_frame_rate`) for all 55 videos. The pipeline reads the rate from each file, so nothing depended on 40/9.)*
  - For GDPR privacy (`arXiv:2502.14185` §IV-1), **the human participant is segmented out with a black mask** in `video_cam1.mp4`, while the **Baxter robot and tabletop objects remain visible**.
  - Therefore, passing `video_cam1.mp4` to `judge` (`qwen2.5vl:7b`), `judge-large` (`gemma4:26b`), and `action-probe` (`siglip-b16-224-u9`) shows the robot's physical action with the participant's facial and body reactions masked out. *(**Corrected November 4, 2026:** only the face and identity are masked. The black silhouette keeps the participant's posture and gestures visible, such as leaning or a hand raised to the head; see "As built" below.)*
  - Meanwhile, the REFLEX authors ran **OpenFace 2.0**, **Facetorch**, **MediaPipe Pose**, and **Hume Expression Measurement (Face + Speech Prosody)** on the **unmasked raw recordings before anonymization** and released the per-frame and per-utterance outputs in the CSV files.
- **CSV schemas & frame/time alignment (pinned in [`05_wave_b_reflex.md`](05_wave_b_reflex.md) §2–§4):**
  - `time.csv`: columns `Frame, Stamp, Timestamp, Seconds, Time`. `Frame` is 1-indexed (`1 … N`, matching `video_cam1.mp4` frames); `Seconds` is elapsed wall-clock seconds from Frame 1.
  - `analysis.csv`: columns `Participant, Strategy, Round No., Object, Action, Explanation Level, State, Start Frame, End Frame, Start, End, Resolved?`.
    - Contains **604 robot-failure episodes** across the 55 participants (2,416 rows: each episode is a contiguous 4-row block with `State` in `Pre`, `Failure`, `Explanation`, `Resolution`).
    - On the 604 `Failure` rows, `Resolved?` is `"TRUE"` on **477**, `"FALSE"` on **116**, and empty/other on **11**. *(**Corrected November 4, 2026:** the 11 are participant `C3-2`'s episodes, whose header is misspelled `Resloved?`. All 11 are `TRUE`, so the true counts are 488 `TRUE` and 116 `FALSE`.)*
    - **Leakage warning (Lesson L12):** the `Object` column contains annotator failure descriptions (e.g. `"White Cube - Drop"`, `"Red Toy (Nothing inside)"`, `"Cup (Wrong Cup)"`). `Object` is **never** included in `context_text` or `meta.noun`.
  - `openface.csv`: 1-indexed `frame`, `confidence`, `success`, 8 gaze columns, 3 head rotation columns (`pose_Rx, pose_Ry, pose_Rz`), 17 continuous AU intensities (`AU*_r`), and 18 binary AU presences (`AU*_c`).
  - `facetorch.csv`: 0-indexed `Frame ID` (`= Frame - 1`), `Valence`, `Arousal`, `Dominance`.
  - `body.csv`: 0-indexed `Frame` (`= Frame - 1`), `Crossed Arms` (`TRUE`/`FALSE`), `Arms behind back` (`TRUE`/`FALSE`), and 33 MediaPipe 3D pose landmarks.
  - `speech.csv`: utterance-level `BeginTime`, `EndTime` (in wall-clock seconds matching `time.csv` `Seconds`), `Transcript`, `Id` (`"robot"`, `"participant"`, `"robot/participant"`), and 48 Hume vocal prosody dimensions (`Admiration` … `Triumph`). *(**Corrected November 4, 2026:**)*
    - *the columns are `Id, Text, BeginTime, EndTime, Confidence, SpeakerConfidence` and then the 48 prosody columns (6–53). There is no `Transcript` column; the words are in `Text`;*
    - *`Id` also takes `"robot/ participant"` (12 rows, with a space), `"spk_0"` … `"spk_6"` (63 rows) and `"unknown"` (5 rows).*
- **Two evaluation datasets (pinned in [`05_wave_b_reflex.md`](05_wave_b_reflex.md) §3):**
  1. **`reflex` (1,208 candidate items across 55 participant groups):** immediate reaction to robot failure (`Pre` `label = 1` vs. `Failure` `label = 0`), with `react-face` (`reflex-face-v1`, 153-d `float32`) as the primary reaction channel.
  2. **`reflex_resolve` (593 candidate items across 55 participant groups):** hidden-outcome collaborative resolution (`[Failure.Start Frame, Explanation.End Frame]` action window before resolution $\rightarrow$ `Resolved?` `TRUE` `label = 1` vs. `FALSE` `label = 0`), evaluated with both `react-face` (`reflex-face-v1`, 153-d) and `react-nonverbal` (`reflex-speech-v1`, 96-d), plus `step-prior` on `(action, explanation_level)`.

### As built (B1–B4; verified by the designer, November 4, 2026)

- **Items:**
  - `reflex` has 1,208 items (604 / 604).
  - `reflex_resolve` has **604** items (488 resolved / 116 not), because B1 normalized `C3-2`'s `Resloved?` header.
  - The designer recomputed every item's frame bounds, container seconds and wall-clock seconds from the CSVs: 0 mismatches.
- **Window lengths differ by class in `reflex`.** Reaction windows average 45.7 frames for `fail` vs. 35.5 for `pre`; action windows average 23.7 vs. 35.5. Window length alone predicts the `reflex` label at 0.735 on test (`05_wave_b_reflex.md` §7).
- **`reflex_resolve`'s outcome follows the step:**
  - resolved: `pick` 219/219, `carry` 198/220, `place` 71/165;
  - by explanation level only 74–88% (high 165/187, medium 115/143, zero 102/131, low 106/143);
  - a step prior on `(action, explanation level)` scores 0.938 on test.
- **What the masked video shows:** the robot, the table and the participant's black silhouette. Posture and gestures are visible; the face is not.

## BAD / ERR@HRI / REACT: reactions to robots (superseded by REFLEX, November 1, 2026)

**If BAD access is granted, these commitments from the access request bind the project:**
- BAD is a held-out evaluation set only, with no training on it;
- it is stored only in an encrypted volume on the maintainer's personal hardware (never employer devices, never cloud sync);
- it is processed **locally only**. Frames are never sent to any cloud service or API, **including the Gemini frontier judge**;
- only aggregate metrics are published; never frames, face crops, embeddings or per-participant results;
- the data and everything derived from it are destroyed at study end, or 12 months after download, whichever comes first, and QDR is notified.

These are the only sources where the reactions are *to a robot*. They are H2 targets, never training data, so that "trained on web video, transferred to robots" stays a clean zero-shot claim. **BAD requires an IRB-reviewed protocol from an affiliated institution**, which the maintainer (unaffiliated) lacks. The access request asks whether an exception is possible. **Issue 4** records the decision of October 9: an own reaction study and an academic partner were both declined, so if the request is refused, H2 transfer is shown on HoloAssist and AM-FED+ with no robot-reaction target.

## Ego4D v0 corpus (archived; optional reuse)

991 clips, 23,378 segments, already on the SSD. It has no outcome labels, so it cannot test H1 or H2. It is still useful as a **false-positive set**: mostly steady-state co-activity (cleaning, cards, groceries) where no evaluative reaction occurs. A reward model that fires confidently here is hallucinating social feedback. Do not extend it.

## Licensing policy

- Each dataset's own license governs it.
- Web video follows the v0 **dehydration rule**: never redistribute pixels we do not own. Releases contain video IDs, timestamps, labels and derived features only.
- How web video may be *downloaded* for research is **Issue 1** in [`ongoing_general_errors.md`](ongoing_general_errors.md).
- Raw video lives only on the external SSD under `DATA_ROOT` (`src/config.py`).
