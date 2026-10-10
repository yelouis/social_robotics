# Engineering Issues & Decisions: Working Log

**What this file is:** the live queue of open issues, the decisions the maintainer has made or still has to make, maintainer-only actions, deferred work with its triggers, and a one-line index of resolved work.
- The build spec is [`agent_execution_guide.md`](agent_execution_guide.md). This file is where findings and choices live.
- The project's decision log lives in [`00_thesis.md`](00_thesis.md) (one source).

**Filing format.** An open issue states its status, the facts with dates and sources, two or more options, each with pros and cons, a recommendation, and a final `Your selection: _____` line. **That line belongs to the maintainer, and an agent must never fill it in.**

---

## 1. Open & in-flight

**October 8, 2026: the project was reoriented.**
- **What happened:** the v0 Social-Affective Filter is archived at tag `v0-saf-final` (why: [`LESSONS_v0.md`](LESSONS_v0.md)). The tree now holds the new grounding docs plus three utilities: `src/shared/vlm_client.py`, `src/models_config.py` and `tools/`.
- **Approved build:** **Wave A**, the evaluation harness plus the first H1 numbers on Oops! and HoloAssist ([`agent_execution_guide.md`](agent_execution_guide.md)).
- **Decisions pending:**
  - Issue 1 blocks Wave B only.
  - Issue 4 was decided October 9: the BAD request, else no robot-reaction target.
  - Issue 2 was resolved October 9: independent (item A4).
  - Issue 3 (SSD space) was resolved October 8.

---

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Verdict-video sourcing: download public videos (release only IDs, timestamps and labels) or Creative Commons only?

**Status:** ⚠️ Awaiting selection. **Blocks Wave B only.** Wave A uses Oops! and HoloAssist, which are downloaded from their official pages.

**Facts (October 8, 2026):**
1. **YouTube's terms prohibit downloading** except where the service expressly authorizes it, or with written permission from YouTube and the rights holders. They separately prohibit automated access (robots, scrapers) without permission ([terms](https://www.youtube.com/terms)). **This holds for Creative Commons videos too.** A CC license grants *copyright* permission to reuse; it does not change YouTube's terms on *how you obtain the file*.
2. **YouTube actively blocks bulk downloaders.**
   - 2026 yt-dlp issue reports show "Sign in to confirm you're not a bot" walls on public videos ([#15865](https://github.com/yt-dlp/yt-dlp/issues/15865)).
   - One user was walled after about 600 downloads ([#16147](https://github.com/yt-dlp/yt-dlp/issues/16147)).
   - **Standing rule for this project: never circumvent a bot check** (no cookies, PO-token providers, proxies or client spoofing). Any YouTube bulk download is therefore capped by how far plain, polite downloading gets.
3. **Search and metadata are authorized through the YouTube Data API.**
   - `search.list` accepts `videoLicense=creativeCommon`, so both pools can be *counted* legitimately ([API reference](https://googleapis.github.io/google-api-python-client/docs/dyn/youtube_v3.search.html)).
   - The [YouTube Researcher Program](https://research.youtube.com/) expands API quota for researchers affiliated with an accredited higher-education institution. That is metadata only, not video files.
4. **Precedent:** Kinetics, AudioSet and HowTo100M distribute YouTube IDs plus annotations. **Link rot is real.** Kinetics' older editions lost enough videos that a snapshot is now hosted separately ([Kinetics-700-2020 note](https://arxiv.org/abs/2010.10864)).
5. **What we release, under any option:** IDs, timestamps, verdict labels, model scores. **Never face or voice embeddings of identifiable people.**
6. **The maintainer is not affiliated with a university** (stated October 9, 2026). The consequences:
   - the YouTube Researcher Program is unavailable;
   - there is no institutional ethics review or counsel, and the EU research text-and-data-mining exception (for research organisations) does not apply, so any terms or copyright risk is personal;
   - academic-only datasets are closed: Emognition needs an academic email, and DEAP needs a permanent academic position.
7. **Licensed reaction-plus-verdict data does exist, behind agreements without a stated affiliation rule:**
   - **AM-FED / AM-FED+:** webcam reactions to a few Super Bowl ads, with self-reported "Did you like the video?"; AM-FED+ has 1,044 videos. Access is a signed non-commercial EULA, emailed to `amfed@affectiva.com` ([AM-FED](https://www.affectiva.com/facial-expression-dataset-/)).
   - **A taste-liking database:** 2,970 videos of taste-induced expressions from 495 people (*Automatic Estimation of Taste Liking Through Facial Expression Dynamics*, IEEE TAC 2020). No public download was found; ask the authors.

**Option A: Download public videos for research; release only IDs, timestamps and labels (as v0 did)**
- *Pros:*
  - by far the largest pool for the taste-test/review genre, which is what makes the H2 scaling curve possible;
  - least selection bias;
  - established practice in video ML;
  - consistent with the v0 dehydration rule.
- *Cons:*
  - **conflicts with YouTube's terms** (a contract risk, separate from copyright);
  - fragile, because downloads stop at the first bot wall and we will not circumvent it;
  - link rot erodes reproducibility;
  - reviewers will ask about consent for affect analysis of identifiable faces;
  - research copyright exceptions (US fair use, EU/UK text-and-data-mining) depend on jurisdiction and facts. If you have an institution, its view matters.

**Option B: Creative Commons only**
- *Pros:*
  - copyright allows redistribution, so we can ship the clips themselves (fully reproducible, no link rot);
  - the cleanest licensing and ethics story;
  - matches v0's "Ring 2" plan.
- *Cons:*
  - **does not fix the download problem**, because the CC videos are still on YouTube;
  - probably far smaller for *this* genre, since popular food reviewers rarely choose CC. That risks making the scaling curve impossible;
  - CC uploads include mislabeled re-uploads, so licenses need verifying;
  - CC covers the uploader's copyright, not the consent of others on screen.

**Option C (recommended): measure first via the API, then decide by a fixed rule**
- One day of API-only counting (authorized) of verdict-video candidates in both pools, with the query list and title filter fixed in the Wave B spec.
- **The rule:**
  - if the CC pool has ≥ 1,000 candidates → **B**;
  - otherwise → **A at pilot scale** (≤ 300 videos), stopping at the first bot wall. Pursue authorized access in parallel: the Researcher Program if you are affiliated, and direct permission from the top channels.
- *Pros:* decides with numbers instead of guesses; Wave A is unaffected either way.
- *Cons:* about a day of delay; needs a YouTube Data API key (maintainer action M3).

**Option D: record our own taste tests** — ❌ **Declined by the maintainer, October 9, 2026:** *"I will not do this."* Kept for the record; do not re-propose.
- Friends and family taste things, react, and state a score out of 10.
- *Pros:*
  - real consent;
  - controlled audio and video;
  - doubles as a rehearsal for stage A (live reactions).
- *Cons:*
  - small (tens to low hundreds of items);
  - takes people's time. It is collection, not rating, but still not "free at scale";
  - cannot produce the H2 scaling curve alone.
- Best as a **complement** to A/B/C, not a replacement.

**Option E: licensed reaction-plus-verdict datasets** (AM-FED+, plus the taste-liking database if its authors share it)
- *Pros:*
  - real verdict labels, legally obtained through an agreement;
  - no YouTube involvement;
  - in AM-FED+, many viewers see the same ad, so an action-only judge is blind *by design*. Only reactions can explain who liked it. That is the cleanest test of "this person, right now".
- *Cons:*
  - the stimuli are ads, not food, so it is a different domain from verdict videos;
  - fixed size (no scaling curve);
  - depends on the EULA holders accepting an independent researcher.

**Recommendation (revised again October 9, 2026, after option D was declined):** **E as the hidden-outcome core, with creator permission (and non-YouTube CC) for scale.**
- **E:** request AM-FED+ (maintainer action M4), and ask the taste-liking authors (M5).
- **Scale:**
  - email taste-test and review creators for permission (template in [`maintainer_access_requests.md`](maintainer_access_requests.md)). C's API count, if M3 is done, is the way to *find* and rank those channels;
  - add CC videos from platforms that allow downloading (Internet Archive, Wikimedia Commons, Vimeo with downloads enabled).
- **A (unauthorized public download)** stays not recommended.
- **The risk this leaves:** every hidden-outcome verdict source now depends on someone else saying yes (Affectiva/Smart Eye, the paper's authors, creators). If none does, Wave B has no verdict data. The kill rule would then fall back to HoloAssist, the only hidden-outcome set already in hand, with its pitch-shift caveat. The designer will state that fallback in the Wave B spec rather than leave it implicit.
- **Worth evaluating in the Wave B spec (not yet verified):** Ego-Exo4D's expert commentary and proficiency labels. Ego4D-style licenses were granted to the maintainer as an individual before. Before any use it needs the same independence check as HoloAssist (A4): were the proficiency labels assigned from the commentary?

Your selection: _____

---

### Issue 2: HoloAssist label independence

**Status:** ✅ **Resolved: independent (October 9, 2026, item A4).**

**The concern:** if HoloAssist's mistake labels were assigned *from* the instructor's interventions, then "the instructor's reaction predicts the mistake" is circular. The intervention correlating with mistakes is expected; it is the signal. *Deriving the label from it* is the problem.

**Resolution (item A4):**
- **Protocol quote verbatim:**
  > *"Action Correctness: Indicate whether the action is correct or a mistake to achieve the task. The options are: Correct action, Wrong action, corrected by instructor verbally, Wrong action, corrected by performer, Wrong action, not corrected, Others"* (HoloAssist README)
  > *"Incorrect Action Explanation: Provided by the human annotators to explain why they believe the action is wrong."* (HoloAssist README)
  > *"Mistakes include the ones that are 'self-corrected by the task performers', are 'verbally corrected by the instructors', and 'are not corrected labeled'. Our human annotators annotate all three mistake types separately, but for benchmark evaluation, we will consolidate them into one mistake class. We defer the detailed study of differentiating whether and how the mistakes are corrected to future work. To ensure the annotation quality, we additionally ask the third-person annotators to explain why the action is a mistake and also assign a mapping to every mistake that is corrected by an instructor verbally to the conversation sentence whose type is 'instructor correcting mistakes'."* (Paper arXiv:2309.17024 §3.2)
- **Findings:**
  - Mistakes were annotated from third-person video observation of whether the physical movement achieved the task, with human explanations provided per mistake.
  - 43.8% of mistakes (3,153 / 7,205) were corrected by the student or uncorrected, with no instructor intervention. In 32.0% of mistakes (2,307 / 7,205), no instructor utterance overlapped `[start, end + 5.0]`.
  - The `react-spoke` speech rate is **67.98% for mistakes (4,898 / 7,205)** vs. **30.87% for correct actions (43,736 / 141,691)**.
  - The official splits are **not participant-disjoint** (236 / 340 performer prefixes straddle splits), so A8 will use a grouped 70/30 split.
- **Verdict:** Issue 2 is resolved as independent. A8 is cleared to proceed when its prerequisites land.

---

### Issue 3: SSD capacity for Wave A downloads

**Status:** ✅ **Resolved, October 8, 2026, by the maintainer's selection:** *"Yes, clean up any videos you want from the v0 leftovers. Feel free to delete Ego4D if you think that is the right choice."*

**What was done (designer):**
- **Deleted:** the v0 **videos** only, 36,506 files, 1.336 TB.
  - `social_robotics/raw_videos/ego4d/v2/full_scale/`: 1,083 Ego4D clips, 1.2 TB.
  - Charades-Ego: `ego_videos/`, `tp_videos/`, `CharadesEgo_v1_480.tar` and `Charades_v1_480.zip`.
- **Manifest:** every path plus the 1,083 Ego4D clip ids and re-download instructions, in `/Volumes/Extreme SSD/social_robotics/DELETED_2026-10-08.json`.
- **Free space:** 311 GiB → **1.5 TiB**.

**Kept, because it is not video or is small and referenced:**
- `ego4d_data/`: annotations plus 68 GB of precomputed Omnivore features; it never held the videos.
- The Ego4D annotations under `raw_videos/ego4d/v2/annotations/`.
- `full_run_2026_06_18/` (13 GB of derived v0 results) and `bench_v0/` (4.5 GB).
- The Charades-Ego annotations.
- The Wan2.1 weights (102 GB) and `saf_env/` (19 GB) are dead weight, but they are not videos, so they were left for a separate decision.

**Why Ego4D was deleted rather than pruned:** nothing in Waves A or B uses it. Its bystander footage was shown to be the wrong data for H1 (no outcome labels; rare evaluative reactions). The only possible reuse, the optional false-positive set (DW4), can re-download a chosen subset by clip id.

---

### Issue 4: The robot-reaction H2 target (BAD) requires an institutional ethics review

**Status:** ✅ **Decided by elimination, October 9, 2026.** The maintainer declined option C (*"I will not do a study for people watching robot clips on a webcam"*) and option B (*"I also do not intend on partnering with an academic institution"*). **The path:** A, with the maintainer's QDR access request in progress (form answers in `maintainer_access_requests.md` M1a). **If refused → D:** H2 transfer is shown on HoloAssist and AM-FED+, with no "reactions to robots" target.

**Facts (October 9, 2026, from QDR's metadata API and the dataset page):**
- **BAD is QDR "Controlled Access."** Only 4 documentation files (0.3 MB) are public. The data (54 participant video zips + survey, 2.71 GB) is restricted.
- **The Terms of Access require:**
  - a description of use and human-subjects protections;
  - **"a protocol for your research study that has been reviewed by an IRB or ethics approval committee at your affiliated institution"**;
  - a special download agreement (no redistribution; use limited to the described study within human-interaction research; no identifying or harming participants).
- **The maintainer is not affiliated** (Issue 1, fact 6).
- **ERR@HRI 3.0's copy of BAD** is likely gated similarly. The 2024 edition's EULA was approved by the University of Cambridge's data protection officer and ethics committee, and signed copies went to a research office. Not confirmed for 3.0.

**Option A: ask anyway, offering an independent IRB** (the M1a draft)
- *Pros:* free to ask; it keeps the published robot-reaction benchmark in play.
- *Cons:* likely refused as written. An independent (commercial) IRB review costs money and takes weeks.

**Option B: partner with an academic collaborator** — ❌ declined October 9. who holds the data under their IRB, and runs or co-authors the BAD evaluation.
- *Pros:* the standard route; it also adds a co-author.
- *Cons:* depends on finding someone; the data stays with them.

**Option C: run our own BAD-style reaction study** — ❌ **Declined October 9.** Originally flagged as possibly ruled out: the maintainer declined self-recorded taste tests (Issue 1 D, October 9), and this is the same kind of self-run collection. Confirm before planning on it.
- Consenting friends and family watch short clips of a robot succeeding and failing (our own footage; later the microduck), on a webcam, with a consent form that covers research use.
- It can share sessions with the Issue 1 option D taste tests.
- *Pros:*
  - fully owned and consented data, which we can publish;
  - doubles as the stage A (live reactions) rehearsal;
  - independent of anyone's approval.
- *Cons:* small (tens of people); not the published benchmark, so less comparable to prior work; needs robot footage first.

**Option D: drop the robot-reaction target from H2.** Transfer is shown on HoloAssist (first person) and AM-FED+ (individual preference) instead.
- *Pros:* no new collection.
- *Cons:* loses the "reactions *to robots*" claim, which is the most robotics-relevant result.

**Recommendation:** send A now (free). If C is also out, the realistic routes to a robot-reaction result are **B** (an academic partner) or a yes on A. Otherwise accept **D**: H2 transfer is shown on HoloAssist and AM-FED+, and "reactions *to robots*" waits for stage A (the microduck).

Your selection: _____

---

## Maintainer actions (not agent work)

| Id | Action | Why | Status |
|---|---|---|---|
| M1 | Request **BAD dataset** access (QDR account → Request Access / "Contact Owner"), asking whether an independent researcher can qualify, and email the **ERR@HRI 3.0** organizers about post-challenge access | H2 targets: reactions to robots. BAD requires an IRB-reviewed protocol from an affiliated institution (Issue 4). Drafts: `maintainer_access_requests.md` M1a/M1b | Open |
| M2 | Add an "Archived (October 2026)" note to the 6 public `louisye/social-robotics-*` Hugging Face cards | They describe the v0 pipeline as current | ✅ Done October 9 (wording confirmed by the maintainer; 6 HF commits) |
| M3 | Create a **YouTube Data API v3** key (a Google Cloud project with the API enabled) | Only for *counting* (Issue 1, option C); never for downloading | Optional |
| M4 | Sign and send the **AM-FED+** EULA | Issue 1 option E: reactions + self-reported liking | Open, pending the Issue 1 selection |
| M5 | Ask the authors of the **taste-liking** database for access | Issue 1 option E | Optional |

---

## 2. Lessons that still bite

Condensed from [`LESSONS_v0.md`](LESSONS_v0.md). Each is a trap that is live for the new code.

- **L1: No outcome label, no evaluation.** v0's features could only be judged by humans, and humans could not judge them. Every item needs a data-derived label.
- **L2: A pipeline that never crashes can hide a stage that never ran.** v0's layers were gated down to ~1% yield and still "succeeded". Every stage logs counts in, out and excluded (by reason), and a condition that could not run is written as a row saying so.
- **L3: Honest nulls.** A failure is null plus a reason, never a zero. v0's silent mock-emotion fallback produced real-looking fake scores.
- **L4: Anchor windows to the observer.** v0 measured where the *actor* moved and missed the observer four times over. Reaction windows are defined per dataset in `02_data_sources.md`.
- **L5: Count distinct things.** Re-anchored windows photocopied one reaction up to 70×. One item = one (source event, window), and `item_id`s are unique.
- **L6: Decode sequentially; never seek per frame on H.264.** Cut windows with ffmpeg.
- **L7: Detached runs or dead runs.** Agent background tasks are reaped after ~1–2 h. Use `tools/daemonize.py` + `tools/run_supervised.sh`.
- **L8: A green suite proves nothing about spec fidelity.** Read the code against the contract.
- **L9: This Mac is shared, and its memory is not ours.** On October 9, 2026 (19:50–19:52) it ran out of memory: the `animated_infographics` agent ran four image-generation gates at once (two at about 27 GB each), Ollama's `llama-server` held 10.5 GB, and this project's A5/A6 slow tests loaded models in the same window. macOS killed its own services. **Every model load is now admitted through a machine-wide heavy lock shared by both projects, and every long loop backs off between items** (`03_eval_harness.md` §12; guide item A6b).

---

## 3. Resolved index

One line per delivered item: `<id> — <title> — git log --grep "(<id>)" — <measured result>`.

**Reorientation (October 8, 2026):**
- R0 — Retire SAF v0; outcome-supervised reaction-reward thesis; grounding docs — `454b40d` — v0 archived at `v0-saf-final`; tests 2/2.
- R1 — Wave A spec: execution guide, harness contract, tracking doc, `AGENTS.md` — see `git log --grep "agent execution guide"`.
- R2 — Issue 3: v0 videos deleted (1.336 TB; Ego4D + Charades-Ego); manifest `DATA_ROOT/DELETED_2026-10-08.json`; 1.5 TiB free — same commit as R1.

**Wave A:** *(the implementing agent adds one line per item here, in the item's own commit)*
- R3 — A6b specified (memory guard; October 10, after the October 9 out-of-memory). Contract in `03_eval_harness.md` §12; sequenced before A7.
- A1 — Battery green and scripted — git log --grep "(a1)" — scripts/battery.sh exit 0 (G1 exit 0, G2 exit 0 [2 passed], G3 skipped).
- A2 — Metrics, scorecard, self-test — git log --grep "(a2)" — G3 selftest passes 4/4 checks (planted AUROC=0.830 [0.807, 0.852], null CI=[0.459, 0.523], grouping width ratio=4.66, paired delta exact 0 on same scores).
- A3 — Items and grouped splits — git log --grep "(a3)" — Item validation rejects all 5 error conditions with verbatim messages; make_group_split guarantees group-disjoint splits and sha256 tamper verification.
- A4 — HoloAssist labels + independence check — git log --grep "(a4)" — Issue 2 resolved independent; labels downloaded (111 MB); react-spoke 67.98% mistakes (4,898/7,205) vs 30.87% correct (43,736/141,691); 236/340 performer prefixes straddle official splits.
- A5 — Encoders and feature cache — git log --grep "(a5)" — FeatureCache with atomic save/load and sanitization; FrameEncoder (siglip-b16-224, 768 float32); NonverbalAudioEncoder (e2v-plus-large, 1024 float32); resumable extract.py; G4 slow tests pass (determinism, shape/dtype, information falsification).
- A6 — The VLM judge (local + frontier) — git log --grep "(a6)" — OllamaJudge (qwen2.5vl:7b) + GeminiJudge (gemini-3.6-flash); build_prompt verbatim; prompt_hash; regex parser with retry logic; payload isolation verified (no audio, no label leakage); G4 live synthetic inference pass.

---

## 4. Deferred work (do not start without its trigger)

| Id | Work | Trigger to start |
|---|---|---|
| DW1 | **Wave B: verdict-video H1 pilot** (~200 videos; face + non-verbal audio; ASR verdict parsing) | Issue 1 selected **and** Wave A closed **and** the designer has written the Wave B spec |
| DW2 | Face encoder (`react-face`) | Part of the Wave B spec |
| DW3 | **H2 transfer** to HoloAssist / AM-FED+ (+ BAD only if the QDR request is granted); scaling curves | Wave B closed |
| DW4 | Ego4D v0 false-positive set | Wave B+. The raw videos were deleted October 8 (Issue 3); re-download the chosen clip ids from `DELETED_2026-10-08.json` |
| DW5 | **Wave D, the robot check (offline H3)**: a data-efficiency curve on RoboReward, with vs. without pre-training on reaction-labeled human task video (`00_thesis.md` H3) | Wave A shows reaction signal on ≥ 1 dataset **and** the designer has written the Wave D spec |
| DW6 | **Stage A live, as a personal demo with the maintainer as the only reactor.** The microduck learns *which of its existing behaviors, and which styles of them, the maintainer likes* from their live reactions. The Mac runs the reaction reader (a webcam on the maintainer, or the duck's front-camera WebRTC stream) and sends choices over the duck's control channel. Measures: trials-to-preference vs. explicit thumbs-up/down, and reaction-reader accuracy against a quick rating after each trial. **Not in scope:** learning motor skills from reactions (that needs millions of trials, in simulation). Facts (October 9, 2026): 15-DoF, 25 cm biped, RK3566 with 1 GB RAM, front camera, 7 shipped moves; new policies are trained in sim with `microduck_rl` (mjlab/MuJoCo Warp, PPO), which **needs a CUDA GPU**, i.e. not this Mac | Hardware acquired **and** H2 done **and** the maintainer opts in (deferred October 9: *"maybe we don't worry about the someone reacting to the robot yet"*) |
