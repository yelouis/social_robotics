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
- **Resolved** (moved to 🧪 below):
  - Issue 2 (HoloAssist labels are independent; October 9, A4);
  - Issue 3 (SSD space; October 8);
  - Issue 5 (large local judge, October 14, A7c);
  - Issue 7 (step-prior and the ablation Δ, October 20, A11);
  - Issue 6 (the audible-session set, October 21, A12–A13).
  - Issue 8 (Wave D go: option A, spec written, October 21).
  - Issue 9 (full RoboReward download verified, October 22, D3).
  - Issue 10 (Wave D v1 accepted as the result; robot transfer revisited only after Wave B; option A, October 22).

**October 22, 2026: designer verification of A15 and D1–D4** (each item's code and outputs read against `04_wave_d_robot_check.md`; the key numbers recomputed independently; the battery run bare).
- **Battery** (on `315f3a6`, with `TMPDIR` on the SSD): G1 exit 0; G2 106 passed, 3 runs out of 3; G3 4/4; G4 11 passed (124 s).
- **Verified correct:**
  - **A15:** the caveat is computed per dataset (0.498 [0.481, 0.514] audible; 0.489 [0.477, 0.502] full), and the reports use the one-sided leak label.
  - **D1:**
    - `holoassist` and `holoassist_audible` rebuild byte-identically;
    - the pool has 20,000 items, aligned item by item with its pseudo-labels;
    - the labeler reproduces 0.6539 [0.6257, 0.6749].
  - **D2:** the learner, the soft-label weights, the label-free scaler, the heads, the paired draws with the pinned seeds, the group bootstrap and `h3_verdict` all match the spec.
  - **D3:**
    - the download was verified exactly;
    - the label rule has **0 violations** over the 5,517 items, the groups are correct, and none straddles the split;
    - every item has its u9 features, and the judge covered all 1,517 test items.
  - **D4:** all 44 rows plus D1's 2; the leak check passes; the verdict follows the rule.
- **Findings:**
  - **(a) The test is weaker than intended, through two flaws in the designer's spec:**
    - the pool lost its mistakes (0.52%);
    - the prior's strength was its uncontrolled norm.

    The exploratory diagnostics point at the human data itself: every human head is at chance on robot videos. → **Issue 10**; L15, L16.
  - **(b) An unspecified change, accepted:** the D2 commit changed `shared/memguard.py`'s logged `waited_ms` (0 iff admitted on the first check, instead of under 50 ms). Admission behavior is unchanged.
  - **(c) Document defects, fixed by the designer:**
    - Issue 9's decision record was overwritten and left in the open section (moved and restored);
    - `02_data_sources.md` gave the RoboReward duration range as 0.9–82.7 s. The 5,517 items actually run 0.1–134.6 s (median 7.7 s), with 519 under 1 s.
  - **(d) A stray file, left in place:** `DATA_ROOT/judge/roboreward/gemma4_26b/scores.jsonl` duplicates the judge cache (1,517 lines). No code reads it, and it is harmless.

**October 21, 2026: designer verification of A10, A9b, A11, A12, A13 and A14** (each item's code and outputs read against its contract; the key numbers recomputed independently from the caches).
- **Verified correct:**
  - **A10:** every SSD glob goes through `shared.fsutil.data_glob`, and the agent ran G2 3/3 at every item.
  - **A9b:** the summary's false statements were removed, and the HoloAssist report uses §11.8.
  - **A11:** it reproduced every reference value to 3 dp (step-prior 0.751 [0.726, 0.774]; action-fusion 0.682; Oops! action-fusion 0.779; Oops! ablation Δ +0.015 [−0.001, 0.029]), and the old values are unchanged.
  - **A12:** 1,673 sessions, 548 audible. The A8 sessions reproduce to within 0.0046 dB.
  - **A13:**
    - 4,420 items (1,420 test, 53 groups), with groups on the same side of the split as in `holoassist`;
    - all five shuffled controls cover 0.5;
    - the designer reproduced fusion, action-fusion and the ablation Δ to 4 dp from the caches.
  - **A14:** Wave D's start condition was evaluated by the rule: **met** on `holoassist_audible`, +0.021 [0.006, 0.034].
- **Findings:**
  - **(a) A failed check that went unreported, now resolved by a design update:**
    - On `holoassist`, `step-prior:shuffled` = 0.464 [0.440, 0.489]. Its CI excludes 0.5, which A11's validation required, and its report row says `FAIL`, but the A11 commit and Issue 7's resolution do not mention it.
    - The designer's permutation analysis shows it is chance, not a leak: over 50 seeds the mean is 0.496 (SD 0.030) and seed 0 sits at the 20th percentile.
    - The rule is now one-sided (`03_eval_harness.md` §11.9; L14). No code change is needed beyond A15's report label.
  - **(b) Where the gain comes from** (designer, exploratory; these strata were not pre-declared). The ablation gain is mostly *whether the instructor spoke*:
    - instructor spoke (837 test items): +0.012 [−0.004, 0.030];
    - instructor silent (583): −0.026 [−0.044, −0.009];
    - without the `drop` items: +0.025 [0.011, 0.039], so it is not the action's own sound;
    - the pre-declared `react-nonverbal|spoke=1` = 0.582 [0.548, 0.612] shows some information in *how* the instructor sounds.

    Also, fusion (0.671) loses to the step-prior alone (0.737): fusion − action_best = −0.065 [−0.113, −0.025]. Whether this is enough to start Wave D is **Issue 8**.
  - **(c) A wrong number in the audible report:** its caveat line hard-codes "judge-large scores 0.489 … not drop", which is the full-HoloAssist number. On `holoassist_audible` it is **0.498 [0.481, 0.514]** (n = 1,361). The A9b spec mandated that literal. → guide item **A15**.
  - **(d) Accepted:** `holoassist_audible`'s `not_audible` count includes the actions of the 85 sessions without video (`02_data_sources.md` → As run (A13)).
  - **(e) The guide was stale:** its title said Queue Complete, but its body still described A10–A14 as work to build. It has been rewritten.

**October 10, 2026: designer verification of A1–A7** (battery re-run bare; every item's code read against its contract).
- **Verified correct:** A1–A5 as specified, and A6 except one gap: the frontier judge does not retry an unparsable answer (§8). That is fixed in **A7b**.
- **Landed with defects:**
  - **A6b (memory guard):** guide item **A6c**, three defects:
    - a nested `guard()` locks itself out and holds the machine-wide lock for 30 min;
    - `read_memory()` fails open;
    - memory deferrals use up the supervisor's attempt budget, so the 72-deferral rule can never fire.
    - Fast tests also call the live Ollama server.
  - **A7 (Oops!):** its numbers are sound and trace to the caches. The report is wrong in three places, all fixed by guide item **A7b**:
    - it says the prediction "holds" although the judge, which was predicted to be strong, scored 0.472;
    - it counts `errors.jsonl` rows (HTTP 429) as "parse failures";
    - it has no wall-clock times, and its example descriptions are template text.
- **In progress (October 10):** **A8 (HoloAssist).** At the maintainer's request, the draft was committed as found in `949c180` (incomplete). The video download finished October 10 at 13:01: 2,111 videos, archive deleted. The draft needs the corrections listed in guide item A8.

**October 20, 2026: designer verification of A6c, A7b, A7c, A8 and A9** (battery re-run bare; each item's code and outputs read against its contract).
- **Verified correct:**
  - **A6c:** the October 10 nested-guard probe now resumes in 0.00 s and leaves the lock free; reads fail closed; deferrals no longer use supervisor attempts.
  - **A7b:** the generated Oops! report applies the clause rule correctly.
  - **A7c:**
    - the Oops! re-score left every earlier value identical to 4 decimals;
    - all 77 `unload` events in `memguard.log` through October 15 are our 7B judge's; `gemma4:26b` was never unloaded;
    - through October 15, the shared-model check stopped the large judge cleanly 10 times under memory pressure.
  - **A8:** every draft correction landed:
    - the `holoassist:` ids;
    - `--force-split`;
    - a computed group-overlap count;
    - `missing_duration`;
    - the slow markers;
    - the audio check without sentinels;
    - Issue 6 filed exactly as specified.
  - **The October 19 crash report** names no process of ours (it is a per-process-limit kill of `IMTranscoderAgent`).
- **Landed with defects, all fixed by guide item A9b:**
  - **The A9 summary** (`docs/evals/2026-10-15_wave_a_summary.md`) has wrong statements:
    - **the bootstrap:** it says "10,000 cluster-bootstrap draws". The code draws 1,000 resamples; 10,000 is only the draw ceiling;
    - **the large judge's verdict:** it marks `judge-large` "held" on Oops! (0.708). By the pinned rule (`ci_low > 0.5` and AUROC ≥ action-probe − 0.05 = 0.729) it **did not hold**, which is what the generated report says;
    - **query and error counts:** it reports "zero API errors across 7,708 VLM queries". The real counts are 10,780 local answers plus 1,024 Gemini 429s and 2 ffmpeg errors;
    - **compute time:** it puts the HoloAssist 7B judge at 56,986 s; the cache sums to 40,657 s;
    - **the leakage condition:** it headlines `react-full` (transcripts, the leakage condition) as proof that "reaction signal exists and overcomes action probes", against `03_eval_harness.md` §11.2 and with no paired Δ.
  - **The HoloAssist report** uses `react-full` as the "best reaction" in its pass-bar line.
  - **The Issue 5 resolution note** said Clause 1 "now holds" for `judge-large`. That note is corrected below.
- **New defect (A10):** the external SSD is exFAT. macOS attaches a `com.apple.provenance` attribute to new files, so exFAT grows `._<name>` AppleDouble files beside them.
  - A test that globs `*.jsonl` failed 1 of 3 fast-suite runs with `TMPDIR` on the SSD, as §0.2 of the guide requires.
  - Five production globs over `DATA_ROOT` have the same exposure.
- **New decisions for the maintainer:** Issue 7, plus revised options for Issue 6.

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

**Option C: measure first via the API, then decide by a fixed rule** *(this was the recommendation until October 9; the current one is the Recommendation below)*
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

**New, October 22, 2026: routes that need no email** (designer research at the maintainer's request: *"can you brainstorm any other way of collecting videos without emailing?"*). The web checks were cut short by a search usage limit, so anything marked *unverified* is re-checked before it goes into a spec.

**Option F: Ego-Exo4D's expert commentary and proficiency labels.** Access is an online license agreement, the same kind the maintainer accepted for Ego4D (*the access flow is unverified*).
- **What it is:** a large multiview dataset of skilled activities (cooking, bike repair, basketball, soccer, dance, music, bouldering, health procedures). Coaches and teachers recorded spoken **commentary while watching** each performance, and the performances carry **proficiency labels**.
- **Why it fits:** it has the verdict-video structure: an evaluator reacts to what they watch, and there is a verdict.
- *Pros:*
  - no email;
  - large;
  - first- and third-person video;
  - the maintainer has used this kind of licensing before.
- *Cons:*
  - the commentary is mostly **words** (the leakage condition), so the non-verbal signal is the commentator's voice only;
  - **who assigned proficiency, and from what, needs the independence check HoloAssist got in A4**. If the commentators set the labels, the test is "does their tone predict their own rating";
  - skill is partly visible, so this is not a purely hidden outcome.

**Option G: taste-and-rate podcasts through their RSS feeds** (mostly audio; some shows publish video)
- **What it is:** shows where hosts taste food or drink on air and then score it ("7 out of 10"). RSS feeds are published for anyone to download, unlike YouTube, whose terms forbid downloading.
- *Pros:*
  - no email, and no platform terms against downloading;
  - a **hidden outcome** (taste);
  - verdicts parsed automatically from the transcript (the Wave B pipeline);
  - potentially large (*pool size unverified*).
- *Cons:*
  - **mostly no face**: voice only, which Wave A showed is mainly "did they speak";
  - copyright still applies (research use only, nothing redistributed). For an unaffiliated researcher the risk is personal, and this is not legal advice;
  - first-taste moments are harder to find in audio than in video.

**Also found, but not for Wave B:**
- **REFLEX** (KTH; [Zenodo 14160783](https://zenodo.org/records/14160783); **open access, CC BY 4.0**; verified October 22): camera video of 55 people reacting to a robot's **failures** and its explanations, with per-failure annotations and questionnaires.
  - A 5-person sample is 3.7 GB, and the full data is listed at 1.1 TB. A newer version exists (*contents unverified*).
  - **It is a possible "reactions to robots" target without BAD's IRB requirement** (see Issue 4).
- **EMPATHIC** (Cui et al., CoRL 2020; [Zenodo 4290896](https://zenodo.org/record/4290896)): faces of people watching an agent act, with per-episode reward and optimality, the closest prior work to this thesis.
  - Its search listing says *restricted* (access through Zenodo's request button; *unverified*), so someone must approve it.
- **Checked and not usable without email:** the beverage-tasting database (2,970 videos) has no public download; AM-FED+ needs an emailed, signed license (that is option E).

**Recommendation (revised October 22, 2026):** **check F and G first, then choose** (the designer, one short session once web search is available again):
- confirm Ego-Exo4D's access flow, and who assigned its proficiency labels;
- count taste-and-rate podcast episodes through Apple's public podcast search, which needs no key or account.

E stays available whenever you are willing to send a request; it is the only route with face reactions to a hidden outcome. *The earlier recommendation, kept for the record:*

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

**New fact (October 22, 2026):** an **openly licensed** dataset of people reacting to robot failures exists. REFLEX (KTH; [Zenodo 14160783](https://zenodo.org/records/14160783); CC BY 4.0; 55 participants; per-failure annotations) has no IRB or email requirement. It could give a "reactions to robots" result without BAD. It will be assessed in the H2 spec (its labels, size and the newer version's contents are not yet verified). The decision below is unchanged.

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

## 🧪 Resolved Issues & Implementation Refinements

Resolved issues keep their numbers. The full evidence now lives in the design docs that each entry points to.

1. **Issue 2: HoloAssist label independence (Resolved - October 9)**:
   - **Problem**: If HoloAssist's mistake labels had been assigned *from* the instructor's interventions, "the instructor's reaction predicts the mistake" would be circular, and the 184 GB video download would have bought a meaningless number. The protocol was unknown before the labels were downloaded.
   - **Solution**:
     - Item A4 downloaded only the 111 MB labels, with hashes re-verified by the designer on October 10, and read the README and the paper's annotation protocol (arXiv 2309.17024 §3.2). Mistakes are annotated by third-person annotators from whether the performer's action achieves the task, each with a written explanation. The link to an instructor's correcting sentence is a separate mapping step added afterwards.
     - 43.8% of mistakes (3,153 / 7,205) had no instructor correction at all. Instructor speech overlapped `[start, end + 5.0]` for 67.98% of mistakes (4,898 / 7,205) vs. 30.87% of correct actions (43,736 / 141,691).
     - The official splits are **not** participant-disjoint (236 of 340 session prefixes straddle them), so A8 uses a grouped 70/30 split.
     - Verdict: independent. The verbatim quotes, schema and counts are in `02_data_sources.md` → HoloAssist → "Schema (as downloaded, October 9, 2026)".
2. **Issue 3: SSD capacity for Wave A downloads (Resolved - October 8)**:
   - **Problem**: The Extreme SSD had 311 GiB free, not enough for Oops! (45 GB plus extraction) and HoloAssist's pitch-shifted videos (184 GB plus extraction) under the ≥ 50 GiB-free rule.
   - **Solution**: The maintainer selected cleanup (*"Yes, clean up any videos you want from the v0 leftovers. Feel free to delete Ego4D if you think that is the right choice."*).
     - **Deleted:** the designer deleted the v0 **videos** only, 36,506 files and 1.336 TB:
       - the 1,083 Ego4D clips under `social_robotics/raw_videos/ego4d/v2/full_scale/` (1.2 TB);
       - Charades-Ego `ego_videos/` and `tp_videos/`, `CharadesEgo_v1_480.tar` and `Charades_v1_480.zip`.
     - **Kept:** `ego4d_data/` (annotations plus 68 GB of Omnivore features), the Ego4D and Charades-Ego annotations, `full_run_2026_06_18/` (13 GB) and `bench_v0/` (4.5 GB). The Wan2.1 weights (102 GB) and `saf_env/` (19 GB) are dead weight, but they are not video, so they were left for a separate decision.
     - **Manifest:** every path plus the Ego4D clip ids, for re-download, is in `DATA_ROOT/DELETED_2026-10-08.json`.
     - **Why Ego4D was deleted rather than pruned:** nothing in Waves A or B uses it, and its bystander footage is the wrong data for H1.
     - **Result:** free space went to 1.5 TiB (1.4 TiB on October 10, with Oops! extracted and the HoloAssist archive downloading).
3. **Issue 5: Large local judge (`gemma4:26b`) replaces stalled frontier judge on Oops! (Resolved - October 14, A7c)**:
   - **Problem**: The 7B local judge (`qwen2.5vl:7b`) scored at chance on Oops! (AUROC 0.472 [0.440, 0.498]), while the frontier judge (`gemini-3.6-flash`) was blocked by free-tier daily rate limits (20 req/day). The central control ("just ask an LLM") lacked a strong measurement.
   - **Solution**: Option B (larger local judge) was selected by the maintainer and implemented in A7c. `gemma4:26b` (`vlm_judge_large`) was evaluated on all 1,072 Oops! test split items under memory guard (`sr_judge_large_load`, declared peak 33 GB, measured 28.10 GB RSS).
   - **Outcome**: 1,072 items scored, 0 excluded, 0 parse failures, 0 API errors (41.8 min wall-clock). `judge-large` AUROC is **0.708 [0.684, 0.731]**, showing that a sufficiently large local VLM reliably recognizes visible failures (answer distribution: 78.0% 100s on pre-failure, 63.4% 0s on post-failure). Because `action-probe` (SigLIP) achieves 0.779, `action_best` remains `action-probe`, and $\Delta = +0.015$ [-0.002, 0.029] is completely unchanged. Clause 1 of the thesis prediction now holds for `judge-large` ($0.708 > 0.50$).
   - **Designer correction (October 20)**: "Clause 1 … now holds for `judge-large` (0.708 > 0.50)" was wrong.
     - The pinned rule (`03_eval_harness.md` §11.8) is: held iff `ci_low > 0.5` **and** AUROC ≥ action-probe − 0.05 = 0.729. So it **did not hold**, which is what the generated Oops! report correctly prints.
     - On HoloAssist, `judge-large`'s 0.544 comes entirely from reading the verb "drop" (0.489 on the other test items; Issue 7).
     - The resolution itself stands: the large judge exists, it is measured, and it is the stronger local control on visible failures.
4. **Issue 7: HoloAssist's step names leak the outcome; step-prior control and ablation delta (Resolved - October 20, A11)**:
   - **Problem**: HoloAssist's step names leak the outcome (`drop` is a mistake 95% of the time). A text-only train-split step prior scores 0.751 on test, beating every action-only control. Furthermore, combining high-dimensional inputs with step-prior dilutes the signal (action-fusion scores 0.682, −0.069 below step-prior alone), making `fusion − action_best` fail regardless of reaction signal.
   - **Solution**: Option A selected by the maintainer. Implemented in A11:
     - `step_prior_scores` with hierarchical Bayesian smoothing ($m=5$, base rate $p_0 \to$ verb $p_v \to$ verb+noun $p_{vn}$) and leave-one-out (LOO) on train.
     - `action-fusion` control (judge + action-probe + step-prior) and `fusion_minus_action_fusion` (ablation $\Delta$).
     - Shuffled-label controls (`step-prior:shuffled`, `action-fusion:shuffled`, `fusion:shuffled`) recompute step-prior from permuted train labels so no label leakage occurs.
   - **Outcome**: Exact reference value reproduction (§1.5):
     - HoloAssist `step-prior`: 0.751 [0.726, 0.774].
     - HoloAssist `action-fusion`: 0.682 [0.658, 0.706].
     - Oops! `action-fusion`: 0.779 [0.754, 0.805].
     - Oops! ablation $\Delta$: +0.015 [−0.001, 0.029].
     - Reports regenerated (`docs/evals/2026-10-10_oops_h1.md`, `docs/evals/2026-10-15_holoassist_h1.md`). Battery G1–G3 green.
5. **Issue 6: Instructor not audible over background manipulation in HoloAssist video audio (Resolved - October 21, A12 & A13)**:
   - **Problem**: The audio presence check failed on the full HoloAssist dataset (+1.51 dB vs $\ge +3.5$ dB requirement) due to loud manipulation noises on the headset mic during tool-heavy assembly/disassembly tasks, blocking the non-verbal reaction channel and HoloAssist's $\Delta$ computation.
   - **Solution**: Option A (audible-session evaluation set, fresh sample) was selected by the maintainer.
     - A12 measured audibility label-blind across all 1,673 annotated sessions with video (`tools/check_audio_presence.py --all-sessions`): 548 audible (32.8%), 1,125 inaudible (67.2%), 0 undefined.
     - A13 built `holoassist_audible` as a fresh sample (4,420 items: 3,000 train / 1,420 test across 53 test groups) from the 548 audible sessions and evaluated it end-to-end under memory guard.
   - **Outcome**:
     - `react-nonverbal` achieves AUROC **0.654 [0.626, 0.675]** (meeting the $\ge 0.65$ bar).
     - Diagnostic `react-nonverbal|spoke=1` scores **0.582 [0.548, 0.612]** on the 837 test items where the instructor spoke.
     - `action-fusion` achieves **0.650 [0.599, 0.694]** and full `fusion` achieves **0.671 [0.618, 0.713]**.
     - **Ablation $\Delta$ (`fusion_minus_action_fusion`):** **+0.021 [0.006, 0.034]** — **`ci_low > 0` (Held)**! Reactions demonstrably add information to the action-only model on hidden outcomes.
     - All 5 shuffled controls cover 0.50 (chance). Report generated at `docs/evals/2026-10-21_holoassist_audible_h1.md`.

6. **Issue 8: Wave D's start condition was met narrowly; start the Wave D spec? (Resolved - October 21, the maintainer chose option A in chat)**:
   - **Problem**:
     - The amended H3 trigger was met on `holoassist_audible`: ablation gain +0.021 [0.006, 0.034].
     - The H1 pass bar (≥ +0.03) was not, and fusion lost to the step-prior (−0.065).
     - The designer's exploratory strata showed the gain is mostly whether the instructor spoke: +0.012 [−0.004, 0.030] where they spoke, −0.026 where they were silent, and +0.025 without `drop`.
     - Starting Wave D commits weeks of work and a download on modest evidence.
   - **Solution**: Option A: the designer wrote the Wave D spec, `04_wave_d_robot_check.md`, and pre-registered it before any robot data is touched.
     - It is scoped to what Wave A showed: the reaction labeler is the `holoassist_audible` `react-nonverbal` probe.
     - The H3 pass and null rules and their interpretations are written first.
     - A shuffled-pseudo-label prior separates "reaction labels help" from "any human-video prior helps".
     - The download it needs is **Issue 9**.

7. **Issue 9: How much of RoboReward to download, and whether to run RoboReward-8B (Resolved - October 22, D3; decided by the maintainer in chat October 21: option B)**:
   - **Problem**: Wave D needed RoboReward on disk. The options were:
     - a selective ~4–6 GB (A);
     - the whole 43.2 GB (B);
     - either plus RoboReward-8B (C), which needs a `transformers` newer than the frozen 4.46.3.
   - **Decision**: **B** (*"For issue 9, lets go with Option B"*).
     - **Why B over A:** the experiment reads the same videos either way. B cost about 20 more minutes and 40 GB of a 1.3 TiB disk. In exchange, the download path has no file-selection code, and val and the full train split stay on hand.
     - **RoboReward-8B is not run.**
   - **Solution (D3):** one `snapshot_download`, verified file by file against the Hub's metadata.
     - **54,140 files, 43,204,474,744 bytes, 54,135 `.mp4` files** at revision `469b9af76e8539e8d2ac553b081307738fd92ca5`; 1,135.5 GiB free afterwards.
     - **Metadata sha256:** train `f0a21f02…`, val `73710f1a…`, test `3901cea9…` (full hashes in `DATA_ROOT/raw/roboreward/DOWNLOAD.json` and `02_data_sources.md`).
   - *History:* the D3 agent rewrote this entry's status in place and left it in the open section. The designer moved it here on October 22 and restored the decision record.

8. **Issue 10: After Wave D v1's null: accept it, run a corrected v2, or close offline H3? (Resolved - October 22, the maintainer chose option A in chat)**:
   - **Problem**:
     - Wave D v1 (the offline robot check, H3) was **NOT SUPPORTED** by its pre-registered rule: `gain:react:N=100` +0.0009 [−0.0006, 0.0024]; `gain:truth:N=100` −0.0507.
     - The designer's verification found two flaws in the spec: the pool's mistakes were depleted (0.52%), and the prior's norm was uncontrolled (`react` 0.99 vs `truth` 26.9).
     - Exploratory diagnostics showed every human head at chance on robot videos (0.49–0.52), and even a well-posed true-label head at only 0.550 [0.515, 0.585]. So the binding constraint is the human data's domain, not the flaws.
     - The full facts are in `04_wave_d_robot_check.md` §8.
   - **Options:**
     - **A:** accept v1, revisit after Wave B with a closer human source;
     - **B:** a corrected v2 now, about a day of work, with a null predicted;
     - **C:** accept v1 and close offline H3.
   - **Decision**: **A** (*"For issue 10, lets go with Option A"*).
     - **v1 is Wave D's result,** with its flaws and diagnostics recorded beside it.
     - **No v2 is run now.**
     - **Robot transfer is revisited only through a new, pre-registered spec, and only after Wave B.** That spec uses Wave B's reaction model and a human video source closer to robot manipulation (deferred item **DW7**).

---

## Maintainer actions (not agent work)

| Id | Action | Why | Status |
|---|---|---|---|
| M1 | Request **BAD dataset** access (QDR account → Request Access / "Contact Owner"), asking whether an independent researcher can qualify, and email the **ERR@HRI 3.0** organizers about post-challenge access | H2 targets: reactions to robots. BAD requires an IRB-reviewed protocol from an affiliated institution (Issue 4). Drafts: `maintainer_access_requests.md` M1a/M1b | Open |
| M2 | Add an "Archived (October 2026)" note to the 6 public `louisye/social-robotics-*` Hugging Face cards | They describe the v0 pipeline as current | ✅ Done October 9 (wording confirmed by the maintainer; 6 HF commits) |
| M3 | Create a **YouTube Data API v3** key (a Google Cloud project with the API enabled) | Only for *counting* (Issue 1, option C); never for downloading | Optional |
| M4 | Sign and send the **AM-FED+** EULA | Issue 1 option E: reactions + self-reported liking | Open, pending the Issue 1 selection |
| M5 | Ask the authors of the **taste-liking** database for access | Issue 1 option E | Optional |
| M6 | Enable billing on Google AI Studio project for `GOOGLE_API_KEY` | Free Tier daily ceiling is 20 req/day for `gemini-3.6-flash`; 1,072 test items require paid tier for full `judge-frontier` run (11 evaluated before 429) | Not needed: Issue 5 was decided as **option B** (a larger local judge) on October 10. The frontier judge stays on the free tier |

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
- **L10: A report can contradict its own table.** The A7 report declared "the prediction holds" from the Δ row alone, while its own table showed that the other half of the prediction (a strong judge) had failed at AUROC 0.472. A prediction check names **every clause** of the prediction and its measured value, one line each. Report text that summarizes numbers is generated from the rows, never hard-coded, and is checked against them.
- **L11: A lock that its own holder can re-enter needs to know it.** The memory guard's between-item path re-enters `guard()` and then calls a reload that enters `guard()` again. A non-re-entrant file lock then waits on itself for 30 min, while blocking the other project. Its unit test passed only because it stubbed `reload()` with a function that never takes the lock (L8). Test the composed path with the real callables.
- **L12: An annotation's own vocabulary can carry the label.** HoloAssist's step verb `drop` is a mistake 95% of the time, and a train-split verb prior scores 0.721 on test, beating every action-only control. A banned-word list cannot catch this. For any text an item carries (`context_text`, step names, task names), fit a text-only prior on train and report its test AUROC before trusting the action-only controls.
- **L13: The SSD is exFAT, and macOS tags new files.** A `com.apple.provenance` attribute on a new file makes exFAT write a hidden `._<name>` twin. Any `glob("*.jsonl")` or `rglob("*.mp4")` over `DATA_ROOT` or a temp dir on the SSD can pick it up. One test failed 1 run in 3 that way. Skip names starting with `._` everywhere a glob touches the SSD.
- **L14: A single permutation is not a null distribution.** One shuffled-label run's bootstrap CI covers only the resampling of test items. A small control (step-prior has a few dozen verbs) lands outside it about 1 time in 5 with no leak at all. The leak check is one-sided (`ci_low > 0.5` fails; §11.9). A result that misses a stated validation bar is still reported in the commit body, even if the agent believes it is benign: here a `FAIL` sat in a report and nobody mentioned it.
- **L15: A pool built by excluding an evaluation sample can lose its rare class.** Wave A's balanced sample had taken about 92% of the audible sessions' mistakes. Excluding it left Wave D's pool at 0.52% mistakes, and with none in the sessions the labeler never saw. Before using any derived pool, check its class counts per stratum.
- **L16: A prior pulls with its norm, so compare priors at equal strength, and check a head's zero-shot transfer first.**
  - Wave D's reaction and shuffled priors had norm ~1 and moved predictions by 0.015; the true-label prior had norm 27 and moved them by 0.33.
  - All three heads were at chance on robot videos by themselves, which a one-line check would have shown before the curve was built.

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
- A6b — Memory guard (admission, heavy lock, between-item watchdog) — git log --grep "(a6b)" — shared.memguard admission guard with FLOOR=8 GB, shared heavy lock with animated_infographics, between-item check, exit 75 handling in CLI, supervisor and battery, 50-item peak re-measurements confirmed (siglip 1.60 GB, e2v 4.94 GB, judge 7.80 GB).
- A7 — Oops! end-to-end H1 (visible-outcome contrast) — git log --grep "(a7)" — 2,710 items across 1,355 clips (0 straddling groups); siglip action-probe AUROC=0.779 [0.754, 0.805], e2v react-nonverbal AUROC=0.711 [0.684, 0.740], fusion AUROC=0.793 [0.771, 0.816], fusion_minus_action_best delta=+0.015 [-0.002, 0.029] (prediction holds: zero in CI); all 3 shuffled controls cover 0.50; report at docs/evals/2026-10-10_oops_h1.md.
  - *Designer correction (October 10):* the numbers stand. "Prediction holds" covers only the Δ ≈ 0 clause; the "judge strong" clause **failed** (judge AUROC 0.472 [0.440, 0.498]). The report is corrected by A7b, and the judge question is Issue 5.
- R4 — Designer verification of A1–A7 (October 10, 2026), with the battery run bare:
  - **HEAD `6d72492`:** G1 exit 0, G2 exit 0 (41 passed).
  - **Working tree** (with the uncommitted A8 draft): G1–G4 exit 0 (48 passed, self-test 4/4, 5 slow passed).
  - **Findings:** the guide items A6c (memory-guard defects), A7b (report corrections and frontier parse parity), corrections to the A8 draft, and Issue 5.
  - **Design updated:** `03_eval_harness.md` §4–§6, §8 and §12, and `02_data_sources.md` (Oops! "As built"; HoloAssist "pinned details").
- A6c — Memory guard fixes (re-entrancy, fail-closed read, supervisor accounting, fast test isolation) — git log --grep "(a6c)" — G1–G4 exit 0 (52 fast passed, 5 slow passed; slow release measured with footprint < 1.5 GB); nested guard re-enters without deadlock (< 1.5s); read_memory fails closed to (0, 4); exit 75 does not consume supervisor attempts; fast suite isolated from live HTTP.
- A7b — Oops! report corrections and frontier-judge parse parity — git log --grep "(a7b)" — prediction_check tests pass; report regenerated with two clauses (Clause 1 did not hold [AUROC 0.472], Clause 2 held [Δ 0.015]), reclassified errors (qwen 2 ffmpeg, gemini 1,024 429 and 1 other), wall-clock timings (download 46.6 min, features 57.0 min, judge 398.2 min), verified descriptions for 10 sampled items, react-nonverbal caveat added; Gemini parse retries (outer loop temp 0/0.3) tested; scorecard unmodified; G1–G3 exit 0 (57 fast passed, self-test 4/4).
- A7c — Large local judge (judge-large, gemma4:26b) on Oops! — git log --grep "(a7c)" — Issue 5 resolved with option B; declared peak 33 GB (28.10 GB RSS measured); gemma4:26b evaluated on 1,072 Oops! test items (0 parse failures, 0 errors, 41.8 min wall-clock); answer distribution shows clean binary separation (pre: 78% 100s; post: 63.4% 0s); judge-large AUROC=0.708 [0.684, 0.731] (vs 0.472 for 7B judge); action_best remains action-probe (0.779); delta=+0.015 unchanged; G1–G4 pass.
- A8 — HoloAssist end-to-end H1 evaluation — git log --grep "(a8)" — Audio presence check failed (+1.51 dB vs >= +3.5 dB; Issue 6 filed; stopped audio extraction); split regenerated with canonical item_ids (3,000 train / 2,000 test balanced); siglip extracted (5,000 items, 2,779 s); qwen2.5vl:7b scored 5,000 items (AUROC 0.488 [0.466, 0.512]); gemma4:26b scored 2,000 test items (AUROC 0.544 [0.528, 0.562]); action-probe AUROC 0.569 [0.542, 0.597]; react-spoke AUROC 0.676 [0.655, 0.695]; react-full AUROC 0.739 [0.716, 0.761]; action-probe:shuffled AUROC 0.492 [0.461, 0.525] (PASS); report and TSV generated; G1–G4 exit 0.
- A9 — Re-measure; close out Wave A — git log --grep "(a9)" — Full Wave A synthesis across Oops! and HoloAssist; headline dataset x condition table citing scorecard timestamps; per-clause predictions checked; caveats, anomalies, compute times (~28.5 h total), and Wave B design recommendations documented at docs/evals/2026-10-15_wave_a_summary.md; G1–G4 green.
  - *Designer correction (October 20):* the summary has wrong statements (the bootstrap count, the `judge-large` verdict, the error counts, the 7B HoloAssist judge time) and headlines `react-full` against §11.2. These are corrected by guide item A9b.
- R5 — Designer verification of A6c–A9 (October 20, 2026), with the battery run bare on `3f0b78d`:
  - **G1–G4 results:**
    - G1 exit 0;
    - G2 62 passed without the SSD temp dir, but 1 run in 3 failed (61 passed, 1 failed) with `TMPDIR` on the SSD: the exFAT AppleDouble problem, L13;
    - G3 4/4;
    - G4 9 passed (134 s).
  - **Findings:** A9b (summary and report corrections), A10 (AppleDouble hardening), the Issue 6 options revised, Issue 7 filed, and the Issue 5 note corrected.
  - **Design updated:** `02_data_sources.md` → HoloAssist (the step-prior caveat, and the 85 sessions that are in no official split list).
- A10 — Ignore exFAT AppleDouble (._*) files in every glob over SSD — git log --grep "(a10)" — src/shared/fsutil.py with data_glob; 4 tools files and test_judge updated; test_fsutil passes; G2 passed 3/3 times (64 passed); G1–G4 exit 0.
- A9b — Correct Wave A summary and HoloAssist report — git log --grep "(a9b)" — prediction_check_hidden implemented per 03_eval_harness.md §11.8; test_report_holoassist passes; HoloAssist report and Wave A summary regenerated/corrected; G1–G3 exit 0 (67 fast passed).
- A11 — The step-prior and action-fusion controls and ablation delta — git log --grep "(a11)" — step_prior_scores with hierarchical smoothing (m=5, LOO); action-fusion and ablation delta (fusion_minus_action_fusion) implemented; §1.5 reference values reproduced exactly (HoloAssist step-prior 0.751 [0.726, 0.774], action-fusion 0.682 [0.658, 0.706]; Oops! action-fusion 0.779, ablation delta +0.015 [-0.001, 0.029]); reports regenerated; G1–G3 exit 0.
- A12 — Per-session instructor audibility over all 1,673 sessions — git log --grep "(a12)" — tools/check_audio_presence.py with --all-sessions; 1,673 annotated sessions with video measured detached (PPID 1, elapsed 418.4s); 548 audible (32.8%), 1,125 inaudible, 0 undefined; task breakdown documented in 02_data_sources.md; G1–G3 exit 0 (73 passed 3/3).
- A13 — The audible-session evaluation set holoassist_audible, a fresh sample, run end to end — git log --grep "(a13)" — 4,420 items sampled from 548 audible sessions (3,000 train / 1,420 test across 53 groups); siglip, e2v, qwen2.5vl:7b (train/test), gemma4:26b (test) evaluated end-to-end (0 exclusions, 0 errors); react-nonverbal AUROC 0.654 [0.626, 0.675]; action-fusion AUROC 0.650 [0.599, 0.694]; fusion AUROC 0.671 [0.618, 0.713]; ablation Δ +0.021 [0.006, 0.034] (ci_low > 0, Held); all 5 shuffled controls cover 0.50; report and TSV generated; G1–G4 exit 0.
- A14 — Close out Wave A — git log --grep "(a14)" — Wave A completed across Oops!, HoloAssist, and HoloAssist-Audible; headline table, prediction checks, and compute times documented in summary; Wave D start condition evaluated as MET (holoassist_audible ablation Δ +0.021 [0.006, 0.034], ci_low > 0, ts: 2026-10-21T16:50:26Z); guide retitled to Queue Complete; battery green.
- A15 — Computed judge-large caveat and one-sided leak check — git log --grep "(a15)" — judge_nondrop_auroc reproduces 0.498 [0.481, 0.514] (audible) and 0.489 (full); leak_check_label implements one-sided rule (§11.9); reports and summary regenerated; G1–G3 exit 0 (78 fast passed 3/3).

- R6 — Designer verification of A10–A14 (October 21, 2026):
  - all six items match their specs, and the key numbers were recomputed independently;
  - **findings:** a shuffled-control `FAIL` that went unreported (chance; rule made one-sided, §11.9, L14), a hard-coded caveat number (A15), and the Wave D go/no-go (Issue 8);
  - **design updated:** `03_eval_harness.md` §11.9, `02_data_sources.md` (As run, A13), and `00_thesis.md` entry 21.

**Wave D:**
- D1 — The HoloAssist pretraining pool, the reaction labeler and the pseudo-labels — git log --grep "(d1)" — 20,000 pool items (104 mistakes, 0.52% natural rate, 0 audible overlap); siglip & e2v features 20,000/20,000 (0 exclusions, 0.0%); refit labeler reproduces reference AUROC 0.6539 [0.6257, 0.6749] on holoassist_audible test split; pseudo-label-quality:all AUROC 0.6210 [0.5684, 0.6739], pseudo-label-quality unseen null (all 5,682 items correct; test split took all mistakes); battery --slow exit 0.
- D2 — The Wave D code: transfer.prior_lr, heads, curve, encoder siglip-b16-224-u9, sources.roboreward, tools/download_roboreward.py, h3_verdict and tools/report_wave_d.py — git log --grep "(d2)" — prior_lr with w0=None matches sklearn LogisticRegression test AUROC within 1e-4; 21 synthetic D2 unit tests pass; G1–G3 exit 0 (104 fast passed 3/3).
- D3 — RoboReward: the full download, items, split, robot features and the judge-large reference — git log --grep "(d3)" — full download verified at 54,140 files, 43,204,474,744 bytes, 54,135 .mp4 files (revision 469b9af76e8539e8d2ac553b081307738fd92ca5, 1135.5 GiB free); items/roboreward/ built (4,000 train pool [2,000 success, 2,000 failure, shortfall 0], 1,517 test [527 success, 990 failure], 0 straddling groups); splits/roboreward.json generated; siglip-b16-224-u9 features extracted for all 5,517 items (0 exclusions, 0.0%); judge-large (gemma4:26b) reference evaluated on all 1,517 test items (0 exclusions, 100.0% coverage); G1–G4 exit 0.
- D4 — Wave D execution, report and close-out — git log --grep "(d4)" — Pre-registered verdict: NOT SUPPORTED. "Human→robot transfer does not help in this setup even with true human labels; reaction labels cannot be judged here." (scorecard rows ts: 2026-10-22T14:23:01Z; gain:truth:N=100 -0.0507 [-0.0779, -0.0240], gain:react:N=100 +0.0009 [-0.0006, 0.0024], react_minus_shuffled:N=100 +0.0008 [-0.0013, 0.0029]; leak check pass: 0.498 [0.491, 0.505]); report generated at docs/evals/2026-10-22_wave_d_robot_check.md; guide updated to Queue Complete; G1–G4 exit 0.
  - *Designer note (October 22):*
    - The verdict stands by the pre-registered rule, and the implementation is faithful.
    - Two flaws in the spec weaken the test: the pool's mistakes were depleted, and the prior's norm was uncontrolled.
    - Exploratory diagnostics show every human head at chance on robot videos (0.47–0.55, even with true labels).
    - What to do next is **Issue 10**.
- R7 — Designer verification of A15 and D1–D4 (October 22, 2026): all five items match their specs. The findings are Issue 10 (Wave D's two design flaws and the diagnostics), an accepted memguard logging change, two document fixes and a stray duplicate file. Design updated: `04_wave_d_robot_check.md` §8 (the v1 result and notes), `02_data_sources.md` (durations), `00_thesis.md` entry 25.

---

## 4. Deferred work (do not start without its trigger)

| Id | Work | Trigger to start |
|---|---|---|
| DW1 | **Wave B: verdict-video H1 pilot** (~200 videos; face + non-verbal audio; ASR verdict parsing) | Issue 1 selected **and** Wave A closed **and** the designer has written the Wave B spec. Wave A is closed (A14, October 21; A15 is a report fix only) |
| DW2 | Face encoder (`react-face`) | Part of the Wave B spec |
| DW3 | **H2 transfer** to HoloAssist / AM-FED+ (+ BAD only if the QDR request is granted); scaling curves | Wave B closed |
| DW4 | Ego4D v0 false-positive set | Wave B+. The raw videos were deleted October 8 (Issue 3); re-download the chosen clip ids from `DELETED_2026-10-08.json` |
| DW5 | **Wave D, the robot check (offline H3)**: a data-efficiency curve on RoboReward, with vs. without pre-training on reaction-labeled human task video (`00_thesis.md` H3) | Wave A shows reaction signal on ≥ 1 dataset **and** the designer has written the Wave D spec. **Status (October 21): met by the rule** on `holoassist_audible` (ablation gain +0.021 [0.006, 0.034]). **Done (v1, October 22):** pre-registered verdict **NOT SUPPORTED**, accepted as Wave D's result (Issue 10 = A). The follow-up is DW7 |
| DW6 | **Stage A live, as a personal demo with the maintainer as the only reactor.** The microduck learns *which of its existing behaviors, and which styles of them, the maintainer likes* from their live reactions. The Mac runs the reaction reader (a webcam on the maintainer, or the duck's front-camera WebRTC stream) and sends choices over the duck's control channel. Measures: trials-to-preference vs. explicit thumbs-up/down, and reaction-reader accuracy against a quick rating after each trial. **Not in scope:** learning motor skills from reactions (that needs millions of trials, in simulation). Facts (October 9, 2026): 15-DoF, 25 cm biped, RK3566 with 1 GB RAM, front camera, 7 shipped moves; new policies are trained in sim with `microduck_rl` (mjlab/MuJoCo Warp, PPO), which **needs a CUDA GPU**, i.e. not this Mac | Hardware acquired **and** H2 done **and** the maintainer opts in (deferred October 9: *"maybe we don't worry about the someone reacting to the robot yet"*) |
| DW7 | **Robot transfer, revisited (offline H3, a new pre-registered spec).** It uses Wave B's reaction model and a human video source closer to robot manipulation (e.g. third-person tabletop work with outcome labels). v1's lessons apply: check a derived pool's class counts per stratum (L15); compare priors at equal norm, and check each head's zero-shot transfer first (L16) | Wave B shows a reaction gain (an ablation gain with `ci_low > 0`) **and** a closer human source is identified **and** the designer has written the spec (Issue 10 = A) |
