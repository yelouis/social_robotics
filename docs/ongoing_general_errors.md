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
  - **Issues 6 and 7 were decided October 20 (option A for both). Two sub-choices were made in chat:**
    - a **fresh** audible-session sample;
    - the pass bar measured as the **ablation gain**.

    They are being implemented by guide items A11–A13 (`00_thesis.md` entry 20).
  - Issue 1 blocks Wave B only.
  - Issue 4 was decided October 9: the BAD request, else no robot-reaction target.
- **Resolved** (moved to 🧪 below): Issue 2 (HoloAssist labels are independent; October 9, A4), Issue 3 (SSD space; October 8), and Issue 5 (large local judge `gemma4:26b` evaluated on Oops!, October 14, A7c).

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

### Issue 6: Instructor not audible over background manipulation in HoloAssist video audio (audio presence check: +1.51 dB vs >= +3.5 dB requirement)

**Status**: ✅ **Decided: option A, by the maintainer, October 20, 2026.**
- **Sub-choice, made in chat:** build the audible set as a **fresh sample** (≈ 3,000–4,000 items, ≈ 1,000–1,300 of them test), not as a subset of the existing 5,000.
- **Definition:** `02_data_sources.md` → "Audible-session evaluation set".
- **Implementation:** guide items A12 (per-session audibility) and A13 (`holoassist_audible`). The entry moves to 🧪 Resolved when A13 lands.

*Filed by the implementing agent in A8 (October 15), exactly as the guide required.* **The options were revised by the designer on October 20**, after verification showed what the failure costs (facts 4–6). The agent's original options are kept at the end of this entry.

**What is blocked:** `react-nonverbal`, `fusion` and therefore **HoloAssist's Δ row**. The non-audio conditions (`judge`, `judge-large`, `action-probe`, `react-spoke`, `react-full`) are scored.

**Facts (October 15, 2026):**
1. **Audio presence check failed the $\ge +3.5$ dB threshold:** `tools/check_audio_presence.py` was run on 20 randomly sampled sessions (`default_rng(0)`, sorted by `video_name`, restricted to sessions with $\ge 1$ instructor utterance and $\ge 1$ no-utterance gap $\ge 2$ s, where video exists).
   - **Median per-session difference:** **+1.51 dB** (requirement: $\ge +3.5$ dB). Verdict: **FAIL**. Summary saved to `DATA_ROOT/runs/holoassist_audio_presence.json`.
2. **Breakdown across tasks:**
   - In quiet manipulation tasks (e.g. `z129-aug-11-22-dslr`, `z094-july-11-22-switch`, `z195-sep-17-22-gopro`), instructor utterances are clearly audible and substantially louder than silence (+14.32 dB, +8.41 dB, +8.05 dB).
   - In noisy manipulation tasks involving tools and furniture assembly/disassembly (e.g. `z202-sep-23-22-knarrevik_disassemble`, `z046-june-24-22-knarrevik_disassemble`, `z175-sep-05-22-gladom_disassemble`), long unannotated periods contain loud table thuds, metal clattering, screwdriver ratchets, and part handling (-41 to -45 dBFS). This is louder than the instructor's pitch-shifted voice (-50 to -55 dBFS), yielding differences down to -10.36 dB.
   - The microphone is on the wearer's HoloLens headset, so hands-on tool noise inches from the mic physically overwhelms the instructor's voice (who is sitting at a remote screen or further away).
3. **Execution guide requirement (§3 A8.1):**
   > *"Requirement: a median per-session difference $\ge +3.5$ dB. If it fails, **STOP**, and file **Issue 6** ('instructor not audible in the video audio') with options A (`react-spoke`/`react-full` from annotations only) and B (drop HoloAssist's audio channel), and a `Your selection: _____` line. Continue only with what does not use audio."*

4. **Per-session spread (designer, from `DATA_ROOT/runs/holoassist_audio_presence.json`):**
   - **6 of the 20 sessions pass +3.5 dB on their own:**
     - five device setups: `z129 dslr` +14.3, `z094 switch` +8.4, `z195 gopro` +8.0, `z167 gopro` +5.4, `z035 printer_big` +4.5;
     - one stool disassembly: `z124 marius` +5.9.
   - Two more are just under: `z108 gladom` +3.47 and `z189 printer_big` +3.15.
   - The furniture sessions (`knarrevik`, `gladom`, `rashult`) range from −10.4 to +3.5 dB. The other device sessions (`R076`/`R099` DSLR, `nespresso`, `ATV`) sit at +0.5 to +1.7 dB.
   - So the instructor is audible in roughly a third of sessions, mostly the quieter device tasks.
5. **What it costs today:** HoloAssist was Wave A's only hidden-outcome test, and its Δ row reads `not run: fusion not run`.
   - Oops!'s Δ is +0.015 [−0.002, 0.029], which includes 0.
   - So **Wave A has not shown a reaction gain on any dataset.** The start condition for Wave D (the offline robot check, DW5: "Wave A shows reaction signal on ≥ 1 dataset", `00_thesis.md` H3) is **not met**.
   - The kill rule is unaffected: it is decided on Wave B's verdict videos only.
6. **Any HoloAssist Δ also depends on Issue 7.** A step-name prior beats every action-only control used so far. Decide both before the Δ is computed.

**Option A (recommended): Measure audibility in every session, then run the non-verbal channel only where the instructor is audible**
- Compute the presence difference for every annotated session that has video (1,673), with the existing tool and rule, on audio levels only and **before any probe is fit**. Keep the sessions at ≥ +3.5 dB.
- Extract `e2v-plus-large` for the items in those sessions. Report `react-nonverbal`, `react-nonverbal|spoke=1`, `fusion` and the Δ on that subset, each row's `notes` naming the subset rule and its size.
- *Pros:*
  - it tests the thesis's actual channel (non-verbal voice) where it exists;
  - the filter is blind to labels, and its threshold was fixed before any outcome was seen;
  - it is cheap: about 1–2 h of session audio levels plus minutes of `emotion2vec+`.
- *Cons:*
  - **the subset is smaller and skewed:** about a third of sessions, mostly camera, console, printer and GoPro setups. Expect roughly 600 test items and a wider CI;
  - the rule was chosen after the check failed (though not after any outcome was seen), so it must be reported as such.

**Option B: Use the annotated speech timing (`spoke`) as HoloAssist's reaction channel in fusion, on all items**
- `fusion-spoke` = [`logit(judge)`, action-probe features, `spoke`], with its own Δ against `action_best`.
- *Pros:*
  - minutes of work; all 2,000 test items;
  - whether a supervisor spoke up is a behavioral reaction a robot could observe.
- *Cons:*
  - it comes from human-annotated utterance spans, not from an automatic detector;
  - it is crude: whether someone spoke, not how they sounded;
  - 56% of mistakes are by definition "corrected by instructor verbally", so a reviewer may call the gain near-definitional. A4 still holds: the labels were not derived from speech;
  - it is not the face/voice signal the thesis is about.
- **It can be combined with A** (separate, clearly named rows).

**Option C: Extract `e2v-plus-large` on all items anyway, with the +1.51 dB caveat** (the agent's original option C)
- *Pros:* the condition table as originally specified, on the full set.
- *Cons:* in most sessions the audio is the performer's tool noise, so the Δ would measure manipulation sounds (as on Oops!) and could not be read as a reaction effect.

**Option D: No HoloAssist Δ.** Keep the current rows, with `react-spoke` and `react-full` as descriptive only (the agent's original options A and B).
- *Pros:* no work; no risk of over-claiming.
- *Cons:* Wave A ends with no hidden-outcome Δ. Wave D can then start only on Wave B's results.

**Recommendation:** **A**, optionally with **B** as a second row. A answers the question the thesis asks, on the data where it can be asked. B is cheap context, but reviewers will discount it.

Your selection: Proceed with Option A.

*The agent's original options (October 15), for the record:*
- **A:** use `react-spoke`/`react-full` only and do not extract audio (now **D**);
- **B:** drop HoloAssist's audio channel (now inside **D**);
- **C:** extract anyway (now **C**).

---

### Issue 7: HoloAssist's step names leak the outcome; a step-name prior beats every action-only control

**Status**: ✅ **Decided: option A, by the maintainer, October 20, 2026.**
- **Follow-up choice, made in chat: the pass bar uses the ablation gain.** That is fusion − `action-fusion`, the same model without the reaction features.
- **Why it was needed:** the designer measured option A's intended combined model as written. Judge + step-prior + SigLIP scored 0.682, **−0.069 [−0.091, −0.047] below the step-prior alone (0.751)**. The high-dimensional inputs dilute the one strong feature, so "fusion − action_best" would fail whatever the reactions carry.
- **Contract:** `03_eval_harness.md` §4 (the ablation Δ row) and §6 (`step-prior`, `action-fusion`, the exact formula).
- **Implementation:** guide item A11. The entry moves to 🧪 Resolved when A11 lands.

*Found in the designer's verification, October 20, 2026, with a few lines over `DATA_ROOT/items/holoassist/items.jsonl` and `splits/holoassist.json`.*

**Facts (October 20, 2026):**
1. **The judges are given the step:** `context_text` = `"Task: <task>. Step: <verb> <noun>."` (`02_data_sources.md` → HoloAssist). The spec chose this so the judge would know what the person was trying to do, like a dish's name in verdict videos.
2. **One verb is an outcome, not a step:** `drop` (435 of the 5,000 items, 182 in test) is a mistake 95% of the time. The leakage test banned `mistake`, `correct`, `wrong`, `error`, `fix` and `instead`, but not outcome-type verbs.
3. **A step-name prior beats every action-only control.**
   - Method: the per-verb mistake rate learned on the train split (smoothed toward the train base rate), scored on test.
   - With all items: verb only **0.721**, verb + noun **0.727**, task only 0.613.
   - Without the `drop` items it still scores **0.666** (verb + noun 0.682). The class-balanced sample has different step mixes per class: the mistake share is `touch` 0.79, `pull` 0.74, `press` 0.70 and `open` 0.70, against `disassemble` 0.20 and `withdraw` 0.29.
   - For comparison: `action-probe` 0.569, `judge` 0.488, `judge-large` 0.544, `react-spoke` 0.676.
4. **`judge-large`'s HoloAssist signal is the leak.**
   - On the 1,818 test items without `drop` it scores **0.489** (chance). On the 182 `drop` items its mean P(good) is 0.30, against 0.92 elsewhere.
   - The 7B judge barely uses it (0.468 without `drop`).
5. **The reaction signal survives without `drop`:** `react-spoke` scores **0.733** on the non-`drop` test items (0.676 on all).
6. **Why it matters:** the H1 Δ is the reactions' gain over the best *action-only* control. Knowing which step was performed is action-only information, and it is stronger than any action-only control used so far. A Δ against `action-probe` (0.569) would overstate what reactions add.

**Option A (recommended): Add a `step-prior` action-only control, and give it to fusion too**
- `step-prior` = the train-split mistake rate per (verb, noun), backing off to the verb and then to the base rate, smoothed with 5 pseudo-counts. It is text only, with no frames. HoloAssist only.
- It enters `action_best`, and `logit(step-prior)` is added to `fusion`'s inputs, so fusion is not handicapped against it.
- The judges' input is unchanged.
- *Pros:*
  - an honest bar: the reactions must beat "knowing what step it was";
  - cheap: minutes, with no new extraction or judge runs;
  - a robot knows which step it was commanded to do, so the control is realistic.
- *Cons:*
  - it raises the bar the reactions must clear;
  - `drop` stays in the judges' input. It is disclosed, and it is harmless to the Δ because `step-prior` absorbs it.

**Option B: Remove outcome-type verbs, then rebuild**
- Drop the `drop` items, re-sample, rebuild the items and the split, and re-run features and both judges (~13 h).
- *Pros:* the clearest leak is removed from the items and the judge's input.
- *Cons:*
  - expensive;
  - the item definition changes after results were seen;
  - the remaining verbs still carry a 0.666 prior, so A's control is still needed for a fair Δ.

**Option C: Keep everything; report the 0.721 step prior as a caveat only**
- *Pros:* no work.
- *Cons:* any HoloAssist reaction gain is measured against a weaker action-only control than one a reviewer can build in a few lines.

**Recommendation:** **A.** It is cheap, keeps every number already computed, and makes the HoloAssist Δ (whatever Issue 6 selects) a fair test.

Your selection: Proceed with Option A.

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


---

## 4. Deferred work (do not start without its trigger)

| Id | Work | Trigger to start |
|---|---|---|
| DW1 | **Wave B: verdict-video H1 pilot** (~200 videos; face + non-verbal audio; ASR verdict parsing) | Issue 1 selected **and** Wave A closed **and** the designer has written the Wave B spec. Wave A is closed once guide items A9b and A10 land; Issues 6 and 7 do not block Wave B |
| DW2 | Face encoder (`react-face`) | Part of the Wave B spec |
| DW3 | **H2 transfer** to HoloAssist / AM-FED+ (+ BAD only if the QDR request is granted); scaling curves | Wave B closed |
| DW4 | Ego4D v0 false-positive set | Wave B+. The raw videos were deleted October 8 (Issue 3); re-download the chosen clip ids from `DELETED_2026-10-08.json` |
| DW5 | **Wave D, the robot check (offline H3)**: a data-efficiency curve on RoboReward, with vs. without pre-training on reaction-labeled human task video (`00_thesis.md` H3) | Wave A shows reaction signal on ≥ 1 dataset **and** the designer has written the Wave D spec. **Status (October 20): not met.** Oops!'s Δ CI includes 0, and HoloAssist's Δ was not run (Issues 6 and 7) |
| DW6 | **Stage A live, as a personal demo with the maintainer as the only reactor.** The microduck learns *which of its existing behaviors, and which styles of them, the maintainer likes* from their live reactions. The Mac runs the reaction reader (a webcam on the maintainer, or the duck's front-camera WebRTC stream) and sends choices over the duck's control channel. Measures: trials-to-preference vs. explicit thumbs-up/down, and reaction-reader accuracy against a quick rating after each trial. **Not in scope:** learning motor skills from reactions (that needs millions of trials, in simulation). Facts (October 9, 2026): 15-DoF, 25 cm biped, RK3566 with 1 GB RAM, front camera, 7 shipped moves; new policies are trained in sim with `microduck_rl` (mjlab/MuJoCo Warp, PPO), which **needs a CUDA GPU**, i.e. not this Mac | Hardware acquired **and** H2 done **and** the maintainer opts in (deferred October 9: *"maybe we don't worry about the someone reacting to the robot yet"*) |
