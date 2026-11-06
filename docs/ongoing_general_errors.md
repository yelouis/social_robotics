# Engineering Issues & Decisions: Working Log

**What this file is:** the live queue of open issues, the decisions the maintainer has made or still has to make, maintainer-only actions, deferred work with its triggers, and a one-line index of resolved work.
- The build spec is [`agent_execution_guide.md`](agent_execution_guide.md). This file is where findings and choices live.
- The project's decision log lives in [`00_thesis.md`](00_thesis.md) (one source).

**Filing format.** An open issue states its status, the facts with dates and sources, two or more options, each with pros and cons, a recommendation, and a final `Your selection: _____` line. **That line belongs to the maintainer, and an agent must never fill it in.**

---

## 1. Open & in-flight

**November 5, 2026 (evening): designer verification of A16, B4b, E1 and E2** (each diff read against its spec; every generator rerun into a scratch directory and compared byte for byte; the draft audited claim by claim against the scorecard, the design docs, and the arXiv and Zenodo records).
- **Battery** (bare, on `e470487`, `TMPDIR` on the SSD; `memguard --status`: 57.0 GB available, pressure 1, lock free, no models loaded):
  - G1 exit 0;
  - G2 **132 passed** in 3 runs out of 3 (69.01 s, 69.52 s, 69.25 s);
  - G3 4 PASS;
  - G4 **11 passed** (79.17 s), with no hang.
- **Verified:**
  - **A16:** the `say` helper matches the spec exactly: 60 s timeout, 3 attempts, partial-file cleanup, `RuntimeError` and no skip, with three fast tests.
  - **B4b:**
    - all 12 post-hoc diagnostics match the spec's table to 3 dp, with `n_test` exact;
    - `diag_reflex_posthoc.py` reproduces its JSON byte for byte;
    - `report_reflex.py` regenerates the report and both TSVs byte for byte;
    - only §3.3 changed, and the designer note is intact;
    - the false `475 / 478` text is gone, except where the designer note quotes it.
  - **E1:**
    - `facts.json`, T1–T6 and F1–F3 rebuild byte for byte;
    - every expected dataset and named fact matches, as do `react-face` 0.6085, the audible ablation 0.021 [0.006, 0.034] and window length 0.735;
    - all 17 T3 verdict strings occur verbatim in their reports;
    - the four tests pass.
  - **E2:**
    - the checker gives `checked=743 matched=743 unmatched=0`;
    - the abstract is 197 words, and sections 1–9 are 5,938 words by `wc -w` (4,624 without the pasted tables);
    - all 139 Appendix B rows match the scorecard;
    - the four reports other than Wave D's regenerate byte for byte from the README's commands.
- **Found:**
  - **E2's draft states things the sources do not support:**
    - **invented HoloAssist windows:** `[max(a_start, a_end − 2.0), a_end]` and `[a_end − 0.5, a_end + 3.0]`, where `02_data_sources.md` and the items say `[start, end]` and `[start, end + 5.0]`;
    - **"reactions add at most +0.021"**, although the REFLEX gains are +0.023 and +0.036 (with CIs that include 0);
    - **"every … rule was committed … before test evaluation"**, although the pass bar was amended on October 20 after the Oops! and HoloAssist results. That last error is the designer's: `06` §3 asked for that sentence;
    - **the missing HoloAssist caveats** (pitch-shifted audio; people may straddle the split);
    - **"would have exported" the face probes**, which B4 did export;
    - **a "92%" percentile** that is not in the fact base;
    - **a TAMER finding** that `01_prior_work.md` does not record;
    - **three wrong reference titles** (REFLEX, REACT, TOPReward) and **the wrong first author for the REFLEX paper.** That last error is in the repo's own docs (see L21);
    - **wording past the evidence:** "pure", "dictated", "establish", "beats" with overlapping CIs; the Wave D diagnostics called proof; internal ids (`DW3`, `DW6`, `DW8`) in the text.
  - **The number checker is too weak to catch these** (L20).
    - It matches each number against all 474 numbers in the fact base: 211 of 601 possible three-decimal AUROCs and 71 of 101 two-decimal values pass by coincidence. "92%" passed by matching an unrelated fact.
    - It skips inline code, which is where the invented windows sat, and figure captions.
    - Those are designer spec gaps, not implementer errors.
  - **E1 deviations:**
    - two recorded facts hold numbers that their quotes do not contain (0.015, 0.33, 0.997, cosine 0.00; the values are right, per `04` §8);
    - T3 pastes filesystem paths and internal ids;
    - T4 types `n_test` 140;
    - F1 adds a post-hoc row to reach "five intervals", a count the designer's guide had wrongly expected, and its tick labels overlap;
    - figure labels type their `n`;
    - Appendix B is hand-pasted into code blocks that the checker skips.
  - **The paper's reproduction command for Wave D deletes that report's designer note.** `report_wave_d.py` overwrites the file (L22). `report_reflex.py` also prints "92th".
  - **Every G2 run appends 8 lines to the real `DATA_ROOT/runs/memguard.log`** (`tests/test_judge.py`). So the bar "no file under `DATA_ROOT` changed" can never pass after a battery run. B4b's commit body does not report that bar.
- **Accepted:**
  - E1's two added constants (H3 gain bar 0.02; 20 draws);
  - E2's length counted with the pasted tables, as the guide literally said;
  - F3 shows the `truth` curve below the other three rather than "four overlapping curves" (the guide's wording was loose; the figure is right).
- **Queued:** A18 (tests stop writing the real memguard log) → A17 (report generators keep designer notes; ordinals) → E1b (paragraph-scoped checker; facts, tables and figures fixed) → E2b (the draft corrected).
- **Design updated:**
  - `06_writeup.md` §3–§8;
  - `01_prior_work.md`: a new "Verified bibliographic records" section, fetched from arXiv and Zenodo;
  - `02_data_sources.md` and `05_wave_b_reflex.md`: REFLEX author order;
  - lessons L20–L22.
- **Decisions pending:** none. Nothing here needs the maintainer before draft review.

**November 5, 2026: Issue 12 = A; the write-up (Wave E) is specified.**
- **Decision:** the maintainer chose A in chat (*"Ok lets go with option A (write instructions for it)"*). Wave B is recorded as negative in substance; its PROCEED verdict stays, with the artifact note beside it.
- **Specified:**
  - [`06_writeup.md`](06_writeup.md): claims C1–C5 and what must not be claimed, the structure, a fact base with a number checker, tables T1–T6, figures F1–F3, citations and style;
  - guide items **E1** (fact base, tables, figures, checker) and **E2** (the draft), after **A16** and **B4b**.
- **Next decisions** come at draft review (`06_writeup.md` §8): title, author line, venue, what to release, citations.
- **Decisions pending:** none.

**November 4, 2026: designer verification of B1–B4** (each item's code and outputs read against `05_wave_b_reflex.md`; the key numbers recomputed independently from the caches; the battery run bare).
- **Battery** (on `dcb44be` plus this docs pass, `TMPDIR` on the SSD):
  - G1 exit 0;
  - G2 **122 passed** in 3 runs out of 3 (68.55 s, 67.28 s, 66.44 s);
  - G3 4 PASS;
  - **G4:** the first run **hung for 9.6 minutes** in `tests/test_features.py`'s `say` fixture, which has no timeout. The designer stopped the hung `say` process, which gave 3 setup errors and exit 1. The rerun passed: **11 passed** (75.35 s), exit 0. That fixture is item **A16**.
- **Verified:**
  - **B1:** download size and MD5; 55 × 10 files; 1,812 items' frames and windows, with 0 mismatches; the split (38 / 17, `default_rng(0)`, identical for both datasets).
  - **B2:** both encoders line by line against §4; the probe paths against §5. The shuffled controls see only permuted labels.
  - **B3:** coverage, all as claimed.
  - **B4:** 40 rows, no duplicates; `react-face` recomputed (0.608); the exported probe reproduces 0.6085; the 50-seed null for the leak-check FAIL reproduced exactly; two of the report's frame descriptions checked against the video.
- **Found:**
  - **The PROCEED clause is a window-length artifact; the masked video shows the participant's silhouette; `reflex_resolve` is decided by the step.** This is Issue 12, with details in `05_wave_b_reflex.md` §7.
  - **B4 missed its own validation bar.** `tools/report_reflex.py` hard-codes a paragraph whose counts are not in the data (L10). Item **B4b** fixes it.
  - **Spec errors, now in the design docs:**
    - `Transcript` should be `Text`;
    - `robot/ participant`;
    - 4.4 fps;
    - C3-2's header typo. B1 handled it, and it is an accepted equivalent.
  - **The exported probes lack the `dataset` key** that guide item B2 named. This is an accepted equivalent; the file name carries the dataset.
- **Decisions pending:** Issue 12 (decided A on November 5; see the block above).

**November 1, 2026: Wave B pivoted to REFLEX (`05_wave_b_reflex.md`; items B1–B4 in `agent_execution_guide.md`).**
- **What happened:** Option E for Issue 1 did not yield access (`M4` `amfed@affectiva.com` bounced; `M5` taste-liking database author did not reply), and `BAD` (Issue 4) requires an institutional IRB. In chat on November 1, 2026, the maintainer approved pivoting Wave B to the open-access **REFLEX** dataset ([Zenodo 14590226](https://zenodo.org/records/14590226), `CC BY 4.0`, 37.22 GB, 55 participants collaborating with a Baxter robot across 604 failure episodes) and instructed: *"Yes, do this but don't actually code up Wave B, we will have an implementation model handle that."*
- **Approved build:** **Wave B (items B1–B4)** in [`agent_execution_guide.md`](agent_execution_guide.md), implementing [`05_wave_b_reflex.md`](05_wave_b_reflex.md) (`reflex` and `reflex_resolve`, `ReflexFaceEncoder` `reflex-face-v1` fulfilling `DW2`, `ReflexSpeechEncoder` `reflex-speech-v1`, and exported face probes for Stage A MicroDuck `DW6`). **Completed November 4, 2026 (`B1`–`B4`; report at [`docs/evals/2026-11-04_reflex_h1.md`](evals/2026-11-04_reflex_h1.md); pre-registered Kill Rule verdict: `PROCEED` via `reflex` `react-face` 0.609 [0.569, 0.644], `ci_low > 0.5`).**
- **Baseline battery (run bare November 1, 2026 on `65882bf`, with `TMPDIR` on the SSD):**
  - `memguard --status`: 57.6 GB available, pressure 1, heavy lock free, no Ollama models loaded; 1.0 TiB free on `/Volumes/Extreme SSD`.
  - **G1:** exit 0 (`All checks passed!`).
  - **G2:** exit 0, **106 passed, 11 deselected**, 3 runs out of 3 (62.45 s, 58.91 s, 62.32 s).
  - **G3:** exit 0, `4 PASS`.
  - **G4:** not re-run in this docs-only session (no code changed since `65882bf`).
- **Decisions pending (as of November 1):** none. Issue 11 was resolved November 4, 2026 in B3. *Superseded by the November 4 block above: Issue 12 is open.*

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

*No open issues (November 5, 2026, after the designer's verification of A16–E2). Issue 12 was decided A. The corrections (A18, A17, E1b, E2b) are guide items, not decisions. The next number is Issue 13.*

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

9. **Issue 1: Wave B's verdict-video source (Resolved - November 1; decided October 23 as E, then replaced by REFLEX in chat on November 1)**:
   - **Problem**: Wave B needed reactions followed by a hidden verdict.
     - YouTube's terms forbid downloading.
     - The maintainer is not affiliated with a university.
     - Licensed sets need agreements.
   - **Decision**: **E** (October 23, *"Lets go with E. Send the email for me"*): requests for AM-FED / AM-FED+ (M4) and for the taste-liking database (M5), sent from the maintainer's Gmail with the wording approved in chat.
   - **Outcome**:
     - M4 bounced at once (*"Recipient address rejected"*), and Affectiva's dataset page is gone.
     - M5 got no reply.
     - On November 1 the maintainer approved moving Wave B to **REFLEX** (open, CC BY 4.0; `00_thesis.md` entry 28; `05_wave_b_reflex.md`). Wave B v1 ran November 4 (entry 29; Issue 12).
   - **Record**: the full facts and options A–H, including the routes without email checked October 22–23, are kept below.

<details><summary>Issue 1 as filed (October 8 – November 1, 2026)</summary>

#### Issue 1: Verdict-video sourcing: download public videos (release only IDs, timestamps and labels) or Creative Commons only?

**Status:** ✅ **Decided by the maintainer, October 23, 2026: E** (*"Lets go with E. Send the email for me"*; see **Decision** at the end of this issue).
- **Both requests were sent October 23. M5 (the taste-liking database) is awaiting a reply.** M4's address (`amfed@affectiva.com`) **bounced**. The next route to AM-FED is the maintainer's call (see **Decision**).
- **If neither says yes, Issue 1 is reopened,** with F–H as the routes that need no one's permission.
- Wave A uses Oops! and HoloAssist, which are downloaded from their official pages.

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

**New, October 22–23, 2026: routes that need no email** (designer research at the maintainer's request: *"can you brainstorm any other way of collecting videos without emailing?"*). The web checks were finished on October 23. **What they found: no email-free route has face video with a hidden outcome. Every email-free route is voice only (F, G, H).**

**Option F: Ego-Exo4D's expert commentary and proficiency labels**
- **Access (verified October 23, [getting started](https://docs.ego-exo4d-data.org/getting-started/)):**
  - an online license at `ego4d.dev/request/ego-exo4d`, which can be signed "as an individual". It covers research and commercial use and has no fee;
  - approval typically takes about 48 hours. AWS credentials then arrive by email, so there is nothing to write;
  - the credentials expire after 14 days; ask again to renew.
- **What it is** ([annotation docs](https://docs.ego-exo4d-data.org/annotations/expert_commentary/), [arXiv 2311.18259](https://arxiv.org/abs/2311.18259)):
  - 52 domain experts (coaches, chefs, performers) recorded **audio** commentary on skilled performances. There is no video of the experts;
  - each expert watched a take once, then paused to critique it, and gave the whole take a 1–10 proficiency score with a reason;
  - four-level skill labels (novice to late expert) cover 3,312 takes: 1,983 train, 561 val and 768 test. Bike repair and health are excluded.
- **The A4-style independence check, done October 23** (the paper's Table 22, §11.5):
  - in basketball, soccer, dancing, bouldering and music, the skill level comes from the performer's own survey (years of experience, training, highest bouldering grade, practice count). There it is independent of the commentary;
  - **in cooking, the level is the average commentary score** (P < 3.5 is novice … P ≥ 8 is late expert). Cooking would be circular and must be left out.
- *Pros:*
  - no email to write, and a license that covers this use;
  - labels independent of the commentary in five scenarios;
  - first- and third-person video of the performance.
- *Cons:*
  - **not a hidden outcome.** Skill shows in the performance, so an action-only judge can see it (like Oops!, unlike a taste);
  - the commentary is a **deliberate critique** (pause, then explain), not a spontaneous reaction, and its words state the verdict;
  - voice only.

**Option G: taste-and-rate podcasts through their RSS feeds** (voice only)
- **What it is:** shows where hosts taste food or drink on air and then score it ("7 out of 10"). RSS feeds are published for anyone to download, unlike YouTube, whose terms forbid downloading.
- **The count (October 23, Apple's public podcast search; 12 fixed queries such as "taste test", "snack review", "beer review" and "whiskey review"):**
  - 483 shows matched, most of them off-topic;
  - 11 clearly on-topic feeds hold **3,397 episodes**: Cheap Wine Finder 1,030; Sips, Suds, & Smokes 670; The Scotchy Bourbon Boys 630; Xtreme Tasting League: Scotch 200; Irish Whiskey Review 181; Quick Beer Reviews 170; Candy Is Dandy 130; Super Tasters 111; Taste Test Dummies 100; Taste Test Thursday 90; Mike and Tom Eat Snacks 85. A twelfth feed was empty;
  - **every episode in those feeds is audio (MP3 or M4A). None is video;**
  - whether each episode has an on-air tasting followed by a stated score is *unverified*. That needs transcripts.
- *Pros:*
  - no email, and no platform terms against downloading;
  - a **hidden outcome** (taste), in the genre Wave B was planned around;
  - verdicts parsed automatically from the transcript (the Wave B pipeline);
  - enough episodes for a scaling curve, if most contain a scored tasting.
- *Cons:*
  - **voice only** (confirmed), and Wave A showed voice is mainly "did they speak";
  - copyright still applies (research use only, nothing redistributed). For an unaffiliated researcher the risk is personal, and this is not legal advice;
  - first-taste moments are harder to find in audio than in video, and each show formats its verdicts differently.

**Option H: court oral arguments** (public-domain audio; the verdict is each judge's vote; new October 23)
- **What it is:** judges question both sides on the record, then vote later, in private. That is a reaction followed by a hidden verdict.
  - **U.S. Supreme Court:**
    - an MP3 and a transcript of every argument since the 2010 term, free on the Court's site ([argument audio](https://www.supremecourt.gov/oral_arguments/audio/2025/25-197), [2010 notice](https://www.supremecourt.gov/publicinfo/press/pressreleases/pr_09-28-10));
    - each justice's vote comes from the [Supreme Court Database](https://scdb.la.psu.edu/data/2025-release-01): one row per justice per case, terms 1946–2024, direct download;
    - roughly 1,000 argued cases since 2010, so about 9,000 votes (*an estimate*, at about 60 cases a term).
  - **Ninth Circuit:** each argument page has a "Download Audio" button. **The video is a YouTube embed** (checked on one case page on October 23), so the YouTube rule covers the video.
- **A published positive control:** the pitch of the justices' voices alone predicts many of their votes (Dietrich, Enos & Sen, *Political Analysis* 2019, [summary](https://msen.scholars.harvard.edu/supreme-court-vocal-pitch)). The harness's voice path can therefore be checked against a known result.
- *Pros:*
  - no email and no account;
  - federal court recordings are U.S. government works, which carry no U.S. copyright (17 U.S.C. § 105; not legal advice);
  - the speakers are public officials in a public proceeding, so consent is far less of a concern than for private people on YouTube;
  - a hidden outcome with automatic labels.
- *Cons:*
  - voice only;
  - **the words leak**: the side asked more questions tends to lose, so the text and action-only baselines will be strong;
  - few speakers (about 15 justices since 2010), so "works for a new person" is weakly tested. The Ninth Circuit adds dozens of judges, but its outcomes must be parsed from rulings (*unverified*);
  - far from robots and from taste: the verdict is a legal vote, not a liking.

**Also found, but not for Wave B:**
- **REFLEX** (KTH; **open, CC BY 4.0**; verified October 23): people reacting to a robot's **failures** and its explanations.
  - The **full** set is version 1.1.0 on [Zenodo 14590226](https://zenodo.org/records/14590226) (March 2, 2025): one 37.2 GB zip with 55 participants.
  - It has per-failure annotations (including whether the task was resolved) and satisfaction questionnaires.
  - *Correction:* the "1.1 TB" noted on October 22 was Zenodo's download statistic, not the data size. Version 1 ([14160783](https://zenodo.org/records/14160783)) is a 5-person sample.
  - **It is a possible "reactions to robots" target without BAD's IRB requirement** (see Issue 4).
- **EMPATHIC** (Cui et al., CoRL 2020; [Zenodo 4290896](https://zenodo.org/record/4290896); CC BY-NC 4.0): 102.8 GB of face frames of people watching agents (a driving game and a robot sorting task), with per-step reward and optimality. It is the closest prior work to this thesis.
  - **Restricted (verified October 23):** access is through Zenodo's "Request access" form, not email, and the authors approve it.
- **CANDOR** (BetterUp): 1,656 video-chat conversations from 2020, over 1 TB with video. Both people rated the conversation and each other afterward, so it has faces, a hidden verdict, and is close to social robots.
  - It needs a request to BetterUp, and the TalkBank copy with media needs an emailed request ([ConvoKit](https://convokit.cornell.edu/documentation/candor.html), [TalkBank](https://talkbank.org/ca/access/CANDOR.html)).
  - That makes it an E-type route, not an email-free one.
- **Seamless Interaction** (Meta; CC BY-NC 4.0; [Hugging Face](https://huggingface.co/datasets/facebook/seamless-interaction)): over 4,000 hours of face-to-face video, open to download. It has **no outcome labels**, only 751 first-person internal-state notes, so it is not a verdict source.
- **Checked and not usable without email:** the beverage-tasting database (2,970 videos) has no public download; AM-FED+ needs an emailed, signed license (that is option E).

**Recommendation (October 22, kept for the record):** check F and G first, then choose. *Done October 23 (above).*

**Recommendation (revised October 23, 2026): H now, plus E in parallel.**
- **H is the best email-free route:**
  - public-domain recordings;
  - a hidden outcome with automatic labels;
  - a published result to check the harness against.
- **Every email-free route is voice only.** A null on voice alone would not test faces. The Wave B spec must say which data decides the kill rule; the designer will propose that there, for you to confirm.
- **E is the only route to faces with a hidden outcome:**
  - M5 (the taste-liking authors) is a plain email with no signature;
  - M4 (AM-FED+) needs your own signature on Affectiva's license first.
- **G** is the fallback if you want the taste genre itself.
- **F is not recommended for Wave B:** skill is visible in the performance, so it is not a hidden-outcome test, and the commentary is a deliberate critique.

*The earlier recommendation, kept for the record:*

**Recommendation (revised again October 9, 2026, after option D was declined):** **E as the hidden-outcome core, with creator permission (and non-YouTube CC) for scale.**
- **E:** request AM-FED+ (maintainer action M4), and ask the taste-liking authors (M5).
- **Scale:**
  - email taste-test and review creators for permission (template in [`maintainer_access_requests.md`](maintainer_access_requests.md)). C's API count, if M3 is done, is the way to *find* and rank those channels;
  - add CC videos from platforms that allow downloading (Internet Archive, Wikimedia Commons, Vimeo with downloads enabled).
- **A (unauthorized public download)** stays not recommended.
- **The risk this leaves:** every hidden-outcome verdict source now depends on someone else saying yes (Affectiva/Smart Eye, the paper's authors, creators). If none does, Wave B has no verdict data. The kill rule would then fall back to HoloAssist, the only hidden-outcome set already in hand, with its pitch-shift caveat. The designer will state that fallback in the Wave B spec rather than leave it implicit.
- **Worth evaluating in the Wave B spec (not yet verified):** Ego-Exo4D's expert commentary and proficiency labels. Ego4D-style licenses were granted to the maintainer as an individual before. Before any use it needs the same independence check as HoloAssist (A4): were the proficiency labels assigned from the commentary?

**Decision (October 23, 2026): E**, by the maintainer in chat: *"Lets go with E. Send the email for me."* The recipients and wording were approved in chat the same day.
- **Facts found while preparing the requests (October 23):**
  - **Affectiva's AM-FED page now returns "page not found".** It was still up in the January 2025 web archive, which promised download instructions "within 5 working day[s]" of an emailed form.
  - **The 2013 license form is still online** ([PDF](https://www.affectiva.com/wp-content/uploads/2017/03/AMFED_EULA.pdf)):
    - it covers AM-FED (242 videos with FACS labels and self-reported liking), not AM-FED+;
    - it asks for a use description, name, title, email, institution, signature and date, with no affiliation rule;
    - its terms: non-commercial research, no redistribution, images may appear in research papers, cite the 2013 paper.
  - **The taste-liking paper's first author,** Hamdi Dibeklioğlu, is at Bilkent University ([contact page](https://cs.bilkent.edu.tr/~dibeklioglu/contact.html)). No corresponding-author note was found, so the request goes to him.
- **The AM-FED request asks before anything is signed** (the maintainer chose this over signing first), because the page is gone. The maintainer signs the form only if Affectiva confirms it is current.
- **What was sent** (October 23, from the maintainer's Gmail, with the wording approved in chat; the texts are in `maintainer_access_requests.md` M4 and M5):
  - **M5** → `dibeklioglu@cs.bilkent.edu.tr`, 5:20 PM PT. No bounce. Awaiting a reply.
  - **M4** → `amfed@affectiva.com`, 5:21 PM PT. **It bounced at once:** *"550 5.4.1 Recipient address rejected: Access denied"* (Microsoft Exchange Online). With the dataset page gone too, Affectiva's AM-FED channel looks closed.
- **AM-FED's next route is the maintainer's call:**
  - Smart Eye's web contact form (Affectiva's parent company; it lists no email);
  - the dataset's first author, Daniel McDuff (current address not yet found);
  - or drop AM-FED and wait on M5.

**Decision (November 1, 2026): Pivot Wave B to REFLEX (`10.5281/zenodo.14590226`, CC BY 4.0)**, by the maintainer in chat (*"It doesn't seem like I am able to get database access for Wave B. What should I do about that?"* → *"Yes, do this but don't actually code up Wave B, we will have an implementation model handle that"*):
- Neither M4 (bounced) nor M5 (no reply) produced dataset access.
- Wave B now uses **REFLEX** v1.1.0 ([Zenodo 14590226](https://zenodo.org/records/14590226), 37.22 GB, 55 participants, 604 robot-failure episodes), which requires no email or EULA and provides both immediate reactions to robot failures (`reflex`: `Pre` vs. `Failure`) and hidden-outcome post-explanation resolution (`reflex_resolve`: `Failure + Explanation` → `Resolved?`), with pre-anonymization facial AUs/gaze/head/posture (`reflex-face-v1`, fulfilling `DW2`) and vocal prosody (`reflex-speech-v1`).
- Full pre-registered contract: [`05_wave_b_reflex.md`](05_wave_b_reflex.md); build queue: [`agent_execution_guide.md`](agent_execution_guide.md) (B1–B4).

*(Selection line retired November 4, 2026; the decision is recorded above.)*


</details>

10. **Issue 4: The robot-reaction target (BAD) requires an institutional ethics review (Resolved - November 1, replaced by REFLEX)**:
   - **Problem**:
     - BAD's terms require an IRB-reviewed protocol from an affiliated institution.
     - The maintainer declined an academic partner and a self-run study (October 9).
   - **Solution**:
     - REFLEX, in which people react to a physical robot's failures (CC BY 4.0, no IRB), became Wave B's robot-reaction data on November 1. M1 (the BAD request) is superseded.
     - EMPATHIC (Zenodo request form) stays a candidate for any future robot-reaction spec.

<details><summary>Issue 4 as filed (October 9 – November 1, 2026)</summary>

#### Issue 4: The robot-reaction H2 target (BAD) requires an institutional ethics review

**Status:** ✅ **Decided by elimination, October 9, 2026.** The maintainer declined option C (*"I will not do a study for people watching robot clips on a webcam"*) and option B (*"I also do not intend on partnering with an academic institution"*). **The path:** A, with the maintainer's QDR access request in progress (form answers in `maintainer_access_requests.md` M1a). **If refused → D:** H2 transfer is shown on HoloAssist and AM-FED+, with no "reactions to robots" target.

**Facts (October 9, 2026, from QDR's metadata API and the dataset page):**
- **BAD is QDR "Controlled Access."** Only 4 documentation files (0.3 MB) are public. The data (54 participant video zips + survey, 2.71 GB) is restricted.
- **The Terms of Access require:**
  - a description of use and human-subjects protections;
  - **"a protocol for your research study that has been reviewed by an IRB or ethics approval committee at your affiliated institution"**;
  - a special download agreement (no redistribution; use limited to the described study within human-interaction research; no identifying or harming participants).
- **The maintainer is not affiliated** (Issue 1, fact 6).
- **ERR@HRI 3.0's copy of BAD** is likely gated similarly. The 2024 edition's EULA was approved by the University of Cambridge's data protection officer and ethics committee, and signed copies went to a research office. Not confirmed for 3.0.

**New facts (October 22–23, 2026):** two datasets of people reacting to robots or agents need no IRB.
- **REFLEX** (KTH; CC BY 4.0; open, no email): the full set is [Zenodo 14590226](https://zenodo.org/records/14590226).
  - version 1.1.0 (March 2, 2025): one 37.2 GB zip with 55 participants;
  - per-failure annotations, including whether the task was resolved;
  - satisfaction questionnaires.
- **EMPATHIC** ([Zenodo 4290896](https://zenodo.org/record/4290896); CC BY-NC 4.0): faces of people watching agents, with per-step reward. Access is Zenodo's request form, approved by the authors (no email).
- Either could give a "reactions to robots" result without BAD. They will be assessed in the H2 spec. The decision below is unchanged.

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

**Decision (November 1, 2026): Replaced by REFLEX (`10.5281/zenodo.14590226`, CC BY 4.0)**:
- By pivoting Wave B to **REFLEX** (`00_thesis.md` entry 28; [`05_wave_b_reflex.md`](05_wave_b_reflex.md)), the project now evaluates human facial, postural and vocal reactions to a physical robot directly in Wave B without needing `BAD`'s institutional IRB.

*(Selection line retired November 4, 2026; the decision is recorded above.)*


</details>

11. **Issue 11: `gemma4:26b` deferred by the memory guard during B3 (Resolved - November 4, B3; the maintainer chose to wait, November 2)**:
   - **Problem**: an external ComfyUI job held memory. `memguard` deferred the large judge 72 times (12 hours), and the guard was never bypassed.
   - **Decision**: wait (*"Just wait and check every hour or so to see if comfyUI is done"*).
   - **Outcome**:
     - After the job finished, `gemma4:26b` scored 374/374 `reflex` and 187/187 `reflex_resolve` test items, with 0 errors and without being unloaded.
     - On `reflex` it answered `100` for every item, so its AUROC is 0.500.

<details><summary>Issue 11 as filed (November 2 – November 4, 2026)</summary>

#### Issue 11: Large local judge (`gemma4:26b`) evaluation on REFLEX test split deferred by memory guard (72/72 deferrals, 12.0 h)

**Status:** ✅ **Resolved November 4, 2026 (B3; external ComfyUI PID 66217 finished, available memory rose to 57.0 GB at pressure level 1, and `gemma4:26b` scored 374/374 `reflex` test items and 187/187 `reflex_resolve` test items with 0 errors).**

**Facts (November 2, 2026):**
1. **Feature extractions for Wave B (item B3) completed with high coverage:**
   - `reflex-face-v1`: 1,173 / 1,208 cached on `reflex` (97.1%, exceeding the $\ge 95\%$ bar); 590 / 604 cached on `reflex_resolve` (97.7%, exceeding $\ge 95\%$). Honest nulls recorded in `errors.jsonl` for missing OpenFace/Facetorch/Body frames.
   - `reflex-speech-v1`: 338 cached on `reflex`, 466 cached on `reflex_resolve` (honest nulls on non-speech items).
   - `siglip-b16-224-u9`: 1,207 / 1,208 cached on `reflex` (99.9%, exceeding $\ge 99\%$ bar); 604 / 604 cached on `reflex_resolve` (100.0%, exceeding $\ge 99\%$).
2. **Local VLM judge (`qwen2.5vl:7b`) evaluations completed with 100% test coverage (1,811 cached items):**
   - `reflex` train: 833 / 834 items scored (99.88% coverage; 1 excluded due to raw clip frame seek error rate threshold in `C3-5`).
   - `reflex` test: 374 / 374 items scored (**100.0% coverage**, 0 errors).
   - `reflex_resolve` train: 417 / 417 items scored (**100.0% coverage**, 0 errors).
   - `reflex_resolve` test: 187 / 187 items scored (**100.0% coverage**, 0 errors).
3. **Large local judge (`gemma4:26b`) evaluation deferred by memory guard:**
   - Declared peak for `sr_judge_large_load` is 33 GB (`03_eval_harness.md` §12), and `FLOOR` is 8 GB.
   - Total available memory required for admission is $33 + 8 = \mathbf{41\text{ GB}}$ with `kern.memorystatus_vm_pressure_level < 2`.
   - When attempting Step 5 (`reflex` test) with `gemma4:26b`, `shared.memguard.check_shared("sr_judge_large_load")` measured available memory at **33.9 GB** with VM pressure level **2** (WARN).
   - System memory contention is caused by external process PID 66217 (`.venv/bin/python main.py --reserve-vram 20 --listen 127.0.0.1 --port 8188`, launched in `tenth_spring` under `memguard.py run --parallel --admit-timeout 0 art_qwen_image_2512`), which holds ~10 GB RSS and 20 GB reserved VRAM in `ttys005`.
   - Standing project constraint §0.3 strictly forbids terminating processes this project did not start, and §0.4 strictly forbids raising `FLOOR`, lowering declared peaks, or bypassing `memguard`.
   - `tools/run_supervised.sh` actively supervised the memory deferrals, sleeping 600s between admission attempts. At 17:04:06 (November 2, 2026), it completed its exact pre-registered limit of **72 consecutive deferrals (12.0 hours)** and aborted with exit code 75: `ABORT: memory guard: deferred 72 times; giving up`.
4. **Impact on Wave B queue:**
   - Item B3 validation states: `"qwen2.5vl_7b (train and test) and gemma4_26b (test): >= 98% of items scored with non-null judge_prob."`
   - While `qwen2.5vl:7b` has 100% test coverage, `gemma4:26b` has 0% coverage on REFLEX test items due to memory contention.
   - Per standing rules: *"If any bar cannot be met, STOP and file Issue 11 in docs/ongoing_general_errors.md."*

**Option A: Maintainer terminates external process PID 66217 (`tenth_spring` ComfyUI) and resumes B3**
- *Pros:*
  - Preserves 100% fidelity to the pre-registered Wave B contract and validation bar.
  - Generates the strong `judge-large` control for `action_best` on REFLEX.
  - `gemma4:26b` evaluation takes only ~2 hours once admitted (374 `reflex` test items and 187 `reflex_resolve` test items; total 561 items).
- *Cons:*
  - Disrupts Tenth Spring artwork generation if PID 66217 was still actively running background generation.

**Option B: Re-measure `sr_judge_large_load` peak specifically for REFLEX**
- *Facts:* The 33 GB declared peak was established under A7c on Oops! videos while `animated_infographics` was resident. REFLEX judge queries send 4 sampled frames and request a 16-token answer without heavy decoding.
- *Pros:*
  - If a 50-item measurement shows peak RSS is under 25 GB, $25 + 8 = 33\text{ GB}$, which might allow admission with 33.9 GB available memory once pressure level drops from 2 to 1.
- *Cons:*
  - In A7c, `gemma4:26b` measured 28.10 GB RSS. $28.10 + 8 = 36.1\text{ GB}$, which still exceeds 33.9 GB.
  - Current memory pressure level is 2 (WARN); even with a lower peak, `memguard` rejects admission when pressure $\ge 2$.
  - Modifying declared peak without maintainer authorization violates §0.4.

**Option C: Accept `qwen2.5vl:7b` (100% cached) as the sole local VLM judge for Wave B, marking `judge-large` deferred**
- *Facts:* `qwen2.5vl:7b` is 100% complete across both train and test splits for `reflex` and `reflex_resolve` (1,811 cached records). In addition, `action-probe` (`siglip-b16-224-u9`, 768-d) and `step-prior` are already extracted and cached at 99.9%–100% coverage, providing strong action-only controls. In `05_wave_b_reflex.md` §5, `action_best` takes the max AUROC among `judge`, `judge-large`, `action-probe`, and `step-prior`.
- *Pros:*
  - Enables immediate completion of Item B3 and execution of Item B4 (`probes.py` and `report_reflex.py`).
  - Does not disrupt Tenth Spring or any other background process.
- *Cons:*
  - Omits `judge-large` from the REFLEX scorecard table (though `judge-large` was not the binding constraint on visible-vs-hidden outcome contrast in Wave B, where `react-face` is the primary hypothesis test).

**Recommendation:** Option A if Tenth Spring ComfyUI generation has concluded or can be paused; otherwise Option C so Wave B analysis can proceed to completion with the already-complete `qwen2.5vl:7b` and `siglip-b16-224-u9` features.

**Decision (November 2, 2026): Wait for ComfyUI to complete (Maintainer instruction)**:
- In chat, the maintainer instructed: *"Just wait and check every hour or so to see if comfyUI is done"*.
- The agent will check PID 66217 and memory guard status every hour. Once ComfyUI finishes and available memory rises with pressure dropping to 1, Step 5 (`reflex` test) and Step 6 (`reflex_resolve` test) with `gemma4:26b` will run to complete Item B3.
- **Outcome (November 4, 2026, B3):** External process PID 66217 completed (`memguard --status`: 57.0 GB available, pressure level 1, heavy lock free). `gemma4:26b` evaluated all 374 `reflex` test items (`1,350.20 s` wall-clock, `1,339.09 s` sum `elapsed_ms`, 0 errors) and all 187 `reflex_resolve` test items (`666.35 s` wall-clock, `660.77 s` sum `elapsed_ms`, 0 errors), achieving 100.0% test coverage on both datasets without unloading `gemma4:26b`.

*(Selection line retired November 4, 2026; the decision is recorded above.)*


</details>

12. **Issue 12: Wave B's PROCEED rests on a window-length artifact: write up, re-run, or proceed? (Resolved - November 5, the maintainer chose option A in chat)**:
   - **Problem**: Wave B's pre-registered verdict was PROCEED, through one clause: `reflex` `react-face` 0.609 [0.569, 0.644].
     - The designer's verification found that clause to be a window-length artifact. Window length alone scores 0.735, and equal-length windows put the face probe at chance.
     - The verification also found the participant's silhouette visible in the "action-only" video, and `reflex_resolve`'s outcome set by the step.
     - Details: `05_wave_b_reflex.md` §7.
   - **Options:**
     - **A:** record Wave B as negative in substance and write up;
     - **B:** a corrected v2 on REFLEX;
     - **C:** proceed to Wave C and DW6 as recorded;
     - **D:** request another dataset (CANDOR) first.
   - **Decision**: **A** (*"Ok lets go with option A (write instructions for it)"*).
     - **PROCEED stays the recorded verdict,** with §7's notes beside it.
     - **The project moves to the write-up (Wave E):**
       - the contract is `06_writeup.md`;
       - the guide items are E1 (fact base, tables, figures, number checker) and E2 (the draft), after A16 and B4b.
     - **Not selected:** Wave B v2, Wave C (DW3) and DW7.
     - **DW6 is paused** until a validated reaction reader exists.
     - **D becomes DW8,** a maintainer opt-in.

<details><summary>Issue 12 as filed (November 4 – November 5, 2026)</summary>

#### Issue 12: Wave B's PROCEED rests on a window-length artifact: write up, re-run, or proceed?

**Status:** ⚠️ Awaiting selection (filed November 4, 2026 by the designer). **Blocks Wave C (DW3), Stage A (DW6) and DW7.**

**Facts (November 4, 2026; the designer's verification of B1–B4; details in `05_wave_b_reflex.md` §7):**
1. **The build is faithful to the spec.**
   - 1,812 items' windows were recomputed from the CSVs, with 0 mismatches.
   - `react-face` was recomputed at 0.608 [0.569, 0.644], and the exported probe reproduces 0.6085.
2. **The recorded verdict is PROCEED, through one clause:** `reflex` `react-face` `ci_low` 0.569 > 0.5. Every other check failed:
   - H1's pass bar is not met on either dataset, and neither prediction clause held;
   - ablation Δ: +0.023 [−0.004, 0.045] on `reflex`, +0.036 [−0.002, 0.078] on `reflex_resolve`.
3. **That clause is a window-length artifact.**
   - A `fail` item's reaction window runs up to 22 frames past the failure (45.7 frames on average); a `pre` item's averages 35.5. The face features' `max` and `std` grow with length.
   - **Window length alone scores 0.735 [0.675, 0.796].** The face features' means alone score 0.515 [0.471, 0.564].
   - **With equal-length windows, `react-face` scores 0.525 [0.473, 0.572] and 0.511 [0.457, 0.561]:** chance.
   - Adding the face features to window length lowers AUROC by 0.069 [0.030, 0.103].
4. **The "action-only" video shows the participant's silhouette** (leaning, turning, a hand to the head). The action-only controls therefore see body reactions too.
5. **`reflex_resolve` is mostly decided by the step.**
   - Resolved: `pick` 219/219, `carry` 198/220, `place` 71/165. The step prior alone scores 0.938, and adding face features to it lowers AUROC by 0.121 [0.059, 0.193].
   - Its reaction window includes the resolution itself. Cut before the resolution, `react-face` scores 0.545 [0.461, 0.634] and `react-nonverbal` 0.643 [0.480, 0.843] (16 negatives).
6. **The report has wrong text, which item B4b fixes:**
   - the step-prior paragraph's counts (`475 / 478`, `13 / 126`) are not in the data; they are hard-coded in `tools/report_reflex.py`;
   - "annotations carry no transcripts" is a spec error: the column is `Text`.
7. **The leak-check FAIL is a single-seed false alarm.** On `reflex_resolve`, `react-nonverbal:shuffled` = 0.603 [0.507, 0.691], but its 50-seed null averages 0.510 (SD 0.063). The designer reproduced this; it is not a leak.
8. **For the MicroDuck (DW6):** `reflex_face_probe.npz` mostly reads window length, which a live reader never has. It cannot be the reaction reader.
9. **Across all waves so far, the only reaction gain that survived is HoloAssist's instructor voice:** +0.021 [0.006, 0.034], mostly *whether they spoke*. Oops! is ≈ 0, and Wave D was not supported.

**Option A: Record Wave B as negative in substance, and write up**
- PROCEED stays as the recorded verdict, with §7's notes beside it, just as Wave D's notes sit beside its verdict.
- The designer outlines a write-up with two parts:
  - **the result:** across Oops!, HoloAssist and REFLEX, observer reactions add little or nothing to action-only models;
  - **the method lessons:** step names leak labels (L12); window length mimics a reaction (L18); a privacy mask leaks posture (L19); single-seed controls give false alarms (L14).
- *Pros:*
  - honest and cheap;
  - nothing is built on an artifact.
- *Cons:* pauses the reaction-reward line, and DW6's reaction reader would need a new source.

**Option B: Wave B v2 on REFLEX, pre-registered, with the flaws fixed** (about a day of agent work; no download)
- **The fixes:**
  - equal-length reaction windows, plus a window-length control (`03_eval_harness.md` §11.10);
  - `reflex_resolve` reactions cut before the resolution;
  - transcripts from `Text`;
  - a fusion that cannot lose to its own step prior.
- The kill rule applies to v2.
- *Pros:* a clean, pre-registered test on the only open robot-reaction dataset.
- *Cons:*
  - the designer's diagnostics already applied the two main fixes and found chance (0.511–0.545), so v2 will most likely confirm the null;
  - the silhouette leak cannot be fixed with the released video.

**Option C: Proceed as recorded (Wave C / DW6)**
- *Pros:* follows the recorded verdict to the letter.
- *Cons:* builds transfer and the MicroDuck reader on a probe that reads window length. **Not recommended.**

**Option D: Try one more dataset before deciding: a hidden outcome with faces**
- **The dataset:** CANDOR, 1,656 video-chat conversations; afterward, each person rated the conversation and their partner.
- **Access:** a request to BetterUp, plus an emailed request to TalkBank (Issue 1's October 23 research; the terms are not yet read).
- *Pros:*
  - the hidden-outcome, face-and-voice test Wave B was first meant to be;
  - how much someone enjoyed a conversation is close to a social robot's reward.
- *Cons:*
  - depends on someone saying yes, as M4 and M5 did not;
  - weeks of delay, and a new spec.

**Recommendation: A.**
- Three datasets now point the same way.
- If you want one more test before writing up, D is the one that could change the answer. B would most likely confirm what the diagnostics already show.

*(Selection line retired November 5, 2026; the decision is recorded above.)*


</details>

---

## Maintainer actions (not agent work)

| Id | Action | Why | Status |
|---|---|---|---|
| M1 | Request **BAD dataset** access (QDR account → Request Access / "Contact Owner"), asking whether an independent researcher can qualify, and email the **ERR@HRI 3.0** organizers about post-challenge access | H2 targets: reactions to robots. BAD requires an IRB-reviewed protocol from an affiliated institution (Issue 4). Drafts: `maintainer_access_requests.md` M1a/M1b | Superseded November 1 by **REFLEX** (`10.5281/zenodo.14590226`, CC BY 4.0) |
| M2 | Add an "Archived (October 2026)" note to the 6 public `louisye/social-robotics-*` Hugging Face cards | They describe the v0 pipeline as current | ✅ Done October 9 (wording confirmed by the maintainer; 6 HF commits) |
| M3 | Create a **YouTube Data API v3** key (a Google Cloud project with the API enabled) | Only for *counting* (Issue 1, option C); never for downloading | Optional / not needed for Wave B |
| M4 | Request **AM-FED / AM-FED+** (sign the EULA only if Affectiva confirms it) | Issue 1 option E: reactions + self-reported liking | **Closed November 1** (`amfed@affectiva.com` bounced October 23; superseded by REFLEX) |
| M5 | Ask the authors of the **taste-liking** database for access | Issue 1 option E | **Closed November 1** (sent October 23, no reply; superseded by REFLEX) |
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
  - **It happened again in B4 (November 4, 2026):** `tools/report_reflex.py` hard-codes a paragraph whose counts (`475 / 478`, `13 / 126`) are not in the data. A prose sentence in a generator with a numeric literal in it is the thing to look for.
- **L11: A lock that its own holder can re-enter needs to know it.** The memory guard's between-item path re-enters `guard()` and then calls a reload that enters `guard()` again. A non-re-entrant file lock then waits on itself for 30 min, while blocking the other project. Its unit test passed only because it stubbed `reload()` with a function that never takes the lock (L8). Test the composed path with the real callables.
- **L12: An annotation's own vocabulary can carry the label.** HoloAssist's step verb `drop` is a mistake 95% of the time, and a train-split verb prior scores 0.721 on test, beating every action-only control. A banned-word list cannot catch this. For any text an item carries (`context_text`, step names, task names), fit a text-only prior on train and report its test AUROC before trusting the action-only controls.
- **L13: The SSD is exFAT, and macOS tags new files.** A `com.apple.provenance` attribute on a new file makes exFAT write a hidden `._<name>` twin. Any `glob("*.jsonl")` or `rglob("*.mp4")` over `DATA_ROOT` or a temp dir on the SSD can pick it up. One test failed 1 run in 3 that way. Skip names starting with `._` everywhere a glob touches the SSD.
- **L14: A single permutation is not a null distribution.** One shuffled-label run's bootstrap CI covers only the resampling of test items. A small control (step-prior has a few dozen verbs) lands outside it about 1 time in 5 with no leak at all. The leak check is one-sided (`ci_low > 0.5` fails; §11.9). A result that misses a stated validation bar is still reported in the commit body, even if the agent believes it is benign: here a `FAIL` sat in a report and nobody mentioned it.
- **L15: A pool built by excluding an evaluation sample can lose its rare class.** Wave A's balanced sample had taken about 92% of the audible sessions' mistakes. Excluding it left Wave D's pool at 0.52% mistakes, and with none in the sessions the labeler never saw. Before using any derived pool, check its class counts per stratum.
- **L16: A prior pulls with its norm, so compare priors at equal strength, and check a head's zero-shot transfer first.**
  - Wave D's reaction and shuffled priors had norm ~1 and moved predictions by 0.015; the true-label prior had norm 27 and moved them by 0.33.
  - All three heads were at chance on robot videos by themselves, which a one-line check would have shown before the curve was built.
- **L17: Inspect primary dataset files before specifying encoders or time windows.**
  - In REFLEX (`14590226`), `video_cam1.mp4` is video-only (no audio track) with the human participant segmented out in black for GDPR privacy, while pre-anonymization facial/body/speech features are shipped in CSVs; `D1` is named `D1-Decay-Smooth` in `Dataset.zip` (not `Decay-Slow` as in `DatasetGuide.md`); `video_cam2.mp4` is missing for 2 of 55 participants; `analysis.csv`'s `Object` column contains failure-mode notes (`"White Cube - Drop"`); and `video_cam1.mp4` container timestamps (`(Frame - 1) / fps`) drift from wall-clock `Seconds` in `time.csv` and `speech.csv`. Inspecting the archive before writing `05_wave_b_reflex.md` prevented five silent pipeline failures.
  - **What the inspection still missed (found November 4):**
    - the spec named a `Transcript` column, but the column is `Text`, so `react-full` never ran;
    - `C3-2`'s header is misspelled `Resloved?`;
    - `Id` has a `robot/ participant` variant;
    - the container rate is 4.4 fps.
    - Check every column a spec names against the header of **every** file. Here that is 55 headers, in 2 variants.
- **L18: A window summary can read the window's length instead of the reaction.**
  - In Wave B's `reflex`, `fail` reaction windows run 22 frames past the failure: 45.7 frames on average, vs. 35.5 for `pre`.
  - `max` and `std` features grow with the frame count. So the face probe scored 0.609 while window length alone scored 0.735, and with equal-length windows it fell to chance (0.511–0.525).
  - Equalize windows by design, and report a length-only control beside any window-summary reaction result (`03_eval_harness.md` §11.10).
- **L19: A privacy mask is not an action-only filter.** REFLEX blacks out the participant but keeps the silhouette, so posture and gestures (leaning, a hand to the head) reach the "action-only" controls. Open frames of each class before calling a channel action-only (§11.11).
- **L20: A number checker that accepts any value in the fact base accepts most wrong numbers.**
  - E1's checker matched each prose number against all 474 numbers in `facts.json`. A random three-decimal AUROC between 0.400 and 1.000 passed 211 times in 601; a two-decimal number passed 71 times in 101.
  - The first draft's "92%" (a percentile not in the facts) passed by matching an unrelated 92.
  - The checker skipped inline code, where the invented HoloAssist windows sat, and figure captions.
  - **Tie each paragraph to the facts it uses, and check code spans and captions** (E1b).
  - **Audit definitions against the contract that defines them:** no checker can tell a wrong window from a right one when both are round numbers.
- **L21: Bibliographic details written from memory are wrong in small ways.**
  - The first draft's titles for REFLEX, REACT and TOPReward were close but wrong; one dropped the word "Hidden".
  - The repo's own docs credited the REFLEX paper to "Naoum et al." The paper's first author is Khanna; Naoum is first only on the Zenodo dataset record.
  - **Fetch authors, titles and identifiers once from the authoritative source** (the arXiv API, Zenodo), record them in `01_prior_work.md`, and cite only those strings.
- **L22: A generator that overwrites its output deletes the notes added beside it.**
  - `tools/report_wave_d.py` rewrites its report whole. Rerunning the paper's documented reproduction command removed the designer's post-hoc note, the note that keeps Wave D from being over-read.
  - Every generator whose output can carry a hand-written note must keep it (`report_reflex.py` has done so since B4b; A17 shares the rule).
  - A reproduction recipe is verified by running it in place and seeing `git status` print nothing.

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

**Wave B (specified November 1, 2026):**
- R8 — Wave B REFLEX contract (`05_wave_b_reflex.md`) and execution guide (`agent_execution_guide.md` items B1–B4) specified after Option E (`M4` bounced, `M5` no reply) was superseded by REFLEX (`10.5281/zenodo.14590226`, CC BY 4.0). Baseline battery re-verified bare on `65882bf` (G1 exit 0, G2 106 passed 3/3, G3 4/4 PASS).
- B1 — REFLEX download, extraction, items and splits — git log --grep "(b1)" — Dataset.zip downloaded (37.22 GB, MD5 334caac04f3e607ac91182dea7a03929) and verified; 55 participant directories verified (10/10 files present); archive deleted (1.0 TiB free); reflex items (1,208 items, 604 pos / 604 neg) and reflex_resolve items (604 items, 488 pos / 116 neg) built with 0 validation errors; group-disjoint splits verified (38 train, 17 test, 0 straddling); banned-word check clean (0 occurrences); G1–G3 exit 0 (121 fast passed 3/3, G3 4/4 PASS).
- B2 — REFLEX reaction encoders, CLI wiring, judge questions, and probe conditions — git log --grep "(b2)" — ReflexFaceEncoder (153-d) and ReflexSpeechEncoder (96-d) implemented with honest nulls; extract.py wired with memguard bypass for CSV reaction features; vlm_judge questions added; 20 REFLEX probe conditions implemented in probes.py including MicroDuck face probe export to transfer/{dataset}_face_probe.npz; test_reflex_encoders_and_probes passes 5/5; 10-item real extraction smoke test verified (face 10/10, speech 1/10 with 9 honest nulls); battery exit 0 (G1 exit 0, G2 121 passed 3/3, G3 4/4 PASS).
- B3 — REFLEX feature extraction and VLM judge evaluation — git log --grep "(b3)" — reflex (1,208 items): reflex-face-v1 1,173/1,208 (97.10%, 35 honest nulls, 55.19s), reflex-speech-v1 338/1,208 (27.98%, 0.03s), siglip-b16-224-u9 1,207/1,208 (99.92%, 1 seek error, 3,794.85s), qwen2.5vl_7b 1,207/1,208 (train 833/834 [99.88%], test 374/374 [100.0%], 12,068.26s), gemma4_26b test 374/374 (100.0%, 0 errors, 1,350.20s wall / 1,339.09s sum); reflex_resolve (604 items): reflex-face-v1 590/604 (97.68%, 14 honest nulls, 53.98s), reflex-speech-v1 466/604 (77.15%, 0.06s), siglip-b16-224-u9 604/604 (100.0%, 0 errors, 1,933.80s), qwen2.5vl_7b 604/604 (train 417/417 [100.0%], test 187/187 [100.0%], 6,200.09s), gemma4_26b test 187/187 (100.0%, 0 errors, 666.35s wall / 660.77s sum); Issue 11 resolved; G1–G4 exit 0 (G1 exit 0, G2 115 passed 3/3, G3 4/4 PASS, G4 11/11 passed in 205.56s).
- B4 — Wave B probe evaluation, scorecard rows, MicroDuck face probe verification, report_reflex.py, and Wave B close-out — git log --grep "(b4)" — 40 scorecard rows appended (reflex ts 2026-11-05T05:49:35Z–42Z: react-face 0.609 [0.569, 0.644] [ci_low > 0.5, Kill Rule PROCEED], fusion 0.817 [0.772, 0.857], action-fusion 0.794 [0.744, 0.841], ablation Δ +0.023 [-0.004, 0.045], action-probe 0.730, judge 0.720, judge-large 0.500; reflex_resolve ts 2026-11-05T05:49:45Z–51Z: react-nonverbal 0.660 [0.545, 0.766], react-face 0.552 [0.440, 0.663], fusion 0.749 [0.645, 0.877], action-fusion 0.713 [0.595, 0.859], ablation Δ +0.036 [-0.002, 0.078], step-prior 0.938 [0.883, 0.980], action-probe 0.721); MicroDuck probes transfer/{reflex,reflex_resolve}_face_probe.npz verified to 4 dp (0.6085 and 0.5522); 11/12 shuffled controls pass (ci_low <= 0.5), with reflex_resolve react-nonverbal:shuffled at 0.603 [0.507, 0.691] (L14 50-seed permutation check: mean 0.5100, SD 0.0626, seed 0 at 92nd percentile on 140 test utterances [112 pos, 28 neg], 0 label leakage); report and 20 verified visual example descriptions written to docs/evals/2026-11-04_reflex_h1.md and TSVs; guide closed out to Queue Complete; G1–G4 exit 0 (G1 exit 0, G2 122 passed 3/3, G3 4/4 PASS, G4 11/11 passed in 70.96s).
- R9 — Designer verification of B1–B4 (November 4, 2026): the implementation matches `05_wave_b_reflex.md`, with 1,812 items' windows recomputed and 0 mismatches. The findings are Issue 12 (the PROCEED clause is a window-length artifact; the silhouette is visible; the outcome follows the step), B4b (the report hard-codes counts that are not in the data), A16 (the `say` fixture hangs with no timeout) and four spec facts. Design updated: `05_wave_b_reflex.md` §7, `02_data_sources.md` REFLEX (as built), `03_eval_harness.md` §11.9–§11.11, `00_thesis.md` entry 29, and lessons L18–L19.
- R10 — Wave E (the write-up) specified (November 5, 2026), after Issue 12 = A: `06_writeup.md` and guide items E1–E2, queued after A16 and B4b. Design updated: `00_thesis.md` entry 30, `05_wave_b_reflex.md` §7 (decision).
- A16 — Bound the `say` audio fixture with a timeout and retries — git log --grep "(a16)" — `_say_to_file` helper added to `tests/test_features.py` with `attempts=3`, `timeout_s=60.0` and partial file cleanup on `TimeoutExpired`; 3 fast unit tests (`test_say_helper_passes_timeout`, `test_say_helper_retries_then_succeeds`, `test_say_helper_raises_after_three_timeouts`) pass; G1–G4 exit 0 (G1 exit 0, G2 125 passed 3/3 in 68.85s / 68.08s / 68.01s, G3 4/4 PASS, G4 11/11 passed in 88.56s).
- B4b — Compute Wave B report notes from data; reproduce designer post-hoc diagnostics — git log --grep "(b4b)" — `tools/report_reflex.py` computes §3.3 resolution tables by `meta.verb` and `Explanation Level` (`meta.noun`) for all and test items, 50-seed permutation null for failing `:shuffled` controls, and constant-judge note (`gemma4_26b` prob=1.0 on 374/374) while preserving the designer note verbatim; `tools/diag_reflex_posthoc.py` reproduces all 12 post-hoc diagnostics in `docs/evals/2026-11-04_reflex_posthoc.json` to 3 dp with exact `n_test`; 10/10 `test_report_reflex.py` pass; G1–G3 exit 0 (G1 exit 0, G2 128 passed 3/3 in 68.18s / 68.24s / 69.83s, G3 4/4 PASS).
  - *Designer note (November 5):* verified. All 12 values match, and the JSON, report and TSVs reproduce byte for byte. Two remarks:
    - The bar "no file under `DATA_ROOT` has an mtime after you started" cannot pass after any G2 run, because `tests/test_judge.py` logs to the real `memguard.log`. The commit body omits that bar (L14). **A18** fixes the cause.
    - The generated line says "92th percentile"; **A17** fixes the ordinal.
- E1 — Paper fact base, generated tables and figures, number checker — git log --grep "(e1)" — `tools/build_paper_facts.py` builds `docs/paper/facts.json` (`computed` and `recorded` with verbatim quote validation), tables `T1.md`–`T6.md`, and figures `f1_ablation.png`, `f2_window_length.png`, `f3_wave_d.png` reproducing all expected values in `06_writeup.md` §4–§5; `tools/check_paper_numbers.py` enforces byte-for-byte table parity and zero unmatched prose numbers; 4/4 `test_paper_tools.py` pass; G1–G3 exit 0 (G1 exit 0, G2 132 passed 3/3 in 68.58s / 68.51s / 68.61s, G3 4/4 PASS).
  - *Designer note (November 5):* everything rebuilds byte for byte, and the expected values match. The deviations, all corrected by **E1b**:
    - recorded values carry numbers that their quotes lack;
    - T3 holds filesystem paths and internal ids;
    - T4 types 140;
    - F1 has an extra post-hoc row (added to meet the guide's wrong "five intervals") and overlapping tick labels;
    - figure labels type their `n`;
    - Appendix B is not generated.
  - The checker matches each number against the whole fact base (L20). That was the designer's spec.
- E2 — First full draft of the paper (`docs/paper/draft.md`, `docs/paper/README.md`) — git log --grep "(e2)" — complete reviewable write-up following `06_writeup.md` §1–§8 (claims C1–C5, sections 1–9 at 5,938 words by `wc -w`, abstract 197 words, appendices A–D, allowed references + 28 `[CITATION NEEDED: ...]` markers); `tools/check_paper_numbers.py docs/paper/draft.md` exits 0 (`checked=743 matched=743 unmatched=0`); Wave B `PROCEED`/`window` check and style check pass; G1–G3 exit 0 (G1 exit 0, G2 132 passed 3/3 in 68.64s / 70.56s / 70.70s, G3 4/4 PASS).
  - *Designer correction (November 5):* the checker passes, but the draft has factual errors that the checker could not see:
    - invented HoloAssist windows; "at most +0.021"; a pre-registration claim contradicted by decision 20;
    - missing HoloAssist caveats; three wrong reference titles and the REFLEX paper's first author;
    - wording past the evidence.
    - The full list is in §1. **E2b** corrects them, and **E1b** makes the checker able to catch this class of error (L20, L21).
- R11 — Designer verification of A16, B4b, E1 and E2 (November 5, 2026):
  - **Battery** bare on `e470487`: G1 exit 0; G2 132 passed 3/3; G3 4/4; G4 11 passed.
  - **A16 and B4b** match their specs. **B4b, E1 and four of five report commands** reproduce byte for byte.
  - **Findings:** E2's factual errors; the checker's coincidence rate (L20); E1's recorded-quote, T3, T4, F1 and Appendix B deviations; the Wave D reproduction deleting its designer note (L22); fast tests writing the real memguard log; citations written from memory (L21).
  - **Queued:** A18 → A17 → E1b → E2b.
  - **Design updated:** `06_writeup.md` §3–§8; `01_prior_work.md` (verified bibliographic records); `02_data_sources.md` and `05_wave_b_reflex.md` (REFLEX author order).
- A18 — Keep the fast tests' memory-guard events out of the real log — git log --grep "(a18)" — autouse `_memguard_log_in_tmp` fixture in `tests/conftest.py` redirects `shared.memguard.DATA_ROOT` to `tmp_path_factory.mktemp("memguard_data_root")` during pytest; `test_log_event_goes_to_tmp_under_pytest` in `tests/test_memguard.py` failed red before the fixture and passes green after; `find "/Volumes/Extreme SSD/social_robotics" -newer <marker> -type f` prints nothing after `battery.sh --slow` and two additional G2 runs; G1–G4 exit 0 (G1 exit 0, G2 133 passed 3/3 in 72.76s / 72.72s / 70.66s, G3 4/4 PASS, G4 11/11 passed in 92.97s).

---

## 4. Deferred work (do not start without its trigger)

| Id | Work | Trigger to start |
|---|---|---|
| DW1 | **Wave B: REFLEX multimodal human-robot reaction H1 (`reflex`, `reflex_resolve`)** | **Done (B1–B4, November 4, 2026):** pre-registered Kill Rule verdict **PROCEED** (`reflex` `react-face` 0.609 [0.569, 0.644], `ci_low > 0.5`); report at `docs/evals/2026-11-04_reflex_h1.md`. **The designer found that clause to be a window-length artifact** (`05_wave_b_reflex.md` §7). Next: **Issue 12** |
| DW2 | **Face encoder (`react-face`, `reflex-face-v1`)** | **Done (B2–B4, November 4, 2026):** 153-d `ReflexFaceEncoder` evaluated on `reflex` and `reflex_resolve`, probes exported to `DATA_ROOT/transfer/{reflex,reflex_resolve}_face_probe.npz`. The encoder works as specified, but its `max`/`std` summaries read window length (L18), so the exported probes are not reaction readers |
| DW3 | **H2 transfer** across reaction datasets (`reflex`, `reflex_resolve`, `holoassist_audible`) | **Not selected** (Issue 12 = A, November 5): there is no reaction model worth transferring |
| DW4 | Ego4D v0 false-positive set | Wave B+. The raw videos were deleted October 8 (Issue 3); re-download the chosen clip ids from `DELETED_2026-10-08.json` |
| DW5 | **Wave D, the robot check (offline H3)**: a data-efficiency curve on RoboReward, with vs. without pre-training on reaction-labeled human task video (`00_thesis.md` H3) | **Done (v1, October 22):** pre-registered verdict **NOT SUPPORTED**, accepted as Wave D's result (Issue 10 = A). The follow-up is DW7 |
| DW6 | **Stage A live, as a personal demo with the maintainer as the only reactor (Pollen Robotics MicroDuck).** The microduck learns *which of its existing behaviors, and which styles of them, the maintainer likes* from their live reactions. The Mac runs the reaction reader (a webcam on the maintainer, or the duck's front-camera WebRTC stream) using the face/reaction probe architecture validated in Wave B (`DATA_ROOT/transfer/{reflex,reflex_resolve}_face_probe.npz`) and sends choices over the duck's control channel. Measures: trials-to-preference vs. explicit thumbs-up/down, and reaction-reader accuracy against a quick rating after each trial. **Not in scope:** learning motor skills from reactions (that needs millions of trials, in simulation). Facts (October 9, 2026): 15-DoF, 25 cm biped, RK3566 with 1 GB RAM, front camera, 7 shipped moves; new policies are trained in sim with `microduck_rl` (mjlab/MuJoCo Warp, PPO), which **needs a CUDA GPU**, i.e. not this Mac | **Paused** (Issue 12 = A): needs a validated reaction reader (the REFLEX face probes read window length, L18) **and** hardware ready **and** the maintainer's opt-in |
| DW7 | **Robot transfer, revisited (offline H3, a new pre-registered spec).** It uses Wave B's REFLEX reaction model and third-person tabletop robot/human manipulation data (`reflex` → `roboreward`). v1's lessons apply: check a derived pool's class counts per stratum (L15); compare priors at equal norm, and check each head's zero-shot transfer first (L16) | Wave B shows a reaction gain (an ablation gain with `ci_low > 0`) **and** the designer has written the spec (Issue 10 = A). **Not met by Wave B v1** (ablation Δ `ci_low` −0.004 and −0.002). **Not selected** (Issue 12 = A) |
| DW8 | **A decisive hidden-outcome test with faces** (Issue 12's option D): for example CANDOR, 1,656 video-chat conversations with each person's private rating afterward. It needs a request to BetterUp and an emailed request to TalkBank | **Maintainer opt-in only.** The designer drafts any request and the maintainer approves it before it is sent |
