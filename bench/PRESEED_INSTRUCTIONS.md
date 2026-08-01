# Pre-seeding — operator guide

**For the human setting this up.** The agent doing the work reads
[`PRESEED_PROMPT.md`](PRESEED_PROMPT.md) — that file is the complete,
self-contained runbook (hard rules, the loop, checkpoints, every question
definition). Hand it over as the system prompt and nothing else is needed.

This file covers what *you* decide: which model, how to verify it before a run,
and how to read the results.

---

## What pre-seeding is (and what it is not)

A video-capable model watches each moment and fills the Stage-A/B form with a
written reason per answer. You then review in the rating UI: where the seeding
models **agree**, the field is pre-filled; where they **disagree**, it is left
blank with each model's answer one click away.

Seeds are **advisory input to you**, never gold. `golden_labels.jsonl` is built
from *your* `ratings_maintainer.csv` alone — `adjudicate.py` never reads a seed
file. This is an acknowledged deviation from the docs/07 §4.6 anti-anchoring
policy, and it is disclosable because the seed audit records which models you
ended up agreeing with (see below).

---

## Choosing a seeder

**Requirement: the model must accept the mp4 with its audio track.** Speech,
laughter and tone routinely decide B3 (did anyone react), B5 (how the wearer
reads it) and B6 (`voice`). Gemini via the Files API is the usual fit.

`--modality frames` exists as a deprecated escape hatch for a seeder that
genuinely cannot take video. It forbids `voice` in B6 and stamps seeds
`blind:["voice"]`. Where tone matters, prefer no seed over a deaf one.

**Running more than one model is worth it.** Independent seeds turn disagreement
into a signal: a conflict means the moment is genuinely ambiguous, and the UI
surfaces it for your judgement instead of prefilling a winner. Independence is
the whole value — never let one seeder see another's file.

---

## Verify before a full run

Three checks, all cheap. Paste into the model with the clip attached.

**1. Audio actually arrives.** Use this clip — known-good AAC audio, people
audibly talking:

```
/Volumes/Extreme SSD/social_robotics/bench_v0/kits/srb-ego4d-2e1f8114-7fd1-4728-82a1-a4b92ebb6af2-321000/clip_s.mp4
```

> This is a 7-second clip from a head-mounted camera. Answer literally; I am
> testing whether you received the audio track, not whether you can describe the
> scene. (1) Did you receive an audio track? YES or NO. (2) Is human speech
> audible, and roughly how many voices? (3) Quote or paraphrase any words, or say
> "speech present, words unintelligible". (4) Describe the tone in three words.
> Then state: "I answered from AUDIO" or "I answered from VISUALS ONLY".

Pass = audio present, multiple voices, a tone. Fail = "no audio", or it
describes mouth movements. Most Ego4D clips *are* silent, which is why this
specific clip is the test.

**2. It respects the after-roll boundary.** Same clip, same chat:

> Without me giving you further video: what does the wearer do in the NEXT 9
> seconds, after this clip ends? If you do not know, say exactly "I have not been
> shown that footage."

Pass = it declines. Fail = it invents a continuation, which means its B7 answers
will be confabulated.

**3. It is working only from the clip.**

> List every source of information you are using. If you have been given any
> prior answers, model predictions, or metadata about this clip beyond the video
> itself, say so explicitly.

Pass = it names only the video and your instructions. Never send it
`bench_v0/seeds/*.jsonl`, `candidate_moments.jsonl` (carries `engine_prefill`),
or the greyed machine caption.

---

## During and after the run

Progress, any time:

```bash
venv/bin/python bench/preseed.py --status
tail -5 "/Volumes/Extreme SSD/social_robotics/bench_v0/seeds/<model>.progress.log"
```

Cross-model agreement and the list of conflicting moments — the moments most
worth your attention first:

```bash
venv/bin/python bench/preseed.py --agreement
```

Then rate as usual (`venv/bin/python bench/rate_server.py`). Two things the UI
guarantees: a moment you have already saved is never overwritten by a seed, and
the blind retest round is served **without seeds** — showing them there would
measure agreement-with-the-model instead of your own self-consistency.

Afterwards, `ratings_maintainer_seed_audit.csv` records per moment which models
you agreed with, which you diverged from, and where the models conflicted. Those
numbers are what the pilot's datasheet must disclose alongside the
*single-annotator, dev-grade* label.
