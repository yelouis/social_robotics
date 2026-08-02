# Pre-seeding runbook — read this, then work the loop until it says DONE

## ★ GOAL — the task is not finished until this is true

**Seed every moment in the kit set.** You are done when, and only when:

```bash
venv/bin/python bench/preseed.py --model $MODEL --status
# -> "this_model": { "remaining": 0 }
```

equivalently, when `--next` returns an empty array `[]`. That is the **single**
completion condition. At the time of writing there are ~349 moments; `--status`
is the authority, not this number.

### Do not stop before `remaining` is 0

This is a long, repetitive job — a few hundred moments, five at a time. It is
*expected* to take many batches. Specifically:

- **Finishing a batch is not finishing the task.** After every `--record`, go
  straight back to §2 Step 1 and pull the next batch.
- **Do not ask for permission to continue.** You already have it. Continue
  automatically until `remaining` is 0.
- **Reporting progress is not stopping.** Print your one-line status and keep
  going in the same turn.
- **Do not stop because the work feels repetitive, or because you have done
  "enough", or because a milestone looks tidy.** There are no milestones other
  than `remaining: 0`.
- **A rejected batch is not a stop condition.** Fix the fields the recorder
  listed and re-run the same `--record`.
- **If you are interrupted or restarted**, resume with §3 and carry on — the
  goal is unchanged and your progress is on disk.

### The only legitimate reasons to stop early

Stop, and say plainly which one applies:

1. The audio gate in §1 fails — you cannot hear the known-good clip.
2. The same batch fails validation three times and you cannot see why.
3. `--status` reports 0 kits, or the paths in this file do not exist.
4. The human tells you to stop.

Anything else: keep going.

---

You are pre-rating short clips from a head-mounted (egocentric) camera for the
**SocialRobotics-Bench v0** pilot. A human reviews and corrects everything you
produce. Your job is an honest first pass **with a stated reason for every
answer** — not to be right at all costs.

This file is your only source of truth. **Return to §2 (THE LOOP) after every
batch.** You will be interrupted or run out of context long before the work is
finished; §3 tells you how to resume with nothing remembered.

Project root (all commands run from here):

```
/Users/louisye/Desktop/Louis/social_robotics
```

Set these once per session and reuse them:

```bash
cd /Users/louisye/Desktop/Louis/social_robotics
PY=venv/bin/python
MODEL=<your-model-id>        # e.g. gemini-3-pro — pick ONE and never change it
```

**You must watch the actual video with its audio.** This task is video-only:
seeding from extracted stills is not an option here, because speech, laughter
and tone routinely decide B3, B5 and B6, and a silent read of them is
systematically wrong in ways the reviewer cannot see. If you cannot receive an
mp4 with a working audio track, **stop and tell the human** — do not substitute
frames, screenshots, or a description of the video obtained from elsewhere.

---

## 0. Hard rules (violating these silently ruins the dataset)

1. **The judgement must be yours, formed by watching the clip.** Never open,
   read, or ask for another model's seeds (`bench_v0/seeds/*.jsonl`) or the
   human's `ratings_maintainer.csv`. Several models seed the same moments
   *independently*; agreement between them is only meaningful if you never saw
   the others. This is the single most important rule.

   You **may** build or use tools that help *you* perceive the clip better —
   a script that slows playback, extracts a crop, isolates or amplifies the
   audio, or produces a speech transcript. Those give you raw perception.
   You **may not** hand the clip to another model and copy back its *answer* —
   a captioner, a VLM, or an agent asked "what social feedback is this?" is a
   judgement, not a perception aid. The line: a tool may tell you **what is
   there**; only you may decide **what it means**.

   Whenever a tool contributed to an answer, say so in that field's
   `rationale` (e.g. "transcript from ASR: '…'; tone judged from the audio").

2. **Never let the project's own pipeline inform your answers.** Do not read
   `candidate_moments.jsonl` (it carries `engine_prefill`), anything under
   `bench_v0/03*_result.json`, the `segment_dataset/`, or the greyed
   machine-generated caption. Do not run the engine's layers (`src/layer_*`) to
   help you decide. The benchmark exists to test whether models can read social
   feedback; seeding from the pipeline's guesses would make it measure itself.
   The commands in this runbook never expose those to you — keep it that way.
3. **Every answered field needs a `rationale` entry.** A seed without one is
   rejected. Cite what you saw or heard and roughly when ("frames 3–4", "at ~2 s").
4. **`voice` in B6 only if you actually heard it in this clip.** Never infer
   speech from moving mouths and never claim to have heard something you did
   not. Many clips are silent; that is a fact about the clip, not a failure.
5. **Never edit `ratings_maintainer.csv`.** That file belongs to the human.
   You write only via `preseed.py --record`.
6. **Say when you are unsure.** `unsure` / `cant_tell` / `unclear` are real,
   scored answers. A confident wrong seed costs the reviewer more than an honest
   uncertain one, because it is more likely to be accepted without scrutiny.

---

## 1. First run only — confirm your setup

```bash
$PY bench/preseed.py --model $MODEL --status
```

Prints how many kits exist and how many you have already seeded. If
`this_model.seeded` is `0`, you are starting fresh. If it is non-zero, you are
resuming — that is normal, go straight to the loop.

**Audio gate — do this before seeding anything.** Open this clip, which is known
to contain audible conversation:

```
/Volumes/Extreme SSD/social_robotics/bench_v0/kits/srb-ego4d-2e1f8114-7fd1-4728-82a1-a4b92ebb6af2-321000/clip_s.mp4
```

Confirm you can hear speech in it and report what you hear. Many Ego4D clips are
genuinely silent, so this specific clip is the test — a "no audio" answer here
means the audio is not reaching you, not that the clip is quiet.

If you cannot hear it: **stop and tell the human.** Do not proceed on stills and
do not ask another model to describe it for you.

---

## 2. THE LOOP — repeat until `remaining` is 0

### Step 1 — get your next batch (clips are found for you)

```bash
$PY bench/preseed.py --model $MODEL --next 5
```

This prints a JSON array of moments you have **not** yet seeded. Each entry has:

- `moment_id` — the id you must echo back in your seed
- `clip_moment` — absolute path to the 7 s moment clip (**with audio**)
- `clip_after` — absolute path to the ~9 s after-roll, or `null` if none exists
- `t_climax_sec`, `has_splus`

**You never need a human to give you file paths.** If the array comes back
empty, every moment is seeded — you are DONE; report that and stop.

Start with 5 per batch. Video calls are heavy; do not exceed what you can hold
without losing the batch.

### Step 2 — watch the moment clip only

Watch `clip_moment` — video and audio together. **Do not open `clip_after`
yet.** Answer A1–A2, then B1–B6.

Withholding the after-roll is not a formality: benchmark family F4 asks a model
to *predict* what B7 records. If you let the after-roll inform B1–B6, you leak
that answer into the question.

### Step 3 — then the after-roll, for B7 only

If B3 is `yes` and `clip_after` is not `null`, open it now and answer B7. If
`clip_after` is `null`, leave B7 empty.

### Step 4 — write the batch to a file

Write a JSON **array** (one object per moment) to a temp path, e.g.
`/tmp/seeds_batch.json`. Schema and definitions are in §4 and §5.

### Step 5 — record it (this is your checkpoint)

```bash
$PY bench/preseed.py --model $MODEL --record /tmp/seeds_batch.json
```

- Success prints `{"accepted": N, "total_seeded": M, "remaining": R, ...}` and
  appends a line to `bench_v0/seeds/$MODEL.progress.log`.
- Failure prints `{"rejected": [...]}` with the exact problems and writes
  **nothing** — fix the listed fields and re-run the same command.

**Record after every single batch.** An unrecorded batch is lost work. Never
hold two batches of unrecorded seeds.

### Step 6 — go back to Step 1 (do not end your turn here)

Do not carry anything forward in your head. Step 1 recomputes what is left.

**Check the completion condition, then act on it:**

- `remaining > 0` → immediately run Step 1 again. Do not pause, do not ask, do
  not summarise-and-halt. The loop continues in the same turn.
- `remaining == 0` → you are DONE. Report the final count and stop.

If you find yourself about to write "I have completed a batch" or "let me know
if you'd like me to continue" while `remaining > 0`, that is the failure this
runbook exists to prevent. Pull the next batch instead.

---

## 3. Checkpoint & resume protocol

**Your checkpoint is on disk, not in your context.** `preseed.py --record`
writes `bench_v0/seeds/$MODEL.jsonl` (your seeds) and appends to
`bench_v0/seeds/$MODEL.progress.log` (the history). `--next` skips anything
already in your seed file, so the loop is idempotent.

To resume — after a crash, a context reset, or days later — do exactly this:

```bash
cd /Users/louisye/Desktop/Louis/social_robotics
venv/bin/python bench/preseed.py --model <same-model-id> --status
tail -5 "/Volumes/Extreme SSD/social_robotics/bench_v0/seeds/<same-model-id>.progress.log"
```

Then go to §2 Step 1. Nothing else needs to be recovered.

**If you are interrupted mid-batch**, discard the partial batch and re-request
it — re-rating 5 moments is cheaper than reasoning about which ones landed.

**Report to the human** at each checkpoint, in one line:
`seeded <total>/<kits>, <remaining> left` plus anything
that surprised you (clips with no audio, unreadable footage, repeated `unclear`).

---

## 4. Output schema

```json
[
  {
    "moment_id": "srb-ego4d-<uuid>-<ms>",
    "A1": "yes", "A2": "yes", "A3": "",
    "B1": "hands a card to the player opposite",
    "B2": "yes", "B3": "yes", "B4": "wearer", "B5": "approving",
    "B6": "face;head_gesture", "B7": "continues", "B8": "somewhat",
    "rationale": {
      "A1": "Two people seated opposite are visible throughout.",
      "A2": "Dim but faces and hands are readable.",
      "B1": "Wearer's right hand extends a card across the table at ~3 s.",
      "B2": "Both players face the wearer across a small table.",
      "B3": "The player opposite goes from a flat expression to a broad smile at ~4 s.",
      "B4": "The smile starts right after the card is extended and they are looking at the wearer's hand.",
      "B5": "Smile plus a nod reads as acceptance of the offer.",
      "B6": "Judged from facial expression and a head dip; no speech audible.",
      "B7": "In the after-roll the wearer continues dealing to the next player.",
      "B8": "Clear footage, but the nod is brief."
    }
  }
]
```

**Skip logic — enforced, violations are rejected:**

- `A1` ≠ `yes` **or** `A2` ≠ `yes` → leave **every** B field empty (`""`).
- `B3` ≠ `yes` → leave `B4`, `B5`, `B7` empty.
- `B4` ≠ `wearer` → leave `B5` empty.

---

## 5. The questions

You see through the eyes of the **camera wearer**; their own hands enter frame
from the bottom. Everyone else is a **bystander**.

**A1 — Is at least one other person visible?** `yes` / `no` / `unsure`
Counts: any identifying body part (hands, arms, torso, legs); someone partly out
of frame or briefly passing; someone distant but clearly a person.
Not: the **wearer's own** hands/limbs/reflection (the commonest mistake); people
on a screen, poster or photo; shapes too small to identify.

**A2 — Watchable enough to judge behaviour if someone reacted?** `yes` / `no`
A legibility floor, not video quality. Dim, grainy or wobbly is still `yes`.
`no` only for near-black, covered lens, or whip-pan throughout.

**A3 — Content flags**, `;`-joined or empty: `minor` (a child is a main
participant) · `nudity_private` · `sensitive_info` (readable screens, documents,
plates) · `distress` · `other`.

**B1 — What does the camera wearer do?** Free text, verb first, present tense,
5–12 words, naming the target: "hands a card to the player opposite".
`conversation` if they are only talking/listening with no discrete physical act.
`unclear` if you cannot tell what they did (not merely hard to phrase).
*Trap*: when the most eye-catching motion belongs to a **bystander**, still
describe the **wearer** — usually `conversation` or "watches …".

**B2 — Is anyone positioned to notice that action?** `yes` / `no` / `unsure`
**Opportunity to perceive**, not whether they responded. `yes` if facing the
wearer, oriented to the shared activity, or close enough that ordinary speech
carries. `no` if turned away *and* distant, absorbed elsewhere, or behind a
barrier. `B2=yes` with `B3=no` is a valuable combination, not a contradiction.

**B3 — Does any visible person noticeably react?** `yes` / `no` / `unsure`
A **change of state**, not a steady state: compare the person at the start of
the clip with the same person at the end.
Counts: expression change; head nod/shake/turn; gesture; body movement (steps
closer or back, leans, flinches); audible speech onset, laugh or exclamation;
gaze shifting to or away from the wearer.
Not: simply continuing an activity; merely being present; apparent motion caused
by the **camera** moving — that is the wearer, not them.
Answer for **any** visible person, even one clearly reacting to their phone.

**B4 — What is that reaction responding to?** (only if `B3=yes`)
`wearer` / `something_else` / `cant_tell`
Weigh timing (did it follow the wearer's action closely?), orientation (turned
toward the wearer or the object involved?), and plausibility. `cant_tell` is a
real scored answer — never force a guess.

**B5 — How would the wearer read that reaction?** (only if `B4=wearer`)
`approving` / `disapproving` / `neutral` / `mixed`
Take the wearer's perspective: as feedback on what I just did, was this
positive, negative, or neither? Not "was the person happy" in general.
`neutral` = noticed with no evaluation. `mixed` = genuinely both, or readable
either way. `neutral` is **not** the same as "can't tell".

**B6 — Which signals did you actually use?** `;`-joined:
`face` · `head_gesture` · `hand_body` · `proxemics` · `gaze` · `voice`
Only what you used, not everything present. **`voice` only if you heard audio.**

**B7 — In the after-roll, does the wearer change what they are doing?**
(only if `B3=yes` and an after-roll exists)
`continues` / `adjusts` / `stops` / `cant_tell`
Descriptive only — what happened, not what should have. Judge the **wearer**,
not the bystanders. No claim that the reaction caused it.

**B8 — Your confidence:** `confident` / `somewhat` / `guessing`
Use `guessing` freely; those moments are dropped automatically, so honesty costs
nothing. If your answer hinges on motion you can only infer between stills, or
on audio you did not receive, `somewhat` is the ceiling.
