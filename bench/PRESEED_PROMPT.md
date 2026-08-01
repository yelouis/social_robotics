# Pre-seeding runbook — read this, then work the loop until it says DONE

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
MODALITY=video               # 'video' if you receive the mp4 WITH audio, else 'frames'
```

---

## 0. Hard rules (violating these silently ruins the dataset)

1. **Never open, read, or ask for another model's seeds.** Not
   `bench_v0/seeds/*.jsonl`, not the human's `ratings_maintainer.csv`. Several
   models seed the same moments *independently*; agreement between them is only
   meaningful if you never saw the others. This is the single most important rule.
2. **Never read the pipeline's own predictions** — `candidate_moments.jsonl`
   (it contains `engine_prefill`), or anything under `bench_v0/03*_result.json`.
   The benchmark exists to test whether models can do this; seeding from the
   pipeline's guesses would make it circular. The commands below never expose
   them to you.
3. **Every answered field needs a `rationale` entry.** A seed without one is
   rejected. Cite what you saw or heard and roughly when ("frames 3–4", "at ~2 s").
4. **`voice` in B6 only if you actually heard audio.** On `MODALITY=frames` the
   recorder rejects it. Never claim to have heard something you did not.
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

If `MODALITY=video`, verify on your first clip that audio really reached you
(ask yourself: can I hear speech?). Many clips are genuinely silent, so test on
one where people are visibly talking. If you cannot hear anything on such a
clip, switch to `MODALITY=frames` and say so to the human.

---

## 2. THE LOOP — repeat until `remaining` is 0

### Step 1 — get your next batch (clips are found for you)

```bash
$PY bench/preseed.py --model $MODEL --modality $MODALITY --next 5
```

This prints a JSON array of moments you have **not** yet seeded. Each entry has:

- `moment_id` — the id you must echo back in your seed
- `clip_moment` — absolute path to the 7 s moment clip (**with audio**)
- `clip_after` — absolute path to the ~9 s after-roll, or `null` if none exists
- `frames_moment` / `frames_after` — extracted JPEGs (only on `--modality frames`)
- `t_climax_sec`, `has_splus`

**You never need a human to give you file paths.** If the array comes back
empty, every moment is seeded — you are DONE; report that and stop.

Start with 5 per batch. Video calls are heavy; do not exceed what you can hold
without losing the batch.

### Step 2 — watch the moment clip only

Open `clip_moment` (or the `frames_moment` stills). **Do not open `clip_after`
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
$PY bench/preseed.py --model $MODEL --modality $MODALITY --record /tmp/seeds_batch.json
```

- Success prints `{"accepted": N, "total_seeded": M, "remaining": R, ...}` and
  appends a line to `bench_v0/seeds/$MODEL.progress.log`.
- Failure prints `{"rejected": [...]}` with the exact problems and writes
  **nothing** — fix the listed fields and re-run the same command.

**Record after every single batch.** An unrecorded batch is lost work. Never
hold two batches of unrecorded seeds.

### Step 6 — go back to Step 1

Do not carry anything forward in your head. Step 1 recomputes what is left.

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
`seeded <total>/<kits>, <remaining> left, modality=<video|frames>` plus anything
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
