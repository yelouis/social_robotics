# Pre-seeding instructions (for a model/agent, including a future session of me)

You are pre-rating social-interaction moments for **SocialRobotics-Bench v0**.
A human maintainer reviews and corrects every answer you produce — your job is
to give them a **defensible first pass with a stated reason for each answer**,
not to be right at all costs.

Read this whole file before your first batch. It is written to be resumable:
you will run out of context long before the work is done, and everything below
is designed so the next session (yours or another model's) continues cleanly.

---

## 0. Hard rules

1. **Answer from the video frames only.** Never read, request, or copy the
   engine's own predictions (`engine_prefill` in `candidate_moments.jsonl`, or
   anything under `03*_result.json`). Those are the *pipeline's* guesses; the
   whole point of this benchmark is that its labels are not derived from them.
   `preseed.py` deliberately never exposes them to you.
2. **State a reason for every field you answer.** A seed without a rationale is
   rejected by the recorder. The rationale is what makes your answer reviewable
   — the human reads it to decide whether to keep or change your pick.
3. **Know which modality you are using — it changes what you may claim.**
   - **Frame track** (`--modality frames`, the default): you see extracted
     stills only. You cannot hear anything. Never tick `voice` in B6 — the
     recorder rejects it — and where audio would plausibly decide the answer,
     say so in the rationale and lower B8. Seeds are stamped `blind:["voice"]`.
   - **Native-video track** (`--modality video`): the model received the mp4
     itself, audio included (see §2b). You may cite tone, speech, and laughter,
     and you may tick `voice`. Seeds are stamped `blind:[]`.

   Do not use `--modality video` unless the model genuinely received the audio
   track. A silent-video model claiming voice evidence is worse than a frame
   model admitting it cannot hear.
4. **Say when you are unsure.** `unsure` / `cant_tell` / `unclear` are real,
   scored answers. A confident wrong seed costs the maintainer more time than
   an honest uncertain one, because it is more likely to be accepted silently.
5. **Never edit `ratings_maintainer.csv`.** That file is the human's. You write
   only to `bench_v0/seeds/<your-model-id>.jsonl` via `preseed.py --record`.

---

## 1. What you are labelling

Each **moment** is a ~7 s clip cut around a detected social climax, plus a ~9 s
**after-roll** showing what the camera wearer did next. Footage is egocentric:
you see through the wearer's eyes, and the wearer's own hands enter frame from
the bottom.

The full, authoritative definition of every question — what counts, what does
not, what each option means — is in **`bench/rater.html`** in the `SCHEMA`
array (each question's `help` block). **Read those definitions before you start.**
`bench/LABELING_GUIDE.md` adds worked examples for the ambiguous cases. Do not
work from your own intuition about the question wording; the definitions are
deliberately operational so that different raters (and different models) resolve
edge cases the same way.

Quick reference of the fields you produce:

| Field | Values |
|---|---|
| `A1` | `yes` / `no` / `unsure` — another person visible |
| `A2` | `yes` / `no` — watchable enough to judge behaviour |
| `A3` | `` or `;`-joined: `minor`,`nudity_private`,`sensitive_info`,`distress`,`other` |
| `B1` | free text, verb first (or `conversation` / `unclear`) |
| `B2` | `yes` / `no` / `unsure` — someone positioned to notice |
| `B3` | `yes` / `no` / `unsure` — someone visibly reacts |
| `B4` | `wearer` / `something_else` / `cant_tell` (only if `B3=yes`) |
| `B5` | `approving` / `disapproving` / `neutral` / `mixed` (only if `B4=wearer`) |
| `B6` | `;`-joined: `face`,`head_gesture`,`hand_body`,`proxemics`,`gaze` (never `voice`) |
| `B7` | `continues` / `adjusts` / `stops` / `cant_tell` (only if `B3=yes`, needs after-roll) |
| `B8` | `confident` / `somewhat` / `guessing` |

**Skip logic is enforced.** If `A1≠yes` or `A2≠yes`, leave every B field empty.
If `B3≠yes`, leave B4/B5/B7 empty. If `B4≠wearer`, leave B5 empty. The recorder
rejects violations, so you cannot silently corrupt the form.

---

## 2. The loop (repeat until done)

```bash
cd /Users/louisye/Desktop/Louis/social_robotics
PY=venv/bin/python
MODEL=claude-opus-5            # your model id — pick one and keep it stable
MODALITY=frames                # or: video   (see §2b)

# 1. Where am I?
$PY bench/preseed.py --model $MODEL --status

# 2. Get the next batch (start with 5; tune to what your context allows)
$PY bench/preseed.py --model $MODEL --modality $MODALITY --next 5
```

`--next` prints a JSON array. Each entry has `moment_id`, `t_climax_sec`,
`has_splus`, plus:

- `clip_moment` / `clip_after` — absolute paths to the **mp4s with audio**
  (use these on the native-video track);
- `frames_moment` / `frames_after` — 6 extracted JPEGs each (frame track only;
  omitted when `--modality video`).

**3. Watch the moment.** On the frame track, open every `frames_moment` image.
On the native-video track, send `clip_moment` itself. Either way, hold back
`clip_after` / `frames_after` until you need B7 — answer B1–B6 first, because
knowing what happened next biases the read of the reaction itself (the same
rule the human raters follow).

**4. Write your seeds** to a temp JSON file — a list of objects:

```json
[
  {
    "moment_id": "srb-ego4d-<uuid>-<ms>",
    "A1": "yes", "A2": "yes", "A3": "",
    "B1": "hands a card to the player opposite",
    "B2": "yes", "B3": "yes", "B4": "wearer", "B5": "approving",
    "B6": "face;head_gesture", "B7": "continues", "B8": "somewhat",
    "rationale": {
      "A1": "Two people seated across the table are visible in all six frames.",
      "A2": "Indoor light is dim but faces and hands are clearly readable.",
      "B1": "Wearer's hand extends a card toward the player opposite in frames 3-4.",
      "B2": "Both players face the wearer across a small table, well within a metre.",
      "B3": "The player opposite shifts from a flat expression to a broad smile between frames 4 and 5, and their head dips.",
      "B4": "The smile begins immediately after the card is extended and they are looking at the wearer's hand, not elsewhere.",
      "B5": "Smile plus a nod reads as acceptance of the offered card.",
      "B6": "Judged from facial expression and the head dip; audio unavailable.",
      "B7": "In the after-roll the wearer continues dealing to the next player.",
      "B8": "Frames are clear, but a nod is hard to confirm from stills alone."
    }
  }
]
```

**5. Record it** (validates enums + skip logic, appends, and is idempotent):

```bash
$PY bench/preseed.py --model $MODEL --modality $MODALITY --record /tmp/seeds_batch.json
```

It prints `{"accepted": N, "total_seeded": M, "remaining": R}`. **That printed
`remaining` is your checkpoint** — nothing else needs to be remembered between
sessions. If it prints `rejected`, fix the listed fields and re-record; nothing
is written unless the whole batch validates.

**6. Repeat from step 1.** A new session picks up exactly where the last stopped,
because `--next` skips moments already in your model's seed file.

---

## 2b. Native-video track (models that accept the mp4 with audio)

If your model takes video directly — Gemini via the Files API is the usual case
— **prefer it**. The clips carry their original audio, and speech, laughter and
tone are decisive evidence for B3 (did anyone react), B5 (how the wearer would
read it), and B6 (`voice`). A frame-based seeder is structurally blind to all of
it, so an audio-capable seed is the more valuable one.

Run with `--modality video`. Send `clip_moment` (7 s, audio included), and only
fetch `clip_after` when you reach B7.

A minimal Gemini recipe — one moment, one call:

```python
import json, os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
moment = json.loads(open("/tmp/batch.json").read())[0]      # from --next

f = client.files.upload(file=moment["clip_moment"])         # audio comes with it
prompt = open("bench/PRESEED_PROMPT.md").read() + "\n\nmoment_id: " + moment["moment_id"]

resp = client.models.generate_content(
    model="gemini-3-pro",                                   # any video+audio model
    contents=[f, prompt],
    config={"response_mime_type": "application/json"},
)
print(resp.text)          # -> append to your batch seeds file
```

Notes that matter:

- **Confirm the audio actually arrived.** Ask the model, on the first clip, to
  describe any sound it hears. Many Ego4D clips are genuinely silent, so a
  single "no audio" answer proves nothing — check on a clip where people are
  visibly talking. If it cannot hear, switch back to `--modality frames`; a
  silent model claiming voice evidence is the one failure mode this design
  cannot detect for you.
- **Upload the moment clip only** until you need B7. Sending both at once leaks
  the after-roll into the B1–B6 answers and destroys the F4 boundary.
- **Batch by API limits, not context.** Video calls are heavier than image
  calls; 3–5 moments per run is a sane start.
- Ask for JSON matching the schema in step 4 exactly (`response_mime_type` above
  does most of the work), then hand the array straight to `--record`.

`bench/PRESEED_PROMPT.md` holds a ready-made system prompt with the question
definitions inlined, so a non-agentic model can be driven from a plain script.

---

## 3. If you are a *different* model from the one that started

Use your own `--model` id. Seeds are stored per model in
`bench_v0/seeds/<model>.jsonl`, so several models can seed the same moments
independently and none overwrites another. **Do not read another model's seed
file before forming your own answer** — independent seeds are the entire value
of running more than one model. Where models agree, the UI pre-fills the answer;
where they disagree, it deliberately leaves the field blank and shows both
options for the human to choose between. Agreement is only meaningful if the
seeds were produced independently.

Check cross-model agreement any time:

```bash
$PY bench/preseed.py --agreement
```

---

## 4. Judgement guidance that repeatedly matters

- **B1 describes the WEARER**, never the most eye-catching person. If a bystander
  is doing something dramatic and the wearer is just present, that is
  `conversation` or `watches …`.
- **B3 is a change of state**, not a steady state. Compare the first frame with
  the last: would someone say "something happened there"? Camera movement is the
  *wearer* moving, not a reaction.
- **B3 covers anyone visible**, including a person clearly reacting to their
  phone. Whether it was directed at the wearer is B4's job, not B3's.
- **B2 is opportunity, not response.** `B2=yes` with `B3=no` is a valuable
  combination (someone could have reacted and did not), not a contradiction.
- **B5 is the wearer's reading of the reaction**, not the bystander's mood in
  general.
- **Stills under-detect nods and flinches.** When your answer hinges on motion
  you can only infer between frames, say so and set `B8` to `somewhat` at best.
- **`unclear` for B1** means you cannot tell what the wearer did — not that it is
  hard to phrase.

---

## 5. Batch sizing and cost

Six frames per clip and up to twelve per moment with the after-roll. Start at
**5 moments per batch**; if context allows, raise it. Record after *every*
batch — an unrecorded batch is lost work when the session ends. Never hold more
than one batch of unrecorded seeds.

The current kit set is ~200 moments, so expect on the order of 40 batches of 5.
There is no deadline and no penalty for stopping: `--status` always tells the
next session exactly what remains.
