# SRB Labeling Guide (v0)

You are producing the **golden labels** for SocialRobotics-Bench. The engine's
guesses are hidden from you on purpose (anti-anchoring, docs/07 §4.6): answer
only from what you see and hear in the clips. There are no right-answer
quotas — honest "Unsure"/"Can't tell" answers make the benchmark *better*
(they become the calibration classes models are scored on).

## The form, one moment at a time

Open the moment's kit folder: watch `clip_s.mp4` (the moment, with audio),
then `clip_splus.mp4` (what happened next) when present. `strip.jpg` is only
a thumbnail for orientation. The grayed `task_label_hint` in the kit manifest
is machine-generated — ignore it whenever it disagrees with your eyes.

**Stage A (triage, ~10–15 s)** — A1: another person visible? A2: watchable?
A3: content flags (`minor / nudity_private / sensitive_info / distress / other`,
semicolon-separated). Any A3 flag quarantines the moment; keep rating the rest.

**Stage B (gold, ~60–90 s)** — answer with the short codes:

| Field | Question | Codes |
|---|---|---|
| B1 | What does the WEARER do? | free text, verb first; sentinels `conversation`, `unclear` |
| B2 | Anyone positioned to notice? | `yes / no / unsure` |
| B3 | Does anyone visibly react? | `yes / no / unsure` → if not `yes`, LEAVE B4–B7 BLANK |
| B4 | Reaction responding to…? | `wearer / something_else / cant_tell` → if not `wearer`, LEAVE B5 BLANK |
| B5 | Valence as the wearer would read it | `approving / disapproving / neutral / mixed` |
| B6 | Evidence channels (multi) | `face; head_gesture; hand_body; proxemics; voice; gaze` |
| B7 | What did the wearer do next? (needs clip_splus + B3=yes) | `continues / adjusts / stops / cant_tell` |
| B8 | Your confidence | `confident / somewhat / guessing` (guessing = auto-dropped from test, still useful) |
| B9 | Notes | free text |
| C1 | Missed moment in this clip? | timestamp + note (becomes a `human_flag` candidate) |

`seconds_spent` (optional but valuable): rough seconds this moment took you —
it decides the v1 tooling choice (docs/07 ⚠️ Issue 3).

## Worked examples (one per ambiguity-taxonomy case, docs/06)

1. **Conversation only** — wearer chats at a dinner table, hands idle.
   B1=`conversation` (NOT "talks about the food" — no discrete physical act).
   *Counter-example*: wearer chats WHILE passing a dish → B1="passes a dish
   to the person opposite" (the physical act wins).
2. **Indeterminate action** — motion blur during a whip-pan; you can't tell
   what the wearer did. B1=`unclear`, answer B2/B3 anyway (you can often
   still see people and reactions). *Counter-example*: action is small but
   visible (adjusts a knob) → caption it; `unclear` is for *cannot tell*,
   not *hard to phrase*.
3. **Attribution confound** — a bystander laughs, but they're looking at a
   phone, not the wearer. B3=`yes`, B4=`something_else`. If their gaze is
   off-screen and you genuinely can't tell the target: B4=`cant_tell` —
   that's a legitimate gold class, not a failure.
4. **Continuous companionship** — walking together, no discrete action.
   B1 = the locomotion phrase ("walks alongside the group"); B3 usually `no`
   (steady presence ≠ reaction). A nod mid-walk IS a reaction → B3=`yes`.
5. **Direction inversion** — the bystander acted FIRST and the wearer reacts.
   Rate what's asked: B1 = wearer's action in the window (often
   `conversation`), B4 = is the *bystander's* visible behavior a response to
   the wearer? If they initiated, usually `something_else` + note in B9.
6. **No-audience moments** — wearer acts alone (these are engine controls).
   A1=`no` is fine and expected — that IS the label. Don't stretch to find
   people in reflections.
7. **Masked/occluded faces** — visible person, unreadable face. B3 can still
   be `yes` via posture/movement/voice; tick only the B6 channels you
   actually used (e.g. `proxemics; voice`, no `face`).
8. **Bystander-action captions** — the most salient motion is a *bystander*
   bending/reaching, wearer just watching. B1 describes the WEARER
   (`conversation` or "watches the person dig") — never the bystander's act.

## Mechanics

- Fill `bench_v0/ratings_maintainer.csv` (any spreadsheet app; save as CSV).
- Validate early, in batches: `python bench/adjudicate.py --validate <csv>` —
  it catches enum typos and skip-logic slips while the clips are fresh.
- Retest pass (later, ≥3 days): same form, fresh order, no peeking at
  round-one answers — it measures YOUR consistency, and only counts if blind.
