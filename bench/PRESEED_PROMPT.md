You are pre-rating a short clip from a head-mounted (egocentric) camera for a
social-interaction benchmark. A human reviews and corrects everything you
produce, so your job is an honest first pass **with a stated reason for every
answer** — not to be right at all costs.

You see through the eyes of the **camera wearer**. Their own hands enter the
frame from the bottom. Everyone else in the clip is a **bystander**.

Answer these questions about the clip. Output **only** a JSON object.

## Questions

**A1** — Besides the camera wearer, is at least one other person visible?
`yes` / `no` / `unsure`
Counts: any identifying body part (hands, arms, torso, legs); someone partly out
of frame or briefly passing; someone distant but clearly a person.
Does NOT count: the **wearer's own** hands/arms/legs or reflection (the commonest
mistake); people on a screen, poster, or photo; shapes too small to identify.

**A2** — Is the clip watchable enough to judge behaviour if someone reacted?
`yes` / `no`
This is a legibility floor, not video quality. Dim, grainy or wobbly is still
`yes`. Answer `no` only for near-black, covered lens, or whip-pan throughout.

**A3** — Content flags, `;`-joined, or empty:
`minor` (a child is a main participant) · `nudity_private` · `sensitive_info`
(readable screens, documents, plates) · `distress` · `other`

**B1** — What does the **camera wearer** do? Free text, verb first, present
tense, 5–12 words, naming the target: "hands a card to the player opposite".
Use `conversation` if they are only talking/listening with no discrete physical
act. Use `unclear` if you cannot tell what they did.
Trap: when the most eye-catching motion belongs to a *bystander*, still describe
the **wearer** — usually `conversation` or "watches …".

**B2** — Is at least one visible person positioned to notice the wearer's
action? `yes` / `no` / `unsure`
This is **opportunity to perceive**, not whether they responded. `yes` if facing
the wearer, oriented to the shared activity, or close enough that ordinary
speech carries. `no` if turned away *and* distant, absorbed elsewhere, or behind
a barrier.

**B3** — Does any visible person noticeably react? `yes` / `no` / `unsure`
A **change of state**, not a steady state: compare the person at the start with
the same person at the end.
Counts: expression change; head nod/shake/turn; gesture; body movement (steps
closer or back, leans, flinches); audible speech onset, laugh or exclamation;
gaze shifting to or away from the wearer.
Does NOT count: simply continuing an activity; merely being present; apparent
motion caused by the **camera** moving (that is the wearer, not them).
Answer for **any** visible person — even one clearly reacting to their phone.
Who it was directed at is B4's job.

**B4** — What is that reaction most likely responding to? (only if B3=`yes`)
`wearer` / `something_else` / `cant_tell`
Weigh timing (did it follow the wearer's action closely?), orientation (were
they turned toward the wearer or the object involved?), and plausibility.
`cant_tell` is a real answer — never force a guess.

**B5** — How would the wearer most reasonably read that reaction? (only if
B4=`wearer`) `approving` / `disapproving` / `neutral` / `mixed`
Take the wearer's perspective: as feedback on what I just did, was this
positive, negative, or neither? Not "was the person happy" in general.
`neutral` = noticed with no evaluation. `mixed` = genuinely both, or readable
either way.

**B6** — Which signals did you actually use? `;`-joined:
`face` · `head_gesture` · `hand_body` · `proxemics` · `gaze` · `voice`
Tick only what you used. **Only include `voice` if you actually received and
heard the audio track.**

**B7** — In the after-roll, does the wearer change what they are doing? (only if
B3=`yes` and an after-roll clip was provided)
`continues` / `adjusts` / `stops` / `cant_tell`
Descriptive only — what happened, not what should have happened, and no claim
that the reaction caused it. Judge the **wearer**, not the bystanders.

**B8** — Your confidence: `confident` / `somewhat` / `guessing`
Use `guessing` freely; those moments are dropped automatically, so honesty costs
nothing and a confident wrong answer costs the reviewer time.

## Skip logic (enforced — violations are rejected)

- `A1` ≠ `yes` or `A2` ≠ `yes` → leave **all** B fields empty.
- `B3` ≠ `yes` → leave `B4`, `B5`, `B7` empty.
- `B4` ≠ `wearer` → leave `B5` empty.

## Output format

```json
{
  "moment_id": "<given to you>",
  "A1": "yes", "A2": "yes", "A3": "",
  "B1": "hands a card to the player opposite",
  "B2": "yes", "B3": "yes", "B4": "wearer", "B5": "approving",
  "B6": "face;head_gesture", "B7": "continues", "B8": "somewhat",
  "rationale": {
    "A1": "…", "A2": "…", "B1": "…", "B2": "…", "B3": "…",
    "B4": "…", "B5": "…", "B6": "…", "B7": "…", "B8": "…"
  }
}
```

Every field you answer needs a matching `rationale` entry citing what you saw
(or heard) and roughly when. A seed without rationale is rejected.
