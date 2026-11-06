# Paper Artifact & Report Reproduction (`docs/paper/`)

All commands below run from the repository root (`/Users/louisye/Desktop/Louis/social_robotics`) using `./venv/bin/python` (Python 3.9.6) with `PYTHONPATH=src` and the external SSD environment variables exported. They are strictly read-only over `DATA_ROOT` (`/Volumes/Extreme SSD/social_robotics`) and `results/scorecard.jsonl`.

## 0. Environment Setup

```bash
export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
export TMPDIR="/Volumes/Extreme SSD/tmp"
export SR_NO_MODEL_BANNER=1
```

## 1. Rebuild Wave B Post-Hoc Diagnostics (Item B4b)

Reproduces the 12 post-hoc diagnostics in `docs/evals/2026-11-04_reflex_posthoc.json`:

```bash
PYTHONPATH=src ./venv/bin/python tools/diag_reflex_posthoc.py
```

## 2. Rebuild Paper Fact Base, Tables (`T1`–`T6`, `B1.md`–`B8.md`), and Figures (`F1`–`F3`) (Items E1, E1b & E1c)

Writes `docs/paper/facts.json`, `docs/paper/tables/T1.md`–`T6.md`, `docs/paper/tables/B1.md`–`B8.md`, and `docs/paper/figures/{f1_ablation,f2_window_length,f3_wave_d}.png`:

```bash
PYTHONPATH=src ./venv/bin/python tools/build_paper_facts.py
```

## 3. Verify Paper Draft Numbers and Embedded Tables (Items E1b, E2b & E1c)

Verifies that every `<!-- table:Xn -->` block (`T1`–`T6`, `B1.md`–`B8.md`) in `docs/paper/draft.md` matches `docs/paper/tables/Xn.md` byte for byte and that every numeric token in prose matches the facts named in its paragraph's `<!-- facts: ... -->` tag (`unmatched=0`):

```bash
PYTHONPATH=src ./venv/bin/python tools/check_paper_numbers.py docs/paper/draft.md
```

## 4. Regenerate Dated Evaluation Reports in Place (`docs/evals/`)

The five commands below regenerate the dated evaluation reports in `docs/evals/` **in place** (preserving each file's trailing `*Designer note*` block per A17). Running `git status --short docs/evals docs/paper` afterward must print nothing:

```bash
# Wave A: Oops! visible-outcome H1 report (docs/evals/2026-10-10_oops_h1.md)
PYTHONPATH=src ./venv/bin/python tools/report_oops.py --date 2026-10-10 --updated-date 2026-10-21 --updated-a15-date 2026-10-21

# Wave A: HoloAssist full H1 report (docs/evals/2026-10-15_holoassist_h1.md)
PYTHONPATH=src ./venv/bin/python tools/report_holoassist.py --dataset holoassist --date 2026-10-15 --corrected-date 2026-10-20 --updated-date 2026-10-21 --updated-a15-date 2026-10-21

# Wave A: HoloAssist audible-session H1 report (docs/evals/2026-10-21_holoassist_audible_h1.md)
PYTHONPATH=src ./venv/bin/python tools/report_holoassist.py --dataset holoassist_audible --date 2026-10-21 --updated-a15-date 2026-10-21

# Wave D: RoboReward offline H3 report (docs/evals/2026-10-22_wave_d_robot_check.md)
PYTHONPATH=src ./venv/bin/python tools/report_wave_d.py --date 2026-10-22

# Wave B: REFLEX multimodal human-robot reaction H1 report (docs/evals/2026-11-04_reflex_h1.md)
PYTHONPATH=src ./venv/bin/python tools/report_reflex.py --date 2026-11-04

# Verify clean reproduction in place (must print nothing)
git status --short docs/evals docs/paper
```
