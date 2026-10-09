# Social Robotics: Human Reactions as Robot Reward

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Thesis.** People react to what others do: a wince, a laugh, a "yes!", a face at the first bite. Those reactions are a reward signal. We test whether that signal:

1. tells us something about an action's outcome **that a VLM judge watching the action cannot infer**;
2. can be learned **at internet scale from ordinary video, with no human labeling**;
3. makes **robot learning better** when added to the conventional RL reward.

**Principle.** Supervise on **outcomes**, never on emotion. We never label which emotion someone shows. We only ask whether their reaction predicts if the action went well, using outcome labels that come from the data itself: a spoken verdict, a mistake annotation, a failure timestamp. Every metric is automatic.

## Status (October 2026)

- **Reoriented.** The v0 Social-Affective Filter pipeline is archived at tag [`v0-saf-final`](../../tree/v0-saf-final). Why: [`docs/LESSONS_v0.md`](docs/LESSONS_v0.md).
- **Next:** Phase 0. Build the eval harness and request dataset access. Then Phase 1, the H1 pilot on verdict videos. See the roadmap in [`docs/00_thesis.md`](docs/00_thesis.md).

## Documentation

| Doc | Contents |
|---|---|
| [`00_thesis.md`](docs/00_thesis.md) | **Read first.** Thesis, hypotheses H1–H3 with pass/kill criteria, roadmap, decision log |
| [`01_prior_work.md`](docs/01_prior_work.md) | EMPATHIC, BAD, ERR@HRI and others, and where we differ |
| [`02_data_sources.md`](docs/02_data_sources.md) | Verdict videos, HoloAssist, Oops!, robot-reaction sets; labeling and leakage controls |
| [`03_eval_harness.md`](docs/03_eval_harness.md) | The automatic scorecard: conditions, metrics, splits, cadence |
| [`LESSONS_v0.md`](docs/LESSONS_v0.md) | What the v0 pipeline taught us (negative results + operations) |

## Layout

```
docs/                 grounding docs (above)
src/config.py         DATA_ROOT (external SSD)
src/models_config.py  local model tier registry (vlm_judge)
src/shared/           vlm_client.py: ollama calls with an enforced timeout
tools/                daemonize.py + run_supervised.sh for long unattended runs
tests/
```

New code lands in `src/` packages as the phases need it: data loaders per source, feature extraction, `eval/`.

## Setup

- **Host:** Mac Studio (M4 Max, 64 GB). Raw video lives only on the external SSD (`DATA_ROOT`, see `.env.example`).
- **Environment:** `venv` (Python 3.9). [Ollama](https://ollama.com/) for the local VLM judge. Optional Gemini API key for the frontier judge.
- **Tests:** `./venv/bin/python -m pytest tests/`

## License & Ethics

MIT. We never redistribute video pixels we do not own: releases contain IDs, timestamps, labels and derived features only.
