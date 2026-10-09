# 03: Evaluation Harness — "Is the thesis working?"

The harness is how we find out, **often and without humans**, whether reactions are a reward signal. It is built *before* any model is tuned (Phase 0). v0 inverted that order and spent four months on features it could not evaluate.

## Contract

- **One command** prints the current scorecard: `python -m eval.scorecard` (to be built in Phase 0).
- **Every run appends** one row per `(hypothesis, dataset, condition, metric)` to `results/scorecard.jsonl` (tracked in git), stamped with the git SHA, date and config hash. That file *is* the project's progress history.
- **No step requires a human.** Labels come from the data ([`02_data_sources.md`](02_data_sources.md)); pass and kill thresholds come from [`00_thesis.md`](00_thesis.md).
- **Cheap by construction.** Expensive things are computed once and cached on the SSD: encoder features per `(dataset, clip, encoder)` and VLM-judge answers per `(clip, model, prompt hash)`. A scorecard refresh retrains small probes only, in minutes on the Mac Studio.

## H1 conditions (per dataset)

| Condition | Model sees | Purpose |
|---|---|---|
| **judge** | Action frames + item/task text, **reactions masked and audio muted** | The "just ask a VLM" control |
| **react-face** | Reactor face crops only | Implicit visual reaction |
| **react-nonverbal** | Non-verbal audio embeddings only (no transcript) | Implicit vocal reaction |
| **react-full** | Face + full audio + transcript | Upper bound; includes explicit words |
| **fusion** | judge score + react-face + react-nonverbal | The thesis claim: reaction adds to the judge |

## Metrics

- **AUROC** with a 95% bootstrap CI (1,000 resamples over *groups*, not items). Also **Spearman** against the raw score where one exists (verdict out of 10).
- **Δ = fusion − judge**, with its own bootstrap CI. This is the headline H1 number.
- **H2:** a zero-shot AUROC matrix (train source × target), plus a **scaling curve** (AUROC vs. training hours of web data on a log axis).
- **False-positive rate** on the archived Ego4D steady-state set: the share of moments where the model asserts a confident good or bad outcome when no evaluative reaction exists. This carries forward v0's "social hallucination" idea.

## Splits

- Always **grouped**: verdict videos by channel/reviewer; HoloAssist by instructor–performer pair; BAD by participant. An item's person never appears on both sides of a split.
- Split files are tracked in git (`splits/<dataset>.json`) with fixed seeds, and never regenerated silently.

## Encoders & judges (candidates; chosen in Phase 1 by what runs well on MPS)

- **Visual:** frame embeddings (SigLIP/CLIP-class) or a video encoder (V-JEPA/VideoMAE-class); a face detector + face-crop embedding for `react-face`.
- **Audio:** paralinguistic embeddings (emotion2vec+, which ran on this Mac in v0's 03c; used here as an *embedding*, never as emotion categories), plus a Whisper encoder.
- **Probe:** logistic regression or a small MLP over temporally pooled features. Bigger heads only once a small one shows signal.
- **VLM judge:**
  - local `qwen2.5vl` via `src/shared/vlm_client.py` (`models_config.get_model("vlm_judge")`);
  - Gemini API as the frontier anchor (the maintainer's Gemini access, as used for v0 pre-seeding).
  - The judge outputs a probability of a good outcome so AUROC is defined. Prompts are versioned, and a prompt hash is part of the cache key.

## Cadence

| Trigger | What reruns |
|---|---|
| Any code change to probes or fusion | Probe scorecard (minutes) |
| New data added | Feature extraction for the new clips only, then the scorecard |
| Judge prompt or model change | Judge answers (expensive, cached), then the scorecard |
| Weekly | Full scorecard, including H2 transfer and the scaling curve |

## Guardrails against fooling ourselves

1. **The judge control is mandatory.** No H1 result is reported without it.
2. **Leakage conditions are reported separately**, never only `react-full`.
3. **Grouped CIs only.** Item-level CIs overstate certainty when one reviewer contributes 50 items.
4. **Targets are never training data.** BAD, ERR@HRI and HoloAssist are H2 targets only, so transfer claims stay zero-shot.
5. **Kill criteria are written before results** ([`00_thesis.md`](00_thesis.md)). They change only through a dated entry in the decision log.
