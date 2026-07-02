# E2E Run Archive (May–June 2026)

The raw artifacts of the early validation runs (result JSONs, debug frames, per-run
logs — ~120 MB, git-ignored) were **deleted on July 1, 2026** during a cleanup.
Every run's *report* is preserved verbatim in this directory
(`<run_dir>__<report_name>.md`), and the engineering conclusions each run produced
live durably in the per-layer docs' Resolved/Unresolved history — this index maps
runs → findings → where the detail lives.

**What was kept as raw data** (still in `e2e_reports/`):
- `2026_06_29_density_landing_top200/` — the CURRENT top-200 landing: `input_top200.dense.json`
  (the manifest of record), 03d/03f dense results, per-layer datasets, `LANDING_REPORT.txt`.
- `2026_06_16_per_layer_publish/` — the exact data published to HF + the publish/card tooling
  (`publish.py`, `republish.py`, `generate_cards.py`).
- `2026_06_14_layer03e_50/03a_attention_result_50.json` — the only pre-991 full 03a trace
  (23 clips; climax-independent, still valid input for 03e re-evals).
- `500_video_test_report.md` (e2e_reports root).

## Run index

| Run | What it validated | Key finding (detail in) |
|---|---|---|
| 2026_05_16 → 05_18 (100-video ×3) | Nodes 01+02 e2e at 100 videos, iterating gate fixes | Filter precision/recall iterations; VLM verification cascade tuning → docs/02 Resolved #5, #9–#14 |
| 2026_05_19 (500-video) | Node 01+02 scale-up | Throughput + memory bounds at 500; survived → docs/02 |
| 2026_05_22 (150-video) | Post gate-accuracy fixes (+ then-Layer 1a) | Gate accuracy re-verified; Layer 1a later removed entirely (PR #90) |
| 2026_06_02_layer03a (10-clip, v1–v5) | 03a functional smell test | Gaze scores match visual attention; face-gate + bbox-redetect iterations → docs/03a Resolved #1, #2, #5 |
| 2026_06_04_layer03b (10-clip) | 03b functional smell test | Emotion direction matches faces on resolvable crops → docs/03b |
| 2026_06_04_layer03b_50 (50-clip) | 03b at 50 + cost levers | **Only ~2/50 clips yield confident emotion** (egocentric faces too small); climax was the dominant cost → face-quality prefilter (84% skip, 0 false neg) + climax sequential-decode speedup → docs/03b Resolved #7, #8 |
| 2026_06_08_layer03c_50 + 06_09 postfix | 03c at 50 | Many Ego4D clips are silent — `audio_present:false` is normal; no-audio ffmpeg flood fixed → docs/03c |
| 2026_06_09/06_10/06_13 layer03d (50-clip ×4) | 03d smell test + fixes | June 9: 0/50 dead yield (strict windows) → window anchoring + SAM MPS float32 → June 10 producing; June 13: identity collision + span cap fixed → docs/03d Resolved #1–#6 |
| 2026_06_14_layer03d_50_spotcheck | 03d visual spot-check | Boxed-frame QA of approach/avoidance calls (index.html) → docs/03d |
| 2026_06_14_layer03e_50 | 03e at 50 (kept 03a trace) | Multi-window climax rescued 03e 0/200→producing; window re-anchoring + dedup guardrails → docs/03e Resolved #1–#6 |
| 2026_06_14_layer03e_headpose | 6DoF head-pose recovery experiment | Did NOT validate — gaze-derived gestures are noise (VOR decouples gaze from head-nod), head-pose-only is the trustworthy signal → docs/03e Resolved #11 |
| 2026_06_14/06_15 layer03f (50-clip ×4) | 03f sparse→dense→gated iterations | Sparse: 0/50; dense pose sampling: producing; FP gate: 9→2; dense-track gate: camera-motion FPs gone → docs/03f Resolved #9–#12 |
| 2026_06_16_layer04_smoke | 04 dehydrated export | Aggregation + dehydration produce publishable parquet; canonical `03*_result.json` naming contract → docs/04 |
| 2026_06_29_density_landing_top200 (KEPT) | Dense-manifest landing on top-200 | 03f motor resonance 65→180, 03d non-Neutral 185→207 + `proxemic_trajectory_shape` distribution → `LANDING_REPORT.txt` |

## Superseded-data warning

All manifests/results above predate the **June 30 Layer 02b climax rework**
(`docs/02b_task_climax_layer.md`): their `task_temporal_metadata` was produced by the
retired optical-flow detector (windows trail the climax; velocity-scaled lengths).
Numbers derived from *window placement* (03d/03e/03f yields) are not directly comparable
to post-02b runs. The 03a traces are climax-independent and remain valid.
