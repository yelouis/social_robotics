# tools/

Versioned dev / QA helpers, promoted out of the gitignored `scratch/` workspace
so they are tracked and shareable. **None of these are part of the production
pipeline** — they only *drive* or *inspect* the productionized layers in `src/`
(layer logic lives in `src/`, not here).

## Conventions
- Run from the **repo root** with the project venv: `./venv/bin/python tools/<tool>.py …`
  (a couple need `PYTHONPATH=src`, noted below). Heavy deps (cv2, mediapipe,
  hsemotion-onnx, scipy, ultralytics) come from `venv`.
- Tools whose paths point at a dated `e2e_reports/<run>/` directory are
  **smell-test templates**: edit the `RDIR` constant near the top to target a
  new run. The reusable part is the logic, not the hardcoded path.

## Generic (argument-driven) tools
| tool | what it does | usage |
|---|---|---|
| `analyze_e2e.py` | Summarize a Node-02 E2E filter run: pass rates, per-gate rejection attribution (parsed from the detector's stdout log), best/worst clips, synthetic true-positive rate. | `./venv/bin/python tools/analyze_e2e.py --results RESULTS.json --log DETECTOR.log` |
| `extract_spotcheck.py` | Extract spot-check frames from any clip, optionally drawing a bystander box pulled from a result JSON. | `./venv/bin/python tools/extract_spotcheck.py --video PATH --prefix NAME --outdir DIR [--timestamps 10,60,120 | --n 3] [--boxes-from RESULTS.json --video-id ID]` |

## Layer smell-test runners (templates — edit `RDIR`)
| tool | what it does | usage |
|---|---|---|
| `run_03a_10.py` | Run Layer 03a (Attention) over a bounded sample manifest; resumable. | `./venv/bin/python tools/run_03a_10.py` |
| `run_03b_50.py` | Populate climax metadata, then run Layer 03b (Reasonable Emotion) over a sample manifest. | `./venv/bin/python tools/run_03b_50.py` |

## Visual spot-check renderers (templates — edit `RDIR`)
| tool | what it does | usage |
|---|---|---|
| `spotcheck_03a.py` | Render a gaze-overlay frame for a 03a result (bystander box + gaze-direction arrow + score/target), to judge whether the score matches where the bystander looks. | `./venv/bin/python tools/spotcheck_03a.py <video_id_prefix> <max|min> <out_name> [person_idx]` |
| `spotcheck_03b.py` | Sample a bystander crop across the reaction window, run the same ONNX HSEmotion, and overlay the emotion label + magnitude, to judge whether the detected emotion matches the face. | `PYTHONPATH=src ./venv/bin/python tools/spotcheck_03b.py <video_id_prefix> <out_prefix>` |

## Validators
| tool | what it does | usage |
|---|---|---|
| `validate_climax.py` | Recompute climax with the sequential-decode path on clips that already have an old `cap.set`-cached climax, compare timestamps, and measure the speedup (regression guard for `src/shared/climax_extraction.py`). Run from repo root. | `./venv/bin/python tools/validate_climax.py` |
