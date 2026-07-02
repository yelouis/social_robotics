# Layer 04 — Dehydrated Export Smoke Test

**Date**: June 16, 2026 · **Goal**: confirm 04 aggregates real layer outputs and produces a publishable, dehydrated dataset (post-03g-removal).

## Setup
Real 03a–03f result JSONs from prior 50-clip runs, copied into `data/` under the **canonical** names the aggregator's `glob("03*_result.json")` + `stem.replace("_result","")` expect (the e2e `_50` suffix would otherwise be skipped / mis-named), over the canonical 50-clip `filtered_manifest.json`. Driver: `run_export.py`.

## Result: ✅ works end-to-end
- **Aggregated 50 rows × 36 cols**, `schema_version = 1.0.0+5dffe5`, `pipeline_git_sha = e860fc1`.
- **Dehydration validation PASSED** → wrote `export/social_metadata.parquet` (224 KB) + `export/export_metadata.json`.
- **Registry summary columns surface and reconcile with source runs**:
  | column | coverage | truthy | reconciles with |
  |---|---|---|---|
  | `03f_..._max_ego_chaos_score` | 34/50 | 34 | 03f dense-track: 34 scored |
  | `03f_..._any_motor_resonance` | 34/50 | 0 | 03f Resolved #10 gate (2→0) |
  | `03f_..._any_mirroring` | 34/50 | 4 | 6 mirroring person-rows → 4 videos (per-video `any`) |
  | `03c_..._avg_prosody_scalar` | 50/50 | 9 | |
  | `03d_..._max_proxemic_confidence` | 35/50 | 8 | |
  | `03e_..._any_nod_detected` | 10/50 | 5 | |
  | `03b_..._avg_task_score` | 2/50 | 2 | 03b sparse on this corpus |
- **Manifest join 50/50** (video_id, source_dataset, task_labels, duration_sec, fps).
- **NaN-fill** correct for partial layers (03a raw 23/50, 03b 2/50).

## Safety checks (all pass)
- **Dehydration guards are real (not no-ops)** — `export_parquet` RAISES on: a `/Volumes/...` raw path leak, a `synthetic_` video_id, and raw `bytes`; no partial parquet is written when it raises.
- **Publish path is wired + safely gated** — `upload_to_huggingface(token=None)` with `HF_TOKEN` unset logs "Skipping… Local files will remain intact" and returns **before any network call**. No upload performed.

## Published ✅ (private HF dataset `louisye/testing`, 2026-06-16)
Pushed via the project's own `upload_to_huggingface()` (`push_to_hf.py`). `whoami` = `louisye` (token_role **write**); repo went `['.gitattributes']` → 5 files: `social_metadata.parquet`, `export_metadata.json`, `README.md` (auto-generated dataset card), `rehydrate_dataset.py`. **Round-trip verified**: re-downloaded the parquet from the hub → 50 rows × 36 cols, 50 unique video_ids, 0 synthetic leaks, queryable columns intact.

Token handling: a plain `export HF_TOKEN=` in an interactive shell does NOT reach the tool-shell; used `HF_HOME='/Volumes/Extreme SSD/huggingface_cache' ./venv/bin/hf auth login` (SSD credential store) + resolved via `get_token()` and passed `token=` explicitly (the project upload only reads `HF_TOKEN` env, not the login store).

## Artifacts
`run_export.py`, `data/` (manifest + 6 canonically-named layer results), `export/` (parquet + metadata).
