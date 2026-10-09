# 03: Evaluation Harness: "Is the thesis working?"

The harness tells us, **often and without humans**, whether reactions are a reward signal. It is built *before* any model is tuned (Wave A). v0 inverted that order and spent four months on features it could not evaluate.

**This document is the contract.** Every path, field name, constant and prompt below is a decision. The build spec that implements it, item by item, is [`agent_execution_guide.md`](agent_execution_guide.md).

---

## 1. Principles

- **One command** prints the current scorecard: `PYTHONPATH=src ./venv/bin/python -m harness.scorecard`.
- **Every evaluation run appends** rows to `results/scorecard.jsonl` (tracked in git). That file *is* the project's progress history. Rows are never edited or deleted. A wrong row is superseded by a later row with `notes` explaining why.
- **No step requires a human.** Labels come from the data ([`02_data_sources.md`](02_data_sources.md)); pass and kill thresholds come from [`00_thesis.md`](00_thesis.md).
- **Cheap by construction.** Encoder features and judge answers are computed once and cached on the SSD. A scorecard refresh retrains small probes only, in minutes.

## 2. Code layout

```
src/harness/items.py       Item dataclass + validate_items(); read/write items.jsonl
src/harness/splits.py      make_group_split(), load_split()
src/harness/metrics.py     auroc_ci(), delta_auroc_ci(), spearman_ci()
src/harness/probes.py      fit/evaluate the probe conditions (§6)
src/harness/scorecard.py   ScorecardRow, append_rows(), CLI (python -m harness.scorecard)
src/features/cache.py      FeatureCache
src/features/visual.py     FrameEncoder       (SigLIP)
src/features/audio.py      NonverbalAudioEncoder (emotion2vec+)
src/judge/vlm_judge.py     VLMJudge (local ollama + Gemini anchor)
src/sources/oops.py        Oops! adapter      -> items.jsonl
src/sources/holoassist.py  HoloAssist adapter -> items.jsonl
splits/<dataset>.json      tracked split files
results/scorecard.jsonl    tracked scorecard history
docs/evals/                dated evaluation reports
```

Package names are deliberate: `harness` (not `eval`, which shadows a builtin) and `sources` (not `datasets`, which shadows the Hugging Face library).

## 3. Items: the unit of evaluation

Each source adapter writes `DATA_ROOT/items/<dataset>/items.jsonl`, one JSON object per line:

| field | type | meaning |
|---|---|---|
| `item_id` | str | `"<dataset>:<source-native id>:<index>"`, unique, stable across reruns |
| `dataset` | str | `oops` \| `holoassist` \| `verdict` \| `bad` |
| `group_id` | str | the person/session unit that must never straddle a split (§5) |
| `video_path` | str | absolute path under `DATA_ROOT` |
| `action_window_sec` | [float, float] | what the judge and the action probe see |
| `reaction_window_sec` | [float, float] | what the reaction channels see |
| `label` | int | **1 = good outcome**, 0 = bad outcome (per-dataset definitions in [`02_data_sources.md`](02_data_sources.md)) |
| `label_source` | str | where the label came from, e.g. `"holoassist.mistake_attribute"` |
| `context_text` | str | the text the judge is given. **Must not reveal the outcome** |
| `official_split` | str \| null | the dataset's own split name, if any |
| `meta` | object | anything else (raw annotation ids, transcript snippets) |

`validate_items()` rejects:
- a duplicate `item_id`;
- a window with `start >= end`, or `start < 0`;
- a `label` not in {0, 1};
- an empty `group_id`;
- a `video_path` that does not exist.

## 4. Scorecard row schema

`results/scorecard.jsonl`, one JSON object per line:

```json
{"ts": "2026-10-09T12:00:00Z", "git_sha": "abc1234", "dirty": false,
 "hypothesis": "H1", "dataset": "holoassist", "split": "test",
 "condition": "fusion", "metric": "auroc",
 "value": 0.71, "ci_low": 0.66, "ci_high": 0.76,
 "n_items": 2000, "n_groups": 41, "n_excluded": 12,
 "config_hash": "3f9a0c21b7de", "notes": ""}
```

- **`config_hash`:** the first 12 hex digits of the SHA-256 of the canonical JSON (sorted keys) of the run config. The config includes the encoder ids, judge model, prompt hash, probe hyperparameters and split file hash.
- **`n_excluded`:** items dropped from this condition (missing feature, judge parse failure), always reported.
- **The Δ row:**
  - `condition` = `"fusion_minus_action_best"`, `metric` = `"delta_auroc"`.
  - `action_best` is whichever of `judge` / `action-probe` has the higher test AUROC. That is conservative for our claim.
  - Its CI is computed on paired group-bootstrap resamples (§7), over the test items where **both** scores are non-null. Its `n_items` is that intersection.
- `python -m harness.scorecard` with no arguments prints the **latest** row per `(hypothesis, dataset, split, condition, metric)` as a table, with the newest first. `--history <dataset>` prints every row for one dataset in time order.

## 5. Splits

- **Grouped always.** No `group_id` appears in two splits.
- **If the dataset ships an official split** that is already group-disjoint (verify; never assume), use it: `train` for fitting, the official held-out split as `test`. Otherwise: a grouped 70/30 split, `numpy.random.default_rng(0)`, shuffling sorted unique `group_id`s, the first 70% to train.
- **Written once** to `splits/<dataset>.json`: `{"dataset", "seed", "source": "official"|"grouped_70_30", "train": [item_ids], "test": [item_ids], "sha256": <hash of the sorted item lists>}`.
- `make_group_split` **refuses to overwrite** an existing split file unless called with `force=True`, and a forced rewrite must be named in the commit body.

## 6. Conditions (H1)

| Condition | Input | Model |
|---|---|---|
| `judge` | Action-window frames + `context_text`; **no audio, no reaction window** | Zero-shot VLM (§8). Score = P(good) |
| `action-probe` | SigLIP features of the action window | Logistic regression |
| `react-nonverbal` | emotion2vec+ embedding of the reaction-window audio; **no transcript** | Logistic regression |
| `react-spoke` | One binary feature: did the reactor speak in the reaction window (from annotations) | The feature itself as the score. *HoloAssist only.* It answers "is the emotion signal more than 'they said something'?" |
| `react-full` | Transcript text in the reaction window | TF-IDF (`ngram_range=(1,2)`, `min_df=2`) + logistic regression. *Only where the dataset provides transcripts* |
| `fusion` | `[logit(judge), action-probe features, react-nonverbal features]` | Logistic regression |

**Probe hyperparameters (fixed; never tuned on test):**
- `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")`.
- A `StandardScaler` fit on train only.
- `logit(judge)` clips P to [0.01, 0.99] first.

A condition that cannot run on a dataset is written as **one row with `value: null` and `notes: "not run: <reason>"`**, never silently absent.

## 7. Metrics

- **AUROC** (`sklearn.metrics.roc_auc_score`) on the test split.
- **Group bootstrap:**
  - 1,000 resamples, seeded `numpy.random.default_rng(0)`. Each resample draws `n_groups` groups with replacement and takes all of their items.
  - A resample containing one class only is redrawn. If more than 10,000 total draws are needed, raise.
  - The CI is the 2.5th and 97.5th percentiles.
- **Δ CI:** the same resampled groups are used for both conditions in each draw (paired).
- **Spearman** (verdict datasets with raw scores only), with the same group bootstrap.
- **False-positive rate** on the archived Ego4D steady-state set *(Wave B+, not Wave A)*: the share of items with a fusion P(good) outside [0.2, 0.8].

## 8. VLM judge

**Local judge** (always run):
- **Model:** `models_config.get_model("vlm_judge")` (`qwen2.5vl:7b` on this host), via `src/shared/vlm_client.ollama_chat`.
- **Frames:** 4 frames at `t_k = s + (k + 0.5)·(e − s)/4`, `k = 0..3`, over the action window `[s, e]`. Each is resized so its longer side is 768 px, and JPEG-encoded at quality 90. **No audio is ever sent.**
- **Options:** `{"temperature": 0, "num_ctx": 8192, "num_predict": 16}`, `timeout=180`.
- **Prompt, verbatim** (`{n}` = 4):

```
You are shown {n} frames, in time order, from a video of a person doing something.
Context: {context_text}
Question: {question}
Answer with a single integer from 0 to 100: your probability (in percent) that the answer is YES. Output only the integer.
```

- **`{question}` per dataset, verbatim:**
  - `holoassist`: `Did the person perform this step correctly, without a mistake?`
  - `oops`: `Is everything still going as the person intended at this point, with no accident or failure yet?`
  - `verdict` (Wave B): `Will the person rate this item highly (8 or more out of 10)?`
- **Parse:** the first match of `\b(100|[1-9]?\d)\b`, divided by 100. If no match, retry up to 2 more times with `temperature` 0.3. If all 3 attempts fail, `judge_prob = null`; the item is excluded from `judge` and `fusion` and counted in `n_excluded`.
- **Cache:** `DATA_ROOT/judge/<dataset>/<model_tag>/<prompt_hash>.jsonl`, one line per item: `{item_id, judge_prob, raw, attempts, elapsed_ms}`. The `prompt_hash` is the first 12 hex digits of the SHA-256 of `template + question + "frames=4;side=768;q=90"`. A cached item is never re-queried.

**Frontier anchor** (test split only):
- **Model:** Gemini `gemini-3.6-flash` (the model used for v0 pre-seeding) via `google-genai`, with `GOOGLE_API_KEY` from `.env`.
- **Inputs:** the same 4 frames, prompt and parse rule.
- **Cap:** at most 2,000 items per dataset; if the test split is larger, sample with `default_rng(0)`.
- **Errors:** on HTTP 429 or 5xx, sleep 30 s and retry, up to 5 tries; then `null`.
- **Rows:** written as condition `judge-frontier`. It does not enter `fusion` (it has no train-split scores) but **does** enter `action_best` for the Δ row.

## 9. Encoders

| Encoder id | What | Exact spec |
|---|---|---|
| `siglip-b16-224` | Action-window visual | `transformers` `SiglipModel.from_pretrained("google/siglip-base-patch16-224").get_image_features`. Frames at `t = s, s+1, s+2, …` strictly below `e`, plus `e` itself (so at least 2); each L2-normalized, then mean-pooled → 768-d float32. Device `mps` if available, else `cpu` |
| `e2v-plus-large` | Reaction-window non-verbal audio | `funasr` `AutoModel(model="iic/emotion2vec_plus_large")`, `generate(<16 kHz mono wav>, granularity="utterance", extract_embedding=True)` → 1024-d float32. Audio is cut with ffmpeg (`-ac 1 -ar 16000`). **The embedding is used as features; its emotion-category outputs are never used** |

**Cache:** `DATA_ROOT/features/<dataset>/<encoder_id>/<sanitized item_id>.npy`, plus `index.jsonl` (`item_id`, `shape`, `window`, `elapsed_ms`). Extraction is resumable: existing files are skipped unless `--force`. A failure is logged to `errors.jsonl` with its traceback, and never written as zeros.

## 10. Cadence

| Trigger | What reruns |
|---|---|
| Any change to probes or fusion | Probes + scorecard (minutes) |
| New items | Features and judge for the new items only, then the scorecard |
| Judge prompt or model change | Judge (expensive, new `prompt_hash`), then the scorecard |

## 11. Guardrails against fooling ourselves

1. **No H1 result without both action-only controls** (`judge` and `action-probe`) and the Δ row.
2. **Leakage conditions are reported separately.** `react-nonverbal` is the headline reaction number; `react-full` is never reported alone.
3. **Grouped CIs only.**
4. **H2 targets are never training data** (BAD, ERR@HRI; HoloAssist when used as an H2 target).
5. **Kill and pass criteria are written before results** ([`00_thesis.md`](00_thesis.md)). They change only through a dated decision-log entry.
6. **`context_text` never reveals the outcome.** Each adapter's choice is stated in [`02_data_sources.md`](02_data_sources.md).
