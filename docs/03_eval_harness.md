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
src/shared/memguard.py     memory admission, heavy lock, between-item watchdog (§12)
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

## 12. Memory guard (added October 10, 2026)

**Why.** On October 9, 2026 at 19:50–19:52 the 64 GB Mac ran out of memory. The macOS jetsam reports (`/Library/Logs/DiagnosticReports/JetsamEvent-2026-10-09-19505*.ips`) show:
- two `python3.12` processes at 26.6–27.2 GB each, which were the `animated_infographics` agent running four image-generation gates at once (its `docs/design_system_architecture.md` §11);
- Ollama's `llama-server` at 10.5 GB;
- this project's A5/A6 slow tests, which loaded SigLIP and emotion2vec and called the Ollama judge in the same window.

macOS killed its own services for lack of compressor space. Nothing in either project checked memory. **Other programs start and stop at will** (the other project, browsers, editors, other agents), so the memory free when a run starts says little about ten minutes later.

**Principles** (shared with `animated_infographics`, so the two projects cooperate):
1. **Admission, not hope.** Check the memory available *now* before loading any model.
2. **One heavy step at a time on the whole machine.** Take the **same** machine-wide lock file as `animated_infographics`.
3. **Re-check between items.** Back off when someone else needs the memory.
4. **Free our own memory first; never touch another program's.** We only ever unload our own Ollama model (`get_model("vlm_judge")`). Never `gemma4:26b` or any other.
5. **Fail clean.** A job that cannot get memory waits, then exits **75** (`EX_TEMPFAIL`) with its progress saved, and the supervisor relaunches it later. The kernel never gets to kill one of ours.

**Available memory and pressure** (the same formula as `animated_infographics`):
- `available = hw.memsize × kern.memorystatus_level / 100`. That is the kernel's free percentage, the one jetsam acts on.
- `pressure = kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning, 4 critical.
- Both are read with `sysctl -n`, through one function `read_memory() -> (available_bytes, pressure)` that tests replace with a fake.
- **`FLOOR = 8 GB`** must remain available after any admission.

**Heavy steps and their declared peaks** (`src/shared/memguard.py`, `HEAVY_STEPS`). Declared peak = measured × 1.15, rounded up to a whole GB:

| Step | Where | Measured (designer, October 10, 2026, M4 Max) | Declared peak |
|---|---|---|---|
| `sr_siglip` | `FrameEncoder` model load | 1.56 GB process footprint after load + one window (MPS 1.03 GB) | **2 GB** |
| `sr_e2v` | `NonverbalAudioEncoder` model load | 4.76 GB peak RSS during load + one window; 3.0 GB steady | **6 GB** |
| `sr_judge_load` | first `OllamaJudge` call while `qwen2.5vl:7b` is not loaded | `llama-server` 7.72 GB RSS; `/api/ps` 6.8 GiB at `num_ctx` 8192 | **9 GB** |

- After `release()`, the process measured 1.10 GB, against 0.14 GB before loading.
- The implementing agent **re-measures** each step with `/usr/bin/time -l` ("peak memory footprint") on a 50-item run. If measured × 1.15 rounds up above a declared peak, the constant and this table are raised in the same commit.

**The lock (shared with `animated_infographics`):**
- **File:** `fcntl.flock(LOCK_EX)` on `<lock dir>/heavy.lock`. The lock dir is `$INFOGRAPHICS_LOCK_DIR` if set, else `~/.cache/animated_infographics/locks`, which is that project's default. **The path must equal theirs**, or the two projects stop seeing each other.
- **Contents:** while holding it, write one line, `<step> pid <pid>`, the format their `get_heavy_lock_holder()` parses.
- The OS releases a `flock` when its process dies, so there is never a stale lock.

**Admission:** `with guard("<step>"):` wraps every model load: `FrameEncoder._ensure_loaded`, `NonverbalAudioEncoder._ensure_loaded`, and `OllamaJudge` before a call while its model is absent from `GET /api/ps`. Every code path that loads a model (CLIs, the A7/A8 runs, slow tests) is therefore guarded, with no caller remembering to.
1. Take the heavy lock, polling non-blocking every 0.5 s.
2. Admit when `available − peak(step) ≥ FLOOR`.
3. If not admitted, and the step is not `sr_judge_load`, and our judge model is loaded: unload **only** it (`POST /api/generate {"model": <get_model("vlm_judge")>, "keep_alive": 0}`), wait up to 30 s for `/api/ps` to drop it, then check again.
4. Otherwise wait, checking every 5 s. At most every 30 s, log `memory guard: waiting for <step>: need <peak> GB + floor 8 GB, available <a> GB, pressure <p>, heavy lock <free|held by pid N>`.
5. After `SR_MEM_WAIT_S` (default **1800 s**) in total, raise `MemoryDeferred`. CLIs turn that into **exit 75**.
6. Hold the lock only through the load and the first item (the peak), then release it. A loaded, idle model is ordinary used memory that the other project's admission already accounts for.

**Between items** (`memguard.check(release, reload)`, called before every item in `features.extract` and the judge loop):
- **Critical** (`pressure == 4` or `available < FLOOR / 2`):
  - call `release()`;
  - save progress;
  - log `action=stop`;
  - raise `MemoryDeferred`, so the CLI exits **75**.
- **Warning** (`pressure == 2` or `available < FLOOR`):
  - call `release()`;
  - re-enter `guard()` (which waits for room, with natural hysteresis: re-admission needs `peak + 8 GB` free);
  - call `reload()`;
  - continue.
- **`release()`** drops the model references, runs `gc.collect()` and `torch.mps.empty_cache()`, and for the judge unloads only our own Ollama model. The process must return within **1.5 GB** of its pre-load footprint.
- **The judge also releases when its run ends,** unloading only our model.

**The supervisor** (`tools/run_supervised.sh`):
- Exit **75** means "deferred by memory guard". Sleep `SR_MEMWAIT_SLEEP_S` (default **600 s**), relaunch, and **do not** count the attempt toward the no-progress abort.
- After `SR_MAX_MEM_DEFERRALS` (default **72**, about 12 h) consecutive deferrals, abort with `memory guard: deferred <n> times; giving up`.
- Any other non-zero exit keeps today's behavior.

**Tests and the battery:**
- `tests/conftest.py` turns an uncaught `MemoryDeferred` into `pytest.exit("memory guard: deferred: <message>", returncode=75)`. A guarded slow test **never** skips silently and never passes.
- `scripts/battery.sh` prints `G<n> <name>: exit 75 (deferred by memory guard; not run)` for that code. The battery's exit is the maximum code as before, so a deferral is never read as green.

**Status:** `PYTHONPATH=src ./venv/bin/python -m shared.memguard --status` prints:
- the available GB and pressure level;
- the heavy-lock holder's pid and step, if any;
- the Ollama models loaded, with their sizes, marking ours.

Agents run it before any long job.

**A test-only override:** `SR_MEMGUARD_FAKE_AVAILABLE_GB=<n>` makes `read_memory()` report `n` GB at pressure 1, so the deferral path can be drilled without consuming memory. While it is set, every read logs `memory guard: FAKE MEMORY READING (<n> GB)`. No run that counts toward results may set it.

**The log:** every admission, wait summary, pause, stop and unload appends one line to `DATA_ROOT/runs/memguard.log`: `ts=<iso> pid=<p> step=<s> action=<admit|pause|resume|stop|unload|deferred> waited_ms=<w> available_gb=<a> pressure=<p>`.
