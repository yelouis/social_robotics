# Agent Execution Guide: Active Build: Wave A, remaining items (A6c → A7b → A8 → A9), October 10, 2026

**You are an engineering agent with no memory of this project.**
- **What happened:** on October 8, 2026 the project was reoriented. The v0 pipeline (six hand-built affect layers over Ego4D, plus a human-rated benchmark) is archived at git tag `v0-saf-final` and removed from the tree, because it could not be validated ([`LESSONS_v0.md`](LESSONS_v0.md)).
- **What exists:** Wave A items **A1–A7 and A6b have landed** on `main` (`651039b` … `6d72492`, October 9–10):
  - the harness (items, splits, metrics, scorecard);
  - the encoders and the feature cache;
  - the VLM judge;
  - the memory guard;
  - the first H1 numbers, on Oops!.

  The designer verified them on October 10 by reading every item's code against its contract and re-running the battery (§1).
- **What you build:** the rest of Wave A, in the §2 order:
  - **A6c:** fix three memory-guard defects found in verification;
  - **A7b:** correct the Oops! report, and give the frontier judge the same parse retries as the local one;
  - **A8:** finish HoloAssist, resuming an uncommitted draft, with corrections;
  - **A9:** close out Wave A.

**Read before anything else, in this order:**
1. [`00_thesis.md`](00_thesis.md): the thesis, H1–H3, the pass and kill criteria, and the decision log (entries 15–17 are this verification).
2. [`03_eval_harness.md`](03_eval_harness.md): **the contract** for every path, schema, constant and prompt. §4–§6, §8 and §12 were amended on October 10; each amendment is marked *(added/clarified October 10, 2026)*.
3. [`02_data_sources.md`](02_data_sources.md): Oops! "As built" and HoloAssist "Item construction: pinned details" (both October 10).
4. [`ongoing_general_errors.md`](ongoing_general_errors.md): §1 (the verification summary), **Issue 5**, and §2 (the lessons, especially L10 and L11).

**The maintainer's words:**
- *"The fundamental core of this project is to use human emotion as a policy for robot learning in addition to the traditional RL stack for robots. We want to add this additional reward signal and investigate if this additional reward signal helped the robot learn a task better and perhaps allow robots to train on much more data like vlogs or any POV youtube video."*
- *"Ideally we should avoid human in the loop ratings because that is not scalable and we want to be able to somehow measure whether our thesis is working often."*
- *"How is this any different from asking an LLM model what is socially appropriate…?"* This is why every H1 result is measured against action-only controls, and why Issue 5 matters.
- *"No need to open up a new branch, just push to the repo."*
- *"During another agent's last implementation and testing it seems like we ran out of memory. Write guards so that we don't run out of memory. Assume that other program can start and stop which will take from the available memory."* (October 10, 2026)

**Status:** **Active Build: Wave A, remaining items.** Next is **A6c**.
- **One maintainer decision is open: Issue 5** (the frontier judge). It blocks no item. A8 and A9 write honest `not run`/`partial` rows if it is not selected, and A9 has a conditional step if it is.
- **Issue 1** gates only Wave B, which is not in this guide.

**Every path, number, field name and literal string in this guide and in `03_eval_harness.md` is a decision, not a suggestion. Implement as written; do not substitute your own.**

---

## 0. Standing constraints (apply to every item)

1. **The battery is the regression bar.**
   - Run `scripts/battery.sh` bare after every item (`--slow` when the item touches models), and update §1.4.
   - Read every exit code bare; never pipe a gate through something that swallows its code.
2. **Environment.**
   - Run from the repo root with `./venv/bin/python` (Python **3.9.6**) and `PYTHONPATH=src`.
   - **Python 3.9 syntax only:** no `match`; no `X | Y` unions evaluated at runtime (use `typing.Optional`/`Union`, or `from __future__ import annotations`).
   - **Before any run that loads models or touches video:**
     ```bash
     export HF_HOME="/Volumes/Extreme SSD/huggingface_cache"
     export TMPDIR="/Volumes/Extreme SSD/tmp"
     export SR_NO_MODEL_BANNER=1
     ```
3. **Dependencies are frozen.**
   - **Install no new pip packages.** Everything is in `venv` (§1.1).
   - **No new model downloads.** `google/siglip-base-patch16-224` and `iic/emotion2vec_plus_large` are cached.
   - **Pull no Ollama models.**
4. **Data placement and disk.**
   - All data lives under `DATA_ROOT` (`src/config.py`, default `/Volumes/Extreme SSD/social_robotics`). **No video or audio on the internal disk, ever.**
   - The free space *after* any download and extraction must stay **≥ 50 GiB**.
5. **Authorized downloads:** only the HoloAssist pitch-shifted videos, which are **already downloading** (§1.6). **No other download of any kind. No YouTube or web video. Never circumvent a login wall, CAPTCHA or bot check.**
6. **Never delete anything you did not create.** You may delete a downloaded archive only after you have verified its extraction.
7. **Long runs (> 30 min) run detached:**
   ```bash
   ./venv/bin/python tools/daemonize.py <log> bash tools/run_supervised.sh <progress.json> <runner…>
   ```
   - Then confirm `PPID 1` (`tools/README.md`). Never use your own background-task feature: it is reaped after ~1–2 h.
   - Every long runner is resumable and maintains `<out_dir>/progress.json`, a JSON list of finished `item_id`s, rewritten atomically.
8. **Determinism.** Seeds: `numpy.random.default_rng(0)` and `torch.manual_seed(0)`. Iterate inputs in sorted `item_id` order.
9. **No outcome leakage.**
   - The judge never receives audio, the reaction window outside the action window, labels, Oops! descriptions, or HoloAssist mistake/purpose labels.
   - Probes never see test labels at fit time.
   - **Nothing is tuned on the test split.**
10. **The scorecard is append-only.** Never edit or delete a row in `results/scorecard.jsonl`. A wrong row is superseded by a new row whose `notes` say why.
11. **Honest nulls (L3).**
    - A failure is `null` plus a reason, never a zero and never a sentinel such as `-100`.
    - **A "not run" reason is derived from the run's evidence, never hard-coded** (`03_eval_harness.md` §6).
12. **Every stage logs its counts (L2).** Every CLI ends by printing one line: `items_in=<a> items_out=<b> excluded=<c> elapsed_s=<t>`. Exclusions are broken down by reason.
13. **Reports are generated from the rows, clause by clause (L10).** A sentence that states a result is computed from the scorecard row it describes, and a prediction check gives every clause of the prediction its own measured line. Template prose that does not depend on the data is not a description.
14. **Commits.**
    - One item = one Conventional Commit on `main`, scope = item id (`fix(a6c): …`, `fix(a7b): …`, `feat(a8): …`).
    - The body states the WHY, the red run and the green run.
    - Push after every item with **`/usr/bin/git push origin main`**. **No branches, no PRs. Never amend a pushed commit.**
    - **Stage by explicit path only** (`/usr/bin/git add <path> <path> …`). Never `git add -A`, `git add .` or `git commit -a`. Until A8's commit, the tree holds the uncommitted A8 draft (§1.3), and it must not leak into A6c's or A7b's commit.
15. **Record the resolution in the same commit:** one line under **"Wave A"** in `ongoing_general_errors.md` §3: `A<n> — <title> — git log --grep "(a<n>)" — <measured result>`.
16. **When this guide and a contract doc disagree, STOP and file it** in `ongoing_general_errors.md` as a new issue (the next number is **Issue 6**), with options.
17. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer.
18. **Memory.** This Mac is shared with other programs, including the `animated_infographics` agent, browsers and editors, that start and stop at will.
    - Every model load goes through `shared.memguard.guard()`, and every long loop calls `memguard.check()` between items (`03_eval_harness.md` §12).
    - **Never run two of this project's model-loading jobs at once.**
    - Run `PYTHONPATH=src ./venv/bin/python -m shared.memguard --status` before any long job, and record its output in the run log.
    - **A memory deferral (exit 75) is never a pass.** Never raise `FLOOR`, a declared peak or a wait limit to make a run go through.
    - **Never stop, signal or unload anything this project did not start.** `gemma4:26b` in Ollama is another program's.
19. **Running jobs are not yours to stop** (§1.6). Leave the HoloAssist download and the Gemini job running.
    - **Never edit `tools/run_supervised.sh` in place while a supervisor is running it.** Bash reads a script from its open file as it executes. Write the new version to a temp file and `mv` it over the original: the rename gives a new inode, and the running shells keep the old one.

---

## 1. Verified baseline (October 10, 2026; designer, this session)

### 1.1 Environment

- **Host:** Mac Studio M4 Max, 64 GB; macOS (Darwin 25.6); ffmpeg 8.1 (`/opt/homebrew/bin/ffmpeg`); `/usr/bin/footprint` available.
- **Ollama:** `qwen2.5vl:7b` (ours: `get_model("vlm_judge")`). Also installed: `gemma4:26b`, `gemma4:latest`, `glm4:latest`, `moondream:latest`, `tinyllama:latest`.
- **`venv`, Python 3.9.6:**
  - ML: torch 2.8.0 · torchaudio 2.8.0 · transformers 4.46.3 · funasr 1.3.1 · scikit-learn 1.6.1 · scipy 1.13.1;
  - data: numpy 2.0.2 · pandas 2.3.3 · pyarrow 21.0.0 · soundfile;
  - clients: google-genai 1.47.0 · httpx 0.28.1 · huggingface_hub 0.36.2;
  - tooling: ruff 0.15.11 · pytest 8.4.2 · psutil.
- **`.env`** defines `GOOGLE_API_KEY` (**free tier: 20 requests/day for `gemini-3.6-flash`**, Issue 5) and `HF_TOKEN`.

### 1.2 Storage (Extreme SSD)

- 1.8 TiB total, **1.4 TiB free** (October 10, 11:09).
- **Oops!:** `raw/oops/oops_dataset/` holds 29,940 extracted videos. The archive was deleted after verification (`raw/oops/DOWNLOAD.json`).
- **HoloAssist:**
  - `raw/holoassist/labels/`: hashes re-verified October 10 against `DOWNLOAD.json`;
  - `raw/holoassist/video_pitch_shifted.tar`: downloading (§1.6).
- **Still in place; do not touch:** `huggingface_cache/`, `ego4d_data/`, `saf_env/`, `social_robotics/full_run_2026_06_18/`, `social_robotics/bench_v0/`, `DELETED_2026-10-08.json`.

### 1.3 Repository and working tree at hand-off

- `HEAD` = `6d72492` (`feat(a7)`), pushed. v0 is at tag `v0-saf-final`.
- **The uncommitted A8 draft, written by a previous agent.** It is not yours to discard; A8 resumes it:

  | State | Path | What it is |
  |---|---|---|
  | modified | `src/sources/holoassist.py` | `--build`: the item builder, group split and balanced sampling (+294 lines) |
  | modified | `src/harness/probes.py` | HoloAssist `react-spoke` and `react-full` branches |
  | modified | `src/harness/splits.py` | `make_group_split(…, source=, seed=)` |
  | modified | `tests/test_probes.py` | `test_run_probes_holoassist_synthetic` |
  | modified | `tests/test_memguard.py` | one line patching `is_our_judge_loaded` in test (b). **This hunk belongs to A6c**; commit it there |
  | new | `splits/holoassist.json` | 3,000 train / 2,000 test. Its item ids are in the **wrong format**, so A8 regenerates it |
  | new | `tests/test_sources_holoassist.py` | 6 tests; 3 read SSD data in the fast suite |
  | new | `tools/download_holoassist.py` | the running download (§1.6) |
  | new | `tools/check_audio_presence.py` | A8 step 2 (needs corrections) |
  | new | `tools/report_holoassist.py` | A8 report generator (not yet run) |

- **If any draft file's modification time is later than 2026-10-10 11:30,** another session may still be working in this tree. **STOP and ask the maintainer** before editing.

### 1.4 Gates (run bare October 10, 2026; the regression bar)

| # | Gate | Command | `HEAD` (clean export) | Working tree (with the A8 draft) |
|---|---|---|---|---|
| G1 | Lint | `./venv/bin/ruff check src tests tools` | exit 0 · clean | exit 0 · clean |
| G2 | Fast tests | `SR_NO_MODEL_BANNER=1 ./venv/bin/python -m pytest -q -m "not slow" tests/` | exit 0 · **41 passed** | exit 0 · **48 passed** |
| G3 | Harness self-test | `PYTHONPATH=src ./venv/bin/python -m harness.scorecard --selftest` | exit 0 · 4 PASS | exit 0 · 4 PASS |
| G4 | Slow tests | `… -m pytest -q -m slow tests/` | not run on the export | exit 0 · **5 passed** (82 s) |

- `memguard --status` before G4: 40.3 GB available, pressure 1, heavy lock free, `gemma4:26b` loaded (not ours).
- **What the battery does not catch** (it is green, and these were found by reading the code; they are A6c's and A7b's gaps):
  - the nested-guard lockout;
  - the fail-open memory read;
  - the supervisor's attempt accounting;
  - the Oops! report's errors.

### 1.5 Data, caches and results on the SSD

| What | Where | State |
|---|---|---|
| Oops! items | `items/oops/items.jsonl`, `stats.json` | 2,710 items (1,355 clips); 1,638 train / 1,072 test; 90 test groups |
| Oops! split | `splits/oops.json` (tracked) | `source: official`, hash-verified |
| Oops! features | `features/oops/{siglip-b16-224,e2v-plus-large}/` | 2,708 `.npy` each. 1 undecodable clip (2 items, train) is in `errors.jsonl` |
| Oops! local judge | `judge/oops/qwen2.5vl_7b/2409b2876016.jsonl` | 2,708 answers, **0 null**. 2 ffmpeg errors (the same clip) in `errors.jsonl` |
| Oops! frontier judge | `judge/oops/gemini-3.6-flash/2409b2876016.jsonl` | 11 of 1,072 scored. `errors.jsonl` holds 429s (Issue 5) |
| Oops! rows | `results/scorecard.jsonl` | 11 rows, `2026-10-10T17:33:22Z`–`…26Z`. Judge **0.472** [0.440, 0.498]; action-probe 0.779; react-nonverbal 0.711; fusion 0.793; Δ +0.015 [−0.002, 0.029]; shuffled controls 0.510 / 0.468 / 0.513 |
| HoloAssist items (draft) | `items/holoassist/items.jsonl`, `stats.json` | 5,000 items (2,500 per class), built with the draft. **Rebuilt in A8** (item-id format) |
| Memory-guard log | `runs/memguard.log` | No real `pause` or `stop` yet (the only `stop`/`deferred` lines are fake-reading drills). No jetsam report since October 9 |

### 1.6 Processes running at hand-off (October 10, 11:09)

| PID | What | Note |
|---|---|---|
| 40612 (PPID 1) → 40623 → curl 40626 | `run_supervised.sh raw/holoassist/progress.json ./venv/bin/python tools/download_holoassist.py` | 62.1 of 184.2 GB at 11:09 (~37 MB/s). It then hashes, extracts into `raw/holoassist/videos/`, counts `.mp4` (deleting the archive only if ≥ 1,000) and sets `progress.json` to `["verified"]` |
| 37143 (PPID 1) → 37154 | `run_supervised.sh …/judge/oops/gemini-3.6-flash/progress.json ./venv/bin/python -m judge.vlm_judge --dataset oops --split test --backend gemini --limit 2000` | Retrying 429s. It is futile until Issue 5 is decided, but harmless (network only), and its cache is valid. **Leave it** |
| 38143 | Ollama `llama-server` with `gemma4:26b` | Another program's. Never unload it |

Check with: `ps -axo pid,ppid,etime,command | grep -E "run_supervised|download_holoassist|vlm_judge" | grep -v grep`.

---

## 2. Execution order

| # | Item | Why this position |
|---|---|---|
| A6c | Memory-guard fixes: a re-entrant guard, a fail-closed memory read, supervisor accounting, and no live HTTP in fast tests | **A8's feature extraction is the longest model run of Wave A** (5,000 items × 2 encoders, then 5,000 judge calls at ~9 s each), and it runs under contention. Today the first `warning` reading would make extraction hold the machine-wide lock for 30 min and then exit 75. It lands while the download finishes, and touches no A8 draft file except `tests/test_memguard.py`, whose draft hunk belongs here |
| A7b | Oops! report corrections; frontier parse parity | A8's report must not copy A7's defects, so the corrected generator is A8's template. The Gemini fix must land before any Gemini run on HoloAssist (Issue 5 = A). It touches no A8 draft file |
| A8 | HoloAssist: end-to-end H1, resuming the draft with corrections | Needs A6c (extraction), A7b (the report pattern) and the download marked `["verified"]` (§1.6) |
| A9 | Re-measure; close out Wave A | Measures the finished system, and writes the summary the designer uses to spec Wave B |

---

## 3. The items

### A6c: Memory-guard fixes

**What this means for the maintainer:** today, the first time memory gets tight during a feature run, our job grabs the lock it shares with the other project and sits on it for 30 minutes doing nothing. That also stops the other project from starting anything heavy. If macOS ever fails to report memory, the guard assumes 64 GB is free.

**The gap** (read at `6d72492`; contract `03_eval_harness.md` §12, amended October 10):
1. **A nested `guard()` locks itself out.**
   - `memguard.check()` (`src/shared/memguard.py:298`) runs `with guard(step): reload()`, and the encoders' `reload` is `_ensure_loaded`, which enters `guard()` again (`src/features/visual.py:58`, `src/features/audio.py:45`).
   - The inner call's `get_heavy_lock_holder()` (`memguard.py:202`) sees the lock held by its own pid and waits `SR_MEM_WAIT_S`.
   - **Reproduced by the designer** with fakes: 40 GB available, pressure 1, a 4 s wait: `deferred ['release'] 4.0s`.
   - Test (e) missed it because it stubs `reload` with a function that never takes the lock (L11).
2. **`read_memory()` fails open:** `memguard.py:77` returns `64 GB, pressure 1` on any error.
3. **Memory deferrals consume the supervisor's attempts:** `tools/run_supervised.sh:70` (`for attempt in $(seq 1 "$MAX_ATTEMPTS")`) also counts 75 exits. The `SR_MAX_MEM_DEFERRALS=72` rule is therefore unreachable: the run ends at 50 deferrals with exit 1 and the wrong message.
4. **Fast tests reach the live Ollama server.**
   - Test (c) (`tests/test_memguard.py`, `guard("sr_e2v")` with 5 GB) calls the real `/api/ps`, and if `qwen2.5vl:7b` is loaded it really unloads it.
   - The draft hunk on test (b) patches one instance; the rest remain.
5. **The slow release test measures RSS (`psutil`), not the footprint** that §12 names (`footprint -p <pid>`). On Apple silicon, Metal allocations are in the footprint and not reliably in RSS.

**Implementation:**
1. **Make `guard()` re-entrant** (`src/shared/memguard.py`):
   - Add a module-level `_HELD_DEPTH: int = 0`.
   - **Outer path** (`_HELD_DEPTH == 0`): unchanged, except that you set `_HELD_DEPTH += 1` immediately after `flock` succeeds and the memory re-check admits. Decrement it in `finally` **before** unlocking.
   - **Nested path** (`_HELD_DEPTH > 0`):
     - Never open, `flock`, truncate or write the lock file.
     - Loop until the timeout:
       - `available, pressure = read_memory()`;
       - if `available − peak ≥ FLOOR`: `log_event(step, "admit", …)`, `_HELD_DEPTH += 1`, `yield`, then decrement in `finally`;
       - else apply step 3 (unload only our judge, if `step != "sr_judge_load"`), print the §12 waiting line at most every 30 s with `heavy lock held by pid <own pid>`, and sleep 5 s.
     - On timeout, `log_event(step, "deferred", …)` and raise `MemoryDeferred`, exactly as the outer path does.
   - Factor the shared memory-wait into one private helper. Do not duplicate it.
2. **Make `read_memory()` fail closed:** on any exception, or on fewer than 3 parsed fields, write `memory guard: READ FAILED (<error>)` to stderr and return `(0, 4)`. The fake-reading override is unchanged.
3. **Supervisor accounting** (`tools/run_supervised.sh`; **write to a temp file, then `mv`**, per §0.19):
   - Replace the `for` loop with `attempt=0; while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do …`.
   - Increment `attempt` only for exits other than 75.
   - The 75 branch is unchanged: it logs, checks `SR_MAX_MEM_DEFERRALS`, sleeps and continues. At the limit it logs `ABORT: memory guard: deferred <n> times; giving up` and exits **75**.
   - Update the header comment and `tools/README.md`: *"memory deferrals do not count toward SR_SUPERVISE_MAX_ATTEMPTS"*.
4. **`tests/test_memguard.py`:**
   - **The autouse fixture `_no_live_http`:** unless the test carries the `slow` marker:
     - `monkeypatch` `httpx.get` and `httpx.post` with a recorder that appends the URL to a list and raises `httpx.ConnectError("blocked in fast test")`;
     - **at teardown, assert the list is empty**, with the message `live HTTP in a fast memguard test: <urls>`.
     - The assertion must live at teardown because `is_our_judge_loaded()` and `unload_own_judge()` swallow every exception, so an exception raised inside the call would pass silently.
     - Tests (d) and any others that need HTTP install their own fakes after it. Keep the draft hunk on test (b).
     - **Red first:** add the fixture alone. Test (c) as written at `6d72492` must go red at teardown, because it reaches `/api/ps`; record it. Then give (c) its own `is_our_judge_loaded` fake → green. That red run is also the fixture's falsification.
   - **New test (i), the nested guard:**
     - Set `SR_MEM_WAIT_S=2` and `INFOGRAPHICS_LOCK_DIR=<tmp>`, and use a fake reader that returns `(7 GB, 2)` once and then `(40 GB, 1)`.
     - Call `check("sr_siglip", release, reload)`, where `reload` is `lambda: <enter guard("sr_siglip") and record "reload-admitted">`.
     - **Assert:** it returns in **< 1.5 s**; the calls are `["release", "reload-admitted"]`; and afterwards a child process can take `flock(LOCK_EX | LOCK_NB)` on `<tmp>/heavy.lock`.
   - **New test (j), the real callables (L11):**
     - Build `FrameEncoder()` with `features.visual.AutoProcessor.from_pretrained` and `features.visual.SiglipModel.from_pretrained` monkeypatched to return a dummy, whose `.to()` returns itself and which has `.eval()`.
     - Use the same fake reader as (i), and call `check("sr_siglip", enc.release, enc._ensure_loaded)`.
     - **Assert:** it returns in < 1.5 s, and `enc._model` is the dummy.
   - **New test (k), fail-closed:** with `subprocess.check_output` monkeypatched to raise `OSError`, `read_memory()` returns `(0, 4)`, and stderr contains `memory guard: READ FAILED`.
   - **New supervisor test:**
     - With `SR_SUPERVISE_MAX_ATTEMPTS=3`, `SR_MEMWAIT_SLEEP_S=0` and `SR_MAX_MEM_DEFERRALS=100`, a runner that exits 75 **five** times and then 0 must finish with exit 0 and `DONE`.
     - With `SR_MAX_MEM_DEFERRALS=4`, a runner that always exits 75 must exit **75** with `deferred 4 times; giving up` in stderr.
   - **The slow release test:** measure with `footprint -p <pid>`. Parse the `Footprint: <n> <KB|MB|GB>` line; the bar is still < 1.5 GB above pre-load.

**Validation:**
- **Red first:** run new tests (i), (j), (k) and the supervisor test against the current code. (i) and (j) **must fail** (timeout → `MemoryDeferred`), (k) must fail (it returns 64 GB), and the supervisor test must fail (exit 1 at 3 attempts). Record all four in the commit body.
- **Green:** all four pass.
- **Falsify:** delete the `_HELD_DEPTH > 0` branch → (i) and (j) go red → restore → green.
- **Drills (unchanged, they must still hold):**
  - `SR_MEMGUARD_FAKE_AVAILABLE_GB=1 SR_MEM_WAIT_S=5 python -m features.extract --dataset test_ds --encoder e2v-plus-large --limit 2` exits **75** in under 15 s, and `memguard.log` gains `action=deferred`.
  - `SR_MEMGUARD_FAKE_AVAILABLE_GB=1 scripts/battery.sh --slow` prints `G4 slow: exit 75 (deferred by memory guard; not run)`.
- **No live HTTP:** the fast suite is green with the fixture in place, and the fixture's falsification (above) is recorded red.
- G1–G4 green.

**Blast radius:** `src/shared/memguard.py`, `tools/run_supervised.sh`, `tools/README.md`, `tests/test_memguard.py`; `ongoing_general_errors.md` §3; this guide's §1.4. **Do not** touch the A8 draft files.

---

### A7b: Oops! report corrections and frontier-judge parse parity

**What this means for the maintainer:** today the Oops! report says the prediction came true, when half of it (a strong judge) did not. Its judge-error counts are mislabeled, and its "example descriptions" are the same two sentences for every clip. Anyone reading only the report would be misled.

**The gap** (read at `6d72492`):
- **`tools/report_oops.py:119`** writes "The prediction **holds**" from the Δ row alone. The judge's AUROC (0.472 [0.440, 0.498]), the first clause of the prediction, is never checked.
- **`tools/report_oops.py:53` and `:91`** glob `*.jsonl`, which includes `errors.jsonl`, and count every error row as a "parse failure". The real numbers:
  - local judge: **0 parse failures** and 2 ffmpeg errors (one undecodable clip);
  - Gemini: **11 scored**; the "202 parse failures" are HTTP 429 rows.
- **`tools/report_oops.py:191`:** no wall-clock times, although A7 required them.
- **`tools/report_oops.py:211` and `:227`:** each example's "description" is one of two fixed sentences.
- **The `react-nonverbal` caveat is absent:** the `post` audio contains the failure's own sound (`02_data_sources.md` → Oops! → "As built").
- **`src/judge/vlm_judge.py:253–255`:** `GeminiJudge` returns after one unparsable answer. §8 gives it the local parse rule: up to 2 more attempts at `temperature` 0.3.

**Implementation:**
1. **`tools/report_oops.py`:**
   - **(a) Judge diagnostics.**
     - Read answers only from `judge/oops/<model_tag>/<prompt_hash>.jsonl`. Parse failures = rows with `judge_prob: null`.
     - Read `errors.jsonl` separately. Count distinct `item_id`s and classify each by its `error` text: `429` if it contains `429`; `5xx` if it matches `\b5\d\d\b`; `ffmpeg` if it contains `ffmpeg`; else `other`.
     - For Gemini, also print `scored <k> of <n> test items`.
   - **(b) The prediction check:** a function `prediction_check(rows: Dict[str, dict]) -> List[str]` that returns exactly these three lines, filled from the rows:
     - `- **Clause 1, "the judge is strong":** judge AUROC <v> [<lo>, <hi>] → <held|did not hold>`. It **held** iff `ci_low > 0.5` **and** the judge's AUROC ≥ the action-probe's AUROC − 0.05.
     - `- **Clause 2, "Δ ≈ 0":** Δ <v> [<lo>, <hi>] (action_best = <name>) → <held|did not hold>`. It **held** iff `ci_low ≤ 0 ≤ ci_high`.
     - `- **Frontier anchor:** <judge-frontier row: value and CI, or its not-run/partial note> (Issue 5)`.
     - No other verdict sentence anywhere in the report.
   - **(c) Wall-clock time per stage:**
     - features: the sum of `elapsed_ms` over the **last** `index.jsonl` entry per `item_id`, for each encoder;
     - judge: the sum of `elapsed_ms` over the cache rows, per model;
     - download: from `raw/oops/DOWNLOAD.json` or `download.log` timestamps, else `not recorded`;
     - probes: `not recorded (seconds)`.
   - **(d) Examples:** keep the current selection of 10 items (`default_rng(0)`; 5 `pre`, 5 `post`).
     - For each, extract the 4 judge frames (`judge.vlm_judge.sample_frames`) into `$TMPDIR/oops_examples/`. **Open and look at them yourself**, and write one plain sentence of what is visible into `docs/evals/2026-10-10_oops_examples.tsv` (`item_id<TAB>sentence`).
     - The generator reads that file through `--descriptions <path>` and prints `Seen in the frames: <sentence>`; a missing sentence prints `not described`.
     - Delete the frames afterwards. **No pixels in git.**
   - **(e) The caveat,** verbatim under the conditions table: `react-nonverbal on Oops! is not evidence of reaction signal: the reaction window equals the action window, so the post-failure audio contains the failure's own sound and any compilation music (02_data_sources.md → Oops! → As built).`
2. **Regenerate `docs/evals/2026-10-10_oops_h1.md` in place,** and add as its second line: `*Corrected <YYYY-MM-DD> (A7b): prediction check split by clause; judge error counts reclassified; wall-clock times and example descriptions added. The numbers are unchanged.*` **Do not rerun probes and do not touch the scorecard.**
3. **`src/judge/vlm_judge.py` `GeminiJudge.judge_item`:**
   - Wrap the existing per-call 429/5xx loop in an outer parse loop of up to 3 attempts: attempt 1 at `temperature` 0, attempts 2–3 at 0.3.
   - Return `(prob, raw, attempt, ms)` on the first parsable answer, or `(None, raw, 3, ms)` after 3 unparsable ones.
   - A 429/5xx that survives its 5 tries still **raises**, so the item is not cached (§8).
   - The running Gemini supervisor picks up the change at its next relaunch; that is intended.
4. **Tests:**
   - `tests/test_judge.py`, with a fake `genai` client:
     - replies `"none"`, `"none"`, `"42"` → `0.42`, attempts 3, and the 2nd and 3rd calls had `temperature=0.3`;
     - `"none"` × 3 → `None`, attempts 3;
     - five raised `429` errors → `judge_item` raises, and `run_judge` writes an `errors.jsonl` line and **no** cache line.
   - **New `tests/test_report_oops.py`:** `prediction_check` on two fixture row sets:
     - the real Oops! values → clause 1 `did not hold`, clause 2 `held`;
     - `judge 0.85 [0.80, 0.90]`, `action-probe 0.78`, and Δ `0.05 [0.02, 0.08]` → clause 1 `held`, clause 2 `did not hold`.

**Validation:**
- **Red first:** the new tests fail on the current code. Record them.
- **The regenerated report shows:**
  - clause 1 `did not hold` with `0.472 [0.440, 0.498]`, and clause 2 `held` with `0.015 [-0.002, 0.029]`;
  - the local judge: 2,708 answers, 0 parse failures, 2 `ffmpeg` errors;
  - Gemini: `scored 11 of 1072`, with its 429 count equal to `python3 -c "import json; print(len({json.loads(l)['item_id'] for l in open('<…>/gemini-3.6-flash/errors.jsonl') if '429' in json.loads(l)['error']}))"`;
  - 10 example sentences, each different and each consistent with its window type.
- `git diff results/scorecard.jsonl` is empty.
- G1–G3 green.

**Blast radius:** `tools/report_oops.py`, `docs/evals/2026-10-10_oops_h1.md`, `docs/evals/2026-10-10_oops_examples.tsv` (new), `src/judge/vlm_judge.py`, `tests/test_judge.py`, `tests/test_report_oops.py` (new); `ongoing_general_errors.md` §3; this guide's §1.4.

---

### A8: HoloAssist, end-to-end H1 (resume the draft)

**What this means for the maintainer:** the first test where the outcome can be partly hidden from the camera, and where the reactor (the instructor) is watching someone else work. That is the closest Wave A gets to "a person watching a robot work".

**Start condition:**
- A6c and A7b are pushed;
- `raw/holoassist/progress.json` reads `["verified"]` and `DOWNLOAD.json` has the video entry with `extracted: true`.

If the download supervisor aborted, read its log, relaunch it exactly as in §1.6 (it resumes with `curl -C -`), and wait. Issue 2 is resolved ("independent").

**Keep from the draft** (checked by the designer; do not rework):
- the builder's grouped 70/30 split by session prefix, done **before** per-class sampling (`03_eval_harness.md` §5);
- balanced sampling capped at 1,500/1,000 per class;
- `react-spoke` = `1 − spoke`;
- `react-full` = TF-IDF on `meta.transcript`;
- `fix` → `repair` in task names;
- the leakage regex and its injection test.

**Corrections to the draft** (each one fails a contract today):
1. **The `item_id` format** (`src/sources/holoassist.py:276`): it must be `f"holoassist:{vname}:{idx}"`, where `idx` is the event's index in `events`. Put the annotation's own `id` in `meta.event_id`.
2. **The forced split** (`holoassist.py:432`): pass `force=args.force_split`, with a new `--force-split` flag defaulting to off (§5).
   - Regenerate `splits/holoassist.json` **once** with `--force-split`, because the draft file's ids are in the old format, and say so in the commit body.
   - **Check, and report in the commit body,** that the regenerated split has the same group partition as the draft: 238 train groups and 102 test groups, from the same seeded shuffle. A different partition means the candidate set changed; find out why before going on.
3. **The hard-coded `"straddling_groups": 0`** (`holoassist.py:375`): compute it from the split map.
4. **The duration fallback of 10⁹ s** (`holoassist.py:210`): skip with reason `missing_duration` (expected count 0; every session has `videoMetadata.duration.seconds`).
5. **Session coverage:** after extraction, count the annotated sessions (of 1,758) whose `get_holoassist_video_path` exists. Record the count and the directory layout under a new `02_data_sources.md` → HoloAssist subsection **"Videos (as extracted, <date>)"**. Missing videos are skipped as `missing_video`, never silently.
6. **SSD-dependent tests in the fast suite:**
   - mark `@pytest.mark.slow` on `test_holoassist_builder_and_group_disjointness`, `test_real_items_leakage_check` and `test_real_split_group_disjointness`;
   - add a **fast** builder test on a 3-session synthetic annotation JSON in `tmp_path`;
   - add `assert all(i.item_id.startswith("holoassist:") for i in items)` to both builder tests.
7. **`tools/check_audio_presence.py`:**
   - **No `-100.0` sentinels** (`:36`, and the means).
   - Draw the 20 sessions (`default_rng(0)`, sessions sorted by `video_name`) **only from sessions with ≥ 1 instructor utterance and ≥ 1 no-utterance gap of ≥ 2 s**. A span with no samples is excluded and counted, not scored.
   - Write the summary to `DATA_ROOT/runs/holoassist_audio_presence.json`, and put the temp WAV under `TMPDIR`.
   - **Add a fast falsifying test:** synthetic 16 kHz audio with instructor spans +10 dB over the gaps → `passed: true`; flat noise → `passed: false`.
8. **`src/harness/probes.py`:**
   - **(a) The frontier row:** delete the hard-coded reason (`:198`) and the 50% rule (`:182`). Implement `03_eval_harness.md` §6/§8 exactly:
     - `k = 0` → `not run: <derived reason>`: `no cache file <path>`, or `0 of <n> scored; last error: <first 80 chars> (errors.jsonl)`;
     - `0 < k < n` → a real row with `notes: "partial: <k> of <n> scored"`;
     - it enters `action_best` (`:347`) only if `k ≥ 0.9·n`.
   - **(b)** Add the `react-nonverbal|spoke=1` diagnostic row (§6) for HoloAssist.
   - **(c)** Add synthetic tests: frontier `k = 0` → a derived note containing `0 of`; `k = 10 of 100` → a `partial` row that is **not** in `action_best`; `k = 95 of 100` → it enters `action_best`.
9. **`tools/download_holoassist.py`:** fix the disk check for the future: require `free ≥ archive bytes + 50 GiB` before extraction, and record `free_gib_before_extract` in `DOWNLOAD.json`. The running process keeps its old code; just record what happened.
10. **`tools/report_holoassist.py`:** build it on A7b's corrected pattern: per-clause prediction check, judge diagnostics, wall-clock, and descriptions you write after looking at the frames (TSV), as in A7b.

**Then run, in order** (§0.7 detached; §0.18 one model job at a time; record `memguard --status` before each):
1. `python tools/check_audio_presence.py`.
   - **Requirement:** a median per-session difference **≥ +3.5 dB**.
   - If it fails, **STOP**, and file **Issue 6** (*"instructor not audible in the video audio"*) with options A (`react-spoke`/`react-full` from annotations only) and B (drop HoloAssist's audio channel), and a `Your selection: _____` line. Continue only with what does not use audio.
2. `python -m sources.holoassist --build --force-split` (once). `validate_items` returns no errors, and `stats.json` shows every skip reason.
3. Features, one after the other, each under its own supervisor:
   - `python -m features.extract --dataset holoassist --encoder siglip-b16-224`;
   - then `--encoder e2v-plus-large`.
4. The judge: `python -m judge.vlm_judge --dataset holoassist --split train --backend ollama`, then `--split test`.
   - Gemini on test **only if Issue 5 = A and billing is on**; otherwise do not launch it.
5. `python -m harness.probes --dataset holoassist`.
6. `python tools/report_holoassist.py` → `docs/evals/<YYYY-MM-DD>_holoassist_h1.md`, with:
   - **at the top:** the pitch-shift caveat, and the group caveat (`02_data_sources.md` → HoloAssist → pinned details);
   - the conditions table, the Δ row, the diagnostic row and the shuffled controls;
   - the per-clause prediction check (*hidden outcome: reaction-only > action_best; fusion > action_best*);
   - the H1 pass bar (*proposed*, `00_thesis.md`: reaction-only ≥ 0.65 **and** Δ ≥ +0.03 with the CI excluding 0), with the measured values. It is stated as reported, never as the kill decision;
   - one plain paragraph on *"is the voice signal more than 'the instructor said something'?"*, answered from `react-spoke` and `react-nonverbal|spoke=1`;
   - the audio-presence measurement;
   - judge diagnostics and wall-clock time;
   - 10 examples (5 mistakes, 5 correct; `default_rng(0)`) described after looking at their frames.

**Validation:**
- **Leakage (falsifying):** no `context_text` matches `\b(mistake|correct|wrong|error|fix|instead)\b` (case-insensitive), over the real items; injecting `"mistake"` into one fixture item turns the test red.
- Every `item_id` starts with `holoassist:`.
- `straddling_groups` in `stats.json` is computed, and equals 0.
- `splits/holoassist.json` is group-disjoint and hash-verified.
- **The shuffled controls:** every `:shuffled` row's CI contains 0.5. If any `ci_low` is > 0.5, **STOP** and find the leak.
- Every §6 condition has a row (real, `partial` or `not run` with a derived reason), and the Δ row and `react-nonverbal|spoke=1` exist.
- The audio check's falsifying test is red on flat noise.
- G1–G4 green.

**Commit:** `feat(a8): …`, staging explicitly:
- the draft files of §1.3 (except `tests/test_memguard.py`, already committed in A6c);
- `results/scorecard.jsonl`;
- `docs/evals/<date>_holoassist_h1.md` and its TSV;
- `docs/02_data_sources.md`, `docs/ongoing_general_errors.md`, and this guide's §1.4.

**Blast radius:** all of the above; `splits/holoassist.json` (forced once, named in the body).

---

### A9: Re-measure; close out Wave A

**What this means for the maintainer:** one page that says whether Wave A found reaction signal beyond the action-only controls, what it could not measure, and what the designer needs to know to write Wave B.

**Implementation:**
1. **Conditional, only if Issue 5 = A was selected and billing is enabled:**
   - Let the Gemini runs finish on both test splits; check the cache coverage.
   - Re-run `python -m harness.probes --dataset oops` and `--dataset holoassist`. This appends superseding rows; put `rerun: judge-frontier scored <k>/<n> (Issue 5 A)` in each new row's `notes` by passing it through a new `--notes` CLI argument.
   - Regenerate both reports.
2. Run `scripts/battery.sh --slow` bare and update §1.4.
3. Write `docs/evals/<YYYY-MM-DD>_wave_a_summary.md`:
   - **the headline table:** dataset × condition, with CIs, n and `n_excluded`; the Δ rows; the diagnostic row; the shuffled controls;
   - **the per-clause prediction checks** for both datasets;
   - **the caveats,** stated in full:
     - Oops! `react-nonverbal` is not evidence of reaction signal;
     - HoloAssist audio is pitch-shifted, and its groups are pairs, not persons;
     - **the state of the "just ask an LLM" control** (Issue 5);
   - **anything suspicious:** an excluded share > 5%, a judge parse-failure rate > 2%, a condition that could not run;
   - compute time per stage;
   - **observations for Wave B:** encoder failures, the judge's behavior (the 90/100 answer pattern), and the observed speed of data handling.
4. Make sure each item has its line in `ongoing_general_errors.md` §3, and add A6c, A7b, A8 and A9 to §5.1 below.
5. **Rewrite this guide's title and status** to **"Queue Complete: waiting on Issue 1 and the Wave B spec"**, adding **"and Issue 5"** if it is still unselected. **Then stop. Do not invent work.**

**Validation:** every number in the summary matches a row in `results/scorecard.jsonl` (cite its `ts`); G1–G4 green.

---

## 4. Deferred: do NOT start

- **Wave B (hidden-outcome verdict data), DW1.** Needs Issue 1's selection **and** a Wave B spec from the designer. Do not write it yourself.
- **The face encoder** (`react-face`), DW2: part of Wave B.
- **H2 transfer** (HoloAssist / AM-FED+, BAD only if granted), DW3: needs Wave B.
- **The Ego4D false-positive set,** DW4.
- **Wave D, the robot check (offline H3; RoboReward)**, DW5: the designer writes its spec after Wave A's results. Do not download RoboReward.
- **Stage A live** (the microduck), DW6.
- **A larger local judge** (`gemma4:26b`): only if the maintainer selects Issue 5 option B **and** the designer specs it.
- **Any judge prompt or question change** (Issue 5 option D is not recommended).
- **Any web or YouTube video acquisition. Re-downloading any Ego4D or Charades-Ego video. Anything from tag `v0-saf-final`.**

---

## 5. Do NOT change

### 5.1 Already delivered (verified by the designer, October 10, 2026)

- **R0, the reorientation** (`454b40d`). **R1, the Wave A spec.** **R3, the A6b spec.** **R4, this verification.**
- **A1** `651039b`: `scripts/battery.sh` (G1–G4, maximum-exit rule, the exit-75 line) and `pytest.ini`.
- **A2** `d9de9d3`: `harness.metrics` (`auroc_ci`, `delta_auroc_ci`, `spearman_ci`, the group bootstrap, the 10,000-draw guard), and `harness.scorecard` (`ScorecardRow` in §4 order, `append_rows` with fsync, `config_hash`, the CLI with `--history` and `--selftest`).
- **A3** `dd7fa0f`: `harness.items` (5 verbatim rejections) and `harness.splits` (seeded group shuffle, straddle error, overwrite refusal, tamper check).
- **A4** `11df49f`: the HoloAssist labels (hashes re-verified), `--stats`, and the independence verdict.
- **A5** `5983241`: `FeatureCache`, `FrameEncoder` (768-d), `NonverbalAudioEncoder` (1024-d) and `features.extract`.
- **A6** `0f16799`: `build_prompt` (verbatim), `prompt_hash`, `parse_prob`, `sample_frames`, `OllamaJudge`, `GeminiJudge` and the cache. A7b changes only Gemini's parse retries.
- **A6b** `76e71cc`: `shared.memguard` (the floor, peaks re-measured at 1.60 / 4.94 / 7.80 GB, the shared lock path, unloading only our model, `--status`, the fake-reading override), plus exit 75 in the CLIs, the supervisor and the battery. A6c changes only what §3 lists.
- **A7** `6d72492`: the Oops! download, schema, adapter, split, features, judge, probes and 11 scorecard rows. A7b corrects only the report.
- `shared/vlm_client.ollama_chat`'s enforced timeout is load-bearing (`LESSONS_v0.md`, "Operations"). Do not replace it with the `ollama` Python client.

### 5.2 Accepted equivalents (checked October 10; do not "fix" these back)

- **Oops! "onsets missing" skips a clip if any worker marked no failure** (`n_notfound > 0`), rather than taking the median of the rest. It is stricter, and the caps were never binding (`02_data_sources.md` → Oops! → As built).
- **`features.visual.extract_frame` retries a seek at `t − 0.05 … t − 0.3 s`** when ffmpeg returns no frame at a clip's final timestamp. The frame is within 0.3 s of `e`.
- **`GeminiJudge` uses `max_output_tokens=32` and `thinking_budget=0`, and does not cache an item after 5 failed 429/5xx tries** (now written into §8).
- **`guard()` holds the heavy lock through the model load only, not through the first item** (§12 step 6). The first-window increment is small next to the load peak (`sr_e2v`: 4.76 GB peak during load vs. 3.0 GB steady).
- **An item's scorecard rows carry its parent commit's `git_sha` with `dirty: true`,** because rows are written before the item's one commit. The code of record is the item's own commit: the next commit that touches `results/scorecard.jsonl`.
- **`metrics.MetricResult`** is a 3-tuple subclass carrying `n_excluded`. It unpacks as `(point, low, high)`, as specified.
- **`make_group_split(…, source=, seed=)`**, and HoloAssist's group pre-assignment before sampling (written into §5).
- **HoloAssist task `fix motorcycle` → `repair motorcycle`** in `context_text` (`02_data_sources.md`, pinned details).
- **`extract.py` skips `memguard.check()` before the first item.** The first item loads through `guard()` itself.

### 5.3 Maintainer decisions

**October 8, 2026:**
- **The thesis:** human reactions as an additional reward signal for robot learning, learnable from web-scale video. **No human-in-the-loop rating.** Measure whether the thesis works, often and automatically.
- **SAF v0 retired.**
- **Both deployment stages,** in order: web reactions as free labels (B) first, then live reactions (A).
- **Any footage for training;** results reported on both robot-reaction (BAD) and first-person (HoloAssist) data.
- **"Just ask an LLM" is the control to beat.**
- **Hidden-outcome data first.**
- **Three-month goal: a paper-grade H1 + H2. Compute: the Mac Studio only.**
- **The designer does not code; an implementing agent builds from this guide. Commit straight to `main`.**
- **Issue 3 → the v0 videos were deleted.**

**October 9, 2026:**
- **No self-recorded data, no human-subjects study, no academic partner** (Issues 1D, 4B, 4C).
- **The offline robot check (Wave D) is in scope; the microduck demo is deferred.**

### 5.4 Invariants and intentional design decisions

- **Supervise on outcomes, never on emotion.** No emotion categories as features or targets, from any model. Embeddings are allowed.
- **`label = 1` always means a good outcome.** Hence `react-spoke` = `1 − spoke`.
- **The judge never hears audio** and never sees anything outside the action window.
- **Two action-only controls:** the Δ is taken against `action_best`. `judge-frontier` joins `action_best` only at ≥ 90% coverage.
- **Grouped splits and grouped CIs only.** Split files are written once; a forced rewrite is named in the commit.
- **The scorecard is append-only. H2 targets are never training data. Fixed probe hyperparameters. Nulls with reasons, never zeros or sentinels. No licensed-dataset pixels in git.**
- **`guard()` is re-entrant within a process; `read_memory()` fails closed; memory deferrals never consume supervisor attempts** (§12, from A6c onward).

### 5.5 Assessed and rejected: do NOT re-propose

- **Human rating rounds, golden labels, rater UIs, pre-seeded review.** The v0 benchmark failed this way (0/349 rated).
- **Hand-built affect channels** as the representation.
- **Self-recorded data collection, any human-subjects study, or an academic partnership** (declined October 9, 2026).
- **Ego4D bystander footage as H1 data. A first-person-only data restriction.**
- **Using Oops! descriptions or HoloAssist mistake/purpose labels as model inputs.**
- **Tuning prompts, hyperparameters, windows or caps on test results,** including rewriting the judge question because it scored 0.472 on Oops!. That is Issue 5 option D, and it is the maintainer's call.
- **Circumventing YouTube or any site's bot checks.**

---

## 6. Where the contracts live

| What | Where |
|---|---|
| The thesis, H1–H3, pass/kill, the decision log | `00_thesis.md` |
| Items, splits, the scorecard schema, conditions (incl. the diagnostic row and the not-run/partial rules), metrics, the judge, encoders | `03_eval_harness.md` §3–§9 |
| The memory guard: floor, peaks, the shared lock, re-entrancy, fail-closed reads, exit 75, supervisor and battery behavior | `03_eval_harness.md` §12 |
| Oops! and HoloAssist item definitions, schemas, as-built details and caveats | `02_data_sources.md` |
| Issues 1, 4, 5 (open); Issues 2, 3 (resolved); maintainer actions; lessons L1–L11; the resolved index | `ongoing_general_errors.md` |
| Operations (detached runs, ollama, decoding, storage) | `LESSONS_v0.md` "Operations"; `tools/README.md` |

---

## 7. Validation standard

- **Red first.** Before building an item, run its falsifying check against the current code and record the failure.
- **Every gate must be able to fail.** Show it red, then green.
- **Test the composed path (L11).** When a fix is about how two pieces call each other, at least one test uses the real callables, not a stub that cannot reproduce the bug.
- **Falsify the pipeline, not just the code:** the shuffled-label controls must sit at chance.
- **Report denominators** (`n_items`, `n_groups`, `n_excluded`) with every number.
- **Never loosen a bar or a threshold to pass it.** File it with the measurement and options.
- **Read your own outputs.** Open `stats.json`, the reports, a sample of judge `raw` answers and the example frames, and describe what you saw in the commit body. A report sentence that would be the same for every dataset is not a reading.

---

## 8. THE LOOP

```
(1) Is there an approved item? A6c, A7b, A8, A9, in §2 order. If all are done
    or blocked, STOP. Never start §4 work. Never fill in a `Your selection:`
    line.
(2) Read the item and EVERY contract section it names. Copy paths,
    constants, prompts and error strings VERBATIM.
(3) RED FIRST: run the item's falsifying check; record the failure.
(4) Build only what the item says. Nothing from §5.5.
(5) GREEN; then falsify (break, see red, restore, see green).
(6) Open every artefact you produced and describe it.
(7) scripts/battery.sh bare (add --slow when the item touched models).
    Update §1.4.
(8) ONE commit on main, scope = item id, staging EXPLICIT PATHS ONLY.
    WHY + red/green in the body. ONE line under "Wave A" in
    ongoing_general_errors.md §3. Never amend after pushing.
(9) /usr/bin/git push origin main
(10) Next item. A failed bar, an impossible rule or a missing input:
     file it (next: Issue 6), then continue only with items that do not
     depend on it.
```

---

## 9. Definition of Done: Wave A

- [ ] A6c, A7b, A8 and A9 each landed as one pushed commit on `main`, scoped to its id, with red and green runs recorded.
- [ ] The memory guard: tests (i) and (j) pass with the real nesting and were shown red without the depth branch; `read_memory()` fails closed; the supervisor finishes after more deferrals than `SR_SUPERVISE_MAX_ATTEMPTS`; fast tests never reach live Ollama; no jetsam report during a Wave A run names one of our processes.
- [ ] The Oops! report carries the per-clause prediction check (judge clause: did not hold), correct judge diagnostics, wall-clock times, real example descriptions and the `react-nonverbal` caveat, with the scorecard untouched.
- [ ] HoloAssist: items in the `holoassist:` id format; the audio-presence check run (or Issue 6 filed); every §6 condition has a row (real, partial or derived not-run), the Δ row and the `spoke=1` diagnostic exist; the shuffled controls sit at chance; the report is written with both caveats at the top.
- [ ] `docs/evals/<date>_wave_a_summary.md` is written, and every number in it is traceable to a scorecard row.
- [ ] `scripts/battery.sh` exits 0 and `--slow` exits 0; §1.4 is re-measured bare.
- [ ] This guide is rewritten to **Queue Complete** (naming what it waits on). **Then stop. Do not invent work.**
