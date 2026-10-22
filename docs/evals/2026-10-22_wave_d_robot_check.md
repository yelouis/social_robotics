# Wave D Robot Check Evaluation Report (2026-10-22)

## 1. Executive Summary & Verdict (H3 Offline)

- **Pre-registered Verdict:** **NOT SUPPORTED**
- **Interpretation:** Human→robot transfer does not help in this setup even with true human labels; reaction labels cannot be judged here.

### Key N=100 Comparisons

- **`gain:truth:N=100`:** -0.0507 [-0.0779, -0.0240]
- **`gain:react:N=100`:** +0.0009 [-0.0006, 0.0024]
- **`react_minus_shuffled:N=100`:** +0.0008 [-0.0013, 0.0029]

## 2. Data-Efficiency Curve (mean20 Test AUROC)

| Budget (N) | None (w0=0) | React Prior | Shuffled Prior | Truth Prior | Gain (React - None) | Gain (React - Shuffled) |
|---|---|---|---|---|---|---|
| N=50 | 0.562 [0.547, 0.577] | 0.563 [0.548, 0.577] | 0.562 [0.547, 0.578] | 0.512 [0.484, 0.540] | 0.001 [-0.001, 0.003] | 0.001 [-0.002, 0.004] |
| N=100 | 0.579 [0.559, 0.598] | 0.580 [0.559, 0.598] | 0.579 [0.558, 0.598] | 0.528 [0.500, 0.555] | 0.001 [-0.001, 0.002] | 0.001 [-0.001, 0.003] |
| N=200 | 0.596 [0.575, 0.617] | 0.597 [0.575, 0.617] | 0.596 [0.574, 0.617] | 0.554 [0.528, 0.580] | 0.000 [-0.001, 0.002] | 0.000 [-0.001, 0.002] |
| N=500 | 0.611 [0.588, 0.630] | 0.611 [0.588, 0.631] | 0.611 [0.588, 0.631] | 0.584 [0.560, 0.607] | 0.000 [-0.001, 0.001] | -0.000 [-0.002, 0.001] |
| N=1000 | 0.625 [0.602, 0.647] | 0.625 [0.602, 0.647] | 0.625 [0.603, 0.647] | 0.618 [0.593, 0.642] | -0.000 [-0.001, 0.000] | -0.000 [-0.001, 0.000] |

## 3. Reference Lines (Context, No Verdict)

| Condition | Metric | Value | 95% CI | Notes |
|---|---|---|---|---|
| `robot-full` | auroc | 0.657 | [0.624, 0.690] | trained on full 4,000-item robot train pool (w0=0) |
| `judge-large` | auroc | 0.747 | [0.720, 0.774] | gemma4:26b zero-shot judge on test set |
| `roboreward-8b` | auroc | null | — | not run: RoboReward-8B is Qwen3-VL-based and needs a transformers newer than the frozen 4.46.3 (Issue 9 chose B) |

## 4. Pseudo-Label Quality (HoloAssist Pool)

| Condition | Metric | Value | 95% CI | N (items) | Notes |
|---|---|---|---|---|---|
| `pseudo-label-quality` | auroc | null | — | — | not run: fewer than 2 classes in labeler_unseen_group (all items are correct actions; test split consumed all mistakes) |
| `pseudo-label-quality:all` | auroc | 0.6210 | [0.5684, 0.6739] | 20000 | all scored pool items |

## 5. Shuffled Control (Leak Check)

Leak check (03_eval_harness.md §11.9): a shuffled control fails only if its ci_low > 0.5.

| Condition | Metric | Value | 95% CI | Leak check | Notes |
|---|---|---|---|---|---|
| `detector:none:N=1000:shuffled` | auroc_mean20 | 0.498 | [0.491, 0.505] | pass | shuffled-label control; robot labels permuted per draw (seed=1000*N + r + 7) |

