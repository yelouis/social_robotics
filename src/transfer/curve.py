from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import roc_auc_score

from config import DATA_ROOT
from harness.items import read_items
from harness.metrics import auroc_ci, drop_missing_scores
from harness.probes import load_features_for_items
from features.cache import FeatureCache
from harness.scorecard import ScorecardRow, append_rows, config_hash, make_row
from harness.splits import load_split
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash
from transfer.heads import load_human_heads, load_transfer_scaler
from transfer.prior_lr import prior_regularized_logistic_regression, predict_proba


BUDGETS = [50, 100, 200, 500, 1000]
N_DRAWS = 20


def compute_curve_metrics_and_cis(
    y_test: np.ndarray,
    groups_test: List[str],
    scores_by_prior: Dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Compute mean20 AUROC, gains, and group-bootstrap 95% CIs for curve at a budget N."""
    y_clean, _, groups_clean, _ = drop_missing_scores(y_test, scores_by_prior["none"][0], groups_test)
    unique_groups = np.unique(groups_clean)
    n_groups = len(unique_groups)
    group_to_indices = {g: np.where(groups_clean == g)[0] for g in unique_groups}

    # Evaluate point estimates on full test set
    point_estimates: Dict[str, float] = {}
    test_aucs_by_prior: Dict[str, np.ndarray] = {}

    for p_name, score_matrix in scores_by_prior.items():
        # score_matrix is shape (n_draws, n_test)
        aucs = np.array([roc_auc_score(y_clean, draw_scores) for draw_scores in score_matrix])
        test_aucs_by_prior[p_name] = aucs
        point_estimates[p_name] = float(np.mean(aucs))

    for p in ["react", "shuffled", "truth"]:
        if p in test_aucs_by_prior and "none" in test_aucs_by_prior:
            point_estimates[f"gain_{p}"] = float(np.mean(test_aucs_by_prior[p] - test_aucs_by_prior["none"]))

    if "react" in test_aucs_by_prior and "shuffled" in test_aucs_by_prior:
        point_estimates["react_minus_shuffled"] = float(np.mean(test_aucs_by_prior["react"] - test_aucs_by_prior["shuffled"]))

    # 1,000 group bootstrap resamples
    rng = np.random.default_rng(seed)
    boot_aucs: Dict[str, List[float]] = {k: [] for k in point_estimates}

    draws = 0
    while len(boot_aucs["none"]) < n_boot:
        draws += 1
        if draws > 10000:
            raise RuntimeError("bootstrap: >10000 draws needed for two-class resamples")
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        y_b = y_clean[idx]
        if len(np.unique(y_b)) < 2:
            continue

        # Compute mean20 for each prior on this bootstrap sample
        b_means: Dict[str, float] = {}
        for p_name, score_matrix in scores_by_prior.items():
            aucs = [roc_auc_score(y_b, draw_scores[idx]) for draw_scores in score_matrix]
            m = float(np.mean(aucs))
            b_means[p_name] = m
            boot_aucs[p_name].append(m)

        for p in ["react", "shuffled", "truth"]:
            if p in b_means and "none" in b_means:
                boot_aucs[f"gain_{p}"].append(b_means[p] - b_means["none"])

        if "react" in b_means and "shuffled" in b_means:
            boot_aucs["react_minus_shuffled"].append(b_means["react"] - b_means["shuffled"])

    results: Dict[str, Dict[str, float]] = {}
    for k, pt in point_estimates.items():
        ci_low, ci_high = np.percentile(boot_aucs[k], [2.5, 97.5])
        results[k] = {
            "point": float(pt),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    return results


def run_curve(
    data_root: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
    scorecard_path: Optional[Union[str, Path]] = None,
) -> List[ScorecardRow]:
    root = Path(data_root or DATA_ROOT)
    splits_path = Path(splits_dir or "splits")
    sc_path = Path(scorecard_path or "results/scorecard.jsonl")

    # 1. Load items and split
    items_path = root / "items" / "roboreward" / "items.jsonl"
    stats_path = root / "items" / "roboreward" / "stats.json"
    if not items_path.exists():
        raise FileNotFoundError(f"RoboReward items not found: {items_path}")

    all_items = read_items(items_path)
    item_map = {it.item_id: it for it in all_items}

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_pool_ids = stats["train_pool_ids"]

    split_data = load_split("roboreward", path_dir=splits_path)
    test_ids = split_data["test"]

    train_pool_items = [item_map[i] for i in train_pool_ids]
    test_items = [item_map[i] for i in test_ids]

    # Load u9 features
    u9_cache = FeatureCache(dataset="roboreward", encoder_id="siglip-b16-224-u9", data_root=root)
    X_train_raw, valid_tr, _ = load_features_for_items(train_pool_items, u9_cache)
    X_test_raw, valid_te, _ = load_features_for_items(test_items, u9_cache)

    if not np.all(valid_tr):
        raise RuntimeError("Some robot train pool items missing u9 features")
    if not np.all(valid_te):
        raise RuntimeError("Some robot test items missing u9 features")

    # Load scaler and scale features
    scaler = load_transfer_scaler(root / "transfer" / "scaler.npz")
    X_train_pool = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_train_pool = np.array([it.label for it in train_pool_items], dtype=int)
    y_test = np.array([it.label for it in test_items], dtype=int)
    groups_test = [it.group_id for it in test_items]
    n_groups_test = len(set(groups_test))

    # Load human heads
    heads = load_human_heads(root / "transfer" / "heads.npz")

    # Separate train pool into successes and failures
    succ_idx = np.where(y_train_pool == 1)[0]
    fail_idx = np.where(y_train_pool == 0)[0]

    all_scores: Dict[str, np.ndarray] = {}
    rows: List[ScorecardRow] = []

    cfg_curve = {
        "budgets": BUDGETS,
        "n_draws": N_DRAWS,
        "encoder": "siglip-b16-224-u9",
        "dataset": "roboreward",
    }
    c_hash = config_hash(cfg_curve)

    print("Running Wave D curve across budgets...")

    for N in BUDGETS:
        half_n = N // 2
        scores_by_prior: Dict[str, np.ndarray] = {
            "none": np.empty((N_DRAWS, len(test_items)), dtype=np.float64),
            "react": np.empty((N_DRAWS, len(test_items)), dtype=np.float64),
            "shuffled": np.empty((N_DRAWS, len(test_items)), dtype=np.float64),
            "truth": np.empty((N_DRAWS, len(test_items)), dtype=np.float64),
        }

        if N == 1000:
            shuf_ctrl_scores = np.empty((N_DRAWS, len(test_items)), dtype=np.float64)

        for r in range(N_DRAWS):
            rng = np.random.default_rng(1000 * N + r)
            draw_succ = rng.choice(succ_idx, size=half_n, replace=False)
            draw_fail = rng.choice(fail_idx, size=half_n, replace=False)
            draw_idx = np.concatenate([draw_succ, draw_fail])
            # Sorted by item_id order
            draw_idx.sort()

            X_draw = X_train_pool[draw_idx]
            y_draw = y_train_pool[draw_idx]

            # Fit 4 priors on the same draw
            for p_name in ["none", "react", "shuffled", "truth"]:
                w0 = heads[p_name][0] if p_name != "none" else None
                w, b = prior_regularized_logistic_regression(X_draw, y_draw, w0=w0, C=1.0)
                probs = predict_proba(X_test, w, b)
                scores_by_prior[p_name][r] = probs

            # Leak check control for N=1000
            if N == 1000:
                rng_shuf = np.random.default_rng(1000 * N + r + 7)
                y_draw_shuf = rng_shuf.permutation(y_draw)
                w_s, b_s = prior_regularized_logistic_regression(X_draw, y_draw_shuf, w0=None, C=1.0)
                shuf_ctrl_scores[r] = predict_proba(X_test, w_s, b_s)

        # Record raw scores
        for p_name in ["none", "react", "shuffled", "truth"]:
            all_scores[f"{p_name}_N_{N}"] = scores_by_prior[p_name]
        if N == 1000:
            all_scores["shuffled_ctrl_N_1000"] = shuf_ctrl_scores

        # Compute metrics and CIs
        curve_metrics = compute_curve_metrics_and_cis(
            y_test=y_test,
            groups_test=groups_test,
            scores_by_prior=scores_by_prior,
            n_boot=1000,
            seed=0,
        )

        # 4 detector rows
        for p_name in ["none", "react", "shuffled", "truth"]:
            m = curve_metrics[p_name]
            rows.append(make_row(
                hypothesis="H3",
                dataset="roboreward",
                split="test",
                condition=f"detector:{p_name}:N={N}",
                metric="auroc_mean20",
                value=m["point"],
                ci_low=m["ci_low"],
                ci_high=m["ci_high"],
                n_items=len(test_items),
                n_groups=n_groups_test,
                n_excluded=0,
                config_hash=c_hash,
                notes=f"prior={p_name}, N={N}, 20 draws paired",
            ))

        # 3 gain rows
        for p_name in ["react", "shuffled", "truth"]:
            m = curve_metrics[f"gain_{p_name}"]
            rows.append(make_row(
                hypothesis="H3",
                dataset="roboreward",
                split="test",
                condition=f"gain:{p_name}:N={N}",
                metric="delta_auroc_mean20",
                value=m["point"],
                ci_low=m["ci_low"],
                ci_high=m["ci_high"],
                n_items=len(test_items),
                n_groups=n_groups_test,
                n_excluded=0,
                config_hash=c_hash,
                notes=f"gain {p_name} over none, N={N}, 20 draws paired",
            ))

        # 1 react_minus_shuffled row
        m_rvs = curve_metrics["react_minus_shuffled"]
        rows.append(make_row(
            hypothesis="H3",
            dataset="roboreward",
            split="test",
            condition=f"react_minus_shuffled:N={N}",
            metric="delta_auroc_mean20",
            value=m_rvs["point"],
            ci_low=m_rvs["ci_low"],
            ci_high=m_rvs["ci_high"],
            n_items=len(test_items),
            n_groups=n_groups_test,
            n_excluded=0,
            config_hash=c_hash,
            notes=f"react minus shuffled, N={N}, 20 draws paired",
        ))

        # Shuffled leak check at N=1000
        if N == 1000:
            m_leak = compute_curve_metrics_and_cis(
                y_test=y_test,
                groups_test=groups_test,
                scores_by_prior={"none": shuf_ctrl_scores},
                n_boot=1000,
                seed=0,
            )["none"]
            rows.append(make_row(
                hypothesis="H3",
                dataset="roboreward",
                split="test",
                condition="detector:none:N=1000:shuffled",
                metric="auroc_mean20",
                value=m_leak["point"],
                ci_low=m_leak["ci_low"],
                ci_high=m_leak["ci_high"],
                n_items=len(test_items),
                n_groups=n_groups_test,
                n_excluded=0,
                config_hash=c_hash,
                notes="shuffled-label control; robot labels permuted per draw (seed=1000*N + r + 7)",
            ))

    # Reference lines
    # 1. robot-full
    w_full, b_full = prior_regularized_logistic_regression(X_train_pool, y_train_pool, w0=None, C=1.0)
    full_probs = predict_proba(X_test, w_full, b_full)
    res_full = auroc_ci(y_test, full_probs, groups_test)
    rows.append(make_row(
        hypothesis="H3",
        dataset="roboreward",
        split="test",
        condition="robot-full",
        metric="auroc",
        value=res_full.point,
        ci_low=res_full.ci_low,
        ci_high=res_full.ci_high,
        n_items=len(test_items),
        n_groups=n_groups_test,
        n_excluded=0,
        config_hash=c_hash,
        notes="trained on full 4,000-item robot train pool (w0=0)",
    ))

    # 2. judge-large
    p_hash = prompt_hash(DATASET_QUESTIONS["roboreward"])
    j_cache_path = root / "judge" / "roboreward" / "gemma4_26b" / f"{p_hash}.jsonl"
    judge_scores_map = {}
    if j_cache_path.exists():
        for line in j_cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("judge_prob") is not None:
                        judge_scores_map[d["item_id"]] = float(d["judge_prob"])
                except Exception:
                    pass

    judge_probs = np.array([judge_scores_map.get(it.item_id, np.nan) for it in test_items])
    valid_j = ~np.isnan(judge_probs)
    if np.any(valid_j):
        res_judge = auroc_ci(y_test[valid_j], judge_probs[valid_j], [groups_test[i] for i in range(len(groups_test)) if valid_j[i]])
        rows.append(make_row(
            hypothesis="H3",
            dataset="roboreward",
            split="test",
            condition="judge-large",
            metric="auroc",
            value=res_judge.point,
            ci_low=res_judge.ci_low,
            ci_high=res_judge.ci_high,
            n_items=int(np.sum(valid_j)),
            n_groups=n_groups_test,
            n_excluded=int(np.sum(~valid_j)),
            config_hash=c_hash,
            notes="gemma4:26b zero-shot judge on test set",
        ))
    else:
        rows.append(make_row(
            hypothesis="H3",
            dataset="roboreward",
            split="test",
            condition="judge-large",
            metric="auroc",
            value=None,
            ci_low=None,
            ci_high=None,
            n_items=len(test_items),
            n_groups=n_groups_test,
            n_excluded=len(test_items),
            config_hash=c_hash,
            notes="not run",
        ))

    # 3. roboreward-8b
    rows.append(make_row(
        hypothesis="H3",
        dataset="roboreward",
        split="test",
        condition="roboreward-8b",
        metric="auroc",
        value=None,
        ci_low=None,
        ci_high=None,
        n_items=len(test_items),
        n_groups=n_groups_test,
        n_excluded=len(test_items),
        config_hash=c_hash,
        notes="not run: RoboReward-8B is Qwen3-VL-based and needs a transformers newer than the frozen 4.46.3 (Issue 9 chose B)",
    ))

    # Save curve scores
    scores_npz_path = root / "transfer" / "curve_scores.npz"
    scores_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(scores_npz_path, **all_scores)
    print(f"Saved raw curve scores to {scores_npz_path}")

    # Append rows to scorecard
    append_rows(rows, sc_path)
    print(f"Appended {len(rows)} Wave D curve and reference rows to {sc_path}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wave D data-efficiency curve")
    parser.add_argument("--data-root", default=None, help="Data root path")
    parser.add_argument("--splits-dir", default=None, help="Splits directory")
    parser.add_argument("--scorecard", default="results/scorecard.jsonl", help="Scorecard path")
    args = parser.parse_args()

    run_curve(data_root=args.data_root, splits_dir=args.splits_dir, scorecard_path=args.scorecard)


if __name__ == "__main__":
    main()
