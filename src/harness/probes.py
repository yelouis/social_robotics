from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import DATA_ROOT
from features.cache import FeatureCache
from harness.items import Item, read_items
from harness.metrics import auroc_ci, delta_auroc_ci
from harness.scorecard import ScorecardRow, append_rows, config_hash, make_row
from harness.splits import load_split
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash


def logit(prob: np.ndarray) -> np.ndarray:
    """Computes logit clipping P to [0.01, 0.99] per 03_eval_harness.md §6."""
    clipped = np.clip(prob, 0.01, 0.99)
    return np.log(clipped / (1.0 - clipped))


def load_features_for_items(
    items: List[Item],
    cache: FeatureCache,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Loads features for items from cache.

    Returns (features_array, valid_mask, n_missing).
    """
    feats = []
    valid_mask = []
    missing_count = 0

    for it in items:
        if cache.has(it.item_id):
            arr = cache.load(it.item_id)
            feats.append(arr)
            valid_mask.append(True)
        else:
            feats.append(None)
            valid_mask.append(False)
            missing_count += 1

    feature_dim = 0
    for f in feats:
        if f is not None:
            feature_dim = len(f)
            break

    dense_feats = []
    for f in feats:
        if f is not None:
            dense_feats.append(f)
        else:
            dense_feats.append(np.zeros(feature_dim, dtype=np.float32))

    return np.array(dense_feats, dtype=np.float32), np.array(valid_mask, dtype=bool), missing_count


def load_judge_scores(
    items: List[Item],
    dataset: str,
    model_tag: str,
    p_hash: str,
    data_root: Path,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Loads cached judge probabilities.

    Returns (scores_array, valid_mask, n_excluded).
    """
    cache_file = data_root / "judge" / dataset / model_tag / f"{p_hash}.jsonl"
    cached: Dict[str, Optional[float]] = {}
    if cache_file.exists():
        import json
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                    cached[entry["item_id"]] = entry.get("judge_prob")
                except Exception:
                    pass

    scores = []
    valid = []
    excluded = 0
    for it in items:
        prob = cached.get(it.item_id)
        if prob is not None:
            scores.append(float(prob))
            valid.append(True)
        else:
            scores.append(np.nan)
            valid.append(False)
            excluded += 1

    return np.array(scores, dtype=np.float64), np.array(valid, dtype=bool), excluded


def build_test_only_judge_row(
    condition: str,
    dataset: str,
    split_name: str,
    model_tag: str,
    p_hash: str,
    data_root: Path,
    test_items: List[Item],
    y_test: np.ndarray,
    groups_test: List[str],
    c_hash: str,
) -> Tuple[ScorecardRow, np.ndarray, Optional[float], int]:
    """Builds scorecard row for test-only judge condition (judge-large, judge-frontier) per §6."""
    cache_file = data_root / "judge" / dataset / model_tag / f"{p_hash}.jsonl"
    errors_file = data_root / "judge" / dataset / model_tag / "errors.jsonl"

    scores, valid, _ = load_judge_scores(test_items, dataset, model_tag, p_hash, data_root)
    k = int(np.sum(valid))
    n_total_test = len(test_items)
    n_groups_test = len(set(groups_test))

    if k == 0:
        if not cache_file.exists():
            notes = f"not run: no cache file {cache_file}"
        else:
            err_str = None
            if errors_file.exists():
                try:
                    lines = [line.strip() for line in errors_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if lines:
                        err_str = lines[-1][:80]
                except Exception:
                    pass
            if err_str:
                notes = f"not run: 0 of {n_total_test} scored; last error: {err_str}"
            else:
                notes = f"not run: 0 of {n_total_test} scored"

        row = make_row(
            hypothesis="H1",
            dataset=dataset,
            split=split_name,
            condition=condition,
            metric="auroc",
            value=None,
            ci_low=None,
            ci_high=None,
            n_items=n_total_test,
            n_groups=n_groups_test,
            n_excluded=n_total_test,
            config_hash=c_hash,
            notes=notes,
        )
        return row, scores, None, k

    elif 0 < k < n_total_test:
        try:
            res = auroc_ci(y_test, scores, groups_test)
            row = make_row(
                hypothesis="H1",
                dataset=dataset,
                split=split_name,
                condition=condition,
                metric="auroc",
                value=res.point,
                ci_low=res.ci_low,
                ci_high=res.ci_high,
                n_items=n_total_test - res.n_excluded,
                n_groups=n_groups_test,
                n_excluded=res.n_excluded,
                config_hash=c_hash,
                notes=f"partial: {k} of {n_total_test} scored",
            )
            return row, scores, res.point, k
        except Exception as exc:
            row = make_row(
                hypothesis="H1",
                dataset=dataset,
                split=split_name,
                condition=condition,
                metric="auroc",
                value=None,
                ci_low=None,
                ci_high=None,
                n_items=k,
                n_groups=n_groups_test,
                n_excluded=n_total_test - k,
                config_hash=c_hash,
                notes=f"partial: {k} of {n_total_test} scored; metric undefined: {exc}",
            )
            return row, scores, None, k

    else:
        # k == n_total_test
        res = auroc_ci(y_test, scores, groups_test)
        row = make_row(
            hypothesis="H1",
            dataset=dataset,
            split=split_name,
            condition=condition,
            metric="auroc",
            value=res.point,
            ci_low=res.ci_low,
            ci_high=res.ci_high,
            n_items=n_total_test - res.n_excluded,
            n_groups=n_groups_test,
            n_excluded=res.n_excluded,
            config_hash=c_hash,
            notes="",
        )
        return row, scores, res.point, k


def run_probes(
    dataset: str,
    split_name: str = "test",
    data_root: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
    scorecard_path: Optional[Union[str, Path]] = None,
    judge_model_tag: str = "qwen2.5vl_7b",
    notes: str = "",
) -> List[ScorecardRow]:
    root = Path(data_root or DATA_ROOT)
    splits_path = Path(splits_dir or "splits")

    items_path = root / "items" / dataset / "items.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(f"Items not found: {items_path}")
    all_items = read_items(items_path)
    item_map = {it.item_id: it for it in all_items}

    split_data = load_split(dataset, path_dir=splits_path)
    train_ids = split_data["train"]
    test_ids = split_data["test"]

    train_items = [item_map[i] for i in train_ids if i in item_map]
    test_items = [item_map[i] for i in test_ids if i in item_map]

    y_train = np.array([it.label for it in train_items], dtype=int)
    y_test = np.array([it.label for it in test_items], dtype=int)
    groups_test = [it.group_id for it in test_items]

    # Load Action & React features
    action_cache = FeatureCache(dataset=dataset, encoder_id="siglip-b16-224", data_root=root)
    react_cache = FeatureCache(dataset=dataset, encoder_id="e2v-plus-large", data_root=root)

    X_action_train, valid_act_tr, n_ex_act_tr = load_features_for_items(train_items, action_cache)
    X_action_test, valid_act_te, n_ex_act_te = load_features_for_items(test_items, action_cache)

    X_react_train, valid_re_tr, n_ex_re_tr = load_features_for_items(train_items, react_cache)
    X_react_test, valid_re_te, n_ex_re_te = load_features_for_items(test_items, react_cache)

    # Load Judge scores
    q = DATASET_QUESTIONS.get(dataset, "Is everything going as intended?")
    p_hash = prompt_hash(q)
    judge_train_scores, valid_j_tr, n_ex_j_tr = load_judge_scores(train_items, dataset, judge_model_tag, p_hash, root)
    judge_test_scores, valid_j_te, n_ex_j_te = load_judge_scores(test_items, dataset, judge_model_tag, p_hash, root)

    # Run Config Hash
    cfg = {
        "dataset": dataset,
        "encoder_action": "siglip-b16-224",
        "encoder_react": "e2v-plus-large",
        "judge_model": judge_model_tag,
        "judge_large_model": "gemma4:26b",
        "judge_large_options": {"temperature": 0, "num_ctx": 16384, "num_predict": 16},
        "prompt_hash": p_hash,
        "probe_C": 1.0,
        "probe_class_weight": "balanced",
        "probe_max_iter": 2000,
        "split_sha256": split_data["sha256"],
    }
    c_hash = config_hash(cfg)

    rows: List[ScorecardRow] = []
    n_total_test = len(test_items)
    n_groups_test = len(set(groups_test))

    # 1. Condition: judge
    j_res = auroc_ci(y_test, judge_test_scores, groups_test)
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="judge", metric="auroc",
        value=j_res.point, ci_low=j_res.ci_low, ci_high=j_res.ci_high,
        n_items=n_total_test - j_res.n_excluded, n_groups=n_groups_test, n_excluded=j_res.n_excluded,
        config_hash=c_hash, notes="",
    ))

    # 2. Condition: judge-large (test split only)
    row_large, large_test_scores, large_point, k_large = build_test_only_judge_row(
        condition="judge-large",
        dataset=dataset,
        split_name=split_name,
        model_tag="gemma4_26b",
        p_hash=p_hash,
        data_root=root,
        test_items=test_items,
        y_test=y_test,
        groups_test=groups_test,
        c_hash=c_hash,
    )
    rows.append(row_large)

    # 3. Condition: judge-frontier (test split only)
    row_front, frontier_test_scores, front_point, k_front = build_test_only_judge_row(
        condition="judge-frontier",
        dataset=dataset,
        split_name=split_name,
        model_tag="gemini-3.6-flash",
        p_hash=p_hash,
        data_root=root,
        test_items=test_items,
        y_test=y_test,
        groups_test=groups_test,
        c_hash=c_hash,
    )
    rows.append(row_front)

    # 3. Condition: action-probe
    act_mask_tr = valid_act_tr
    act_scaler = StandardScaler().fit(X_action_train[act_mask_tr])
    X_act_tr_scaled = act_scaler.transform(X_action_train[act_mask_tr])
    clf_act = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf_act.fit(X_act_tr_scaled, y_train[act_mask_tr])

    act_test_probs = np.full(len(test_items), np.nan)
    if np.any(valid_act_te):
        X_act_te_scaled = act_scaler.transform(X_action_test[valid_act_te])
        act_test_probs[valid_act_te] = clf_act.predict_proba(X_act_te_scaled)[:, 1]

    act_res = auroc_ci(y_test, act_test_probs, groups_test)
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="action-probe", metric="auroc",
        value=act_res.point, ci_low=act_res.ci_low, ci_high=act_res.ci_high,
        n_items=n_total_test - act_res.n_excluded, n_groups=n_groups_test, n_excluded=act_res.n_excluded,
        config_hash=c_hash, notes="",
    ))

    # 4. Condition: react-nonverbal
    re_mask_tr = valid_re_tr
    re_scaler = StandardScaler().fit(X_react_train[re_mask_tr])
    X_re_tr_scaled = re_scaler.transform(X_react_train[re_mask_tr])
    clf_re = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf_re.fit(X_re_tr_scaled, y_train[re_mask_tr])

    re_test_probs = np.full(len(test_items), np.nan)
    if np.any(valid_re_te):
        X_re_te_scaled = re_scaler.transform(X_react_test[valid_re_te])
        re_test_probs[valid_re_te] = clf_re.predict_proba(X_re_te_scaled)[:, 1]

    re_res = auroc_ci(y_test, re_test_probs, groups_test)
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="react-nonverbal", metric="auroc",
        value=re_res.point, ci_low=re_res.ci_low, ci_high=re_res.ci_high,
        n_items=n_total_test - re_res.n_excluded, n_groups=n_groups_test, n_excluded=re_res.n_excluded,
        config_hash=c_hash, notes="",
    ))

    # 5. Condition: react-spoke & react-full
    if dataset == "oops":
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="react-spoke", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="not run: no annotations of reactor speech in Oops!",
        ))
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="react-full", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="not run: no annotations of reactor speech in Oops!",
        ))
    else:
        # react-spoke: binary feature did reactor speak in reaction window
        # Direction: silence=1.0 (correct), speech=0.0 (mistake)
        spoke_scores = np.array([1.0 - float(it.meta.get("spoke", 0)) for it in test_items])
        spk_res = auroc_ci(y_test, spoke_scores, groups_test)
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="react-spoke", metric="auroc",
            value=spk_res.point, ci_low=spk_res.ci_low, ci_high=spk_res.ci_high,
            n_items=n_total_test - spk_res.n_excluded, n_groups=n_groups_test, n_excluded=spk_res.n_excluded,
            config_hash=c_hash, notes="",
        ))

        # react-full: TF-IDF (ngram_range=(1,2), min_df=2) + logistic regression on meta.transcript
        texts_tr = [it.meta.get("transcript") or "" for it in train_items]
        texts_te = [it.meta.get("transcript") or "" for it in test_items]
        has_transcripts = any(t.strip() for t in texts_tr)

        if has_transcripts:
            import warnings
            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            X_text_tr = tfidf.fit_transform(texts_tr)
            X_text_te = tfidf.transform(texts_te)

            clf_text = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                clf_text.fit(X_text_tr, y_train)

            text_probs = clf_text.predict_proba(X_text_te)[:, 1]
            rf_res = auroc_ci(y_test, text_probs, groups_test)
            rows.append(make_row(
                hypothesis="H1", dataset=dataset, split=split_name,
                condition="react-full", metric="auroc",
                value=rf_res.point, ci_low=rf_res.ci_low, ci_high=rf_res.ci_high,
                n_items=n_total_test - rf_res.n_excluded, n_groups=n_groups_test, n_excluded=rf_res.n_excluded,
                config_hash=c_hash, notes="",
            ))
        else:
            rows.append(make_row(
                hypothesis="H1", dataset=dataset, split=split_name,
                condition="react-full", metric="auroc",
                value=None, ci_low=None, ci_high=None,
                n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
                config_hash=c_hash, notes="not run: annotations carry no transcripts",
            ))

    # 6. Condition: fusion [logit(judge), action, react]
    fusion_tr_mask = valid_j_tr & valid_act_tr & valid_re_tr
    fusion_te_mask = valid_j_te & valid_act_te & valid_re_te

    logit_j_tr = logit(judge_train_scores[fusion_tr_mask])
    X_fusion_tr_raw = np.hstack([
        logit_j_tr[:, None],
        X_action_train[fusion_tr_mask],
        X_react_train[fusion_tr_mask],
    ])
    fusion_scaler = StandardScaler().fit(X_fusion_tr_raw)
    X_fusion_tr = fusion_scaler.transform(X_fusion_tr_raw)

    clf_fusion = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf_fusion.fit(X_fusion_tr, y_train[fusion_tr_mask])

    fusion_test_probs = np.full(len(test_items), np.nan)
    if np.any(fusion_te_mask):
        logit_j_te = logit(judge_test_scores[fusion_te_mask])
        X_fusion_te_raw = np.hstack([
            logit_j_te[:, None],
            X_action_test[fusion_te_mask],
            X_react_test[fusion_te_mask],
        ])
        X_fusion_te = fusion_scaler.transform(X_fusion_te_raw)
        fusion_test_probs[fusion_te_mask] = clf_fusion.predict_proba(X_fusion_te)[:, 1]

    fu_res = auroc_ci(y_test, fusion_test_probs, groups_test)
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="fusion", metric="auroc",
        value=fu_res.point, ci_low=fu_res.ci_low, ci_high=fu_res.ci_high,
        n_items=n_total_test - fu_res.n_excluded, n_groups=n_groups_test, n_excluded=fu_res.n_excluded,
        config_hash=c_hash, notes="",
    ))

    # 7. Condition: fusion_minus_action_best
    action_candidates = []
    k_judge = int(np.sum(valid_j_te))
    if k_judge >= 0.9 * n_total_test and j_res.point is not None:
        action_candidates.append(("judge", judge_test_scores, j_res.point))
    if k_large >= 0.9 * n_total_test and large_point is not None:
        action_candidates.append(("judge-large", large_test_scores, large_point))
    if k_front >= 0.9 * n_total_test and front_point is not None:
        action_candidates.append(("judge-frontier", frontier_test_scores, front_point))
    if act_res.point is not None:
        action_candidates.append(("action-probe", act_test_probs, act_res.point))

    valid_candidates = [c for c in action_candidates if c[2] is not None]
    best_candidate = max(valid_candidates, key=lambda c: c[2])
    action_best_name, action_best_scores, _ = best_candidate

    d_res = delta_auroc_ci(
        y_test, action_best_scores, fusion_test_probs, groups_test
    )
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="fusion_minus_action_best", metric="delta_auroc",
        value=d_res.point, ci_low=d_res.ci_low, ci_high=d_res.ci_high,
        n_items=n_total_test - d_res.n_excluded, n_groups=n_groups_test, n_excluded=d_res.n_excluded,
        config_hash=c_hash, notes=f"action_best={action_best_name}",
    ))

    # 8. Shuffled-label controls (permuted train labels)
    rng = np.random.default_rng(0)
    y_train_shuffled = y_train.copy()
    rng.shuffle(y_train_shuffled)

    # action-probe:shuffled
    clf_act_sh = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf_act_sh.fit(X_act_tr_scaled, y_train_shuffled[act_mask_tr])
    act_sh_probs = np.full(len(test_items), np.nan)
    if np.any(valid_act_te):
        act_sh_probs[valid_act_te] = clf_act_sh.predict_proba(X_act_te_scaled)[:, 1]
    ash_res = auroc_ci(y_test, act_sh_probs, groups_test)
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="action-probe:shuffled", metric="auroc",
        value=ash_res.point, ci_low=ash_res.ci_low, ci_high=ash_res.ci_high,
        n_items=n_total_test - ash_res.n_excluded, n_groups=n_groups_test, n_excluded=ash_res.n_excluded,
        config_hash=c_hash, notes="shuffled-label control",
    ))

    # react-nonverbal:shuffled
    clf_re_sh = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf_re_sh.fit(X_re_tr_scaled, y_train_shuffled[re_mask_tr])
    re_sh_probs = np.full(len(test_items), np.nan)
    if np.any(valid_re_te):
        re_sh_probs[valid_re_te] = clf_re_sh.predict_proba(X_re_te_scaled)[:, 1]
    rsh_res = auroc_ci(y_test, re_sh_probs, groups_test)
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="react-nonverbal:shuffled", metric="auroc",
        value=rsh_res.point, ci_low=rsh_res.ci_low, ci_high=rsh_res.ci_high,
        n_items=n_total_test - rsh_res.n_excluded, n_groups=n_groups_test, n_excluded=rsh_res.n_excluded,
        config_hash=c_hash, notes="shuffled-label control",
    ))

    # fusion:shuffled
    clf_fu_sh = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf_fu_sh.fit(X_fusion_tr, y_train_shuffled[fusion_tr_mask])
    fu_sh_probs = np.full(len(test_items), np.nan)
    if np.any(fusion_te_mask):
        fu_sh_probs[fusion_te_mask] = clf_fu_sh.predict_proba(X_fusion_te)[:, 1]
    fsh_res = auroc_ci(y_test, fu_sh_probs, groups_test)
    rows.append(make_row(
        hypothesis="H1", dataset=dataset, split=split_name,
        condition="fusion:shuffled", metric="auroc",
        value=fsh_res.point, ci_low=fsh_res.ci_low, ci_high=fsh_res.ci_high,
        n_items=n_total_test - fsh_res.n_excluded, n_groups=n_groups_test, n_excluded=fsh_res.n_excluded,
        config_hash=c_hash, notes="shuffled-label control",
    ))

    if notes:
        for r in rows:
            if r.notes:
                r.notes = f"{r.notes}; {notes}"
            else:
                r.notes = notes

    # Append rows to scorecard
    target_scorecard = scorecard_path or "results/scorecard.jsonl"
    append_rows(rows, path=target_scorecard)

    print(f"\nScorecard rows computed for {dataset} ({len(rows)} conditions):")
    print(f"{'Condition':30s} {'Metric':12s} {'Value':>7s} {'CI':>18s} {'n':>6s} {'Excl':>5s}")
    print("-" * 84)
    for r in rows:
        val_str = f"{r.value:.3f}" if r.value is not None else "null"
        ci_str = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]" if r.ci_low is not None and r.ci_high is not None else "-"
        print(f"{r.condition:30s} {r.metric:12s} {val_str:>7s} {ci_str:>18s} {r.n_items:>6d} {r.n_excluded:>5d}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate probes on cached features and judge scores")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="test", help="Split name")
    parser.add_argument("--scorecard", default="results/scorecard.jsonl", help="Scorecard JSONL path")
    parser.add_argument("--notes", default="", help="Optional notes to append to all rows")
    args = parser.parse_args()

    run_probes(
        dataset=args.dataset,
        split_name=args.split,
        scorecard_path=args.scorecard,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
