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


HOLOASSIST_DATASETS = ("holoassist", "holoassist_audible")


def logit(prob: np.ndarray) -> np.ndarray:
    """Computes logit clipping P to [0.01, 0.99] per 03_eval_harness.md §6."""
    clipped = np.clip(prob, 0.01, 0.99)
    return np.log(clipped / (1.0 - clipped))


def step_prior_scores(
    train_items: List[Item],
    target_items: List[Item],
    train_labels: np.ndarray,
    loo: bool = False,
) -> np.ndarray:
    """Computes train-split P(correct | verb, noun) with hierarchical smoothing (m=5) per 03_eval_harness.md §6.

    Levels: base rate -> verb -> (verb, noun), m=5.
    If loo=True, each item in target_items (assumed to be train_items) has its own label
    removed at all three levels.
    """
    N = len(train_items)
    K = int(np.sum(train_labels))
    p0 = K / N if N > 0 else 0.5

    n_verb: Dict[str, int] = {}
    k_verb: Dict[str, int] = {}
    n_vn: Dict[Tuple[str, str], int] = {}
    k_vn: Dict[Tuple[str, str], int] = {}

    for it, y in zip(train_items, train_labels):
        v = it.meta.get("verb", "")
        n = it.meta.get("noun", "")
        vn = (v, n)
        n_verb[v] = n_verb.get(v, 0) + 1
        k_verb[v] = k_verb.get(v, 0) + int(y == 1)
        n_vn[vn] = n_vn.get(vn, 0) + 1
        k_vn[vn] = k_vn.get(vn, 0) + int(y == 1)

    scores = np.zeros(len(target_items), dtype=np.float64)

    if loo:
        for i, (it, y) in enumerate(zip(target_items, train_labels)):
            v = it.meta.get("verb", "")
            n = it.meta.get("noun", "")
            vn = (v, n)

            N_loo = N - 1
            K_loo = K - int(y == 1)
            p0_loo = K_loo / N_loo if N_loo > 0 else 0.5

            n_v_loo = n_verb.get(v, 0) - 1
            k_v_loo = k_verb.get(v, 0) - int(y == 1)
            p_v_loo = (k_v_loo + 5.0 * p0_loo) / (n_v_loo + 5.0)

            n_vn_loo = n_vn.get(vn, 0) - 1
            k_vn_loo = k_vn.get(vn, 0) - int(y == 1)
            p_vn_loo = (k_vn_loo + 5.0 * p_v_loo) / (n_vn_loo + 5.0)

            scores[i] = p_vn_loo
    else:
        for j, it in enumerate(target_items):
            v = it.meta.get("verb", "")
            n = it.meta.get("noun", "")
            vn = (v, n)

            if v in n_verb:
                p_v = (k_verb[v] + 5.0 * p0) / (n_verb[v] + 5.0)
            else:
                p_v = p0

            if vn in n_vn:
                p_vn = (k_vn[vn] + 5.0 * p_v) / (n_vn[vn] + 5.0)
            else:
                p_vn = p_v

            scores[j] = p_vn

    return scores


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
        "step_prior": {"levels": ["base", "verb", "verb+noun"], "m": 5, "loo_train": True},
        "ablation": "fusion_minus_action_fusion",
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
    act_point = None
    k_act_tr = int(np.sum(valid_act_tr))
    if k_act_tr > 0 and len(set(y_train[valid_act_tr])) > 1:
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
        act_point = act_res.point
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="action-probe", metric="auroc",
            value=act_res.point, ci_low=act_res.ci_low, ci_high=act_res.ci_high,
            n_items=n_total_test - act_res.n_excluded, n_groups=n_groups_test, n_excluded=act_res.n_excluded,
            config_hash=c_hash, notes="",
        ))
    else:
        act_test_probs = np.full(len(test_items), np.nan)
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="action-probe", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes=f"not run: 0 of {len(train_items)} items cached in {action_cache.cache_dir}",
        ))

    # 4. Condition: react-nonverbal
    k_re_tr = int(np.sum(valid_re_tr))
    if k_re_tr > 0 and len(set(y_train[valid_re_tr])) > 1:
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
    else:
        re_test_probs = np.full(len(test_items), np.nan)
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="react-nonverbal", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes=f"not run: 0 of {len(train_items)} items cached in {react_cache.cache_dir}",
        ))

    # 4b. HoloAssist diagnostic: react-nonverbal|spoke=1
    if dataset in HOLOASSIST_DATASETS:
        spoke_mask_te = np.array([bool(it.meta.get("spoke", 0)) for it in test_items], dtype=bool)
        n_spoke_items = int(np.sum(spoke_mask_te))
        spoke_groups = [groups_test[i] for i in range(len(test_items)) if spoke_mask_te[i]]
        n_spoke_groups = len(set(spoke_groups))

        if k_re_tr > 0 and n_spoke_items > 0 and len(set(y_train[valid_re_tr])) > 1:
            spoke_test_probs = np.full(n_spoke_items, np.nan)
            valid_re_spoke = valid_re_te[spoke_mask_te]
            if np.any(valid_re_spoke):
                X_re_spoke = X_react_test[spoke_mask_te][valid_re_spoke]
                X_re_spk_scaled = re_scaler.transform(X_re_spoke)
                spoke_test_probs[valid_re_spoke] = clf_re.predict_proba(X_re_spk_scaled)[:, 1]

            y_test_spoke = y_test[spoke_mask_te]
            valid_probs_mask = ~np.isnan(spoke_test_probs)
            if len(set(y_test_spoke[valid_probs_mask])) > 1:
                rspk_res = auroc_ci(y_test_spoke, spoke_test_probs, spoke_groups)
                rows.append(make_row(
                    hypothesis="H1", dataset=dataset, split=split_name,
                    condition="react-nonverbal|spoke=1", metric="auroc",
                    value=rspk_res.point, ci_low=rspk_res.ci_low, ci_high=rspk_res.ci_high,
                    n_items=n_spoke_items - rspk_res.n_excluded, n_groups=n_spoke_groups, n_excluded=rspk_res.n_excluded,
                    config_hash=c_hash, notes="diagnostic: test items with spoke=1",
                ))
            else:
                rows.append(make_row(
                    hypothesis="H1", dataset=dataset, split=split_name,
                    condition="react-nonverbal|spoke=1", metric="auroc",
                    value=None, ci_low=None, ci_high=None,
                    n_items=n_spoke_items, n_groups=n_spoke_groups, n_excluded=n_spoke_items,
                    config_hash=c_hash, notes="diagnostic: test items with spoke=1; metric undefined: only one class in spoke=1 subset",
                ))
        else:
            rows.append(make_row(
                hypothesis="H1", dataset=dataset, split=split_name,
                condition="react-nonverbal|spoke=1", metric="auroc",
                value=None, ci_low=None, ci_high=None,
                n_items=n_spoke_items, n_groups=n_spoke_groups, n_excluded=n_spoke_items,
                config_hash=c_hash, notes=f"diagnostic: test items with spoke=1; not run: react-nonverbal not run ({rows[-1].notes})",
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
    elif dataset in HOLOASSIST_DATASETS:
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

    # 5b. Condition: step-prior
    step_prior_point = None
    step_prior_test_scores = None
    step_prior_train_loo = None

    if dataset in HOLOASSIST_DATASETS:
        step_prior_train_loo = step_prior_scores(train_items, train_items, y_train, loo=True)
        step_prior_test_scores = step_prior_scores(train_items, test_items, y_train, loo=False)
        sp_res = auroc_ci(y_test, step_prior_test_scores, groups_test)
        step_prior_point = sp_res.point
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="step-prior", metric="auroc",
            value=sp_res.point, ci_low=sp_res.ci_low, ci_high=sp_res.ci_high,
            n_items=n_total_test - sp_res.n_excluded, n_groups=n_groups_test, n_excluded=sp_res.n_excluded,
            config_hash=c_hash, notes="text-only: train-split P(correct | verb, noun), hierarchical smoothing m=5",
        ))
    elif dataset == "oops":
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="step-prior", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="not run: context_text is constant on Oops! (one string for every item)",
        ))

    # 6. Condition: fusion
    fusion_tr_mask = valid_j_tr & valid_act_tr & valid_re_tr
    fusion_te_mask = valid_j_te & valid_act_te & valid_re_te

    fu_point = None
    fusion_test_probs = np.full(len(test_items), np.nan)
    fusion_notes = ""

    fusion_runnable = np.any(fusion_tr_mask) and len(set(y_train[fusion_tr_mask])) > 1
    if fusion_runnable:
        logit_j_tr = logit(judge_train_scores[fusion_tr_mask])
        if dataset in HOLOASSIST_DATASETS:
            logit_sp_tr = logit(step_prior_train_loo[fusion_tr_mask])
            X_fusion_tr_raw = np.hstack([
                logit_j_tr[:, None],
                logit_sp_tr[:, None],
                X_action_train[fusion_tr_mask],
                X_react_train[fusion_tr_mask],
            ])
        elif dataset == "oops":
            X_fusion_tr_raw = np.hstack([
                logit_j_tr[:, None],
                X_action_train[fusion_tr_mask],
                X_react_train[fusion_tr_mask],
            ])

        fusion_scaler = StandardScaler().fit(X_fusion_tr_raw)
        X_fusion_tr = fusion_scaler.transform(X_fusion_tr_raw)

        clf_fusion = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf_fusion.fit(X_fusion_tr, y_train[fusion_tr_mask])

        if np.any(fusion_te_mask):
            logit_j_te = logit(judge_test_scores[fusion_te_mask])
            if dataset in HOLOASSIST_DATASETS:
                logit_sp_te = logit(step_prior_test_scores[fusion_te_mask])
                X_fusion_te_raw = np.hstack([
                    logit_j_te[:, None],
                    logit_sp_te[:, None],
                    X_action_test[fusion_te_mask],
                    X_react_test[fusion_te_mask],
                ])
            elif dataset == "oops":
                X_fusion_te_raw = np.hstack([
                    logit_j_te[:, None],
                    X_action_test[fusion_te_mask],
                    X_react_test[fusion_te_mask],
                ])
            X_fusion_te = fusion_scaler.transform(X_fusion_te_raw)
            fusion_test_probs[fusion_te_mask] = clf_fusion.predict_proba(X_fusion_te)[:, 1]

        fu_res = auroc_ci(y_test, fusion_test_probs, groups_test)
        fu_point = fu_res.point
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="fusion", metric="auroc",
            value=fu_res.point, ci_low=fu_res.ci_low, ci_high=fu_res.ci_high,
            n_items=n_total_test - fu_res.n_excluded, n_groups=n_groups_test, n_excluded=fu_res.n_excluded,
            config_hash=c_hash, notes="",
        ))
    else:
        missing_reasons = []
        if not np.any(valid_j_tr):
            missing_reasons.append("judge not run on train")
        if not np.any(valid_act_tr):
            missing_reasons.append("action-probe not run on train")
        if not np.any(valid_re_tr):
            missing_reasons.append("react-nonverbal not run on train")
        reason_str = ", ".join(missing_reasons) if missing_reasons else "no overlapping items across modalities"
        fusion_notes = f"not run: {reason_str}"
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="fusion", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes=fusion_notes,
        ))

    # 6b. Condition: action-fusion (fusion's inputs minus the reaction features)
    # When fusion can run, fit on exactly fusion_tr_mask and score on fusion_te_mask.
    # When fusion cannot run, fit on valid_j & valid_act.
    act_fu_point = None
    act_fu_test_probs = np.full(len(test_items), np.nan)

    if fusion_runnable:
        act_fu_tr_mask = fusion_tr_mask
        act_fu_te_mask = fusion_te_mask
    else:
        act_fu_tr_mask = valid_j_tr & valid_act_tr
        act_fu_te_mask = valid_j_te & valid_act_te

    act_fu_runnable = np.any(act_fu_tr_mask) and len(set(y_train[act_fu_tr_mask])) > 1

    if act_fu_runnable:
        logit_j_tr_af = logit(judge_train_scores[act_fu_tr_mask])
        if dataset in HOLOASSIST_DATASETS:
            logit_sp_tr_af = logit(step_prior_train_loo[act_fu_tr_mask])
            X_act_fu_tr_raw = np.hstack([
                logit_j_tr_af[:, None],
                logit_sp_tr_af[:, None],
                X_action_train[act_fu_tr_mask],
            ])
        elif dataset == "oops":
            X_act_fu_tr_raw = np.hstack([
                logit_j_tr_af[:, None],
                X_action_train[act_fu_tr_mask],
            ])

        act_fu_scaler = StandardScaler().fit(X_act_fu_tr_raw)
        X_act_fu_tr = act_fu_scaler.transform(X_act_fu_tr_raw)

        clf_act_fu = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf_act_fu.fit(X_act_fu_tr, y_train[act_fu_tr_mask])

        if np.any(act_fu_te_mask):
            logit_j_te_af = logit(judge_test_scores[act_fu_te_mask])
            if dataset in HOLOASSIST_DATASETS:
                logit_sp_te_af = logit(step_prior_test_scores[act_fu_te_mask])
                X_act_fu_te_raw = np.hstack([
                    logit_j_te_af[:, None],
                    logit_sp_te_af[:, None],
                    X_action_test[act_fu_te_mask],
                ])
            elif dataset == "oops":
                X_act_fu_te_raw = np.hstack([
                    logit_j_te_af[:, None],
                    X_action_test[act_fu_te_mask],
                ])
            X_act_fu_te = act_fu_scaler.transform(X_act_fu_te_raw)
            act_fu_test_probs[act_fu_te_mask] = clf_act_fu.predict_proba(X_act_fu_te)[:, 1]

        act_fu_res = auroc_ci(y_test, act_fu_test_probs, groups_test)
        act_fu_point = act_fu_res.point
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="action-fusion", metric="auroc",
            value=act_fu_res.point, ci_low=act_fu_res.ci_low, ci_high=act_fu_res.ci_high,
            n_items=n_total_test - act_fu_res.n_excluded, n_groups=n_groups_test, n_excluded=act_fu_res.n_excluded,
            config_hash=c_hash, notes="",
        ))
    else:
        missing_reasons = []
        if not np.any(valid_j_tr):
            missing_reasons.append("judge not run on train")
        if not np.any(valid_act_tr):
            missing_reasons.append("action-probe not run on train")
        reason_str = ", ".join(missing_reasons) if missing_reasons else "no overlapping items across modalities"
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="action-fusion", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes=f"not run: {reason_str}",
        ))

    # 6c. Condition: fusion_minus_action_fusion (ablation Delta)
    if fusion_runnable and act_fu_runnable and fu_point is not None and act_fu_point is not None:
        d_abl_res = delta_auroc_ci(y_test, act_fu_test_probs, fusion_test_probs, groups_test)
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="fusion_minus_action_fusion", metric="delta_auroc",
            value=d_abl_res.point, ci_low=d_abl_res.ci_low, ci_high=d_abl_res.ci_high,
            n_items=n_total_test - d_abl_res.n_excluded, n_groups=n_groups_test, n_excluded=d_abl_res.n_excluded,
            config_hash=c_hash, notes="ablation: reaction features removed",
        ))
    else:
        fu_reason = fusion_notes or "fusion not run"
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="fusion_minus_action_fusion", metric="delta_auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes=f"not run: fusion not run ({fu_reason})",
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
    if act_point is not None:
        action_candidates.append(("action-probe", act_test_probs, act_point))
    if dataset in HOLOASSIST_DATASETS and step_prior_point is not None:
        action_candidates.append(("step-prior", step_prior_test_scores, step_prior_point))
    if act_fu_point is not None:
        action_candidates.append(("action-fusion", act_fu_test_probs, act_fu_point))

    valid_candidates = [c for c in action_candidates if c[2] is not None]
    if valid_candidates:
        best_candidate = max(valid_candidates, key=lambda c: c[2])
        action_best_name, action_best_scores, _ = best_candidate
    else:
        action_best_name = "none"
        action_best_scores = np.full(len(test_items), np.nan)

    if fu_point is not None and valid_candidates:
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
    else:
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="fusion_minus_action_best", metric="delta_auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes=f"action_best={action_best_name}; not run: fusion not run" if valid_candidates else "not run",
        ))

    # 8. Shuffled-label controls (permuted train labels)
    rng = np.random.default_rng(0)
    y_train_shuffled = y_train.copy()
    rng.shuffle(y_train_shuffled)

    # action-probe:shuffled
    if k_act_tr > 0 and len(set(y_train[valid_act_tr])) > 1:
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
    else:
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="action-probe:shuffled", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="shuffled-label control; not run: action-probe not run",
        ))

    # react-nonverbal:shuffled
    if k_re_tr > 0 and len(set(y_train[valid_re_tr])) > 1:
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
    else:
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="react-nonverbal:shuffled", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="shuffled-label control; not run: react-nonverbal not run",
        ))

    # step-prior:shuffled (recomputed from y_train_shuffled)
    if dataset in HOLOASSIST_DATASETS:
        sp_sh_train_loo = step_prior_scores(train_items, train_items, y_train_shuffled, loo=True)
        sp_sh_test_scores = step_prior_scores(train_items, test_items, y_train_shuffled, loo=False)
        spsh_res = auroc_ci(y_test, sp_sh_test_scores, groups_test)
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="step-prior:shuffled", metric="auroc",
            value=spsh_res.point, ci_low=spsh_res.ci_low, ci_high=spsh_res.ci_high,
            n_items=n_total_test - spsh_res.n_excluded, n_groups=n_groups_test, n_excluded=spsh_res.n_excluded,
            config_hash=c_hash, notes="shuffled-label control",
        ))
    elif dataset == "oops":
        sp_sh_train_loo = None
        sp_sh_test_scores = None
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="step-prior:shuffled", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="shuffled-label control; not run: context_text is constant on Oops! (one string for every item)",
        ))

    # action-fusion:shuffled (step-prior recomputed from y_train_shuffled)
    if act_fu_runnable:
        logit_j_tr_af = logit(judge_train_scores[act_fu_tr_mask])
        if dataset in HOLOASSIST_DATASETS:
            logit_sp_tr_af_sh = logit(sp_sh_train_loo[act_fu_tr_mask])
            X_act_fu_tr_sh_raw = np.hstack([
                logit_j_tr_af[:, None],
                logit_sp_tr_af_sh[:, None],
                X_action_train[act_fu_tr_mask],
            ])
        elif dataset == "oops":
            X_act_fu_tr_sh_raw = np.hstack([
                logit_j_tr_af[:, None],
                X_action_train[act_fu_tr_mask],
            ])

        act_fu_sh_scaler = StandardScaler().fit(X_act_fu_tr_sh_raw)
        X_act_fu_tr_sh = act_fu_sh_scaler.transform(X_act_fu_tr_sh_raw)

        clf_act_fu_sh = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf_act_fu_sh.fit(X_act_fu_tr_sh, y_train_shuffled[act_fu_tr_mask])

        act_fu_sh_probs = np.full(len(test_items), np.nan)
        if np.any(act_fu_te_mask):
            logit_j_te_af = logit(judge_test_scores[act_fu_te_mask])
            if dataset in HOLOASSIST_DATASETS:
                logit_sp_te_af_sh = logit(sp_sh_test_scores[act_fu_te_mask])
                X_act_fu_te_sh_raw = np.hstack([
                    logit_j_te_af[:, None],
                    logit_sp_te_af_sh[:, None],
                    X_action_test[act_fu_te_mask],
                ])
            elif dataset == "oops":
                X_act_fu_te_sh_raw = np.hstack([
                    logit_j_te_af[:, None],
                    X_action_test[act_fu_te_mask],
                ])
            X_act_fu_te_sh = act_fu_sh_scaler.transform(X_act_fu_te_sh_raw)
            act_fu_sh_probs[act_fu_te_mask] = clf_act_fu_sh.predict_proba(X_act_fu_te_sh)[:, 1]

        afush_res = auroc_ci(y_test, act_fu_sh_probs, groups_test)
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="action-fusion:shuffled", metric="auroc",
            value=afush_res.point, ci_low=afush_res.ci_low, ci_high=afush_res.ci_high,
            n_items=n_total_test - afush_res.n_excluded, n_groups=n_groups_test, n_excluded=afush_res.n_excluded,
            config_hash=c_hash, notes="shuffled-label control",
        ))
    else:
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="action-fusion:shuffled", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="shuffled-label control; not run: action-fusion not run",
        ))

    # fusion:shuffled (step-prior recomputed from y_train_shuffled)
    if fusion_runnable:
        logit_j_tr = logit(judge_train_scores[fusion_tr_mask])
        if dataset in HOLOASSIST_DATASETS:
            logit_sp_tr_sh = logit(sp_sh_train_loo[fusion_tr_mask])
            X_fusion_tr_sh_raw = np.hstack([
                logit_j_tr[:, None],
                logit_sp_tr_sh[:, None],
                X_action_train[fusion_tr_mask],
                X_react_train[fusion_tr_mask],
            ])
        elif dataset == "oops":
            X_fusion_tr_sh_raw = np.hstack([
                logit_j_tr[:, None],
                X_action_train[fusion_tr_mask],
                X_react_train[fusion_tr_mask],
            ])

        fu_sh_scaler = StandardScaler().fit(X_fusion_tr_sh_raw)
        X_fu_tr_sh = fu_sh_scaler.transform(X_fusion_tr_sh_raw)

        clf_fu_sh = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf_fu_sh.fit(X_fu_tr_sh, y_train_shuffled[fusion_tr_mask])

        fu_sh_probs = np.full(len(test_items), np.nan)
        if np.any(fusion_te_mask):
            logit_j_te = logit(judge_test_scores[fusion_te_mask])
            if dataset in HOLOASSIST_DATASETS:
                logit_sp_te_sh = logit(sp_sh_test_scores[fusion_te_mask])
                X_fusion_te_sh_raw = np.hstack([
                    logit_j_te[:, None],
                    logit_sp_te_sh[:, None],
                    X_action_test[fusion_te_mask],
                    X_react_test[fusion_te_mask],
                ])
            elif dataset == "oops":
                X_fusion_te_sh_raw = np.hstack([
                    logit_j_te[:, None],
                    X_action_test[fusion_te_mask],
                    X_react_test[fusion_te_mask],
                ])
            X_fu_te_sh = fu_sh_scaler.transform(X_fusion_te_sh_raw)
            fu_sh_probs[fusion_te_mask] = clf_fu_sh.predict_proba(X_fu_te_sh)[:, 1]

        fsh_res = auroc_ci(y_test, fu_sh_probs, groups_test)
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="fusion:shuffled", metric="auroc",
            value=fsh_res.point, ci_low=fsh_res.ci_low, ci_high=fsh_res.ci_high,
            n_items=n_total_test - fsh_res.n_excluded, n_groups=n_groups_test, n_excluded=fsh_res.n_excluded,
            config_hash=c_hash, notes="shuffled-label control",
        ))
    else:
        rows.append(make_row(
            hypothesis="H1", dataset=dataset, split=split_name,
            condition="fusion:shuffled", metric="auroc",
            value=None, ci_low=None, ci_high=None,
            n_items=n_total_test, n_groups=n_groups_test, n_excluded=n_total_test,
            config_hash=c_hash, notes="shuffled-label control; not run: fusion not run",
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
