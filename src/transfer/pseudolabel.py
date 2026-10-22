from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import DATA_ROOT
from features.cache import FeatureCache
from harness.items import read_items
from harness.metrics import auroc_ci
from harness.probes import load_features_for_items
from harness.scorecard import ScorecardRow, append_rows, config_hash, make_row
from harness.splits import load_split


class ReactionLabeler:
    """Reaction labeler fit on holoassist_audible train split."""

    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        coef: np.ndarray,
        intercept: float,
    ) -> None:
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float32)
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = float(intercept)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = (X_arr - self.scaler_mean) / self.scaler_scale
        z = X_scaled @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            coef=self.coef,
            intercept=np.array([self.intercept], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> ReactionLabeler:
        data = np.load(path)
        return cls(
            scaler_mean=data["scaler_mean"],
            scaler_scale=data["scaler_scale"],
            coef=data["coef"],
            intercept=float(data["intercept"][0]),
        )


def fit_labeler(X: np.ndarray, y: np.ndarray) -> ReactionLabeler:
    """Fit StandardScaler and balanced LogisticRegression(C=1.0)."""
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0)
    clf.fit(X_scaled, y)
    return ReactionLabeler(
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        coef=clf.coef_[0],
        intercept=float(clf.intercept_[0]),
    )


def score_pool_items(
    labeler: ReactionLabeler,
    items: List[Any],
    X: np.ndarray,
    test_groups: Set[str],
) -> List[Dict[str, Any]]:
    """Score items with labeler and set labeler_unseen_group flag."""
    probs = labeler.predict_proba(X)
    results: List[Dict[str, Any]] = []
    for it, q in zip(items, probs):
        gid = it.group_id if hasattr(it, "group_id") else it["group_id"]
        iid = it.item_id if hasattr(it, "item_id") else it["item_id"]
        lbl = it.label if hasattr(it, "label") else it["label"]
        is_unseen = gid in test_groups
        results.append({
            "item_id": iid,
            "q": float(q),
            "label": int(lbl),
            "group_id": gid,
            "labeler_unseen_group": bool(is_unseen),
        })
    return results


def run_pseudolabeling(
    data_root: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
    scorecard_path: Optional[Union[str, Path]] = None,
) -> Tuple[ReactionLabeler, List[ScorecardRow]]:
    root = Path(data_root or DATA_ROOT)
    splits_path = Path(splits_dir or "splits")
    sc_path = Path(scorecard_path or "results/scorecard.jsonl")

    # 1. Load holoassist_audible items and split
    ha_items_path = root / "items" / "holoassist_audible" / "items.jsonl"
    if not ha_items_path.exists():
        raise FileNotFoundError(f"Items not found: {ha_items_path}")
    ha_items = read_items(ha_items_path)
    ha_map = {it.item_id: it for it in ha_items}

    split_data = load_split("holoassist_audible", path_dir=splits_path)
    train_ids = split_data["train"]
    test_ids = split_data["test"]
    train_items = [ha_map[i] for i in train_ids if i in ha_map]
    test_items = [ha_map[i] for i in test_ids if i in ha_map]

    y_train = np.array([it.label for it in train_items], dtype=int)
    y_test = np.array([it.label for it in test_items], dtype=int)
    groups_test = [it.group_id for it in test_items]
    test_groups_set = set(groups_test)

    # 2. Fit labeler on train split
    e2v_cache_ha = FeatureCache(dataset="holoassist_audible", encoder_id="e2v-plus-large", data_root=root)
    X_train, valid_tr, n_ex_tr = load_features_for_items(train_items, e2v_cache_ha)
    if n_ex_tr > 0 or not np.all(valid_tr):
        raise RuntimeError(f"holoassist_audible train split missing {n_ex_tr} e2v features")

    labeler = fit_labeler(X_train, y_train)

    # Save labeler
    labeler_path = root / "transfer" / "labeler.npz"
    labeler.save(labeler_path)
    print(f"Saved reaction labeler to {labeler_path}")

    # Evaluate on test split (reproduce §1.5 reference 0.6539 [0.6257, 0.6749])
    X_test, valid_te, n_ex_te = load_features_for_items(test_items, e2v_cache_ha)
    test_probs = labeler.predict_proba(X_test)
    test_res = auroc_ci(y_test, test_probs, groups_test)
    print(
        f"Labeler on holoassist_audible test split: AUROC = {test_res.point:.4f} "
        f"[{test_res.ci_low:.4f}, {test_res.ci_high:.4f}] (reference: 0.6539 [0.6257, 0.6749])"
    )

    # 3. Score pool items
    pool_items_path = root / "items" / "holoassist_pool" / "items.jsonl"
    if not pool_items_path.exists():
        raise FileNotFoundError(f"Pool items not found: {pool_items_path}")
    pool_items = read_items(pool_items_path)

    e2v_cache_pool = FeatureCache(dataset="holoassist_pool", encoder_id="e2v-plus-large", data_root=root)
    X_pool, valid_pool, n_ex_pool = load_features_for_items(pool_items, e2v_cache_pool)

    # Keep items with valid features
    valid_items = [it for it, v in zip(pool_items, valid_pool) if v]
    X_pool_valid = X_pool[valid_pool]

    if len(valid_items) == 0:
        raise RuntimeError("No pool items have e2v-plus-large features cached")

    scored_entries = score_pool_items(labeler, valid_items, X_pool_valid, test_groups_set)

    # Write pseudolabels JSONL
    out_jsonl = root / "transfer" / "holoassist_pool_pseudolabels.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry in scored_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(scored_entries)} pseudolabel records to {out_jsonl}")

    # 4. Compute quality metrics and append scorecard rows
    # All pool items
    labels_all = np.array([e["label"] for e in scored_entries], dtype=int)
    q_all = np.array([e["q"] for e in scored_entries], dtype=float)
    groups_all = [e["group_id"] for e in scored_entries]
    n_groups_all = len(set(groups_all))
    res_all = auroc_ci(labels_all, q_all, groups_all)

    # Unseen groups
    unseen_entries = [e for e in scored_entries if e["labeler_unseen_group"]]
    labels_unseen = np.array([e["label"] for e in unseen_entries], dtype=int)
    q_unseen = np.array([e["q"] for e in unseen_entries], dtype=float)
    groups_unseen = [e["group_id"] for e in unseen_entries]
    n_groups_unseen = len(set(groups_unseen))

    if len(set(labels_unseen)) >= 2:
        res_unseen = auroc_ci(labels_unseen, q_unseen, groups_unseen)
        val_unseen = res_unseen.point
        ci_lo_unseen = res_unseen.ci_low
        ci_hi_unseen = res_unseen.ci_high
        notes_unseen = "labeler_unseen_group: session never seen by reaction labeler"
    else:
        val_unseen = None
        ci_lo_unseen = None
        ci_hi_unseen = None
        notes_unseen = "not run: fewer than 2 classes in labeler_unseen_group (all items are correct actions; test split consumed all mistakes)"

    cfg = {
        "dataset": "holoassist_pool",
        "labeler_train_split": "holoassist_audible.train",
        "encoder": "e2v-plus-large",
    }
    c_hash = config_hash(cfg)

    row_unseen = make_row(
        hypothesis="H3",
        dataset="holoassist_pool",
        split="pool",
        condition="pseudo-label-quality",
        metric="auroc",
        value=val_unseen,
        ci_low=ci_lo_unseen,
        ci_high=ci_hi_unseen,
        n_items=len(unseen_entries),
        n_groups=n_groups_unseen,
        n_excluded=n_ex_pool,
        config_hash=c_hash,
        notes=notes_unseen,
    )

    row_all = make_row(
        hypothesis="H3",
        dataset="holoassist_pool",
        split="pool",
        condition="pseudo-label-quality:all",
        metric="auroc",
        value=res_all.point,
        ci_low=res_all.ci_low,
        ci_high=res_all.ci_high,
        n_items=len(scored_entries),
        n_groups=n_groups_all,
        n_excluded=n_ex_pool,
        config_hash=c_hash,
        notes="all scored pool items",
    )

    append_rows([row_unseen, row_all], sc_path)
    print(f"Appended pseudo-label-quality rows to {sc_path}:")
    if row_unseen.value is not None and row_unseen.ci_low is not None and row_unseen.ci_high is not None:
        print(f"  pseudo-label-quality: {row_unseen.value:.4f} [{row_unseen.ci_low:.4f}, {row_unseen.ci_high:.4f}], n={row_unseen.n_items}")
    else:
        print(f"  pseudo-label-quality: {row_unseen.value} ({row_unseen.notes}), n={row_unseen.n_items}")
    print(f"  pseudo-label-quality:all: {row_all.value:.4f} [{row_all.ci_low:.4f}, {row_all.ci_high:.4f}], n={row_all.n_items}")

    return labeler, [row_unseen, row_all]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reaction labeler and pseudolabel generator for HoloAssist pool")
    parser.add_argument("--data-root", default=None, help="Root data directory")
    parser.add_argument("--splits-dir", default=None, help="Splits directory")
    parser.add_argument("--scorecard", default=None, help="Scorecard path")
    args = parser.parse_args()

    run_pseudolabeling(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        scorecard_path=args.scorecard,
    )


if __name__ == "__main__":
    main()
