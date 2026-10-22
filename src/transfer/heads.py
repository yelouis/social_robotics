from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from transfer.prior_lr import prior_regularized_logistic_regression


def fit_and_save_scaler(
    X_pool_siglip: np.ndarray,
    X_robot_train_u9: np.ndarray,
    out_path: Optional[Union[str, Path]] = None,
) -> StandardScaler:
    """Fit label-free StandardScaler on concatenation of pool SigLIP and robot train u9 features."""
    X_all = np.vstack([np.asarray(X_pool_siglip, dtype=np.float32), np.asarray(X_robot_train_u9, dtype=np.float32)])
    scaler = StandardScaler().fit(X_all)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, mean=scaler.mean_, scale=scaler.scale_)
        print(f"Saved transfer scaler to {p}")

    return scaler


def load_transfer_scaler(path: Union[str, Path]) -> StandardScaler:
    """Load StandardScaler from npz."""
    data = np.load(path)
    scaler = StandardScaler()
    scaler.mean_ = data["mean"]
    scaler.scale_ = data["scale"]
    scaler.var_ = data["scale"] ** 2
    scaler.n_features_in_ = len(data["mean"])
    return scaler


def fit_human_heads(
    X_pool_scaled: np.ndarray,
    q: np.ndarray,
    y_true: np.ndarray,
    seed: int = 0,
    out_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Fit the three human heads (w_react, w_shuffled, w_truth) per contract §4."""
    X_scaled = np.asarray(X_pool_scaled, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.int64)

    # 1. w_react (soft labels q)
    w_react, b_react = prior_regularized_logistic_regression(
        X_scaled, q_arr, is_soft=True, w0=None, C=1.0
    )

    # 2. w_shuffled (q permuted with default_rng(seed))
    rng = np.random.default_rng(seed)
    q_shuffled = rng.permutation(q_arr)
    w_shuffled, b_shuffled = prior_regularized_logistic_regression(
        X_scaled, q_shuffled, is_soft=True, w0=None, C=1.0
    )

    # 3. w_truth (true hard balanced labels)
    w_truth, b_truth = prior_regularized_logistic_regression(
        X_scaled, y_arr, is_soft=False, w0=None, C=1.0
    )

    heads = {
        "react": (w_react, b_react),
        "shuffled": (w_shuffled, b_shuffled),
        "truth": (w_truth, b_truth),
    }

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            w_react=w_react,
            b_react=b_react,
            w_shuffled=w_shuffled,
            b_shuffled=b_shuffled,
            w_truth=w_truth,
            b_truth=b_truth,
        )
        print(f"Saved human heads to {p}")

    return heads


def load_human_heads(path: Union[str, Path]) -> Dict[str, Tuple[np.ndarray, float]]:
    """Load human heads from npz."""
    data = np.load(path)
    return {
        "react": (data["w_react"], float(data["b_react"])),
        "shuffled": (data["w_shuffled"], float(data["b_shuffled"])),
        "truth": (data["w_truth"], float(data["b_truth"])),
    }


def ensure_scaler_and_heads(data_root: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
    """Ensure transfer scaler and human heads exist on disk, fitting them if missing."""
    from config import DATA_ROOT
    from features.cache import FeatureCache
    from harness.items import read_items
    from harness.probes import load_features_for_items
    import json

    root = Path(data_root or DATA_ROOT)
    scaler_path = root / "transfer" / "scaler.npz"
    heads_path = root / "transfer" / "heads.npz"

    if scaler_path.exists() and heads_path.exists():
        return scaler_path, heads_path

    # 1. Pool items and features
    pool_items_path = root / "items" / "holoassist_pool" / "items.jsonl"
    if not pool_items_path.exists():
        raise FileNotFoundError(f"Missing pool items: {pool_items_path}")
    pool_items = read_items(pool_items_path)
    pool_cache = FeatureCache(dataset="holoassist_pool", encoder_id="siglip-b16-224", data_root=root)
    X_pool_raw, valid_p, _ = load_features_for_items(pool_items, pool_cache)
    if not np.all(valid_p):
        raise RuntimeError("Missing pool siglip features")

    # 2. Robot train pool items and features
    stats_path = root / "items" / "roboreward" / "stats.json"
    robot_items_path = root / "items" / "roboreward" / "items.jsonl"
    if not stats_path.exists() or not robot_items_path.exists():
        raise FileNotFoundError("Missing roboreward items or stats")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_pool_ids = stats["train_pool_ids"]
    all_robot_items = read_items(robot_items_path)
    robot_map = {it.item_id: it for it in all_robot_items}
    ordered_train_pool_items = [robot_map[i] for i in train_pool_ids]

    u9_cache = FeatureCache(dataset="roboreward", encoder_id="siglip-b16-224-u9", data_root=root)
    X_robot_train, valid_tr, _ = load_features_for_items(ordered_train_pool_items, u9_cache)
    if not np.all(valid_tr):
        raise RuntimeError("Missing robot train pool u9 features")

    # 3. Fit scaler if missing
    if not scaler_path.exists():
        print(f"Fitting transfer scaler on pool ({len(pool_items)}) and robot train ({len(ordered_train_pool_items)})...")
        scaler = fit_and_save_scaler(X_pool_raw, X_robot_train, out_path=scaler_path)
    else:
        scaler = load_transfer_scaler(scaler_path)

    # 4. Fit human heads if missing
    if not heads_path.exists():
        print(f"Fitting human heads (react, shuffled, truth) on {len(pool_items)} pool items...")
        X_pool_scaled = scaler.transform(X_pool_raw)
        pseudolabels_path = root / "transfer" / "holoassist_pool_pseudolabels.jsonl"
        if not pseudolabels_path.exists():
            raise FileNotFoundError(f"Missing {pseudolabels_path}")
        q_list = []
        y_list = []
        for line in pseudolabels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                q_list.append(float(d["q"]))
                y_list.append(int(d["label"]))
        q_arr = np.array(q_list, dtype=np.float64)
        y_arr = np.array(y_list, dtype=np.int64)

        fit_human_heads(X_pool_scaled, q=q_arr, y_true=y_arr, seed=0, out_path=heads_path)

    return scaler_path, heads_path
