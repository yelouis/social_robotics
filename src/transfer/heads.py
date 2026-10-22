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
