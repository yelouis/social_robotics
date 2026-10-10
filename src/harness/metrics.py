from __future__ import annotations

from typing import Any, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.stats


class MetricResult(tuple):
    """3-tuple of (point, ci_low, ci_high) with attached metadata."""

    point: float
    ci_low: float
    ci_high: float
    n_excluded: int

    def __new__(
        cls,
        point: float,
        ci_low: float,
        ci_high: float,
        n_excluded: int = 0,
    ) -> MetricResult:
        obj = super().__new__(cls, (float(point), float(ci_low), float(ci_high)))
        obj.point = float(point)
        obj.ci_low = float(ci_low)
        obj.ci_high = float(ci_high)
        obj.n_excluded = int(n_excluded)
        return obj


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    try:
        return bool(np.isnan(val))
    except (TypeError, ValueError):
        return False


def drop_missing_scores(
    y: Any,
    score: Any,
    groups: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)

    valid_mask = np.zeros(len(y_arr), dtype=bool)
    for i in range(len(y_arr)):
        s_val = score[i] if isinstance(score, (list, tuple)) else score[i]
        if _is_missing(s_val) or _is_missing(y_arr[i]):
            valid_mask[i] = False
        else:
            valid_mask[i] = True

    score_arr = np.asarray([score[i] for i in range(len(y_arr)) if valid_mask[i]], dtype=float)
    n_excluded = int((~valid_mask).sum())
    return y_arr[valid_mask].astype(int), score_arr, groups_arr[valid_mask], n_excluded


def auroc_ci(
    y: Any,
    score: Any,
    groups: Any,
    n_boot: int = 1000,
    seed: int = 0,
    return_excluded: bool = False,
) -> Union[MetricResult, Tuple[float, float, float, int]]:
    y_clean, score_clean, groups_clean, n_excluded = drop_missing_scores(y, score, groups)
    if len(y_clean) == 0:
        raise ValueError("Cannot compute AUROC: all items were excluded.")
    if len(np.unique(y_clean)) < 2:
        raise ValueError("Cannot compute AUROC: fewer than 2 classes present.")

    point = float(roc_auc_score(y_clean, score_clean))

    unique_groups = np.unique(groups_clean)
    n_groups = len(unique_groups)
    group_to_indices = {g: np.where(groups_clean == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    boot_values = []
    draws = 0
    while len(boot_values) < n_boot:
        draws += 1
        if draws > 10000:
            raise RuntimeError("bootstrap: >10000 draws needed for 1000 two-class resamples")
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        y_boot = y_clean[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        boot_values.append(roc_auc_score(y_boot, score_clean[idx]))

    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
    result = MetricResult(point, float(ci_low), float(ci_high), n_excluded=n_excluded)
    if return_excluded:
        return (result.point, result.ci_low, result.ci_high, result.n_excluded)
    return result


def _item_auroc_ci(
    y: Any,
    score: Any,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Private helper for self-test (c): item-level bootstrap."""
    y_clean, score_clean, _, _ = drop_missing_scores(y, score, np.arange(len(y)))
    point = float(roc_auc_score(y_clean, score_clean))
    rng = np.random.default_rng(seed)
    n_items = len(y_clean)
    boot_values = []
    draws = 0
    while len(boot_values) < n_boot:
        draws += 1
        if draws > 10000:
            raise RuntimeError("bootstrap: >10000 draws needed for 1000 two-class resamples")
        idx = rng.choice(n_items, size=n_items, replace=True)
        y_boot = y_clean[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        boot_values.append(roc_auc_score(y_boot, score_clean[idx]))
    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
    return (point, float(ci_low), float(ci_high))


def delta_auroc_ci(
    y: Any,
    score_a: Any,
    score_b: Any,
    groups: Any,
    n_boot: int = 1000,
    seed: int = 0,
    return_excluded: bool = False,
) -> Union[MetricResult, Tuple[float, float, float, int]]:
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)

    valid_mask = np.zeros(len(y_arr), dtype=bool)
    for i in range(len(y_arr)):
        sa = score_a[i] if isinstance(score_a, (list, tuple)) else score_a[i]
        sb = score_b[i] if isinstance(score_b, (list, tuple)) else score_b[i]
        if _is_missing(sa) or _is_missing(sb) or _is_missing(y_arr[i]):
            valid_mask[i] = False
        else:
            valid_mask[i] = True

    y_clean = y_arr[valid_mask].astype(int)
    groups_clean = groups_arr[valid_mask]
    score_a_clean = np.asarray([score_a[i] for i in range(len(y_arr)) if valid_mask[i]], dtype=float)
    score_b_clean = np.asarray([score_b[i] for i in range(len(y_arr)) if valid_mask[i]], dtype=float)
    n_excluded = int((~valid_mask).sum())

    if len(y_clean) == 0:
        raise ValueError("Cannot compute delta AUROC: all items were excluded.")
    if len(np.unique(y_clean)) < 2:
        raise ValueError("Cannot compute delta AUROC: fewer than 2 classes present.")

    point = float(roc_auc_score(y_clean, score_b_clean) - roc_auc_score(y_clean, score_a_clean))

    unique_groups = np.unique(groups_clean)
    n_groups = len(unique_groups)
    group_to_indices = {g: np.where(groups_clean == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    boot_values = []
    draws = 0
    while len(boot_values) < n_boot:
        draws += 1
        if draws > 10000:
            raise RuntimeError("bootstrap: >10000 draws needed for 1000 two-class resamples")
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        y_boot = y_clean[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        auc_b = roc_auc_score(y_boot, score_b_clean[idx])
        auc_a = roc_auc_score(y_boot, score_a_clean[idx])
        boot_values.append(auc_b - auc_a)

    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
    result = MetricResult(point, float(ci_low), float(ci_high), n_excluded=n_excluded)
    if return_excluded:
        return (result.point, result.ci_low, result.ci_high, result.n_excluded)
    return result


def spearman_ci(
    x: Any,
    y: Any,
    groups: Any,
    n_boot: int = 1000,
    seed: int = 0,
    return_excluded: bool = False,
) -> Union[MetricResult, Tuple[float, float, float, int]]:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)

    valid_mask = np.zeros(len(x_arr), dtype=bool)
    for i in range(len(x_arr)):
        xv = x_arr[i]
        yv = y_arr[i]
        if _is_missing(xv) or _is_missing(yv):
            valid_mask[i] = False
        else:
            valid_mask[i] = True

    x_clean = x_arr[valid_mask].astype(float)
    y_clean = y_arr[valid_mask].astype(float)
    groups_clean = groups_arr[valid_mask]
    n_excluded = int((~valid_mask).sum())

    if len(x_clean) == 0:
        raise ValueError("Cannot compute Spearman: all items were excluded.")

    point = float(scipy.stats.spearmanr(x_clean, y_clean).statistic)

    unique_groups = np.unique(groups_clean)
    n_groups = len(unique_groups)
    group_to_indices = {g: np.where(groups_clean == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    boot_values = []
    draws = 0
    while len(boot_values) < n_boot:
        draws += 1
        if draws > 10000:
            raise RuntimeError("bootstrap: >10000 draws needed for 1000 resamples")
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        r = float(scipy.stats.spearmanr(x_clean[idx], y_clean[idx]).statistic)
        if np.isnan(r):
            continue
        boot_values.append(r)

    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
    result = MetricResult(point, float(ci_low), float(ci_high), n_excluded=n_excluded)
    if return_excluded:
        return (result.point, result.ci_low, result.ci_high, result.n_excluded)
    return result
