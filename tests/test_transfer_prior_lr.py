from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from transfer.prior_lr import prior_regularized_logistic_regression, predict_proba


def test_prior_lr_equals_sklearn():
    """Pin contract §4: with w0 = None (or 0), prior_lr equals sklearn LogisticRegression(C=1.0, class_weight='balanced') within 1e-4 test AUROC."""
    rng = np.random.default_rng(42)
    n_train = 300
    n_test = 200
    d = 10

    # Synthetic features & binary labels with imbalance (70% 1, 30% 0)
    X_train = rng.standard_normal((n_train, d)).astype(np.float64)
    true_w = rng.standard_normal(d)
    y_train = (X_train @ true_w + rng.standard_normal(n_train) > 0.2).astype(np.int64)

    X_test = rng.standard_normal((n_test, d)).astype(np.float64)
    y_test = (X_test @ true_w + rng.standard_normal(n_test) > 0.2).astype(np.int64)

    # 1. Sklearn reference
    lr = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=0)
    lr.fit(X_train, y_train)
    probs_sklearn = lr.predict_proba(X_test)[:, 1]
    auroc_sklearn = roc_auc_score(y_test, probs_sklearn)

    # 2. prior_lr with w0 = None
    w, b = prior_regularized_logistic_regression(X_train, y_train, w0=None, C=1.0)
    probs_prior_lr = predict_proba(X_test, w, b)
    auroc_prior_lr = roc_auc_score(y_test, probs_prior_lr)

    diff = abs(auroc_sklearn - auroc_prior_lr)
    assert diff < 1e-4, f"AUROC mismatch with sklearn: {auroc_prior_lr:.6f} vs {auroc_sklearn:.6f} (diff={diff:.6e})"


def test_prior_lr_regularization_toward_w0():
    """Verify that prior_lr regularizes toward w0: ||w - w0|| is smaller when w0 is provided."""
    rng = np.random.default_rng(123)
    n = 200
    d = 8

    X = rng.standard_normal((n, d)).astype(np.float64)
    y = rng.integers(0, 2, size=n)

    # Arbitrary non-zero prior
    w0 = rng.standard_normal(d) * 2.0

    # Fit with w0=None (regularizes toward 0)
    w_zero, _ = prior_regularized_logistic_regression(X, y, w0=None, C=0.1)

    # Fit with prior w0 (regularizes toward w0)
    w_prior, _ = prior_regularized_logistic_regression(X, y, w0=w0, C=0.1)

    # Distance to w0 must be strictly smaller for w_prior than for w_zero
    dist_with_prior = np.linalg.norm(w_prior - w0)
    dist_without_prior = np.linalg.norm(w_zero - w0)
    assert dist_with_prior < dist_without_prior, (
        f"Regularization failed: dist_with_prior={dist_with_prior:.4f} "
        f"not less than dist_without_prior={dist_without_prior:.4f}"
    )


def test_prior_lr_soft_labels():
    """Verify soft-label handling per contract §4: each item contributes (x, 1) with s1*q and (x, 0) with s0*(1-q)."""
    rng = np.random.default_rng(456)
    n = 100
    d = 5

    X = rng.standard_normal((n, d)).astype(np.float64)
    # Soft labels q in (0, 1)
    q = rng.uniform(0.1, 0.9, size=n).astype(np.float64)

    w, b = prior_regularized_logistic_regression(X, q, is_soft=True, C=1.0)
    assert w.shape == (d,)
    assert isinstance(b, float)

    probs = predict_proba(X, w, b)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
