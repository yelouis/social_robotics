from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from transfer.pseudolabel import fit_labeler, score_pool_items


def test_fit_labeler_equals_sklearn():
    """Verify that fit_labeler reproduces StandardScaler + LogisticRegression(C=1.0, class_weight='balanced')."""
    rng = np.random.default_rng(42)
    n = 200
    d = 16
    X = rng.standard_normal((n, d)).astype(np.float32)
    # Binary labels with imbalance (70% 1, 30% 0)
    y = (rng.uniform(size=n) > 0.3).astype(np.int64)

    # Reference fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0)
    lr.fit(X_scaled, y)
    expected_probs = lr.predict_proba(X_scaled)[:, 1]

    # Function under test
    labeler = fit_labeler(X, y)
    probs = labeler.predict_proba(X)

    np.testing.assert_allclose(labeler.scaler_mean, scaler.mean_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(labeler.scaler_scale, scaler.scale_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(labeler.coef, lr.coef_[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(labeler.intercept, lr.intercept_[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(probs, expected_probs, rtol=1e-5, atol=1e-5)


def test_labeler_save_load_npz():
    """Verify labeler saves to npz and restores identically."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=50)

    labeler = fit_labeler(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "labeler.npz"
        labeler.save(npz_path)

        from transfer.pseudolabel import ReactionLabeler
        loaded = ReactionLabeler.load(npz_path)

        probs_orig = labeler.predict_proba(X)
        probs_loaded = loaded.predict_proba(X)
        np.testing.assert_array_equal(probs_orig, probs_loaded)


def test_labeler_unseen_group_flag():
    """Verify that labeler_unseen_group is True iff the group is in test_groups."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 4)).astype(np.float32)
    y = np.array([1, 0, 1, 0], dtype=np.int64)
    labeler = fit_labeler(X, y)

    items = [
        {"item_id": "holoassist_pool:s1:0", "label": 1, "group_id": "G_train_1"},
        {"item_id": "holoassist_pool:s2:0", "label": 0, "group_id": "G_test_1"},
        {"item_id": "holoassist_pool:s3:0", "label": 1, "group_id": "G_train_2"},
        {"item_id": "holoassist_pool:s4:0", "label": 0, "group_id": "G_test_2"},
    ]

    test_groups = {"G_test_1", "G_test_2"}

    results = score_pool_items(labeler, items, X, test_groups)

    assert len(results) == 4
    assert results[0]["labeler_unseen_group"] is False
    assert results[1]["labeler_unseen_group"] is True
    assert results[2]["labeler_unseen_group"] is False
    assert results[3]["labeler_unseen_group"] is True
    for r in results:
        assert 0.0 <= r["q"] <= 1.0
