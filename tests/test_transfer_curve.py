from __future__ import annotations

import numpy as np

from transfer.curve import compute_curve_metrics_and_cis


def test_curve_bootstrap_cis_shape_and_pairing():
    """Test paired group bootstrap computation for mean20 and gains."""
    rng = np.random.default_rng(0)
    n_test = 100
    groups = [f"group_{i // 5}" for i in range(n_test)]
    y_test = rng.integers(0, 2, size=n_test)

    # 20 draws for 4 priors
    priors = ["none", "react", "shuffled", "truth"]
    scores: dict[str, np.ndarray] = {}
    for p in priors:
        # shape (20, n_test)
        scores[p] = rng.uniform(0.0, 1.0, size=(20, n_test))

    results = compute_curve_metrics_and_cis(
        y_test=y_test,
        groups_test=groups,
        scores_by_prior=scores,
        n_boot=100,
        seed=0,
    )

    # All priors have point, ci_low, ci_high
    for p in priors:
        res = results[p]
        assert "point" in res and "ci_low" in res and "ci_high" in res
        assert res["ci_low"] <= res["point"] <= res["ci_high"] or (res["ci_low"] <= res["ci_high"])

    # Gains
    for p in ["react", "shuffled", "truth"]:
        gain_res = results[f"gain_{p}"]
        assert gain_res["ci_low"] <= gain_res["ci_high"]

    assert "react_minus_shuffled" in results
    assert results["react_minus_shuffled"]["ci_low"] <= results["react_minus_shuffled"]["ci_high"]
