from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from harness.metrics import (
    _item_auroc_ci,
    auroc_ci,
    delta_auroc_ci,
    drop_missing_scores,
    spearman_ci,
)
from harness.scorecard import (
    append_rows,
    config_hash,
    make_row,
    read_rows,
)


def test_auroc_equals_sklearn_fixed_arrays():
    # 3 fixed arrays
    cases = [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], ["g1", "g1", "g2", "g2"]),
        ([0, 1, 0, 1, 0, 1], [0.2, 0.9, 0.1, 0.8, 0.4, 0.7], ["g1", "g2", "g3", "g4", "g5", "g6"]),
        ([1, 0, 0, 1, 1], [0.9, 0.1, 0.2, 0.8, 0.7], ["g1", "g1", "g2", "g3", "g3"]),
    ]
    for y, score, groups in cases:
        expected = roc_auc_score(y, score)
        res = auroc_ci(y, score, groups, n_boot=50, seed=0)
        assert abs(res.point - expected) < 1e-9


def test_selftest_check_a_planted_signal():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 20)
    y = rng.binomial(1, 0.5, size=1200)
    score = y + 0.8 * rng.standard_normal(size=1200)
    res = auroc_ci(y, score, groups, n_boot=1000, seed=0)
    assert res.point >= 0.75
    assert res.ci_low > 0.5


def test_selftest_check_b_null():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 20)
    y = rng.binomial(1, 0.5, size=1200)
    _score_a = y + 0.8 * rng.standard_normal(size=1200)
    score_null = rng.standard_normal(size=1200)
    res = auroc_ci(y, score_null, groups, n_boot=1000, seed=0)
    assert res.ci_low <= 0.5 <= res.ci_high


def test_selftest_check_c_grouping_real():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 25)
    group_labels = rng.binomial(1, 0.5, size=40)
    group_offsets = group_labels + rng.standard_normal(size=40)
    y = np.repeat(group_labels, 25)
    group_offset_items = np.repeat(group_offsets, 25)
    score = group_offset_items + 0.1 * rng.standard_normal(size=1000)

    res_grouped = auroc_ci(y, score, groups, n_boot=1000, seed=0)
    res_item = _item_auroc_ci(y, score, n_boot=1000, seed=0)
    grouped_width = res_grouped.ci_high - res_grouped.ci_low
    item_width = res_item[2] - res_item[1]
    assert grouped_width >= 1.5 * item_width


def test_selftest_check_d_paired_delta():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 20)
    y = rng.binomial(1, 0.5, size=1200)
    score_a = y + 0.8 * rng.standard_normal(size=1200)
    score_b = score_a + 1.5 * y

    res_diff = delta_auroc_ci(y, score_a, score_b, groups, n_boot=1000, seed=0)
    assert res_diff.ci_low > 0

    res_same = delta_auroc_ci(y, score_a, score_a, groups, n_boot=1000, seed=0)
    assert res_same.point == 0.0
    assert res_same.ci_low == 0.0
    assert res_same.ci_high == 0.0


def test_nan_scores_excluded_and_counted():
    y = [0, 1, 0, 1, 1, 0]
    score = [0.1, np.nan, 0.2, None, 0.9, 0.3]
    groups = ["g1", "g1", "g2", "g2", "g3", "g3"]

    y_clean, s_clean, g_clean, n_excl = drop_missing_scores(y, score, groups)
    assert n_excl == 2
    assert len(y_clean) == 4
    assert len(s_clean) == 4
    assert len(g_clean) == 4

    res = auroc_ci(y, score, groups, n_boot=50, seed=0)
    assert res.n_excluded == 2


def test_append_rows_appends(tmp_path: Path):
    scorecard_path = tmp_path / "scorecard.jsonl"
    r1 = make_row("H1", "d", "test", "c1", "auroc", 0.7, 0.6, 0.8, 100, 10, 0, "hash1", "n1")
    r2 = make_row("H1", "d", "test", "c2", "auroc", 0.75, 0.65, 0.85, 100, 10, 0, "hash2", "n2")
    r3 = make_row("H1", "d", "test", "c3", "auroc", 0.8, 0.7, 0.9, 100, 10, 0, "hash3", "n3")

    append_rows([r1, r2], path=scorecard_path)
    assert len(read_rows(scorecard_path)) == 2

    append_rows([r3], path=scorecard_path)
    rows_read = read_rows(scorecard_path)
    assert len(rows_read) == 3
    assert rows_read[0].condition == "c1"
    assert rows_read[1].condition == "c2"
    assert rows_read[2].condition == "c3"


def test_config_hash_order_independent():
    c1 = {"b": 2, "a": 1, "c": {"y": 20, "x": 10}}
    c2 = {"a": 1, "c": {"x": 10, "y": 20}, "b": 2}
    h1 = config_hash(c1)
    h2 = config_hash(c2)
    assert h1 == h2
    assert len(h1) == 12


def test_spearman_ci():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [1.1, 2.2, 2.9, 4.1, 5.2, 5.9]
    groups = ["g1", "g1", "g2", "g2", "g3", "g3"]
    res = spearman_ci(x, y, groups, n_boot=50, seed=0)
    assert res.point > 0.9
    assert res.ci_low > 0.8

