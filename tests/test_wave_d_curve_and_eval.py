from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import DATA_ROOT
from harness.items import read_items
from harness.probes import load_features_for_items
from harness.splits import load_split
from features.cache import FeatureCache

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.report_wave_d import h3_verdict  # noqa: E402


@pytest.mark.slow
def test_real_data_prior_lr_sklearn_equivalence():
    """Spot check per contract: for draw (N=1000, r=0), prior_lr with w0=None

    matches sklearn LogisticRegression(C=1.0, class_weight="balanced") within 1e-4 AUROC.
    """
    from transfer.heads import ensure_scaler_and_heads, load_transfer_scaler
    from transfer.prior_lr import prior_regularized_logistic_regression, predict_proba

    items_path = DATA_ROOT / "items" / "roboreward" / "items.jsonl"
    stats_path = DATA_ROOT / "items" / "roboreward" / "stats.json"
    splits_path = Path("splits")

    assert items_path.exists() and stats_path.exists()

    all_items = read_items(items_path)
    item_map = {it.item_id: it for it in all_items}
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_pool_ids = stats["train_pool_ids"]

    split_data = load_split("roboreward", path_dir=splits_path)
    test_ids = split_data["test"]

    train_pool_items = [item_map[i] for i in train_pool_ids]
    test_items = [item_map[i] for i in test_ids]

    u9_cache = FeatureCache(dataset="roboreward", encoder_id="siglip-b16-224-u9", data_root=DATA_ROOT)
    X_train_raw, valid_tr, _ = load_features_for_items(train_pool_items, u9_cache)
    X_test_raw, valid_te, _ = load_features_for_items(test_items, u9_cache)
    assert np.all(valid_tr) and np.all(valid_te)

    # Ensure scaler exists and scale features
    ensure_scaler_and_heads(DATA_ROOT)
    scaler = load_transfer_scaler(DATA_ROOT / "transfer" / "scaler.npz")
    X_train_pool = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_train_pool = np.array([it.label for it in train_pool_items], dtype=int)
    y_test = np.array([it.label for it in test_items], dtype=int)

    succ_idx = np.where(y_train_pool == 1)[0]
    fail_idx = np.where(y_train_pool == 0)[0]

    # Draw for N=1000, r=0
    N = 1000
    r = 0
    rng = np.random.default_rng(1000 * N + r)
    draw_succ = rng.choice(succ_idx, size=N // 2, replace=False)
    draw_fail = rng.choice(fail_idx, size=N // 2, replace=False)
    draw_idx = np.concatenate([draw_succ, draw_fail])
    draw_idx.sort()

    X_draw = X_train_pool[draw_idx]
    y_draw = y_train_pool[draw_idx]

    # 1. prior_lr with w0=None
    w_opt, b_opt = prior_regularized_logistic_regression(X_draw, y_draw, w0=None, C=1.0)
    probs_prior_lr = predict_proba(X_test, w_opt, b_opt)
    auroc_prior_lr = roc_auc_score(y_test, probs_prior_lr)

    # 2. sklearn LogisticRegression(C=1.0, class_weight="balanced")
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit(X_draw, y_draw)
    probs_sklearn = clf.predict_proba(X_test)[:, 1]
    auroc_sklearn = roc_auc_score(y_test, probs_sklearn)

    diff = abs(auroc_prior_lr - auroc_sklearn)
    print(f"prior_lr AUROC: {auroc_prior_lr:.6f}, sklearn AUROC: {auroc_sklearn:.6f}, diff: {diff:.6e}")
    assert diff <= 1e-4, f"AUROC diff {diff} exceeds tolerance 1e-4"


def test_wave_d_scorecard_rows_and_leak_check():
    """Verify all 44 Wave D scorecard rows exist and leak check passes."""
    sc_file = Path("results/scorecard.jsonl")
    assert sc_file.exists(), "scorecard.jsonl does not exist"

    rows = []
    for line in sc_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("hypothesis") == "H3" and d.get("dataset") == "roboreward":
                rows.append(d)

    latest_rows = {r["condition"]: r for r in rows}

    budgets = [50, 100, 200, 500, 1000]
    priors = ["none", "react", "shuffled", "truth"]

    # 20 detector rows
    for b in budgets:
        for p in priors:
            cond = f"detector:{p}:N={b}"
            assert cond in latest_rows, f"Missing {cond}"
            assert latest_rows[cond]["metric"] == "auroc_mean20"
            assert latest_rows[cond]["value"] is not None

    # 15 gain rows
    for b in budgets:
        for p in ["react", "shuffled", "truth"]:
            cond = f"gain:{p}:N={b}"
            assert cond in latest_rows, f"Missing {cond}"
            assert latest_rows[cond]["metric"] == "delta_auroc_mean20"
            assert latest_rows[cond]["value"] is not None

    # 5 react_minus_shuffled rows
    for b in budgets:
        cond = f"react_minus_shuffled:N={b}"
        assert cond in latest_rows, f"Missing {cond}"
        assert latest_rows[cond]["metric"] == "delta_auroc_mean20"
        assert latest_rows[cond]["value"] is not None

    # Reference lines: robot-full, judge-large, roboreward-8b
    assert "robot-full" in latest_rows
    assert latest_rows["robot-full"]["value"] is not None

    assert "judge-large" in latest_rows
    assert latest_rows["judge-large"]["value"] is not None
    assert latest_rows["judge-large"]["n_items"] == 1517

    assert "roboreward-8b" in latest_rows
    assert latest_rows["roboreward-8b"]["value"] is None

    # Leak check: detector:none:N=1000:shuffled
    assert "detector:none:N=1000:shuffled" in latest_rows
    leak_row = latest_rows["detector:none:N=1000:shuffled"]
    assert leak_row["ci_low"] is not None
    assert leak_row["ci_low"] <= 0.5, f"Leak check FAILED: ci_low={leak_row['ci_low']} > 0.5"


def test_wave_d_report_consistency():
    """Verify that Wave D report exists, verdict matches h3_verdict, and numbers match scorecard."""
    reports = glob.glob("docs/evals/*_wave_d_robot_check.md")
    assert len(reports) > 0, "No Wave D report found in docs/evals/"

    latest_report = max(reports)
    content = Path(latest_report).read_text(encoding="utf-8")

    # Load scorecard rows
    sc_file = Path("results/scorecard.jsonl")
    h3_rows = {}
    for line in sc_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("hypothesis") == "H3" and d.get("dataset") == "roboreward":
                h3_rows[d["condition"]] = d

    verdict, text = h3_verdict(h3_rows)
    assert f"- **Pre-registered Verdict:** **{verdict}**" in content
    assert f"- **Interpretation:** {text}" in content
