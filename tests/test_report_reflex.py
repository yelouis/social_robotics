import json
from pathlib import Path
import tempfile

import numpy as np

from features.face import get_reflex_face_feature_names
from tools.report_reflex import (
    VERIFIED_EXAMPLE_DESCRIPTIONS,
    evaluate_kill_rule,
    extract_top_weights,
    generate_reflex_report,
    prediction_check_reflex,
)


def test_prediction_check_reflex_synthetic_held_met():
    """Clause 1 held, Clause 2 held, H1 pass bar met."""
    rows = {
        "react-face": {
            "value": 0.72,
            "ci_low": 0.67,
            "ci_high": 0.77,
            "notes": "",
        },
        "action-probe": {
            "value": 0.58,
            "ci_low": 0.54,
            "ci_high": 0.62,
            "notes": "",
        },
        "fusion_minus_action_best": {
            "value": 0.14,
            "ci_low": 0.09,
            "ci_high": 0.19,
            "notes": "action_best=action-probe",
        },
        "fusion_minus_action_fusion": {
            "value": 0.05,
            "ci_low": 0.02,
            "ci_high": 0.08,
            "notes": "ablation: reaction features removed",
        },
        "judge": {"value": 0.52, "ci_low": 0.49, "ci_high": 0.55},
        "judge-large": {"value": 0.55, "ci_low": 0.52, "ci_high": 0.58},
        "react-nonverbal": {"value": 0.51, "ci_low": 0.47, "ci_high": 0.55},
        "react-spoke": {"value": 0.50, "ci_low": 0.46, "ci_high": 0.54},
        "react-full": {"value": 0.52, "ci_low": 0.48, "ci_high": 0.56},
    }

    lines = prediction_check_reflex(rows, dataset="reflex")
    c1 = next(line for line in lines if "Clause 1" in line)
    c2 = next(line for line in lines if "Clause 2" in line)
    pb = next(line for line in lines if "H1 pass bar" in line)
    ctx = next(line for line in lines if "Context, no verdict" in line)

    assert "→ held" in c1
    assert "→ held" in c2
    assert "→ met" in pb
    assert "react-face" in c1
    assert "react-nonverbal" in ctx


def test_prediction_check_reflex_synthetic_not_held_not_met():
    """Clause 1 did not hold, Clause 2 did not hold, H1 pass bar not met."""
    rows = {
        "react-face": {
            "value": 0.55,
            "ci_low": 0.51,
            "ci_high": 0.59,
            "notes": "",
        },
        "action-probe": {
            "value": 0.60,
            "ci_low": 0.56,
            "ci_high": 0.64,
            "notes": "",
        },
        "fusion_minus_action_best": {
            "value": -0.05,
            "ci_low": -0.09,
            "ci_high": -0.01,
            "notes": "action_best=action-probe",
        },
        "fusion_minus_action_fusion": {
            "value": -0.01,
            "ci_low": -0.04,
            "ci_high": 0.02,
            "notes": "",
        },
    }

    lines = prediction_check_reflex(rows, dataset="reflex")
    c1 = next(line for line in lines if "Clause 1" in line)
    c2 = next(line for line in lines if "Clause 2" in line)
    pb = next(line for line in lines if "H1 pass bar" in line)

    assert "→ did not hold" in c1
    assert "→ did not hold" in c2
    assert "→ not met" in pb


def test_prediction_check_reflex_missing_null():
    """Null/missing rows -> not tested, not testable."""
    rows = {
        "react-face": None,
        "fusion_minus_action_fusion": {
            "value": None,
            "notes": "not run",
        },
    }

    lines = prediction_check_reflex(rows, dataset="reflex")
    c1 = next(line for line in lines if "Clause 1" in line)
    c2 = next(line for line in lines if "Clause 2" in line)
    pb = next(line for line in lines if "H1 pass bar" in line)

    assert "not tested" in c1
    assert "not tested" in c2
    assert "not testable" in pb


def test_evaluate_kill_rule():
    # 1. PROCEED via positive ablation on reflex
    status, msg = evaluate_kill_rule(
        reflex_rows={"fusion_minus_action_fusion": {"value": 0.04, "ci_low": 0.01}},
        reflex_resolve_rows={"react-face": {"value": 0.51, "ci_low": 0.48}},
    )
    assert status == "PROCEED"
    assert "positive ablation" in msg

    # 2. PROCEED via react-face beats chance on reflex_resolve
    status, msg = evaluate_kill_rule(
        reflex_rows={"fusion_minus_action_fusion": {"value": 0.01, "ci_low": -0.02}},
        reflex_resolve_rows={"react-face": {"value": 0.68, "ci_low": 0.63}},
    )
    assert status == "PROCEED"
    assert "beats chance" in msg

    # 3. KILL when neither positive ablation nor react-face beats chance on both
    status, msg = evaluate_kill_rule(
        reflex_rows={
            "react-face": {"value": 0.51, "ci_low": 0.47},
            "fusion_minus_action_fusion": {"value": 0.00, "ci_low": -0.03},
        },
        reflex_resolve_rows={
            "react-face": {"value": 0.52, "ci_low": 0.48},
            "fusion_minus_action_fusion": {"value": -0.01, "ci_low": -0.04},
        },
    )
    assert status == "KILL"

    # 4. PENDING when uncomputed
    status, msg = evaluate_kill_rule(reflex_rows={}, reflex_resolve_rows={})
    assert status == "PENDING"


def test_extract_top_weights():
    with tempfile.TemporaryDirectory() as td:
        probe_path = Path(td) / "test_probe.npz"
        coef = np.zeros((1, 153), dtype=np.float32)
        # Set specific coefficients
        coef[0, 0] = 2.5  # gaze_0_x_mean
        coef[0, 1] = -1.8  # gaze_0_y_mean
        coef[0, 10] = 3.2  # pose_Rx_mean
        coef[0, 152] = -4.0  # Arms_behind_back_max

        np.savez(probe_path, mean=np.zeros(153), scale=np.ones(153), coef=coef, intercept=np.zeros(1))

        top_pos, top_neg = extract_top_weights(probe_path)
        assert len(top_pos) == 5
        assert len(top_neg) == 5

        # Check largest positive
        feat_names = get_reflex_face_feature_names()
        assert top_pos[0][0] == feat_names[10]
        assert round(top_pos[0][1], 2) == 3.2

        # Check most negative
        assert top_neg[0][0] == feat_names[152]
        assert round(top_neg[0][1], 2) == -4.0


def test_generate_reflex_report_synthetic():
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        scorecard_file = tmp_dir / "scorecard.jsonl"
        data_root = tmp_dir / "data"
        splits_dir = tmp_dir / "splits"
        out_file = tmp_dir / "report.md"

        data_root.mkdir(parents=True)
        splits_dir.mkdir(parents=True)

        # Write mock scorecard rows for reflex and reflex_resolve
        mock_rows = [
            {
                "hypothesis": "H1", "dataset": "reflex", "split": "test",
                "condition": "react-face", "metric": "auroc",
                "value": 0.69, "ci_low": 0.64, "ci_high": 0.74,
                "n_items": 200, "n_groups": 17,
            },
            {
                "hypothesis": "H1", "dataset": "reflex", "split": "test",
                "condition": "action-probe", "metric": "auroc",
                "value": 0.58, "ci_low": 0.53, "ci_high": 0.63,
                "n_items": 200, "n_groups": 17,
            },
            {
                "hypothesis": "H1", "dataset": "reflex", "split": "test",
                "condition": "fusion_minus_action_best", "metric": "delta_auroc",
                "value": 0.11, "ci_low": 0.06, "ci_high": 0.16,
                "notes": "action_best=action-probe",
            },
            {
                "hypothesis": "H1", "dataset": "reflex", "split": "test",
                "condition": "fusion_minus_action_fusion", "metric": "delta_auroc",
                "value": 0.04, "ci_low": 0.01, "ci_high": 0.07,
            },
            {
                "hypothesis": "H1", "dataset": "reflex", "split": "test",
                "condition": "react-face:shuffled", "metric": "auroc",
                "value": 0.49, "ci_low": 0.45, "ci_high": 0.53,
            },
            {
                "hypothesis": "H1", "dataset": "reflex_resolve", "split": "test",
                "condition": "react-face", "metric": "auroc",
                "value": 0.66, "ci_low": 0.61, "ci_high": 0.71,
                "n_items": 150, "n_groups": 17,
            },
            {
                "hypothesis": "H1", "dataset": "reflex_resolve", "split": "test",
                "condition": "action-probe", "metric": "auroc",
                "value": 0.55, "ci_low": 0.50, "ci_high": 0.60,
                "n_items": 150, "n_groups": 17,
            },
            {
                "hypothesis": "H1", "dataset": "reflex_resolve", "split": "test",
                "condition": "fusion_minus_action_best", "metric": "delta_auroc",
                "value": 0.11, "ci_low": 0.05, "ci_high": 0.17,
                "notes": "action_best=action-probe",
            },
            {
                "hypothesis": "H1", "dataset": "reflex_resolve", "split": "test",
                "condition": "fusion_minus_action_fusion", "metric": "delta_auroc",
                "value": 0.03, "ci_low": 0.005, "ci_high": 0.06,
            },
        ]
        with open(scorecard_file, "w", encoding="utf-8") as f:
            for r in mock_rows:
                f.write(json.dumps(r) + "\n")

        # Mock probe npz
        transfer_dir = data_root / "transfer"
        transfer_dir.mkdir(parents=True)
        probe_path = transfer_dir / "reflex_face_probe.npz"
        np.savez(probe_path, mean=np.zeros(153), scale=np.ones(153), coef=np.ones((1, 153)), intercept=np.zeros(1))

        generate_reflex_report(
            date_str="2026-11-01",
            scorecard_path=scorecard_file,
            data_root=data_root,
            splits_dir=splits_dir,
            output_path=out_file,
        )

        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "REFLEX Multimodal Human-Robot Reactions" in content
        assert "Executive Summary" in content
        assert "Pre-Registered Prediction Checks & Pass Bar" in content
        assert "Clause 1" in content
        assert "Clause 2" in content
        assert "Scorecard Conditions Table" in content
        assert "Stage A MicroDuck Transfer Probes" in content
        assert "Compute & Wall-Clock Times" in content
        assert "Sampled Test Split Examples (Verified Visual Descriptions)" in content


def test_verified_example_descriptions_complete():
    assert len(VERIFIED_EXAMPLE_DESCRIPTIONS) == 20
    reflex_keys = [k for k in VERIFIED_EXAMPLE_DESCRIPTIONS if k.startswith("reflex:")]
    resolve_keys = [k for k in VERIFIED_EXAMPLE_DESCRIPTIONS if k.startswith("reflex_resolve:")]
    assert len(reflex_keys) == 10
    assert len(resolve_keys) == 10
    for k, desc in VERIFIED_EXAMPLE_DESCRIPTIONS.items():
        assert len(desc) > 50, f"Description too short for {k}"
        assert "not described" not in desc.lower()

