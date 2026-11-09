"""Tests for tools/diag_reflex_v2.py (Item B5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from harness.items import Item
from harness.metrics import delta_auroc_ci
from tools.diag_reflex_v2 import (
    CORE_HUME_COLS,
    FACETORCH_COLS,
    GAZE_FEATURE_NAMES,
    ReflexFaceV2Encoder,
    build_window_dicts,
    compute_fusion_minus_action_best,
    compute_reflex_v2_diagnostics,
    get_reflex_face_v2_feature_names,
    select_action_best_scores,
)


def _write_synthetic_reflex_trial(trial_dir: Path) -> None:
    """Write synthetic gaze.csv, hume.csv, and facetorch.csv into trial_dir."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    frames = list(range(10, 30))
    labels = ["Task"] * 10 + ["Robot"] * 6 + ["Miscellaneous"] * 4
    gaze_df = pd.DataFrame({"Frame": frames, "Gaze": labels})
    gaze_df.to_csv(trial_dir / "gaze.csv", index=False)

    rng = np.random.default_rng(0)
    hume_data: Dict[str, object] = {"Frame": frames}
    for col in CORE_HUME_COLS:
        hume_data[col] = rng.uniform(0.0, 1.0, size=len(frames))
    pd.DataFrame(hume_data).to_csv(trial_dir / "hume.csv", index=False)

    ft_data: Dict[str, object] = {"Frame ID": [f - 1 for f in frames]}
    for col in FACETORCH_COLS:
        ft_data[col] = rng.normal(0.0, 1.0, size=len(frames))
    pd.DataFrame(ft_data).to_csv(trial_dir / "facetorch.csv", index=False)


def test_encoder_returns_47_finite_values_and_raises_on_empty_gaze(tmp_path: Path) -> None:
    """Fast check 1: ReflexFaceV2Encoder returns 47 finite values and raises RuntimeError on empty gaze."""
    trial_dir = tmp_path / "trial_01"
    _write_synthetic_reflex_trial(trial_dir)

    feat_names = get_reflex_face_v2_feature_names()
    assert len(feat_names) == 47
    assert len(GAZE_FEATURE_NAMES) == 9
    assert len(CORE_HUME_COLS) == 17
    assert len(FACETORCH_COLS) == 2

    enc = ReflexFaceV2Encoder()
    vec = enc.encode(trial_dir, 10, 29)
    assert vec.shape == (47,)
    assert vec.dtype == np.float32
    assert bool(np.all(np.isfinite(vec)))

    # Window [0, 5] has no gaze rows (frames are 10..29) -> must raise RuntimeError
    with pytest.raises(RuntimeError, match="no valid gaze frames"):
        enc.encode(trial_dir, 0, 5)


def test_equal_start_and_action_equal_end_match_episode_lengths() -> None:
    """Fast check 2: equal_start and action_equal_end give both items of an episode the same length."""
    items = [
        Item(
            item_id="reflex:ep1:pre",
            dataset="reflex",
            group_id="p01",
            video_path="/tmp/ep1/video.mp4",
            action_window_sec=[0.0, 2.3],
            reaction_window_sec=[0.0, 2.3],
            label=1,
            label_source="reflex_phase",
            context_text="pick cup",
            meta={
                "action_frames": [10, 45],  # length 36
                "reaction_frames": [10, 45],  # length 36
            },
        ),
        Item(
            item_id="reflex:ep1:fail",
            dataset="reflex",
            group_id="p01",
            video_path="/tmp/ep1/video.mp4",
            action_window_sec=[3.0, 4.5],
            reaction_window_sec=[3.0, 6.0],
            label=0,
            label_source="reflex_phase",
            context_text="pick cup",
            meta={
                "action_frames": [46, 69],  # length 24
                "reaction_frames": [46, 91],  # length 46
            },
        ),
    ]

    orig_w, start_w, end_w, act_end_w = build_window_dicts(items)

    # Per-item reaction window (orig_w) has different lengths (36 vs 46)
    orig_len_pre = orig_w["reflex:ep1:pre"][1] - orig_w["reflex:ep1:pre"][0] + 1
    orig_len_fail = orig_w["reflex:ep1:fail"][1] - orig_w["reflex:ep1:fail"][0] + 1
    assert orig_len_pre != orig_len_fail

    # equal_start gives both items of the episode the exact same length (min = 36)
    start_len_pre = start_w["reflex:ep1:pre"][1] - start_w["reflex:ep1:pre"][0] + 1
    start_len_fail = start_w["reflex:ep1:fail"][1] - start_w["reflex:ep1:fail"][0] + 1
    assert start_len_pre == start_len_fail == 36

    # equal_end also matches length (36)
    end_len_pre = end_w["reflex:ep1:pre"][1] - end_w["reflex:ep1:pre"][0] + 1
    end_len_fail = end_w["reflex:ep1:fail"][1] - end_w["reflex:ep1:fail"][0] + 1
    assert end_len_pre == end_len_fail == 36

    # action_equal_end gives both items of the episode the exact same length (min action = 24)
    act_len_pre = act_end_w["reflex:ep1:pre"][1] - act_end_w["reflex:ep1:pre"][0] + 1
    act_len_fail = act_end_w["reflex:ep1:fail"][1] - act_end_w["reflex:ep1:fail"][0] + 1
    assert act_len_pre == act_len_fail == 24


def test_action_best_helper_subtracts_best_control_not_action_fusion() -> None:
    """Fast check 3: action_best helper selects step-prior when it beats action-fusion, differing from delta_af."""
    rng = np.random.default_rng(0)
    y_test = np.array([0, 1] * 25, dtype=int)
    groups_test = [f"g{i // 5:02d}" for i in range(len(y_test))]

    # step-prior is nearly perfect; action-fusion is noisy; fusion is intermediate
    step_prior_scores = y_test * 0.85 + rng.uniform(0.0, 0.15, size=len(y_test))
    action_fusion_scores = y_test * 0.25 + rng.uniform(0.0, 0.75, size=len(y_test))
    fusion_scores = y_test * 0.45 + rng.uniform(0.0, 0.55, size=len(y_test))

    candidates = {
        "action-fusion": action_fusion_scores,
        "step-prior": step_prior_scores,
    }
    best_name, best_scores, best_auc = select_action_best_scores(y_test, groups_test, candidates)
    assert best_name == "step-prior"
    assert best_auc > 0.90
    np.testing.assert_allclose(best_scores, step_prior_scores)

    chosen_name, delta_ab = compute_fusion_minus_action_best(
        y_test=y_test,
        fusion_scores=fusion_scores,
        groups_test=groups_test,
        candidate_scores=candidates,
    )
    delta_af = delta_auroc_ci(y_test, action_fusion_scores, fusion_scores, groups_test)

    assert chosen_name == "step-prior"
    assert delta_ab.point < 0.0
    assert delta_af.point > 0.0
    assert abs(delta_ab.point - delta_af.point) > 0.15


@pytest.mark.slow
def test_diag_reflex_v2_json_matches_expected_values() -> None:
    """Slow check 4: tool's JSON matches all expected values in the review to three decimals."""
    out_path = Path("docs/evals/2026-11-08_reflex_v2_posthoc.json")
    computed = compute_reflex_v2_diagnostics()
    assert out_path.exists(), f"Missing {out_path}"
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert computed == saved

    assert saved["_status"] == (
        "exploratory, post hoc: features and windows chosen after about 80 test-set evaluations "
        "on the 17 REFLEX test participants (November 7-8, 2026); not pre-registered; never scorecard rows"
    )

    expected_points = {
        "reflex.v2_equal_start": 0.759,
        "reflex.v2_action_equal_end": 0.736,
        "reflex.v2_equal_end": 0.695,
        "reflex.v2_orig": 0.722,
        "reflex.v2_fusion_minus_action_fusion_equal_start": 0.056,
        "reflex.v2_fusion_minus_action_fusion_action_equal_end": 0.068,
        "reflex.v2_fusion_minus_action_fusion_equal_end": 0.029,
        "reflex.v2_fusion_minus_action_fusion_orig": 0.044,
        "reflex.v2_gaze_only_equal_start": 0.718,
        "reflex.v2_gaze_only_action_equal_end": 0.720,
        "reflex.v2_face_only_equal_start": 0.643,
        "reflex.v2_face_only_action_equal_end": 0.618,
        "reflex.v2_gaze_only_fusion_minus_action_fusion_equal_start": 0.038,
        "reflex.v2_gaze_only_fusion_minus_action_fusion_action_equal_end": 0.053,
        "reflex.v2_face_only_fusion_minus_action_fusion_equal_start": 0.012,
        "reflex.v2_face_only_fusion_minus_action_fusion_action_equal_end": 0.004,
        "reflex.placebo_early_vs_late_pre": 0.679,
        "reflex.onset_straddle_all": 0.598,
        "reflex.onset_straddle_gaze": 0.600,
        "reflex.onset_straddle_face": 0.495,
        "reflex.resplit_mean": 0.763,
        "reflex.resplit_p5": 0.723,
        "reflex.resplit_p95": 0.787,
        "reflex.robot_gaze_share_pre": 0.061,
        "reflex.robot_gaze_share_fail": 0.128,
        "reflex_resolve.v2_action_window": 0.696,
        "reflex_resolve.window_length": 0.652,
        "reflex_resolve.length_plus_v2_minus_length": 0.104,
        "reflex_resolve.v2_last10": 0.658,
        "reflex_resolve.v2_last20": 0.661,
        "reflex_resolve.step_prior": 0.937,
        "reflex_resolve.step_prior_plus_v2_minus_step_prior": -0.062,
        "reflex_resolve.v2_action_fusion": 0.743,
        "reflex_resolve.v2_fusion": 0.775,
        "reflex_resolve.v2_fusion_minus_action_fusion": 0.031,
        "reflex_resolve.v2_fusion_minus_action_best": -0.163,
        "reflex_resolve.v2_within_place": 0.505,
        "reflex_resolve.v2_step_auroc_pick": 0.852,
        "reflex_resolve.v2_step_auroc_place": 0.838,
        "reflex_resolve.v2_step_auroc_carry": 0.709,
        "reflex_resolve.nonverbal_v1_rule": 0.660,
        "reflex_resolve.nonverbal_participant_only": 0.501,
        "reflex_resolve.speech_mixed_share": 0.788,
    }
    for key, exp_val in expected_points.items():
        assert key in saved, f"Missing key {key}"
        assert round(float(saved[key]["value"]), 3) == exp_val, (
            f"Key {key}: got {saved[key]['value']} (rounded {round(float(saved[key]['value']), 3)}), expected {exp_val}"
        )

    assert saved["reflex_resolve.nonverbal_v1_rule"]["n_test"] == 140
    assert saved["reflex_resolve.nonverbal_participant_only"]["n_test"] == 66
    assert round(float(saved["reflex_resolve.nonverbal_participant_only"]["ci_low"]), 3) == 0.309
    assert round(float(saved["reflex_resolve.nonverbal_participant_only"]["ci_high"]), 3) == 0.734

    s3 = saved["reflex_resolve.nonverbal_participant_only_before_resolution"]
    assert s3["value"] is None
    assert s3["note"] == "not estimable: two unresolved test items"

    mixed = saved["reflex_resolve.speech_mixed_share"]
    assert mixed["n_mixed"] == 367
    assert mixed["n"] == 466

    assert saved["reflex.speech_rows_v1_rule"] == {"value": 1093}
    assert saved["reflex.speech_rows_robot_participant"] == {"value": 622}
    assert saved["reflex.speech_rows_participant"] == {"value": 471}
