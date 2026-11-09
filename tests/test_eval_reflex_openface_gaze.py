from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from harness.items import Item
from harness.metrics import MetricResult, auroc_ci
from harness.probes import step_prior_scores
from tools.eval_reflex_openface_gaze import (
    DEFAULT_OUT_PATH,
    OPENFACE_GAZE_COLS,
    SAME_PHASE_PAIRS,
    STATUS_TEXT,
    ReflexOpenFaceGazeEncoder,
    build_same_phase_items_and_windows,
    evaluate_confirmatory_verdict,
    evaluate_reflex_openface_gaze,
    get_openface_gaze_feature_names,
)


def _write_synthetic_openface_csv(pdir: Path) -> None:
    cols = ["frame", "face_id", "timestamp", "confidence", "success"] + list(OPENFACE_GAZE_COLS)
    rows = [
        # frame 1: valid
        [1, 0, 0.0, 0.95, 1, 0.10, -0.20, -0.90, 0.12, -0.18, -0.92, 0.11, -0.19],
        # frame 2: invalid (success=0) -> must be ignored even with extreme values
        [2, 0, 0.2, 0.99, 0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        # frame 3: invalid (confidence < 0.7) -> must be ignored
        [3, 0, 0.4, 0.50, 1, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0],
        # frame 4: valid
        [4, 0, 0.6, 0.85, 1, 0.20, -0.10, -0.80, 0.22, -0.08, -0.82, 0.21, -0.09],
        # frame 5: valid
        [5, 0, 0.8, 0.90, 1, 0.30, 0.00, -0.70, 0.32, 0.02, -0.72, 0.31, 0.01],
        # frame 6: valid
        [6, 0, 1.0, 0.92, 1, 0.40, 0.10, -0.60, 0.42, 0.12, -0.62, 0.41, 0.11],
        # frame 25: invalid in [20, 30]
        [25, 0, 5.0, 0.30, 0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    ]
    lines = [", ".join(cols)]
    for r in rows:
        lines.append(", ".join(str(x) for x in r))
    (pdir / "openface.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_openface_gaze_encoder_synthetic_and_honest_null(tmp_path: Path) -> None:
    _write_synthetic_openface_csv(tmp_path)
    names = get_openface_gaze_feature_names()
    assert len(names) == 26

    enc = ReflexOpenFaceGazeEncoder()
    vec = enc.encode(tmp_path, 1, 10)
    assert vec.shape == (26,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))

    # Mean of gaze_0_x over the 4 valid frames (0.10, 0.20, 0.30, 0.40) is 0.25
    assert vec[0] == pytest.approx(0.25, abs=1e-5)
    # Delta (second half [0.30, 0.40] minus first half [0.10, 0.20]) is +0.20
    assert vec[16] == pytest.approx(0.20, abs=1e-5)
    # Angular step speed between consecutive valid frames has dx=0.10, dy=0.10 -> sqrt(0.02)
    assert vec[24] == pytest.approx(float(np.sqrt(0.02)), abs=1e-5)
    assert vec[25] == pytest.approx(0.0, abs=1e-5)

    with pytest.raises(RuntimeError, match="no valid OpenFace frames"):
        enc.encode(tmp_path, 20, 30)


def test_same_phase_windows_equal_length_and_balanced_step_prior(tmp_path: Path) -> None:
    # Build two synthetic participants (P1 in train, P2 in test) with all 5 phase-matched pairs
    reflex_items: List[Item] = []
    for pid_idx, pid in enumerate(["P1", "P2"]):
        pdir = tmp_path / "raw" / "reflex" / "Dataset" / "C1-Fixed-Low" / pid
        pdir.mkdir(parents=True, exist_ok=True)
        vid = pdir / "video_cam1.mp4"
        vid.write_bytes(b"fake")

        specs = [
            ("1", "B", "Pick", 100, 140, 140, 165),
            ("1", "B", "Carry", 200, 245, 245, 270),
            ("1", "D", "Place", 300, 338, 338, 360),
            ("2", "A", "Carry", 400, 432, 432, 460),
            ("3", "A", "Place", 500, 528, 528, 555),
            ("3", "B", "Carry", 600, 642, 642, 668),
            ("4", "A", "Pick", 700, 735, 735, 756),
            ("4", "C", "Place", 800, 824, 824, 850),
        ]
        acsv_lines = [
            "Participant,Strategy,Round No.,Object,Action,Explanation Level,State,Start Frame,End Frame,Start,End,Resolved?"
        ]
        for ep_idx, (rnd, obj, act, psf, pef, fsf, fef) in enumerate(specs):
            acsv_lines.append(
                f"{pid},C1 Fixed Low,{rnd},{obj},{act},Low,Pre,{psf},{pef},1.0,2.0,TRUE"
            )
            acsv_lines.append(
                f"{pid},C1 Fixed Low,{rnd},{obj},{act},Low,Failure,{fsf},{fef},2.0,3.0,TRUE"
            )
            for kind, label, sf, ef in [("pre", 1, psf, pef), ("fail", 0, fsf, fef)]:
                reflex_items.append(
                    Item(
                        item_id=f"reflex:{pid}:{ep_idx}:{kind}",
                        dataset="reflex",
                        group_id=pid,
                        video_path=str(vid),
                        action_window_sec=[1.0, 2.0],
                        reaction_window_sec=[1.0, 2.0],
                        label=label,
                        label_source="analysis.csv",
                        context_text="",
                        meta={
                            "action_frames": [sf, ef],
                            "reaction_frames": [sf, ef + (10 if label == 0 else 0)],
                            "verb": act.lower(),
                            "noun": "object",
                            "round_no": rnd,
                            "episode_idx": ep_idx,
                            "participant": pid,
                        },
                    )
                )
        (pdir / "analysis.csv").write_text("\n".join(acsv_lines) + "\n", encoding="utf-8")

    split_data: Dict[str, List[str]] = {
        "train": [it.item_id for it in reflex_items if it.group_id == "P1"],
        "test": [it.item_id for it in reflex_items if it.group_id == "P2"],
    }

    all_sp, tr_sp, te_sp, windows, _ = build_same_phase_items_and_windows(
        reflex_items=reflex_items,
        split_data=split_data,
        root=tmp_path,
    )
    assert len(all_sp) == 20
    assert len(tr_sp) == 10
    assert len(te_sp) == 10
    assert len(SAME_PHASE_PAIRS) == 5

    by_pair: Dict[str, List[Item]] = {}
    for it in all_sp:
        by_pair.setdefault(str(it.meta["pair_key"]), []).append(it)
    assert len(by_pair) == 10
    for pair_items in by_pair.values():
        assert len(pair_items) == 2
        labels = sorted(it.label for it in pair_items)
        assert labels == [0, 1]
        verbs = {str(it.meta["verb"]) for it in pair_items}
        assert len(verbs) == 1
        assert next(iter(verbs)) in {"pick", "carry"}
        lengths = [windows[it.item_id][1] - windows[it.item_id][0] + 1 for it in pair_items]
        assert lengths[0] == lengths[1]
        # End-anchored at action_frames[1]
        for it in pair_items:
            assert windows[it.item_id][1] == int(it.meta["action_frames"][1])

    # Physical-phase step_prior on balanced pairs must be 0.5 everywhere (no label leakage)
    y_tr = np.array([it.label for it in tr_sp], dtype=int)
    y_te = np.array([it.label for it in te_sp], dtype=int)
    sp_te = step_prior_scores(list(tr_sp), list(te_sp), y_tr, loo=False)
    assert np.allclose(sp_te, 0.5)

    # Falsify: if we left the episode failure verb unnormalized ("carry"/"place" on pos vs "pick"/"carry" on neg),
    # step_prior would leak the label with AUROC > 0.85
    orig_map = {it.item_id: it for it in reflex_items}
    leaked_tr = [orig_map[str(it.meta["orig_item_id"])] for it in tr_sp]
    leaked_te = [orig_map[str(it.meta["orig_item_id"])] for it in te_sp]
    leaked_scores = step_prior_scores(leaked_tr, leaked_te, y_tr, loo=False)
    leaked_res = auroc_ci(y_te, leaked_scores, [it.group_id for it in te_sp])
    assert leaked_res.point is not None and float(leaked_res.point) > 0.80


def test_confirmatory_verdict_helper() -> None:
    res_pass = evaluate_confirmatory_verdict(
        standalone=MetricResult(point=0.68, ci_low=0.55, ci_high=0.79, n_excluded=0),
        ablation_delta=MetricResult(point=0.04, ci_low=0.01, ci_high=0.07, n_excluded=0),
    )
    assert res_pass["standalone_above_chance"] is True
    assert res_pass["ablation_gain_positive"] is True
    assert res_pass["confirmatory_supported"] is True
    assert res_pass["h1_pass_bar_met"] is True
    assert res_pass["verdict"] == "SUPPORTED"

    res_null_gain = evaluate_confirmatory_verdict(
        standalone=MetricResult(point=0.62, ci_low=0.52, ci_high=0.71, n_excluded=0),
        ablation_delta=MetricResult(point=0.005, ci_low=-0.018, ci_high=0.029, n_excluded=0),
    )
    assert res_null_gain["standalone_above_chance"] is True
    assert res_null_gain["ablation_gain_positive"] is False
    assert res_null_gain["confirmatory_supported"] is False
    assert res_null_gain["h1_pass_bar_met"] is False
    assert res_null_gain["verdict"] == "NOT SUPPORTED"

    res_null_standalone = evaluate_confirmatory_verdict(
        standalone=MetricResult(point=0.51, ci_low=0.43, ci_high=0.59, n_excluded=0),
        ablation_delta=MetricResult(point=0.02, ci_low=0.002, ci_high=0.04, n_excluded=0),
    )
    assert res_null_standalone["standalone_above_chance"] is False
    assert res_null_standalone["ablation_gain_positive"] is True
    assert res_null_standalone["confirmatory_supported"] is False
    assert res_null_standalone["verdict"] == "NOT SUPPORTED"


@pytest.mark.slow
def test_eval_reflex_openface_gaze_reproduces_json() -> None:
    computed = evaluate_reflex_openface_gaze()
    assert computed["_status"] == STATUS_TEXT
    assert computed["same_phase.n_train_items"] == {"value": 380}
    assert computed["same_phase.n_test_items"] == {"value": 170}
    assert computed["same_phase.n_train_valid_openface_items"] == {"value": 376}
    assert computed["same_phase.n_test_valid_openface_items"] == {"value": 157}
    assert computed["same_phase.n_train_valid_openface_pairs"] == {"value": 186}
    assert computed["same_phase.n_test_valid_openface_pairs"] == {"value": 77}
    assert computed["same_phase.window_length"]["value"] == pytest.approx(0.5, abs=1e-6)
    assert computed["same_phase.step_prior"]["value"] == pytest.approx(0.5, abs=1e-6)

    assert DEFAULT_OUT_PATH.exists(), f"Missing {DEFAULT_OUT_PATH}"
    saved = json.loads(DEFAULT_OUT_PATH.read_text(encoding="utf-8"))
    assert saved == computed
