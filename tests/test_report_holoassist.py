import json
from pathlib import Path

from tools.report_holoassist import prediction_check_hidden


def test_prediction_check_hidden_real_rows():
    """(a) Real HoloAssist rows -> Clause 1 not tested, Clause 2 not tested, pass bar not testable.
    No line containing held or met also contains react-full.
    """
    sc_path = Path("results/scorecard.jsonl")
    assert sc_path.exists()
    rows = {}
    for line in sc_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("dataset") == "holoassist":
                rows[r["condition"]] = r

    lines = prediction_check_hidden(rows)
    assert any("Clause 1" in line and "not tested" in line for line in lines)
    assert any("Clause 2" in line and "not tested" in line for line in lines)
    assert any("H1 pass bar" in line and "not testable" in line for line in lines)

    # Ensure react-full never appears in held or met lines
    for line in lines:
        if "held" in line.lower() or "met" in line.lower():
            assert "react-full" not in line


def test_prediction_check_hidden_synthetic_held_met():
    """(b) react-nonverbal 0.80 [0.76, 0.84], action_best = action-probe 0.57,
    ablation Delta 0.05 [0.02, 0.08] -> held, held, met.
    """
    rows = {
        "react-nonverbal": {
            "value": 0.80, "ci_low": 0.76, "ci_high": 0.84, "notes": ""
        },
        "action-probe": {
            "value": 0.57, "ci_low": 0.54, "ci_high": 0.60, "notes": ""
        },
        "fusion_minus_action_best": {
            "value": 0.23, "ci_low": 0.18, "ci_high": 0.28, "notes": "action_best=action-probe"
        },
        "fusion_minus_action_fusion": {
            "value": 0.05, "ci_low": 0.02, "ci_high": 0.08, "notes": "ablation: reaction features removed"
        },
        "judge": {"value": 0.488, "ci_low": 0.466, "ci_high": 0.512},
        "judge-large": {"value": 0.544, "ci_low": 0.528, "ci_high": 0.562},
        "react-spoke": {"value": 0.676, "ci_low": 0.655, "ci_high": 0.695},
        "react-full": {"value": 0.739, "ci_low": 0.716, "ci_high": 0.761},
    }

    lines = prediction_check_hidden(rows)
    c1 = next(line for line in lines if "Clause 1" in line)
    c2 = next(line for line in lines if "Clause 2" in line)
    pb = next(line for line in lines if "H1 pass bar" in line)

    assert "→ held" in c1
    assert "→ held" in c2
    assert "→ met" in pb


def test_prediction_check_hidden_falsifying_react_full():
    """(c, falsifying) react-full 0.90, react-nonverbal null -> pass bar not testable."""
    rows = {
        "react-full": {
            "value": 0.90, "ci_low": 0.88, "ci_high": 0.92, "notes": "leakage condition"
        },
        "react-nonverbal": {
            "value": None, "ci_low": None, "ci_high": None, "notes": "not run: audio presence check failed"
        },
        "action-probe": {
            "value": 0.57, "ci_low": 0.54, "ci_high": 0.60, "notes": ""
        },
        "fusion_minus_action_best": {
            "value": None, "ci_low": None, "ci_high": None, "notes": "not run: fusion not run; action_best=action-probe"
        },
        "fusion_minus_action_fusion": {
            "value": None, "ci_low": None, "ci_high": None, "notes": "not run: fusion not run"
        },
    }

    lines = prediction_check_hidden(rows)
    pb = next(line for line in lines if "H1 pass bar" in line)
    assert "not testable" in pb
    assert "met" not in pb.split("→")[1]
