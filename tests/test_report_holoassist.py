import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.report_holoassist import generate_holoassist_report, prediction_check_hidden  # noqa: E402


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


def test_generate_holoassist_report_audible_synthetic(tmp_path):
    """Test generate_holoassist_report with dataset='holoassist_audible'."""
    # Write a mock scorecard
    sc_file = tmp_path / "scorecard.jsonl"
    mock_rows = [
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "action-probe",
            "metric": "auroc",
            "value": 0.58,
            "ci_low": 0.55,
            "ci_high": 0.61,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "step-prior",
            "metric": "auroc",
            "value": 0.72,
            "ci_low": 0.69,
            "ci_high": 0.75,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "react-nonverbal",
            "metric": "auroc",
            "value": 0.68,
            "ci_low": 0.65,
            "ci_high": 0.71,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "fusion",
            "metric": "auroc",
            "value": 0.75,
            "ci_low": 0.72,
            "ci_high": 0.78,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "action-fusion",
            "metric": "auroc",
            "value": 0.69,
            "ci_low": 0.66,
            "ci_high": 0.72,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "fusion_minus_action_fusion",
            "metric": "delta_auroc",
            "value": 0.06,
            "ci_low": 0.03,
            "ci_high": 0.09,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "ablation: reaction features removed",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "fusion_minus_action_best",
            "metric": "delta_auroc",
            "value": 0.03,
            "ci_low": 0.00,
            "ci_high": 0.06,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "action_best=step-prior",
        },
    ]
    with open(sc_file, "w", encoding="utf-8") as f:
        for r in mock_rows:
            f.write(json.dumps(r) + "\n")

    out_md = tmp_path / "report.md"
    rep_path = generate_holoassist_report(
        dataset="holoassist_audible",
        date_str="2026-10-20",
        data_root=tmp_path,
        scorecard_path=sc_file,
        out_path=out_md,
    )

    assert rep_path.exists()
    content = rep_path.read_text(encoding="utf-8")
    assert "HoloAssist (Audible Sessions) H1 Evaluation Report" in content or "HoloAssist Audible" in content
    assert "The subset" in content or "The Subset" in content
    assert "Clause 1" in content
    assert "Clause 2" in content

