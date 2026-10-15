from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.report_oops import prediction_check  # noqa: E402


def test_prediction_check_oops_real_values():
    """Real Oops! values: judge AUROC is at chance (0.472 [0.440, 0.498]), action-probe is 0.779,

    Delta is 0.015 [-0.002, 0.029].
    Clause 1 did not hold; Clause 2 held.
    """
    rows = {
        "judge": {
            "condition": "judge",
            "metric": "auroc",
            "value": 0.472,
            "ci_low": 0.440,
            "ci_high": 0.498,
            "n_items": 1072,
            "notes": "",
        },
        "judge-frontier": {
            "condition": "judge-frontier",
            "metric": "auroc",
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "n_items": 1072,
            "notes": "not run: GOOGLE_API_KEY daily quota exhausted (20 req/day on Free Tier)",
        },
        "action-probe": {
            "condition": "action-probe",
            "metric": "auroc",
            "value": 0.779,
            "ci_low": 0.754,
            "ci_high": 0.805,
            "n_items": 1072,
            "notes": "",
        },
        "fusion_minus_action_best": {
            "condition": "fusion_minus_action_best",
            "metric": "delta_auroc",
            "value": 0.015,
            "ci_low": -0.002,
            "ci_high": 0.029,
            "n_items": 1072,
            "notes": "action-probe",
        },
    }

    lines = prediction_check(rows)
    text = "\n".join(lines)

    # Clause 1 for judge: did not hold
    assert '- **Clause 1, "the judge is strong" (judge):** AUROC 0.472 [0.440, 0.498] (1072 scored) → did not hold' in text
    # Clause 2: held
    assert '- **Clause 2, "Δ ≈ 0":** Δ 0.015 [-0.002, 0.029] (action_best = action-probe) → held' in text
    # Null condition listed with notes
    assert "- **judge-frontier:** not run: GOOGLE_API_KEY daily quota exhausted (20 req/day on Free Tier)" in text


def test_prediction_check_strong_judge_nonzero_delta():
    """Synthetic values: judge is strong (0.85 [0.80, 0.90]), action-probe is 0.78,

    Delta is 0.050 [0.020, 0.080] (excludes 0).
    Clause 1 held; Clause 2 did not hold.
    """
    rows = {
        "judge": {
            "condition": "judge",
            "metric": "auroc",
            "value": 0.850,
            "ci_low": 0.800,
            "ci_high": 0.900,
            "n_items": 1000,
            "notes": "",
        },
        "action-probe": {
            "condition": "action-probe",
            "metric": "auroc",
            "value": 0.780,
            "ci_low": 0.740,
            "ci_high": 0.820,
            "n_items": 1000,
            "notes": "",
        },
        "fusion_minus_action_best": {
            "condition": "fusion_minus_action_best",
            "metric": "delta_auroc",
            "value": 0.050,
            "ci_low": 0.020,
            "ci_high": 0.080,
            "n_items": 1000,
            "notes": "judge",
        },
    }

    lines = prediction_check(rows)
    text = "\n".join(lines)

    # Clause 1 for judge: held
    assert '- **Clause 1, "the judge is strong" (judge):** AUROC 0.850 [0.800, 0.900] (1000 scored) → held' in text
    # Clause 2: did not hold
    assert '- **Clause 2, "Δ ≈ 0":** Δ 0.050 [0.020, 0.080] (action_best = judge) → did not hold' in text
