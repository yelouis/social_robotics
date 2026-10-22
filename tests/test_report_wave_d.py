from __future__ import annotations


from tools.report_wave_d import h3_verdict


def test_h3_verdict_supported():
    """gain:react:N=100 >= 0.02 with ci_low > 0 AND react_minus_shuffled:N=100 ci_low > 0 -> Supported."""
    rows = {
        "gain:truth:N=100": {"value": 0.05, "ci_low": 0.02, "ci_high": 0.08},
        "gain:react:N=100": {"value": 0.03, "ci_low": 0.01, "ci_high": 0.05},
        "react_minus_shuffled:N=100": {"value": 0.02, "ci_low": 0.005, "ci_high": 0.035},
    }
    verdict, text = h3_verdict(rows)
    assert verdict == "SUPPORTED"
    assert "Supported" in text


def test_h3_verdict_truth_fails():
    """gain:truth:N=100 ci_low <= 0 -> human video does not help even with true labels."""
    rows = {
        "gain:truth:N=100": {"value": 0.01, "ci_low": -0.01, "ci_high": 0.03},
        "gain:react:N=100": {"value": 0.00, "ci_low": -0.02, "ci_high": 0.02},
        "react_minus_shuffled:N=100": {"value": 0.00, "ci_low": -0.01, "ci_high": 0.01},
    }
    verdict, text = h3_verdict(rows)
    assert verdict == "NOT SUPPORTED"
    assert text == "Human→robot transfer does not help in this setup even with true human labels; reaction labels cannot be judged here."


def test_h3_verdict_react_fails_truth_passes():
    """gain:truth passes, but gain:react ci_low <= 0 -> reaction-derived labels too weak."""
    rows = {
        "gain:truth:N=100": {"value": 0.04, "ci_low": 0.01, "ci_high": 0.07},
        "gain:react:N=100": {"value": 0.01, "ci_low": -0.01, "ci_high": 0.03},
        "react_minus_shuffled:N=100": {"value": 0.00, "ci_low": -0.01, "ci_high": 0.01},
    }
    verdict, text = h3_verdict(rows)
    assert verdict == "NOT SUPPORTED"
    assert text == "Labeled human video transfers, but reaction-derived labels are too weak to."


def test_h3_verdict_shuffled_not_separated():
    """gain:react passes, but react_minus_shuffled ci_low <= 0 -> not a reaction effect."""
    rows = {
        "gain:truth:N=100": {"value": 0.04, "ci_low": 0.01, "ci_high": 0.07},
        "gain:react:N=100": {"value": 0.03, "ci_low": 0.01, "ci_high": 0.05},
        "react_minus_shuffled:N=100": {"value": 0.005, "ci_low": -0.005, "ci_high": 0.015},
    }
    verdict, text = h3_verdict(rows)
    assert verdict == "NOT SUPPORTED"
    assert text == "A human-video prior helps regardless of its labels; the gain is not a reaction effect."


def test_h3_verdict_below_bar():
    """gain:react ci_low > 0, react_minus_shuffled ci_low > 0, but gain:react < 0.02."""
    rows = {
        "gain:truth:N=100": {"value": 0.04, "ci_low": 0.01, "ci_high": 0.07},
        "gain:react:N=100": {"value": 0.015, "ci_low": 0.005, "ci_high": 0.025},
        "react_minus_shuffled:N=100": {"value": 0.010, "ci_low": 0.002, "ci_high": 0.018},
    }
    verdict, text = h3_verdict(rows)
    assert verdict == "NOT SUPPORTED"
    assert text == "The gain is real but below the +0.02 bar."
