from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.report_wave_d import h3_verdict  # noqa: E402


def verify_report(
    report_path: str = "docs/evals/2026-10-22_wave_d_robot_check.md",
    scorecard_path: str = "results/scorecard.jsonl",
) -> None:
    sc_file = Path(scorecard_path)
    h3_rows = {}
    for line in sc_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("hypothesis") == "H3":
                h3_rows[d["condition"]] = d

    verdict, text = h3_verdict(h3_rows)
    report_text = Path(report_path).read_text(encoding="utf-8")

    # 1. Executive Summary
    assert f"- **Pre-registered Verdict:** **{verdict}**" in report_text, "Verdict header mismatch"
    assert f"- **Interpretation:** {text}" in report_text, "Verdict text mismatch"

    for cond in ["gain:truth:N=100", "gain:react:N=100", "react_minus_shuffled:N=100"]:
        r = h3_rows[cond]
        val_str = f"{r['value']:+.4f}"
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        expected_line = f"- **`{cond}`:** {val_str} {ci_str}"
        assert expected_line in report_text, f"Key comparison mismatch: {expected_line}"

    # 2. Curve Table
    budgets = [50, 100, 200, 500, 1000]
    for n in budgets:
        r_none = h3_rows[f"detector:none:N={n}"]
        r_react = h3_rows[f"detector:react:N={n}"]
        r_shuf = h3_rows[f"detector:shuffled:N={n}"]
        r_truth = h3_rows[f"detector:truth:N={n}"]
        r_gain = h3_rows[f"gain:react:N={n}"]
        r_react_shuf = h3_rows[f"react_minus_shuffled:N={n}"]

        row_str = (
            f"| N={n} | "
            f"{r_none['value']:.3f} [{r_none['ci_low']:.3f}, {r_none['ci_high']:.3f}] | "
            f"{r_react['value']:.3f} [{r_react['ci_low']:.3f}, {r_react['ci_high']:.3f}] | "
            f"{r_shuf['value']:.3f} [{r_shuf['ci_low']:.3f}, {r_shuf['ci_high']:.3f}] | "
            f"{r_truth['value']:.3f} [{r_truth['ci_low']:.3f}, {r_truth['ci_high']:.3f}] | "
            f"{r_gain['value']:.3f} [{r_gain['ci_low']:.3f}, {r_gain['ci_high']:.3f}] | "
            f"{r_react_shuf['value']:.3f} [{r_react_shuf['ci_low']:.3f}, {r_react_shuf['ci_high']:.3f}] |"
        )
        assert row_str in report_text, f"Curve row N={n} mismatch:\nExpected: {row_str}"

    # 3. Reference lines
    r_rf = h3_rows["robot-full"]
    rf_line = f"| `robot-full` | auroc | {r_rf['value']:.3f} | [{r_rf['ci_low']:.3f}, {r_rf['ci_high']:.3f}] | {r_rf['notes']} |"
    assert rf_line in report_text, f"robot-full mismatch: {rf_line}"

    r_jl = h3_rows["judge-large"]
    jl_line = f"| `judge-large` | auroc | {r_jl['value']:.3f} | [{r_jl['ci_low']:.3f}, {r_jl['ci_high']:.3f}] | {r_jl['notes']} |"
    assert jl_line in report_text, f"judge-large mismatch: {jl_line}"

    r8b_line = "| `roboreward-8b` | auroc | null | — | not run: RoboReward-8B is Qwen3-VL-based and needs a transformers newer than the frozen 4.46.3 (Issue 9 chose B) |"
    assert r8b_line in report_text, f"roboreward-8b mismatch: {r8b_line}"

    # 4. Pseudo-label quality
    r_pl_all = h3_rows["pseudo-label-quality:all"]
    pl_line = f"| `pseudo-label-quality:all` | auroc | {r_pl_all['value']:.4f} | [{r_pl_all['ci_low']:.4f}, {r_pl_all['ci_high']:.4f}] | {r_pl_all['n_items']} | {r_pl_all['notes']} |"
    assert pl_line in report_text, f"pseudo-label quality mismatch: {pl_line}"

    # 5. Leak check
    r_leak = h3_rows["detector:none:N=1000:shuffled"]
    leak_line = f"| `detector:none:N=1000:shuffled` | auroc_mean20 | {r_leak['value']:.3f} | [{r_leak['ci_low']:.3f}, {r_leak['ci_high']:.3f}] | pass | {r_leak['notes']} |"
    assert leak_line in report_text, f"leak check mismatch: {leak_line}"

    print("SUCCESS: Every number in the report matches a scorecard row exactly!")
    print(f"h3_verdict: {verdict}")
    print(f"Interpretation: {text}")


if __name__ == "__main__":
    verify_report()
