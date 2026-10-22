from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from harness.reporting import leak_check_label


def h3_verdict(rows: Dict[str, Any]) -> Tuple[str, str]:
    """Pre-registered verdict rule (04_wave_d_robot_check.md §6).

    Supported iff both:
      - gain:react:N=100 >= +0.02, with ci_low > 0;
      - and react_minus_shuffled:N=100 has ci_low > 0.
    Not supported otherwise, with interpretation from table in order (first match).
    """
    r_truth = rows.get("gain:truth:N=100")
    r_react = rows.get("gain:react:N=100")
    r_react_shuffled = rows.get("react_minus_shuffled:N=100")

    gain_truth_ci_low = r_truth.get("ci_low") if r_truth else None
    gain_react_val = r_react.get("value") if r_react else None
    gain_react_ci_low = r_react.get("ci_low") if r_react else None
    react_shuffled_ci_low = r_react_shuffled.get("ci_low") if r_react_shuffled else None

    # Supported condition
    if (
        gain_react_val is not None
        and gain_react_val >= 0.02
        and gain_react_ci_low is not None
        and gain_react_ci_low > 0.0
        and react_shuffled_ci_low is not None
        and react_shuffled_ci_low > 0.0
    ):
        return "SUPPORTED", "H3 is Supported: human reaction prior improves robot detector by >= +0.02 AUROC over no-prior baseline and beats shuffled human prior."

    # First match in contract §6 table
    if gain_truth_ci_low is None or gain_truth_ci_low <= 0.0:
        return "NOT SUPPORTED", "Human→robot transfer does not help in this setup even with true human labels; reaction labels cannot be judged here."
    elif gain_react_ci_low is None or gain_react_ci_low <= 0.0:
        return "NOT SUPPORTED", "Labeled human video transfers, but reaction-derived labels are too weak to."
    elif react_shuffled_ci_low is None or react_shuffled_ci_low <= 0.0:
        return "NOT SUPPORTED", "A human-video prior helps regardless of its labels; the gain is not a reaction effect."
    else:
        return "NOT SUPPORTED", "The gain is real but below the +0.02 bar."


def generate_wave_d_report(
    date_str: Optional[str] = None,
    scorecard_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    today = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sc_file = Path(scorecard_path or "results/scorecard.jsonl")

    rows: List[Dict[str, Any]] = []
    if sc_file.exists():
        for line in sc_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("hypothesis") == "H3":
                        rows.append(r)
                except Exception:
                    pass

    latest_rows: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest_rows[r["condition"]] = r

    verdict, verdict_text = h3_verdict(latest_rows)

    lines: List[str] = [
        f"# Wave D Robot Check Evaluation Report ({today})",
        "",
        "## 1. Executive Summary & Verdict (H3 Offline)",
        "",
        f"- **Pre-registered Verdict:** **{verdict}**",
        f"- **Interpretation:** {verdict_text}",
        "",
        "### Key N=100 Comparisons",
        "",
    ]

    for cond in ["gain:truth:N=100", "gain:react:N=100", "react_minus_shuffled:N=100"]:
        r = latest_rows.get(cond)
        if r and r.get("value") is not None:
            ci_s = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if r.get("ci_low") is not None else "—"
            lines.append(f"- **`{cond}`:** {r['value']:+.4f} {ci_s}")
        else:
            lines.append(f"- **`{cond}`:** not run")

    lines.extend([
        "",
        "## 2. Data-Efficiency Curve (mean20 Test AUROC)",
        "",
        "| Budget (N) | None (w0=0) | React Prior | Shuffled Prior | Truth Prior | Gain (React - None) | Gain (React - Shuffled) |",
        "|---|---|---|---|---|---|---|",
    ])

    budgets = [50, 100, 200, 500, 1000]
    for n in budgets:
        r_none = latest_rows.get(f"detector:none:N={n}")
        r_react = latest_rows.get(f"detector:react:N={n}")
        r_shuf = latest_rows.get(f"detector:shuffled:N={n}")
        r_truth = latest_rows.get(f"detector:truth:N={n}")
        r_gain_react = latest_rows.get(f"gain:react:N={n}")
        r_react_shuf = latest_rows.get(f"react_minus_shuffled:N={n}")

        def fmt(r: Optional[Dict[str, Any]]) -> str:
            if not r or r.get("value") is None:
                return "—"
            ci = f" [{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r.get("ci_low") is not None else ""
            return f"{r['value']:.3f}{ci}"

        lines.append(
            f"| N={n} | {fmt(r_none)} | {fmt(r_react)} | {fmt(r_shuf)} | {fmt(r_truth)} | {fmt(r_gain_react)} | {fmt(r_react_shuf)} |"
        )

    lines.extend([
        "",
        "## 3. Reference Lines (Context, No Verdict)",
        "",
        "| Condition | Metric | Value | 95% CI | Notes |",
        "|---|---|---|---|---|",
    ])

    for cond in ["robot-full", "judge-large", "roboreward-8b"]:
        r = latest_rows.get(cond)
        if r and r.get("value") is not None:
            ci_s = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r.get("ci_low") is not None else "—"
            val_s = f"{r['value']:.3f}"
            notes = r.get("notes", "")
        else:
            val_s = "null"
            ci_s = "—"
            notes = r.get("notes", "not run") if r else "not run"
        lines.append(f"| `{cond}` | auroc | {val_s} | {ci_s} | {notes} |")

    lines.extend([
        "",
        "## 4. Pseudo-Label Quality (HoloAssist Pool)",
        "",
        "| Condition | Metric | Value | 95% CI | N (items) | Notes |",
        "|---|---|---|---|---|---|",
    ])

    for cond in ["pseudo-label-quality", "pseudo-label-quality:all"]:
        r = latest_rows.get(cond)
        if r and r.get("value") is not None:
            ci_s = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if r.get("ci_low") is not None else "—"
            val_s = f"{r['value']:.4f}"
            n_items = str(r.get("n_items", "—"))
            notes = r.get("notes", "")
        else:
            val_s = "null"
            ci_s = "—"
            n_items = "—"
            notes = r.get("notes", "not run") if r else "not run"
        lines.append(f"| `{cond}` | auroc | {val_s} | {ci_s} | {n_items} | {notes} |")

    lines.extend([
        "",
        "## 5. Shuffled Control (Leak Check)",
        "",
        "Leak check (03_eval_harness.md §11.9): a shuffled control fails only if its ci_low > 0.5.",
        "",
        "| Condition | Metric | Value | 95% CI | Leak check | Notes |",
        "|---|---|---|---|---|---|",
    ])

    r_leak = latest_rows.get("detector:none:N=1000:shuffled")
    if r_leak and r_leak.get("value") is not None:
        ci_s = f"[{r_leak['ci_low']:.3f}, {r_leak['ci_high']:.3f}]" if r_leak.get("ci_low") is not None else "—"
        val_s = f"{r_leak['value']:.3f}"
        leak_status = leak_check_label(r_leak.get("value"), r_leak.get("ci_low"), r_leak.get("ci_high"))
        notes = r_leak.get("notes", "shuffled-label control")
    else:
        val_s = "null"
        ci_s = "—"
        leak_status = "— (not run)"
        notes = "not run"

    lines.append(f"| `detector:none:N=1000:shuffled` | auroc_mean20 | {val_s} | {ci_s} | {leak_status} | {notes} |")
    lines.append("")

    target_path = Path(out_path or f"docs/evals/{today}_wave_d_robot_check.md")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated Wave D report at {target_path}")

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wave D Robot Check Report")
    parser.add_argument("--date", default=None, help="Date string YYYY-MM-DD")
    parser.add_argument("--scorecard", default="results/scorecard.jsonl", help="Scorecard path")
    parser.add_argument("--out", default=None, help="Output markdown path")
    args = parser.parse_args()

    generate_wave_d_report(date_str=args.date, scorecard_path=args.scorecard, out_path=args.out)


if __name__ == "__main__":
    main()
