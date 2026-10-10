from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import DATA_ROOT
from harness.items import read_items
from harness.splits import load_split


def generate_oops_report(
    date_str: Optional[str] = None,
    data_root: Optional[Path] = None,
    scorecard_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    wall_clocks: Optional[Dict[str, float]] = None,
) -> Path:
    root = data_root or DATA_ROOT
    today = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sc_file = scorecard_path or Path("results/scorecard.jsonl")

    # 1. Read scorecard rows for Oops
    rows: List[Dict[str, Any]] = []
    if sc_file.exists():
        for line in sc_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("dataset") == "oops":
                        rows.append(r)
                except Exception:
                    pass

    # Group latest by condition
    latest_rows: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest_rows[r["condition"]] = r

    # 2. Judge stats
    judge_stats = {"ollama": {"total": 0, "parse_failures": 0, "api_errors": 0},
                   "gemini": {"total": 0, "parse_failures": 0, "api_errors": 0}}
    judge_dir = root / "judge" / "oops"
    if judge_dir.exists():
        for model_dir in judge_dir.iterdir():
            if not model_dir.is_dir():
                continue
            backend = "gemini" if "gemini" in model_dir.name else "ollama"
            for jf in model_dir.glob("*.jsonl"):
                for line in jf.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                        judge_stats[backend]["total"] += 1
                        if e.get("judge_prob") is None:
                            raw = e.get("raw", "")
                            if "API error" in raw or "error" in raw.lower():
                                judge_stats[backend]["api_errors"] += 1
                            else:
                                judge_stats[backend]["parse_failures"] += 1
                    except Exception:
                        pass

    # 3. 10 example items (5 pre, 5 post, default_rng(0))
    items_path = root / "items" / "oops" / "items.jsonl"
    all_items = read_items(items_path)
    split_data = load_split("oops", path_dir="splits")
    test_set = set(split_data.get("test", []))
    test_items = [it for it in all_items if it.item_id in test_set]

    pre_items = [it for it in test_items if it.label == 1]
    post_items = [it for it in test_items if it.label == 0]

    rng = np.random.default_rng(0)
    pre_sample_idx = rng.choice(len(pre_items), size=min(5, len(pre_items)), replace=False)
    post_sample_idx = rng.choice(len(post_items), size=min(5, len(post_items)), replace=False)

    sampled_pre = [pre_items[i] for i in sorted(pre_sample_idx)]
    sampled_post = [post_items[i] for i in sorted(post_sample_idx)]

    # Load judge cached predictions
    judge_cached: Dict[str, Optional[float]] = {}
    if judge_dir.exists():
        for model_dir in judge_dir.iterdir():
            if "qwen" in model_dir.name and model_dir.is_dir():
                for jf in model_dir.glob("*.jsonl"):
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                e = json.loads(line)
                                judge_cached[e["item_id"]] = e.get("judge_prob")
                            except Exception:
                                pass

    # Build report markdown
    lines = [
        f"# Oops! H1 Evaluation Report ({today})",
        "",
        "## 1. Executive Summary & Prediction Check",
        "",
        "**Prediction Check:**",
        "> *Prediction:* In fail videos where failures are plainly visible in the action frames, the VLM judge is expected to perform strongly, and non-verbal reactions should add little or no incremental signal over the action-only baseline ($\\Delta \\approx 0$).",
        "",
    ]

    # Delta row check
    delta_row = latest_rows.get("fusion_minus_action_best")
    if delta_row:
        d_val = delta_row.get("value")
        d_low = delta_row.get("ci_low")
        d_high = delta_row.get("ci_high")
        lines.append(f"- **Result:** $\\Delta = {d_val:.3f}$ [95% CI: {d_low:.3f}, {d_high:.3f}] (action_best = `{delta_row.get('notes')}`).")
        if d_low is not None and d_low <= 0 <= d_high:
            lines.append("- **Verdict:** The prediction **holds**. Reactions add no statistically significant gain over the action-only judge on visible failure outcomes (zero is inside the 95% CI).")
        else:
            lines.append(f"- **Verdict:** $\\Delta$ is {d_val:.3f} with 95% CI [{d_low:.3f}, {d_high:.3f}].")
    lines.append("")

    # Conditions Table
    lines.extend([
        "## 2. Conditions Table (Test Split)",
        "",
        "| Condition | Metric | Value | 95% CI | N (included) | N (excluded) | Notes |",
        "|---|---|---|---|---|---|---|",
    ])
    core_conditions = [
        "judge",
        "judge-frontier",
        "action-probe",
        "react-nonverbal",
        "react-spoke",
        "react-full",
        "fusion",
        "fusion_minus_action_best",
    ]
    for c in core_conditions:
        r = latest_rows.get(c)
        if r:
            val_s = f"{r['value']:.3f}" if r.get("value") is not None else "null"
            ci_s = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r.get("ci_low") is not None and r.get("ci_high") is not None else "—"
            notes = r.get("notes", "")
            lines.append(f"| `{r['condition']}` | {r['metric']} | {val_s} | {ci_s} | {r['n_items']} | {r['n_excluded']} | {notes} |")
        else:
            lines.append(f"| `{c}` | auroc | null | — | — | — | not run |")
    lines.append("")

    # Shuffled Controls Table
    lines.extend([
        "## 3. Shuffled-Label Controls",
        "",
        "Validation requirement: every `:shuffled` condition must have a 95% CI covering chance (0.50).",
        "",
        "| Condition | Metric | Value | 95% CI | Covers 0.5? | Notes |",
        "|---|---|---|---|---|---|",
    ])
    shuffled_conditions = ["action-probe:shuffled", "react-nonverbal:shuffled", "fusion:shuffled"]
    for c in shuffled_conditions:
        r = latest_rows.get(c)
        if r:
            val_s = f"{r['value']:.3f}" if r.get("value") is not None else "null"
            ci_low = r.get("ci_low")
            ci_high = r.get("ci_high")
            ci_s = f"[{ci_low:.3f}, {ci_high:.3f}]" if ci_low is not None and ci_high is not None else "—"
            covers_05 = "PASS (Yes)" if (ci_low is not None and ci_high is not None and ci_low <= 0.5 <= ci_high) else "FAIL (No)"
            lines.append(f"| `{r['condition']}` | {r['metric']} | {val_s} | {ci_s} | {covers_05} | {r.get('notes', '')} |")
    lines.append("")

    # Judge Execution Stats
    lines.extend([
        "## 4. Judge Diagnostics",
        "",
        f"- **Local Judge (`qwen2.5vl:7b`):** total queries = {judge_stats['ollama']['total']}, parse failures = {judge_stats['ollama']['parse_failures']}, API errors = {judge_stats['ollama']['api_errors']}",
        f"- **Frontier Judge (`gemini-3.6-flash`):** total queries = {judge_stats['gemini']['total']}, parse failures = {judge_stats['gemini']['parse_failures']}, API errors = {judge_stats['gemini']['api_errors']}",
        "",
    ])

    # Wall-clock timing
    lines.extend([
        "## 5. Wall-Clock Compute Time",
        "",
    ])
    if wall_clocks:
        for stage, secs in wall_clocks.items():
            lines.append(f"- **{stage}:** {secs:.1f} s ({secs / 60.0:.1f} min)")
    else:
        lines.append("- Timings recorded per supervisor run logs under `raw/oops/` and `runs/`.")
    lines.append("")

    # 10 example items
    lines.extend([
        "## 6. Example Items (Test Split)",
        "",
        "Ten items sampled with `default_rng(0)` (5 pre-failure, 5 post-failure), described in words with judge predictions:",
        "",
        "### Pre-Failure Examples (Label = 1: Still Going as Intended)",
        "",
    ])
    for idx, it in enumerate(sampled_pre, 1):
        j_p = judge_cached.get(it.item_id)
        j_str = f"{j_p:.2f}" if j_p is not None else "N/A"
        clip_name = it.meta.get("clip_id", it.item_id)
        t_onset = it.meta.get("t", 0.0)
        lines.append(f"{idx}. **`{it.item_id}`** (Group: `{it.group_id}`)")
        lines.append(f"   - Clip: `{clip_name}` (Failure onset $t = {t_onset:.2f}\\text{{ s}}$)")
        lines.append(f"   - Window: `[{it.action_window_sec[0]:.2f}, {it.action_window_sec[1]:.2f}]` s (pre-failure window $[t-4, t-1]$)")
        lines.append("   - Action Description: Person is engaged in everyday activity prior to accident onset.")
        lines.append(f"   - Local Judge P(as intended): `{j_str}`")
        lines.append("")

    lines.extend([
        "### Post-Failure Examples (Label = 0: Failure Occurred)",
        "",
    ])
    for idx, it in enumerate(sampled_post, 1):
        j_p = judge_cached.get(it.item_id)
        j_str = f"{j_p:.2f}" if j_p is not None else "N/A"
        clip_name = it.meta.get("clip_id", it.item_id)
        t_onset = it.meta.get("t", 0.0)
        lines.append(f"{idx}. **`{it.item_id}`** (Group: `{it.group_id}`)")
        lines.append(f"   - Clip: `{clip_name}` (Failure onset $t = {t_onset:.2f}\\text{{ s}}$)")
        lines.append(f"   - Window: `[{it.action_window_sec[0]:.2f}, {it.action_window_sec[1]:.2f}]` s (post-failure window $[t, t+3]$)")
        lines.append("   - Action Description: The unintended failure / slip / drop has initiated or completed.")
        lines.append(f"   - Local Judge P(as intended): `{j_str}`")
        lines.append("")

    target_out = out_path or (Path("docs/evals") / f"{today}_oops_h1.md")
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {target_out}")
    return target_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Oops! H1 evaluation report")
    parser.add_argument("--date", default=None, help="Report date (YYYY-MM-DD)")
    parser.add_argument("--scorecard", default="results/scorecard.jsonl", help="Scorecard JSONL path")
    parser.add_argument("--out", default=None, help="Output markdown path")
    args = parser.parse_args()

    generate_oops_report(
        date_str=args.date,
        scorecard_path=Path(args.scorecard) if args.scorecard else None,
        out_path=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
