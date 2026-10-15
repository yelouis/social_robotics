from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import numpy as np

from config import DATA_ROOT
from harness.items import read_items
from harness.splits import load_split


def prediction_check(rows: Dict[str, dict]) -> List[str]:
    """Check hypothesis predictions against scorecard rows.

    Returns lines in order:
    1. One Clause 1 line per judge condition with a non-null value, in the order
       `judge`, `judge-large`, `judge-frontier`:
       Held iff ci_low > 0.5 and AUROC >= action_probe_auroc - 0.05.
    2. Clause 2 line for Delta approx 0:
       Held iff ci_low <= 0 <= ci_high.
    3. For each judge condition whose value is null:
       - **<condition>:** <its notes>
    """
    judge_conditions = ["judge", "judge-large", "judge-frontier"]
    action_probe_row = rows.get("action-probe")
    action_probe_auroc = (
        action_probe_row.get("value")
        if action_probe_row and action_probe_row.get("value") is not None
        else None
    )

    clause1_lines: List[str] = []
    null_lines: List[str] = []

    for cond in judge_conditions:
        if cond in rows:
            r = rows[cond]
            val = r.get("value")
            if val is not None:
                ci_low = r.get("ci_low")
                ci_high = r.get("ci_high")
                n_items = r.get("n_items", 0)
                held = False
                if ci_low is not None and ci_low > 0.5:
                    if action_probe_auroc is not None:
                        held = val >= (action_probe_auroc - 0.05)
                    else:
                        held = True
                verdict = "held" if held else "did not hold"
                clause1_lines.append(
                    f'- **Clause 1, "the judge is strong" ({cond}):** AUROC {val:.3f} [{ci_low:.3f}, {ci_high:.3f}] ({n_items} scored) → {verdict}'
                )
            else:
                notes = r.get("notes", "")
                null_lines.append(f"- **{cond}:** {notes}")

    clause2_lines: List[str] = []
    delta_row = rows.get("fusion_minus_action_best")
    if delta_row:
        d_val = delta_row.get("value")
        d_low = delta_row.get("ci_low")
        d_high = delta_row.get("ci_high")
        notes = delta_row.get("notes", "")
        if "action_best=" in notes:
            best_name = notes.split("action_best=")[1].split(";")[0].strip()
        elif ";" in notes:
            best_name = notes.split(";")[0].strip()
        else:
            best_name = notes.strip()
        held = (d_low is not None and d_high is not None and d_low <= 0 <= d_high)
        verdict = "held" if held else "did not hold"
        if d_val is not None and d_low is not None and d_high is not None:
            clause2_lines.append(
                f'- **Clause 2, "Δ ≈ 0":** Δ {d_val:.3f} [{d_low:.3f}, {d_high:.3f}] (action_best = {best_name}) → {verdict}'
            )

    return clause1_lines + clause2_lines + null_lines


def compute_oops_wall_clocks(root: Path) -> Dict[str, str]:
    timings: Dict[str, str] = {}

    # 1. Download
    dl_log = root / "raw" / "oops" / "download.log"
    dl_recorded = False
    if dl_log.exists():
        try:
            ts_matches = []
            for line in dl_log.read_text(encoding="utf-8").splitlines():
                m = re.search(r"\[supervise\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]", line)
                if m:
                    ts_matches.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            if len(ts_matches) >= 2:
                dur_s = (ts_matches[-1] - ts_matches[0]).total_seconds()
                timings["Download"] = f"{dur_s:,.1f} s ({dur_s / 60.0:.1f} min)"
                dl_recorded = True
        except Exception:
            pass
    if not dl_recorded:
        timings["Download"] = "not recorded"

    # 2. Features
    feat_dir = root / "features" / "oops"
    if feat_dir.exists():
        for enc in ["siglip-b16-224", "e2v-plus-large"]:
            idx_file = feat_dir / enc / "index.jsonl"
            if idx_file.exists():
                last_ms: Dict[str, float] = {}
                for line in idx_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            d = json.loads(line)
                            iid = d.get("item_id")
                            if iid:
                                last_ms[iid] = float(d.get("elapsed_ms", 0.0))
                        except Exception:
                            pass
                total_s = sum(last_ms.values()) / 1000.0
                timings[f"Features (`{enc}`)"] = f"{total_s:,.1f} s ({total_s / 60.0:.1f} min)"

    # 3. Judge
    judge_dir = root / "judge" / "oops"
    if judge_dir.exists():
        for model_dir in sorted(judge_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            total_ms = 0.0
            for jf in model_dir.glob("*.jsonl"):
                if jf.name == "errors.jsonl":
                    continue
                for line in jf.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            d = json.loads(line)
                            total_ms += float(d.get("elapsed_ms", 0.0))
                        except Exception:
                            pass
            total_s = total_ms / 1000.0
            timings[f"Judge (`{model_dir.name}`)"] = f"{total_s:,.1f} s ({total_s / 60.0:.1f} min)"

    # 4. Probes
    timings["Probes"] = "not recorded (seconds)"

    return timings


def generate_oops_report(
    date_str: Optional[str] = None,
    data_root: Optional[Path] = None,
    scorecard_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    descriptions_path: Optional[Path] = None,
    corrected_date: Optional[str] = None,
) -> Path:
    root = data_root or DATA_ROOT
    today = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    corr_date = corrected_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

    # 2. Judge stats & answers
    split_data = load_split("oops", path_dir="splits")
    test_set = set(split_data.get("test", []))

    judge_dir = root / "judge" / "oops"
    judge_cached: Dict[str, Optional[float]] = {}
    judge_diag_lines: List[str] = []

    if judge_dir.exists():
        for model_dir in sorted(judge_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            # (a) Read answers only from <prompt_hash>.jsonl (exclude errors.jsonl)
            ans_files = [f for f in model_dir.glob("*.jsonl") if f.name != "errors.jsonl"]
            total_answers = 0
            parse_failures = 0
            scored_test_count = 0

            for jf in ans_files:
                for line in jf.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                        total_answers += 1
                        prob = e.get("judge_prob")
                        iid = e.get("item_id")
                        if prob is None:
                            parse_failures += 1
                        elif iid in test_set:
                            scored_test_count += 1
                        if "qwen" in model_dir.name and iid:
                            judge_cached[iid] = prob
                    except Exception:
                        pass

            # Read errors.jsonl separately and classify
            err_file = model_dir / "errors.jsonl"
            err_by_id: Dict[str, str] = {}
            if err_file.exists():
                for line in err_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                        iid = e.get("item_id")
                        err_text = str(e.get("error", ""))
                        if "429" in err_text:
                            cat = "429"
                        elif re.search(r"\b5\d\d\b", err_text):
                            cat = "5xx"
                        elif "ffmpeg" in err_text:
                            cat = "ffmpeg"
                        else:
                            cat = "other"
                        if iid:
                            err_by_id[iid] = cat
                    except Exception:
                        pass

            cat_counts = {"429": 0, "5xx": 0, "ffmpeg": 0, "other": 0}
            for cat in err_by_id.values():
                cat_counts[cat] += 1

            err_parts = []
            for c in ["429", "5xx", "ffmpeg", "other"]:
                count = cat_counts[c]
                if count > 0:
                    err_parts.append(f"{count:,} `{c}` error" + ("s" if count != 1 else ""))
            err_summary = ", ".join(err_parts) if err_parts else "0 errors"

            mname_lower = model_dir.name.lower()
            if "gemini" in mname_lower:
                judge_diag_lines.append(
                    f"- **Frontier Judge (`{model_dir.name}`):** scored {scored_test_count} of {len(test_set)} test items, "
                    f"{parse_failures} parse failures, {err_summary}"
                )
            elif "qwen" in mname_lower:
                judge_diag_lines.append(
                    f"- **Local Judge (`qwen2.5vl:7b`):** {total_answers:,} answers, "
                    f"{parse_failures} parse failures, {err_summary}"
                )
            elif "gemma" in mname_lower:
                judge_diag_lines.append(
                    f"- **Large Local Judge (`gemma4:26b`):** {total_answers:,} answers, "
                    f"{parse_failures} parse failures, {err_summary}"
                )
            else:
                judge_diag_lines.append(
                    f"- **Judge (`{model_dir.name}`):** {total_answers:,} answers, "
                    f"{parse_failures} parse failures, {err_summary}"
                )

    # Compute answer distributions by window type for local judges
    def compute_distribution(model_name: str) -> Optional[Dict[str, Dict[str, int]]]:
        m_dir = judge_dir / model_name
        if not m_dir.exists():
            return None
        counts = {
            "pre": {"100": 0, "90": 0, "80 or below": 0, "0": 0},
            "post": {"100": 0, "90": 0, "80 or below": 0, "0": 0},
        }
        ans_files = [f for f in m_dir.glob("*.jsonl") if f.name != "errors.jsonl"]
        has_any = False
        for jf in ans_files:
            for line in jf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    prob = e.get("judge_prob")
                    if prob is None:
                        continue
                    iid = e.get("item_id", "")
                    w = "pre" if iid.endswith(":pre") else "post"
                    p_pct = round(prob * 100)
                    if p_pct == 100:
                        counts[w]["100"] += 1
                    elif p_pct == 90:
                        counts[w]["90"] += 1
                    elif p_pct == 0:
                        counts[w]["0"] += 1
                    elif p_pct <= 80:
                        counts[w]["80 or below"] += 1
                    else:
                        counts[w]["80 or below"] += 1
                    has_any = True
                except Exception:
                    pass
        return counts if has_any else None

    dist_lines: List[str] = [
        "",
        "### Answer Distribution by Window Type",
        "",
    ]
    for model_key, label in [("qwen2.5vl_7b", "Local Judge (`qwen2.5vl:7b`)"), ("gemma4_26b", "Large Local Judge (`gemma4:26b`)")]:
        dist = compute_distribution(model_key)
        if dist:
            dist_lines.extend([
                f"#### {label}",
                "",
                "| Window | 100 | 90 | 80 or below | 0 |",
                "|---|---|---|---|---|",
                f"| `pre` (before the failure) | {dist['pre']['100']} | {dist['pre']['90']} | {dist['pre']['80 or below']} | {dist['pre']['0']} |",
                f"| `post` (after the failure) | {dist['post']['100']} | {dist['post']['90']} | {dist['post']['80 or below']} | {dist['post']['0']} |",
                "",
            ])

    # 3. 10 example items (5 pre, 5 post, default_rng(0))
    items_path = root / "items" / "oops" / "items.jsonl"
    all_items = read_items(items_path)
    test_items = [it for it in all_items if it.item_id in test_set]

    pre_items = [it for it in test_items if it.label == 1]
    post_items = [it for it in test_items if it.label == 0]

    rng = np.random.default_rng(0)
    pre_sample_idx = rng.choice(len(pre_items), size=min(5, len(pre_items)), replace=False)
    post_sample_idx = rng.choice(len(post_items), size=min(5, len(post_items)), replace=False)

    sampled_pre = [pre_items[i] for i in sorted(pre_sample_idx)]
    sampled_post = [post_items[i] for i in sorted(post_sample_idx)]

    # Load descriptions
    descriptions: Dict[str, str] = {}
    desc_p = descriptions_path or Path("docs/evals/2026-10-10_oops_examples.tsv")
    if desc_p.exists():
        for line in desc_p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    descriptions[parts[0].strip()] = parts[1].strip()

    # Build report markdown
    lines = [
        f"# Oops! H1 Evaluation Report ({today})",
        f"*Corrected {corr_date} (A7b): prediction check split by clause; judge error counts reclassified; wall-clock times and example descriptions added. The numbers are unchanged.*",
        f"*Updated {corr_date} (A7c): judge-large added (Issue 5, option B).*",
        "",
        "## 1. Executive Summary & Prediction Check",
        "",
        "**Prediction Check:**",
        "> *Prediction:* In fail videos where failures are plainly visible in the action frames, the VLM judge is expected to perform strongly, and non-verbal reactions should add little or no incremental signal over the action-only baseline ($\\Delta \\approx 0$).",
        "",
    ]

    # Prediction check lines
    pred_lines = prediction_check(latest_rows)
    lines.extend(pred_lines)
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
        "judge-large",
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
    lines.append("react-nonverbal on Oops! is not evidence of reaction signal: the reaction window equals the action window, so the post-failure audio contains the failure's own sound and any compilation music (02_data_sources.md → Oops! → As built).")
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
    ])
    lines.extend(judge_diag_lines)
    lines.extend(dist_lines)
    lines.append("")

    # Wall-clock timing
    lines.extend([
        "## 5. Wall-Clock Compute Time",
        "",
    ])
    wall_clocks = compute_oops_wall_clocks(root)
    for stage, dur_str in wall_clocks.items():
        lines.append(f"- **{stage}:** {dur_str}")
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
        desc = descriptions.get(it.item_id, "not described")
        lines.append(f"{idx}. **`{it.item_id}`** (Group: `{it.group_id}`)")
        lines.append(f"   - Clip: `{clip_name}` (Failure onset $t = {t_onset:.2f}\\text{{ s}}$)")
        lines.append(f"   - Window: `[{it.action_window_sec[0]:.2f}, {it.action_window_sec[1]:.2f}]` s (pre-failure window $[t-4, t-1]$)")
        lines.append(f"   - Seen in the frames: {desc}")
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
        desc = descriptions.get(it.item_id, "not described")
        lines.append(f"{idx}. **`{it.item_id}`** (Group: `{it.group_id}`)")
        lines.append(f"   - Clip: `{clip_name}` (Failure onset $t = {t_onset:.2f}\\text{{ s}}$)")
        lines.append(f"   - Window: `[{it.action_window_sec[0]:.2f}, {it.action_window_sec[1]:.2f}]` s (post-failure window $[t, t+3]$)")
        lines.append(f"   - Seen in the frames: {desc}")
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
    parser.add_argument("--descriptions", default="docs/evals/2026-10-10_oops_examples.tsv", help="Path to TSV containing example descriptions")
    parser.add_argument("--data-root", default=None, help="Root path containing dataset artifacts")
    args = parser.parse_args()

    generate_oops_report(
        date_str=args.date,
        data_root=Path(args.data_root) if args.data_root else None,
        scorecard_path=Path(args.scorecard) if args.scorecard else None,
        out_path=Path(args.out) if args.out else None,
        descriptions_path=Path(args.descriptions) if args.descriptions else None,
    )


if __name__ == "__main__":
    main()
