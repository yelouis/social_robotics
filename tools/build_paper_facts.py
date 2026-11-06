#!/usr/bin/env python3
"""tools/build_paper_facts.py — Build docs/paper/facts.json, tables T1..T6, and figures F1..F3.

Implements Item E1 (06_writeup.md §4–§5, agent_execution_guide.md §3).
Strictly read-only over DATA_ROOT and results/scorecard.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.items import read_items  # noqa: E402
from harness.reporting import leak_check_label  # noqa: E402
from harness.splits import load_split  # noqa: E402
from tools.report_holoassist import prediction_check_hidden  # noqa: E402
from tools.report_oops import prediction_check  # noqa: E402
from tools.report_reflex import evaluate_kill_rule, prediction_check_reflex  # noqa: E402
from tools.report_wave_d import h3_verdict  # noqa: E402


RECORDED_FACTS: Dict[str, dict] = {
    "wave_a/audible_strata": {
        "value": {
            "spoke_value": 0.012,
            "spoke_ci_low": -0.004,
            "spoke_ci_high": 0.030,
            "silent_value": -0.026,
            "without_drop_value": 0.025,
        },
        "source": "docs/ongoing_general_errors.md §2 Resolved entry 6 (Issue 8)",
        "quote": (
            "The designer's exploratory strata showed the gain is mostly whether the instructor spoke: "
            "+0.012 [−0.004, 0.030] where they spoke, −0.026 where they were silent, and +0.025 without `drop`."
        ),
    },
    "wave_d/head_zero_shot": {
        "value": {
            "react": {"value": 0.508, "ci_low": 0.475, "ci_high": 0.540},
            "shuffled": {"value": 0.521, "ci_low": 0.487, "ci_high": 0.554},
            "truth": {"value": 0.491, "ci_low": 0.457, "ci_high": 0.522},
            "cosine": 0.0,
        },
        "source": "docs/04_wave_d_robot_check.md §8",
        "quote": (
            "**Each human head on its own, scored on the robot test set:** "
            "`react` 0.508 [0.475, 0.540], `shuffled` 0.521 [0.487, 0.554], `truth` 0.491 [0.457, 0.522]."
        ),
    },
    "wave_d/well_posed_head": {
        "value": {
            "audible_train": {"value": 0.550, "ci_low": 0.515, "ci_high": 0.585},
            "full_train": {"value": 0.472, "ci_low": 0.437, "ci_high": 0.509},
        },
        "source": "docs/04_wave_d_robot_check.md §8",
        "quote": (
            "**Well-posed true-label heads:** one fit on `holoassist_audible` train (1,500 + 1,500) "
            "scores 0.550 [0.515, 0.585] on robot test; one fit on full `holoassist` train scores 0.472 [0.437, 0.509]."
        ),
    },
    "wave_d/head_norms": {
        "value": {
            "react": 0.99,
            "shuffled": 1.03,
            "truth": 26.9,
            "move_react_shuffled": 0.015,
            "move_truth": 0.33,
            "truth_in_sample_auroc": 0.997,
        },
        "source": "docs/04_wave_d_robot_check.md §8",
        "quote": "The norms are `react` 0.99, `shuffled` 1.03 and `truth` 26.9.",
    },
    "wave_d/pool_mistakes": {
        "value": {
            "mistakes": 104,
            "total": 20000,
            "rate": 0.0052,
            "rate_pct": 0.52,
            "audible_mistakes_taken_pct": 92,
        },
        "source": "docs/04_wave_d_robot_check.md §8",
        "quote": (
            "Wave A's balanced `holoassist_audible` sample had taken about 92% of the audible sessions' mistakes, "
            "so the pool has 104 mistakes in 20,000 (0.52%), and none in the sessions the labeler never saw."
        ),
    },
    "holoassist/step_prior_shuffled_null": {
        "value": {
            "mean": 0.496,
            "sd": 0.030,
            "seeds": 50,
            "below_047_pct": 20,
            "threshold": 0.47,
        },
        "source": "docs/03_eval_harness.md §11.9",
        "quote": (
            "Over 50 permutation seeds the same control averages 0.496 (SD 0.030), "
            "and 20% of seeds fall below 0.47 (designer, October 21)."
        ),
    },
}


def validate_recorded_facts(recorded: Dict[str, dict], repo_root: Path = REPO_ROOT) -> Dict[str, dict]:
    """Assert that every recorded fact's quote appears verbatim in its source file."""
    for key, entry in recorded.items():
        if not isinstance(entry, dict) or "source" not in entry or "quote" not in entry or "value" not in entry:
            raise ValueError(f"Malformed recorded entry for {key!r}")
        src_spec = str(entry["source"])
        file_part = src_spec.split()[0].rstrip(",")
        candidates = [repo_root / file_part, repo_root / "docs" / file_part]
        src_path: Optional[Path] = None
        for cand in candidates:
            if cand.is_file():
                src_path = cand
                break
        if src_path is None:
            raise ValueError(f"Recorded fact {key!r}: source file not found for {src_spec!r}")
        text = src_path.read_text(encoding="utf-8")
        quote = str(entry["quote"])
        if quote not in text:
            raise ValueError(f"Recorded fact {key!r}: verbatim quote not found in {src_path}: {quote!r}")
    return recorded


def load_latest_scorecard_rows(scorecard_path: Path) -> Dict[str, Dict[str, dict]]:
    """For each (dataset, condition), select the row with the latest ts."""
    latest_raw: Dict[Tuple[str, str], dict] = {}
    if scorecard_path.exists():
        for line in scorecard_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ds = str(row.get("dataset", ""))
            cond = str(row.get("condition", ""))
            ts = str(row.get("ts", ""))
            k = (ds, cond)
            if k not in latest_raw or ts >= str(latest_raw[k].get("ts", "")):
                latest_raw[k] = row

    result: Dict[str, Dict[str, dict]] = {}
    for (ds, cond), row in latest_raw.items():
        result.setdefault(ds, {})[cond] = {
            "value": row.get("value"),
            "ci_low": row.get("ci_low"),
            "ci_high": row.get("ci_high"),
            "n_items": row.get("n_items"),
            "n_groups": row.get("n_groups"),
            "ts": row.get("ts", ""),
            "notes": row.get("notes", ""),
        }
    return result


def _fmt_cell(row: Optional[dict]) -> str:
    if not row or row.get("value") is None:
        return "n/a"
    v = float(row["value"])
    lo = row.get("ci_low")
    hi = row.get("ci_high")
    if lo is not None and hi is not None:
        return f"{v:.3f} [{float(lo):.3f}, {float(hi):.3f}]"
    return f"{v:.3f}"


def _build_dataset_facts(data_root: Path, splits_dir: Path) -> Dict[str, Any]:
    ds_facts: Dict[str, Any] = {}
    h1_datasets = ["oops", "holoassist", "holoassist_audible", "reflex", "reflex_resolve"]
    for ds in h1_datasets:
        items_path = data_root / "items" / ds / "items.jsonl"
        all_items = read_items(items_path)
        imap = {it.item_id: it for it in all_items}
        sp = load_split(ds, path_dir=splits_dir)
        tr_items = [imap[i] for i in sp["train"] if i in imap]
        te_items = [imap[i] for i in sp["test"] if i in imap]

        n_items = len(all_items)
        n_pos_all = sum(1 for it in all_items if it.label == 1)
        n_neg_all = n_items - n_pos_all

        n_train = len(tr_items)
        n_train_groups = len(set(it.group_id for it in tr_items))
        n_train_pos = sum(1 for it in tr_items if it.label == 1)
        n_train_neg = n_train - n_train_pos

        n_test = len(te_items)
        n_test_groups = len(set(it.group_id for it in te_items))
        n_test_pos = sum(1 for it in te_items if it.label == 1)
        n_test_neg = n_test - n_test_pos
        test_pos_rate = (n_test_pos / n_test) if n_test > 0 else 0.0

        ds_facts[ds] = {
            "items": n_items,
            "positives": n_pos_all,
            "negatives": n_neg_all,
            "train_items": n_train,
            "train_groups": n_train_groups,
            "train_positives": n_train_pos,
            "train_negatives": n_train_neg,
            "test_items": n_test,
            "test_groups": n_test_groups,
            "test_positives": n_test_pos,
            "test_negatives": n_test_neg,
            "test_pos_rate": test_pos_rate,
        }

    # roboreward from stats.json + splits/roboreward.json
    rr_stats_path = data_root / "items" / "roboreward" / "stats.json"
    rr_items_path = data_root / "items" / "roboreward" / "items.jsonl"
    rr_stats = json.loads(rr_stats_path.read_text(encoding="utf-8")) if rr_stats_path.exists() else {}
    rr_all = read_items(rr_items_path) if rr_items_path.exists() else []
    rr_map = {it.item_id: it for it in rr_all}
    rr_sp = load_split("roboreward", path_dir=splits_dir)
    rr_te = [rr_map[i] for i in rr_sp["test"] if i in rr_map]
    rr_test_groups = len(set(it.group_id for it in rr_te))
    n_train_pool = int(rr_stats.get("n_train_pool", 4000))
    n_train_pos = int(rr_stats.get("n_train_success", 2000))
    n_train_neg = int(rr_stats.get("n_train_failure", 2000))
    n_test_rr = int(rr_stats.get("n_test", 1517))
    n_test_pos_rr = int(rr_stats.get("n_test_success", 527))
    n_test_neg_rr = int(rr_stats.get("n_test_failure", 990))

    ds_facts["roboreward"] = {
        "items": n_train_pool + n_test_rr,
        "train_items": n_train_pool,
        "train_positives": n_train_pos,
        "train_negatives": n_train_neg,
        "test_items": n_test_rr,
        "test_groups": rr_test_groups,
        "test_positives": n_test_pos_rr,
        "test_negatives": n_test_neg_rr,
        "test_pos_rate": n_test_pos_rr / n_test_rr if n_test_rr > 0 else 0.0,
    }
    return ds_facts


def _build_named_facts(data_root: Path) -> Dict[str, Any]:
    # 1. holoassist/verb_drop
    ha_items = read_items(data_root / "items" / "holoassist" / "items.jsonl")
    drop_items = [it for it in ha_items if it.meta.get("verb") == "drop"]
    drop_total = len(drop_items)
    drop_mistakes = sum(1 for it in drop_items if it.label == 0)
    drop_rate = (drop_mistakes / drop_total) if drop_total > 0 else 0.0

    # 2. reflex/reaction_frames_mean
    reflex_items = read_items(data_root / "items" / "reflex" / "items.jsonl")
    fail_lens = [
        int(it.meta["reaction_frames"][1]) - int(it.meta["reaction_frames"][0]) + 1
        for it in reflex_items
        if it.label == 0
    ]
    pre_lens = [
        int(it.meta["reaction_frames"][1]) - int(it.meta["reaction_frames"][0]) + 1
        for it in reflex_items
        if it.label == 1
    ]
    fail_mean = float(np.mean(fail_lens))
    pre_mean = float(np.mean(pre_lens))

    # 3 & 4. reflex_resolve resolution by verb and by Explanation Level
    rr_items = read_items(data_root / "items" / "reflex_resolve" / "items.jsonl")
    by_verb: Dict[str, Dict[str, Any]] = {}
    for v in ["pick", "carry", "place"]:
        sub = [it for it in rr_items if it.meta.get("verb") == v]
        res = sum(1 for it in sub if it.label == 1)
        tot = len(sub)
        by_verb[v] = {
            "resolved": res,
            "total": tot,
            "rate": (res / tot) if tot > 0 else 0.0,
            "rate_pct": (100.0 * res / tot) if tot > 0 else 0.0,
        }

    by_level: Dict[str, Dict[str, Any]] = {}
    for lvl in ["high", "medium", "low", "zero"]:
        sub = [it for it in rr_items if it.meta.get("noun") == lvl]
        res = sum(1 for it in sub if it.label == 1)
        tot = len(sub)
        by_level[lvl] = {
            "resolved": res,
            "total": tot,
            "rate": (res / tot) if tot > 0 else 0.0,
            "rate_pct": (100.0 * res / tot) if tot > 0 else 0.0,
        }

    return {
        "holoassist/verb_drop": {
            "total": drop_total,
            "mistakes": drop_mistakes,
            "mistake_rate": drop_rate,
            "mistake_rate_pct": 100.0 * drop_rate,
        },
        "reflex/reaction_frames_mean": {
            "fail": round(fail_mean, 1),
            "pre": round(pre_mean, 1),
            "fail_exact": fail_mean,
            "pre_exact": pre_mean,
        },
        "reflex_resolve/resolved_by_verb": by_verb,
        "reflex_resolve/resolved_by_level": by_level,
    }


def _build_tables(
    scorecard: Dict[str, Dict[str, dict]],
    posthoc: Dict[str, Any],
    ds_facts: Dict[str, Any],
    pred_checks: Dict[str, Any],
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    # T1: Datasets
    meta_rows = [
        ("oops", "Visible", "Non-verbal audio (`e2v-plus-large`)"),
        ("holoassist", "Hidden / partly visible", "Instructor voice (`e2v-plus-large`), speech presence (`react-spoke`)"),
        ("holoassist_audible", "Hidden / partly visible", "Instructor voice (`e2v-plus-large`), speech presence (`react-spoke`)"),
        ("reflex", "Partly visible", "Face/gaze/posture (`reflex-face-v1`), speech prosody (`reflex-speech-v1`)"),
        ("reflex_resolve", "Hidden", "Face/gaze/posture (`reflex-face-v1`), speech prosody (`reflex-speech-v1`)"),
        ("roboreward", "Visible", "None (robot rollouts; H3 transfer target)"),
    ]
    t1_lines = [
        "| Name | Outcome visibility | Reaction channel(s) | Items | Train / test items | Test groups | Test positive rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, vis, chan in meta_rows:
        df = ds_facts[name]
        rate_str = f"{df['test_pos_rate']:.3f} ({df['test_positives']}/{df['test_items']})"
        t1_lines.append(
            f"| `{name}` | {vis} | {chan} | {df['items']:,} | {df['train_items']:,} / {df['test_items']:,} | {df['test_groups']} | {rate_str} |"
        )
    (tables_dir / "T1.md").write_text("\n".join(t1_lines) + "\n", encoding="utf-8")

    # T2: H1 results
    h1_cols = ["oops", "holoassist", "holoassist_audible", "reflex", "reflex_resolve"]
    t2_conds = [
        "judge",
        "judge-large",
        "action-probe",
        "step-prior",
        "action-fusion",
        "react-nonverbal",
        "react-face",
        "react-spoke",
        "fusion",
        "fusion_minus_action_fusion",
        "fusion_minus_action_best",
    ]
    t2_lines = [
        "| Condition | `oops` | `holoassist` | `holoassist_audible` | `reflex` | `reflex_resolve` |",
        "|---|---|---|---|---|---|",
    ]
    for cond in t2_conds:
        cells = [_fmt_cell(scorecard.get(ds, {}).get(cond)) for ds in h1_cols]
        t2_lines.append(f"| `{cond}` | " + " | ".join(cells) + " |")
    (tables_dir / "T2.md").write_text("\n".join(t2_lines) + "\n", encoding="utf-8")

    # T3: Pre-registered checks per H1 dataset
    t3_lines = [
        "| Dataset | Check | Verdict string |",
        "|---|---|---|",
    ]
    for ds in ["oops", "holoassist", "holoassist_audible", "reflex", "reflex_resolve"]:
        for line in pred_checks[ds]:
            clean = line.lstrip("- ").strip()
            if clean.startswith("**Context") or clean.startswith("**Caveat") or clean.startswith("**judge-frontier"):
                continue
            if ":**" in clean:
                chk, rest = clean.split(":**", 1)
                chk_name = chk.replace("**", "").strip()
                t3_lines.append(f"| `{ds}` | {chk_name} | {rest.strip()} |")
            else:
                t3_lines.append(f"| `{ds}` | Check | {clean} |")
    kr_status, kr_msg = pred_checks["reflex_kill_rule"]
    t3_lines.append(f"| `reflex` / `reflex_resolve` | Wave B Kill Rule | **{kr_status}** — {kr_msg} |")
    (tables_dir / "T3.md").write_text("\n".join(t3_lines) + "\n", encoding="utf-8")

    # T4: REFLEX post-hoc diagnostics
    t4_lines = [
        "| Diagnostic key | Value [95% CI] | `n_test` |",
        "|---|---|---|",
    ]
    for k, v in posthoc.items():
        if k == "reflex_resolve.nonverbal_shuffled_null":
            val_s = (
                f"mean {v['mean']:.3f} (SD {v['sd']:.3f}, range [{v['min']:.3f}, {v['max']:.3f}], "
                f"seed 0 = {v['seed0']:.3f}; true {v['true_value']:.3f} beats {v['true_beats']}/50)"
            )
            t4_lines.append(f"| `{k}` | {val_s} | 140 |")
        else:
            val_s = f"{v['value']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
            t4_lines.append(f"| `{k}` | {val_s} | {v['n_test']} |")
    (tables_dir / "T4.md").write_text("\n".join(t4_lines) + "\n", encoding="utf-8")

    # T5: Wave D
    rr_sc = scorecard.get("roboreward", {})
    t5_lines = [
        "| Budget `N` / Reference | `detector:none` | `gain:react` | `gain:shuffled` | `gain:truth` | `react_minus_shuffled` |",
        "|---|---|---|---|---|---|",
    ]
    for n_b in [50, 100, 200, 500, 1000]:
        c_none = _fmt_cell(rr_sc.get(f"detector:none:N={n_b}"))
        c_g_react = _fmt_cell(rr_sc.get(f"gain:react:N={n_b}"))
        c_g_shuf = _fmt_cell(rr_sc.get(f"gain:shuffled:N={n_b}"))
        c_g_truth = _fmt_cell(rr_sc.get(f"gain:truth:N={n_b}"))
        c_rms = _fmt_cell(rr_sc.get(f"react_minus_shuffled:N={n_b}"))
        t5_lines.append(f"| `N={n_b}` | {c_none} | {c_g_react} | {c_g_shuf} | {c_g_truth} | {c_rms} |")
    t5_lines.append(f"| `robot-full` | {_fmt_cell(rr_sc.get('robot-full'))} | n/a | n/a | n/a | n/a |")
    t5_lines.append(f"| `judge-large` | {_fmt_cell(rr_sc.get('judge-large'))} | n/a | n/a | n/a | n/a |")
    (tables_dir / "T5.md").write_text("\n".join(t5_lines) + "\n", encoding="utf-8")

    # T6: Shuffled controls
    t6_lines = [
        "| Dataset | Passing (`ci_low <= 0.5`) / Run | Failing `:shuffled` row(s) & 50-seed permutation null |",
        "|---|---|---|",
    ]
    for ds in ["oops", "holoassist", "holoassist_audible", "reflex", "reflex_resolve"]:
        ds_rows = scorecard.get(ds, {})
        sh_rows = [
            (c, r)
            for c, r in ds_rows.items()
            if c.endswith(":shuffled") and r.get("value") is not None
        ]
        n_run = len(sh_rows)
        n_pass = 0
        fail_notes: List[str] = []
        for c, r in sh_rows:
            lk = leak_check_label(r.get("value"), r.get("ci_low"), r.get("ci_high"))
            if lk.startswith("FAIL"):
                if ds == "reflex_resolve" and c == "react-nonverbal:shuffled":
                    nv_null = posthoc["reflex_resolve.nonverbal_shuffled_null"]
                    fail_notes.append(
                        f"`{c}` {_fmt_cell(r)} (`{lk}`); 50-seed null mean {nv_null['mean']:.3f}, "
                        f"SD {nv_null['sd']:.3f}, range [{nv_null['min']:.3f}, {nv_null['max']:.3f}], "
                        f"seed 0 = {nv_null['seed0']:.3f}, true {nv_null['true_value']:.3f} beats {nv_null['true_beats']}/50"
                    )
                else:
                    fail_notes.append(f"`{c}` {_fmt_cell(r)} (`{lk}`)")
            else:
                n_pass += 1
        fail_col = "; ".join(fail_notes) if fail_notes else "none"
        t6_lines.append(f"| `{ds}` | {n_pass} / {n_run} | {fail_col} |")
    (tables_dir / "T6.md").write_text("\n".join(t6_lines) + "\n", encoding="utf-8")


def _build_figures(
    scorecard: Dict[str, Dict[str, dict]],
    posthoc: Dict[str, Any],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # F1: Forest plot of ablation Delta (fusion_minus_action_fusion) for the 5 H1 datasets
    # Note: holoassist was stopped at the audio check; we also show reflex length-controlled Delta so 5 intervals are plotted.
    f1_entries = [
        ("oops (n=1,072)", scorecard["oops"]["fusion_minus_action_fusion"], "o", "#222222"),
        ("holoassist (audio check FAIL; not run)", None, "x", "#888888"),
        ("holoassist_audible (n=1,420)", scorecard["holoassist_audible"]["fusion_minus_action_fusion"], "s", "#000000"),
        ("reflex (n=345)", scorecard["reflex"]["fusion_minus_action_fusion"], "^", "#333333"),
        ("reflex_resolve (n=173)", scorecard["reflex_resolve"]["fusion_minus_action_fusion"], "D", "#444444"),
        ("reflex: [length+face] - [length] (n=345)", posthoc["reflex.length_plus_face_minus_length"], "v", "#666666"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    y_positions = np.arange(len(f1_entries))[::-1]
    ax.axvline(0.0, color="#333333", linestyle="-", linewidth=1.1, label="Zero gain (Δ = 0.00)")
    ax.axvline(0.03, color="#555555", linestyle="--", linewidth=1.1, label="H1 pass bar (Δ = +0.03)")

    for y_pos, (label, row, marker, color) in zip(y_positions, f1_entries):
        if row is not None and row.get("value") is not None:
            v = float(row["value"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            ax.errorbar(
                v,
                y_pos,
                xerr=[[v - lo], [hi - v]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=4,
                markersize=6.5,
            )
        else:
            ax.text(0.002, y_pos, "not run (audio check < +3.5 dB)", va="center", ha="left", fontsize=8.5, color="#555555")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([e[0] for e in f1_entries], fontsize=9)
    ax.set_xlabel("Paired Ablation Gain Δ AUROC (fusion − action-fusion, dimensionless)", fontsize=9.5)
    ax.set_title("H1 Reaction Ablation Gain Across Datasets (95% Group-Bootstrap CI)", fontsize=10.5)
    ax.set_xlim(-0.12, 0.10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "f1_ablation.png", dpi=150)
    plt.close(fig)

    # F2: REFLEX react-face AUROC under original windows, means only, equal-length start, equal-length end, and window length alone
    f2_entries = [
        ("react-face (original windows, n=345)", posthoc["reflex.react_face"], "o", "#111111"),
        ("react-face (50 _mean dims only, n=345)", posthoc["reflex.face_means_only"], "s", "#444444"),
        ("react-face (equal-length start, n=342)", posthoc["reflex.face_equal_length_start"], "^", "#555555"),
        ("react-face (equal-length end, n=343)", posthoc["reflex.face_equal_length_end"], "v", "#666666"),
        ("window length alone (-L, n=345)", posthoc["reflex.window_length"], "D", "#000000"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    x_pos = np.arange(len(f2_entries))
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.1, label="Chance (AUROC = 0.50)")
    for idx, (label, row, marker, color) in enumerate(f2_entries):
        v = float(row["value"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        ax.errorbar(
            idx,
            v,
            yerr=[[v - lo], [hi - v]],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=5,
            markersize=7,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [
            "Original\nwindows",
            "50 _mean\ndims only",
            "Equal-length\n(start)",
            "Equal-length\n(end)",
            "Window length\nalone (-L)",
        ],
        fontsize=8.5,
    )
    ax.set_ylabel("Test AUROC (dimensionless)", fontsize=9.5)
    ax.set_title("REFLEX Immediate Reaction (reflex): Face Probe vs. Window-Length Artifact", fontsize=10.2)
    ax.set_ylim(0.42, 0.84)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "f2_window_length.png", dpi=150)
    plt.close(fig)

    # F3: Wave D test AUROC against N (log axis) for none, react, shuffled, truth detectors with CI bands
    budgets = [50, 100, 200, 500, 1000]
    priors = [
        ("none", "No human prior (w0 = 0)", "o", "-", "#000000"),
        ("react", "Reaction pseudo-label prior", "s", "--", "#333333"),
        ("shuffled", "Shuffled pseudo-label prior", "^", "-.", "#666666"),
        ("truth", "True human-label prior", "D", ":", "#888888"),
    ]
    rr_sc = scorecard["roboreward"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for key, label, marker, ls, color in priors:
        vals = [float(rr_sc[f"detector:{key}:N={n}"]["value"]) for n in budgets]
        los = [float(rr_sc[f"detector:{key}:N={n}"]["ci_low"]) for n in budgets]
        his = [float(rr_sc[f"detector:{key}:N={n}"]["ci_high"]) for n in budgets]
        ax.plot(budgets, vals, marker=marker, linestyle=ls, color=color, linewidth=1.6, markersize=6, label=label)
        ax.fill_between(budgets, los, his, color=color, alpha=0.12)

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Labeled Robot Training Clips N (clips, log scale)", fontsize=9.5)
    ax.set_ylabel("20-Draw Mean Test AUROC (dimensionless)", fontsize=9.5)
    ax.set_title("Wave D (RoboReward, n_test=1,517): Robot Success Detection vs. Label Budget", fontsize=10.2)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "f3_wave_d.png", dpi=150)
    plt.close(fig)


def build_all_paper_artifacts(
    data_root: Path,
    scorecard_path: Path = Path("results/scorecard.jsonl"),
    splits_dir: Path = Path("splits"),
    posthoc_path: Path = Path("docs/evals/2026-11-04_reflex_posthoc.json"),
    paper_dir: Path = Path("docs/paper"),
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    recorded = validate_recorded_facts(RECORDED_FACTS, repo_root=repo_root)
    scorecard = load_latest_scorecard_rows(scorecard_path)
    posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
    ds_facts = _build_dataset_facts(data_root, splits_dir)
    named_facts = _build_named_facts(data_root)

    pred_checks: Dict[str, Any] = {
        "oops": prediction_check(scorecard.get("oops", {})),
        "holoassist": prediction_check_hidden(
            scorecard.get("holoassist", {}),
            dataset="holoassist",
            data_root=data_root,
            splits_dir=splits_dir,
        ),
        "holoassist_audible": prediction_check_hidden(
            scorecard.get("holoassist_audible", {}),
            dataset="holoassist_audible",
            data_root=data_root,
            splits_dir=splits_dir,
        ),
        "reflex": prediction_check_reflex(scorecard.get("reflex", {}), dataset="reflex"),
        "reflex_resolve": prediction_check_reflex(scorecard.get("reflex_resolve", {}), dataset="reflex_resolve"),
        "reflex_kill_rule": list(
            evaluate_kill_rule(scorecard.get("reflex", {}), scorecard.get("reflex_resolve", {}))
        ),
        "roboreward_h3": list(h3_verdict(scorecard.get("roboreward", {}))),
    }

    constants = {
        "chance_auroc": 0.5,
        "pass_bar_react_auroc": 0.65,
        "pass_bar_ablation_delta": 0.03,
        "h3_pass_bar_gain": 0.02,
        "bootstrap_resamples": 1000,
        "permutation_seeds": 50,
        "reflex_fail_window_extension_frames": 22,
        "reflex_fps": 4.4,
        "wave_d_budgets": [50, 100, 200, 500, 1000],
        "wave_d_draws_per_budget": 20,
        "reflex_participants": 55,
    }

    computed: Dict[str, Any] = {
        "scorecard": scorecard,
        "posthoc": posthoc,
        "datasets": ds_facts,
        "named": named_facts,
        "prediction_checks": pred_checks,
        "constants": constants,
    }

    # Also populate flat slash-separated keys per 06_writeup.md §4.1
    for ds_name, conds in scorecard.items():
        for cond_name, row_dict in conds.items():
            computed[f"scorecard/{ds_name}/{cond_name}"] = row_dict
            for f_name in ("value", "ci_low", "ci_high", "n_items", "n_groups"):
                computed[f"scorecard/{ds_name}/{cond_name}/{f_name}"] = row_dict.get(f_name)

    for ph_key, ph_dict in posthoc.items():
        computed[f"posthoc/{ph_key}"] = ph_dict
        if isinstance(ph_dict, dict):
            for f_name, f_val in ph_dict.items():
                computed[f"posthoc/{ph_key}/{f_name}"] = f_val

    for nf_key, nf_val in named_facts.items():
        computed[nf_key] = nf_val

    for c_key, c_val in constants.items():
        computed[f"constants/{c_key}"] = c_val

    facts_obj = {
        "computed": computed,
        "recorded": recorded,
    }

    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "facts.json").write_text(json.dumps(facts_obj, indent=2) + "\n", encoding="utf-8")

    _build_tables(scorecard, posthoc, ds_facts, pred_checks, paper_dir / "tables")
    _build_figures(scorecard, posthoc, paper_dir / "figures")

    return facts_obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper facts.json, tables T1..T6, and figures F1..F3.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/Volumes/Extreme SSD/social_robotics")),
        help="Path to DATA_ROOT",
    )
    parser.add_argument("--scorecard", type=Path, default=Path("results/scorecard.jsonl"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--posthoc", type=Path, default=Path("docs/evals/2026-11-04_reflex_posthoc.json"))
    parser.add_argument("--paper-dir", type=Path, default=Path("docs/paper"))
    args = parser.parse_args()

    build_all_paper_artifacts(
        data_root=args.data_root,
        scorecard_path=args.scorecard,
        splits_dir=args.splits_dir,
        posthoc_path=args.posthoc,
        paper_dir=args.paper_dir,
    )
    print(f"Built {args.paper_dir / 'facts.json'}, tables T1..T6, and figures F1..F3")


if __name__ == "__main__":
    main()
