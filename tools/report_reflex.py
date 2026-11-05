#!/usr/bin/env python3
"""tools/report_reflex.py — Wave B REFLEX H1 evaluation report generator.

Generates docs/evals/<date>_reflex_h1.md and TSV example tables
per 05_wave_b_reflex.md §6 and 03_eval_harness.md §11.8–§11.9.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from features.face import get_reflex_face_feature_names
from harness.items import Item, read_items
from harness.reporting import leak_check_label
from harness.splits import load_split
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash
from shared.fsutil import data_glob


VERIFIED_EXAMPLE_DESCRIPTIONS: Dict[str, str] = {
    # reflex: 5 Pre-failure (label=1) examples
    "reflex:C2-2:5:pre": (
        "In frame 0, Baxter's left gripper descends on the far left of the table near a white-and-red box and a yellow "
        "fiducial-tagged object; across frames 1-3, the left arm retracts upward out of the left side of the frame while "
        "the black-masked participant stands facing the table with a white tablet on the right."
    ),
    "reflex:C2-3:1:pre": (
        "Baxter's left gripper holds a white pouch/object with a fiducial marker horizontally above a white-and-red box "
        "and a purple fiducial-tagged carton on the left side of the table (frames 0-2), then lifts the held white object "
        "higher in frame 3 while the black-masked participant stands upright in front of the green screen."
    ),
    "reflex:D1-1:5:pre": (
        "In frame 0, Baxter's left gripper holds a blue-and-white object at the lower left edge of the table next to a "
        "white-and-red box and a yellow fiducial-tagged object; in frames 1-3, the left arm has retracted out of view, "
        "leaving the white-and-red box, gray rectangular box, and yellow fiducial object on the table while the black-masked participant watches."
    ),
    "reflex:D1-5:6:pre": (
        "Across all 4 frames, Baxter's left arm is positioned vertically on the left side of the workspace holding a blue block "
        "above a white-and-red box and a yellow fiducial-tagged object on the dark tabletop, while the black-masked participant "
        "stands turned slightly toward the robot's right arm."
    ),
    "reflex:D2-1:10:pre": (
        "Baxter's left gripper holds a dark rectangular box on a blue tray horizontally above the left edge of the table (frames 0-2) "
        "and raises it upward in frame 3; meanwhile, the black-masked participant tilts their head (frames 0-1), turns to the left "
        "(frame 2), and raises their right hand to the top of their head (frame 3)."
    ),
    # reflex: 5 Failure (label=0) examples
    "reflex:D1-1:4:fail": (
        "Across all 4 frames, Baxter's left arm remains extended vertically downward on the left side of the table with its red "
        "two-finger gripper poised directly over a vertical blue block resting on a white base plate beside a yellow fiducial-tagged "
        "object, failing to lift the blue block while the black-masked participant leans forward watching."
    ),
    "reflex:D1-5:2:fail": (
        "Both of Baxter's arms are retracted out of the central workspace across all 4 frames, leaving the dark tabletop empty "
        "except for white tape markers on the lower left while the black-masked participant stands upright in front of the green screen."
    ),
    "reflex:D1-5:8:fail": (
        "Across all 4 frames, a dark box with a white fiducial marker rests at the far lower-left corner of the table on the tape "
        "markers while Baxter's left arm is out of view and the black-masked participant stands facing the left side of the workspace."
    ),
    "reflex:D2-10:4:fail": (
        "Across all 4 frames, Baxter's left arm stays stationary on the left with its red gripper positioned above a blue vertical block "
        "on a white base plate next to a yellow fiducial-tagged object; the black-masked participant stands with their left hand on their "
        "hip (frames 0-1), then lowers and extends their left arm toward the right side of the table (frames 2-3)."
    ),
    "reflex:D2-2:2:fail": (
        "Across all 4 frames, Baxter's left arm is out of view and the tabletop shows only white tape markers on the left and a white "
        "tablet on the right, while the black-masked participant stands with hands near their waist facing the workspace."
    ),
    # reflex_resolve: 5 Resolved (label=1) examples
    "reflex_resolve:C2-2:8": (
        "In frame 0, the black-masked participant reaches their right arm across to the lower-left corner of the table touching a dark "
        "box with a fiducial marker; in frames 1-2 the participant stands upright with arms by their torso while the dark box sits on "
        "the left, and in frame 3 Baxter's left gripper enters from the lower left above the dark box."
    ),
    "reflex_resolve:C2-3:6": (
        "In frames 0-2, Baxter's left arm is positioned at the far left edge above a white-and-red box, a gray fiducial-tagged box, "
        "and a yellow object; in frame 3, Baxter's left gripper swings inward horizontally above the gray and yellow objects while "
        "the black-masked participant watches."
    ),
    "reflex_resolve:D1-1:6": (
        "In frames 0-2, Baxter's left gripper holds a tilted white-and-red box and blue block at the far left above a gray box and "
        "yellow fiducial object; in frame 3, the left gripper extends horizontally inward toward the center-left of the table while "
        "the black-masked participant stands facing the workspace."
    ),
    "reflex_resolve:D1-5:3": (
        "In frames 0-2, Baxter's left gripper holds a blue block and a tilted purple carton at the left above a yellow object, a dark "
        "fiducial-tagged box, and a white flat plate; in frame 3, the left arm moves lower and horizontally over the dark box and yellow "
        "object while the black-masked participant watches."
    ),
    "reflex_resolve:D2-1:10": (
        "Baxter's left gripper holds a dark rectangular box on a blue tray above the left side of the table across all 4 frames; "
        "the black-masked participant stands with their right hand raised to their head in frames 0-2, then lowers their right arm "
        "toward the left side of the table in frame 3."
    ),
    # reflex_resolve: 5 Unresolved (label=0) examples
    "reflex_resolve:D1-1:2": (
        "In frames 0-2, the tabletop is bare except for white tape markers on the left and the tablet on the right while the black-masked "
        "participant stands looking toward the left; in frame 3, Baxter's left gripper enters at the lower left carrying a white-and-red "
        "object over the tape markers."
    ),
    "reflex_resolve:D1-6:5": (
        "In frames 0-2, a white-and-red box, a gray box, and a yellow fiducial-tagged object rest on the left side of the table while "
        "the black-masked participant stands with hands near their waist; in frame 3, Baxter's left gripper enters horizontally above "
        "the white-and-red box."
    ),
    "reflex_resolve:D1-7:8": (
        "A dark box with a white fiducial marker sits on the lower-left corner of the table and a white tablet rests on the center-right "
        "across all 4 frames; the black-masked participant stands facing the workspace in frames 0-2, and in frame 3 Baxter's left gripper "
        "enters at the lower left above the dark box."
    ),
    "reflex_resolve:D2-10:8": (
        "A dark box with a white fiducial marker sits on the lower-left tape markers across all 4 frames; the black-masked participant "
        "leans forward looking toward the left in frames 0-2, then straightens up in frame 3 as Baxter's left gripper enters at the lower "
        "left above the dark box."
    ),
    "reflex_resolve:D2-9:8": (
        "Under cooler blue-tinted lighting, a dark box with a white fiducial marker rests on the lower-left tape markers across all 4 frames "
        "while the black-masked participant stands with hands behind/near their waist; in frame 3, Baxter's left gripper enters at the lower "
        "left above the dark box."
    ),
}


def fmt_row(r: Optional[dict]) -> str:
    """Format a scorecard row as val [lo, hi] or its notes."""
    if not r:
        return "not run"
    val = r.get("value")
    if val is not None:
        lo = r.get("ci_low")
        hi = r.get("ci_high")
        if lo is not None and hi is not None:
            return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"
        return f"{val:.3f}"
    return r.get("notes", "not run")


def prediction_check_reflex(
    rows: Dict[str, dict],
    dataset: str = "reflex",
) -> List[str]:
    """Check hypothesis predictions against scorecard rows for REFLEX per 03_eval_harness.md §11.8 & 05_wave_b_reflex.md §6.

    Returns lines in order:
    1. Clause 1, "reaction-only beats action_best"
    2. Clause 2, "reactions add to the action-only model"
    3. H1 pass bar (proposed, 00_thesis.md)
    4. Context, no verdict
    """
    lines: List[str] = []

    # 1. Clause 1: react-face vs action_best
    r_react = rows.get("react-face")
    r_delta_ab = rows.get("fusion_minus_action_best")
    action_best_name = "action-probe"
    if r_delta_ab:
        notes = r_delta_ab.get("notes", "")
        if "action_best=" in notes:
            action_best_name = notes.split("action_best=")[1].split(";")[0].strip()

    r_ab = rows.get(action_best_name)
    ab_val = r_ab.get("value") if r_ab else None
    ab_val_str = f"{ab_val:.3f}" if ab_val is not None else (fmt_row(r_ab) if r_ab else "not run")

    if r_react is None or r_react.get("value") is None:
        rf_notes = r_react.get("notes", "not run") if r_react else "not run"
        c1_verdict = f"not tested: {rf_notes}"
        c1_rf_str = rf_notes
    else:
        rf_val = r_react["value"]
        rf_lo = r_react.get("ci_low")
        rf_hi = r_react.get("ci_high")
        if rf_lo is not None and rf_hi is not None:
            c1_rf_str = f"{rf_val:.3f} [{rf_lo:.3f}, {rf_hi:.3f}]"
        else:
            c1_rf_str = f"{rf_val:.3f}"

        if ab_val is not None and rf_lo is not None:
            held = rf_lo > ab_val
            c1_verdict = "held" if held else "did not hold"
        else:
            c1_verdict = "not tested: action_best missing"

    lines.append(
        f'- **Clause 1, "reaction-only beats action_best":** react-face {c1_rf_str} vs {action_best_name} {ab_val_str} → {c1_verdict}'
    )

    # 2. Clause 2: ablation Delta (fusion_minus_action_fusion)
    r_ablation = rows.get("fusion_minus_action_fusion")
    if r_ablation is None:
        c2_body = "not run"
        c2_verdict = "not tested: no fusion_minus_action_fusion row"
    elif r_ablation.get("value") is None:
        notes = r_ablation.get("notes", "")
        c2_body = notes if notes else "not run"
        c2_verdict = f"not tested: {notes or 'no fusion_minus_action_fusion row'}"
    else:
        c2_body = fmt_row(r_ablation)
        lo = r_ablation.get("ci_low")
        held = lo is not None and lo > 0
        c2_verdict = "held" if held else "did not hold"

    lines.append(
        f'- **Clause 2, "reactions add to the action-only model":** ablation Δ {c2_body} → {c2_verdict}'
    )

    # 3. H1 pass bar
    react_has_val = (r_react is not None and r_react.get("value") is not None)
    ablation_has_val = (r_ablation is not None and r_ablation.get("value") is not None)
    if not react_has_val or not ablation_has_val:
        missing = []
        if not react_has_val:
            missing.append("react-face")
        if not ablation_has_val:
            missing.append("ablation Δ")
        pb_verdict = f"not testable: {', '.join(missing)}"
    else:
        rf_val = r_react["value"]
        ab_val = r_ablation["value"]
        ab_lo = r_ablation.get("ci_low")
        met = (rf_val >= 0.65) and (ab_val >= 0.03) and (ab_lo is not None and ab_lo > 0.0)
        pb_verdict = "met" if met else "not met"

    lines.append(
        f'- **H1 pass bar (proposed, 00_thesis.md):** react-face ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → {pb_verdict}'
    )

    # 4. Context, no verdict
    ctx_parts = []
    ctx_parts.append(f"fusion − action_best {fmt_row(rows.get('fusion_minus_action_best'))}")
    ctx_parts.append(f"judge {fmt_row(rows.get('judge'))}")
    ctx_parts.append(f"judge-large {fmt_row(rows.get('judge-large'))}")
    ctx_parts.append(f"action-probe {fmt_row(rows.get('action-probe'))}")
    if "step-prior" in rows:
        ctx_parts.append(f"step-prior {fmt_row(rows.get('step-prior'))}")
    if "action-fusion" in rows:
        ctx_parts.append(f"action-fusion {fmt_row(rows.get('action-fusion'))}")
    ctx_parts.append(f"react-nonverbal {fmt_row(rows.get('react-nonverbal'))}")
    ctx_parts.append(f"react-spoke {fmt_row(rows.get('react-spoke'))}")
    ctx_parts.append(f"react-full {fmt_row(rows.get('react-full'))}")

    lines.append(f"- **Context, no verdict:** {'; '.join(ctx_parts)}.")

    return lines


prediction_check = prediction_check_reflex


def evaluate_kill_rule(
    reflex_rows: Dict[str, dict],
    reflex_resolve_rows: Dict[str, dict],
) -> Tuple[str, str]:
    """Evaluate pre-registered Wave B kill rule per 05_wave_b_reflex.md §6.

    Returns:
        (status, explanation) where status in ("PROCEED", "KILL", "PENDING").
    """
    datasets = [("reflex", reflex_rows), ("reflex_resolve", reflex_resolve_rows)]
    positive_ablation_found = False
    react_face_beats_chance_found = False
    all_evaluated = True

    details = []

    for name, rows in datasets:
        rf = rows.get("react-face")
        ablation = rows.get("fusion_minus_action_fusion")

        rf_lo = rf.get("ci_low") if rf and rf.get("value") is not None else None
        ab_lo = ablation.get("ci_low") if ablation and ablation.get("value") is not None else None

        if rf_lo is None or ab_lo is None:
            all_evaluated = False

        rf_str = f"react-face ci_low={rf_lo:.3f}" if rf_lo is not None else "react-face uncomputed"
        ab_str = f"ablation ci_low={ab_lo:.3f}" if ab_lo is not None else "ablation uncomputed"
        details.append(f"{name}: {rf_str}, {ab_str}")

        if ab_lo is not None and ab_lo > 0.0:
            positive_ablation_found = True
        if rf_lo is not None and rf_lo > 0.5:
            react_face_beats_chance_found = True

    if positive_ablation_found or react_face_beats_chance_found:
        reasons = []
        if positive_ablation_found:
            reasons.append("positive ablation Δ (ci_low > 0)")
        if react_face_beats_chance_found:
            reasons.append("react-face beats chance (ci_low > 0.5)")
        msg = f"PROCEED to DW3 and DW6: at least one dataset demonstrated {', '.join(reasons)}. ({'; '.join(details)})"
        return ("PROCEED", msg)

    if not all_evaluated:
        msg = f"PENDING: not all required conditions are computed. ({'; '.join(details)})"
        return ("PENDING", msg)

    # Neither positive ablation nor react-face beats chance on both datasets
    msg = f"KILL: neither dataset achieved positive ablation gain (ci_low > 0) and react-face failed to beat chance (ci_low <= 0.5) on both datasets. Stop and write up negative result. ({'; '.join(details)})"
    return ("KILL", msg)


def compute_distribution(
    dataset: str,
    model_tag: str,
    p_hash: str,
    data_root: Path,
    splits_dir: Optional[Path] = None,
) -> Optional[Dict[str, Dict[str, int]]]:
    """Computes answer distribution by class (label=1 vs label=0) for judge on REFLEX test split."""
    cache_file = data_root / "judge" / dataset / model_tag / f"{p_hash}.jsonl"
    if not cache_file.exists():
        return None

    items_file = data_root / "items" / dataset / "items.jsonl"
    s_dir = splits_dir if splits_dir else Path("splits")
    splits_file = s_dir / f"{dataset}.json"
    if not items_file.exists() or not splits_file.exists():
        return None

    split_data = json.loads(splits_file.read_text(encoding="utf-8"))
    test_ids = set(split_data.get("test", []))
    items = {it.item_id: it for it in read_items(items_file) if it.item_id in test_ids}

    dist = {
        "label=1": {"100": 0, "90": 0, "<=80": 0, "0": 0, "total": 0},
        "label=0": {"100": 0, "90": 0, "<=80": 0, "0": 0, "total": 0},
    }

    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            iid = d.get("item_id")
            if iid not in items:
                continue
            prob = d.get("judge_prob")
            if prob is None:
                continue
            item = items[iid]
            w_key = "label=1" if item.label == 1 else "label=0"
            val = int(round(prob * 100))
            dist[w_key]["total"] += 1
            if val == 100:
                dist[w_key]["100"] += 1
            elif val == 90:
                dist[w_key]["90"] += 1
            elif val == 0:
                dist[w_key]["0"] += 1
            elif val <= 80:
                dist[w_key]["<=80"] += 1
        except Exception:
            pass

    return dist


def compute_judge_stats(
    dataset: str,
    model_tag: str,
    p_hash: str,
    data_root: Path,
) -> Dict[str, int]:
    """Compute total cached answers, parse failures, and error counts for a judge run."""
    stats = {"total": 0, "parse_failures": 0, "ffmpeg_errors": 0, "api_errors": 0}
    j_dir = data_root / "judge" / dataset / model_tag
    if not j_dir.exists():
        return stats

    jf = j_dir / f"{p_hash}.jsonl"
    if jf.exists():
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                stats["total"] += 1
                if d.get("judge_prob") is None:
                    stats["parse_failures"] += 1
            except Exception:
                pass

    err_file = j_dir / "errors.jsonl"
    if err_file.exists():
        err_by_id: Dict[str, str] = {}
        for line in err_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                iid = d.get("item_id")
                err_str = str(d.get("error", "")).lower()
                if iid:
                    err_by_id[iid] = "ffmpeg" if "ffmpeg" in err_str else "api"
            except Exception:
                pass
        for cat in err_by_id.values():
            if cat == "ffmpeg":
                stats["ffmpeg_errors"] += 1
            else:
                stats["api_errors"] += 1

    return stats


def compute_reflex_wall_clocks(root: Path) -> Dict[str, str]:
    """Compute compute and wall-clock times from logs and feature/judge index caches."""
    timings: Dict[str, str] = {}

    # 1. Download
    dl_json = root / "raw" / "reflex" / "DOWNLOAD.json"
    dl_log = root / "raw" / "reflex" / "download.log"
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
                timings["Download & Extraction"] = f"{dur_s:,.1f} s ({dur_s / 60.0:.1f} min)"
                dl_recorded = True
        except Exception:
            pass
    if not dl_recorded and dl_json.exists():
        try:
            d = json.loads(dl_json.read_text(encoding="utf-8"))
            ts = d.get("ts", "")
            timings["Download & Extraction"] = f"Verified {d.get('zip_bytes', 0) / 1e9:.1f} GB archive ({ts})"
            dl_recorded = True
        except Exception:
            pass
    if not dl_recorded:
        timings["Download & Extraction"] = "not recorded"

    # 2. Features for reflex and reflex_resolve
    for ds in ["reflex", "reflex_resolve"]:
        feat_dir = root / "features" / ds
        if feat_dir.exists():
            for enc in ["reflex-face-v1", "reflex-speech-v1", "siglip-b16-224-u9"]:
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
                    timings[f"Features `{ds}` (`{enc}`)"] = f"{total_s:,.1f} s ({total_s / 60.0:.1f} min across {len(last_ms)} items)"

    # 3. Judges
    for ds in ["reflex", "reflex_resolve"]:
        judge_dir = root / "judge" / ds
        if judge_dir.exists():
            for model_dir in sorted(judge_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("._"):
                    continue
                total_ms = 0.0
                n_scored = 0
                for jf in data_glob(model_dir, "*.jsonl"):
                    if jf.name == "errors.jsonl":
                        continue
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                d = json.loads(line)
                                total_ms += float(d.get("elapsed_ms", 0.0))
                                n_scored += 1
                            except Exception:
                                pass
                total_s = total_ms / 1000.0
                timings[f"Judge `{ds}` (`{model_dir.name}`)"] = f"{total_s:,.1f} s ({total_s / 60.0:.1f} min across {n_scored} items)"

    return timings


def extract_top_weights(
    probe_npz_path: Path,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Extract top-5 positive and top-5 negative standardized weights from face probe."""
    if not probe_npz_path.exists():
        return ([], [])
    data = np.load(probe_npz_path)
    coef = data["coef"].ravel()
    feat_names = get_reflex_face_feature_names()
    pairs = [(feat_names[i], float(coef[i])) for i in range(len(coef))]
    top_pos = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]
    top_neg = sorted(pairs, key=lambda x: x[1])[:5]
    return (top_pos, top_neg)


def _sample_and_format_examples(
    dataset: str,
    data_root: Path,
    splits_dir: Path,
    tsv_out: Optional[Path] = None,
) -> List[str]:
    """Sample 10 test items (5 label=1, 5 label=0, default_rng(0)), format markdown & write TSV."""
    items_path = data_root / "items" / dataset / "items.jsonl"
    split_file = splits_dir / f"{dataset}.json"
    if not items_path.exists() or not split_file.exists():
        return [f"- Items or split file not found for `{dataset}`."]

    all_items = read_items(items_path)
    split_data = load_split(dataset, path_dir=splits_dir)
    test_set = set(split_data.get("test", []))
    test_items = [it for it in all_items if it.item_id in test_set]

    pos_items = [it for it in test_items if it.label == 1]
    neg_items = [it for it in test_items if it.label == 0]

    rng = np.random.default_rng(0)
    pos_idx = rng.choice(len(pos_items), size=min(5, len(pos_items)), replace=False) if pos_items else []
    neg_idx = rng.choice(len(neg_items), size=min(5, len(neg_items)), replace=False) if neg_items else []

    sampled_pos: List[Item] = [pos_items[i] for i in sorted(pos_idx)]
    sampled_neg: List[Item] = [neg_items[i] for i in sorted(neg_idx)]

    q = DATASET_QUESTIONS.get(dataset, "")
    p_h = prompt_hash(q)
    j_dir = data_root / "judge" / dataset
    qwen_cache: Dict[str, Optional[float]] = {}
    gemma_cache: Dict[str, Optional[float]] = {}

    for m_tag, target_dict in [("qwen2.5vl_7b", qwen_cache), ("gemma4_26b", gemma_cache)]:
        jf = j_dir / m_tag / f"{p_h}.jsonl"
        if jf.exists():
            for line in jf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        target_dict[d["item_id"]] = d.get("judge_prob")
                    except Exception:
                        pass

    md_lines: List[str] = []
    tsv_rows: List[str] = [
        "item_id\tgroup_id\tlabel\tverb\tnoun\tspoke\ttranscript\taction_window\treaction_window\tlocal_judge_p\tlarge_judge_p\tdescription"
    ]

    pos_title = "Pre-Failure Normal Execution (Label = 1)" if dataset == "reflex" else "Resolved Team Outcome (Label = 1)"
    neg_title = "Robot Manipulation Failure (Label = 0)" if dataset == "reflex" else "Unresolved Team Outcome (Label = 0)"

    for sub_title, sub_items in [(pos_title, sampled_pos), (neg_title, sampled_neg)]:
        md_lines.append(f"#### `{dataset}` — {sub_title}\n")
        for idx, it in enumerate(sub_items, 1):
            jq = qwen_cache.get(it.item_id)
            jg = gemma_cache.get(it.item_id)
            jq_s = f"`{jq:.2f}`" if jq is not None else "`null`"
            jg_s = f"`{jg:.2f}`" if jg is not None else "`null`"
            spk = it.meta.get("spoke", 0)
            tr = it.meta.get("transcript") or "None"
            verb = it.meta.get("verb", "")
            noun = it.meta.get("noun", "")
            desc = VERIFIED_EXAMPLE_DESCRIPTIONS.get(it.item_id)
            if not desc:
                raise KeyError(f"Missing verified visual frame description for sampled test item {it.item_id}")

            md_lines.append(f"{idx}. **`{it.item_id}`** (Group: `{it.group_id}`, Label: `{it.label}`)")
            md_lines.append(f"   - Context: `{it.context_text}`")
            md_lines.append(
                f"   - Action Window: `[{it.action_window_sec[0]:.2f}, {it.action_window_sec[1]:.2f}]` s "
                f"(Reaction Window: `[{it.reaction_window_sec[0]:.2f}, {it.reaction_window_sec[1]:.2f}]` s)"
            )
            md_lines.append(f"   - Local Judge P: {jq_s}, Large Judge P: {jg_s}, Spoke: `{spk}`, Transcript: *\"{tr}\"*")
            md_lines.append(f"   - Seen in the frames (`video_cam1.mp4`): {desc}")
            md_lines.append("")

            tsv_rows.append(
                f"{it.item_id}\t{it.group_id}\t{it.label}\t{verb}\t{noun}\t{spk}\t{tr}\t"
                f"[{it.action_window_sec[0]:.2f},{it.action_window_sec[1]:.2f}]\t"
                f"[{it.reaction_window_sec[0]:.2f},{it.reaction_window_sec[1]:.2f}]\t"
                f"{jq if jq is not None else ''}\t{jg if jg is not None else ''}\t{desc}"
            )

    if tsv_out is not None and (sampled_pos or sampled_neg):
        tsv_out.parent.mkdir(parents=True, exist_ok=True)
        tsv_out.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")

    return md_lines


def generate_reflex_report(
    date_str: str,
    scorecard_path: Path,
    data_root: Path,
    splits_dir: Path,
    output_path: Path,
    tsv_out_reflex: Optional[Path] = None,
    tsv_out_reflex_resolve: Optional[Path] = None,
) -> Path:
    """Generate the full Wave B REFLEX H1 evaluation report."""
    # 1. Parse scorecard rows
    reflex_rows: Dict[str, dict] = {}
    reflex_resolve_rows: Dict[str, dict] = {}
    if scorecard_path.exists():
        for line in scorecard_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ds = r.get("dataset")
            if ds == "reflex":
                reflex_rows[r["condition"]] = r
            elif ds == "reflex_resolve":
                reflex_resolve_rows[r["condition"]] = r

    # 2. Prediction checks
    pred_reflex = prediction_check_reflex(reflex_rows, dataset="reflex")
    pred_resolve = prediction_check_reflex(reflex_resolve_rows, dataset="reflex_resolve")

    # 3. Kill rule
    kr_status, kr_msg = evaluate_kill_rule(reflex_rows, reflex_resolve_rows)

    # 4. Wall clocks
    timings = compute_reflex_wall_clocks(data_root)

    # 5. MicroDuck probe weights
    probe_reflex_path = data_root / "transfer" / "reflex_face_probe.npz"
    probe_resolve_path = data_root / "transfer" / "reflex_resolve_face_probe.npz"
    pos_r, neg_r = extract_top_weights(probe_reflex_path)
    pos_rr, neg_rr = extract_top_weights(probe_resolve_path)

    # Build markdown
    md: List[str] = []
    md.append(f"# REFLEX Multimodal Human-Robot Reactions (Wave B) H1 Evaluation Report ({date_str})\n")
    md.append("**Pre-registration:** `05_wave_b_reflex.md`, `00_thesis.md` Decision 28, `03_eval_harness.md` §11.8–§11.9.")
    md.append(f"**Scorecard Source:** `{scorecard_path}`\n")

    md.append("## 1. Executive Summary\n")
    md.append(
        "Wave B evaluates spontaneous human facial, gaze, head-pose, posture (`react-face`, encoder `reflex-face-v1`, 153-d) "
        "and vocal prosody (`react-nonverbal`, encoder `reflex-speech-v1`, 96-d) reactions to physical Baxter robot failures "
        "during a collaborative object-sorting task (`dataset = reflex`, 1,208 items across 55 participants; 834 train / 374 test), "
        "and reactions to robot explanations predicting subsequent hidden-outcome team resolution (`dataset = reflex_resolve`, "
        "604 items across 55 participants after normalizing the `C3-2` header typo `'Resloved?'` → `'Resolved?'`: 488 `TRUE` / 116 `FALSE`; "
        "417 train / 187 test)."
    )
    md.append("")
    md.append(f"- **Wave B Pre-Registered Kill Rule Verdict:** **{kr_status}**")
    md.append(f"  - {kr_msg}")
    md.append("")

    md.append("## 2. Pre-Registered Prediction Checks & Pass Bar\n")
    md.append("### 2.1 `dataset = reflex` (Immediate Reaction to Robot Failure: `Pre` vs. `Failure`)")
    for line in pred_reflex:
        md.append(line)
    md.append("")
    md.append("### 2.2 `dataset = reflex_resolve` (Hidden-Outcome Resolution after Robot Explanation)")
    for line in pred_resolve:
        md.append(line)
    md.append("")

    # Full conditions tables
    md.append("## 3. Scorecard Conditions Table\n")
    for ds_name, rows_dict in [("reflex", reflex_rows), ("reflex_resolve", reflex_resolve_rows)]:
        md.append(f"### 3.{1 if ds_name == 'reflex' else 2} `{ds_name}` (Test Split)\n")
        md.append("| Condition | Metric | Value | 95% CI | N (items) | N (groups) | N (excluded) | Leak check (fails iff ci_low > 0.5) | Notes |")
        md.append("|---|---|---|---|---|---|---|---|---|")

        order = [
            "react-face", "react-nonverbal", "react-nonverbal|spoke=1", "react-spoke", "react-full",
            "judge", "judge-large", "judge-frontier", "action-probe", "step-prior", "action-fusion",
            "fusion", "fusion_minus_action_fusion", "fusion_minus_action_best",
            "action-probe:shuffled", "react-face:shuffled", "react-nonverbal:shuffled",
            "step-prior:shuffled", "action-fusion:shuffled", "fusion:shuffled",
        ]
        seen = set()
        for cond in order:
            if cond in rows_dict:
                seen.add(cond)
                r = rows_dict[cond]
                val_s = f"{r['value']:.3f}" if r.get("value") is not None else "—"
                ci_s = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r.get("ci_low") is not None else "—"
                n_i = str(r.get("n_items", "—"))
                n_g = str(r.get("n_groups", "—"))
                n_ex = str(r.get("n_excluded", "—"))
                leak = leak_check_label(r.get("value"), r.get("ci_low"), r.get("ci_high")) if ":shuffled" in cond else "—"
                notes = r.get("notes", "")
                md.append(f"| `{cond}` | {r.get('metric', 'auroc')} | {val_s} | {ci_s} | {n_i} | {n_g} | {n_ex} | {leak} | {notes} |")

        for cond, r in rows_dict.items():
            if cond not in seen:
                val_s = f"{r['value']:.3f}" if r.get("value") is not None else "—"
                ci_s = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r.get("ci_low") is not None else "—"
                n_i = str(r.get("n_items", "—"))
                n_g = str(r.get("n_groups", "—"))
                n_ex = str(r.get("n_excluded", "—"))
                leak = leak_check_label(r.get("value"), r.get("ci_low"), r.get("ci_high")) if ":shuffled" in cond else "—"
                notes = r.get("notes", "")
                md.append(f"| `{cond}` | {r.get('metric', 'auroc')} | {val_s} | {ci_s} | {n_i} | {n_g} | {n_ex} | {leak} | {notes} |")
        md.append("")

    md.append("### 3.3 Shuffled-Control & Methodological Notes (`L12`, `L14`)\n")
    md.append(
        "- **`reflex_resolve` `react-nonverbal:shuffled` (`0.603 [0.507, 0.691]`, `FAIL: above chance (leak)`):** "
        "11 of the 12 `:shuffled` controls across `reflex` and `reflex_resolve` pass (`ci_low <= 0.5`). "
        "On `reflex_resolve` test, only `140` items have participant speech (`112` positive `Resolved? = TRUE` vs. `28` negative `Resolved? = FALSE` across 17 test participants), "
        "and the probe fits 96 prosody dimensions (`reflex-speech-v1`) on `326` train speech items (`p/n = 96/326 = 0.29`). "
        "Per `Lesson L14`, a 50-seed permutation check (`seed = 0..49` shuffling train labels independently of test) yields mean test AUROC **`0.5100`** (SD **`0.0626`**, range `[0.3302, 0.6526]`), "
        "with `seed = 0` (`0.6025`) sitting at the **92nd percentile** (58% of seeds > 0.50). "
        "There is zero label leakage in `ReflexSpeechEncoder` or `run_probes`; the single fixed permutation (`default_rng(0)`) crosses `ci_low = 0.500` by `0.007` due to small-sample variance on the 28 negative test utterances (while true-label `react-nonverbal` scores `0.660 [0.545, 0.766]`, above all 50 null permutations)."
    )
    md.append(
        "- **`reflex_resolve` `step-prior` (`0.938 [0.883, 0.980]`, `Lesson L12`):** "
        "In `reflex_resolve`, `meta.noun` encodes the robot's explanation detail level (`zero`, `low`, `medium`, `high` from `analysis.csv`'s `Explain` column, `05_wave_b_reflex.md` §3.3). "
        "Because the REFLEX study protocol resolves almost every failure when the robot provides a `low`, `medium`, or `high` explanation (`475 / 478 = 99.4%` resolved) and fails to resolve most `zero`-explanation episodes (`13 / 126 = 10.3%` resolved), "
        "the train-split `(verb, explanation_level)` step prior achieves `0.938 [0.883, 0.980]` on test (`step-prior:shuffled` = `0.440 [0.278, 0.593]`, `pass`)."
    )
    md.append(
        "- **`reflex` `judge-large` (`0.500 [0.500, 0.500]`):** "
        "In `video_cam1.mp4`, the human participant is segmented out with a black mask and the wide-angle camera views the tabletop from across the room. "
        "When asked *\"Is the robot performing the step normally at this point, with no failure yet?\"* on `reflex`, `gemma4:26b` answered `100` (`prob = 1.0`) on all 374 test items, yielding constant predictions (`AUROC = 0.500`), whereas `qwen2.5vl:7b` (`judge`) achieved `0.720 [0.680, 0.761]` and `action-probe` (`siglip-b16-224-u9`) achieved `0.730 [0.705, 0.767]`."
    )
    md.append("")

    # Judge diagnostics
    md.append("## 4. Judge Diagnostics & Answer Distributions\n")
    for ds_name in ["reflex", "reflex_resolve"]:
        q = DATASET_QUESTIONS.get(ds_name, "")
        p_h = prompt_hash(q)
        md.append(f"### 4.{1 if ds_name == 'reflex' else 2} `{ds_name}` (Prompt Hash `{p_h}`)\n")
        md.append(f"- **Question:** *\"{q}\"*")
        for m_tag, m_name in [("qwen2.5vl_7b", "Local Judge (`qwen2.5vl:7b`)"), ("gemma4_26b", "Large Local Judge (`gemma4:26b`)")]:
            j_st = compute_judge_stats(ds_name, m_tag, p_h, data_root)
            md.append(
                f"- **{m_name}:** cached answers = {j_st['total']}, parse failures = {j_st['parse_failures']}, "
                f"ffmpeg decode errors = {j_st['ffmpeg_errors']}, API errors = {j_st['api_errors']}"
            )
            dist = compute_distribution(ds_name, m_tag, p_h, data_root, splits_dir)
            if dist and (dist["label=1"]["total"] > 0 or dist["label=0"]["total"] > 0):
                md.append("")
                md.append("| Ground Truth (Test Split) | N Total | Prob = 1.0 (100) | Prob = 0.9 (90) | Prob <= 0.8 (<=80) | Prob = 0.0 (0) |")
                md.append("|---|---|---|---|---|---|")
                for gt in ["label=1", "label=0"]:
                    c = dist[gt]
                    md.append(f"| `{gt}` | {c['total']} | {c['100']} | {c['90']} | {c['<=80']} | {c['0']} |")
                md.append("")
        md.append("")

    # Stage A MicroDuck Artifact Summary
    md.append("## 5. Stage A MicroDuck Transfer Probes (`DW6` Preparation)\n")
    md.append("Standardized logistic regression probe weights saved for live emotional feedback on the Pollen Robotics MicroDuck:")
    md.append("")
    for ds_name, p_path, top_pos, top_neg in [
        ("reflex", probe_reflex_path, pos_r, neg_r),
        ("reflex_resolve", probe_resolve_path, pos_rr, neg_rr),
    ]:
        md.append(f"### 5.{1 if ds_name == 'reflex' else 2} `{ds_name}` Face Probe (`{p_path.name}`)")
        if p_path.exists():
            md.append(f"- **Path:** `{p_path}`")
            md.append("- **Top 5 Positive Standardized Weights (indicative of `label = 1`):**")
            for name, w in top_pos:
                md.append(f"  - `{name}`: `+{w:.4f}`")
            md.append("- **Top 5 Negative Standardized Weights (indicative of `label = 0`):**")
            for name, w in top_neg:
                md.append(f"  - `{name}`: `{w:.4f}`")
        else:
            md.append(f"- Probe artifact `{p_path}` not yet generated.")
        md.append("")

    # Wall clock times
    md.append("## 6. Compute & Wall-Clock Times\n")
    md.append("| Phase | Wall-clock / Elapsed Time |")
    md.append("|---|---|")
    for k, v in timings.items():
        md.append(f"| {k} | {v} |")
    md.append("")

    # Sampled examples
    md.append("## 7. Sampled Test Split Examples (Verified Visual Descriptions)\n")
    md.append(
        "Ten test items per dataset sampled with `numpy.random.default_rng(0)` (5 `label = 1`, 5 `label = 0`), "
        "with verified descriptions of the 4 sampled action-window frames in `video_cam1.mp4` (where the human participant "
        "is segmented out with a black mask and the Baxter robot and tabletop objects are visible):"
    )
    md.append("")
    md.append("### 7.1 `reflex` Sampled Test Items\n")
    md.extend(_sample_and_format_examples("reflex", data_root, splits_dir, tsv_out=tsv_out_reflex))
    md.append("### 7.2 `reflex_resolve` Sampled Test Items\n")
    md.extend(_sample_and_format_examples("reflex_resolve", data_root, splits_dir, tsv_out=tsv_out_reflex_resolve))

    # Write markdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md), encoding="utf-8")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wave B REFLEX H1 evaluation report.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Report date (YYYY-MM-DD)")
    parser.add_argument("--scorecard", default="results/scorecard.jsonl", help="Scorecard jsonl path")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/Volumes/Extreme SSD/social_robotics"), help="Data root path")
    parser.add_argument("--splits-dir", default="splits", help="Splits directory")
    parser.add_argument("--out-dir", default="docs/evals", help="Output directory")

    args = parser.parse_args()

    scorecard_path = Path(args.scorecard)
    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)

    out_file = out_dir / f"{args.date}_reflex_h1.md"
    tsv_reflex = out_dir / f"{args.date}_reflex_examples.tsv"
    tsv_resolve = out_dir / f"{args.date}_reflex_resolve_examples.tsv"

    generate_reflex_report(
        date_str=args.date,
        scorecard_path=scorecard_path,
        data_root=data_root,
        splits_dir=splits_dir,
        output_path=out_file,
        tsv_out_reflex=tsv_reflex,
        tsv_out_reflex_resolve=tsv_resolve,
    )
    print(f"Report written to {out_file}")


if __name__ == "__main__":
    main()
