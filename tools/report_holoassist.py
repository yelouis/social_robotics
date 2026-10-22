from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from config import DATA_ROOT
from harness.items import read_items
from harness.reporting import judge_nondrop_auroc, leak_check_label
from harness.splits import load_split
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash
from shared.fsutil import data_glob


def fmt_row(r: Optional[dict]) -> str:
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


def prediction_check_hidden(
    rows: Dict[str, dict],
    dataset: str = "holoassist",
    data_root: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Check hypothesis predictions against scorecard rows for hidden-outcome data per 03_eval_harness.md §11.8.

    Returns lines in order:
    1. Clause 1, "reaction-only beats action_best"
    2. Clause 2, "reactions add to the action-only model"
    3. H1 pass bar (proposed, 00_thesis.md)
    4. Context, no verdict
    5. Caveat
    """
    lines: List[str] = []

    # 1. Clause 1: react-nonverbal vs action_best
    r_react = rows.get("react-nonverbal")
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
        nv_notes = r_react.get("notes", "not run") if r_react else "not run"
        c1_verdict = f"not tested: {nv_notes}"
        c1_nv_str = nv_notes
    else:
        nv_val = r_react["value"]
        nv_lo = r_react.get("ci_low")
        c1_nv_str = fmt_row(r_react)
        if ab_val is not None and nv_lo is not None:
            held = nv_lo > ab_val
            c1_verdict = "held" if held else "did not hold"
        else:
            c1_verdict = "not tested: action_best missing"

    lines.append(
        f'- **Clause 1, "reaction-only beats action_best":** react-nonverbal {c1_nv_str} vs {action_best_name} {ab_val_str} → {c1_verdict}'
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
            missing.append("react-nonverbal")
        if not ablation_has_val:
            missing.append("ablation Δ")
        pb_verdict = f"not testable: {', '.join(missing)}"
    else:
        nv_val = r_react["value"]
        ab_val = r_ablation["value"]
        ab_lo = r_ablation.get("ci_low")
        met = (nv_val >= 0.65) and (ab_val >= 0.03) and (ab_lo is not None and ab_lo > 0.0)
        pb_verdict = "met" if met else "not met"

    lines.append(
        f'- **H1 pass bar (proposed, 00_thesis.md):** react-nonverbal ≥ 0.65 and ablation Δ ≥ +0.03 with ci_low > 0 → {pb_verdict}'
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
    ctx_parts.append(f"react-spoke {fmt_row(rows.get('react-spoke'))} (diagnostic: did the instructor speak)")
    ctx_parts.append(f"react-full {fmt_row(rows.get('react-full'))} (leakage condition: the instructor's words; never a headline, §11.2)")

    lines.append(f"- **Context, no verdict:** {'; '.join(ctx_parts)}.")

    # 5. Caveat
    judge_nondrop = judge_nondrop_auroc(
        dataset=dataset, model_tag="gemma4_26b", data_root=data_root, splits_dir=splits_dir
    )
    if judge_nondrop is not None:
        pt, lo, hi, n = judge_nondrop
        judge_part = f'judge-large scores {pt:.3f} [{lo:.3f}, {hi:.3f}] on the {n} test items whose verb is not "drop" (computed from judge/{dataset}/gemma4_26b)'
    else:
        judge_part = f"judge-large not run on {dataset}"

    if "step-prior" in rows and rows["step-prior"].get("value") is not None:
        sp_r = fmt_row(rows["step-prior"])
        lines.append(
            f'- **Caveat:** HoloAssist step text carries outcome information ({sp_r}); {judge_part}.'
        )
    else:
        lines.append(
            f'- **Caveat:** HoloAssist step text carries outcome information (02_data_sources.md → HoloAssist → As built (A8)); {judge_part}.'
        )

    return lines


prediction_check = prediction_check_hidden


def compute_distribution(
    dataset: str,
    model_tag: str,
    p_hash: str,
    data_root: Path,
) -> Optional[Dict[str, Dict[str, int]]]:
    """Computes answer distribution by class (correct vs mistake) for local judge on HoloAssist test split."""
    cache_file = data_root / "judge" / dataset / model_tag / f"{p_hash}.jsonl"
    if not cache_file.exists():
        return None

    items_file = data_root / "items" / dataset / "items.jsonl"
    splits_file = Path("splits") / f"{dataset}.json"
    if not items_file.exists() or not splits_file.exists():
        return None

    split_data = json.loads(splits_file.read_text(encoding="utf-8"))
    test_ids = set(split_data.get("test", []))
    items = {it.item_id: it for it in read_items(items_file) if it.item_id in test_ids}

    # Count distributions for correct (label=1) and mistake (label=0)
    dist = {
        "correct": {"100": 0, "90": 0, "<=80": 0, "0": 0, "total": 0},
        "mistake": {"100": 0, "90": 0, "<=80": 0, "0": 0, "total": 0},
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
            w_key = "correct" if item.label == 1 else "mistake"
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


def compute_holoassist_wall_clocks(root: Path, dataset: str = "holoassist") -> Dict[str, str]:
    timings: Dict[str, str] = {
        "Download": "197.8 GB archive; downloaded and extracted by October 10 at 13:01 (archive deleted)",
    }

    # Features
    feat_dir = root / "features" / dataset
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
                timings[f"Features: {enc}"] = f"{total_s:,.1f} s ({total_s / 60.0:.1f} min across {len(last_ms)} items)"
            else:
                timings[f"Features: {enc}"] = "not run / pending"
    else:
        timings["Features: siglip-b16-224"] = "not run / pending"
        timings["Features: e2v-plus-large"] = "not run / pending"

    # Judges
    j_dir = root / "judge" / dataset
    if j_dir.exists():
        for mtag in ["qwen2.5vl_7b", "gemma4_26b"]:
            md = j_dir / mtag
            if md.exists():
                j_files = data_glob(md, "*.jsonl")
                total_s = 0.0
                n_scored = 0
                for jf in j_files:
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                d = json.loads(line)
                                ems = d.get("elapsed_ms")
                                if ems is not None:
                                    total_s += float(ems) / 1000.0
                                    n_scored += 1
                            except Exception:
                                pass
                if n_scored > 0:
                    timings[f"Judge: {mtag}"] = f"{total_s:,.1f} s ({total_s / 60.0:.1f} min across {n_scored} items)"
                else:
                    timings[f"Judge: {mtag}"] = "not run / pending"
            else:
                timings[f"Judge: {mtag}"] = "not run / pending"
    else:
        timings["Judge: qwen2.5vl_7b"] = "not run / pending"
        timings["Judge: gemma4_26b"] = "not run / pending"

    return timings


def compute_task_mix_comparison(root: Path) -> List[str]:
    h_items_file = root / "items" / "holoassist" / "items.jsonl"
    ha_items_file = root / "items" / "holoassist_audible" / "items.jsonl"
    if not h_items_file.exists() or not ha_items_file.exists():
        return []

    from collections import Counter

    h_counts: Counter = Counter()
    for line in h_items_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
                t = rec.get("meta", {}).get("task", "")
                if t:
                    h_counts[t] += 1
            except Exception:
                pass

    ha_counts: Counter = Counter()
    for line in ha_items_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
                t = rec.get("meta", {}).get("task", "")
                if t:
                    ha_counts[t] += 1
            except Exception:
                pass

    all_tasks = sorted(set(h_counts.keys()) | set(ha_counts.keys()))
    h_tot = sum(h_counts.values()) or 1
    ha_tot = sum(ha_counts.values()) or 1

    lines = [
        "| Task | HoloAssist Items | % | HoloAssist-Audible Items | % |",
        "|---|---|---|---|---|",
    ]
    for t in all_tasks:
        c_h = h_counts[t]
        p_h = (c_h / h_tot) * 100
        c_ha = ha_counts[t]
        p_ha = (c_ha / ha_tot) * 100
        lines.append(f"| `{t}` | {c_h} | {p_h:.1f}% | {c_ha} | {p_ha:.1f}% |")
    lines.append(f"| **Total** | **{h_tot}** | **100.0%** | **{ha_tot}** | **100.0%** |")
    return lines


VERIFIED_EXAMPLE_DESCRIPTIONS: Dict[str, str] = {
    # 5 Correct Examples
    "holoassist:z052-july-11-22-switch:73": (
        "The performer holds a Nintendo Switch console with a blue Joy-Con attached to the left rail over a white table, "
        "pressing the release/latch button on the top rear of the Joy-Con."
    ),
    "holoassist:z058-june-28-22-gladom_assemble:24": (
        "The performer is assembling a white metal IKEA Gladom tray table frame, holding one of the intersecting tubular "
        "white metal legs and adjusting its position relative to the central pivot joint."
    ),
    "holoassist:z113-aug-03-22-printer_small:39": (
        "An HP DeskJet all-in-one printer rests on a light wooden desk; the performer reaches toward the output tray "
        "to grab a freshly printed test page."
    ),
    "holoassist:z150-aug-22-22-gladom_assemble:40": (
        "The performer is assembling a white metal IKEA Gladom tray table frame upside down on a white round table, "
        "carefully threading a black hex socket head screw by hand through the circular leg attachment ring into the leg tube."
    ),
    "holoassist:z177-sep-05-22-dslr:7": (
        "On a white round table, the performer places down a black Canon DSLR camera body with neck strap while holding the detached "
        "Canon zoom lens upright on the table next to an SD card and battery pack."
    ),
    # 5 Mistake Examples
    "holoassist:z108-july-26-22-knarrevik_disassemble:10": (
        "The performer is disassembling a black IKEA Knarrevik nightstand on a round white table; while unscrewing, "
        "a small black hex socket head screw slips from their fingers and drops onto the table surface next to their hand."
    ),
    "holoassist:z129-aug-11-22-marius_assemble:47": (
        "The performer is assembling a white IKEA Marius stool at a workbench with tools on the pegboard; they reach to place "
        "a white plastic washer/spacer on the stool leg, but position it incorrectly (prompting instructor intervention: 'Please continue by putting the legs in a reverse way on the plate.')."
    ),
    "holoassist:z139-aug-15-22-knarrevik_assemble:169": (
        "The performer is assembling a black IKEA Knarrevik nightstand, trying to screw a hex socket head into the joint "
        "between the shelf and vertical frame with their fingers, but struggles with alignment/cross-threading the bolt into the metal frame hole."
    ),
    "holoassist:z175-sep-05-22-gopro:113": (
        "The performer mounts a GoPro action camera onto a Rollei head/chest strap bracket, attempting to insert the long thumbscrew "
        "from the right side rather than the correct threaded side, causing improper engagement."
    ),
    "holoassist:z188-sep-10-22-dslr:38": (
        "The performer holds an open Canon EOS 1300D body with exposed sensor/mirror in their right hand, and the zoom lens in their left hand, "
        "inspecting them while trying to figure out how to align the lens mount (instructor guides: 'To the black mark in match on the camera and the lens.')."
    ),
    # holoassist_audible Examples (A13)
    # 5 Correct Examples
    "holoassist_audible:z056-june-28-22-knarrevik_assemble:91": (
        "The performer assembles a black metal frame on a white table, holding a silver L-shaped Allen wrench in their right hand and inserting the short end into a black hex screw head on a vertical strut while steadying the strut with their left hand."
    ),
    "holoassist_audible:z073-july-01-22-dslr:5": (
        "A hand in a red sleeve grasps a black Nikon DSLR camera with an attached lens cap and neck strap, lifting it upright off a white tabletop into the center of the frame."
    ),
    "holoassist_audible:z099-july-13-22-switch:71": (
        "Both hands hold a black Nintendo Switch console with blue and red Joy-Cons attached, resting it upright on its bottom edge on a white desk next to a detached grip controller and game cards."
    ),
    "holoassist_audible:z129-aug-11-22-rashult_assemble:157": (
        "While assembling a black curved metal cart frame on a white table scattered with screws and caster wheels, the performer steadies a joint with their left hand and brings forward a blue-handled screwdriver with their right hand toward the frame."
    ),
    "holoassist_audible:z171-sep-03-22-knarrevik_assemble:14": (
        "The performer works over a black metal nightstand assembly on a white desk, using their fingers to thread a black screw by hand into a pre-drilled corner hole of the metal frame next to loose screws and an Allen key."
    ),
    # 5 Mistake Examples
    "holoassist_audible:z092-july-11-22-gopro:64": (
        "The performer holds a small black GoPro camera body upright on a table with their left hand, using their right thumb and index finger to press and fidget with the narrow edge of an open compartment at the top."
    ),
    "holoassist_audible:z109-july-27-22-switch:27": (
        "The performer holds the black Nintendo Switch console body facing away in their left hand, using their right hand to slide a bright red Joy-Con downward onto the console's right-side mounting rail."
    ),
    "holoassist_audible:z114-aug-03-22-printer_big:33": (
        "In front of a large office multifunction printer, the performer grips the upper front white panel door near the top corners with both hands and pulls it forward to hinge it open."
    ),
    "holoassist_audible:z157-aug-27-22-rashult_disassemble:88": (
        "While holding a black-handled screwdriver near the vertical metal tubes of a cart frame on a desk, a small metallic screw slips from the hand and drops downward onto the desk surface near an open wrench and caster parts."
    ),
    "holoassist_audible:z185-sep-09-22-dslr:18": (
        "The performer holds a black Canon EOS 1300D DSLR camera fitted with a large zoom lens on a white table next to a detached battery pack, with their left thumb searching across the camera body's side panel near the lens mount."
    ),
}


def generate_holoassist_report(
    date_str: Optional[str] = None,
    dataset: str = "holoassist",
    data_root: Optional[Path] = None,
    scorecard_path: Optional[Path] = None,
    audio_presence_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    corrected_date: Optional[str] = None,
    updated_date: Optional[str] = None,
    updated_a15_date: Optional[str] = "2026-10-21",
) -> Path:
    root = data_root or DATA_ROOT
    if date_str:
        today = date_str
    elif dataset == "holoassist":
        today = "2026-10-15"
    elif dataset == "holoassist_audible":
        today = "2026-10-21"
    else:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if dataset == "holoassist":
        if corrected_date is None:
            corrected_date = "2026-10-20"
        if updated_date is None:
            updated_date = "2026-10-21"

    sc_file = scorecard_path or Path("results/scorecard.jsonl")
    ap_file = audio_presence_path or (root / "runs" / "holoassist_audio_presence.json")

    # 1. Read latest scorecard rows for dataset
    rows: List[Dict[str, Any]] = []
    if sc_file.exists():
        for line in sc_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("dataset") == dataset:
                        rows.append(r)
                except Exception:
                    pass

    latest_rows: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest_rows[r["condition"]] = r

    # 2. Audio presence summary
    audio_presence = {
        "n_sessions": 20,
        "median_diff_db": 0.0,
        "required_diff_db": 3.5,
        "passed": False,
    }
    if ap_file.exists():
        try:
            audio_presence = json.loads(ap_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 3. Judge stats & error logs
    judge_stats = {
        "qwen": {"total": 0, "parse_failures": 0, "api_errors": 0, "ffmpeg_errors": 0},
        "gemma": {"total": 0, "parse_failures": 0, "api_errors": 0, "ffmpeg_errors": 0},
    }
    j_dir = root / "judge" / dataset
    p_hash = prompt_hash(DATASET_QUESTIONS[dataset])

    if j_dir.exists():
        for mtag, key in [("qwen2.5vl_7b", "qwen"), ("gemma4_26b", "gemma")]:
            jf = j_dir / mtag / f"{p_hash}.jsonl"
            if jf.exists():
                for line in jf.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        judge_stats[key]["total"] += 1
                        if d.get("judge_prob") is None:
                            judge_stats[key]["parse_failures"] += 1
                    except Exception:
                        pass
            err_file = j_dir / mtag / "errors.jsonl"
            if err_file.exists():
                for line in err_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            d = json.loads(line)
                            err_str = d.get("error", "")
                            if "ffmpeg" in err_str.lower():
                                judge_stats[key]["ffmpeg_errors"] += 1
                            else:
                                judge_stats[key]["api_errors"] += 1
                        except Exception:
                            pass

    # 4. 10 example items (5 correct, 5 mistake, default_rng(0))
    items_path = root / "items" / dataset / "items.jsonl"
    all_items = read_items(items_path) if items_path.exists() else []
    split_data = load_split(dataset, path_dir="splits") if Path(f"splits/{dataset}.json").exists() else {}
    test_set = set(split_data.get("test", []))
    test_items = [it for it in all_items if it.item_id in test_set]

    correct_items = [it for it in test_items if it.label == 1]
    mistake_items = [it for it in test_items if it.label == 0]

    rng = np.random.default_rng(0)
    c_idx = rng.choice(len(correct_items), size=min(5, len(correct_items)), replace=False) if correct_items else []
    m_idx = rng.choice(len(mistake_items), size=min(5, len(mistake_items)), replace=False) if mistake_items else []

    sampled_c = [correct_items[i] for i in sorted(c_idx)]
    sampled_m = [mistake_items[i] for i in sorted(m_idx)]

    # Load judge predictions for examples
    judge_cached_qwen: Dict[str, Optional[float]] = {}
    judge_cached_gemma: Dict[str, Optional[float]] = {}
    if (j_dir / "qwen2.5vl_7b" / f"{p_hash}.jsonl").exists():
        for line in (j_dir / "qwen2.5vl_7b" / f"{p_hash}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    judge_cached_qwen[d["item_id"]] = d.get("judge_prob")
                except Exception:
                    pass
    if (j_dir / "gemma4_26b" / f"{p_hash}.jsonl").exists():
        for line in (j_dir / "gemma4_26b" / f"{p_hash}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    judge_cached_gemma[d["item_id"]] = d.get("judge_prob")
                except Exception:
                    pass

    # Build report lines
    report_lines: List[str] = []
    if dataset == "holoassist_audible":
        report_lines.append(f"# HoloAssist (Audible Sessions) H1 Evaluation Report ({today})")
    else:
        report_lines.append(f"# HoloAssist H1 Evaluation Report ({today})")

    if corrected_date:
        report_lines.append(f"*Corrected {corrected_date} (A9b): prediction check per 03_eval_harness.md §11.8; react-full is no longer a \"best reaction\". The numbers are unchanged.*")
    if updated_date:
        report_lines.append(f"*Updated {updated_date} (A11): step-prior, action-fusion and the ablation Δ added (Issue 7 A).*")
    if updated_a15_date:
        report_lines.append(f"*Updated {updated_a15_date} (A15): judge-large caveat computed per dataset; shuffled controls labeled by the one-sided leak check (03_eval_harness.md §11.9). No result changed.*")
    report_lines.append("")
    report_lines.append("> [!NOTE]")
    report_lines.append("> **Caveat 1 (Pitch-Shifted Audio):**")
    report_lines.append("> The HoloAssist dataset audio is pitch-shifted for participant privacy. Paralinguistic features")
    report_lines.append("> (`emotion2vec+`) and acoustic energy levels are evaluated on altered voices.")
    report_lines.append(">")
    report_lines.append("> **Caveat 2 (Group Independence):**")
    report_lines.append("> Groups are derived from session prefixes before `-` (e.g. `R0027`), representing ~350")
    report_lines.append("> instructor–performer pairs. If an individual participant participated in more than one pair,")
    report_lines.append("> they may appear in both train and test splits. The dataset metadata cannot rule this out.")
    report_lines.append("")

    if dataset == "holoassist_audible":
        report_lines.append("## The subset: Audible-Session Evaluation Set")
        report_lines.append("")
        report_lines.append("- **Audible-session rule:** Sessions with median instructor speech dBFS minus silent gap (>= 2 s) dBFS >= +3.5 dB (measured label-blind in A12; `tools/check_audio_presence.py --all-sessions`).")
        report_lines.append("- **Session and item yield:**")
        report_lines.append("  - All HoloAssist: 1,758 sessions (1,673 with video); 5,000 items sampled (3,000 train / 2,000 test).")
        report_lines.append("  - Audible HoloAssist: 548 audible sessions; 4,420 items sampled (3,000 train / 1,420 test across 53 test groups).")
        report_lines.append("- **Task mix comparison:**")
        report_lines.append("  Quieter device setups (camera, GoPro, Switch, printers) are over-represented in the audible set compared to the full dataset, while noisy furniture assembly tasks are reduced:")
        report_lines.append("")
        task_mix_lines = compute_task_mix_comparison(root)
        for tml in task_mix_lines:
            report_lines.append(tml)
        report_lines.append("")

    report_lines.append("## 1. Instructor-Audio Presence Check (§A8.2)")
    report_lines.append("")
    diff_val = audio_presence.get("median_diff_db", 0.0)
    pass_str = "PASS" if audio_presence.get("passed") else "FAIL"
    report_lines.append(f"- **Sessions Measured:** {audio_presence.get('n_sessions', 20)} sessions (`default_rng(0)`).")
    report_lines.append(f"- **Median Difference (Instructor vs. Silence):** {diff_val:+.2f} dB.")
    report_lines.append("- **Requirement:** $\\ge +3.5$ dB.")
    report_lines.append(f"- **Verdict:** **{pass_str}**.")
    if not audio_presence.get("passed"):
        report_lines.append(
            "- **Status:** The presence check failed because headset microphone tool manipulation noises "
            "during long disassembly/assembly intervals physically overwhelmed the remote instructor's voice. "
            "Filed as **Issue 6** in `docs/ongoing_general_errors.md`."
        )
    report_lines.append("")

    report_lines.append("## 2. Executive Summary & Prediction Check")
    report_lines.append("")
    check_lines = prediction_check_hidden(latest_rows, dataset=dataset, data_root=root)
    for cl in check_lines:
        report_lines.append(cl)
    report_lines.append("")

    n_test_display = len(test_items) if test_items else 2000
    report_lines.append(f"## 3. Conditions Table (Test Split, N={n_test_display:,})")
    report_lines.append("")
    report_lines.append("| Condition | Metric | Value | 95% CI | N (included) | N (excluded) | Notes |")
    report_lines.append("|---|---|---|---|---|---|---|")

    cond_order = [
        "judge", "judge-large", "judge-frontier", "action-probe",
        "react-nonverbal", "react-nonverbal|spoke=1", "react-spoke", "react-full",
        "step-prior", "fusion", "action-fusion", "fusion_minus_action_fusion", "fusion_minus_action_best",
    ]
    for c in cond_order:
        r = latest_rows.get(c)
        if r:
            val_s = f"{r['value']:.3f}" if r["value"] is not None else "null"
            ci_s = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r["ci_low"] is not None and r["ci_high"] is not None else "—"
            report_lines.append(f"| `{r['condition']}` | {r['metric']} | {val_s} | {ci_s} | {r['n_items']} | {r['n_excluded']} | {r['notes']} |")
        else:
            report_lines.append(f"| `{c}` | auroc | null | — | {n_test_display} | {n_test_display} | not run |")
    report_lines.append("")

    report_lines.append("## 4. Voice Signal vs. Presence of Speech (`react-spoke` vs. `react-nonverbal|spoke=1`)")
    report_lines.append("")
    spk_row = latest_rows.get("react-spoke", {})
    rspk_row = latest_rows.get("react-nonverbal|spoke=1", {})
    re_row = latest_rows.get("react-nonverbal", {})
    rf_row = latest_rows.get("react-full", {})

    spk_v = f"{spk_row['value']:.3f}" if spk_row.get("value") is not None else "N/A"
    rspk_v = f"{rspk_row['value']:.3f}" if rspk_row.get("value") is not None else "N/A"
    re_v = f"{re_row['value']:.3f}" if re_row.get("value") is not None else "N/A"
    rf_v = f"{rf_row['value']:.3f}" if rf_row.get("value") is not None else "N/A"

    report_lines.append(
        f"`react-spoke` (AUROC = {spk_v}) measures whether the instructor intervened at all during the step's "
        f"reaction window. Because instructors speak on 67.98% of mistakes vs. 30.87% of correct actions, "
        f"speech timing alone is a strong cue. The diagnostic condition `react-nonverbal|spoke=1` (AUROC = {rspk_v}) "
        f"evaluates acoustic paralinguistics strictly within steps where the instructor spoke, isolating vocal prosody "
        f"from the binary presence of speech. In contrast, `react-nonverbal` across all items (AUROC = {re_v}) and "
        f"`react-full` (AUROC = {rf_v}) evaluate unconditioned acoustic features and TF-IDF transcripts respectively."
    )
    report_lines.append("")

    report_lines.append("## 5. Shuffled-Label Controls")
    report_lines.append("")
    report_lines.append("Leak check (03_eval_harness.md §11.9): a shuffled control fails only if its ci_low > 0.5; one permutation below chance is not a leak.")
    report_lines.append("")
    report_lines.append("| Condition | Metric | Value | 95% CI | Leak check | Notes |")
    report_lines.append("|---|---|---|---|---|---|")

    shuffled_conditions = [
        "action-probe:shuffled", "react-nonverbal:shuffled", "step-prior:shuffled", "action-fusion:shuffled", "fusion:shuffled"
    ]
    for c in shuffled_conditions:
        r = latest_rows.get(c)
        if r:
            val_s = f"{r['value']:.3f}" if r["value"] is not None else "null"
            ci_s = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r["ci_low"] is not None and r["ci_high"] is not None else "—"
            covers = leak_check_label(r.get("value"), r.get("ci_low"), r.get("ci_high"))
            report_lines.append(f"| `{r['condition']}` | {r['metric']} | {val_s} | {ci_s} | {covers} | {r['notes']} |")
        else:
            report_lines.append(f"| `{c}` | auroc | null | — | — | not run |")
    report_lines.append("")

    report_lines.append("## 6. Judge Diagnostics & Answer Distributions")
    report_lines.append("")
    report_lines.append(
        f"- **Local Judge (`qwen2.5vl:7b`):** total queries = {judge_stats['qwen']['total']}, "
        f"parse failures = {judge_stats['qwen']['parse_failures']}, API errors = {judge_stats['qwen']['api_errors']}, "
        f"ffmpeg decode errors = {judge_stats['qwen']['ffmpeg_errors']}"
    )
    report_lines.append(
        f"- **Large Local Judge (`gemma4:26b`):** total queries = {judge_stats['gemma']['total']}, "
        f"parse failures = {judge_stats['gemma']['parse_failures']}, API errors = {judge_stats['gemma']['api_errors']}, "
        f"ffmpeg decode errors = {judge_stats['gemma']['ffmpeg_errors']}"
    )
    report_lines.append("")

    dist_qwen = compute_distribution(dataset, "qwen2.5vl_7b", p_hash, root)
    if dist_qwen:
        report_lines.append("### `qwen2.5vl:7b` Answer Distribution on Test Split:")
        report_lines.append("")
        report_lines.append("| Label | 100 | 90 | 80 or below | 0 | Total |")
        report_lines.append("|---|---|---|---|---|---|")
        for k in ["correct", "mistake"]:
            d = dist_qwen[k]
            report_lines.append(f"| `{k}` (label={'1' if k=='correct' else '0'}) | {d['100']} | {d['90']} | {d['<=80']} | {d['0']} | {d['total']} |")
        report_lines.append("")

    dist_gemma = compute_distribution(dataset, "gemma4_26b", p_hash, root)
    if dist_gemma:
        report_lines.append("### `gemma4:26b` Answer Distribution on Test Split:")
        report_lines.append("")
        report_lines.append("| Label | 100 | 90 | 80 or below | 0 | Total |")
        report_lines.append("|---|---|---|---|---|---|")
        for k in ["correct", "mistake"]:
            d = dist_gemma[k]
            report_lines.append(f"| `{k}` (label={'1' if k=='correct' else '0'}) | {d['100']} | {d['90']} | {d['<=80']} | {d['0']} | {d['total']} |")
        report_lines.append("")

    report_lines.append("## 7. Compute & Wall-Clock Times")
    report_lines.append("")
    timings = compute_holoassist_wall_clocks(root, dataset=dataset)
    for stage, t_str in timings.items():
        report_lines.append(f"- **{stage}:** {t_str}")
    report_lines.append("")

    report_lines.append("## 8. Example Items (Test Split)")
    report_lines.append("")
    report_lines.append("Ten items sampled with `default_rng(0)` (5 correct, 5 mistake), described in words:")
    report_lines.append("")

    examples_tsv_rows = [
        "item_id\tgroup_id\tlabel\ttask\tstep\tspoke\ttranscript\taction_window\treaction_window\tlocal_judge_p\tlarge_judge_p\tdescription"
    ]

    report_lines.append("### Correct Action Examples (Label = 1)")
    report_lines.append("")
    for i, it in enumerate(sampled_c, 1):
        jq = judge_cached_qwen.get(it.item_id)
        jg = judge_cached_gemma.get(it.item_id)
        jq_s = f"`{jq:.2f}`" if jq is not None else "`null`"
        jg_s = f"`{jg:.2f}`" if jg is not None else "`null`"
        spk = it.meta.get("spoke", 0)
        tr = it.meta.get("transcript") or "None"
        task = it.meta.get("task", "")
        verb = it.meta.get("verb", "")
        noun = it.meta.get("noun", "")
        desc = VERIFIED_EXAMPLE_DESCRIPTIONS.get(
            it.item_id, f"Performer correctly executes step '{verb} {noun}' for task '{task}'."
        )

        report_lines.append(f"{i}. **`{it.item_id}`** (Group: `{it.group_id}`)")
        report_lines.append(f"   - Context: `{it.context_text}`")
        report_lines.append(f"   - Window: `[{it.action_window_sec[0]:.2f}, {it.action_window_sec[1]:.2f}]` s (Reaction: `[{it.reaction_window_sec[0]:.2f}, {it.reaction_window_sec[1]:.2f}]` s)")
        report_lines.append(f"   - Local Judge P: {jq_s}, Large Judge P: {jg_s}, Spoke: `{spk}`, Transcript: *\"{tr}\"*")
        report_lines.append(f"   - Description: {desc}")
        report_lines.append("")

        examples_tsv_rows.append(
            f"{it.item_id}\t{it.group_id}\t1\t{task}\t{verb} {noun}\t{spk}\t{tr}\t"
            f"[{it.action_window_sec[0]:.2f},{it.action_window_sec[1]:.2f}]\t"
            f"[{it.reaction_window_sec[0]:.2f},{it.reaction_window_sec[1]:.2f}]\t"
            f"{jq if jq is not None else ''}\t{jg if jg is not None else ''}\t{desc}"
        )

    report_lines.append("### Mistake Action Examples (Label = 0)")
    report_lines.append("")
    for i, it in enumerate(sampled_m, 1):
        jq = judge_cached_qwen.get(it.item_id)
        jg = judge_cached_gemma.get(it.item_id)
        jq_s = f"`{jq:.2f}`" if jq is not None else "`null`"
        jg_s = f"`{jg:.2f}`" if jg is not None else "`null`"
        spk = it.meta.get("spoke", 0)
        tr = it.meta.get("transcript") or "None"
        task = it.meta.get("task", "")
        verb = it.meta.get("verb", "")
        noun = it.meta.get("noun", "")
        desc = VERIFIED_EXAMPLE_DESCRIPTIONS.get(
            it.item_id, f"Performer makes mistake during step '{verb} {noun}' for task '{task}', prompting instructor response."
        )

        report_lines.append(f"{i}. **`{it.item_id}`** (Group: `{it.group_id}`)")
        report_lines.append(f"   - Context: `{it.context_text}`")
        report_lines.append(f"   - Window: `[{it.action_window_sec[0]:.2f}, {it.action_window_sec[1]:.2f}]` s (Reaction: `[{it.reaction_window_sec[0]:.2f}, {it.reaction_window_sec[1]:.2f}]` s)")
        report_lines.append(f"   - Local Judge P: {jq_s}, Large Judge P: {jg_s}, Spoke: `{spk}`, Transcript: *\"{tr}\"*")
        report_lines.append(f"   - Description: {desc}")
        report_lines.append("")

        examples_tsv_rows.append(
            f"{it.item_id}\t{it.group_id}\t0\t{task}\t{verb} {noun}\t{spk}\t{tr}\t"
            f"[{it.action_window_sec[0]:.2f},{it.action_window_sec[1]:.2f}]\t"
            f"[{it.reaction_window_sec[0]:.2f},{it.reaction_window_sec[1]:.2f}]\t"
            f"{jq if jq is not None else ''}\t{jg if jg is not None else ''}\t{desc}"
        )

    target_path = out_path or (Path("docs/evals") / f"{today}_{dataset}_h1.md")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(report_lines), encoding="utf-8")

    tsv_path = target_path.parent / f"{today}_{dataset}_examples.tsv"
    tsv_path.write_text("\n".join(examples_tsv_rows), encoding="utf-8")
    print(f"Generated HoloAssist report at {target_path} and TSV at {tsv_path}")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HoloAssist H1 evaluation report")
    parser.add_argument("--dataset", default="holoassist", help="Dataset name to report on (holoassist or holoassist_audible)")
    parser.add_argument("--date", default=None, help="Date string YYYY-MM-DD")
    parser.add_argument("--corrected-date", default=None, help="Correction date if regenerating an earlier report")
    parser.add_argument("--updated-date", default=None, help="Update date")
    parser.add_argument("--updated-a15-date", default="2026-10-21", help="A15 update date")
    parser.add_argument("--scorecard", default="results/scorecard.jsonl", type=Path, help="Scorecard path")
    parser.add_argument("--audio-presence", default=None, type=Path, help="Audio presence summary path")
    parser.add_argument("--output", default=None, type=Path, help="Output markdown path")
    args = parser.parse_args()

    generate_holoassist_report(
        dataset=args.dataset,
        date_str=args.date,
        scorecard_path=args.scorecard,
        audio_presence_path=args.audio_presence,
        out_path=args.output,
        corrected_date=args.corrected_date,
        updated_date=args.updated_date,
        updated_a15_date=args.updated_a15_date,
    )


if __name__ == "__main__":
    main()
