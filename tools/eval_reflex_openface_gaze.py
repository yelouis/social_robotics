#!/usr/bin/env python3
"""Pre-registered confirmatory OpenFace same-phase gaze evaluation and diagnostic matrix on REFLEX (Item B6).

Implements docs/05_wave_b_reflex.md §8 (Issue 13 = C).
Read-only over DATA_ROOT (/Volumes/Extreme SSD/social_robotics) and results/scorecard.jsonl.
Writes docs/evals/2026-11-09_reflex_openface_gaze.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.cache import FeatureCache  # noqa: E402
from harness.items import Item  # noqa: E402
from harness.metrics import MetricResult, auroc_ci, delta_auroc_ci  # noqa: E402
from harness.probes import (  # noqa: E402
    load_features_for_items,
    load_judge_scores,
    logit,
    step_prior_scores,
)
from harness.splits import load_split  # noqa: E402
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash  # noqa: E402
from tools.diag_reflex_v2 import (  # noqa: E402
    ALL_IDX,
    DATA_ROOT,
    FACE_IDX,
    GAZE_IDX,
    ReflexFaceV2Encoder,
    _build_action_side,
    _ci_entry,
    _encode_items,
    _load_dataset_items,
    build_window_dicts,
    compute_fusion_minus_action_best,
    fit_and_score,
)

DEFAULT_OUT_PATH = REPO_ROOT / "docs" / "evals" / "2026-11-09_reflex_openface_gaze.json"

STATUS_TEXT = (
    "pre-registered confirmatory test (same_phase.openface_gaze and "
    "same_phase.openface_fusion_minus_action_fusion, specified November 9, 2026 in "
    "docs/05_wave_b_reflex.md §8 before any OpenFace feature was extracted, evaluated once) "
    "plus secondary diagnostic comparisons; never scorecard rows"
)

OPENFACE_GAZE_COLS: Tuple[str, ...] = (
    "gaze_0_x",
    "gaze_0_y",
    "gaze_0_z",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_1_z",
    "gaze_angle_x",
    "gaze_angle_y",
)

SAME_PHASE_PAIRS: Tuple[Tuple[str, str, Tuple[str, str, str, str], Tuple[str, str, str, str]], ...] = (
    ("pick_1", "pick", ("2", "A", "Carry", "Pre"), ("1", "B", "Pick", "Failure")),
    ("pick_2", "pick", ("3", "B", "Carry", "Pre"), ("4", "A", "Pick", "Failure")),
    ("carry_1", "carry", ("1", "D", "Place", "Pre"), ("1", "B", "Carry", "Failure")),
    ("carry_2", "carry", ("3", "A", "Place", "Pre"), ("2", "A", "Carry", "Failure")),
    ("carry_3", "carry", ("4", "C", "Place", "Pre"), ("3", "B", "Carry", "Failure")),
)


def get_openface_gaze_feature_names() -> List[str]:
    names: List[str] = []
    for c in OPENFACE_GAZE_COLS:
        names.append(f"of_{c}_mean")
    for c in OPENFACE_GAZE_COLS:
        names.append(f"of_{c}_std")
    for c in OPENFACE_GAZE_COLS:
        names.append(f"of_{c}_delta")
    names.append("of_gaze_angle_speed_mean")
    names.append("of_gaze_angle_speed_delta")
    return names


class ReflexOpenFaceGazeEncoder:
    """Automatic 26-d eye-gaze encoder from OpenFace openface.csv (docs/05_wave_b_reflex.md §8.2)."""

    encoder_id: str = "reflex-openface-gaze-v1"

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _load_participant(self, participant_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        key = str(participant_dir.resolve())
        if key in self._cache:
            return self._cache[key]

        csv_path = participant_dir / "openface.csv"
        frames_list: List[int] = []
        vals_list: List[List[float]] = []

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            raw_hdr = next(reader)
            hdr = [c.strip() for c in raw_hdr]
            fi = hdr.index("frame")
            ci = hdr.index("confidence")
            si = hdr.index("success")
            col_idx = [hdr.index(c) for c in OPENFACE_GAZE_COLS]
            min_len = max([fi, ci, si] + col_idx) + 1

            for row in reader:
                if len(row) < min_len:
                    continue
                try:
                    succ = int(float(row[si]))
                    conf = float(row[ci])
                    if succ != 1 or conf < 0.7:
                        continue
                    fr = int(float(row[fi]))
                    gvals = [float(row[idx]) for idx in col_idx]
                except ValueError:
                    continue
                frames_list.append(fr)
                vals_list.append(gvals)

        if frames_list:
            frames_arr = np.asarray(frames_list, dtype=np.int32)
            vals_arr = np.nan_to_num(
                np.asarray(vals_list, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            order = np.argsort(frames_arr, kind="mergesort")
            frames_arr = frames_arr[order]
            vals_arr = vals_arr[order]
        else:
            frames_arr = np.zeros(0, dtype=np.int32)
            vals_arr = np.zeros((0, len(OPENFACE_GAZE_COLS)), dtype=np.float32)

        entry = (frames_arr, vals_arr)
        self._cache[key] = entry
        return entry

    def encode(self, participant_dir: Path, rsf: int, ref: int) -> np.ndarray:
        frames, vals = self._load_participant(participant_dir)
        mask = (frames >= rsf) & (frames <= ref)
        n_valid = int(np.sum(mask))
        if n_valid == 0:
            raise RuntimeError(f"no valid OpenFace frames in [{rsf}, {ref}]")

        sub = vals[mask].astype(np.float64)
        mean_8 = np.mean(sub, axis=0)
        if n_valid >= 2:
            std_8 = np.std(sub, axis=0)
            half = n_valid // 2
            delta_8 = np.mean(sub[half:], axis=0) - np.mean(sub[:half], axis=0)
            gx = sub[:, 6]
            gy = sub[:, 7]
            step_speed = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
            speed_mean = float(np.mean(step_speed)) if len(step_speed) >= 1 else 0.0
            if len(step_speed) >= 2:
                shalf = len(step_speed) // 2
                speed_delta = float(np.mean(step_speed[shalf:]) - np.mean(step_speed[:shalf]))
            else:
                speed_delta = 0.0
        else:
            std_8 = np.zeros(len(OPENFACE_GAZE_COLS), dtype=np.float64)
            delta_8 = np.zeros(len(OPENFACE_GAZE_COLS), dtype=np.float64)
            speed_mean = 0.0
            speed_delta = 0.0

        vec = np.concatenate(
            [
                mean_8,
                std_8,
                delta_8,
                np.array([speed_mean, speed_delta], dtype=np.float64),
            ]
        )
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def encode_items_with_nulls(
    enc: Any,
    items: Sequence[Item],
    windows: Dict[str, Tuple[int, int]],
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode items over windows, returning (X, valid_mask) with honest nulls."""
    n = len(items)
    X = np.zeros((n, dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, it in enumerate(items):
        pdir = Path(it.video_path).parent
        rsf, ref = windows[it.item_id]
        try:
            vec = enc.encode(pdir, rsf, ref)
            X[i] = vec
            valid[i] = True
        except RuntimeError:
            valid[i] = False
    return X, valid


def build_same_phase_items_and_windows(
    reflex_items: Sequence[Item],
    split_data: Dict[str, Any],
    root: Path,
) -> Tuple[
    List[Item],
    List[Item],
    List[Item],
    Dict[str, Tuple[int, int]],
    Dict[str, str],
]:
    """Build the 275 phase-matched pairs (550 items: 380 train, 170 test) per docs/05_wave_b_reflex.md §8.1."""
    tr_orig_ids = set(split_data["train"])
    by_p_frames: Dict[Tuple[str, str, int, int], Item] = {}
    p_dirs: Dict[str, Path] = {}

    for it in reflex_items:
        pid = it.group_id
        p_dirs[pid] = Path(it.video_path).parent
        state = "Pre" if it.label == 1 else "Failure"
        sf, ef = int(it.meta["action_frames"][0]), int(it.meta["action_frames"][1])
        by_p_frames[(pid, state, sf, ef)] = it

    all_sp: List[Item] = []
    tr_sp: List[Item] = []
    te_sp: List[Item] = []
    windows: Dict[str, Tuple[int, int]] = {}
    orig_id_by_new_id: Dict[str, str] = {}

    for pid in sorted(p_dirs):
        acsv = p_dirs[pid] / "analysis.csv"
        rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
        with open(acsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                key = (
                    r["Round No."].strip(),
                    r["Object"].strip(),
                    r["Action"].strip(),
                    r["State"].strip(),
                )
                rows[key] = r

        for pair_id, phase, pos_spec, neg_spec in SAME_PHASE_PAIRS:
            pr = rows[pos_spec]
            nr = rows[neg_spec]
            psf, pef = int(float(pr["Start Frame"])), int(float(pr["End Frame"]))
            nsf, nef = int(float(nr["Start Frame"])), int(float(nr["End Frame"]))
            pos_orig = by_p_frames[(pid, "Pre", psf, pef)]
            neg_orig = by_p_frames[(pid, "Failure", nsf, nef)]

            K = min(pef - psf + 1, nef - nsf + 1)
            pos_w = (pef - K + 1, pef)
            neg_w = (nef - K + 1, nef)
            pair_key = f"{pid}:{pair_id}"

            for orig_it, suffix, w in [(pos_orig, "pos", pos_w), (neg_orig, "neg", neg_w)]:
                new_id = f"reflex_same_phase:{pid}:{pair_id}:{suffix}"
                new_meta = dict(orig_it.meta)
                new_meta["orig_item_id"] = orig_it.item_id
                new_meta["verb"] = phase
                new_meta["noun"] = "object"
                new_meta["pair_id"] = pair_id
                new_meta["pair_key"] = pair_key
                new_meta["equalized_frames"] = [w[0], w[1]]

                cloned = Item(
                    item_id=new_id,
                    dataset="reflex_same_phase",
                    group_id=orig_it.group_id,
                    video_path=orig_it.video_path,
                    action_window_sec=list(orig_it.action_window_sec),
                    reaction_window_sec=list(orig_it.reaction_window_sec),
                    label=orig_it.label,
                    label_source=orig_it.label_source,
                    context_text=orig_it.context_text,
                    official_split=orig_it.official_split,
                    meta=new_meta,
                )
                all_sp.append(cloned)
                windows[new_id] = w
                orig_id_by_new_id[new_id] = orig_it.item_id
                if orig_it.item_id in tr_orig_ids:
                    tr_sp.append(cloned)
                else:
                    te_sp.append(cloned)

    all_sp.sort(key=lambda x: x.item_id)
    tr_sp.sort(key=lambda x: x.item_id)
    te_sp.sort(key=lambda x: x.item_id)
    return all_sp, tr_sp, te_sp, windows, orig_id_by_new_id


def build_uncorrupted_gap_items_and_windows(
    reflex_items: Sequence[Item],
    split_data: Dict[str, Any],
) -> Tuple[List[Item], List[Item], Dict[str, Tuple[int, int]]]:
    """Build uncorrupted full-object handling pairs (Gap 2: R1 Obj C, Gap 4: R2 Obj B) per §8.3 item 4."""
    tr_orig_ids = set(split_data["train"])
    p_dirs: Dict[str, Path] = {}
    p_is_tr: Dict[str, bool] = {}
    p_video: Dict[str, str] = {}

    for it in reflex_items:
        pid = it.group_id
        p_dirs[pid] = Path(it.video_path).parent
        p_is_tr[pid] = it.item_id in tr_orig_ids
        p_video[pid] = it.video_path

    tr_gap: List[Item] = []
    te_gap: List[Item] = []
    windows: Dict[str, Tuple[int, int]] = {}

    gap_specs = [
        ("gap_1", ("1", "D", "Place", "Pre"), ("1", "D", "Place", "Failure")),
        ("gap_2", ("2", "C", "Pick", "Pre"), ("3", "A", "Place", "Failure")),
    ]

    for pid in sorted(p_dirs):
        acsv = p_dirs[pid] / "analysis.csv"
        rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
        with open(acsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                key = (
                    r["Round No."].strip(),
                    r["Object"].strip(),
                    r["Action"].strip(),
                    r["State"].strip(),
                )
                rows[key] = r

        for gap_id, after_gap_pre_spec, fail_spec in gap_specs:
            if after_gap_pre_spec not in rows or fail_spec not in rows:
                continue
            pre_r = rows[after_gap_pre_spec]
            fail_r = rows[fail_spec]
            gap_end = int(float(pre_r["Start Frame"])) - 1
            fsf = int(float(fail_r["Start Frame"]))
            fef = int(float(fail_r["End Frame"]))
            K = fef - fsf + 1
            pos_w = (gap_end - K + 1, gap_end)
            neg_w = (fef - K + 1, fef)
            pair_key = f"{pid}:{gap_id}"

            for suffix, label, w in [("pos", 1, pos_w), ("neg", 0, neg_w)]:
                gid = f"reflex_gap:{pid}:{gap_id}:{suffix}"
                it = Item(
                    item_id=gid,
                    dataset="reflex_gap",
                    group_id=pid,
                    video_path=p_video[pid],
                    action_window_sec=[1.0, 2.0],
                    reaction_window_sec=[1.0, 2.0],
                    label=label,
                    label_source="analysis.csv",
                    context_text="",
                    meta={"pair_key": pair_key, "gap_id": gap_id},
                )
                windows[gid] = w
                if p_is_tr[pid]:
                    tr_gap.append(it)
                else:
                    te_gap.append(it)

    tr_gap.sort(key=lambda x: x.item_id)
    te_gap.sort(key=lambda x: x.item_id)
    return tr_gap, te_gap, windows


def evaluate_confirmatory_verdict(
    standalone: MetricResult,
    ablation_delta: MetricResult,
) -> Dict[str, Any]:
    """Evaluate the pre-registered confirmatory decision rule (docs/05_wave_b_reflex.md §8.3)."""
    standalone_above_chance = bool(
        standalone.ci_low is not None and float(standalone.ci_low) > 0.5
    )
    ablation_gain_positive = bool(
        ablation_delta.ci_low is not None and float(ablation_delta.ci_low) > 0.0
    )
    confirmatory_supported = bool(standalone_above_chance and ablation_gain_positive)
    h1_pass_bar_met = bool(
        standalone.point is not None
        and float(standalone.point) >= 0.65
        and ablation_delta.point is not None
        and float(ablation_delta.point) >= 0.03
        and ablation_gain_positive
    )
    return {
        "standalone_above_chance": standalone_above_chance,
        "ablation_gain_positive": ablation_gain_positive,
        "confirmatory_supported": confirmatory_supported,
        "h1_pass_bar_met": h1_pass_bar_met,
        "verdict": "SUPPORTED" if confirmatory_supported else "NOT SUPPORTED",
    }


def _build_same_phase_action_side(
    tr_sp: Sequence[Item],
    te_sp: Sequence[Item],
    orig_item_map: Dict[str, Item],
    root: Path,
) -> Dict[str, Any]:
    """Build action-only controls on same_phase using cached reflex features and physical-phase step-prior."""
    tr_orig = [orig_item_map[str(it.meta["orig_item_id"])] for it in tr_sp]
    te_orig = [orig_item_map[str(it.meta["orig_item_id"])] for it in te_sp]

    ytr = np.array([it.label for it in tr_sp], dtype=int)
    yte = np.array([it.label for it in te_sp], dtype=int)
    gte = [it.group_id for it in te_sp]
    n_te = len(te_sp)

    sig_cache = FeatureCache(dataset="reflex", encoder_id="siglip-b16-224-u9", data_root=root)
    Xs_tr, vs_tr, _ = load_features_for_items(tr_orig, sig_cache)
    Xs_te, vs_te, _ = load_features_for_items(te_orig, sig_cache)

    probs_act_valid, _, _ = fit_and_score(Xs_tr[vs_tr], ytr[vs_tr], Xs_te[vs_te])
    probs_act_te = np.full(n_te, np.nan, dtype=np.float64)
    probs_act_te[vs_te] = probs_act_valid

    ph = prompt_hash(DATASET_QUESTIONS["reflex"])
    j_tr, vj_tr, _ = load_judge_scores(tr_orig, "reflex", "qwen2.5vl_7b", ph, root)
    j_te, vj_te, _ = load_judge_scores(te_orig, "reflex", "qwen2.5vl_7b", ph, root)
    j_lg_te, vj_lg_te, _ = load_judge_scores(te_orig, "reflex", "gemma4_26b", ph, root)

    # Physical-phase step_prior on tr_sp / te_sp (where meta["verb"] is the balanced physical step)
    sp_tr = step_prior_scores(list(tr_sp), list(tr_sp), ytr, loo=True)
    sp_te = step_prior_scores(list(tr_sp), list(te_sp), ytr, loo=False)

    mtr = vj_tr & vs_tr
    mte = vj_te & vs_te
    Xaf_tr = np.hstack([logit(j_tr[mtr])[:, None], logit(sp_tr[mtr])[:, None], Xs_tr[mtr]])
    Xaf_te = np.hstack([logit(j_te[mte])[:, None], logit(sp_te[mte])[:, None], Xs_te[mte]])

    p_af_valid, _, _ = fit_and_score(Xaf_tr, ytr[mtr], Xaf_te)
    paf_te = np.full(n_te, np.nan, dtype=np.float64)
    paf_te[mte] = p_af_valid

    candidates: Dict[str, np.ndarray] = {
        "action-probe": probs_act_te,
        "step-prior": sp_te,
        "action-fusion": paf_te,
    }
    if int(np.sum(vj_te)) >= 0.9 * n_te:
        candidates["judge"] = j_te
    if int(np.sum(vj_lg_te)) >= 0.9 * n_te:
        candidates["judge-large"] = j_lg_te

    return {
        "ytr": ytr,
        "yte": yte,
        "gte": gte,
        "n_te": n_te,
        "mtr": mtr,
        "mte": mte,
        "Xaf_tr": Xaf_tr,
        "Xaf_te": Xaf_te,
        "paf_te": paf_te,
        "probs_act_te": probs_act_te,
        "j_te": j_te,
        "j_lg_te": j_lg_te,
        "sp_te": sp_te,
        "candidates": candidates,
    }


def _score_reaction_with_fusion(
    X_tr: np.ndarray,
    v_tr: np.ndarray,
    X_te: np.ndarray,
    v_te: np.ndarray,
    act_side: Dict[str, Any],
) -> Tuple[MetricResult, MetricResult, MetricResult, MetricResult, np.ndarray]:
    """Fit reaction-only probe and fusion probe on valid masks, returning standalone, fusion, delta_af, delta_ab."""
    ytr = act_side["ytr"]
    yte = act_side["yte"]
    gte = act_side["gte"]
    n_te = len(yte)
    mtr_act = act_side["mtr"]
    mte_act = act_side["mte"]

    probs_valid, _, _ = fit_and_score(X_tr[v_tr], ytr[v_tr], X_te[v_te])
    probs_te = np.full(n_te, np.nan, dtype=np.float64)
    probs_te[v_te] = probs_valid
    res_standalone = auroc_ci(yte, probs_te, gte)

    # Full action+reaction fusion on intersection mask
    mtr_all = mtr_act & v_tr
    mte_all = mte_act & v_te

    # Reconstruct full-length Xaf arrays to index by mtr_all / mte_all
    Xaf_tr_full = np.zeros((len(ytr), act_side["Xaf_tr"].shape[1]), dtype=np.float32)
    Xaf_te_full = np.zeros((n_te, act_side["Xaf_te"].shape[1]), dtype=np.float32)
    Xaf_tr_full[mtr_act] = act_side["Xaf_tr"]
    Xaf_te_full[mte_act] = act_side["Xaf_te"]

    # Refit action-fusion on the exact same valid reaction+action mask so paired delta compares identical items
    p_af_sub_valid, _, _ = fit_and_score(Xaf_tr_full[mtr_all], ytr[mtr_all], Xaf_te_full[mte_all])
    paf_sub_te = np.full(n_te, np.nan, dtype=np.float64)
    paf_sub_te[mte_all] = p_af_sub_valid

    X_fus_tr = np.hstack([Xaf_tr_full[mtr_all], X_tr[mtr_all]])
    X_fus_te = np.hstack([Xaf_te_full[mte_all], X_te[mte_all]])
    p_fus_valid, _, _ = fit_and_score(X_fus_tr, ytr[mtr_all], X_fus_te)
    pfus_te = np.full(n_te, np.nan, dtype=np.float64)
    pfus_te[mte_all] = p_fus_valid

    res_fus = auroc_ci(yte, pfus_te, gte)
    res_delta_af = delta_auroc_ci(yte, paf_sub_te, pfus_te, gte)
    _, res_delta_ab = compute_fusion_minus_action_best(
        y_test=yte,
        fusion_scores=pfus_te,
        groups_test=gte,
        candidate_scores=act_side["candidates"],
    )
    return res_standalone, res_fus, res_delta_af, res_delta_ab, probs_te


def _pair_complete_mask(items: Sequence[Item], valid: np.ndarray) -> np.ndarray:
    """Return boolean mask keeping only items whose pair_key has both pos and neg valid."""
    counts: Dict[str, int] = {}
    for it, v in zip(items, valid):
        if bool(v):
            pk = str(it.meta["pair_key"])
            counts[pk] = counts.get(pk, 0) + 1
    return np.array([counts.get(str(it.meta["pair_key"]), 0) == 2 for it in items], dtype=bool)


def evaluate_reflex_openface_gaze(
    root: Path = DATA_ROOT,
    splits_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if splits_path is None:
        splits_path = REPO_ROOT / "splits"

    out: Dict[str, Any] = {"_status": STATUS_TEXT}

    reflex_items, tr_r, te_r = _load_dataset_items("reflex", root, splits_path)
    orig_item_map = {it.item_id: it for it in reflex_items}
    split_data = load_split("reflex", path_dir=splits_path)

    of_enc = ReflexOpenFaceGazeEncoder()
    v2_enc = ReflexFaceV2Encoder()

    # ------------------------------------------------------------------
    # 1. Confirmatory & diagnostic same-phase contrast (550 items)
    # ------------------------------------------------------------------
    _, tr_sp, te_sp, sp_windows, _ = build_same_phase_items_and_windows(
        reflex_items=reflex_items,
        split_data=split_data,
        root=root,
    )
    Xof_sp_tr, vof_sp_tr = encode_items_with_nulls(of_enc, tr_sp, sp_windows, 26)
    Xof_sp_te, vof_sp_te = encode_items_with_nulls(of_enc, te_sp, sp_windows, 26)

    pc_tr_mask = _pair_complete_mask(tr_sp, vof_sp_tr)
    pc_te_mask = _pair_complete_mask(te_sp, vof_sp_te)

    out["same_phase.n_train_items"] = {"value": int(len(tr_sp))}
    out["same_phase.n_test_items"] = {"value": int(len(te_sp))}
    out["same_phase.n_train_valid_openface_items"] = {"value": int(np.sum(vof_sp_tr))}
    out["same_phase.n_test_valid_openface_items"] = {"value": int(np.sum(vof_sp_te))}
    out["same_phase.n_train_valid_openface_pairs"] = {"value": int(np.sum(pc_tr_mask) // 2)}
    out["same_phase.n_test_valid_openface_pairs"] = {"value": int(np.sum(pc_te_mask) // 2)}

    sp_act = _build_same_phase_action_side(tr_sp, te_sp, orig_item_map, root)
    yte_sp = sp_act["yte"]
    gte_sp = sp_act["gte"]
    n_te_sp = sp_act["n_te"]

    out["same_phase.action_probe"] = _ci_entry(auroc_ci(yte_sp, sp_act["probs_act_te"], gte_sp), n_te_sp)
    out["same_phase.judge"] = _ci_entry(auroc_ci(yte_sp, sp_act["j_te"], gte_sp), n_te_sp)
    out["same_phase.judge_large"] = _ci_entry(auroc_ci(yte_sp, sp_act["j_lg_te"], gte_sp), n_te_sp)
    out["same_phase.step_prior"] = _ci_entry(auroc_ci(yte_sp, sp_act["sp_te"], gte_sp), n_te_sp)
    out["same_phase.action_fusion"] = _ci_entry(auroc_ci(yte_sp, sp_act["paf_te"], gte_sp), n_te_sp)

    len_sp_te = np.array(
        [float(sp_windows[it.item_id][1] - sp_windows[it.item_id][0] + 1) for it in te_sp],
        dtype=np.float64,
    )
    out["same_phase.window_length"] = _ci_entry(auroc_ci(yte_sp, len_sp_te, gte_sp), n_te_sp)

    # Primary confirmatory check: automatic OpenFace gaze on same_phase
    res_of_sp, res_of_fus_sp, res_of_daf_sp, res_of_dab_sp, probs_of_sp_te = _score_reaction_with_fusion(
        X_tr=Xof_sp_tr,
        v_tr=vof_sp_tr,
        X_te=Xof_sp_te,
        v_te=vof_sp_te,
        act_side=sp_act,
    )
    out["same_phase.openface_gaze"] = _ci_entry(res_of_sp, n_te_sp)
    out["same_phase.openface_fusion"] = _ci_entry(res_of_fus_sp, n_te_sp)
    out["same_phase.openface_fusion_minus_action_fusion"] = _ci_entry(res_of_daf_sp, n_te_sp)
    out["same_phase.openface_fusion_minus_action_best"] = _ci_entry(res_of_dab_sp, n_te_sp)
    out["_verdict"] = evaluate_confirmatory_verdict(res_of_sp, res_of_daf_sp)

    # Pair-complete sensitivity check
    res_of_pc, _, res_of_daf_pc, _, _ = _score_reaction_with_fusion(
        X_tr=Xof_sp_tr,
        v_tr=pc_tr_mask,
        X_te=Xof_sp_te,
        v_te=pc_te_mask,
        act_side=sp_act,
    )
    out["same_phase.openface_gaze_pair_complete"] = _ci_entry(res_of_pc, n_te_sp)
    out["same_phase.openface_fusion_minus_action_fusion_pair_complete"] = _ci_entry(
        res_of_daf_pc, n_te_sp
    )

    # Phase strata (pick vs carry) for OpenFace gaze
    for phase_name in ["pick", "carry"]:
        tr_ph_mask = np.array([str(it.meta["verb"]) == phase_name for it in tr_sp], dtype=bool) & vof_sp_tr
        te_ph_idx = [i for i, it in enumerate(te_sp) if str(it.meta["verb"]) == phase_name]
        te_ph_valid = [i for i in te_ph_idx if bool(vof_sp_te[i])]
        p_ph_valid, _, _ = fit_and_score(
            Xof_sp_tr[tr_ph_mask],
            sp_act["ytr"][tr_ph_mask],
            Xof_sp_te[te_ph_valid],
        )
        p_ph_te = np.full(len(te_ph_idx), np.nan, dtype=np.float64)
        valid_pos_map = {orig_i: pos_i for pos_i, orig_i in enumerate(te_ph_idx)}
        for orig_i, val in zip(te_ph_valid, p_ph_valid):
            p_ph_te[valid_pos_map[orig_i]] = val
        res_ph = auroc_ci(
            yte_sp[te_ph_idx],
            p_ph_te,
            [gte_sp[i] for i in te_ph_idx],
        )
        out[f"same_phase.openface_gaze_{phase_name}"] = _ci_entry(res_ph, len(te_ph_idx))

    # Semi-manual gaze.csv, face_only, and v2_all on same_phase
    Xv2_sp_tr = _encode_items(v2_enc, tr_sp, sp_windows)
    Xv2_sp_te = _encode_items(v2_enc, te_sp, sp_windows)
    all_valid_tr = np.ones(len(tr_sp), dtype=bool)
    all_valid_te = np.ones(len(te_sp), dtype=bool)

    for label_prefix, col_idx in [
        ("dataset_gaze", GAZE_IDX),
        ("face_only", FACE_IDX),
        ("v2_all", ALL_IDX),
    ]:
        res_st, res_fu, res_daf, res_dab, _ = _score_reaction_with_fusion(
            X_tr=Xv2_sp_tr[:, col_idx],
            v_tr=all_valid_tr,
            X_te=Xv2_sp_te[:, col_idx],
            v_te=all_valid_te,
            act_side=sp_act,
        )
        out[f"same_phase.{label_prefix}"] = _ci_entry(res_st, n_te_sp)
        out[f"same_phase.{label_prefix}_fusion"] = _ci_entry(res_fu, n_te_sp)
        out[f"same_phase.{label_prefix}_fusion_minus_action_fusion"] = _ci_entry(res_daf, n_te_sp)
        if label_prefix == "dataset_gaze":
            out["same_phase.dataset_gaze_fusion_minus_action_best"] = _ci_entry(res_dab, n_te_sp)

    for phase_name in ["pick", "carry"]:
        tr_ph_mask = np.array([str(it.meta["verb"]) == phase_name for it in tr_sp], dtype=bool)
        te_ph_idx = [i for i, it in enumerate(te_sp) if str(it.meta["verb"]) == phase_name]
        p_dg_ph, _, _ = fit_and_score(
            Xv2_sp_tr[tr_ph_mask][:, GAZE_IDX],
            sp_act["ytr"][tr_ph_mask],
            Xv2_sp_te[te_ph_idx][:, GAZE_IDX],
        )
        res_dg_ph = auroc_ci(
            yte_sp[te_ph_idx],
            p_dg_ph,
            [gte_sp[i] for i in te_ph_idx],
        )
        out[f"same_phase.dataset_gaze_{phase_name}"] = _ci_entry(res_dg_ph, len(te_ph_idx))

    # ------------------------------------------------------------------
    # 2. Within-episode reflex comparison (1,208 items)
    # ------------------------------------------------------------------
    _, start_w, _, act_end_w = build_window_dicts(reflex_items)
    r_act = _build_action_side("reflex", tr_r, te_r, root)
    n_te_r = len(r_act["yte"])

    for mode_name, w_dict in [("action_equal_end", act_end_w), ("equal_start", start_w)]:
        Xof_r_tr, vof_r_tr = encode_items_with_nulls(of_enc, tr_r, w_dict, 26)
        Xof_r_te, vof_r_te = encode_items_with_nulls(of_enc, te_r, w_dict, 26)
        res_st, res_fu, res_daf, res_dab, _ = _score_reaction_with_fusion(
            X_tr=Xof_r_tr,
            v_tr=vof_r_tr,
            X_te=Xof_r_te,
            v_te=vof_r_te,
            act_side=r_act,
        )
        out[f"reflex.openface_gaze_{mode_name}"] = _ci_entry(res_st, n_te_r)
        out[f"reflex.openface_fusion_{mode_name}"] = _ci_entry(res_fu, n_te_r)
        out[f"reflex.openface_fusion_minus_action_fusion_{mode_name}"] = _ci_entry(res_daf, n_te_r)
        out[f"reflex.openface_fusion_minus_action_best_{mode_name}"] = _ci_entry(res_dab, n_te_r)

    # Early-vs-late Pre placebo with automatic OpenFace gaze
    pre_tr = [it for it in tr_r if it.label == 1]
    pre_te = [it for it in te_r if it.label == 1]
    X_pl_tr: List[np.ndarray] = []
    y_pl_tr: List[int] = []
    for it in pre_tr:
        pdir = Path(it.video_path).parent
        rsf, ref = int(it.meta["reaction_frames"][0]), int(it.meta["reaction_frames"][1])
        L = ref - rsf + 1
        h = L // 2
        if h < 2:
            continue
        try:
            v_early = of_enc.encode(pdir, rsf, rsf + h - 1)
            v_late = of_enc.encode(pdir, ref - h + 1, ref)
        except RuntimeError:
            continue
        X_pl_tr.extend([v_early, v_late])
        y_pl_tr.extend([1, 0])

    X_pl_te: List[np.ndarray] = []
    y_pl_te: List[int] = []
    g_pl_te: List[str] = []
    v_pl_te: List[bool] = []
    for it in pre_te:
        pdir = Path(it.video_path).parent
        rsf, ref = int(it.meta["reaction_frames"][0]), int(it.meta["reaction_frames"][1])
        L = ref - rsf + 1
        h = L // 2
        for w_pair, lbl in [((rsf, rsf + h - 1), 1), ((ref - h + 1, ref), 0)]:
            y_pl_te.append(lbl)
            g_pl_te.append(it.group_id)
            if h < 2:
                X_pl_te.append(np.zeros(26, dtype=np.float32))
                v_pl_te.append(False)
                continue
            try:
                X_pl_te.append(of_enc.encode(pdir, w_pair[0], w_pair[1]))
                v_pl_te.append(True)
            except RuntimeError:
                X_pl_te.append(np.zeros(26, dtype=np.float32))
                v_pl_te.append(False)

    X_pl_tr_arr = np.asarray(X_pl_tr, dtype=np.float32)
    y_pl_tr_arr = np.asarray(y_pl_tr, dtype=int)
    X_pl_te_arr = np.asarray(X_pl_te, dtype=np.float32)
    y_pl_te_arr = np.asarray(y_pl_te, dtype=int)
    v_pl_te_arr = np.asarray(v_pl_te, dtype=bool)
    p_pl_valid, _, _ = fit_and_score(X_pl_tr_arr, y_pl_tr_arr, X_pl_te_arr[v_pl_te_arr])
    p_pl_te = np.full(len(y_pl_te_arr), np.nan, dtype=np.float64)
    p_pl_te[v_pl_te_arr] = p_pl_valid
    out["reflex.placebo_early_vs_late_pre_openface_gaze"] = _ci_entry(
        auroc_ci(y_pl_te_arr, p_pl_te, g_pl_te),
        len(y_pl_te_arr),
    )

    # Onset straddle with automatic OpenFace gaze
    episodes: Dict[str, List[Item]] = {}
    for it in reflex_items:
        episodes.setdefault(it.item_id.rsplit(":", 1)[0], []).append(it)
    straddle_w: Dict[str, Tuple[int, int]] = {}
    for ep_items in episodes.values():
        it_pre = [it for it in ep_items if it.label == 1][0]
        it_fail = [it for it in ep_items if it.label == 0][0]
        psf, pef = int(it_pre.meta["reaction_frames"][0]), int(it_pre.meta["reaction_frames"][1])
        fsf, fef = int(it_fail.meta["action_frames"][0]), int(it_fail.meta["action_frames"][1])
        K_on = min(15, pef - psf + 1, fef - fsf + 1)
        straddle_w[it_pre.item_id] = (pef - K_on + 1, pef)
        straddle_w[it_fail.item_id] = (fsf, fsf + K_on - 1)

    X_st_tr, v_st_tr = encode_items_with_nulls(of_enc, tr_r, straddle_w, 26)
    X_st_te, v_st_te = encode_items_with_nulls(of_enc, te_r, straddle_w, 26)
    p_st_valid, _, _ = fit_and_score(X_st_tr[v_st_tr], r_act["ytr"][v_st_tr], X_st_te[v_st_te])
    p_st_te = np.full(n_te_r, np.nan, dtype=np.float64)
    p_st_te[v_st_te] = p_st_valid
    out["reflex.onset_straddle_openface_gaze"] = _ci_entry(
        auroc_ci(r_act["yte"], p_st_te, r_act["gte"]),
        n_te_r,
    )

    # ------------------------------------------------------------------
    # 3. Uncorrupted full-object handlings (Gap 2 & Gap 4 vs Place.Failure)
    # ------------------------------------------------------------------
    tr_gap, te_gap, gap_w = build_uncorrupted_gap_items_and_windows(reflex_items, split_data)
    X_gap_tr, v_gap_tr = encode_items_with_nulls(of_enc, tr_gap, gap_w, 26)
    X_gap_te, v_gap_te = encode_items_with_nulls(of_enc, te_gap, gap_w, 26)
    y_gap_tr = np.array([it.label for it in tr_gap], dtype=int)
    y_gap_te = np.array([it.label for it in te_gap], dtype=int)
    g_gap_te = [it.group_id for it in te_gap]

    p_gap_valid, _, _ = fit_and_score(X_gap_tr[v_gap_tr], y_gap_tr[v_gap_tr], X_gap_te[v_gap_te])
    p_gap_te = np.full(len(te_gap), np.nan, dtype=np.float64)
    p_gap_te[v_gap_te] = p_gap_valid
    out["uncorrupted_gap.openface_gaze"] = _ci_entry(
        auroc_ci(y_gap_te, p_gap_te, g_gap_te),
        len(te_gap),
    )

    pc_gap_tr = _pair_complete_mask(tr_gap, v_gap_tr)
    pc_gap_te = _pair_complete_mask(te_gap, v_gap_te)
    p_gap_pc_valid, _, _ = fit_and_score(
        X_gap_tr[pc_gap_tr],
        y_gap_tr[pc_gap_tr],
        X_gap_te[pc_gap_te],
    )
    p_gap_pc_te = np.full(len(te_gap), np.nan, dtype=np.float64)
    p_gap_pc_te[pc_gap_te] = p_gap_pc_valid
    out["uncorrupted_gap.openface_gaze_pair_complete"] = _ci_entry(
        auroc_ci(y_gap_te, p_gap_pc_te, g_gap_te),
        len(te_gap),
    )

    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output JSON path (default: docs/evals/2026-11-09_reflex_openface_gaze.json).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path to DATA_ROOT.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    res = evaluate_reflex_openface_gaze(root=args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
