#!/usr/bin/env python3
"""Post-hoc diagnostic script for exploratory REFLEX v2 and reviewer falsification checks (Item B5).

Read-only over DATA_ROOT (/Volumes/Extreme SSD/social_robotics) and results/scorecard.jsonl.
Writes docs/evals/2026-11-08_reflex_v2_posthoc.json.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features.cache import FeatureCache
from harness.items import Item, read_items
from harness.metrics import MetricResult, auroc_ci, delta_auroc_ci
from harness.probes import (
    _safe_predict_proba,
    load_features_for_items,
    load_judge_scores,
    logit,
    step_prior_scores,
)
from harness.splits import load_split
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash
DATA_ROOT = Path("/Volumes/Extreme SSD/social_robotics")


STATUS_TEXT = (
    "exploratory, post hoc: features and windows chosen after about 80 test-set evaluations "
    "on the 17 REFLEX test participants (November 7-8, 2026); not pre-registered; never scorecard rows"
)

GAZE_FEATURE_NAMES: Tuple[str, ...] = (
    "gaze_robot_prop",
    "gaze_task_prop",
    "gaze_misc_prop",
    "gaze_robot_delta",
    "gaze_task_delta",
    "gaze_misc_delta",
    "gaze_switch_rate",
    "gaze_task_to_robot_rate",
    "gaze_robot_to_task_rate",
)

CORE_HUME_COLS: Tuple[str, ...] = (
    "Surprise (negative)",
    "Confusion",
    "Awkwardness",
    "Doubt",
    "Disappointment",
    "Concentration",
    "Calmness",
    "Interest",
    "Smile",
    "Frown",
    "Wide-eyed",
    "Wince",
    "AU4 Brow Lowerer",
    "AU12 Lip Corner Puller",
    "AU25 Lips Part",
    "AU53 Head Up",
    "AU54 Head Down",
)

FACETORCH_COLS: Tuple[str, ...] = ("Valence", "Arousal")

GAZE_IDX: List[int] = list(range(0, 9))
HUME_IDX: List[int] = list(range(9, 43))
FT_IDX: List[int] = list(range(43, 47))
FACE_IDX: List[int] = HUME_IDX + FT_IDX
ALL_IDX: List[int] = list(range(0, 47))

ROBOTISH_RE = re.compile(
    r"\b(?:I can't|I cannot|I couldn't|I could not|I failed|going for object|hand it to me|take the object|place it for me)\b",
    re.I,
)


def get_reflex_face_v2_feature_names() -> List[str]:
    names: List[str] = list(GAZE_FEATURE_NAMES)
    for c in CORE_HUME_COLS:
        names.append(f"hume_{c}_mean")
    for c in CORE_HUME_COLS:
        names.append(f"hume_{c}_delta")
    for c in FACETORCH_COLS:
        names.append(f"ft_{c}_mean")
    for c in FACETORCH_COLS:
        names.append(f"ft_{c}_delta")
    return names


class ReflexFaceV2Encoder:
    """Length-invariant, baseline-subtracted (late - early delta) encoder for REFLEX."""

    encoder_id: str = "reflex-face-v2"

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, object]] = {}

    def _load_participant(self, participant_dir: Path) -> Dict[str, object]:
        key = str(participant_dir.resolve())
        if key in self._cache:
            return self._cache[key]

        # 1. gaze.csv (1-indexed Frame)
        g_df = pd.read_csv(participant_dir / "gaze.csv")
        g_frames = g_df["Frame"].astype(int).to_numpy()
        g_str = g_df["Gaze"].astype(str).str.strip().to_numpy()
        g_rob = (g_str == "Robot").astype(np.float32)
        g_tsk = (g_str == "Task").astype(np.float32)
        g_msc = (g_str == "Miscellaneous").astype(np.float32)

        # 2. hume.csv (1-indexed Frame)
        h_df = pd.read_csv(participant_dir / "hume.csv")
        h_frames = h_df["Frame"].astype(int).to_numpy()
        h_vals = np.nan_to_num(
            h_df[list(CORE_HUME_COLS)].to_numpy(dtype=np.float32),
            nan=0.0,
        )

        # 3. facetorch.csv (0-indexed Frame ID -> +1 for 1-indexed)
        ft_df = pd.read_csv(participant_dir / "facetorch.csv")
        ft_df.columns = [c.strip() for c in ft_df.columns]
        ft_df = ft_df.dropna(subset=["Valence", "Arousal"])
        ft_frames = ft_df["Frame ID"].astype(int).to_numpy() + 1
        ft_vals = np.nan_to_num(
            ft_df[list(FACETORCH_COLS)].to_numpy(dtype=np.float32),
            nan=0.0,
        )

        entry: Dict[str, object] = {
            "gaze": (g_frames, g_rob, g_tsk, g_msc, g_str),
            "hume": (h_frames, h_vals),
            "ft": (ft_frames, ft_vals),
        }
        self._cache[key] = entry
        return entry

    def encode(self, participant_dir: Path, rsf: int, ref: int) -> np.ndarray:
        d = self._load_participant(participant_dir)

        gf, grob, gtsk, gmsc, gstr = d["gaze"]  # type: ignore[misc]
        g_mask = (gf >= rsf) & (gf <= ref)
        n_g = int(np.sum(g_mask))
        if n_g == 0:
            raise RuntimeError(f"no valid gaze frames in [{rsf}, {ref}]")

        r_sub = grob[g_mask]
        t_sub = gtsk[g_mask]
        m_sub = gmsc[g_mask]
        s_sub = gstr[g_mask]

        p_rob = float(np.mean(r_sub))
        p_tsk = float(np.mean(t_sub))
        p_msc = float(np.mean(m_sub))
        if n_g >= 2:
            half = n_g // 2
            d_rob = float(np.mean(r_sub[half:]) - np.mean(r_sub[:half]))
            d_tsk = float(np.mean(t_sub[half:]) - np.mean(t_sub[:half]))
            d_msc = float(np.mean(m_sub[half:]) - np.mean(m_sub[:half]))
            sw = float(np.mean(s_sub[1:] != s_sub[:-1]))
            t2r = float(np.mean((s_sub[:-1] == "Task") & (s_sub[1:] == "Robot")))
            r2t = float(np.mean((s_sub[:-1] == "Robot") & (s_sub[1:] == "Task")))
        else:
            d_rob, d_tsk, d_msc, sw, t2r, r2t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        g_vec = np.array(
            [p_rob, p_tsk, p_msc, d_rob, d_tsk, d_msc, sw, t2r, r2t],
            dtype=np.float32,
        )

        hf, hvals = d["hume"]  # type: ignore[misc]
        h_sub = hvals[(hf >= rsf) & (hf <= ref)]
        if len(h_sub) == 0:
            h_mean = np.zeros(len(CORE_HUME_COLS), dtype=np.float32)
            h_delta = np.zeros(len(CORE_HUME_COLS), dtype=np.float32)
        else:
            h_mean = np.mean(h_sub, axis=0)
            if len(h_sub) >= 2:
                half = len(h_sub) // 2
                h_delta = np.mean(h_sub[half:], axis=0) - np.mean(h_sub[:half], axis=0)
            else:
                h_delta = np.zeros(len(CORE_HUME_COLS), dtype=np.float32)

        ftf, ftvals = d["ft"]  # type: ignore[misc]
        ft_sub = ftvals[(ftf >= rsf) & (ftf <= ref)]
        if len(ft_sub) == 0:
            ft_mean = np.zeros(2, dtype=np.float32)
            ft_delta = np.zeros(2, dtype=np.float32)
        else:
            ft_mean = np.mean(ft_sub, axis=0)
            if len(ft_sub) >= 2:
                half = len(ft_sub) // 2
                ft_delta = np.mean(ft_sub[half:], axis=0) - np.mean(ft_sub[:half], axis=0)
            else:
                ft_delta = np.zeros(2, dtype=np.float32)

        return np.concatenate([g_vec, h_mean, h_delta, ft_mean, ft_delta])


def fit_and_score(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clf.fit(X_tr_s, y_tr)
    probs = _safe_predict_proba(clf, X_te_s)[:, 1]
    return probs, scaler, clf


def build_window_dicts(
    reflex_items: Sequence[Item],
) -> Tuple[
    Dict[str, Tuple[int, int]],
    Dict[str, Tuple[int, int]],
    Dict[str, Tuple[int, int]],
    Dict[str, Tuple[int, int]],
]:
    """Build the four REFLEX window schemes from eval_reflex_v2.py unchanged."""
    episodes: Dict[str, List[Item]] = {}
    for it in reflex_items:
        ep_key = it.item_id.rsplit(":", 1)[0]
        episodes.setdefault(ep_key, []).append(it)

    orig_w: Dict[str, Tuple[int, int]] = {}
    start_w: Dict[str, Tuple[int, int]] = {}
    end_w: Dict[str, Tuple[int, int]] = {}
    act_end_w: Dict[str, Tuple[int, int]] = {}

    for ep_items in episodes.values():
        K = min(
            int(it.meta["reaction_frames"][1]) - int(it.meta["reaction_frames"][0]) + 1
            for it in ep_items
        )
        K_act = min(
            int(it.meta["action_frames"][1]) - int(it.meta["action_frames"][0]) + 1
            for it in ep_items
        )
        for it in ep_items:
            rsf, ref = int(it.meta["reaction_frames"][0]), int(it.meta["reaction_frames"][1])
            _, aef = int(it.meta["action_frames"][0]), int(it.meta["action_frames"][1])
            orig_w[it.item_id] = (rsf, ref)
            start_w[it.item_id] = (rsf, rsf + K - 1)
            end_w[it.item_id] = (ref - K + 1, ref)
            act_end_w[it.item_id] = (aef - K_act + 1, aef)

    return orig_w, start_w, end_w, act_end_w


def select_action_best_scores(
    y_test: np.ndarray,
    groups_test: Sequence[str],
    candidate_scores: Dict[str, np.ndarray],
) -> Tuple[str, np.ndarray, float]:
    """Select the action-only control with the highest test AUROC (docs/03_eval_harness.md §6)."""
    best_name = ""
    best_scores: Optional[np.ndarray] = None
    best_point = -1.0
    g_list = list(groups_test)
    for name, scores in candidate_scores.items():
        res = auroc_ci(y_test, scores, g_list)
        if res.point is not None and float(res.point) > best_point:
            best_point = float(res.point)
            best_name = name
            best_scores = np.asarray(scores, dtype=np.float64)
    if best_scores is None:
        raise ValueError("No valid action-only candidate scores provided")
    return best_name, best_scores, best_point


def compute_fusion_minus_action_best(
    y_test: np.ndarray,
    fusion_scores: np.ndarray,
    groups_test: Sequence[str],
    candidate_scores: Dict[str, np.ndarray],
) -> Tuple[str, MetricResult]:
    """Compute fusion_minus_action_best against the highest-test-AUROC action-only control."""
    best_name, best_scores, _ = select_action_best_scores(
        y_test=y_test,
        groups_test=groups_test,
        candidate_scores=candidate_scores,
    )
    return best_name, delta_auroc_ci(y_test, best_scores, fusion_scores, list(groups_test))


def _ci_entry(res: MetricResult, n_total: int) -> Dict[str, Any]:
    return {
        "value": float(res.point) if res.point is not None else None,
        "ci_low": float(res.ci_low) if res.ci_low is not None else None,
        "ci_high": float(res.ci_high) if res.ci_high is not None else None,
        "n_test": int(n_total - res.n_excluded),
    }


def _encode_items(
    enc: ReflexFaceV2Encoder,
    items: Sequence[Item],
    windows: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    return np.array([enc.encode(Path(it.video_path).parent, *windows[it.item_id]) for it in items])


def _load_dataset_items(
    dataset: str,
    root: Path,
    splits_path: Path,
) -> Tuple[List[Item], List[Item], List[Item]]:
    items = read_items(root / "items" / dataset / "items.jsonl")
    item_map = {it.item_id: it for it in items}
    split_data = load_split(dataset, path_dir=splits_path)
    tr = [item_map[i] for i in split_data["train"] if i in item_map]
    te = [item_map[i] for i in split_data["test"] if i in item_map]
    return items, tr, te


def _build_action_side(
    dataset: str,
    tr: Sequence[Item],
    te: Sequence[Item],
    root: Path,
) -> Dict[str, Any]:
    """Build action-only inputs and candidate controls on dataset."""
    ytr = np.array([it.label for it in tr], dtype=int)
    yte = np.array([it.label for it in te], dtype=int)
    gte = [it.group_id for it in te]
    n_te = len(te)

    sig_cache = FeatureCache(dataset=dataset, encoder_id="siglip-b16-224-u9", data_root=root)
    Xs_tr, vs_tr, _ = load_features_for_items(list(tr), sig_cache)
    Xs_te, vs_te, _ = load_features_for_items(list(te), sig_cache)

    probs_act_valid, _, _ = fit_and_score(Xs_tr[vs_tr], ytr[vs_tr], Xs_te[vs_te])
    probs_act_te = np.full(n_te, np.nan, dtype=np.float64)
    probs_act_te[vs_te] = probs_act_valid

    ph = prompt_hash(DATASET_QUESTIONS[dataset])
    j_tr, vj_tr, _ = load_judge_scores(list(tr), dataset, "qwen2.5vl_7b", ph, root)
    j_te, vj_te, _ = load_judge_scores(list(te), dataset, "qwen2.5vl_7b", ph, root)
    j_lg_te, vj_lg_te, _ = load_judge_scores(list(te), dataset, "gemma4_26b", ph, root)

    sp_tr = step_prior_scores(list(tr), list(tr), ytr, loo=True)
    sp_te = step_prior_scores(list(tr), list(te), ytr, loo=False)

    mtr = vj_tr & vs_tr
    mte = vj_te & vs_te
    Xaf_tr = np.hstack([logit(j_tr[mtr])[:, None], logit(sp_tr[mtr])[:, None], Xs_tr[mtr]])
    Xaf_te = np.hstack([logit(j_te[mte])[:, None], logit(sp_te[mte])[:, None], Xs_te[mte]])

    p_af_valid, _, _ = fit_and_score(Xaf_tr, ytr[mtr], Xaf_te)
    paf_te = np.full(n_te, np.nan, dtype=np.float64)
    paf_te[mte] = p_af_valid
    res_af = auroc_ci(yte, paf_te, gte)

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
        "mtr": mtr,
        "mte": mte,
        "Xaf_tr": Xaf_tr,
        "Xaf_te": Xaf_te,
        "sp_tr": sp_tr,
        "sp_te": sp_te,
        "probs_act_te": probs_act_te,
        "paf_te": paf_te,
        "res_af": res_af,
        "candidates": candidates,
    }


def _fused_delta(
    action_bundle: Dict[str, Any],
    Xtr: np.ndarray,
    Xte: np.ndarray,
) -> Tuple[MetricResult, MetricResult, MetricResult, np.ndarray]:
    yte = action_bundle["yte"]
    gte = action_bundle["gte"]
    mtr = action_bundle["mtr"]
    mte = action_bundle["mte"]
    paf = action_bundle["paf_te"]
    n_te = len(yte)

    p_fu_valid, _, _ = fit_and_score(
        np.hstack([action_bundle["Xaf_tr"], Xtr[mtr]]),
        action_bundle["ytr"][mtr],
        np.hstack([action_bundle["Xaf_te"], Xte[mte]]),
    )
    pfu = np.full(n_te, np.nan, dtype=np.float64)
    pfu[mte] = p_fu_valid
    return (
        action_bundle["res_af"],
        auroc_ci(yte, pfu, gte),
        delta_auroc_ci(yte, paf, pfu, gte),
        pfu,
    )


def _shuffled_probe(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    gte: Sequence[str],
) -> MetricResult:
    rng = np.random.default_rng(0)
    y_sh = ytr.copy()
    rng.shuffle(y_sh)
    probs_sh, _, _ = fit_and_score(Xtr, y_sh, Xte)
    return auroc_ci(yte, probs_sh, list(gte))


def _load_speech_df(pdir: Path, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    k = str(pdir.resolve())
    if k not in cache:
        d = pd.read_csv(pdir / "speech.csv")
        d.columns = [c.strip() for c in d.columns]
        d["_id"] = d["Id"].astype(str).str.strip().str.lower()
        cache[k] = d
    return cache[k]


def _speech_item_feats(
    it: Item,
    ids: Set[str],
    window_key: str,
    cache: Dict[str, pd.DataFrame],
) -> Optional[np.ndarray]:
    d = _load_speech_df(Path(it.video_path).parent, cache)
    ws, we = it.meta[window_key]
    sel = d[
        d["_id"].isin(ids)
        & (d["BeginTime"].astype(float) <= float(we))
        & (d["EndTime"].astype(float) >= float(ws))
    ]
    pros = sel.iloc[:, 6:54].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pros = pros[np.all(np.isfinite(pros), axis=1)] if len(pros) else pros
    if len(pros) == 0:
        return None
    return np.concatenate([pros.mean(axis=0), pros.max(axis=0)])


def _run_speech_probe(
    tr: Sequence[Item],
    te: Sequence[Item],
    ids: Set[str],
    window_key: str,
    cache: Dict[str, pd.DataFrame],
) -> Tuple[Optional[MetricResult], int, int, int]:
    Xtr_list: List[np.ndarray] = []
    ytr_list: List[int] = []
    Xte_list: List[np.ndarray] = []
    yte_list: List[int] = []
    gte_list: List[str] = []

    for it in tr:
        f = _speech_item_feats(it, ids, window_key, cache)
        if f is not None:
            Xtr_list.append(f)
            ytr_list.append(int(it.label))
    for it in te:
        f = _speech_item_feats(it, ids, window_key, cache)
        if f is not None:
            Xte_list.append(f)
            yte_list.append(int(it.label))
            gte_list.append(it.group_id)

    ytr_arr = np.array(ytr_list, dtype=int)
    yte_arr = np.array(yte_list, dtype=int)
    n_neg_te = int(np.sum(yte_arr == 0)) if len(yte_arr) else 0
    if len(set(ytr_list)) < 2 or len(set(yte_list)) < 2:
        return None, len(yte_list), n_neg_te, len(ytr_list)

    Xtr_arr = np.array(Xtr_list, dtype=np.float64)
    Xte_arr = np.array(Xte_list, dtype=np.float64)
    probs, _, _ = fit_and_score(Xtr_arr, ytr_arr, Xte_arr)
    res = auroc_ci(yte_arr, probs, gte_list)
    return res, len(yte_list), n_neg_te, len(ytr_list)


def compute_reflex_v2_diagnostics(
    data_root: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute all REFLEX v2 exploratory diagnostics and reviewer falsification checks."""
    root = Path(data_root or DATA_ROOT)
    splits_path = Path(splits_dir or "splits")
    enc = ReflexFaceV2Encoder()

    out: Dict[str, Any] = {
        "_status": STATUS_TEXT,
        "meta.counts": {
            "n_test_evals_approx": 80,
            "n_features_all": 47,
            "n_features_gaze": 9,
            "n_features_face": 38,
            "n_features_hume": 34,
            "n_features_facetorch": 4,
            "n_hume_channels": 17,
            "n_facetorch_channels": 2,
            "n_resplits": 40,
            "n_train_participants": 38,
            "n_test_participants": 17,
            "n_total_participants": 55,
            "last10_frames": 10,
            "last20_frames": 20,
        },
    }

    # =========================================================================
    # 1. REFLEX (pre vs fail)
    # =========================================================================
    reflex_items, tr_r, te_r = _load_dataset_items("reflex", root, splits_path)
    ytr = np.array([it.label for it in tr_r], dtype=int)
    yte = np.array([it.label for it in te_r], dtype=int)
    gte = [it.group_id for it in te_r]
    n_te = len(te_r)

    # E. Frame statistics & gap check
    def _rlen(it: Item) -> int:
        a, b = it.meta["reaction_frames"]
        return int(b) - int(a) + 1

    def _alen(it: Item) -> int:
        a, b = it.meta["action_frames"]
        return int(b) - int(a) + 1

    ep_by_label: Dict[str, Dict[int, Item]] = {}
    for it in reflex_items:
        ep_by_label.setdefault(it.item_id.rsplit(":", 1)[0], {})[int(it.label)] = it

    for pool_name, pool in (("all", reflex_items), ("test", te_r)):
        r_pre = [_rlen(it) for it in pool if it.label == 1]
        r_fail = [_rlen(it) for it in pool if it.label == 0]
        a_pre = [_alen(it) for it in pool if it.label == 1]
        a_fail = [_alen(it) for it in pool if it.label == 0]
        out[f"reflex.frames_{pool_name}_reaction_pre"] = {"value": float(np.mean(r_pre))}
        out[f"reflex.frames_{pool_name}_reaction_fail"] = {"value": float(np.mean(r_fail))}
        out[f"reflex.frames_{pool_name}_action_pre"] = {"value": float(np.mean(a_pre))}
        out[f"reflex.frames_{pool_name}_action_fail"] = {"value": float(np.mean(a_fail))}

    gaps = [
        int(e[0].meta["action_frames"][0]) - int(e[1].meta["action_frames"][1])
        for e in ep_by_label.values()
        if 0 in e and 1 in e
    ]
    out["reflex.failure_start_minus_pre_end_gap"] = {
        "value": int(np.median(gaps)),
        "min": int(min(gaps)),
        "max": int(max(gaps)),
    }

    # Build the 4 v2 windows + placebo + onset_straddle windows
    orig_w, start_w, end_w, act_end_w = build_window_dicts(reflex_items)
    placebo_w: Dict[str, Tuple[int, int]] = {}
    straddle_w: Dict[str, Tuple[int, int]] = {}
    for e in ep_by_label.values():
        if 0 not in e or 1 not in e:
            continue
        pre, fail = e[1], e[0]
        ps, pe = map(int, pre.meta["action_frames"])
        fs, fe = map(int, fail.meta["action_frames"])
        Kp = max(2, (pe - ps + 1) // 2)
        placebo_w[pre.item_id] = (ps, ps + Kp - 1)
        placebo_w[fail.item_id] = (pe - Kp + 1, pe)
        Ks = max(2, min(pe - ps + 1, fe - fs + 1) // 2)
        straddle_w[pre.item_id] = (pe - Ks + 1, pe)
        straddle_w[fail.item_id] = (fs, fs + Ks - 1)

    A_reflex = _build_action_side("reflex", tr_r, te_r, root)
    out["reflex.action_probe"] = _ci_entry(
        auroc_ci(yte, A_reflex["probs_act_te"], gte), n_te
    )
    out["reflex.action_fusion"] = _ci_entry(
        auroc_ci(yte, A_reflex["paf_te"], gte), n_te
    )

    # Original window length L_orig for length_plus_v2_minus_length
    L_orig_tr = np.array(
        [float(orig_w[it.item_id][1] - orig_w[it.item_id][0] + 1) for it in tr_r],
        dtype=np.float64,
    )
    L_orig_te = np.array(
        [float(orig_w[it.item_id][1] - orig_w[it.item_id][0] + 1) for it in te_r],
        dtype=np.float64,
    )
    probs_lr_len, _, _ = fit_and_score(L_orig_tr[:, None], ytr, L_orig_te[:, None])

    mode_windows = [
        ("equal_start", start_w),
        ("action_equal_end", act_end_w),
        ("equal_end", end_w),
        ("orig", orig_w),
    ]
    for mode_name, w_dict in mode_windows:
        Xtr = _encode_items(enc, tr_r, w_dict)
        Xte = _encode_items(enc, te_r, w_dict)
        L_te = np.array([w_dict[it.item_id][1] - w_dict[it.item_id][0] + 1 for it in te_r], float)
        out[f"reflex.window_length_auroc_{mode_name}"] = {
            "value": float(roc_auc_score(yte, -L_te)),
            "n_test": n_te,
        }

        probs_te, _, _ = fit_and_score(Xtr, ytr, Xte)
        res_v2 = auroc_ci(yte, probs_te, gte)
        out[f"reflex.v2_{mode_name}"] = _ci_entry(res_v2, n_te)
        out[f"reflex.v2_shuffled_{mode_name}"] = _ci_entry(
            _shuffled_probe(Xtr, ytr, Xte, yte, gte), n_te
        )

        probs_len_plus_v2, _, _ = fit_and_score(
            np.hstack([L_orig_tr[:, None], Xtr]),
            ytr,
            np.hstack([L_orig_te[:, None], Xte]),
        )
        delta_len = delta_auroc_ci(yte, probs_lr_len, probs_len_plus_v2, gte)
        out[f"reflex.length_plus_v2_minus_length_{mode_name}"] = _ci_entry(delta_len, n_te)

        _, res_fus, delta_af, pfu = _fused_delta(A_reflex, Xtr, Xte)
        best_name, delta_ab = compute_fusion_minus_action_best(
            y_test=yte,
            fusion_scores=pfu,
            groups_test=gte,
            candidate_scores=A_reflex["candidates"],
        )
        out[f"reflex.v2_fusion_{mode_name}"] = _ci_entry(res_fus, n_te)
        out[f"reflex.v2_fusion_minus_action_fusion_{mode_name}"] = _ci_entry(delta_af, n_te)
        ab_entry = _ci_entry(delta_ab, n_te)
        ab_entry["action_best"] = best_name
        out[f"reflex.v2_fusion_minus_action_best_{mode_name}"] = ab_entry

        if mode_name in ("equal_start", "action_equal_end"):
            for sub_name, sub_idx in (
                ("gaze_only", GAZE_IDX),
                ("face_only", FACE_IDX),
                ("hume_only", HUME_IDX),
                ("facetorch_only", FT_IDX),
            ):
                p_sub, _, _ = fit_and_score(Xtr[:, sub_idx], ytr, Xte[:, sub_idx])
                out[f"reflex.v2_{sub_name}_{mode_name}"] = _ci_entry(
                    auroc_ci(yte, p_sub, gte), n_te
                )
                out[f"reflex.v2_{sub_name}_shuffled_{mode_name}"] = _ci_entry(
                    _shuffled_probe(Xtr[:, sub_idx], ytr, Xte[:, sub_idx], yte, gte), n_te
                )
                if sub_name in ("gaze_only", "face_only"):
                    _, res_sub_fus, d_sub_af, _ = _fused_delta(
                        A_reflex, Xtr[:, sub_idx], Xte[:, sub_idx]
                    )
                    out[f"reflex.v2_{sub_name}_fusion_{mode_name}"] = _ci_entry(
                        res_sub_fus, n_te
                    )
                    d_entry = _ci_entry(d_sub_af, n_te)
                    out[f"reflex.v2_{sub_name}_fusion_minus_action_fusion_{mode_name}"] = d_entry
                    out[f"reflex.v2_fusion_minus_action_fusion_{sub_name}_{mode_name}"] = dict(
                        d_entry
                    )

    # Placebo (early vs late Pre) and onset_straddle
    for check_prefix, w_dict in (
        ("placebo_early_vs_late_pre", placebo_w),
        ("onset_straddle", straddle_w),
    ):
        Xtr_c = _encode_items(enc, tr_r, w_dict)
        Xte_c = _encode_items(enc, te_r, w_dict)
        L_c = np.array([w_dict[it.item_id][1] - w_dict[it.item_id][0] + 1 for it in te_r], float)
        out[f"reflex.{check_prefix}_len_only"] = {
            "value": float(roc_auc_score(yte, -L_c)),
            "n_test": n_te,
        }
        p_all, _, _ = fit_and_score(Xtr_c, ytr, Xte_c)
        res_all = _ci_entry(auroc_ci(yte, p_all, gte), n_te)
        if check_prefix == "placebo_early_vs_late_pre":
            out["reflex.placebo_early_vs_late_pre"] = res_all
        else:
            out["reflex.onset_straddle_all"] = res_all

        for sub_suffix, sub_idx in (
            ("gaze", GAZE_IDX),
            ("face", FACE_IDX),
            ("hume", HUME_IDX),
            ("facetorch", FT_IDX),
        ):
            p_sub, _, _ = fit_and_score(Xtr_c[:, sub_idx], ytr, Xte_c[:, sub_idx])
            out[f"reflex.{check_prefix}_{sub_suffix}"] = _ci_entry(
                auroc_ci(yte, p_sub, gte), n_te
            )

    # A2: equal_start probe applied directly to early_vs_late_pre placebo windows
    Xtr_eq = _encode_items(enc, tr_r, start_w)
    Xte_eq = _encode_items(enc, te_r, start_w)
    _, sc_eq, clf_eq = fit_and_score(Xtr_eq, ytr, Xte_eq)
    Xte_pl = sc_eq.transform(_encode_items(enc, te_r, placebo_w))
    p_eq_on_pl = _safe_predict_proba(clf_eq, Xte_pl)[:, 1]
    out["reflex.equal_start_probe_on_placebo"] = _ci_entry(
        auroc_ci(yte, p_eq_on_pl, gte), n_te
    )

    # D: gaze label inventory inside action_equal_end test windows
    gaze_inv: Dict[int, collections.Counter] = {
        1: collections.Counter(),
        0: collections.Counter(),
    }
    for it in te_r:
        g_df = pd.read_csv(Path(it.video_path).parent / "gaze.csv")
        a, b = act_end_w[it.item_id]
        sub = g_df[(g_df["Frame"] >= a) & (g_df["Frame"] <= b)]["Gaze"].astype(str).str.strip()
        gaze_inv[int(it.label)].update(sub.tolist())

    for lab_int, lab_name in ((1, "pre"), (0, "fail")):
        tot_g = sum(gaze_inv[lab_int].values())
        out[f"reflex.robot_gaze_share_{lab_name}"] = {
            "value": float(gaze_inv[lab_int]["Robot"] / tot_g)
        }
        out[f"reflex.task_gaze_share_{lab_name}"] = {
            "value": float(gaze_inv[lab_int]["Task"] / tot_g)
        }
        out[f"reflex.misc_gaze_share_{lab_name}"] = {
            "value": float(gaze_inv[lab_int]["Miscellaneous"] / tot_g)
        }

    # R: 40 random 38/17 participant splits (seeds 1..40) on action_equal_end
    gids = sorted({it.group_id for it in reflex_items})
    Xall = _encode_items(enc, reflex_items, act_end_w)
    yall = np.array([it.label for it in reflex_items], dtype=int)
    gall = np.array([it.group_id for it in reflex_items])
    resplit_aucs: List[float] = []
    for seed in range(1, 41):
        rng_s = np.random.default_rng(seed)
        perm = list(gids)
        rng_s.shuffle(perm)
        trg = set(perm[:38])
        mtr_s = np.array([g in trg for g in gall], dtype=bool)
        p_s, _, _ = fit_and_score(Xall[mtr_s], yall[mtr_s], Xall[~mtr_s])
        resplit_aucs.append(float(roc_auc_score(yall[~mtr_s], p_s)))

    resplit_mean = float(np.mean(resplit_aucs))
    resplit_p5 = float(np.percentile(resplit_aucs, 5))
    resplit_p95 = float(np.percentile(resplit_aucs, 95))
    out["reflex.resplit_action_equal_end"] = {
        "value": resplit_mean,
        "ci_low": resplit_p5,
        "ci_high": resplit_p95,
        "n_splits": 40,
    }
    out["reflex.resplit_mean"] = {"value": resplit_mean}
    out["reflex.resplit_p5"] = {"value": resplit_p5}
    out["reflex.resplit_p95"] = {"value": resplit_p95}

    # =========================================================================
    # 2. REFLEX_RESOLVE (step proxy & action window checks)
    # =========================================================================
    items_res, tr_res, te_res = _load_dataset_items("reflex_resolve", root, splits_path)
    ytr_r = np.array([it.label for it in tr_res], dtype=int)
    yte_r = np.array([it.label for it in te_res], dtype=int)
    gte_r = [it.group_id for it in te_res]
    n_te_r = len(te_res)

    act_res_w: Dict[str, Tuple[int, int]] = {
        it.item_id: (int(it.meta["action_frames"][0]), int(it.meta["action_frames"][1]))
        for it in items_res
    }
    Xtr_res = _encode_items(enc, tr_res, act_res_w)
    Xte_res = _encode_items(enc, te_res, act_res_w)

    p_v2_res, _, _ = fit_and_score(Xtr_res, ytr_r, Xte_res)
    out["reflex_resolve.v2_action_window"] = _ci_entry(
        auroc_ci(yte_r, p_v2_res, gte_r), n_te_r
    )
    out["reflex_resolve.v2_action_window_shuffled"] = _ci_entry(
        _shuffled_probe(Xtr_res, ytr_r, Xte_res, yte_r, gte_r), n_te_r
    )

    Ltr_r = np.array(
        [act_res_w[it.item_id][1] - act_res_w[it.item_id][0] + 1 for it in tr_res],
        dtype=np.float64,
    )[:, None]
    Lte_r = np.array(
        [act_res_w[it.item_id][1] - act_res_w[it.item_id][0] + 1 for it in te_res],
        dtype=np.float64,
    )[:, None]
    p_len_r, _, _ = fit_and_score(Ltr_r, ytr_r, Lte_r)
    p_lv_r, _, _ = fit_and_score(
        np.hstack([Ltr_r, Xtr_res]), ytr_r, np.hstack([Lte_r, Xte_res])
    )
    out["reflex_resolve.window_length"] = _ci_entry(
        auroc_ci(yte_r, p_len_r, gte_r), n_te_r
    )
    out["reflex_resolve.length_plus_v2"] = _ci_entry(
        auroc_ci(yte_r, p_lv_r, gte_r), n_te_r
    )
    out["reflex_resolve.length_plus_v2_minus_length"] = _ci_entry(
        delta_auroc_ci(yte_r, p_len_r, p_lv_r, gte_r), n_te_r
    )

    for wn in (10, 20):
        fixed_w = {
            it.item_id: (
                max(act_res_w[it.item_id][0], act_res_w[it.item_id][1] - wn + 1),
                act_res_w[it.item_id][1],
            )
            for it in items_res
        }
        p_wn, _, _ = fit_and_score(
            _encode_items(enc, tr_res, fixed_w),
            ytr_r,
            _encode_items(enc, te_res, fixed_w),
        )
        out[f"reflex_resolve.v2_last{wn}"] = _ci_entry(
            auroc_ci(yte_r, p_wn, gte_r), n_te_r
        )
        Lf_wn = np.array(
            [fixed_w[it.item_id][1] - fixed_w[it.item_id][0] + 1 for it in te_res],
            dtype=np.float64,
        )
        out[f"reflex_resolve.v2_last{wn}_len_only"] = {
            "value": float(roc_auc_score(yte_r, -Lf_wn)),
            "n_test": n_te_r,
        }

    A_res = _build_action_side("reflex_resolve", tr_res, te_res, root)
    out["reflex_resolve.step_prior"] = _ci_entry(
        auroc_ci(yte_r, A_res["sp_te"], gte_r), n_te_r
    )
    Xsp_tr = logit(A_res["sp_tr"])[:, None]
    Xsp_te = logit(A_res["sp_te"])[:, None]
    p_spv, _, _ = fit_and_score(
        np.hstack([Xsp_tr, Xtr_res]), ytr_r, np.hstack([Xsp_te, Xte_res])
    )
    out["reflex_resolve.step_prior_plus_v2"] = _ci_entry(
        auroc_ci(yte_r, p_spv, gte_r), n_te_r
    )
    out["reflex_resolve.step_prior_plus_v2_minus_step_prior"] = _ci_entry(
        delta_auroc_ci(yte_r, A_res["sp_te"], p_spv, gte_r), n_te_r
    )

    r_af_res, r_fu_res, d_af_res, pfu_res = _fused_delta(A_res, Xtr_res, Xte_res)
    best_name_res, d_ab_res = compute_fusion_minus_action_best(
        y_test=yte_r,
        fusion_scores=pfu_res,
        groups_test=gte_r,
        candidate_scores=A_res["candidates"],
    )
    out["reflex_resolve.v2_action_fusion"] = _ci_entry(r_af_res, n_te_r)
    out["reflex_resolve.v2_fusion"] = _ci_entry(r_fu_res, n_te_r)
    out["reflex_resolve.v2_fusion_minus_action_fusion"] = _ci_entry(d_af_res, n_te_r)
    ab_res_entry = _ci_entry(d_ab_res, n_te_r)
    ab_res_entry["action_best"] = best_name_res
    out["reflex_resolve.v2_fusion_minus_action_best"] = ab_res_entry

    # C2: step prediction & within-place checks
    vtr_r = np.array([str(it.meta.get("verb")) for it in tr_res])
    vte_r = np.array([str(it.meta.get("verb")) for it in te_res])
    m_place_te = vte_r == "place"
    m_place_tr = vtr_r == "place"

    out["reflex_resolve.v2_all_steps_within_place"] = {
        "value": float(roc_auc_score(yte_r[m_place_te], p_v2_res[m_place_te])),
        "n_test": int(np.sum(m_place_te)),
        "n_positives": int(np.sum(yte_r[m_place_te])),
    }

    p_place, _, _ = fit_and_score(
        Xtr_res[m_place_tr], ytr_r[m_place_tr], Xte_res[m_place_te]
    )
    gte_place = [g for g, keep in zip(gte_r, m_place_te) if keep]
    within_place_entry = _ci_entry(
        auroc_ci(yte_r[m_place_te], p_place, gte_place), int(np.sum(m_place_te))
    )
    within_place_entry["n_train"] = int(np.sum(m_place_tr))
    out["reflex_resolve.v2_within_place"] = within_place_entry

    sc_step = StandardScaler().fit(Xtr_res)
    clf_step = LogisticRegression(C=1.0, max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clf_step.fit(sc_step.transform(Xtr_res), vtr_r)
        P_step = clf_step.predict_proba(sc_step.transform(Xte_res))
    for k, cls_name in enumerate(clf_step.classes_):
        step_auc = float(roc_auc_score((vte_r == cls_name).astype(int), P_step[:, k]))
        out[f"reflex_resolve.v2_step_auroc_{cls_name}"] = {
            "value": step_auc,
            "n_test": n_te_r,
        }

    by_verb: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
    for it in te_res:
        vb = str(it.meta.get("verb"))
        by_verb[vb][0] += int(it.label)
        by_verb[vb][1] += 1
    out["reflex_resolve.test_step_counts"] = {
        "value": n_te_r,
        "n_test": n_te_r,
        "n_positives": int(np.sum(yte_r)),
        "pick_resolved": by_verb["pick"][0],
        "pick_total": by_verb["pick"][1],
        "carry_resolved": by_verb["carry"][0],
        "carry_total": by_verb["carry"][1],
        "place_resolved": by_verb["place"][0],
        "place_total": by_verb["place"][1],
        "place_train_total": int(np.sum(m_place_tr)),
    }

    # =========================================================================
    # 3. SPEECH CHECKS (verify_speech.py S1-S3, S5 and verify_v2.py [F])
    # =========================================================================
    speech_cache: Dict[str, pd.DataFrame] = {}
    both_ids = {"participant", "robot/participant"}
    only_ids = {"participant"}

    res_s1, n_s1_te, n_s1_neg, n_s1_tr = _run_speech_probe(
        tr_res, te_res, both_ids, "wall_reaction_window_sec", speech_cache
    )
    assert res_s1 is not None
    s1_entry = _ci_entry(res_s1, n_s1_te)
    s1_entry["n_negatives_test"] = n_s1_neg
    s1_entry["n_train"] = n_s1_tr
    out["reflex_resolve.nonverbal_v1_rule"] = s1_entry

    res_s2, n_s2_te, n_s2_neg, n_s2_tr = _run_speech_probe(
        tr_res, te_res, only_ids, "wall_reaction_window_sec", speech_cache
    )
    assert res_s2 is not None
    s2_entry = _ci_entry(res_s2, n_s2_te)
    s2_entry["n_negatives_test"] = n_s2_neg
    s2_entry["n_train"] = n_s2_tr
    out["reflex_resolve.nonverbal_participant_only"] = s2_entry

    # S3: only two unresolved test items -> record as not estimable, never as its AUROC
    _, n_s3_te, n_s3_neg, n_s3_tr = _run_speech_probe(
        tr_res, te_res, only_ids, "wall_action_window_sec", speech_cache
    )
    out["reflex_resolve.nonverbal_participant_only_before_resolution"] = {
        "value": None,
        "ci_low": None,
        "ci_high": None,
        "n_test": n_s3_te,
        "n_negatives_test": n_s3_neg,
        "n_train": n_s3_tr,
        "note": "not estimable: two unresolved test items",
    }

    # [F]: reflex_resolve items with any v1 participant speech row vs mixed robot/participant row
    n_with_speech = 0
    n_with_mixed = 0
    for it in items_res:
        ws, we = it.meta["wall_reaction_window_sec"]
        s_df = _load_speech_df(Path(it.video_path).parent, speech_cache)
        sel = s_df[
            s_df["_id"].isin(both_ids)
            & (s_df["BeginTime"].astype(float) <= float(we))
            & (s_df["EndTime"].astype(float) >= float(ws))
        ]
        if len(sel) > 0:
            n_with_speech += 1
            if bool((sel["_id"] == "robot/participant").any()):
                n_with_mixed += 1

    out["reflex_resolve.speech_mixed_share"] = {
        "value": float(n_with_mixed / n_with_speech),
        "n_mixed": int(n_with_mixed),
        "n": int(n_with_speech),
    }

    # [S5]: row-level inventory over all 55 participants
    by_participant = {it.group_id: it for it in items_res}
    all_speech_rows = pd.concat(
        [
            _load_speech_df(Path(it.video_path).parent, speech_cache)
            for it in by_participant.values()
        ],
        ignore_index=True,
    )
    n_part_rows = int((all_speech_rows["_id"] == "participant").sum())
    n_rob_part_rows = int((all_speech_rows["_id"] == "robot/participant").sum())
    n_v1_rule_rows = n_part_rows + n_rob_part_rows
    part_df = all_speech_rows[all_speech_rows["_id"] == "participant"]
    n_robotish_part = int(
        part_df["Text"].astype(str).str.contains(ROBOTISH_RE).sum()
    )

    out["reflex.speech_rows_v1_rule"] = {"value": n_v1_rule_rows}
    out["reflex.speech_rows_robot_participant"] = {"value": n_rob_part_rows}
    out["reflex.speech_rows_participant"] = {"value": n_part_rows}
    out["reflex.speech_rows_participant_robot_like"] = {"value": n_robotish_part}

    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute REFLEX v2 exploratory post-hoc diagnostics and falsification checks."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evals/2026-11-08_reflex_v2_posthoc.json"),
        help="Path to output JSON file.",
    )
    args = parser.parse_args(argv)

    results = compute_reflex_v2_diagnostics()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
