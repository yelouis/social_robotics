#!/usr/bin/env python3
"""tools/diag_reflex_posthoc.py — Reproduce Wave B post-hoc diagnostics (Item B4b).

Read-only over DATA_ROOT and results/scorecard.jsonl.
Writes docs/evals/2026-11-04_reflex_posthoc.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features.cache import FeatureCache
from features.face import ReflexFaceEncoder, ReflexSpeechEncoder, get_reflex_face_feature_names
from harness.items import Item, read_items
from harness.metrics import auroc_ci, delta_auroc_ci
from harness.probes import _safe_predict_proba, load_features_for_items, logit, step_prior_scores
from harness.splits import load_split


def _participant_dir(it: Item, data_root: Path) -> Path:
    vp = Path(it.video_path)
    if vp.parent.exists():
        return vp.parent
    return data_root / "raw" / "reflex" / "extracted" / it.group_id


def _fit_and_score_lr(
    X_tr: np.ndarray,
    valid_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    valid_te: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler().fit(X_tr[valid_tr])
    X_tr_s = scaler.transform(X_tr[valid_tr])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clf.fit(X_tr_s, y_tr[valid_tr])
    probs = np.full(len(valid_te), np.nan, dtype=np.float64)
    if np.any(valid_te):
        X_te_s = scaler.transform(X_te[valid_te])
        probs[valid_te] = _safe_predict_proba(clf, X_te_s)[:, 1]
    return probs


def _format_ci_entry(y_te: np.ndarray, scores: np.ndarray, groups_te: List[str]) -> Dict[str, object]:
    res = auroc_ci(y_te, scores, groups_te)
    return {
        "value": float(res.point),
        "ci_low": float(res.ci_low),
        "ci_high": float(res.ci_high),
        "n_test": int(len(y_te) - res.n_excluded),
    }


def _format_delta_entry(
    y_te: np.ndarray,
    scores_base: np.ndarray,
    scores_new: np.ndarray,
    groups_te: List[str],
) -> Dict[str, object]:
    res = delta_auroc_ci(y_te, scores_base, scores_new, groups_te)
    return {
        "value": float(res.point),
        "ci_low": float(res.ci_low),
        "ci_high": float(res.ci_high),
        "n_test": int(len(y_te) - res.n_excluded),
    }


def _encode_face_windows(
    items: List[Item],
    windows: Dict[str, Tuple[int, int]],
    face_enc: ReflexFaceEncoder,
    data_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    valid: List[bool] = []
    for it in items:
        a, b = windows[it.item_id]
        p_dir = _participant_dir(it, data_root)
        try:
            vec = face_enc.encode(p_dir, a, b)
            feats.append(vec)
            valid.append(True)
        except RuntimeError:
            feats.append(np.zeros(153, dtype=np.float32))
            valid.append(False)
    return np.array(feats, dtype=np.float32), np.array(valid, dtype=bool)


def _encode_speech_windows(
    items: List[Item],
    windows: Dict[str, Tuple[float, float]],
    speech_enc: ReflexSpeechEncoder,
    data_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    valid: List[bool] = []
    for it in items:
        wr_s, wr_e = windows[it.item_id]
        p_dir = _participant_dir(it, data_root)
        try:
            vec = speech_enc.encode(p_dir, wr_s, wr_e)
            feats.append(vec)
            valid.append(True)
        except RuntimeError:
            feats.append(np.zeros(96, dtype=np.float32))
            valid.append(False)
    return np.array(feats, dtype=np.float32), np.array(valid, dtype=bool)


def compute_posthoc_diagnostics(
    data_root: Path,
    splits_dir: Path,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}

    # --- Dataset 1: reflex ---
    reflex_items = read_items(data_root / "items" / "reflex" / "items.jsonl")
    reflex_map = {it.item_id: it for it in reflex_items}
    reflex_split = load_split("reflex", path_dir=splits_dir)
    tr_r = [reflex_map[i] for i in reflex_split["train"] if i in reflex_map]
    te_r = [reflex_map[i] for i in reflex_split["test"] if i in reflex_map]
    y_tr_r = np.array([it.label for it in tr_r], dtype=int)
    y_te_r = np.array([it.label for it in te_r], dtype=int)
    groups_te_r = [it.group_id for it in te_r]

    face_cache_r = FeatureCache(dataset="reflex", encoder_id="reflex-face-v1", data_root=data_root)
    X_face_tr_r, valid_face_tr_r, _ = load_features_for_items(tr_r, face_cache_r)
    X_face_te_r, valid_face_te_r, _ = load_features_for_items(te_r, face_cache_r)

    # 1. reflex.react_face
    probs_rf_r = _fit_and_score_lr(X_face_tr_r, valid_face_tr_r, y_tr_r, X_face_te_r, valid_face_te_r)
    out["reflex.react_face"] = _format_ci_entry(y_te_r, probs_rf_r, groups_te_r)

    # 2. reflex.window_length
    L_tr_r = np.array(
        [float(it.meta["reaction_frames"][1] - it.meta["reaction_frames"][0] + 1) for it in tr_r],
        dtype=np.float64,
    )
    L_te_r = np.array(
        [float(it.meta["reaction_frames"][1] - it.meta["reaction_frames"][0] + 1) for it in te_r],
        dtype=np.float64,
    )
    tr_auc_L = float(roc_auc_score(y_tr_r[valid_face_tr_r], L_tr_r[valid_face_tr_r]))
    s_sign = 1.0 if tr_auc_L >= 0.5 else -1.0
    len_scores_te = np.full(len(te_r), np.nan, dtype=np.float64)
    len_scores_te[valid_face_te_r] = s_sign * L_te_r[valid_face_te_r]
    out["reflex.window_length"] = _format_ci_entry(y_te_r, len_scores_te, groups_te_r)

    # 3. reflex.face_means_only
    feat_names = get_reflex_face_feature_names()
    mean_idx = [i for i, name in enumerate(feat_names) if name.endswith("_mean")]
    probs_means_r = _fit_and_score_lr(
        X_face_tr_r[:, mean_idx],
        valid_face_tr_r,
        y_tr_r,
        X_face_te_r[:, mean_idx],
        valid_face_te_r,
    )
    out["reflex.face_means_only"] = _format_ci_entry(y_te_r, probs_means_r, groups_te_r)

    # 4. reflex.length_plus_face_minus_length
    probs_lr_len = _fit_and_score_lr(
        L_tr_r[:, None],
        valid_face_tr_r,
        y_tr_r,
        L_te_r[:, None],
        valid_face_te_r,
    )
    probs_lr_len_face = _fit_and_score_lr(
        np.hstack([L_tr_r[:, None], X_face_tr_r]),
        valid_face_tr_r,
        y_tr_r,
        np.hstack([L_te_r[:, None], X_face_te_r]),
        valid_face_te_r,
    )
    out["reflex.length_plus_face_minus_length"] = _format_delta_entry(
        y_te_r, probs_lr_len, probs_lr_len_face, groups_te_r
    )

    # 5 & 6. Equal-length reaction windows per episode on reflex
    episodes: Dict[str, List[Item]] = {}
    for it in reflex_items:
        ep_key = it.item_id.rsplit(":", 1)[0]
        episodes.setdefault(ep_key, []).append(it)

    start_windows: Dict[str, Tuple[int, int]] = {}
    end_windows: Dict[str, Tuple[int, int]] = {}
    for ep_key, ep_items in episodes.items():
        lengths = [int(it.meta["reaction_frames"][1]) - int(it.meta["reaction_frames"][0]) + 1 for it in ep_items]
        K = min(lengths)
        for it in ep_items:
            rsf, ref = int(it.meta["reaction_frames"][0]), int(it.meta["reaction_frames"][1])
            start_windows[it.item_id] = (rsf, rsf + K - 1)
            end_windows[it.item_id] = (ref - K + 1, ref)

    face_enc = ReflexFaceEncoder()
    X_eq_start_tr, v_eq_start_tr = _encode_face_windows(tr_r, start_windows, face_enc, data_root)
    X_eq_start_te, v_eq_start_te = _encode_face_windows(te_r, start_windows, face_enc, data_root)
    probs_eq_start = _fit_and_score_lr(X_eq_start_tr, v_eq_start_tr, y_tr_r, X_eq_start_te, v_eq_start_te)
    out["reflex.face_equal_length_start"] = _format_ci_entry(y_te_r, probs_eq_start, groups_te_r)

    X_eq_end_tr, v_eq_end_tr = _encode_face_windows(tr_r, end_windows, face_enc, data_root)
    X_eq_end_te, v_eq_end_te = _encode_face_windows(te_r, end_windows, face_enc, data_root)
    probs_eq_end = _fit_and_score_lr(X_eq_end_tr, v_eq_end_tr, y_tr_r, X_eq_end_te, v_eq_end_te)
    out["reflex.face_equal_length_end"] = _format_ci_entry(y_te_r, probs_eq_end, groups_te_r)

    # --- Dataset 2: reflex_resolve ---
    resolve_items = read_items(data_root / "items" / "reflex_resolve" / "items.jsonl")
    resolve_map = {it.item_id: it for it in resolve_items}
    resolve_split = load_split("reflex_resolve", path_dir=splits_dir)
    tr_rr = [resolve_map[i] for i in resolve_split["train"] if i in resolve_map]
    te_rr = [resolve_map[i] for i in resolve_split["test"] if i in resolve_map]
    y_tr_rr = np.array([it.label for it in tr_rr], dtype=int)
    y_te_rr = np.array([it.label for it in te_rr], dtype=int)
    groups_te_rr = [it.group_id for it in te_rr]

    face_cache_rr = FeatureCache(dataset="reflex_resolve", encoder_id="reflex-face-v1", data_root=data_root)
    X_face_tr_rr, valid_face_tr_rr, _ = load_features_for_items(tr_rr, face_cache_rr)
    X_face_te_rr, valid_face_te_rr, _ = load_features_for_items(te_rr, face_cache_rr)

    # 7. reflex_resolve.react_face
    probs_rf_rr = _fit_and_score_lr(X_face_tr_rr, valid_face_tr_rr, y_tr_rr, X_face_te_rr, valid_face_te_rr)
    out["reflex_resolve.react_face"] = _format_ci_entry(y_te_rr, probs_rf_rr, groups_te_rr)

    # 8. reflex_resolve.face_before_resolution (window = meta.action_frames)
    act_frame_windows = {
        it.item_id: (int(it.meta["action_frames"][0]), int(it.meta["action_frames"][1]))
        for it in resolve_items
    }
    X_fbr_tr, v_fbr_tr = _encode_face_windows(tr_rr, act_frame_windows, face_enc, data_root)
    X_fbr_te, v_fbr_te = _encode_face_windows(te_rr, act_frame_windows, face_enc, data_root)
    probs_fbr = _fit_and_score_lr(X_fbr_tr, v_fbr_tr, y_tr_rr, X_fbr_te, v_fbr_te)
    out["reflex_resolve.face_before_resolution"] = _format_ci_entry(y_te_rr, probs_fbr, groups_te_rr)

    # 9. reflex_resolve.nonverbal_before_resolution (speech window = meta.wall_action_window_sec)
    speech_enc = ReflexSpeechEncoder()
    wall_act_windows = {
        it.item_id: (float(it.meta["wall_action_window_sec"][0]), float(it.meta["wall_action_window_sec"][1]))
        for it in resolve_items
    }
    X_nvbr_tr, v_nvbr_tr = _encode_speech_windows(tr_rr, wall_act_windows, speech_enc, data_root)
    X_nvbr_te, v_nvbr_te = _encode_speech_windows(te_rr, wall_act_windows, speech_enc, data_root)
    probs_nvbr = _fit_and_score_lr(X_nvbr_tr, v_nvbr_tr, y_tr_rr, X_nvbr_te, v_nvbr_te)
    out["reflex_resolve.nonverbal_before_resolution"] = _format_ci_entry(y_te_rr, probs_nvbr, groups_te_rr)

    # 10. reflex_resolve.step_prior (on test items with valid face features)
    sp_tr_loo = step_prior_scores(tr_rr, tr_rr, y_tr_rr, loo=True)
    sp_te_all = step_prior_scores(tr_rr, te_rr, y_tr_rr, loo=False)
    sp_te_face = np.full(len(te_rr), np.nan, dtype=np.float64)
    sp_te_face[valid_face_te_rr] = sp_te_all[valid_face_te_rr]
    out["reflex_resolve.step_prior"] = _format_ci_entry(y_te_rr, sp_te_face, groups_te_rr)

    # 11. reflex_resolve.step_prior_plus_face_minus_step_prior
    logit_sp_tr = logit(sp_tr_loo)[:, None]
    logit_sp_te = logit(sp_te_all)[:, None]
    probs_sp_plus_face = _fit_and_score_lr(
        np.hstack([logit_sp_tr, X_face_tr_rr]),
        valid_face_tr_rr,
        y_tr_rr,
        np.hstack([logit_sp_te, X_face_te_rr]),
        valid_face_te_rr,
    )
    out["reflex_resolve.step_prior_plus_face_minus_step_prior"] = _format_delta_entry(
        y_te_rr, sp_te_face, probs_sp_plus_face, groups_te_rr
    )

    # 12. reflex_resolve.nonverbal_shuffled_null
    speech_cache_rr = FeatureCache(dataset="reflex_resolve", encoder_id="reflex-speech-v1", data_root=data_root)
    X_sp_tr_rr, valid_sp_tr_rr, _ = load_features_for_items(tr_rr, speech_cache_rr)
    X_sp_te_rr, valid_sp_te_rr, _ = load_features_for_items(te_rr, speech_cache_rr)
    sp_scaler = StandardScaler().fit(X_sp_tr_rr[valid_sp_tr_rr])
    X_sp_tr_s = sp_scaler.transform(X_sp_tr_rr[valid_sp_tr_rr])
    X_sp_te_s = sp_scaler.transform(X_sp_te_rr[valid_sp_te_rr])

    clf_sp_true = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clf_sp_true.fit(X_sp_tr_s, y_tr_rr[valid_sp_tr_rr])
    true_sp_probs = _safe_predict_proba(clf_sp_true, X_sp_te_s)[:, 1]
    true_sp_auc = float(roc_auc_score(y_te_rr[valid_sp_te_rr], true_sp_probs))

    null_vals: List[float] = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        y_sh = y_tr_rr.copy()
        rng.shuffle(y_sh)
        clf_sh = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clf_sh.fit(X_sp_tr_s, y_sh[valid_sp_tr_rr])
        sh_probs = _safe_predict_proba(clf_sh, X_sp_te_s)[:, 1]
        null_vals.append(float(roc_auc_score(y_te_rr[valid_sp_te_rr], sh_probs)))

    arr_null = np.array(null_vals, dtype=np.float64)
    out["reflex_resolve.nonverbal_shuffled_null"] = {
        "mean": float(np.mean(arr_null)),
        "sd": float(np.std(arr_null, ddof=0)),
        "min": float(np.min(arr_null)),
        "max": float(np.max(arr_null)),
        "seed0": float(arr_null[0]),
        "true_value": float(true_sp_auc),
        "true_beats": int(np.sum(true_sp_auc > arr_null)),
    }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Wave B post-hoc diagnostics (Item B4b).")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "/Volumes/Extreme SSD/social_robotics"),
        help="Data root path",
    )
    parser.add_argument("--splits-dir", default="splits", help="Splits directory")
    parser.add_argument(
        "--out",
        default="docs/evals/2026-11-04_reflex_posthoc.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    res = compute_posthoc_diagnostics(Path(args.data_root), Path(args.splits_dir))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
