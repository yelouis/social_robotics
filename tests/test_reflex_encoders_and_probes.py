from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from features.cache import FeatureCache
from features.face import (
    OPENFACE_COLUMNS,
    ReflexFaceEncoder,
    ReflexSpeechEncoder,
    get_reflex_face_feature_names,
)
from harness.items import Item, write_items
from harness.probes import run_probes
from harness.splits import make_group_split
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash


def create_synthetic_participant_dir(p_dir: Path, valid: bool = True) -> Path:
    p_dir.mkdir(parents=True, exist_ok=True)

    # 1. openface.csv (1-indexed frame, whitespace in headers)
    of_path = p_dir / "openface.csv"
    with open(of_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = [" frame", " success", " confidence"] + [f" {c}" for c in OPENFACE_COLUMNS]
        writer.writerow(headers)
        for frame_idx in range(1, 51):
            if valid:
                success = 1
                conf = 0.85
            else:
                success = 0
                conf = 0.50
            # 8 gaze, 3 pose, 17 AU_r, 18 AU_c
            vals = [float(frame_idx * 0.01)] * len(OPENFACE_COLUMNS)
            writer.writerow([f" {frame_idx}", f" {success}", f" {conf}"] + [f" {v}" for v in vals])

    # 2. facetorch.csv (0-indexed Frame ID)
    ft_path = p_dir / "facetorch.csv"
    with open(ft_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame ID", "Valence", "Arousal"])
        for f_0idx in range(0, 50):
            if valid:
                v_val = 0.1 * f_0idx
                a_val = -0.05 * f_0idx
            else:
                v_val = "NA"
                a_val = "NA"
            writer.writerow([f_0idx, v_val, a_val])

    # 3. body.csv (0-indexed Frame)
    body_path = p_dir / "body.csv"
    with open(body_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Crossed Arms", "Arms behind back"])
        for f_0idx in range(0, 50):
            if valid:
                c_arms = "TRUE" if f_0idx % 2 == 0 else "FALSE"
                b_arms = "FALSE"
            else:
                c_arms = "UNKNOWN"
                b_arms = "UNKNOWN"
            writer.writerow([f_0idx, c_arms, b_arms])

    # 4. speech.csv (wall-clock seconds, 6 leading metadata columns, then 48 Hume prosody columns)
    speech_path = p_dir / "speech.csv"
    with open(speech_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = ["Participant", "Strategy", "Round", "Id", "BeginTime", "EndTime"] + [f"prosody_{i}" for i in range(48)]
        writer.writerow(headers)
        # Robot only row (should be ignored)
        writer.writerow(["P1", "C1", "1", "robot", "1.0", "3.0"] + ["0.1"] * 48)
        # Participant row in [2.0, 5.0]
        if valid:
            writer.writerow(["P1", "C1", "1", "participant", "2.5", "4.5"] + ["0.25"] * 48)
            writer.writerow(["P1", "C1", "1", "robot/participant", "3.0", "4.0"] + ["0.35"] * 48)

    return p_dir


def test_reflex_face_encoder_synthetic():
    with tempfile.TemporaryDirectory() as td:
        p_dir = Path(td) / "C1-1"
        create_synthetic_participant_dir(p_dir, valid=True)

        encoder = ReflexFaceEncoder()
        # Test reaction frames [10, 30] (1-indexed)
        feat = encoder.encode(p_dir, rsf=10, ref=30)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (153,)
        assert feat.dtype == np.float32
        assert np.all(np.isfinite(feat))

        # Test item-based encoding
        item = Item(
            item_id="reflex:C1-1:0:fail",
            dataset="reflex",
            group_id="C1-1",
            video_path=str(p_dir / "video_cam1.mp4"),
            action_window_sec=[2.0, 4.0],
            reaction_window_sec=[2.0, 6.0],
            label=0,
            label_source="synthetic",
            context_text="Task: human-robot collaborative object sorting. Step: pick object.",
            meta={"reaction_frames": [10, 30]},
        )
        feat_item = encoder.encode_item(item)
        assert np.allclose(feat, feat_item)


def test_reflex_face_encoder_honest_null():
    with tempfile.TemporaryDirectory() as td:
        p_dir = Path(td) / "C1-2"
        create_synthetic_participant_dir(p_dir, valid=False)

        encoder = ReflexFaceEncoder()
        # Should raise RuntimeError because confidence < 0.7 and success == 0
        with pytest.raises(RuntimeError, match="no valid OpenFace frames"):
            encoder.encode(p_dir, rsf=10, ref=30)


def test_reflex_speech_encoder_synthetic():
    with tempfile.TemporaryDirectory() as td:
        p_dir = Path(td) / "C1-1"
        create_synthetic_participant_dir(p_dir, valid=True)

        encoder = ReflexSpeechEncoder()
        # Window [2.0, 5.0] overlaps the participant utterances
        feat = encoder.encode(p_dir, wr_s=2.0, wr_e=5.0)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (96,)
        assert feat.dtype == np.float32
        assert np.all(np.isfinite(feat))

        # Window [10.0, 15.0] has no participant speech -> raises RuntimeError
        with pytest.raises(RuntimeError, match="no participant speech"):
            encoder.encode(p_dir, wr_s=10.0, wr_e=15.0)


def test_run_probes_reflex_synthetic():
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        scorecard_file = tmp_dir / "scorecard.jsonl"
        splits_dir = tmp_dir / "splits"
        splits_dir.mkdir()
        items_dir = tmp_dir / "items" / "reflex"
        items_dir.mkdir(parents=True)

        # 40 synthetic items across 4 participant groups (2 train, 2 test)
        items = []
        official_map = {}
        rng = np.random.default_rng(0)

        for i in range(40):
            group = f"P_{i // 10}"
            split = "train" if i < 20 else "test"
            label = 1 if i % 2 == 0 else 0
            kind = "pre" if label == 1 else "fail"
            item_id = f"reflex:{group}:{i % 10}:{kind}"
            spoke = 1 if i % 3 == 0 else 0
            trans = "I see what happened" if spoke else ""

            it = Item(
                item_id=item_id,
                dataset="reflex",
                group_id=group,
                video_path=str(tmp_dir / f"{group}.mp4"),
                action_window_sec=[1.0, 3.0],
                reaction_window_sec=[1.0, 5.0],
                label=label,
                label_source="synthetic",
                context_text="Task: human-robot collaborative object sorting. Step: pick object.",
                official_split=split,
                meta={
                    "verb": "pick",
                    "noun": "object",
                    "reaction_frames": [10, 30],
                    "wall_reaction_window_sec": [1.0, 5.0],
                    "spoke": spoke,
                    "transcript": trans,
                },
            )
            items.append(it)
            official_map[item_id] = split

        write_items(items, items_dir / "items.jsonl")
        make_group_split(items, "reflex", official=official_map, path_dir=splits_dir)

        # Create caches
        action_cache = FeatureCache(dataset="reflex", encoder_id="siglip-b16-224-u9", data_root=tmp_dir)
        face_cache = FeatureCache(dataset="reflex", encoder_id="reflex-face-v1", data_root=tmp_dir)
        speech_cache = FeatureCache(dataset="reflex", encoder_id="reflex-speech-v1", data_root=tmp_dir)

        for it in items:
            act_vec = rng.standard_normal(16).astype(np.float32)
            action_cache.save(it.item_id, act_vec, it.action_window_sec, 10)

            face_vec = rng.standard_normal(153).astype(np.float32)
            if it.label == 1:
                face_vec[:10] += 0.8
            face_cache.save(it.item_id, face_vec, it.reaction_window_sec, 10)

            if it.meta["spoke"] == 1:
                sp_vec = rng.standard_normal(96).astype(np.float32)
                speech_cache.save(it.item_id, sp_vec, it.reaction_window_sec, 10)

        # Judge caches
        p_hash = prompt_hash(DATASET_QUESTIONS["reflex"])
        judge_dir = tmp_dir / "judge" / "reflex" / "qwen2.5vl_7b"
        judge_dir.mkdir(parents=True)
        with open(judge_dir / f"{p_hash}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                prob = 0.7 if it.label == 1 else 0.3
                prob += float(rng.uniform(-0.05, 0.05))
                f.write(json.dumps({"item_id": it.item_id, "dataset": "reflex", "judge_prob": prob}) + "\n")

        # Judge-large cache (test split)
        judge_lg_dir = tmp_dir / "judge" / "reflex" / "gemma4_26b"
        judge_lg_dir.mkdir(parents=True)
        with open(judge_lg_dir / f"{p_hash}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                if it.official_split == "test":
                    prob = 0.8 if it.label == 1 else 0.2
                    f.write(json.dumps({"item_id": it.item_id, "dataset": "reflex", "judge_prob": prob}) + "\n")

        # Run probes on reflex
        rows = run_probes(
            dataset="reflex",
            split_name="test",
            data_root=tmp_dir,
            splits_dir=splits_dir,
            scorecard_path=scorecard_file,
        )

        conditions = [r.condition for r in rows]
        expected_conds = [
            "judge", "judge-large", "judge-frontier", "action-probe",
            "react-face", "react-nonverbal", "react-nonverbal|spoke=1",
            "react-spoke", "react-full", "step-prior", "fusion",
            "action-fusion", "fusion_minus_action_fusion", "fusion_minus_action_best",
            "action-probe:shuffled", "react-face:shuffled", "react-nonverbal:shuffled",
            "step-prior:shuffled", "action-fusion:shuffled", "fusion:shuffled",
        ]
        for ec in expected_conds:
            assert ec in conditions, f"Missing condition: {ec}"

        # Verify exported MicroDuck face probe
        face_probe_path = tmp_dir / "transfer" / "reflex_face_probe.npz"
        assert face_probe_path.exists(), "MicroDuck face probe not exported"
        data = np.load(face_probe_path)
        assert "mean" in data and "scale" in data and "coef" in data and "intercept" in data
        assert data["coef"].shape == (1, 153)

        # Verify reproduction of react-face test AUROC to 4 decimal places
        r_react_face = next(r for r in rows if r.condition == "react-face")
        test_face_feats = np.array([face_cache.load(it.item_id) for it in items if it.official_split == "test"])
        y_test_arr = np.array([it.label for it in items if it.official_split == "test"])
        groups_te = [it.group_id for it in items if it.official_split == "test"]

        mean = data["mean"]
        scale = data["scale"]
        coef = data["coef"]
        intercept = data["intercept"]

        X_te_scaled = (test_face_feats - mean) / scale
        logits = np.dot(X_te_scaled, coef.T) + intercept
        probs = 1.0 / (1.0 + np.exp(-logits.ravel()))

        from harness.metrics import auroc_ci
        direct_res = auroc_ci(y_test_arr, probs, groups_te)
        assert round(direct_res.point, 4) == round(r_react_face.value, 4)


def test_reflex_face_feature_names():
    names = get_reflex_face_feature_names()
    assert len(names) == 153
    assert names[0] == "gaze_0_x_mean"
    assert "pose_Rx_vel_std" in names
    assert "Valence_mean" in names
    assert "Crossed_Arms_mean" in names
