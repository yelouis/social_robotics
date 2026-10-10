from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from features.cache import FeatureCache
from harness.items import Item, write_items
from harness.probes import run_probes
from harness.splits import make_group_split
from judge.vlm_judge import DATASET_QUESTIONS, prompt_hash


def test_run_probes_synthetic():
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        scorecard_file = tmp_dir / "scorecard.jsonl"
        splits_dir = tmp_dir / "splits"
        splits_dir.mkdir()
        items_dir = tmp_dir / "items" / "oops"
        items_dir.mkdir(parents=True)

        # Create 100 synthetic items (50 train, 50 test across 10 groups)
        items = []
        official_map = {}
        rng = np.random.default_rng(0)

        for i in range(100):
            split = "train" if i < 50 else "test"
            group = f"group_{i // 10}"
            item_id = f"oops:clip_{i}:pre" if i % 2 == 0 else f"oops:clip_{i}:post"
            label = 1 if i % 2 == 0 else 0
            it = Item(
                item_id=item_id,
                dataset="oops",
                group_id=group,
                video_path=str(tmp_dir / f"clip_{i}.mp4"),
                action_window_sec=[1.0, 4.0],
                reaction_window_sec=[1.0, 4.0],
                label=label,
                label_source="synthetic",
                context_text="A short clip from a home video.",
                official_split=split,
            )
            items.append(it)
            official_map[item_id] = split

        write_items(items, items_dir / "items.jsonl")
        make_group_split(items, "oops", official=official_map, path_dir=splits_dir)

        # Create feature caches
        action_cache = FeatureCache(dataset="oops", encoder_id="siglip-b16-224", data_root=tmp_dir)
        react_cache = FeatureCache(dataset="oops", encoder_id="e2v-plus-large", data_root=tmp_dir)

        for it in items:
            # Action feature: correlated with label + noise, normalized
            act_vec = rng.standard_normal(16).astype(np.float32)
            if it.label == 1:
                act_vec[:4] += 1.0
            act_vec = act_vec / float(np.linalg.norm(act_vec))
            action_cache.save(it.item_id, act_vec, it.action_window_sec, elapsed_ms=10)

            # React feature: noise, normalized
            re_vec = rng.standard_normal(16).astype(np.float32)
            re_vec = re_vec / float(np.linalg.norm(re_vec))
            react_cache.save(it.item_id, re_vec, it.reaction_window_sec, elapsed_ms=10)

        # Create judge cache
        p_hash = prompt_hash(DATASET_QUESTIONS["oops"])
        judge_dir = tmp_dir / "judge" / "oops" / "qwen2.5vl_7b"
        judge_dir.mkdir(parents=True)
        judge_file = judge_dir / f"{p_hash}.jsonl"

        with open(judge_file, "w", encoding="utf-8") as f:
            for it in items:
                prob = 0.8 if it.label == 1 else 0.2
                prob += float(rng.uniform(-0.1, 0.1))
                entry = {
                    "item_id": it.item_id,
                    "dataset": "oops",
                    "judge_prob": float(np.clip(prob, 0.05, 0.95)),
                }
                f.write(json.dumps(entry) + "\n")

        # Run probes
        rows = run_probes(
            dataset="oops",
            split_name="test",
            data_root=tmp_dir,
            splits_dir=splits_dir,
            scorecard_path=scorecard_file,
        )

        assert len(rows) >= 8
        conditions = [r.condition for r in rows]
        assert "judge" in conditions
        assert "action-probe" in conditions
        assert "react-nonverbal" in conditions
        assert "react-spoke" in conditions
        assert "react-full" in conditions
        assert "fusion" in conditions
        assert "fusion_minus_action_best" in conditions
        assert "action-probe:shuffled" in conditions
        assert "react-nonverbal:shuffled" in conditions
        assert "fusion:shuffled" in conditions

        # Check scorecard file written
        assert scorecard_file.exists()
        lines = scorecard_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(rows)


def test_run_probes_holoassist_synthetic():
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        scorecard_file = tmp_dir / "scorecard.jsonl"
        splits_dir = tmp_dir / "splits"
        splits_dir.mkdir()
        items_dir = tmp_dir / "items" / "holoassist"
        items_dir.mkdir(parents=True)

        items = []
        official_map = {}
        rng = np.random.default_rng(0)

        for i in range(100):
            split = "train" if i < 50 else "test"
            group = f"group_{i // 10}"
            item_id = f"holoassist:sess_{i}:{i}"
            label = 1 if i % 2 == 0 else 0
            spoke = 0 if label == 1 else 1
            transcript = "good job continue" if label == 1 else "no that is wrong stop"
            it = Item(
                item_id=item_id,
                dataset="holoassist",
                group_id=group,
                video_path=str(tmp_dir / f"sess_{i}.mp4"),
                action_window_sec=[1.0, 4.0],
                reaction_window_sec=[1.0, 6.0],
                label=label,
                label_source="synthetic",
                context_text="Task: setup gopro. Step: grab lever.",
                official_split=split,
                meta={
                    "spoke": spoke,
                    "transcript": transcript,
                },
            )
            items.append(it)
            official_map[item_id] = split

        write_items(items, items_dir / "items.jsonl")
        make_group_split(items, "holoassist", official=official_map, path_dir=splits_dir)

        action_cache = FeatureCache(dataset="holoassist", encoder_id="siglip-b16-224", data_root=tmp_dir)
        react_cache = FeatureCache(dataset="holoassist", encoder_id="e2v-plus-large", data_root=tmp_dir)

        for it in items:
            act_vec = rng.standard_normal(16).astype(np.float32)
            if it.label == 1:
                act_vec[:4] += 1.0
            act_vec = act_vec / float(np.linalg.norm(act_vec))
            action_cache.save(it.item_id, act_vec, it.action_window_sec, elapsed_ms=10)

            re_vec = rng.standard_normal(16).astype(np.float32)
            react_cache.save(it.item_id, re_vec, it.reaction_window_sec, elapsed_ms=10)

        p_hash = prompt_hash(DATASET_QUESTIONS["holoassist"])
        judge_dir = tmp_dir / "judge" / "holoassist" / "qwen2.5vl_7b"
        judge_dir.mkdir(parents=True)
        judge_file = judge_dir / f"{p_hash}.jsonl"

        with open(judge_file, "w", encoding="utf-8") as f:
            for it in items:
                prob = 0.8 if it.label == 1 else 0.2
                prob += float(rng.uniform(-0.1, 0.1))
                entry = {
                    "item_id": it.item_id,
                    "dataset": "holoassist",
                    "judge_prob": float(np.clip(prob, 0.05, 0.95)),
                }
                f.write(json.dumps(entry) + "\n")

        rows = run_probes(
            dataset="holoassist",
            split_name="test",
            data_root=tmp_dir,
            splits_dir=splits_dir,
            scorecard_path=scorecard_file,
        )

        row_map = {r.condition: r for r in rows}
        assert row_map["react-spoke"].value is not None
        assert row_map["react-spoke"].value > 0.9  # strongly predictive
        assert row_map["react-full"].value is not None
        assert row_map["react-full"].value > 0.9   # strongly predictive

