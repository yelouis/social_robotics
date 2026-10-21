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
            # Some correct actions also have instructor speech (like real data)
            spoke = 1 if (label == 0 or i % 4 == 0) else 0
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
            if it.label == 1:
                re_vec[:4] += 0.5
            re_vec = re_vec / float(np.linalg.norm(re_vec))
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
        assert row_map["react-spoke"].value > 0.7
        assert row_map["react-full"].value is not None
        assert row_map["react-full"].value > 0.8

        # Diagnostic row assertions:
        assert "react-nonverbal|spoke=1" in row_map
        diag_row = row_map["react-nonverbal|spoke=1"]
        assert diag_row.notes == "diagnostic: test items with spoke=1"
        assert diag_row.value is not None
        assert diag_row.n_items < len([it for it in items if official_map[it.item_id] == "test"])

        # Check action_best never picks the diagnostic row
        delta_row = row_map["fusion_minus_action_best"]
        assert "react-nonverbal|spoke=1" not in delta_row.notes


def test_probes_judge_large_and_frontier_rules():
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        scorecard_file = tmp_dir / "scorecard.jsonl"
        splits_dir = tmp_dir / "splits"
        splits_dir.mkdir()
        items_dir = tmp_dir / "items" / "oops"
        items_dir.mkdir(parents=True)

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
                context_text="A short clip.",
                official_split=split,
            )
            items.append(it)
            official_map[item_id] = split

        write_items(items, items_dir / "items.jsonl")
        make_group_split(items, "oops", official=official_map, path_dir=splits_dir)

        action_cache = FeatureCache(dataset="oops", encoder_id="siglip-b16-224", data_root=tmp_dir)
        react_cache = FeatureCache(dataset="oops", encoder_id="e2v-plus-large", data_root=tmp_dir)

        for it in items:
            act_vec = rng.standard_normal(16).astype(np.float32)
            act_vec = act_vec / float(np.linalg.norm(act_vec))
            action_cache.save(it.item_id, act_vec, it.action_window_sec, elapsed_ms=10)

            re_vec = rng.standard_normal(16).astype(np.float32)
            re_vec = re_vec / float(np.linalg.norm(re_vec))
            react_cache.save(it.item_id, re_vec, it.reaction_window_sec, elapsed_ms=10)

        p_hash = prompt_hash(DATASET_QUESTIONS["oops"])

        # Local judge cache
        j_dir = tmp_dir / "judge" / "oops" / "qwen2.5vl_7b"
        j_dir.mkdir(parents=True)
        with open(j_dir / f"{p_hash}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({"item_id": it.item_id, "judge_prob": 0.5}) + "\n")

        # 1. judge-large with k = n and best AUROC (e.g. 0.95) -> it is action_best
        large_dir = tmp_dir / "judge" / "oops" / "gemma4_26b"
        large_dir.mkdir(parents=True)
        large_file = large_dir / f"{p_hash}.jsonl"
        with open(large_file, "w", encoding="utf-8") as f:
            for it in items:
                # Strong correlation with label
                p = 0.95 if it.label == 1 else 0.05
                f.write(json.dumps({"item_id": it.item_id, "judge_prob": p}) + "\n")

        rows = run_probes(
            dataset="oops",
            split_name="test",
            data_root=tmp_dir,
            splits_dir=splits_dir,
            scorecard_path=scorecard_file,
            notes="test run",
        )
        row_map = {r.condition: r for r in rows}

        assert "judge-large" in row_map
        assert row_map["judge-large"].value is not None
        assert row_map["judge-large"].value > 0.9
        assert "action_best=judge-large" in row_map["fusion_minus_action_best"].notes
        assert "test run" in row_map["judge-large"].notes
        assert "test run" in row_map["fusion_minus_action_best"].notes

        # Frontier cache does not exist -> not run: no cache file
        assert "judge-frontier" in row_map
        assert row_map["judge-frontier"].value is None
        assert "not run: no cache file" in row_map["judge-frontier"].notes

        # 2. k = 0.5 * n -> partial row, NOT in action_best
        test_items = [it for it in items if it.official_split == "test"]
        half_items = test_items[: len(test_items) // 2]
        with open(large_file, "w", encoding="utf-8") as f:
            for it in half_items:
                p = 0.99 if it.label == 1 else 0.01
                f.write(json.dumps({"item_id": it.item_id, "judge_prob": p}) + "\n")

        rows2 = run_probes(
            dataset="oops",
            split_name="test",
            data_root=tmp_dir,
            splits_dir=splits_dir,
            scorecard_path=scorecard_file,
        )
        row_map2 = {r.condition: r for r in rows2}
        assert "partial:" in row_map2["judge-large"].notes
        assert "action_best=judge-large" not in row_map2["fusion_minus_action_best"].notes

        # 3. Single-class partial -> value: null with metric undefined
        with open(large_file, "w", encoding="utf-8") as f:
            # write only positive class items
            for it in test_items:
                if it.label == 1:
                    f.write(json.dumps({"item_id": it.item_id, "judge_prob": 0.8}) + "\n")
                    break

        rows3 = run_probes(
            dataset="oops",
            split_name="test",
            data_root=tmp_dir,
            splits_dir=splits_dir,
            scorecard_path=scorecard_file,
        )
        row_map3 = {r.condition: r for r in rows3}
        assert row_map3["judge-large"].value is None
        assert "metric undefined" in row_map3["judge-large"].notes


def test_step_prior_formula_hand_computed():
    """Verify hierarchical smoothing formula against hand-computed values per 03_eval_harness.md §6.

    Arithmetic:
    6 items:
    Item 0: verb="pick", noun="cup", label=1
    Item 1: verb="pick", noun="cup", label=0
    Item 2: verb="pick", noun="plate", label=1
    Item 3: verb="pour", noun="water", label=1
    Item 4: verb="pour", noun="water", label=1
    Item 5: verb="pour", noun="milk", label=0

    Non-LOO:
    N = 6, K = 4
    p0 = 4 / 6 = 2/3

    Verb level:
    n_v("pick") = 3, k_v("pick") = 2
    p_v("pick") = (2 + 5 * (2/3)) / (3 + 5) = (16/3) / 8 = 2/3
    n_v("pour") = 3, k_v("pour") = 2
    p_v("pour") = (2 + 5 * (2/3)) / (3 + 5) = 2/3

    Verb + noun level:
    ("pick", "cup"): n_vn = 2, k_vn = 1
    p_vn("pick", "cup") = (1 + 5 * (2/3)) / (2 + 5) = (13/3) / 7 = 13/21
    ("pick", "plate"): n_vn = 1, k_vn = 1
    p_vn("pick", "plate") = (1 + 5 * (2/3)) / (1 + 5) = (13/3) / 6 = 13/18
    ("pour", "water"): n_vn = 2, k_vn = 2
    p_vn("pour", "water") = (2 + 5 * (2/3)) / (2 + 5) = (16/3) / 7 = 16/21
    ("pour", "milk"): n_vn = 1, k_vn = 0
    p_vn("pour", "milk") = (0 + 5 * (2/3)) / (1 + 5) = (10/3) / 6 = 10/18 = 5/9

    LOO for Item 0 (pick, cup, label=1):
    N_loo = 5, K_loo = 3 -> p0_loo = 3/5 = 0.6
    n_v_loo = 2, k_v_loo = 1 -> p_v_loo = (1 + 5 * 0.6) / (2 + 5) = 4/7
    n_vn_loo = 1, k_vn_loo = 0 -> p_vn_loo = (0 + 5 * (4/7)) / (1 + 5) = 20/42 = 10/21
    Difference (non-LOO - LOO) = 13/21 - 10/21 = 3/21 = 1/7
    """
    from harness.probes import step_prior_scores

    train_items = [
        Item("t0", "holoassist", "g0", "dummy", [0, 1], [0, 1], 1, "ctx", "src", meta={"verb": "pick", "noun": "cup"}),
        Item("t1", "holoassist", "g0", "dummy", [0, 1], [0, 1], 0, "ctx", "src", meta={"verb": "pick", "noun": "cup"}),
        Item("t2", "holoassist", "g1", "dummy", [0, 1], [0, 1], 1, "ctx", "src", meta={"verb": "pick", "noun": "plate"}),
        Item("t3", "holoassist", "g1", "dummy", [0, 1], [0, 1], 1, "ctx", "src", meta={"verb": "pour", "noun": "water"}),
        Item("t4", "holoassist", "g2", "dummy", [0, 1], [0, 1], 1, "ctx", "src", meta={"verb": "pour", "noun": "water"}),
        Item("t5", "holoassist", "g2", "dummy", [0, 1], [0, 1], 0, "ctx", "src", meta={"verb": "pour", "noun": "milk"}),
    ]
    train_labels = np.array([1, 0, 1, 1, 1, 0], dtype=int)

    # non-LOO evaluation on train items
    scores_non_loo = step_prior_scores(train_items, train_items, train_labels, loo=False)
    expected_non_loo = np.array([13 / 21, 13 / 21, 13 / 18, 16 / 21, 16 / 21, 5 / 9])
    np.testing.assert_allclose(scores_non_loo, expected_non_loo, atol=1e-12)

    # LOO evaluation on train items
    scores_loo = step_prior_scores(train_items, train_items, train_labels, loo=True)
    assert abs(scores_loo[0] - (10 / 21)) < 1e-12
    assert abs((scores_non_loo[0] - scores_loo[0]) - (1 / 7)) < 1e-12


def test_shuffled_controls_see_no_true_labels(tmp_path: Path):
    """(b) Synthetic dataset where verb fully determines label.
    step-prior AUROC is 1.0, but step-prior:shuffled and fusion:shuffled have CIs containing 0.5.
    """
    items_dir = tmp_path / "items" / "holoassist"
    items_dir.mkdir(parents=True)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    items = []
    official_map = {}
    rng = np.random.default_rng(42)

    verbs_pos = [f"insert_{k}" for k in range(10)]
    verbs_neg = [f"drop_{k}" for k in range(10)]

    # 200 items: 100 train, 100 test across 20 groups
    for i in range(200):
        split = "train" if i < 100 else "test"
        group = f"group_{i // 10}"
        item_id = f"holoassist:vid_{i}:{i}"
        # Verb fully determines label
        if i % 2 == 0:
            verb = rng.choice(verbs_pos)
            noun = "peg"
            label = 1
        else:
            verb = rng.choice(verbs_neg)
            noun = "screw"
            label = 0

        it = Item(
            item_id=item_id,
            dataset="holoassist",
            group_id=group,
            video_path=str(tmp_path / f"vid_{i}.mp4"),
            action_window_sec=[0.0, 2.0],
            reaction_window_sec=[2.0, 4.0],
            label=label,
            label_source="synthetic",
            context_text=f"Task: test. Step: {verb} {noun}.",
            official_split=split,
            meta={"verb": verb, "noun": noun, "spoke": 0, "transcript": ""},
        )
        items.append(it)
        official_map[item_id] = split

    write_items(items, items_dir / "items.jsonl")
    make_group_split(items, "holoassist", official=official_map, path_dir=splits_dir)

    # Feature caches (pure noise)
    action_cache = FeatureCache(dataset="holoassist", encoder_id="siglip-b16-224", data_root=tmp_path)
    react_cache = FeatureCache(dataset="holoassist", encoder_id="e2v-plus-large", data_root=tmp_path)
    for it in items:
        vec = rng.standard_normal(8).astype(np.float32)
        vec /= float(np.linalg.norm(vec))
        action_cache.save(it.item_id, vec, it.action_window_sec, elapsed_ms=5)
        react_cache.save(it.item_id, vec, it.reaction_window_sec, elapsed_ms=5)

    # Judge cache (pure noise)
    p_hash = prompt_hash(DATASET_QUESTIONS["holoassist"])
    judge_dir = tmp_path / "judge" / "holoassist" / "qwen2.5vl_7b"
    judge_dir.mkdir(parents=True)
    with open(judge_dir / f"{p_hash}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            p = float(rng.uniform(0.1, 0.9))
            f.write(json.dumps({"item_id": it.item_id, "dataset": "holoassist", "judge_prob": p}) + "\n")

    rows = run_probes(
        dataset="holoassist",
        split_name="test",
        data_root=tmp_path,
        splits_dir=splits_dir,
        scorecard_path=tmp_path / "scorecard.jsonl",
    )
    row_map = {r.condition: r for r in rows}

    # True step-prior is 1.0 (verb determines label perfectly)
    assert row_map["step-prior"].value is not None
    assert row_map["step-prior"].value > 0.99

    # Shuffled controls must cover 0.50
    sp_sh = row_map["step-prior:shuffled"]
    assert sp_sh.ci_low <= 0.50 <= sp_sh.ci_high

    fu_sh = row_map["fusion:shuffled"]
    assert fu_sh.ci_low <= 0.50 <= fu_sh.ci_high

    afu_sh = row_map["action-fusion:shuffled"]
    assert afu_sh.ci_low <= 0.50 <= afu_sh.ci_high


def test_action_fusion_fit_on_exact_fusion_tr_mask(tmp_path: Path):
    """(c) action-fusion is fit on exactly fusion_tr_mask when fusion runs."""
    items_dir = tmp_path / "items" / "oops"
    items_dir.mkdir(parents=True)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    items = []
    official_map = {}
    rng = np.random.default_rng(0)

    for i in range(40):
        split = "train" if i < 20 else "test"
        item_id = f"oops:clip_{i}:pre" if i % 2 == 0 else f"oops:clip_{i}:post"
        label = 1 if i % 2 == 0 else 0
        it = Item(
            item_id=item_id,
            dataset="oops",
            group_id=f"g_{i // 4}",
            video_path=str(tmp_path / f"clip_{i}.mp4"),
            action_window_sec=[0.0, 2.0],
            reaction_window_sec=[2.0, 4.0],
            label=label,
            label_source="synthetic",
            context_text="A short clip from a home video.",
            official_split=split,
        )
        items.append(it)
        official_map[item_id] = split

    write_items(items, items_dir / "items.jsonl")
    make_group_split(items, "oops", official=official_map, path_dir=splits_dir)

    action_cache = FeatureCache(dataset="oops", encoder_id="siglip-b16-224", data_root=tmp_path)
    react_cache = FeatureCache(dataset="oops", encoder_id="e2v-plus-large", data_root=tmp_path)

    # Make only a subset of train items have react features so fusion_tr_mask != all items
    for i, it in enumerate(items):
        vec = rng.standard_normal(4).astype(np.float32)
        vec /= float(np.linalg.norm(vec))
        action_cache.save(it.item_id, vec, it.action_window_sec, elapsed_ms=5)
        # Exclude react feature for first 5 train items
        if i >= 5:
            react_cache.save(it.item_id, vec, it.reaction_window_sec, elapsed_ms=5)

    p_hash = prompt_hash(DATASET_QUESTIONS["oops"])
    judge_dir = tmp_path / "judge" / "oops" / "qwen2.5vl_7b"
    judge_dir.mkdir(parents=True)
    with open(judge_dir / f"{p_hash}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"item_id": it.item_id, "dataset": "oops", "judge_prob": 0.5}) + "\n")

    rows = run_probes(
        dataset="oops",
        split_name="test",
        data_root=tmp_path,
        splits_dir=splits_dir,
        scorecard_path=tmp_path / "scorecard.jsonl",
    )
    row_map = {r.condition: r for r in rows}

    # action-fusion must have the exact same n_items and n_excluded as fusion
    assert "action-fusion" in row_map
    assert "fusion" in row_map
    assert row_map["action-fusion"].n_items == row_map["fusion"].n_items
    assert row_map["action-fusion"].n_excluded == row_map["fusion"].n_excluded


def test_oops_writes_step_prior_not_run_and_audible_takes_holoassist_branch(tmp_path: Path):
    """(d) The Oops! branch writes step-prior not run row; holoassist_audible takes HoloAssist branch."""
    from harness.probes import HOLOASSIST_DATASETS
    assert "holoassist" in HOLOASSIST_DATASETS
    assert "holoassist_audible" in HOLOASSIST_DATASETS
    assert "oops" not in HOLOASSIST_DATASETS

