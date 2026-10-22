import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.report_holoassist import generate_holoassist_report, prediction_check_hidden  # noqa: E402


def test_prediction_check_hidden_real_rows():
    """(a) Real HoloAssist rows -> Clause 1 not tested, Clause 2 not tested, pass bar not testable.
    No line containing held or met also contains react-full.
    """
    sc_path = Path("results/scorecard.jsonl")
    assert sc_path.exists()
    rows = {}
    for line in sc_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("dataset") == "holoassist":
                rows[r["condition"]] = r

    lines = prediction_check_hidden(rows)
    assert any("Clause 1" in line and "not tested" in line for line in lines)
    assert any("Clause 2" in line and "not tested" in line for line in lines)
    assert any("H1 pass bar" in line and "not testable" in line for line in lines)

    # Ensure react-full never appears in held or met lines
    for line in lines:
        if "held" in line.lower() or "met" in line.lower():
            assert "react-full" not in line


def test_prediction_check_hidden_synthetic_held_met():
    """(b) react-nonverbal 0.80 [0.76, 0.84], action_best = action-probe 0.57,
    ablation Delta 0.05 [0.02, 0.08] -> held, held, met.
    """
    rows = {
        "react-nonverbal": {
            "value": 0.80, "ci_low": 0.76, "ci_high": 0.84, "notes": ""
        },
        "action-probe": {
            "value": 0.57, "ci_low": 0.54, "ci_high": 0.60, "notes": ""
        },
        "fusion_minus_action_best": {
            "value": 0.23, "ci_low": 0.18, "ci_high": 0.28, "notes": "action_best=action-probe"
        },
        "fusion_minus_action_fusion": {
            "value": 0.05, "ci_low": 0.02, "ci_high": 0.08, "notes": "ablation: reaction features removed"
        },
        "judge": {"value": 0.488, "ci_low": 0.466, "ci_high": 0.512},
        "judge-large": {"value": 0.544, "ci_low": 0.528, "ci_high": 0.562},
        "react-spoke": {"value": 0.676, "ci_low": 0.655, "ci_high": 0.695},
        "react-full": {"value": 0.739, "ci_low": 0.716, "ci_high": 0.761},
    }

    lines = prediction_check_hidden(rows)
    c1 = next(line for line in lines if "Clause 1" in line)
    c2 = next(line for line in lines if "Clause 2" in line)
    pb = next(line for line in lines if "H1 pass bar" in line)

    assert "→ held" in c1
    assert "→ held" in c2
    assert "→ met" in pb


def test_prediction_check_hidden_falsifying_react_full():
    """(c, falsifying) react-full 0.90, react-nonverbal null -> pass bar not testable."""
    rows = {
        "react-full": {
            "value": 0.90, "ci_low": 0.88, "ci_high": 0.92, "notes": "leakage condition"
        },
        "react-nonverbal": {
            "value": None, "ci_low": None, "ci_high": None, "notes": "not run: audio presence check failed"
        },
        "action-probe": {
            "value": 0.57, "ci_low": 0.54, "ci_high": 0.60, "notes": ""
        },
        "fusion_minus_action_best": {
            "value": None, "ci_low": None, "ci_high": None, "notes": "not run: fusion not run; action_best=action-probe"
        },
        "fusion_minus_action_fusion": {
            "value": None, "ci_low": None, "ci_high": None, "notes": "not run: fusion not run"
        },
    }

    lines = prediction_check_hidden(rows)
    pb = next(line for line in lines if "H1 pass bar" in line)
    assert "not testable" in pb
    assert "met" not in pb.split("→")[1]


def test_generate_holoassist_report_audible_synthetic(tmp_path):
    """Test generate_holoassist_report with dataset='holoassist_audible'."""
    # Write a mock scorecard
    sc_file = tmp_path / "scorecard.jsonl"
    mock_rows = [
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "action-probe",
            "metric": "auroc",
            "value": 0.58,
            "ci_low": 0.55,
            "ci_high": 0.61,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "step-prior",
            "metric": "auroc",
            "value": 0.72,
            "ci_low": 0.69,
            "ci_high": 0.75,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "react-nonverbal",
            "metric": "auroc",
            "value": 0.68,
            "ci_low": 0.65,
            "ci_high": 0.71,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "fusion",
            "metric": "auroc",
            "value": 0.75,
            "ci_low": 0.72,
            "ci_high": 0.78,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "action-fusion",
            "metric": "auroc",
            "value": 0.69,
            "ci_low": 0.66,
            "ci_high": 0.72,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "fusion_minus_action_fusion",
            "metric": "delta_auroc",
            "value": 0.06,
            "ci_low": 0.03,
            "ci_high": 0.09,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "ablation: reaction features removed",
        },
        {
            "hypothesis": "H1",
            "dataset": "holoassist_audible",
            "split": "test",
            "condition": "fusion_minus_action_best",
            "metric": "delta_auroc",
            "value": 0.03,
            "ci_low": 0.00,
            "ci_high": 0.06,
            "n_items": 1420,
            "n_groups": 53,
            "n_excluded": 0,
            "config_hash": "abc",
            "notes": "action_best=step-prior",
        },
    ]
    with open(sc_file, "w", encoding="utf-8") as f:
        for r in mock_rows:
            f.write(json.dumps(r) + "\n")

    out_md = tmp_path / "report.md"
    rep_path = generate_holoassist_report(
        dataset="holoassist_audible",
        date_str="2026-10-20",
        data_root=tmp_path,
        scorecard_path=sc_file,
        out_path=out_md,
    )

    assert rep_path.exists()
    content = rep_path.read_text(encoding="utf-8")
    assert "HoloAssist (Audible Sessions) H1 Evaluation Report" in content or "HoloAssist Audible" in content
    assert "The subset" in content or "The Subset" in content
    assert "Clause 1" in content
    assert "Clause 2" in content


def test_leak_check_label():
    from harness.reporting import leak_check_label

    assert leak_check_label(0.51, 0.505, 0.53) == "FAIL: above chance (leak)"
    assert leak_check_label(0.464, 0.440, 0.489) == "below chance (one permutation; not a leak)"
    assert leak_check_label(0.49, 0.46, 0.52) == "pass"
    assert leak_check_label(None, None, None) == "— (not run)"


def test_judge_nondrop_caveat_computed(tmp_path):
    """build a tmp_path DATA_ROOT holding 8 test items (4 drop, 4 not),
    a split file and a judge-large cache where the 4 non-drop items are perfectly separated;
    the caveat line contains 1.000 and on the 4 test items;
    change one non-drop probability so that AUROC = 0.75, and the line contains 0.750.
    """
    data_root = tmp_path / "data"
    splits_dir = tmp_path / "splits"
    data_root.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    items_dir = data_root / "items" / "holoassist_test_ds"
    items_dir.mkdir(parents=True)

    items = []
    # 8 items: 4 drop (2 pos, 2 neg), 4 non-drop (2 pos, 2 neg)
    for i in range(8):
        verb = "drop" if i < 4 else "place"
        label = 1 if i % 2 == 0 else 0
        item_id = f"holoassist_test_ds:video_{i}:0"
        items.append({
            "item_id": item_id,
            "dataset": "holoassist_test_ds",
            "group_id": f"group_{i}",
            "video_path": f"/dummy/video_{i}.mp4",
            "action_window_sec": [0.0, 1.0],
            "reaction_window_sec": [0.0, 1.0],
            "label": label,
            "label_source": "test",
            "context_text": "Task: test",
            "meta": {"verb": verb},
        })

    with open(items_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    import hashlib

    canonical_content = json.dumps({"test": sorted([it["item_id"] for it in items]), "train": []}, sort_keys=True)
    computed_sha = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()
    split_data = {
        "dataset": "holoassist_test_ds",
        "sha256": computed_sha,
        "train": [],
        "test": [it["item_id"] for it in items],
    }
    with open(splits_dir / "holoassist_test_ds.json", "w", encoding="utf-8") as f:
        json.dump(split_data, f)

    judge_dir = data_root / "judge" / "holoassist_test_ds" / "gemma4_26b"
    judge_dir.mkdir(parents=True)

    # 4 non-drop items (indices 4, 5, 6, 7): labels are [1, 0, 1, 0]
    # Perfect separation: pos items prob 0.9, neg items prob 0.1
    cache_file = judge_dir / "cache_123456789abc.jsonl"
    judge_rows = [
        {"item_id": items[0]["item_id"], "judge_prob": 0.5},
        {"item_id": items[1]["item_id"], "judge_prob": 0.5},
        {"item_id": items[2]["item_id"], "judge_prob": 0.5},
        {"item_id": items[3]["item_id"], "judge_prob": 0.5},
        {"item_id": items[4]["item_id"], "judge_prob": 0.9},
        {"item_id": items[5]["item_id"], "judge_prob": 0.1},
        {"item_id": items[6]["item_id"], "judge_prob": 0.9},
        {"item_id": items[7]["item_id"], "judge_prob": 0.1},
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        for jr in judge_rows:
            f.write(json.dumps(jr) + "\n")

    rows = {
        "step-prior": {"value": 0.75, "ci_low": 0.70, "ci_high": 0.80},
    }

    lines = prediction_check_hidden(rows, dataset="holoassist_test_ds", data_root=data_root, splits_dir=splits_dir)
    caveat_line = next(line for line in lines if "Caveat:" in line)
    assert "1.000" in caveat_line
    assert "on the 4 test items" in caveat_line

    # Now change one non-drop probability so AUROC = 0.75
    # item 4 (label 1): 0.9, item 6 (label 1): 0.9
    # item 5 (label 0): 0.1, item 7 (label 0): 0.5 (below 0.9, so 1 item tied or flipped)
    # Let's say item 4 (label 1) is 0.4 (lower than item 7 at 0.5, higher than item 5 at 0.1)
    # pos pairs: (item 4, item 5): 0.4 > 0.1 (win)
    #            (item 4, item 7): 0.4 < 0.5 (loss)
    #            (item 6, item 5): 0.9 > 0.1 (win)
    #            (item 6, item 7): 0.9 > 0.5 (win)
    # Total wins = 3 out of 4 pairs -> AUROC = 0.750!
    judge_rows[4]["judge_prob"] = 0.4
    judge_rows[7]["judge_prob"] = 0.5
    with open(cache_file, "w", encoding="utf-8") as f:
        for jr in judge_rows:
            f.write(json.dumps(jr) + "\n")

    lines2 = prediction_check_hidden(rows, dataset="holoassist_test_ds", data_root=data_root, splits_dir=splits_dir)
    caveat_line2 = next(line for line in lines2 if "Caveat:" in line)
    assert "0.750" in caveat_line2
    assert "on the 4 test items" in caveat_line2


