from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from config import BASE_DIR, DATA_ROOT
from harness.items import read_items
from harness.metrics import auroc_ci
from harness.splits import load_split
from shared.fsutil import data_glob

SPLITS_DIR = BASE_DIR / "splits"


def leak_check_label(
    value: Optional[float], ci_low: Optional[float], ci_high: Optional[float]
) -> str:
    """Return the one-sided leak check label (03_eval_harness.md §11.9)."""
    if value is None or ci_low is None or ci_high is None:
        return "— (not run)"
    if ci_low > 0.5:
        return "FAIL: above chance (leak)"
    if ci_high < 0.5:
        return "below chance (one permutation; not a leak)"
    return "pass"


def judge_nondrop_auroc(
    dataset: str,
    model_tag: str = "gemma4_26b",
    data_root: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[float, float, float, int]]:
    """Compute judge AUROC on test items whose verb != 'drop'."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    s_dir = Path(splits_dir) if splits_dir is not None else SPLITS_DIR

    items_path = root / "items" / dataset / "items.jsonl"
    if not items_path.exists():
        return None

    split_file = s_dir / f"{dataset}.json"
    if not split_file.exists():
        return None

    judge_dir = root / "judge" / dataset / model_tag
    if not judge_dir.exists():
        return None

    jsonl_files = [f for f in data_glob(judge_dir, "*.jsonl") if f.name != "errors.jsonl"]
    if not jsonl_files:
        return None

    probs = {}
    for f in jsonl_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("judge_prob") is not None:
                        probs[d["item_id"]] = float(d["judge_prob"])
                except Exception:
                    pass

    if not probs:
        return None

    split = load_split(dataset, path_dir=s_dir)
    test_ids = set(split.get("test", []))
    items = read_items(items_path)

    kept = [
        it
        for it in items
        if it.item_id in test_ids and it.meta.get("verb") != "drop" and it.item_id in probs
    ]
    if not kept:
        return None

    labels = np.array([it.label for it in kept], dtype=int)
    if len(np.unique(labels)) < 2:
        return None

    scores = np.array([probs[it.item_id] for it in kept], dtype=float)
    groups = np.array([it.group_id for it in kept])

    res = auroc_ci(labels, scores, groups)
    return (float(res.point), float(res.ci_low), float(res.ci_high), len(kept))
