from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np

from harness.items import Item


def make_group_split(
    items: List[Item],
    dataset: str,
    official: Optional[Dict[str, str]] = None,
    force: bool = False,
    path_dir: Union[str, Path] = "splits",
) -> Dict[str, Any]:
    target_path = Path(path_dir) / f"{dataset}.json"
    if target_path.exists() and not force:
        raise FileExistsError(f"split exists: splits/{dataset}.json (use force=True and say why in the commit)")

    if official is not None:
        group_splits: Dict[str, set] = {}
        for item in items:
            split_name = official.get(item.item_id)
            if split_name in ("train", "test"):
                group_splits.setdefault(item.group_id, set()).add(split_name)

        straddling = [g for g, splits in group_splits.items() if len(splits) > 1]
        if straddling:
            raise ValueError(f"official split is not group-disjoint: {len(straddling)} groups straddle")

        train = [item.item_id for item in items if official.get(item.item_id) == "train"]
        test = [item.item_id for item in items if official.get(item.item_id) == "test"]
        source = "official"
        seed: Optional[int] = None
    else:
        unique_groups = sorted(list(set(item.group_id for item in items)))
        rng = np.random.default_rng(0)
        shuffled = list(unique_groups)
        rng.shuffle(shuffled)

        n_train = int(round(0.7 * len(shuffled)))
        train_groups = set(shuffled[:n_train])
        test_groups = set(shuffled[n_train:])

        train = [item.item_id for item in items if item.group_id in train_groups]
        test = [item.item_id for item in items if item.group_id in test_groups]
        source = "grouped_70_30"
        seed = 0

    canonical_content = json.dumps({"train": sorted(train), "test": sorted(test)}, sort_keys=True)
    split_sha = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

    data = {
        "dataset": dataset,
        "seed": seed,
        "source": source,
        "train": sorted(train),
        "test": sorted(test),
        "sha256": split_sha,
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target_path.parent, delete=False, encoding="utf-8") as tf:
        temp_name = tf.name
        json.dump(data, tf, indent=2)
        tf.write("\n")
        tf.flush()
        os.fsync(tf.fileno())

    os.replace(temp_name, target_path)
    return data


def load_split(dataset: str, path_dir: Union[str, Path] = "splits") -> Dict[str, Any]:
    target_path = Path(path_dir) / f"{dataset}.json"
    if not target_path.exists():
        raise FileNotFoundError(f"Split file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    canonical_content = json.dumps({"train": sorted(data["train"]), "test": sorted(data["test"])}, sort_keys=True)
    computed_sha = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

    if computed_sha != data.get("sha256"):
        raise ValueError(f"tamper detected: sha256 mismatch for split {dataset}")

    return data
