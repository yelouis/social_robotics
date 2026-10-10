from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union


@dataclass
class Item:
    item_id: str
    dataset: str
    group_id: str
    video_path: str
    action_window_sec: List[float]
    reaction_window_sec: List[float]
    label: int
    label_source: str
    context_text: str
    official_split: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def validate_items(items: List[Item], check_paths: bool = True) -> List[str]:
    errors: List[str] = []
    seen_ids = set()

    for item in items:
        # 1. Duplicate item_id
        if item.item_id in seen_ids:
            errors.append(f"duplicate item_id: {item.item_id}")
        else:
            seen_ids.add(item.item_id)

        # 2. Window validation
        for field_name in ("action_window_sec", "reaction_window_sec"):
            w = getattr(item, field_name)
            if not isinstance(w, (list, tuple)) or len(w) != 2:
                errors.append(f"bad window {field_name} {w} in {item.item_id}")
            else:
                s, e = w[0], w[1]
                if s < 0 or s >= e:
                    errors.append(f"bad window {field_name} [{s}, {e}] in {item.item_id}")

        # 3. Label validation
        if item.label not in (0, 1):
            errors.append(f"label must be 0 or 1 in {item.item_id}")

        # 4. Group id validation
        if not item.group_id or not str(item.group_id).strip():
            errors.append(f"empty group_id in {item.item_id}")

        # 5. Video path existence
        if check_paths:
            if not item.video_path or not os.path.exists(item.video_path):
                errors.append(f"missing video_path for {item.item_id}: {item.video_path}")

    return errors


def read_items(path: Union[str, Path]) -> List[Item]:
    p = Path(path)
    if not p.exists():
        return []
    items: List[Item] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(Item(**data))
    return items


def write_items(items: List[Item], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(items, key=lambda it: it.item_id)

    with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False, encoding="utf-8") as tf:
        temp_name = tf.name
        for item in sorted_items:
            tf.write(json.dumps(asdict(item)) + "\n")
        tf.flush()
        os.fsync(tf.fileno())

    os.replace(temp_name, p)
