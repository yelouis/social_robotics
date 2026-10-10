from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import List, Optional, Union

import numpy as np

from config import DATA_ROOT


def sanitize_item_id(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", item_id)


class FeatureCache:
    def __init__(
        self,
        dataset: str,
        encoder_id: str,
        data_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.dataset = dataset
        self.encoder_id = encoder_id
        base_dir = Path(data_root or DATA_ROOT)
        self.cache_dir = base_dir / "features" / dataset / encoder_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.jsonl"

    def path(self, item_id: str) -> Path:
        sanitized = sanitize_item_id(item_id)
        return self.cache_dir / f"{sanitized}.npy"

    def has(self, item_id: str) -> bool:
        return self.path(item_id).exists()

    def save(
        self,
        item_id: str,
        array: np.ndarray,
        window: List[float],
        elapsed_ms: float,
    ) -> None:
        target_file = self.path(item_id)
        arr_f32 = np.asarray(array, dtype=np.float32)

        # Atomic save of .npy
        with tempfile.NamedTemporaryFile("wb", dir=self.cache_dir, delete=False, suffix=".npy") as tf:
            temp_name = tf.name
            np.save(tf, arr_f32)
            tf.flush()
            os.fsync(tf.fileno())

        os.replace(temp_name, target_file)

        # Append to index.jsonl
        entry = {
            "item_id": item_id,
            "shape": list(arr_f32.shape),
            "window": [float(w) for w in window],
            "elapsed_ms": float(elapsed_ms),
        }
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load(self, item_id: str) -> np.ndarray:
        p = self.path(item_id)
        if not p.exists():
            raise FileNotFoundError(f"Feature not found for item: {item_id} at {p}")
        return np.load(p)
