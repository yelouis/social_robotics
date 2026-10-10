from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import List, Optional, Union

from config import DATA_ROOT
from features.audio import NonverbalAudioEncoder
from features.cache import FeatureCache
from features.visual import FrameEncoder
from harness.items import read_items
from harness.splits import load_split


def run_extraction(
    dataset: str,
    encoder_id: str,
    split: str = "all",
    force: bool = False,
    limit: Optional[int] = None,
    data_root: Optional[Union[str, Path]] = None,
    splits_dir: Optional[Union[str, Path]] = None,
) -> None:
    start_time = time.time()
    root = Path(data_root or DATA_ROOT)
    cache = FeatureCache(dataset=dataset, encoder_id=encoder_id, data_root=root)

    # Load encoder
    if encoder_id == "siglip-b16-224":
        encoder = FrameEncoder()
    elif encoder_id == "e2v-plus-large":
        encoder = NonverbalAudioEncoder()
    else:
        raise ValueError(f"Unknown encoder: {encoder_id}")

    # Load items
    items_path = root / "items" / dataset / "items.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    all_items = read_items(items_path)

    # Filter by split if requested
    if split in ("train", "test"):
        split_data = load_split(dataset, path_dir=splits_dir or "splits")
        allowed_ids = set(split_data[split])
        items = [it for it in all_items if it.item_id in allowed_ids]
    else:
        items = all_items

    if limit is not None:
        items = items[:limit]

    items_in = len(items)
    items_out = 0
    excluded = 0

    progress_file = cache.cache_dir / "progress.json"
    finished_ids: List[str] = []
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                finished_ids = json.load(f)
        except Exception:
            finished_ids = []

    finished_set = set(finished_ids)
    errors_path = cache.cache_dir / "errors.jsonl"

    for item in items:
        # Check cache
        if not force and cache.has(item.item_id):
            items_out += 1
            if item.item_id not in finished_set:
                finished_ids.append(item.item_id)
                finished_set.add(item.item_id)
            continue

        item_t0 = time.time()
        try:
            if encoder_id == "siglip-b16-224":
                window = item.action_window_sec
            else:
                window = item.reaction_window_sec

            feat = encoder.encode_window(item.video_path, window)
            elapsed_ms = (time.time() - item_t0) * 1000.0

            cache.save(item.item_id, feat, window, elapsed_ms)
            items_out += 1
            if item.item_id not in finished_set:
                finished_ids.append(item.item_id)
                finished_set.add(item.item_id)

            # Atomically update progress.json
            with tempfile.NamedTemporaryFile("w", dir=cache.cache_dir, delete=False, encoding="utf-8") as tf:
                temp_progress = tf.name
                json.dump(finished_ids, tf)
                tf.flush()
                os.fsync(tf.fileno())
            os.replace(temp_progress, progress_file)

        except Exception as exc:
            excluded += 1
            tb_str = traceback.format_exc()
            err_entry = {
                "item_id": item.item_id,
                "error": str(exc),
                "traceback": tb_str,
                "ts": time.time(),
            }
            with open(errors_path, "a", encoding="utf-8") as ef:
                ef.write(json.dumps(err_entry) + "\n")
                ef.flush()
                os.fsync(ef.fileno())

    elapsed_s = time.time() - start_time
    print(f"items_in={items_in} items_out={items_out} excluded={excluded} elapsed_s={elapsed_s:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features for dataset and encoder")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--encoder", required=True, choices=["siglip-b16-224", "e2v-plus-large"], help="Encoder id")
    parser.add_argument("--split", default="all", choices=["train", "test", "all"], help="Split to process")
    parser.add_argument("--force", action="store_true", help="Force recomputation of cached features")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    args = parser.parse_args()

    run_extraction(
        dataset=args.dataset,
        encoder_id=args.encoder,
        split=args.split,
        force=args.force,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
