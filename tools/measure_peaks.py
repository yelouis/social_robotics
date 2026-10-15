from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image

from config import DATA_ROOT
from features.audio import NonverbalAudioEncoder
from features.visual import FrameEncoder
from harness.items import read_items
from harness.splits import load_split
from judge.vlm_judge import DATASET_QUESTIONS, OllamaJudge, make_large_judge
from shared.memguard import is_model_loaded, read_memory


def measure_siglip(vid_path: Path) -> None:
    enc = FrameEncoder()
    dummy_img = Image.new("RGB", (224, 224), color="red")
    for _ in range(50):
        _ = enc.encode_frames([dummy_img, dummy_img])
    enc.release()
    print("siglip 50 items done")


def measure_e2v(vid_path: Path) -> None:
    enc = NonverbalAudioEncoder()
    for _ in range(50):
        _ = enc.encode_window(vid_path, [0.0, 1.0])
    enc.release()
    print("e2v 50 items done")


def measure_judge(vid_path: Path) -> None:
    judge = OllamaJudge()
    prob, raw, attempts, elapsed = judge.judge_item(
        video_path=vid_path,
        window=[0.0, 1.0],
        context_text="A short clip.",
        question=DATASET_QUESTIONS["oops"],
    )
    print(f"judge result: prob={prob}, attempts={attempts}, elapsed={elapsed:.1f}ms")

    # Read llama-server ps RSS
    res = subprocess.run(
        ["ps", "-Ao", "rss,command"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in res.stdout.splitlines():
        if "llama-server" in line:
            parts = line.strip().split()
            rss_kb = int(parts[0])
            rss_gb = rss_kb / (1024 * 1024)
            print(f"llama-server RSS: {rss_kb} KB ({rss_gb:.2f} GB)")


def get_gemma_blob() -> str:
    try:
        res = subprocess.run(["ollama", "show", "gemma4:26b", "--modelfile"], capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            if line.startswith("FROM ") and "blobs" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return Path(parts[1]).name
    except Exception:
        pass
    return ""


def measure_judge_large() -> None:
    # Precondition: /api/ps does not list gemma4:26b
    while is_model_loaded("gemma4:26b"):
        print("Precondition: gemma4:26b is currently loaded in Ollama. Waiting for it to expire (never unload)...")
        time.sleep(15)

    gemma_blob = get_gemma_blob()

    items_path = DATA_ROOT / "items" / "oops" / "items.jsonl"
    if not items_path.exists():
        print(f"Oops items not found: {items_path}", file=sys.stderr)
        sys.exit(1)
    all_items = read_items(items_path)
    split_data = load_split("oops")
    test_ids = set(split_data["test"])
    test_items = [it for it in all_items if it.item_id in test_ids]
    test_items.sort(key=lambda it: it.item_id)
    items_to_score = test_items[:50]
    print(f"Scoring first {len(items_to_score)} Oops! test items with make_large_judge()...")

    avail_before, press_before = read_memory()
    print(f"read_memory before load: available={avail_before / (1024**3):.2f} GB, pressure={press_before}")

    judge = make_large_judge()
    t0 = time.time()
    for idx, it in enumerate(items_to_score):
        p, raw, att, el = judge.judge_item(
            video_path=it.video_path,
            window=it.action_window_sec,
            context_text=it.context_text,
            question=DATASET_QUESTIONS["oops"],
        )
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/50] item={it.item_id} prob={p} elapsed={el:.1f}ms")

    elapsed_total = time.time() - t0
    print(f"50 items scored in {elapsed_total:.1f}s")

    avail_after, press_after = read_memory()
    print(f"read_memory after 50 calls: available={avail_after / (1024**3):.2f} GB, pressure={press_after}")
    drop_bytes = max(0, avail_before - avail_after)
    drop_gb = drop_bytes / (1024**3)

    # Read llama-server ps RSS
    res = subprocess.run(["ps", "-Ao", "rss,command"], capture_output=True, text=True, check=True)
    server_rss_gb = 0.0
    for line in res.stdout.splitlines():
        if "llama-server" in line:
            if not gemma_blob or gemma_blob in line:
                parts = line.strip().split()
                rss_kb = int(parts[0])
                cur_gb = rss_kb / (1024 * 1024)
                if cur_gb > server_rss_gb:
                    server_rss_gb = cur_gb
                print(f"llama-server RSS: {rss_kb} KB ({cur_gb:.2f} GB)")

    measured = max(server_rss_gb, drop_gb)
    declared_peak = math.ceil(measured * 1.15)
    print("\n--- Summary ---")
    print(f"Drop in available memory: {drop_gb:.2f} GB")
    print(f"llama-server RSS: {server_rss_gb:.2f} GB")
    print(f"Measured = max(RSS, drop): {measured:.2f} GB")
    print(f"Declared peak = ceil(measured * 1.15): {declared_peak} GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["siglip", "e2v", "judge", "judge_large"])
    args = parser.parse_args()

    vid_path = DATA_ROOT / "raw" / "test_ds" / "dummy.mp4"

    if args.step == "siglip":
        if not vid_path.exists():
            print(f"Video path not found: {vid_path}", file=sys.stderr)
            sys.exit(1)
        measure_siglip(vid_path)
    elif args.step == "e2v":
        if not vid_path.exists():
            print(f"Video path not found: {vid_path}", file=sys.stderr)
            sys.exit(1)
        measure_e2v(vid_path)
    elif args.step == "judge":
        if not vid_path.exists():
            print(f"Video path not found: {vid_path}", file=sys.stderr)
            sys.exit(1)
        measure_judge(vid_path)
    elif args.step == "judge_large":
        measure_judge_large()


if __name__ == "__main__":
    main()
