from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from PIL import Image

from config import DATA_ROOT
from features.audio import NonverbalAudioEncoder
from features.visual import FrameEncoder
from judge.vlm_judge import OllamaJudge, DATASET_QUESTIONS


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["siglip", "e2v", "judge"])
    args = parser.parse_args()

    vid_path = DATA_ROOT / "raw" / "test_ds" / "dummy.mp4"
    if not vid_path.exists():
        print(f"Video path not found: {vid_path}", file=sys.stderr)
        sys.exit(1)

    if args.step == "siglip":
        measure_siglip(vid_path)
    elif args.step == "e2v":
        measure_e2v(vid_path)
    elif args.step == "judge":
        measure_judge(vid_path)


if __name__ == "__main__":
    main()
