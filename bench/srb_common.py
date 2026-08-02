"""SocialRobotics-Bench (SRB) v0 — shared constants + tiny helpers.

Benchmark code (docs/07). Boundary rule (docs/07 §0): bench/ never imports
from src/layer_* except src/shared video-cutting utilities; engine outputs are
consumed only as ARTIFACTS (manifest JSON, segment_rows.parquet).
"""
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
SSD_ROOT = Path("/Volumes/Extreme SSD/social_robotics")
BENCH_DATA = SSD_ROOT / "bench_v0"           # all v0 working data (internal-only: Ego4D pixels)

EGO4D_META = SSD_ROOT / "raw_videos" / "ego4d" / "ego4d.json"
PUBLISHED_MANIFEST = SSD_ROOT / "full_run_2026_06_18" / "03a" / "input_991.json"

SPLITS_FILE = BENCH_DIR / "splits_ego4d.json"          # docs/06 Issue-6 minimal stamping (tracked in git)
PILOT_WAVE = "socialrobotics-bench-pilot"

# Rating-kit windows (docs/07 §4.2)
CLIP_S_PRE_SEC = 4.0
CLIP_S_POST_SEC = 3.0
CLIP_SPLUS_END_SEC = 12.0
STRIP_FRAMES = 8

# CSV enums (docs/07 §4.3/4.4, verbatim options -> short codes)
ENUMS = {
    "A1": ["yes", "no", "unsure"],
    "A2": ["yes", "no"],
    # A3 free-form flag list (semicolon-separated) — validated by keyword
    "B2": ["yes", "no", "unsure"],
    "B3": ["yes", "no", "unsure"],
    "B4": ["wearer", "something_else", "cant_tell"],
    "B5": ["approving", "disapproving", "neutral", "mixed"],
    "B7": ["continues", "adjusts", "stops", "cant_tell"],
    "B8": ["confident", "somewhat", "guessing"],
}
B6_CHANNELS = ["face", "head_gesture", "hand_body", "proxemics", "voice", "gaze"]
A3_FLAGS = ["minor", "nudity_private", "sensitive_info", "distress", "other"]

RATING_COLUMNS = ["moment_id", "A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5",
                  "B6", "B7", "B8", "B9", "C1", "seconds_spent"]


def moment_id(corpus: str, clip_id: str, t_climax_sec: float) -> str:
    return f"srb-{corpus}-{clip_id}-{int(round(t_climax_sec * 1000))}"


def read_jsonl(path):
    import json
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
