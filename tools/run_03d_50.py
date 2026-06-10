"""Run Layer 03d (Proxemic Kinematics) over a 50-clip sample manifest. Resumable.

On first init this downloads the registry-selected depth + SAM weights
(depth-anything-large-hf + sam-vit-huge on a 64 GB host) into the Extreme SSD
HuggingFace cache (`HF_HOME=/Volumes/Extreme SSD/huggingface_cache`), so the
boot disk stays clear. Template: edit RDIR for a new run.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.layer_03d_proxemic_kinematics.pipeline import ProxemicKinematicsPipeline  # noqa: E402

RDIR = ROOT / "e2e_reports" / "2026_06_09_layer03d_50"
MAN = RDIR / "manifest_03d_50.json"
OUT = RDIR / "03d_result_50.json"

t0 = time.time()
print("[run_03d_50] init (may download depth + SAM weights to the SSD on first run)...", flush=True)
pipe = ProxemicKinematicsPipeline(str(MAN), str(OUT), force=True)
print(f"[run_03d_50] init done in {time.time()-t0:.0f}s on device={pipe.device}; processing 50 clips...", flush=True)
pipe.run()
print(f"[run_03d_50] DONE total {time.time()-t0:.0f}s -> {OUT}", flush=True)
