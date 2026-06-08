"""Run Layer 03a (Attention) over the bounded 10-clip sample. Resumable."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from layer_03a_attention.pipeline import AttentionLayerPipeline  # noqa: E402

RDIR = ROOT / "e2e_reports" / "2026_06_02_layer03a"
MAN = RDIR / "manifest_10.json"
OUT = RDIR / "03a_attention_result_10_v4.json"  # post-fix re-run (face gate + bbox re-detect + new fields)

pipe = AttentionLayerPipeline(str(MAN), str(OUT), force=False)
print(f"[run_03a] gaze model loaded: {pipe.gaze_pipeline is not None} | "
      f"already processed: {len(pipe.processed_ids)}", flush=True)
pipe.run()
