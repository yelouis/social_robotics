import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from shared.climax_extraction import populate_climax_for_manifest
from layer_03b_reasonable_emotion.pipeline import ReasonableEmotionPipeline
RDIR = ROOT / "e2e_reports" / "2026_06_04_layer03b_50"
MAN = RDIR / "manifest_03b_50.json"; OUT = RDIR / "03b_result_50.json"
t0 = time.time()
print("[run_03b_50] Stage 1: climax...", flush=True)
n = populate_climax_for_manifest(MAN, skip_vlm=True)
print(f"[run_03b_50] climax populated {n} in {time.time()-t0:.0f}s", flush=True)
pipe = ReasonableEmotionPipeline(str(MAN), str(OUT), force=False)
print(f"[run_03b_50] emotion_source={getattr(pipe,'emotion_source','?')} ollama={pipe.ollama_available}", flush=True)
pipe.run()
print(f"[run_03b_50] DONE total {time.time()-t0:.0f}s", flush=True)
