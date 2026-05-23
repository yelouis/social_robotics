import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Storage routing. Per docs/01a_synthetic_positive_generation.md §9.1 and
# docs/01_dataset_acquisition.md, model weights and any transient render files
# MUST live on the external "Extreme SSD", never the internal disk. huggingface
# resolves these env vars at import time, so they must be set *before* the
# `import diffusers` below. The shell profile should already set them; the
# setdefault calls are a defensive backstop that never override an explicit
# operator setting.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_SSD_ROOT = Path("/Volumes/Extreme SSD")
if _SSD_ROOT.exists():
    os.environ.setdefault("HF_HOME", str(_SSD_ROOT / "huggingface_cache"))
    os.environ.setdefault("TMPDIR", str(_SSD_ROOT / "tmp"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Make `models_config` importable when this file is run directly.
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:
    torch = None

# diffusers is an optional heavy dependency (see docs/ml_dependencies.md §2).
# The module must import cleanly without it so the registry and tests can run on
# hosts that have not installed the generator stack; generation then raises a
# clear, actionable error instead of an import crash.
try:
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
except ImportError:
    AutoencoderKLWan = None
    WanPipeline = None
    UniPCMultistepScheduler = None
    export_to_video = None


GENERATOR_NAME = "wan2.1-t2v"
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
# ~5 s at 16 fps (doc §3.1/§9.3 committed-fixture length). 81 frames is the
# minimum that yields >=2 sampled frames under Layer 02's 1/3-fps sampler
# (frame_interval=48 -> samples at idx 0 and 48), so the synthetic can clear
# the detector's min_consistency=2 temporal gate. The prior value (33 -> ~2 s)
# yielded a single sampled frame and was structurally unable to pass the
# filter. Overridable via SR_SYNTH_NUM_FRAMES for memory-constrained hosts.
NUM_FRAMES = int(os.getenv("SR_SYNTH_NUM_FRAMES", "81"))
FPS = 16
DEFAULT_STEPS = int(os.getenv("SR_SYNTH_STEPS", "50"))
SMOKE_STEPS = 10

# Reinforces the three anti-stereo phrases in the positive scaffold
# (prompts.json). Keeping clips single-camera is the dominant lever for
# surviving Layer 02's stereo gate — see docs/01a §3.1.
NEGATIVE_PROMPT = (
    "split-screen, side-by-side, stereoscopic, picture-in-picture, letterbox, "
    "third-person, static, blurry, low quality, distorted, watermark, text overlay"
)


def _install_hint():
    return (
        "Install the local generator stack into the project venv:\n"
        '  pip install "diffusers>=0.33" transformers accelerate ftfy imageio imageio-ffmpeg\n'
        "See docs/01a_synthetic_positive_generation.md §9 for full setup."
    )


class SyntheticVideoGenerator:
    def __init__(self, output_dir=None):
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path("/Volumes/Extreme SSD/social_robotics/raw_videos/synthetic_validation")
        )

        # Load prompts configuration
        prompts_path = Path(__file__).parent / "prompts.json"
        with open(prompts_path, "r") as f:
            self.prompts_config = json.load(f)

        self.version = self.prompts_config.get("version", 1)
        self.scaffold = self.prompts_config.get("scaffold", "")
        self.scenarios = self.prompts_config.get("scenarios", {})
        self._pipe = None  # lazily loaded WanPipeline, shared across a batch

    # --- prompt construction -------------------------------------------------
    def get_prompt(self, scenario_tag, seed=0):
        """Construct the prompt from scaffold + scenario filler, with optional seed variation."""
        if scenario_tag not in self.scenarios:
            raise ValueError(f"Unknown scenario tag: {scenario_tag}")

        scenario = self.scenarios[scenario_tag]
        prompt = self.scaffold.format(
            task_description=scenario["task_description"],
            bystander_description=scenario["bystander_description"],
            reaction_description=scenario["reaction_description"],
        )
        if seed > 0:
            prompt += f"\nVariation seed: {seed}."
        return prompt

    def get_prompt_hash(self, prompt):
        """8-character hash over the prompt and the prompt-config version."""
        hasher = hashlib.sha256()
        hasher.update(f"v{self.version}:".encode("utf-8"))
        hasher.update(prompt.encode("utf-8"))
        return hasher.hexdigest()[:8]

    # --- model / device resolution ------------------------------------------
    def resolve_model_id(self):
        """Resolve the per-tier Wan2.1 repo id from the central registry."""
        try:
            from models_config import get_model
            return get_model("synthetic_generator")
        except Exception:
            return DEFAULT_MODEL_ID

    def select_device(self):
        if torch is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _generator_version(model_id):
        if "14B" in model_id:
            return "14b-diffusers"
        if "1.3B" in model_id:
            return "1.3b-diffusers"
        return "diffusers"

    @staticmethod
    def _render_params(model_id):
        """Tier-specific sampling params (mirrors the Wan2.1 model card; see docs/01a §9.3)."""
        is_720p = "14B" in model_id
        height, width = (720, 1280) if is_720p else (480, 832)
        flow_shift = 5.0 if is_720p else 3.0
        guidance_scale = 5.0 if is_720p else 6.0
        return height, width, flow_shift, guidance_scale

    def _load_pipeline(self, model_id):
        if torch is None:
            raise ImportError("PyTorch is required for synthetic generation.\n" + _install_hint())
        if WanPipeline is None:
            raise ImportError("`diffusers` (>=0.33) is required for synthetic generation.\n" + _install_hint())

        _, _, flow_shift, _ = self._render_params(model_id)
        device = self.select_device()
        print(f"[SyntheticVideoGenerator] Loading {model_id} on device '{device}' (bf16)...", flush=True)

        # Wan recommends the VAE in fp32 and the transformer + text encoder in bf16.
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        if UniPCMultistepScheduler is not None:
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=flow_shift,
            )
        pipe.to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.vae.enable_tiling()  # caps the VAE-decode memory spike on long clips
        except Exception:
            pass  # tiling is a memory optimization; absence is non-fatal
        return pipe

    # --- generation ----------------------------------------------------------
    def generate_video(self, scenario_tag, seed=0, force=False, smoke=False):
        """Generate (or reuse the cached) clip for a scenario.

        The pipeline is loaded lazily on the first cache miss and reused across a
        batch; call `release()` (or `generate_batch`, which does it for you) to
        free it before the Node 02 filtering models load.
        """
        prompt = self.get_prompt(scenario_tag, seed)
        prompt_hash = self.get_prompt_hash(prompt)

        scenario_dir = self.output_dir / scenario_tag
        scenario_dir.mkdir(parents=True, exist_ok=True)
        dest_path = scenario_dir / f"{prompt_hash}.mp4"

        if dest_path.exists() and not force:
            print(f"[SyntheticVideoGenerator] Cache hit for {scenario_tag} (seed={seed}, hash={prompt_hash}). Using {dest_path}")
            return dest_path

        print(f"[SyntheticVideoGenerator] Cache miss for {scenario_tag} (seed={seed}, hash={prompt_hash}). Generating...")

        model_id = self.resolve_model_id()
        height, width, _, guidance_scale = self._render_params(model_id)
        steps = SMOKE_STEPS if smoke else DEFAULT_STEPS
        num_frames = 17 if smoke else NUM_FRAMES

        if self._pipe is None:
            self._pipe = self._load_pipeline(model_id)

        # A CPU generator keeps seeding reproducible and portable across MPS/CPU.
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
        result = self._pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=rng,
            output_type="latent",
        )
        latents = result.frames
        del result

        if isinstance(latents, torch.Tensor):
            # Extract VAE and video processor before deleting pipeline to free up memory
            vae = self._pipe.vae
            video_processor = self._pipe.video_processor

            # Move VAE to CPU to avoid MPS backend out of memory.
            # We also cast the input latents to CPU and float32.
            print("[SyntheticVideoGenerator] Offloading VAE and latents to CPU for decoding...", flush=True)
            vae.to("cpu")
            latents = latents.to(device="cpu", dtype=vae.dtype)

            # Delete the pipe object to reclaim memory (~14 GB)
            self._pipe = None
            gc.collect()
            if torch is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # De-normalize latents on CPU
            vae_config = vae.config
            latents_mean = (
                torch.tensor(vae_config.latents_mean)
                .view(1, vae_config.z_dim, 1, 1, 1)
                .to(device="cpu", dtype=latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1).to(
                device="cpu", dtype=latents.dtype
            )
            latents = latents / latents_std + latents_mean

            # Perform the actual decoding on CPU
            print("[SyntheticVideoGenerator] Decoding latents on CPU...", flush=True)
            video = vae.decode(latents, return_dict=False)[0]

            # Postprocess on CPU
            print("[SyntheticVideoGenerator] Postprocessing video on CPU...", flush=True)
            video = video_processor.postprocess_video(video, output_type="np")
            frames = video[0]
        else:
            # For unit tests where self._pipe is mocked and returns mock/list/etc.
            frames = latents[0] if isinstance(latents, list) else latents

        export_to_video(frames, str(dest_path), fps=FPS)

        self._write_sidecar(
            scenario_dir / f"{prompt_hash}.json",
            scenario_tag, prompt, prompt_hash, seed, model_id, height, width, steps, num_frames
        )
        print(f"[SyntheticVideoGenerator] Generated and cached: {dest_path}")
        return dest_path

    def _write_sidecar(self, sidecar_path, scenario_tag, prompt, prompt_hash, seed, model_id, height, width, steps, num_frames):
        """Write the per-clip provenance sidecar consumed by registry.py."""
        meta = {
            "generator": GENERATOR_NAME,
            "generator_version": self._generator_version(model_id),
            "model_id": model_id,
            "scenario_tag": scenario_tag,
            "prompt_hash": prompt_hash,
            "prompt": prompt,
            "seed": int(seed),
            "num_frames": num_frames,
            "fps": FPS,
            "num_inference_steps": steps,
            "resolution": f"{width}x{height}",
            "prompts_version": self.version,
        }
        with open(sidecar_path, "w") as f:
            json.dump(meta, f, indent=2)

    def release(self):
        """Free the pipeline so its footprint never co-resides with the Node 02
        filtering / Layer-03 models (docs/01a §3)."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def generate_batch(self, count=3, force=False, smoke=False):
        """Generate `count` clips round-robin across scenarios, then release the pipeline."""
        tags = list(self.scenarios.keys())
        results = []
        try:
            for idx in range(count):
                tag = tags[idx % len(tags)]
                seed = idx // len(tags)
                path = self.generate_video(tag, seed=seed, force=force, smoke=smoke)
                results.append((tag, seed, path))
        finally:
            self.release()
        return results


def _main():
    parser = argparse.ArgumentParser(description="Wan2.1 synthetic validation video generator (Layer 1a).")
    parser.add_argument("--smoke", action="store_true", help="Render one clip at reduced steps to validate wiring.")
    parser.add_argument("--count", type=int, default=None, help="Generate N clips round-robin across scenarios.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if a cached clip exists.")
    args = parser.parse_args()

    generator = SyntheticVideoGenerator()

    if args.smoke:
        tag = next(iter(generator.scenarios))
        print(f"[SyntheticVideoGenerator] Smoke render for scenario '{tag}' ({SMOKE_STEPS} steps)...")
        path = generator.generate_video(tag, seed=0, force=args.force, smoke=True)
        generator.release()
        print(f"Smoke clip: {path}")
        return

    if args.count is not None:
        results = generator.generate_batch(count=args.count, force=args.force)
        for tag, seed, path in results:
            print(f"  {tag} (seed={seed}) -> {path}")
        return

    # Default: dry-run — print the baseline prompts + hashes, no generation.
    print("Baseline scenario prompts (dry-run; pass --smoke or --count to generate):")
    for tag in generator.scenarios:
        prompt = generator.get_prompt(tag)
        prompt_hash = generator.get_prompt_hash(prompt)
        indented = prompt.replace("\n", "\n  ")
        print(f"- {tag} (hash: {prompt_hash}):\n  {indented}\n")


if __name__ == "__main__":
    _main()
