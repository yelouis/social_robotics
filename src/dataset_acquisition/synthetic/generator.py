import os
import sys
import json
import time
import hashlib
from pathlib import Path

# Add src to path if run directly
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

class SyntheticVideoGenerator:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path("/Volumes/Extreme SSD/social_robotics/raw_videos/synthetic_validation")
        
        # Load prompts configuration
        prompts_path = Path(__file__).parent / "prompts.json"
        with open(prompts_path, "r") as f:
            self.prompts_config = json.load(f)
            
        self.version = self.prompts_config.get("version", 1)
        self.scaffold = self.prompts_config.get("scaffold", "")
        self.scenarios = self.prompts_config.get("scenarios", {})

    def get_prompt(self, scenario_tag: str, seed: int = 0) -> str:
        """Construct the prompt using scaffold and scenario description, with optional seed variation."""
        if scenario_tag not in self.scenarios:
            raise ValueError(f"Unknown scenario tag: {scenario_tag}")
            
        scenario = self.scenarios[scenario_tag]
        prompt = self.scaffold.format(
            task_description=scenario["task_description"],
            bystander_description=scenario["bystander_description"],
            reaction_description=scenario["reaction_description"]
        )
        if seed > 0:
            prompt += f"\nVariation seed: {seed}."
        return prompt

    def get_prompt_hash(self, prompt: str) -> str:
        """Compute 8-character hash from prompt and prompt config version."""
        hasher = hashlib.sha256()
        hasher.update(f"v{self.version}:".encode("utf-8"))
        hasher.update(prompt.encode("utf-8"))
        return hasher.hexdigest()[:8]

    def generate_video(self, scenario_tag: str, seed: int = 0, force: bool = False) -> Path:
        """Generate a video clip for a scenario. Reuses cache unless force is True."""
        prompt = self.get_prompt(scenario_tag, seed)
        prompt_hash = self.get_prompt_hash(prompt)
        
        scenario_dir = self.output_dir / scenario_tag
        scenario_dir.mkdir(parents=True, exist_ok=True)
        dest_path = scenario_dir / f"{prompt_hash}.mp4"
        
        if dest_path.exists() and not force:
            print(f"[SyntheticVideoGenerator] Cache hit for {scenario_tag} (seed={seed}, hash={prompt_hash}). Using {dest_path}")
            return dest_path
            
        print(f"[SyntheticVideoGenerator] Cache miss for {scenario_tag} (seed={seed}, hash={prompt_hash}). Generating...")
        
        # Enforce API Key
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key missing! Please set the GOOGLE_API_KEY or GEMINI_API_KEY environment variable. "
                "Unable to generate new synthetic videos without API credentials."
            )
            
        if genai is None or types is None:
            raise ImportError(
                "The `google-genai` package is not installed. "
                "Run `pip install google-genai` to install it."
            )

        # Adhere to Hardware Management Rule: force TMPDIR to SSD
        os.environ["TMPDIR"] = "/Volumes/Extreme SSD/tmp"
        Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        
        # Initialize Google GenAI client
        client = genai.Client(api_key=api_key)
        
        # We read from models_config.py if possible to resolve model ID
        try:
            from models_config import get_model
            model = get_model("synthetic_generator")
        except ImportError:
            model = "veo-2.0-generate-001"

        print(f"[SyntheticVideoGenerator] Submitting video generation request to model '{model}'...")
        # Submit asynchronous generation request
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9"
            ),
        )
        
        # Poll for completion
        print("[SyntheticVideoGenerator] Waiting for Veo generation to complete...", flush=True)
        while not operation.done:
            time.sleep(15)
            print(".", end="", flush=True)
            try:
                operation = client.operations.get(operation)
            except Exception as e:
                print(f"\n[Warning] Failed to refresh operation status: {e}. Retrying...")
        print()
        
        if not operation.response or not operation.response.generated_videos:
            raise RuntimeError("Video generation completed but no video was returned by the API.")
            
        video_data = operation.response.generated_videos[0]
        
        # Download video file
        print(f"[SyntheticVideoGenerator] Downloading generated video file...")
        downloaded_file_bytes = client.files.download(file=video_data.video)
        
        with open(dest_path, "wb") as f:
            f.write(downloaded_file_bytes)
            
        print(f"[SyntheticVideoGenerator] Successfully generated and cached: {dest_path}")
        return dest_path

    def generate_batch(self, count: int = 10, force: bool = False):
        """Generate a batch of videos distributed round-robin across all scenario tags."""
        tags = list(self.scenarios.keys())
        results = []
        for idx in range(count):
            tag = tags[idx % len(tags)]
            seed = idx // len(tags)
            path = self.generate_video(tag, seed=seed, force=force)
            results.append((tag, seed, path))
        return results

if __name__ == "__main__":
    # If run as script, perform a dry-run check or generate the baseline set
    generator = SyntheticVideoGenerator()
    print("Baseline scenario prompts:")
    for tag in generator.scenarios:
        prompt = generator.get_prompt(tag)
        prompt_hash = generator.get_prompt_hash(prompt)
        print(f"- {tag} (hash: {prompt_hash}):\n  {prompt.replace(chr(10), '  ' + chr(10))}\n")
