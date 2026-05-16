"""Tier-per-host model configuration.

Single source of truth for which model variant each layer should load. The
table below maps every layer's heavy models onto a (small, medium, large) tier
with the identifier the layer must pass to its loader plus the approximate
resident-set cost. Layers read their model id via `get_model(layer_key)` so the
choice is centralized and inspectable from a single file.

Host-tier auto-detection: hosts with at least 48 GB of unified memory default
to `medium` (current Mac Studio M4 Max production target); smaller hosts (e.g.
the legacy 24 GB Mac mini M4 Pro) default to `small`. The override env var
`SR_MODEL_TIER` (`small` | `medium` | `large`) wins over auto-detection, which
researchers can set per-experiment to A/B a larger tier on a small host or pin
small on the studio for fast iteration.

The startup banner (printed once at first import) reports the active tier,
the detected host memory, and the summed approximate resident set so an OOM
regression on a new host is visible in the first line of pipeline output.
"""

import os
from typing import Dict, Optional, Tuple

HIGH_MEMORY_HOST_BYTES = 48 * 2**30
VALID_TIERS = ("small", "medium", "large")

# layer_key -> tier -> (model_id, approx_size_str, approx_size_bytes)
#
# Sizes are nominal — they reflect the documented resident footprint, not the
# on-disk weight. The bytes column is used only for the banner total; a `None`
# entry is excluded from the sum (e.g. shared models counted under another
# layer).
_MODEL_TIERS: Dict[str, Dict[str, Tuple[str, str, int]]] = {
    # Layer 03b: Reasonable Emotion
    "layer_03b_ollama": {
        "small":  ("gemma4:e4b", "~2.5 GB", 2_500_000_000),
        "medium": ("gemma4:26b", "~15 GB", 15_000_000_000),
        "large":  ("gemma4:26b", "~15 GB", 15_000_000_000),
    },
    "layer_03b_face_emotion": {
        "small":  ("enet_b0_8", "~30 MB", 30_000_000),
        "medium": ("enet_b2_8", "~50 MB", 50_000_000),
        "large":  ("enet_b2_8", "~50 MB", 50_000_000),
    },
    # Layer 03c: Acoustic Prosody
    "layer_03c_ser": {
        "small":  ("iic/emotion2vec_plus_base",  "~300 MB", 300_000_000),
        "medium": ("iic/emotion2vec_plus_large", "~600 MB", 600_000_000),
        "large":  ("iic/emotion2vec_plus_seed",  "~2 GB",   2_000_000_000),
    },
    "layer_03c_aed": {
        "small":  ("iic/SenseVoiceSmall", "~500 MB", 500_000_000),
        "medium": ("iic/SenseVoiceSmall", "~500 MB", 500_000_000),
        "large":  ("iic/SenseVoiceSmall", "~500 MB", 500_000_000),
    },
    # Layer 03d: Proxemic Kinematics
    "layer_03d_depth": {
        "small":  ("depth-anything/Depth-Anything-V2-Small-hf", "~100 MB", 100_000_000),
        "medium": ("LiheYoung/depth-anything-large-hf",         "~1.3 GB", 1_300_000_000),
        "large":  ("LiheYoung/depth-anything-large-hf",         "~1.3 GB", 1_300_000_000),
    },
    "layer_03d_sam": {
        "small":  ("facebook/sam-vit-base", "~375 MB", 375_000_000),
        "medium": ("facebook/sam-vit-huge", "~2.5 GB", 2_500_000_000),
        "large":  ("facebook/sam-vit-huge", "~2.5 GB", 2_500_000_000),
    },
    # Layer 03f: Motor Resonance
    "layer_03f_pose": {
        "small":  ("yolov8n-pose.pt", "~6.5 MB", 6_500_000),
        "medium": ("yolov8x-pose.pt", "~100 MB", 100_000_000),
        "large":  ("yolov8x-pose.pt", "~100 MB", 100_000_000),
    },
    # Stage-2 filtering VLM (climax refinement)
    "filtering_vlm": {
        "small":  ("qwen2.5vl:3b", "~3 GB", 3_000_000_000),
        "medium": ("qwen2.5vl:7b", "~7 GB", 7_000_000_000),
        "large":  ("qwen2.5vl:7b", "~7 GB", 7_000_000_000),
    },
    # Shared social-presence YOLO-pose detector (filtering + 02 verification)
    "social_presence_pose": {
        "small":  ("yolov8n-pose.pt", "~6.5 MB", None),  # counted under 03f
        "medium": ("yolov8n-pose.pt", "~6.5 MB", None),
        "large":  ("yolov8n-pose.pt", "~6.5 MB", None),
    },
    # Lightweight VLM used by SocialPresenceDetector to disambiguate ambiguous
    # bystander candidates (loaded lazily, not part of the steady-state set).
    "social_presence_vlm_verify": {
        "small":  ("moondream",     "~1.6 GB", 1_600_000_000),
        "medium": ("moondream",     "~1.6 GB", 1_600_000_000),
        "large":  ("qwen2.5vl:7b",  "~7 GB",   None),  # counted under filtering_vlm
    },
}


def _detect_host_total_bytes() -> int:
    """Total physical memory in bytes. Falls back to 0 when psutil is missing
    so callers downgrade to the conservative `small` default rather than
    crashing on hosts without psutil."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        return 0


def _auto_tier() -> str:
    return "medium" if _detect_host_total_bytes() >= HIGH_MEMORY_HOST_BYTES else "small"


def get_active_tier() -> str:
    """Return the active tier, honoring `SR_MODEL_TIER` over auto-detection.
    Invalid env values fall back to auto-detection with a stderr warning."""
    override = os.getenv("SR_MODEL_TIER", "").strip().lower()
    if override in VALID_TIERS:
        return override
    if override:
        import sys
        print(
            f"[models_config] WARNING: SR_MODEL_TIER='{override}' is not one of "
            f"{VALID_TIERS}; falling back to auto-detection.",
            file=sys.stderr,
        )
    return _auto_tier()


def get_model(layer_key: str, tier: Optional[str] = None) -> str:
    """Return the model identifier this layer should load.

    `layer_key` must be one of the keys in `_MODEL_TIERS`. An unknown key is a
    programmer error (raises KeyError) — the registry is meant to be exhaustive
    so a missing entry indicates a layer was added without registering its
    model variants here.
    """
    if layer_key not in _MODEL_TIERS:
        raise KeyError(
            f"Unknown model layer_key '{layer_key}'. Add it to _MODEL_TIERS in "
            f"src/models_config.py before calling get_model()."
        )
    tier = tier or get_active_tier()
    return _MODEL_TIERS[layer_key][tier][0]


def get_model_info(layer_key: str, tier: Optional[str] = None) -> Tuple[str, str]:
    """Return (model_id, approx_size_str) for the active tier."""
    tier = tier or get_active_tier()
    model_id, size_str, _ = _MODEL_TIERS[layer_key][tier]
    return model_id, size_str


def _estimated_resident_bytes(tier: str) -> int:
    total = 0
    for variants in _MODEL_TIERS.values():
        _, _, b = variants[tier]
        if b is not None:
            total += b
    return total


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _format_banner(tier: str) -> str:
    host_bytes = _detect_host_total_bytes()
    host_str = _human_bytes(host_bytes) if host_bytes else "unknown (no psutil)"
    resident = _estimated_resident_bytes(tier)
    auto = _auto_tier()
    override = os.getenv("SR_MODEL_TIER", "").strip().lower()
    source = f"env override SR_MODEL_TIER={override}" if override in VALID_TIERS else f"auto-detected (host: {host_str})"
    lines = [
        "=" * 72,
        f"[models_config] Active model tier: {tier.upper()} ({source})",
        f"[models_config] Estimated steady-state resident set: {_human_bytes(resident)}",
        "[models_config] Per-layer selections:",
    ]
    for key in _MODEL_TIERS:
        model_id, size_str, _ = _MODEL_TIERS[key][tier]
        lines.append(f"  - {key:32s} -> {model_id}  ({size_str})")
    lines.append("=" * 72)
    return "\n".join(lines)


_BANNER_PRINTED = False


def print_startup_banner(force: bool = False) -> None:
    """Print the model-tier banner once per process. Idempotent so multiple
    layer imports don't spam stdout. Pass `force=True` to reprint (e.g. after
    flipping `SR_MODEL_TIER` mid-session in a notebook)."""
    global _BANNER_PRINTED
    if _BANNER_PRINTED and not force:
        return
    print(_format_banner(get_active_tier()))
    _BANNER_PRINTED = True


# Print the banner the first time this module is imported so any pipeline
# entry-point gets a one-line visible record of the host's model tier without
# the caller having to remember to invoke it. Suppress via SR_NO_MODEL_BANNER=1
# for unit tests / dataset-acquisition scripts that don't load models.
if os.getenv("SR_NO_MODEL_BANNER", "").strip().lower() not in ("1", "true", "yes"):
    print_startup_banner()
