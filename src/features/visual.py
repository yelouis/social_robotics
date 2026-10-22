from __future__ import annotations

import io
from pathlib import Path
import subprocess
from typing import List, Optional, Union

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, SiglipModel


def get_sample_times(s: float, e: float) -> List[float]:
    times = []
    curr = float(s)
    end = float(e)
    while curr < end:
        times.append(curr)
        curr += 1.0
    if len(times) == 0 or times[-1] != end:
        times.append(end)
    return times


def get_sample_times_u9(s: float, e: float) -> List[float]:
    """Sample 9 frames uniformly at t_k = s + k * (e - s) / 8 for k in 0..8."""
    start = float(s)
    end = float(e)
    return [start + k * (end - start) / 8.0 for k in range(9)]


def get_u9_sample_times(duration: float) -> List[float]:
    """Sample 9 frames uniformly at t_k = k * D / 8 for k=0..8."""
    return get_sample_times_u9(0.0, float(duration))


def extract_frame(video_path: Union[str, Path], t: float) -> Image.Image:
    for seek_t in [t, max(0.0, t - 0.05), max(0.0, t - 0.1), max(0.0, t - 0.2), max(0.0, t - 0.3)]:
        cmd = [
            "ffmpeg",
            "-ss", f"{seek_t:.4f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if proc.stdout:
            return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    raise RuntimeError(f"ffmpeg extracted 0 bytes for frame at {t:.4f} in {video_path}")


class FrameEncoder:
    def __init__(self, encoder_id: str = "siglip-b16-224", device: Optional[str] = None) -> None:
        self.encoder_id = encoder_id
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[SiglipModel] = None

    def _ensure_loaded(self) -> None:
        if self._model is None or self._processor is None:
            from shared import memguard

            with memguard.guard("sr_siglip"):
                if self._model is None or self._processor is None:
                    model_name = "google/siglip-base-patch16-224"
                    self._processor = AutoProcessor.from_pretrained(model_name)
                    self._model = SiglipModel.from_pretrained(model_name).to(self.device)
                    self._model.eval()

    def release(self) -> None:
        self._model = None
        self._processor = None
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()

    def encode_frames(self, images: List[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            # L2-normalize each frame feature
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            # Mean pool across frames
            mean_feat = feats.mean(dim=0).cpu().numpy().astype(np.float32)

        return mean_feat

    def encode_window(self, video_path: Union[str, Path], window: List[float]) -> np.ndarray:
        s, e = window[0], window[1]
        if self.encoder_id == "siglip-b16-224-u9":
            duration = max(0.0, e - s)
            sample_times = [round(s + t, 4) for t in get_u9_sample_times(duration)]
        else:
            sample_times = get_sample_times(s, e)
        frames = [extract_frame(video_path, t) for t in sample_times]
        return self.encode_frames(frames)
