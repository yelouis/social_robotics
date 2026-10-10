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


def extract_frame(video_path: Union[str, Path], t: float) -> Image.Image:
    cmd = [
        "ffmpeg",
        "-ss", f"{t:.4f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg extracted 0 bytes for frame at {t:.4f} in {video_path}")
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


class FrameEncoder:
    encoder_id: str = "siglip-b16-224"

    def __init__(self, device: Optional[str] = None) -> None:
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
        sample_times = get_sample_times(s, e)
        frames = [extract_frame(video_path, t) for t in sample_times]
        return self.encode_frames(frames)
