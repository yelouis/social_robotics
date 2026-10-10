from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, List, Optional, Union

import numpy as np


def extract_audio_wav(
    video_path: Union[str, Path],
    s: float,
    e: float,
    out_wav_path: Union[str, Path],
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{float(s):.4f}",
        "-to", f"{float(e):.4f}",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_wav_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio: {proc.stderr.decode('utf-8', errors='replace')}")


class NonverbalAudioEncoder:
    encoder_id: str = "e2v-plus-large"

    def __init__(self) -> None:
        self._model: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from funasr import AutoModel
            self._model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

    def encode_wav(self, wav_path: Union[str, Path]) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        res = self._model.generate(str(wav_path), granularity="utterance", extract_embedding=True)
        if not res or not isinstance(res, list) or "feats" not in res[0]:
            raise RuntimeError(f"emotion2vec failed to extract embedding from {wav_path}")
        feats = res[0]["feats"]
        # Strictly return float32 embedding; never inspect emotion categories
        return np.asarray(feats, dtype=np.float32)

    def encode_window(self, video_path: Union[str, Path], window: List[float]) -> np.ndarray:
        tmp_dir = os.environ.get("TMPDIR", tempfile.gettempdir())
        with tempfile.NamedTemporaryFile("wb", dir=tmp_dir, suffix=".wav", delete=False) as tf:
            wav_path = tf.name

        try:
            extract_audio_wav(video_path, window[0], window[1], wav_path)
            return self.encode_wav(wav_path)
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
