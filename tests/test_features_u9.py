from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import torch

from features.visual import FrameEncoder, get_u9_sample_times


def test_u9_sample_times():
    """Verify 9-frame uniform sampling at t_k = k * D / 8 for k=0..8."""
    times = get_u9_sample_times(16.0)
    expected = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    assert len(times) == 9
    np.testing.assert_allclose(times, expected)

    times_odd = get_u9_sample_times(10.0)
    assert len(times_odd) == 9
    assert times_odd[0] == 0.0
    assert times_odd[-1] == 10.0
    assert times_odd[4] == 5.0


def test_u9_encoder_initialization():
    encoder = FrameEncoder(encoder_id="siglip-b16-224-u9")
    assert encoder.encoder_id == "siglip-b16-224-u9"


def test_encode_frames_mocked():
    """Verify that encode_frames returns (768,) float32 with per-frame L2 norm then mean pool."""
    encoder = FrameEncoder(encoder_id="siglip-b16-224-u9", device="cpu")

    # Mock processor and model
    fake_images = [Image.new("RGB", (224, 224), color=(i * 10, i * 10, i * 10)) for i in range(9)]

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"pixel_values": torch.zeros((9, 3, 224, 224))}
    mock_processor = MagicMock()
    mock_processor.return_value = mock_inputs

    mock_model = MagicMock()
    # Unnormalized 768-d features for 9 frames
    raw_feats = torch.randn(9, 768)
    mock_model.get_image_features.return_value = raw_feats

    encoder._processor = mock_processor
    encoder._model = mock_model

    feat = encoder.encode_frames(fake_images)
    assert feat.shape == (768,)
    assert feat.dtype == np.float32

    # Verify L2 normalization per frame followed by mean
    expected_normed = raw_feats / raw_feats.norm(p=2, dim=-1, keepdim=True)
    expected_mean = expected_normed.mean(dim=0).numpy().astype(np.float32)
    np.testing.assert_allclose(feat, expected_mean, atol=1e-6)
