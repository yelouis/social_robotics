"""Tunable thresholds for the 03c Acoustic Prosody layer.

All seven heuristic constants previously baked into pipeline.py live here as
fields on a frozen dataclass so empirical tuning becomes data, not code.
Override any field via the constructor for tests or A/B runs:

    pipeline = AcousticProsodyPipeline(
        manifest, output,
        config=Layer03cConfig(high_volume_dbfs=-15.0),
    )
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Layer03cConfig:
    # Volume cutoffs for the alarming/discouraging heuristics (dBFS).
    high_volume_dbfs: float = -20.0
    low_volume_dbfs: float = -35.0

    # Additive bonuses applied to the alarming/discouraging scores when the
    # corresponding volume cutoff is crossed.
    high_volume_bonus: float = 0.3
    low_volume_bonus: float = 0.3

    # Multiplier on pitch_contour_variance when computing the soothing score.
    pitch_variance_soothing_weight: float = 0.5

    # Minimum dominant-tone score required for a non-Neutral classification.
    min_dominant_tone_score: float = 0.3

    # SER dominant-emotion confidence below which SenseVoice runs.
    sensevoice_confidence_threshold: float = 0.6

    # Divisor used to normalize raw pitch variance into [0, 1].
    pitch_variance_normalization: float = 10000.0
