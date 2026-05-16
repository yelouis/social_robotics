from pathlib import Path
from shared.social_presence import SocialPresenceDetector

class StreamingFilter:
    def __init__(self, model_path=None):
        # Model resolution is delegated to SocialPresenceDetector, which reads
        # `social_presence_pose` from the tier-per-host registry when no
        # explicit path is passed.
        self.detector = SocialPresenceDetector(model_path)

    def check_social_presence(self, video_path: Path, sample_rate_fps=1) -> bool:
        """
        Returns True if at least one person (other than POV) is detected.
        """
        return self.detector.detect(video_path, sample_rate_fps=sample_rate_fps, fast_mode=True)
