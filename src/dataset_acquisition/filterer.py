from pathlib import Path
from shared.social_presence import SocialPresenceDetector

class StreamingFilter:
    def __init__(self, model_path='yolov8n.pt'):
        self.detector = SocialPresenceDetector(model_path)

    def check_social_presence(self, video_path: Path, sample_rate_fps=1) -> bool:
        """
        Returns True if at least one person (other than POV) is detected.
        """
        return self.detector.detect(video_path, sample_rate_fps=sample_rate_fps, fast_mode=True)
