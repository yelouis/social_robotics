import cv2
from ultralytics import YOLO
from pathlib import Path

class StreamingFilter:
    def __init__(self, model_path='yolov8n.pt'):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"Loading YOLO model for streaming filter: {self.model_path}")
            self._model = YOLO(self.model_path)
        return self._model

    def check_social_presence(self, video_path: Path, sample_rate_fps=1) -> bool:
        """
        Returns True if at least one person (other than POV) is detected.
        """
        if not video_path.exists():
            return False

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Filter error: Could not open {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or total_frames == 0:
            cap.release()
            return False

        frame_interval = int(max(1, fps / sample_rate_fps))
        
        has_bystander = False
        
        # We only need to find ONE frame with a bystander to "keep" the video
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO
            results = self.model(frame, classes=[0], verbose=False) # class 0 is person
            
            for result in results:
                if len(result.boxes) > 0:
                    has_bystander = True
                    break
            
            if has_bystander:
                break
        
        cap.release()
        return has_bystander
