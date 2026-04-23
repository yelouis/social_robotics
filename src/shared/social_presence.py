import cv2
from ultralytics import YOLO
from pathlib import Path

class SocialPresenceDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Lazy loading to save memory if not used
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, video_path: Path, sample_rate_fps=1, fast_mode=False):
        """
        Detect persons in a video.
        
        Args:
            video_path: Path to the video file.
            sample_rate_fps: How many frames to sample per second.
            fast_mode: If True, returns True as soon as ONE person is detected.
                      If False, returns a list of all detections per frame.
        """
        if not video_path.exists():
            return False if fast_mode else []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False if fast_mode else []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or total_frames == 0:
            cap.release()
            return False if fast_mode else []

        frame_interval = int(max(1, fps / sample_rate_fps))
        
        all_detections = []
        has_bystander = False
        
        # In egocentric footage, we assume any detected 'person' is a bystander
        # because the camera wearer is behind the lens.
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            results = self.model(frame, classes=[0], verbose=False) # class 0 is person
            
            frame_detections = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    # Format: [x1, y1, x2, y2]
                    coords = [int(v) for v in box.xyxy[0].tolist()]
                    
                    # Anti-Wearer Heuristic:
                    # In egocentric video, limbs of the wearer often appear at the bottom/sides.
                    # We ignore detections that:
                    # 1. Touch the bottom edge (y2 > 95% height)
                    # 2. Start significantly below the top (y1 > 20% height) - i.e., no head/shoulders
                    x1, y1, x2, y2 = coords
                    img_h, img_w = frame.shape[:2]
                    
                    is_wearer = (y2 > 0.95 * img_h) and (y1 > 0.15 * img_h)
                    
                    if is_wearer:
                        continue

                    frame_detections.append({
                        "timestamp_sec": timestamp,
                        "bounding_box": coords,
                        "confidence": conf
                    })
                    has_bystander = True
            
            if fast_mode and has_bystander:
                cap.release()
                return True
                
            if frame_detections:
                all_detections.append(frame_detections)
        
        cap.release()
        
        if fast_mode:
            return has_bystander
        return all_detections
