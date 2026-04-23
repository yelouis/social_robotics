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

    def detect(self, video_path: Path, sample_rate_fps=1, fast_mode=False, min_consistency=2):
        """
        Detect persons in a video.
        
        Args:
            video_path: Path to the video file.
            sample_rate_fps: How many frames to sample per second.
            fast_mode: If True, returns True as soon as 'min_consistency' person frames are detected.
                      If False, returns a list of all detections per frame.
            min_consistency: Number of frames with social presence required to confirm (default 2).
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
        detected_frames_count = 0
        
        # In egocentric footage, we assume any detected 'person' is a bystander
        # because the camera wearer is behind the lens.
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            # Higher confidence threshold (0.5) to avoid weak false positives
            results = self.model(frame, classes=[0], verbose=False, conf=0.5) 
            
            frame_detections = []
            has_bystander_in_frame = False
            for result in results:
                for box in result.boxes:
                    coords = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1, x2, y2 = coords
                    img_h, img_w = frame.shape[:2]
                    
                    # Refined Anti-Wearer Heuristic:
                    # 1. Limb: Touches bottom AND starts significantly below top.
                    is_limb = (y2 > 0.95 * img_h) and (y1 > 0.15 * img_h)
                    
                    # 2. Ghost Torso: Full height box starting exactly at top (y1=0).
                    # Real bystanders usually have some top-margin unless they are extremely close.
                    is_ghost = (y1 == 0) and (y2 > 0.90 * img_h)
                    
                    if is_limb or is_ghost:
                        continue

                    frame_detections.append({
                        "timestamp_sec": timestamp,
                        "bounding_box": coords,
                        "confidence": float(box.conf[0])
                    })
                    has_bystander_in_frame = True
            
            if has_bystander_in_frame:
                detected_frames_count += 1
                has_bystander = True
            
            if fast_mode and (detected_frames_count >= min_consistency):
                cap.release()
                return True
                
            if frame_detections:
                all_detections.append(frame_detections)
        
        cap.release()
        
        if fast_mode:
            return detected_frames_count >= min_consistency
        return all_detections
