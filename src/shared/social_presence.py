import cv2
import gc
import torch
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
            print(f"[SocialPresenceDetector] Loading YOLO model: {self.model_path}")
            self._model = YOLO(self.model_path)
        return self._model

    def unload(self):
        """ Explicitly unload the model and clear memory """
        if self._model is not None:
            print(f"[SocialPresenceDetector] Unloading YOLO model...")
            del self._model
            self._model = None
            
        # Force garbage collection and clear MPS/CUDA cache
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        try:
            if not cap.isOpened():
                return False if fast_mode else []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or total_frames == 0:
                return False if fast_mode else []

            frame_interval = int(max(1, fps / sample_rate_fps))
            
            all_detections = []
            detected_frames_count = 0
            
            # Use sequential reading instead of seeking for stability on macOS
            current_frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process frames at the specified interval
                if current_frame_idx % frame_interval == 0:
                    if frame is None or frame.size == 0:
                        current_frame_idx += 1
                        continue

                    timestamp = current_frame_idx / fps
                    # Use the specified device or let it auto-select (defaulting to MPS/CPU)
                    results = self.model(frame, classes=[0], verbose=False, conf=0.5) 
                    
                    frame_detections = []
                    has_bystander_in_frame = False
                    for result in results:
                        for box in result.boxes:
                            coords = [int(v) for v in box.xyxy[0].tolist()]
                            x1, y1, x2, y2 = coords
                            img_h, img_w = frame.shape[:2]
                            
                            # Refined Anti-Wearer Heuristic
                            is_limb = (y2 > 0.95 * img_h) and (y1 > 0.15 * img_h)
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
                    
                    if fast_mode and (detected_frames_count >= min_consistency):
                        return True
                        
                    if frame_detections:
                        all_detections.append(frame_detections)
                
                current_frame_idx += 1
                # Small safety break if we exceed total_frames (sometimes cap.read() returns True past total_frames)
                if current_frame_idx >= total_frames:
                    break
            
            if fast_mode:
                return detected_frames_count >= min_consistency
            return all_detections
        finally:
            cap.release()
