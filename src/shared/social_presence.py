import cv2
import gc
import torch
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path

class SocialPresenceDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model_path = model_path
        self._model = None
        self._mp_hands = None
        self._hands_detector = None

    @property
    def model(self):
        if self._model is None:
            # Lazy loading to save memory if not used
            print(f"[SocialPresenceDetector] Loading YOLO model: {self.model_path}")
            self._model = YOLO(self.model_path)
        return self._model

    @property
    def mp_hands(self):
        if self._hands_detector is None:
            print("[SocialPresenceDetector] Loading MediaPipe Hands model...")
            self._mp_hands = mp.solutions.hands
            self._hands_detector = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._hands_detector

    def unload(self):
        """ Explicitly unload the model and clear memory """
        if self._model is not None:
            print(f"[SocialPresenceDetector] Unloading YOLO model...")
            del self._model
            self._model = None
            
        if self._hands_detector is not None:
            print(f"[SocialPresenceDetector] Unloading MediaPipe Hands model...")
            self._hands_detector.close()
            self._hands_detector = None
            
        # Force garbage collection and clear MPS/CUDA cache
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, video_path: Path, sample_rate_fps=1, fast_mode=False, min_consistency=2, return_hands=False):
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
            all_hands = []
            detected_frames_count = 0
            
            batch_frames = []
            batch_timestamps = []
            BATCH_SIZE = 16  # Adjustable batch size based on memory

            # Use sequential reading instead of seeking for stability on macOS
            current_frame_idx = 0
            while True:
                ret, frame = cap.read()
                
                # Only process frames at the specified interval
                if ret and current_frame_idx % frame_interval == 0:
                    if frame is not None and frame.size > 0:
                        batch_frames.append(frame)
                        batch_timestamps.append(current_frame_idx / fps)
                
                current_frame_idx += 1
                is_end = not ret or current_frame_idx >= total_frames
                
                # Process batch if full or at end of video
                if len(batch_frames) >= BATCH_SIZE or (is_end and len(batch_frames) > 0):
                    # Use YOLO internal batching with ByteTrack
                    results = self.model.track(batch_frames, classes=[0], verbose=False, conf=0.5, batch=len(batch_frames), persist=True, tracker="bytetrack.yaml") 
                    
                    for i, result in enumerate(results):
                        timestamp = batch_timestamps[i]
                        frame_detections = []
                        has_bystander_in_frame = False
                        img_h, img_w = batch_frames[i].shape[:2]
                        
                        # MediaPipe Hand Detection
                        frame_rgb = cv2.cvtColor(batch_frames[i], cv2.COLOR_BGR2RGB)
                        hand_results = self.mp_hands.process(frame_rgb)
                        hand_boxes = []
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                x_min, y_min = img_w, img_h
                                x_max, y_max = 0, 0
                                for lm in hand_landmarks.landmark:
                                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                                    x_min, y_min = min(x_min, x), min(y_min, y)
                                    x_max, y_max = max(x_max, x), max(y_max, y)
                                pad = 20
                                hand_boxes.append((max(0, x_min - pad), max(0, y_min - pad), min(img_w, x_max + pad), min(img_h, y_max + pad)))

                        for box in result.boxes:
                            coords = [int(v) for v in box.xyxy[0].tolist()]
                            x1, y1, x2, y2 = coords
                            
                            # Refined Anti-Wearer Heuristic
                            is_limb = (y2 > 0.95 * img_h) and (y1 > 0.15 * img_h)
                            is_ghost = (y1 == 0) and (y2 > 0.90 * img_h)
                            
                            if is_limb or is_ghost:
                                continue

                            # MediaPipe Hand Overlap Suppression
                            is_hand = False
                            p_area = max(0, x2 - x1) * max(0, y2 - y1)
                            if p_area > 0:
                                for (hx1, hy1, hx2, hy2) in hand_boxes:
                                    ix1 = max(x1, hx1)
                                    iy1 = max(y1, hy1)
                                    ix2 = min(x2, hx2)
                                    iy2 = min(y2, hy2)
                                    if ix1 < ix2 and iy1 < iy2:
                                        i_area = (ix2 - ix1) * (iy2 - iy1)
                                        # Use 40% overlap threshold
                                        if (i_area / p_area) > 0.4:
                                            is_hand = True
                                            break
                            if is_hand:
                                continue

                            # Extract tracking ID if available
                            person_id = int(box.id[0]) if box.id is not None else len(frame_detections)

                            frame_detections.append({
                                "person_id": person_id,
                                "timestamp_sec": timestamp,
                                "bounding_box": coords,
                                "confidence": float(box.conf[0])
                            })
                            has_bystander_in_frame = True
                        
                        if has_bystander_in_frame:
                            detected_frames_count += 1
                        
                        if frame_detections or (return_hands and hand_boxes):
                            all_detections.append(frame_detections)
                            all_hands.append({
                                "timestamp_sec": timestamp,
                                "hand_boxes": hand_boxes
                            })
                    
                    batch_frames = []
                    batch_timestamps = []
                    
                    # Return early in fast mode
                    if fast_mode and (detected_frames_count >= min_consistency):
                        return True
                
                if is_end:
                    break
            
            if fast_mode:
                return detected_frames_count >= min_consistency
            if return_hands:
                return all_detections, all_hands
            return all_detections
        finally:
            cap.release()
