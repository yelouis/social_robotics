import cv2
import numpy as np
import os

# COCO Keypoint Pairs for skeleton drawing (YOLOv8-Pose default)
# 0: nose, 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 
# 9: l_wrist, 10: r_wrist, 11: l_hip, 12: r_hip
SKELETON_PAIRS = [
    (0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12), # Upper body
    (5, 7), (7, 9), (6, 8), (8, 10) # Arms
]

def load_class_map(path="/Volumes/Extreme SSD/charades_ego_data/annotations/CharadesEgo/Charades_v1_classes.txt"):
    """Loads Charades class ID to readable label mapping."""
    class_map = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                # Format: c000 Description
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_map[parts[0]] = parts[1]
    return class_map

class DebugOverlay:
    def __init__(self, class_map_path=None):
        if class_map_path:
            self.class_map = load_class_map(class_map_path)
        else:
            self.class_map = load_class_map()

    def draw_skeleton(self, frame, keypoints, confidence_threshold=0.3):
        """Draws the pose skeleton on the frame."""
        for p1_idx, p2_idx in SKELETON_PAIRS:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                p1 = keypoints[p1_idx]
                p2 = keypoints[p2_idx]
                # p1, p2 are [x, y, conf]
                if p1[2] > confidence_threshold and p2[2] > confidence_threshold:
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
        
        # Draw joints
        for i, kp in enumerate(keypoints):
            if kp[2] > confidence_threshold:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

    def annotate_frame(self, frame, action_labels, vlm_state, vlm_conf, keypoints=None):
        """
        Adds all visual debug information to the frame.
        action_labels: semicolon-separated string of annotations (e.g., 'c156 3.90 12.00;...')
        """
        if keypoints is not None:
            self.draw_skeleton(frame, keypoints)

        # Parse and draw action labels
        ids = [part.split(' ')[0] for part in action_labels.split(';') if part]
        readable_labels = [self.class_map.get(cid, cid) for cid in ids]
        
        # Draw VLM Rating
        y = 40
        header_color = (255, 255, 255)
        text_color = (0, 255, 255)
        
        # Background for contrast
        cv2.rectangle(frame, (5, 5), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (400, 150), (255, 255, 255), 1)

        cv2.putText(frame, f"STATE: {vlm_state}", (15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30
        cv2.putText(frame, f"CONFIDENCE: {vlm_conf:.2f}", (15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, header_color, 1)
        y += 30
        cv2.putText(frame, "ANNOTATED ACTIONS:", (15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, header_color, 1)
        
        for label in readable_labels[:3]: # Limit to top 3 for UI space
            y += 20
            cv2.putText(frame, f"> {label}", (25, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    def generate_side_by_side(self, ego_frame, tp_frame):
        """Combines ego and third-person views side-by-side."""
        h1, w1 = ego_frame.shape[:2]
        h2, w2 = tp_frame.shape[:2]
        
        # Standardize height to 480px
        new_h = 480
        ego_resized = cv2.resize(ego_frame, (int(w1 * new_h / h1), new_h))
        tp_resized = cv2.resize(tp_frame, (int(w2 * new_h / h2), new_h))
        
        # Add labels to the combined frame
        final_frame = np.hstack((ego_resized, tp_resized))
        
        # Add overlay labels for views
        cv2.putText(final_frame, "EGOCENTRIC", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(final_frame, "THIRD-PERSON", (ego_resized.shape[1] + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return final_frame

def export_debug_video(frames, output_path):
    """Encodes a list of frames into a video file."""
    if not frames:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (w, h))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Debug video exported to {output_path}")

if __name__ == "__main__":
    # Smoke test with black frames
    overlay = DebugOverlay()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay.annotate_frame(black_frame, "c106 1.0 5.0", "FOCUSED", 0.95)
    cv2.imwrite("debug_test.jpg", black_frame)
    print("Test frame saved to debug_test.jpg")
