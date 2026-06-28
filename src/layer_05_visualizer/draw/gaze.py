"""
Draw helpers for gaze and attention.
"""
import cv2
import math
from src.layer_05_visualizer.colors import hex_to_bgr

def draw_gaze_arrow(canvas, face_pt, pitch_rad, yaw_rad, score, target, color_hex):
    color = hex_to_bgr(color_hex)
    face_x, face_y = face_pt
    
    L = 50 # Arrow length
    dx = int(-L * math.sin(yaw_rad) * math.cos(pitch_rad))
    dy = int(-L * math.sin(pitch_rad))
    
    tip = (face_x + dx, face_y + dy)
    cv2.arrowedLine(canvas, face_pt, tip, color, 2, tipLength=0.3)
    
def draw_engagement_ring(canvas, box_coords, summary_dict):
    x1, y1, x2, y2 = map(int, box_coords)
    face_x = int((x1 + x2) / 2)
    face_y = y1
    
    score = summary_dict.get("average_attention_score", 0.0)
    target = summary_dict.get("gaze_target_classification", "Unknown")
    
    # Lerp red->green by score
    r = int(255 * (1 - score))
    g = int(255 * score)
    color = (0, g, r) # BGR
    
    radius = max(20, int((x2 - x1) * 0.3))
    thickness = max(1, int(4 * score))
    
    cv2.circle(canvas, (face_x, face_y), radius, color, thickness, cv2.LINE_AA)
    
    # Label target
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, target, (face_x + radius + 5, face_y), font, 0.4, color, 1, cv2.LINE_AA)

