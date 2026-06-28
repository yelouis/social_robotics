"""
Draw helpers for bounding boxes and hand boxes.
"""
import cv2
from src.layer_05_visualizer.colors import hex_to_bgr

def draw_box(canvas, coords, hex_color: str, alpha: float, conf: float, label: str, dashed=False):
    x1, y1, x2, y2 = map(int, coords)
    color = hex_to_bgr(hex_color)
    
    if alpha < 1.0:
        overlay = canvas.copy()
        if dashed:
            # Draw dashed rectangle
            # Simplified: just draw normal rectangle with low alpha for phantoms
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    else:
        if dashed:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
    # Label chip
    text = f"{label} {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Chip bg
    chip_y2 = max(y1, th + baseline + 2)
    cv2.rectangle(canvas, (x1, chip_y2 - th - baseline - 2), (x1 + tw + 4, chip_y2), color, -1)
    
    # Text
    cv2.putText(canvas, text, (x1 + 2, chip_y2 - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

