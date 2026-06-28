"""
Draw helpers for windowed badges (emotion, proxemic, gesture, motor).
"""
import cv2
import math

def draw_window_badge(canvas, box_coords, w, t):
    x1, y1, x2, y2 = map(int, box_coords)
    kind = w.get("kind")
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    if kind == "emotion_slice":
        # 03b Emotion
        pair = w.get("transition_pair", ["unknown", "unknown"])
        text = f"{pair[0]}->{pair[1]}"
        direction = w.get("classified_direction", "neutral")
        
        color = (200, 200, 200) # neutral (gray)
        if direction == "approving":
            color = (0, 255, 0) # green
        elif direction in ("skeptical", "negative"):
            color = (0, 0, 255) # red
            
        cv2.putText(canvas, text, (x1, max(20, y1 - 25)), font, font_scale, color, thickness, cv2.LINE_AA)
        
    elif kind == "proxemic":
        # 03d Proxemic
        action = w.get("classified_action", "Neutral")
        if action != "Neutral":
            text = f"Prox: {action}"
            cv2.putText(canvas, text, (x1, y2 + 20), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            
    elif kind == "gesture":
        # 03e Gesture
        gesture = w.get("gesture_detected", "none")
        if gesture != "none":
            hz = w.get("pitch_oscillation_hz", 0.0)
            # Pulsing effect
            scale = 1.0 + 0.3 * math.sin(2 * math.pi * hz * t)
            
            text = f"{gesture} {hz:.1f}Hz"
            cv2.putText(canvas, text, (x1, y1 + 30), font, font_scale * scale, (255, 100, 100), thickness, cv2.LINE_AA)
            
    elif kind == "motor_resonance":
        # 03f Motor Resonance
        flinch = w.get("motor_resonance_detected", False)
        mirror = w.get("mirroring_detected", False)
        
        y_offset = y2 + 40
        if flinch:
            # flash border (draw thick red box)
            cv2.rectangle(canvas, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 3)
            cv2.putText(canvas, "FLINCH", (x1, y_offset), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            y_offset += 20
        if mirror:
            cv2.putText(canvas, "MIRROR", (x1, y_offset), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
