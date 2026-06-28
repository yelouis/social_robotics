"""
Draw helpers for panels and timelines.
"""
import cv2

def draw_climax_flash(canvas, W, H, task):
    # Flash effect
    cv2.rectangle(canvas, (0, 0), (W, H), (255, 255, 255), 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "CLIMAX", (W // 2 - 50, H // 8), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

def draw_readout_panel(canvas, index, t, ego):
    H, W = canvas.shape[:2]
    overlay = canvas.copy()
    
    # Draw translucent box top-left
    panel_w, panel_h = max(300, W // 4), max(200, H // 4)
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    y = 30
    cv2.putText(canvas, f"t={t:.2f}s", (20, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if ego is not None:
        cv2.putText(canvas, f"ego-chaos: {ego:.2f}", (120, y), font, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        
    y += 20
    for track in index.tracks:
        if track.attention:
            att = track.attention.get("summary", {})
            score = att.get("average_attention_score", 0.0)
            target = att.get("gaze_target_classification", "Unknown")
            cv2.putText(canvas, f"P{track.id} attn {score:.2f} target:{target}", (20, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)
            y += 20

def draw_timeline_strip(canvas, index, t):
    H, W = canvas.shape[:2]
    # Draw translucent strip bottom
    overlay = canvas.copy()
    strip_h = 80
    cv2.rectangle(overlay, (0, H - strip_h), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
    
    # Just draw a simple playhead and axis line for now
    cv2.line(canvas, (20, H - strip_h // 2), (W - 20, H - strip_h // 2), (255, 255, 255), 1)
    
    # Assuming duration ~ 100s for a dummy projection (the real renderer should pass duration to this)
    # We'll just draw the time string at the bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "TIMELINE", (20, H - strip_h + 20), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Playhead - pseudo position
    px = 20 + int(t * 10) % (W - 40)
    cv2.line(canvas, (px, H - strip_h), (px, H), (0, 0, 255), 2)
