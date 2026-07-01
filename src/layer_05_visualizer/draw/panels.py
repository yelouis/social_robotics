"""
Draw helpers for panels and timelines.
"""
import cv2

def draw_climax_flash(canvas, W, H, task):
    # Flash effect
    cv2.rectangle(canvas, (0, 0), (W, H), (255, 255, 255), 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "CLIMAX", (W // 2 - 50, H // 8), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# Canonical layer order + display names (always shown, even when a layer is
# absent/null, so the operator can see every layer was considered).
_CANON_LAYERS = [
    ("03a", "attention"),
    ("03b", "emotion"),
    ("03c", "prosody"),
    ("03d", "proxemic"),
    ("03e", "gesture"),
    ("03f", "motor"),
]

# BGR colors for the three per-layer states.
_C_FINDING = (120, 255, 120)   # bright green — an active signal at t
_C_RAN = (200, 200, 200)       # gray — layer ran but nothing at this instant
_C_ABSENT = (110, 110, 110)    # dim — layer did not run for this clip


def _within(t, w):
    return w and len(w) == 2 and w[0] <= t <= w[1]


def _layer_status(index, layer_id, t):
    """Return (text, color) for one layer at time t — always something, so a
    null/inactive layer is still annotated. """
    if layer_id not in index.layers_present:
        return "not run for this clip", _C_ABSENT

    if layer_id == "03a":
        vals = [(trk.id, trk.attention.get("summary", {})) for trk in index.tracks if trk.attention]
        if vals:
            pid, s = vals[0]
            return (f"P{pid} attn {s.get('average_attention_score', 0):.2f} "
                    f"target:{s.get('gaze_target_classification', '?')}"), _C_FINDING
        return "no bystander face tracked", _C_RAN

    if layer_id == "03c":
        act = [a for a in index.audio if _within(t, a.get("window_sec"))]
        a = act[0] if act else (index.audio[0] if index.audio else None)
        if a is None:
            return "no reaction window", _C_RAN
        if not a.get("audio_present", False):
            return "no audio", _C_RAN
        return (f"tone:{a.get('classified_acoustic_tone', '?')} "
                f"emo:{a.get('dominant_emotion', '?')}"), (_C_FINDING if act else _C_RAN)

    # 03b / 03d / 03e / 03f live in the per-person `windows`.
    active, ran = [], False
    for trk in index.tracks:
        for w in trk.windows:
            if w.get("layer") != layer_id:
                continue
            ran = True
            if _within(t, w.get("window_sec")):
                active.append((trk.id, w))
    if active:
        pid, w = active[0]
        if layer_id == "03b":
            pair = w.get("transition_pair", ["?", "?"])
            return f"P{pid} {pair[0]}->{pair[1]} ({w.get('classified_direction', '?')})", _C_FINDING
        if layer_id == "03d":
            return (f"P{pid} {w.get('classified_action', '?')} "
                    f"conf {w.get('proxemic_confidence', 0):.2f} "
                    f"[{w.get('proxemic_trajectory_shape', '-')}]"), _C_FINDING
        if layer_id == "03e":
            return f"P{pid} {w.get('gesture_detected', 'none')} {w.get('pitch_oscillation_hz', 0):.2f}Hz", _C_FINDING
        if layer_id == "03f":
            fl = "FLINCH" if w.get("motor_resonance_detected") else "no flinch"
            mr = " +mirror" if w.get("mirroring_detected") else ""
            return f"P{pid} {fl}{mr} vel {w.get('bystander_pose_velocity_peak', 0):.1f}", _C_FINDING
    return ("ran; no window at this instant" if ran else "no measurable signal"), _C_RAN


def draw_readout_panel(canvas, index, t, ego):
    H, W = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, H / 1400.0)          # scale text to frame height
    lh = int(26 * fs / 0.5)            # line height
    pad = 12
    n_lines = 2 + len(_CANON_LAYERS)   # header + ego + 6 layers
    panel_w = max(360, int(W * 0.34))
    panel_h = pad * 2 + lh * n_lines

    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    x, y = 20, 10 + pad + lh - 6
    cv2.putText(canvas, f"t={t:.2f}s", (x, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    ego_txt = f"ego-chaos {ego:.2f}" if ego is not None else "ego-chaos --"
    cv2.putText(canvas, ego_txt, (x + int(panel_w * 0.45), y), font, fs, (0, 165, 255), 1, cv2.LINE_AA)

    for lid, name in _CANON_LAYERS:
        y += lh
        text, color = _layer_status(index, lid, t)
        label = f"{lid} {name:<9}"
        cv2.putText(canvas, label, (x, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (x + int(panel_w * 0.30), y), font, fs, color, 1, cv2.LINE_AA)

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
