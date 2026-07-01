"""
Offline cv2 burn-in renderer.
"""
import cv2
import subprocess
from pathlib import Path
import bisect

from src.layer_05_visualizer.draw.boxes import draw_box
from src.layer_05_visualizer.draw.gaze import draw_gaze_arrow, draw_engagement_ring
from src.layer_05_visualizer.draw.emotion import draw_window_badge
from src.layer_05_visualizer.draw.panels import draw_climax_flash, draw_readout_panel, draw_timeline_strip

CLIMAX_FLASH_SEC = 0.5

class RenderIndex:
    def __init__(self, bundle, layers, people, show_phantoms):
        if layers == "all":
            self.layers = set(bundle.get("layers_present", []))
        else:
            self.layers = set(layers.split(",")) if isinstance(layers, str) else set(layers)

        # Which layers actually RAN for this clip (independent of the --layers
        # draw filter) + the 03c audio blocks — both consumed by the per-layer
        # readout panel so a layer that found nothing is still shown as such.
        self.layers_present = set(bundle.get("layers_present", []))
        self.audio = bundle.get("audio", [])
        self.show_phantoms = show_phantoms
        
        self.tracks = []
        for trk in bundle.get("tracks", []):
            pid = trk["person_id"]
            if people and str(pid) not in people:
                continue
            
            pinfo = bundle.get("people", {}).get(str(pid), {})
            is_phantom = pinfo.get("phantom", False)
            if is_phantom and not show_phantoms:
                continue
                
            class TrackWrapper:
                def __init__(self, t_dict, c_hex, is_p):
                    self.id = t_dict["person_id"]
                    self.boxes = t_dict.get("boxes", [])
                    self.gap_tolerance_sec = t_dict.get("gap_tolerance_sec", 4.0)
                    self.attention = t_dict.get("attention")
                    self.windows = t_dict.get("windows", [])
                    self.color = c_hex
                    self.is_phantom = is_p
            
            self.tracks.append(TrackWrapper(trk, pinfo.get("color", "#FFFFFF"), is_phantom))
            
        self.tasks = []
        for t in bundle.get("tasks", []):
            class TaskWrapper:
                def __init__(self, t_dict):
                    self.id = t_dict.get("task_id")
                    self.climax_sec = t_dict.get("climax_sec", 0.0)
                    self.reaction_windows = t_dict.get("reaction_windows_sec", [])
                    self.ego_chaos = t_dict.get("ego_kinetic_chaos_score", 0.0)
            self.tasks.append(TaskWrapper(t))

def build_render_index(bundle, layers, people, show_phantoms):
    return RenderIndex(bundle, layers, people, show_phantoms)

def last_sample_at_or_before(sorted_list, t):
    if not sorted_list: return None
    keys = [x["t"] for x in sorted_list]
    idx = bisect.bisect_right(keys, t) - 1
    if idx < 0: return None
    return sorted_list[idx]

def nearest_sample(sorted_list, t):
    if not sorted_list: return None
    keys = [x["t"] for x in sorted_list]
    idx = bisect.bisect_left(keys, t)
    if idx == 0: return sorted_list[0]
    if idx == len(sorted_list): return sorted_list[-1]
    
    before = sorted_list[idx - 1]
    after = sorted_list[idx]
    if (t - before["t"]) < (after["t"] - t):
        return before
    return after

def within(t, window_sec):
    if not window_sec or len(window_sec) != 2:
        return False
    return window_sec[0] <= t <= window_sec[1]

def active_ego_chaos(tasks, t):
    for task in tasks:
        for w in task.reaction_windows:
            if within(t, w):
                return task.ego_chaos
    return None

def compose_frame(frame, index, t, W, H, panels):
    canvas = frame.copy()

    for track in index.tracks:
        s = last_sample_at_or_before(track.boxes, t)
        if s is None: continue
        age = t - s["t"]
        if age > track.gap_tolerance_sec: continue
        
        # fade
        alpha = max(0.3, 1.0 - (age / track.gap_tolerance_sec) * 0.7)
        
        # denorm
        b = s["box"]
        x1 = int(b[0] * W)
        y1 = int(b[1] * H)
        x2 = int(b[2] * W)
        y2 = int(b[3] * H)
        
        draw_box(canvas, (x1, y1, x2, y2), track.color, alpha, conf=s.get("conf", 1.0), label=f"P{track.id}", dashed=track.is_phantom)
        
        face = (int((x1+x2)/2), y1)
        
        if "03a" in index.layers and track.attention:
            g = nearest_sample(track.attention["trace"], t)
            if g:
                draw_gaze_arrow(canvas, face, g.get("pitch_rad", 0), g.get("yaw_rad", 0), g.get("score", 0), g.get("target", ""), color_hex=track.color)
            draw_engagement_ring(canvas, (x1,y1,x2,y2), track.attention["summary"])

        for w in track.windows:
            if w.get("layer") in index.layers and within(t, w.get("window_sec")):
                draw_window_badge(canvas, (x1,y1,x2,y2), w, t)

    for task in index.tasks:
        if abs(t - task.climax_sec) <= CLIMAX_FLASH_SEC:
            draw_climax_flash(canvas, W, H, task)
            
    ego = active_ego_chaos(index.tasks, t)

    if "readout" in panels: draw_readout_panel(canvas, index, t, ego)
    if "timeline" in panels: draw_timeline_strip(canvas, index, t)
    
    return canvas

def mux_audio(tmp_silent, video_path, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_silent), "-i", str(video_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def render_clip(bundle, video_path, out_path, *, scale=1.0, layers="all",
                people=None, show_phantoms=False, panels=("timeline","readout"),
                with_audio=True, clip_range=None):
    
    out_path = Path(out_path)
    video_path = Path(video_path)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
        
    src_fps = bundle["clip"].get("fps") or cap.get(cv2.CAP_PROP_FPS)
    in_w = bundle["clip"]["native_width"]
    in_h = bundle["clip"]["native_height"]
    out_w = round(in_w * scale)
    out_h = round(in_h * scale)

    index = build_render_index(bundle, layers, people, show_phantoms)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_silent = out_path.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(str(tmp_silent), fourcc, src_fps, (out_w, out_h))

    frame_idx = 0
    start_t = clip_range[0] if clip_range else 0.0
    end_t = clip_range[1] if clip_range else float('inf')
    
    # fast forward
    if start_t > 0:
        frame_idx = int(start_t * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t = frame_idx / src_fps
        if t > end_t:
            break
            
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            
        canvas = compose_frame(frame, index, t, out_w, out_h, panels)
        writer.write(canvas)
        frame_idx += 1

    writer.release()
    cap.release()

    has_audio = bundle["clip"].get("has_audio", False)
    if with_audio and has_audio:
        try:
            mux_audio(tmp_silent, video_path, out_path)
            tmp_silent.unlink()
        except subprocess.CalledProcessError:
            # fallback if mux fails
            tmp_silent.rename(out_path)
    else:
        tmp_silent.rename(out_path)
