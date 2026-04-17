import os
import cv2
import shutil
import pandas as pd

def extract_frames(video_path, output_frames_dir, t_start, t_end, fps_sample=2):
    """
    Extracts frames from a video within a specific time window.
    """
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24.0 # Default fallback
        
    frame_interval = int(fps / fps_sample)
    if frame_interval < 1:
        frame_interval = 1
        
    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    extracted_paths = []
    
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if (frame_count - start_frame) % frame_interval == 0:
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_paths.append(frame_path)
            
        frame_count += 1
        
    cap.release()
    return extracted_paths

def cleanup_frames(frames_dir):
    """
    Deletes the extracted frames directory to prevent SSD wear and save space.
    """
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        print(f"Cleaned up frames in {frames_dir}")

class IngestNode:
    def __init__(self, manifest_path):
        self.manifest = pd.read_csv(manifest_path)
        
    def get_video_pair(self, ego_id):
        row = self.manifest[self.manifest['ego_video_id'] == ego_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def process_view(self, ego_id, view="tp", t_start=None, t_end=None, frames_dir="/Volumes/Extreme SSD/charades_ego_data/tmp_frames"):
        """
        Extracts frames for a specific view (ego or tp) and returns path list.
        """
        pair = self.get_video_pair(ego_id)
        if not pair:
            print(f"Error: Pair for {ego_id} not found.")
            return []
            
        video_path = pair['ego_video_path'] if view == "ego" else pair['tp_video_path']
        
        # Use provided window or full video length
        start = t_start if t_start is not None else 0
        end = t_end if t_end is not None else pair['length']
        
        specific_frames_dir = os.path.join(frames_dir, f"{ego_id}_{view}")
        return extract_frames(video_path, specific_frames_dir, start, end)

if __name__ == "__main__":
    # Example usage
    # node = IngestNode("/Volumes/Extreme SSD/charades_ego_data/annotations/paired_clips_manifest.csv")
    # frames = node.process_view("ABCDE", view="tp", t_start=5.0, t_end=10.0)
    # ... process frames ...
    # cleanup_frames("/Volumes/Extreme SSD/charades_ego_data/tmp_frames/ABCDE_tp")
    pass
