import os
import json
import cv2
import gc
import torch
import traceback
from pathlib import Path
from shared.social_presence import SocialPresenceDetector
try:
    from src.models_config import get_model
except ImportError:
    from models_config import get_model
import config


class FilteringPipeline:
    def __init__(self, input_manifest_path, output_manifest_path, force=False, skip_vlm=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_manifest_path = Path(output_manifest_path)
        self.error_log_path = self.output_manifest_path.parent / "02_filter_errors.json"
        self.force = force
        self.skip_vlm = skip_vlm
        
        # Shared components. Default to YOLO-pose + VLM-gated verification
        # (Resolved Issue #22) so the filtering stage rejects the wearer's own
        # limbs / equipment that the bbox-only yolov8n previously kept.
        # Hosts that need to disable the per-candidate-frame VLM round-trip
        # (e.g. throughput-sensitive overnight backfills) opt out via
        # SAF_VLM_VERIFY_SOCIAL=0.
        vlm_verify = os.getenv("SAF_VLM_VERIFY_SOCIAL", "1").lower() in ("1", "true", "yes")
        self.detector = SocialPresenceDetector(get_model("social_presence_pose"), vlm_verify=vlm_verify)

        # Stage-2 climax-refinement VLM is selected from the central tier
        # registry (`filtering_vlm`): `medium`/`large` -> qwen2.5vl:7b on a
        # >=48 GB host, `small` -> qwen2.5vl:3b for the 24 GB Mac mini.
        # SAF_VLM_MODEL_TIER=small is the legacy per-stage override; it still
        # forces the small tier on this layer alone when set.
        if os.getenv("SAF_VLM_MODEL_TIER", "").lower() == "small":
            self.vlm_model = get_model("filtering_vlm", tier="small")
        else:
            self.vlm_model = get_model("filtering_vlm")

        # Gate the single-pass interleaved architecture on host memory. The
        # two-pass path was built for the 24 GB Mac mini where YOLO and the
        # VLM could not co-reside; on a >=48 GB host they fit together, so we
        # interleave per video and read each file off the SSD once, not twice.
        try:
            import psutil
            self.single_pass = psutil.virtual_memory().total >= 48 * 2**30
        except ImportError:
            self.single_pass = False
        arch = "single-pass interleaved" if self.single_pass else "two-pass"
        print(f"Filtering architecture: {arch} (VLM: {self.vlm_model})")

        # Pass storage
        self.initial_registry = [] # For Pass 1
        self.intermediate_results = [] # Pass 1 output
        
        self.processed_ids = set()
        if self.output_manifest_path.exists() and not self.force:
            try:
                with open(self.output_manifest_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing manifest: {e}. Starting fresh.")
        
        # Load Ego4D Metadata
        self.metadata = self._load_metadata()

    # Only Ego4D has a metadata-driven task-labeling path implemented today
    # (see contextual_task_labeling). Other ego datasets like Charades-Ego,
    # EPIC-KITCHENS, and EgoProceL are deferred until their per-dataset
    # annotation parsers are written; in the interim we skip them at intake
    # so they don't burn YOLO time and then silently disappear at the
    # metadata stage with no entry in filtered_manifest.json.
    _SUPPORTED_DATASETS = ("ego4d",)

    def _is_supported_dataset(self, entry, video_id) -> bool:
        dataset = (entry.get('dataset') or '').strip().lower()
        if dataset in self._SUPPORTED_DATASETS:
            return True
        print(
            f"Skipping {video_id}: dataset '{dataset}' is not yet supported by "
            f"the metadata-driven labeling path (supported: {self._SUPPORTED_DATASETS})."
        )
        return False

    def _load_metadata(self):
        """ Load Ego4D metadata and index by video_uid """
        metadata_path = config.EGO4D_METADATA_PATH
        if metadata_path.exists():
            print(f"Loading Ego4D metadata from {metadata_path}...")
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    videos = data.get('videos', [])
                    return {v['video_uid']: v for v in videos}
            except Exception as e:
                print(f"Error loading metadata: {e}")
        else:
            print(f"Metadata file not found at {metadata_path}")
        return {}

        
    def cleanup_yolo(self):
        """ Explicitly free YOLO memory before VLM pass """
        if hasattr(self, 'detector'):
            self.detector.unload()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            self.initial_registry = json.load(f)

        final_results = []
        # Load existing results to append to them if not forcing
        if self.output_manifest_path.exists() and not self.force:
            try:
                with open(self.output_manifest_path, 'r') as f:
                    final_results = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in final_results}
            except:
                pass

        if self.single_pass:
            self._run_single_pass(final_results)
        else:
            self._run_two_pass(final_results)

        print(f"Final count: {len(final_results)} videos in manifest.")

    def _run_two_pass(self, final_results):
        """ Two-pass fallback for <48 GB hosts: run YOLO across all videos,
        unload it, then run the VLM across the survivors. Prevents YOLO and
        Qwen2.5-VL from co-residing in unified memory on memory-constrained
        hosts (e.g. the 24 GB Mac mini M4 Pro). """
        # --- PASS 1: Social Filter (YOLO) ---
        print("\n--- PASS 1: Social Filter (YOLO) ---")
        pass1_queue = []
        for entry in self.initial_registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            if not self._is_supported_dataset(entry, video_id):
                continue

            video_path = Path(entry['file_path'])
            if not video_path.exists(): continue

            print(f"Social Filter: {video_id}")
            bystander_detections, hand_detections = self.social_presence_filter(video_path)
            if bystander_detections:
                entry['bystander_detections'] = bystander_detections
                entry['hand_detections'] = hand_detections
                pass1_queue.append(entry)
            else:
                print(f"Dropped {video_id}: No social presence.")

        # --- MEMORY CLEANUP ---
        self.cleanup_yolo()

        # --- PASS 2: Task Labeling & Climax (VLM) ---
        print("\n--- PASS 2: Task Labeling & Climax (VLM) ---")
        for entry in pass1_queue:
            video_id = entry.get('id', entry.get('video_id'))
            print(f"Task Refinement: {video_id}")
            try:
                result = self.process_video_vlm_pass(entry)
                if result:
                    final_results.append(result)
                    self.processed_ids.add(video_id)

                    # Incremental save
                    with open(self.output_manifest_path, 'w') as f:
                        json.dump(final_results, f, indent=4)
            except Exception as e:
                self.log_error(video_id, e)

    def _run_single_pass(self, final_results):
        """ Single-pass interleaved architecture for >=48 GB hosts. YOLO and
        Qwen2.5-VL co-reside in unified memory, so each video is read off the
        SSD once: the YOLO social filter is immediately followed by the VLM
        task/climax pass, rather than reading the whole dataset twice. """
        print("\n--- SINGLE-PASS: Interleaved Social Filter + Task/Climax (YOLO + VLM) ---")
        for entry in self.initial_registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            if not self._is_supported_dataset(entry, video_id):
                continue

            video_path = Path(entry['file_path'])
            if not video_path.exists(): continue

            print(f"Social Filter: {video_id}")
            try:
                bystander_detections, hand_detections = self.social_presence_filter(video_path)
                if not bystander_detections:
                    print(f"Dropped {video_id}: No social presence.")
                    continue

                entry['bystander_detections'] = bystander_detections
                entry['hand_detections'] = hand_detections

                print(f"Task Refinement: {video_id}")
                result = self.process_video_vlm_pass(entry)
                if result:
                    final_results.append(result)
                    self.processed_ids.add(video_id)

                    # Incremental save
                    with open(self.output_manifest_path, 'w') as f:
                        json.dump(final_results, f, indent=4)
            except Exception as e:
                self.log_error(video_id, e)

    def process_video_vlm_pass(self, entry):
        """ Pass 2: Labeling and Climax Identification """
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['file_path'])
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        
        # 2. Contextual Task Labeling
        identified_tasks = self.contextual_task_labeling(video_id, duration_sec)
        if not identified_tasks:
            cap.release()
            return None

        # 3. Temporal Task Climax — deferred to Layer 03 (see Resolved Issue
        # #8). Layer 02 no longer seeks through every frame for the optical
        # flow pass; whichever Layer 03 runs first calls
        # `shared.climax_extraction.populate_climax_for_manifest` and fills
        # the metadata in place.

        cap.release()
        
        return {
            "video_id": video_id,
            "source_dataset": entry['dataset'],
            "video_path": str(video_path),
            "fps": fps,
            "duration_sec": duration_sec,
            "identified_tasks": identified_tasks,
            "bystander_detections": entry['bystander_detections'],
            "hand_detections": entry.get('hand_detections', [])
        }

    def log_error(self, video_id, error):
        error_entry = {
            "video_id": video_id,
            "error": str(error),
            "traceback": traceback.format_exc()
        }
        print(f"Error processing {video_id}: {error}")
        
        errors = []
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    errors = json.load(f)
            except:
                pass
        
        errors.append(error_entry)
        with open(self.error_log_path, 'w') as f:
            json.dump(errors, f, indent=4)

    # No longer needed as we use two-pass
    # def process_video(self, entry): ...

    def social_presence_filter(self, video_path, sample_rate_fps=1):
        """ Sample frames at sample_rate_fps and detect all persons. """
        detections_by_frame, hands_by_frame = self.detector.detect(video_path, sample_rate_fps=sample_rate_fps, fast_mode=False, return_hands=True)
        
        if not detections_by_frame:
            return [], []
            
        # Group by tracked person_id
        bystanders_map = {}
        for frame in detections_by_frame:
            for det in frame:
                pid = det.get("person_id", 0)
                if pid not in bystanders_map:
                    bystanders_map[pid] = {
                        "person_id": pid,
                        "timestamps_sec": [],
                        "bounding_boxes": [],
                        "detection_confidence": []
                    }
                bystanders_map[pid]["timestamps_sec"].append(det["timestamp_sec"])
                bystanders_map[pid]["bounding_boxes"].append(det["bounding_box"])
                bystanders_map[pid]["detection_confidence"].append(det["confidence"])
                
        bystanders = list(bystanders_map.values())
        return bystanders, hands_by_frame

    def contextual_task_labeling(self, video_id, duration_sec):
        """ Use Ego4D metadata to identify tasks and map velocities """
        if video_id not in self.metadata:
            # Try removing extension if present in video_id
            clean_id = video_id.split('.')[0]
            if clean_id not in self.metadata:
                print(f"Warning: No metadata found for video {video_id}")
                return []
            video_meta = self.metadata[clean_id]
        else:
            video_meta = self.metadata[video_id]

        scenarios = video_meta.get('scenarios', [])
        if not scenarios:
            return []

        valid_scenarios = []
        for scenario in scenarios:
            label = scenario.strip()
            if label and label.lower() not in ["idling", "no clear task", "ambiguous activity"]:
                valid_scenarios.append(label)
        
        identified_tasks = []
        num_tasks = len(valid_scenarios)
        for i, label in enumerate(valid_scenarios):
            # Heuristic velocity mapping
            velocity = self._get_velocity_from_label(label)
            
            task_duration = duration_sec / num_tasks if num_tasks > 0 else duration_sec
            task_start_sec = i * task_duration
            task_end_sec = (i + 1) * task_duration
            
            task = {
                "task_id": f"t_{len(identified_tasks)+1:02d}",
                "task_label": label,
                "task_confidence": 1.0, # Ground truth metadata
                "task_velocity": velocity,
                "task_start_sec": round(task_start_sec, 2),
                "task_end_sec": round(task_end_sec, 2),
                "task_temporal_metadata": {}
            }
            identified_tasks.append(task)

        return identified_tasks

    def _get_velocity_from_label(self, label):
        """ Heuristic to map task labels to physical velocity """
        l = label.lower()
        fast_keywords = ['drop', 'slip', 'fall', 'throw', 'hit', 'impact', 'jump', 'run', 'fast', 'accident', 'shatter']
        slow_keywords = ['read', 'write', 'think', 'solve', 'puzzle', 'watch', 'look', 'sit', 'slow', 'wait', 'rest']
        
        if any(k in l for k in fast_keywords):
            return 'fast'
        if any(k in l for k in slow_keywords):
            return 'slow'
        return 'medium'


