import os
import json
import cv2
import traceback
import numpy as np
from PIL import Image
from pathlib import Path

# Set HuggingFace Cache to Extreme SSD to prevent local drive fillup
# This must be set BEFORE transformers is imported
SSD_HF_CACHE = "/Volumes/Extreme SSD/huggingface_cache"
os.environ['HF_HOME'] = SSD_HF_CACHE

try:
    import torch
    from transformers import pipeline, SamModel, SamProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from src.models_config import get_model
except ImportError:
    from models_config import get_model

class ProxemicKinematicsPipeline:
    # --- Tuning Constants ---
    # All proxemic heuristic thresholds live here so retuning is a single-file
    # surgical edit (or a subclass override) rather than a hunt across helpers.
    OPTICAL_FLOW_NOISE_THRESHOLD = 15.0   # 95th percentile farneback magnitude beyond which we void the proxemic vector
    BBOX_NORM_PCT = 50.0                  # bbox area delta % that maps to ±1.0 after normalization
    DEPTH_NORM_SCALE = 2.0                # multiplier on -depth_delta when normalizing into [-1, 1]
    BBOX_WEIGHT = 0.4                     # weight of bbox heuristic in the fused proxemic vector
    DEPTH_WEIGHT = 0.6                    # weight of depth heuristic in the fused proxemic vector
    MICROMOVEMENT_THRESHOLD = 0.05        # |vector| below this is treated as no movement (jitter rejection)
    APPROACH_THRESHOLD = 0.3              # vector > this -> Approach_Intervention; < -this -> Avoidance
    # Proxemic trajectory (docs/03d Issue 1 -> Option A): with window-dense boxes
    # (docs/02 Issue 1) the bbox-scale signal is a SERIES across the window, not
    # just two endpoints, so we additionally emit its shape + a downsampled
    # scale-vs-first series. Strictly additive — the 2-endpoint delta and
    # classified_action are unchanged.
    TRAJECTORY_FLAT_CV = 0.08             # coeff-of-variation below which the scale series is "flat"
    TRAJECTORY_MAX_SAMPLES = 8            # cap on the downsampled series length (keeps the JSON lean)
    # Pinned vocabulary for `proxemic_trajectory_shape` — the single source of
    # truth the Layer-04 registry (any_eq:* columns) + test_layer_04 enum-sync
    # check against. Keep in sync with `_compute_proxemic_trajectory`.
    TRAJECTORY_SHAPES = ("insufficient", "flat", "monotonic_approach", "monotonic_retreat", "oscillatory")
    # Issue 1 (June 10): detections on each side of the climax-nearest detection
    # used when the strict reaction window holds <2 detections. Node-02 emits
    # bystander detections at a median ~6s cadence vs 2s reaction windows, so a
    # window holds at most ONE detection by construction; +/-1 detection spans
    # 2-3 consecutive detections (~6-12s) — the minimum measurable interval.
    ANCHOR_SPAN_DETECTIONS = 1
    # Span cap (June 11): hard ceiling in SECONDS on the anchored span. The
    # +/-1-detection span is an index span, so a sparse track's neighbor gap can
    # stretch it to minutes (observed max 198s on the June 10 run) — a bbox/depth
    # delta over that interval measures locomotion/track drift, not a reaction.
    # Anchored windows wider than this skip the bystander (None). All 10
    # confident June-10 vectors sat on <=12s spans, so 30s is conservative.
    MAX_ANCHOR_SPAN_SEC = 30.0
    # Track-explosion cap (June 26): Node-02 fragments bystander tracking into many
    # short spurious positive-id tracks — median 48, up to 829 "persons"/clip,
    # median 4 detections each. 03d's per-bystander Depth-Anything + SAM is far too
    # expensive to pay on all of them (18,497 tracks corpus-wide -> a multi-day run)
    # and they are not real people. Process only the N longest tracks per clip (the
    # genuine, sustained bystanders a scene actually has); short fragments sort to
    # the bottom and are dropped. Bounds the run to ~1,945 tracks. 03f needs the
    # same cap.
    MAX_BYSTANDERS_PER_CLIP = 10
    # Issue 1 Option C (June 13): identity-continuity guard. The upstream
    # collision-proof fallback (social_presence.py) fixes new manifests, but
    # already-generated manifests can still carry a person_id whose detections
    # jump between two bodies (the banner-holder -> child case). When consecutive
    # in-window detection boxes have near-zero IoU AND a large area change, the
    # track is not a continuously-tracked body, so its bbox/depth delta is
    # meaningless and the vector is zeroed (identity_discontinuity). The June 13
    # run separates cleanly: genuine vectors have min consecutive IoU >= 0.31,
    # collisions ~0.00. A same-area position jump (IoU~0 but area ratio ~1.0) is
    # a benign camera pan of one person and is intentionally NOT flagged.
    IDENTITY_IOU_FLOOR = 0.1     # consecutive-box IoU at/below this is a discontinuity candidate
    IDENTITY_AREA_LO = 0.6       # area ratio outside [LO, HI] marks a large size jump (different body)
    IDENTITY_AREA_HI = 1.6

    # Auto-tune accelerator cache flush frequency based on host memory.
    # Hosts with >= 48GB can absorb the intermediate cache pressure of many videos.
    FLUSH_EVERY_N_VIDEOS = 25 if (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) >= 48 * 1024**3 else 1

    def __init__(self, input_manifest_path, output_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03d_proxemic_kinematics_errors.json"
        self.force = force
        self.depth_estimator = None
        self.sam_model = None
        self.sam_processor = None
        self.device = 'cpu'
        self._sam_failure_warned = False  # Issue 2: one-time loud fallback warning
        
        self.processed_ids = set()
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh.")
                
        self._init_model()

    def _init_model(self):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Missing required dependency 'transformers>=4.35.0'. "
                               "Install with: pip install transformers>=4.35.0 huggingface_hub torch")

        # Validate the Extreme SSD is actually mounted before letting transformers
        # write multi-hundred-MB model weights into a phantom /Volumes/<name>
        # directory on the internal drive. macOS does not block writes under
        # /Volumes/<name> when nothing is mounted there, so a missing SSD silently
        # fills the boot disk — exactly the failure mode the SSD cache exists to
        # prevent.
        ssd_root = SSD_HF_CACHE.rsplit("/huggingface_cache", 1)[0]
        if not os.path.ismount(ssd_root):
            raise RuntimeError(
                f"Extreme SSD is not mounted at '{ssd_root}'. The Proxemic Kinematics "
                f"layer caches depth + SAM weights (~0.5-4 GB depending on tier) on "
                f"this volume to keep the boot disk clear. Mount the SSD and retry, "
                f"or override SSD_HF_CACHE in pipeline.py if running on a host "
                f"without the SSD."
            )

        try:
            # Create SSD cache directory if it doesn't exist
            os.makedirs(SSD_HF_CACHE, exist_ok=True)

            # Check for MPS/GPU
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'

            device_id = -1
            if self.device == 'mps':
                device_id = 'mps'
            elif self.device == 'cuda':
                device_id = 0

            depth_model_id = get_model("layer_03d_depth")
            sam_model_id = get_model("layer_03d_sam")

            print(f"Initializing depth estimator '{depth_model_id}' on {self.device} (Cache: {SSD_HF_CACHE})...")
            self.depth_estimator = pipeline(task="depth-estimation", model=depth_model_id, device=device_id)

            print(f"Initializing SAM '{sam_model_id}' (bbox-prompted) on {self.device}...")
            # Bbox-prompted SAM via the lower-level SamModel/SamProcessor API.
            # The high-level mask-generation pipeline runs an exhaustive 32x32
            # candidate-point grid (1024 forward passes per crop), which dominates
            # wall-clock cost on the M4 Pro MPS backend and is wasted work given
            # we already have the bystander bbox as a single-mask prompt.
            self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
            self.sam_model.eval()
            self.sam_processor = SamProcessor.from_pretrained(sam_model_id)

            # Validate download path
            cache_slug = "models--" + depth_model_id.replace("/", "--")
            model_cache_path = Path(SSD_HF_CACHE) / "hub" / cache_slug
            if not model_cache_path.exists():
                print("Warning: Model does not appear to be saved in the expected Extreme SSD cache location.")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize depth/SAM stack: {e}")

    def _log_error(self, video_id, error):
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
        
        temp_err = self.error_log_path.with_suffix('.tmp')
        with open(temp_err, 'w') as f:
            json.dump(errors, f, indent=4)
        temp_err.replace(self.error_log_path)

    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)

        results = []
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
            except:
                pass

        videos_processed_in_session = 0
        for entry in registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            videos_processed_in_session += 1
            print(f"Processing Proxemic Kinematics for video: {video_id}")
            try:
                result = self.process_video(entry)
                if result is None:
                    # Sentinel record: persist the skip decision so subsequent
                    # resume runs don't redo the (expensive) optical-flow + depth
                    # scans only to discard the result again. Downstream consumers
                    # filter on `tasks_analyzed` length or `skipped_reason`.
                    result = {
                        "video_id": video_id,
                        "layer": "03d_proxemic_kinematics",
                        "tasks_analyzed": [],
                        "skipped_reason": "no_output_produced"
                    }
                results.append(result)
                self.processed_ids.add(video_id)

                # Atomic write
                temp_out = self.output_result_path.with_suffix('.tmp')
                with open(temp_out, 'w') as f:
                    json.dump(results, f, indent=4)
                temp_out.replace(self.output_result_path)
            except Exception as e:
                self._log_error(video_id, e)
            finally:
                # Bound MPS/CUDA cache growth. Long batches accumulate
                # intermediate activations across Depth Anything + SAM forward passes.
                # Gated on host memory to maximize throughput on larger hosts.
                if videos_processed_in_session % self.FLUSH_EVERY_N_VIDEOS == 0:
                    self._release_accelerator_cache()

        print(f"Final count: {len(results)} videos processed for Proxemic Kinematics.")

    def _release_accelerator_cache(self):
        """Flush the per-device cache after each video. No-op on CPU."""
        if not TRANSFORMERS_AVAILABLE:
            return
        try:
            if self.device == 'mps' and hasattr(torch, 'mps'):
                torch.mps.empty_cache()
            elif self.device == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            # Cache flush is best-effort; never let it break the run loop.
            pass

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['video_path'])
        
        if not video_path.exists():
            print(f"File not found: {video_path}")
            return None
            
        bystanders = entry.get('bystander_detections', [])
        tasks = entry.get('identified_tasks', [])
        
        if not bystanders or not tasks:
            print(f"No bystanders or tasks found for {video_id}.")
            return self._sentinel(video_id, "no_bystanders" if not bystanders else "no_tasks")

        tasks_analyzed = []
        skip_reasons = []  # Issue 2: why each bystander was skipped (sentinel provenance)
        # Cross-layer multi-window guardrail (docs/03 § Multi-Window Reaction
        # Segments): a sparse bystander's segments can re-anchor onto the SAME
        # measurement window, so score each distinct (person, window) once per clip.
        scored_windows = set()
        # Track-explosion cap (see MAX_BYSTANDERS_PER_CLIP): keep only the longest
        # tracks (genuine sustained bystanders). Sorting by detection count puts the
        # real bystanders first; short untracked negative-id fragments sort to the
        # bottom and are dropped here (the untracked filter below is the safety net).
        bystanders = sorted(
            bystanders, key=lambda b: len(b.get('timestamps_sec', [])), reverse=True
        )[:self.MAX_BYSTANDERS_PER_CLIP]
        # Bystander-aware multi-window climax: one reaction segment per cluster.
        try:
            from shared.climax_extraction import expand_task_segments
        except ImportError:
            from src.shared.climax_extraction import expand_task_segments
        for task in expand_task_segments(tasks):
            task_id = task.get('task_id', 'unknown')
            meta = task.get('task_temporal_metadata', {})
            reaction_window = meta.get('task_reaction_window_sec')
            
            if not reaction_window or len(reaction_window) != 2:
                continue
                
            start_sec, end_sec = reaction_window
            climax_sec = meta.get('task_climax_sec', (start_sec + end_sec) / 2.0)
            noise_threshold = self.OPTICAL_FLOW_NOISE_THRESHOLD
            # Issue 1 (June 10): ego-motion noise is measured over each
            # bystander's actual measurement window (which may be anchored and
            # wider than the 2s reaction window); cached per window so
            # bystanders sharing a window pay Farneback once.
            chaos_cache = {}

            per_person = []
            for bystander in bystanders:
                person_id = bystander.get('person_id')
                # Cross-layer guardrail: skip UNTRACKED bystanders — a negative
                # person_id is an untracked box (Resolved #4), a phantom with no
                # real track to read a proxemic vector from.
                if person_id is None or person_id < 0:
                    skip_reasons.append("untracked_track")
                    continue
                timestamps_sec = bystander.get('timestamps_sec', [])
                bounding_boxes = bystander.get('bounding_boxes', [])

                if not timestamps_sec or not bounding_boxes:
                    continue

                # Issue 1 (June 10): per-bystander window anchoring + span.
                # Returns (None, None, reason) when the bystander is skipped;
                # the reason feeds the sentinel provenance (Issue 2).
                win_start, win_end, window_source = self._bystander_measurement_window(
                    timestamps_sec, climax_sec, start_sec, end_sec)
                if win_start is None:
                    skip_reasons.append(window_source)
                    continue

                # Distinct-window dedup: this (person, window) was already scored
                # in an earlier segment (the sparse track re-anchored to the same
                # window) — same vector, don't recompute/emit it. Also saves the
                # depth + Farneback cost on the duplicate.
                dedup_key = (person_id, round(win_start, 2), round(win_end, 2))
                if dedup_key in scored_windows:
                    continue
                scored_windows.add(dedup_key)

                bbox_delta = self._calculate_bbox_scale_delta(timestamps_sec, bounding_boxes, win_start, win_end)

                # Check if person is present in the window at all
                if bbox_delta is None:
                    continue

                # Additive proxemic trajectory (docs/03d Issue 1 -> Option A):
                # the shape of the scale series across the (now-dense) window.
                traj = self._compute_proxemic_trajectory(
                    timestamps_sec, bounding_boxes, win_start, win_end)

                # Issue 1 Option C (June 13): identity-continuity provenance.
                # min_consecutive_iou is recorded on every vector below; an
                # actual identity break is REJECTED after the chaos gate (not
                # before), so the documented ego-motion gate keeps precedence
                # and its attribution, and the guard targets exactly the gap it
                # exists for: confident false positives that SURVIVE chaos.
                min_iou, identity_break = self._window_identity_continuity(
                    timestamps_sec, bounding_boxes, win_start, win_end)

                win_key = (round(win_start, 2), round(win_end, 2))
                if win_key not in chaos_cache:
                    chaos_cache[win_key] = self._extract_ego_motion_noise(video_path, win_start, win_end)
                chaos_score = chaos_cache[win_key]

                if chaos_score > noise_threshold:
                    per_person.append({
                        "person_id": person_id,
                        "bbox_scale_delta_pct": round(bbox_delta, 2),
                        "depth_anything_v2_delta": 0.0,
                        "proxemic_vector": 0.0,
                        "classified_action": "Neutral",
                        "proxemic_confidence": 0.0,
                        "optical_flow_noise": round(chaos_score, 2),
                        "measurement_window_sec": [round(win_start, 2), round(win_end, 2)],
                        "window_source": window_source,
                        "min_consecutive_iou": round(min_iou, 3),
                        **traj
                    })
                    continue

                # Reject a chaos-surviving vector whose in-window detections jump
                # between bodies (an upstream id collision/switch in an already-
                # generated manifest) — the confident false positives the chaos
                # and confidence gates cannot see (e.g. the 599f2f09 -0.48).
                if identity_break:
                    per_person.append({
                        "person_id": person_id,
                        "bbox_scale_delta_pct": round(bbox_delta, 2),
                        "depth_anything_v2_delta": 0.0,
                        "proxemic_vector": 0.0,
                        "classified_action": "Neutral",
                        "proxemic_confidence": 0.0,
                        "optical_flow_noise": round(chaos_score, 2),
                        "measurement_window_sec": [round(win_start, 2), round(win_end, 2)],
                        "window_source": window_source,
                        "identity_discontinuity": True,
                        "min_consecutive_iou": round(min_iou, 3),
                        **traj
                    })
                    continue

                depth_delta = self._calculate_depth_delta(video_path, timestamps_sec, bounding_boxes, win_start, win_end)
                
                # Check if depth calculation succeeded
                if depth_delta is None:
                    continue
                    
                proxemic_vector, action = self._compute_proxemic_vector(bbox_delta, depth_delta)
                
                # Confidence score: if signs disagree, lower confidence
                # e.g., bbox grows (+) but depth increases (-) -> conflict
                # bbox_delta: + approach. depth_delta: - approach.
                # signs agree if (bbox_delta > 0 and depth_delta < 0) or (bbox_delta < 0 and depth_delta > 0)
                # sign of bbox_delta vs sign of (-depth_delta)
                bbox_sign = 1 if bbox_delta > 0 else (-1 if bbox_delta < 0 else 0)
                depth_app_sign = 1 if depth_delta < 0 else (-1 if depth_delta > 0 else 0)
                
                confidence = 1.0 - (abs(bbox_sign - depth_app_sign) / 2.0)
                
                per_person.append({
                    "person_id": person_id,
                    "bbox_scale_delta_pct": round(bbox_delta, 2),
                    "depth_anything_v2_delta": round(depth_delta, 4),
                    "proxemic_vector": round(proxemic_vector, 2),
                    "classified_action": action,
                    "proxemic_confidence": round(confidence, 2),
                    "optical_flow_noise": round(chaos_score, 2),
                    "measurement_window_sec": [round(win_start, 2), round(win_end, 2)],
                    "window_source": window_source,
                    "min_consecutive_iou": round(min_iou, 3),
                    **traj
                })

            if per_person:
                tasks_analyzed.append({
                    "task_id": task_id,
                    "per_person": per_person
                })
                
        if not tasks_analyzed:
            return self._sentinel(video_id, self._aggregate_skip_reason(skip_reasons))

        return {
            "video_id": video_id,
            "layer": "03d_proxemic_kinematics",
            "tasks_analyzed": tasks_analyzed
        }

    def _bystander_measurement_window(self, timestamps, climax_sec, start_sec, end_sec):
        """Issue 1 (June 10): per-bystander measurement window.

        The strict `task_reaction_window_sec` (2s, wearer-climax-anchored) holds
        at most one Node-02 detection (median ~6s cadence), starving the >=2-
        detection precondition — the June 9 smell test scored 0/50. Mirrors the
        03b Resolved #2 pattern: keep the original window when it already holds
        >=2 detections; otherwise anchor to the detection nearest the climax and
        widen to ANCHOR_SPAN_DETECTIONS on each side (>=2 consecutive
        detections, ~6-12s).

        Returns (start, end, window_source); on skip returns (None, None, reason)
        where reason is "single_detection" (<2 usable detections) or
        "span_capped" (anchored span exceeds MAX_ANCHOR_SPAN_SEC). The reason
        feeds the sentinel provenance in process_video (Issue 2).
        """
        # Delegates to the shared cross-layer helper (src/shared/bystander_window.py,
        # June 14). 03d counts DETECTIONS for the dense-window test and needs >= 2
        # (a bbox/depth delta has two endpoints), uses no padding, and skips a
        # sparser track with reason "single_detection".
        try:
            from shared.bystander_window import bystander_measurement_window
        except ImportError:
            from src.shared.bystander_window import bystander_measurement_window
        return bystander_measurement_window(
            timestamps, climax_sec, start_sec, end_sec,
            min_in_reaction_window=2,
            anchor_span_detections=self.ANCHOR_SPAN_DETECTIONS,
            pad_sec=0.0,
            max_anchor_span_sec=self.MAX_ANCHOR_SPAN_SEC,
            allow_single_detection=False,
            no_detection_reason="single_detection",
        )

    @staticmethod
    def _box_area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _window_identity_continuity(self, timestamps, bboxes, win_start, win_end):
        """Issue 1 Option C: detect an upstream identity collision/switch inside
        the measurement window. Returns (min_consecutive_iou, is_discontinuous).

        A continuously-tracked person's box moves/scales smoothly, so consecutive
        in-window detections overlap (IoU well above 0). When a person_id is
        reused for a different body, consecutive boxes have near-zero IoU AND a
        large size change. A same-area position jump (IoU~0 but area ratio ~1.0)
        is a benign camera pan of ONE person and is NOT flagged — that is what
        spares real approach/recede while catching the banner-holder->child
        collision. On the June 13 corpus genuine vectors had min IoU >= 0.31 vs
        ~0.00 for collisions.
        """
        in_win = sorted((t, b) for t, b in zip(timestamps, bboxes)
                        if win_start <= t <= win_end)
        if len(in_win) < 2:
            return 1.0, False
        min_iou = 1.0
        discontinuous = False
        for (_, b0), (_, b1) in zip(in_win, in_win[1:]):
            iou = self._iou(b0, b1)
            min_iou = min(min_iou, iou)
            a0 = self._box_area(b0)
            ratio = (self._box_area(b1) / a0) if a0 > 0 else 1.0
            if iou <= self.IDENTITY_IOU_FLOOR and (ratio < self.IDENTITY_AREA_LO or ratio > self.IDENTITY_AREA_HI):
                discontinuous = True
        return min_iou, discontinuous

    @staticmethod
    def _sentinel(video_id, reason):
        """Sentinel record carrying a specific skip reason (Issue 2)."""
        return {
            "video_id": video_id,
            "layer": "03d_proxemic_kinematics",
            "tasks_analyzed": [],
            "skipped_reason": reason
        }

    @staticmethod
    def _aggregate_skip_reason(reasons):
        """Issue 2: collapse per-bystander skip reasons into one sentinel label."""
        if not reasons:
            return "no_output_produced"
        uniq = set(reasons)
        if uniq == {"span_capped"}:
            return "all_bystanders_span_capped"
        if uniq == {"single_detection"}:
            return "all_bystanders_single_detection"
        return "mixed_skip"

    def _calculate_bbox_scale_delta(self, timestamps, bboxes, start_sec, end_sec):
        # Extract bboxes in the window
        window_areas = []
        for t, bbox in zip(timestamps, bboxes):
            if start_sec <= t <= end_sec:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                window_areas.append((t, area))
                
        if len(window_areas) < 2:
            return None
            
        window_areas.sort(key=lambda x: x[0])
        first_area = window_areas[0][1]
        last_area = window_areas[-1][1]
        
        if first_area <= 0:
            return 0.0
            
        delta_pct = ((last_area - first_area) / first_area) * 100.0
        return delta_pct

    def _compute_proxemic_trajectory(self, timestamps, bboxes, start_sec, end_sec):
        """Additive (docs/03d Issue 1 -> Option A). With window-dense boxes the
        bystander's bbox-scale is a series across the window, so characterize its
        SHAPE and return a downsampled scale-vs-first series, alongside how many
        dense samples backed it. Leaves the existing 2-endpoint delta untouched;
        degrades to "insufficient" when < 3 samples (the sparse case)."""
        areas = []
        for t, bbox in zip(timestamps, bboxes):
            if start_sec <= t <= end_sec:
                x1, y1, x2, y2 = bbox
                a = (x2 - x1) * (y2 - y1)
                if a > 0:
                    areas.append((t, a))
        areas.sort(key=lambda p: p[0])
        n = len(areas)
        out = {"proxemic_trajectory_shape": "insufficient",
               "proxemic_trajectory_pct": [],
               "proxemic_trajectory_n": n}
        if n < 3:
            return out
        a0 = areas[0][1]
        ratios = [a / a0 for _, a in areas]
        mean_r = sum(ratios) / n
        var = sum((r - mean_r) ** 2 for r in ratios) / n
        cv = (var ** 0.5) / mean_r if mean_r else 0.0
        net = ratios[-1] - ratios[0]
        reversals = sum(1 for i in range(1, n - 1)
                        if (ratios[i + 1] - ratios[i]) * (ratios[i] - ratios[i - 1]) < 0)
        if cv < self.TRAJECTORY_FLAT_CV:
            shape = "flat"
        elif reversals >= 2:
            shape = "oscillatory"
        elif net > 0:
            shape = "monotonic_approach"
        else:
            shape = "monotonic_retreat"
        step = max(1, n // self.TRAJECTORY_MAX_SAMPLES)
        series = [round((ratios[i] - 1.0) * 100.0, 1) for i in range(0, n, step)]
        out.update(proxemic_trajectory_shape=shape, proxemic_trajectory_pct=series)
        return out

    def _extract_ego_motion_noise(self, video_path, start_sec, end_sec):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            return 0.0
            
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return 0.0
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
        
        max_chaos = 0.0
        current_frame_idx = start_frame + 1
        
        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            chaos_score = float(np.percentile(mag, 95))
            if chaos_score > max_chaos:
                max_chaos = chaos_score
                
            prev_gray = gray
            current_frame_idx += 1
            
        cap.release()
        return max_chaos

    def _calculate_depth_delta(self, video_path, timestamps, bboxes, start_sec, end_sec):
        if self.depth_estimator is None:
            return None
            
        # Select timestamps in window
        valid_frames = []
        for t, bbox in zip(timestamps, bboxes):
            if start_sec <= t <= end_sec:
                valid_frames.append((t, bbox))
                
        if len(valid_frames) < 2:
            return None
            
        valid_frames.sort(key=lambda x: x[0])
        
        window_duration = end_sec - start_sec
        num_frames = max(5, int(window_duration * 3))
        num_frames = min(20, num_frames)
        
        if len(valid_frames) > num_frames:
            indices = np.linspace(0, len(valid_frames) - 1, num_frames, dtype=int)
            valid_frames = [valid_frames[i] for i in indices]
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            return None
            
        depths = []
        for t, bbox in valid_frames:
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                # depth result is dict with 'depth' key containing PIL image
                result = self.depth_estimator(img)
                depth_img = result['depth']
                depth_arr = np.array(depth_img)
                
                # The depth map may be a different resolution than the
                # original frame.  Rescale bbox coordinates to match.
                dh, dw = depth_arr.shape[:2]
                scale_x = dw / w
                scale_y = dh / h
                dx1 = max(0, int(x1 * scale_x))
                dy1 = max(0, int(y1 * scale_y))
                dx2 = min(dw, int(x2 * scale_x))
                dy2 = min(dh, int(y2 * scale_y))
                
                # SAM-1 bbox-prompted segmentation. We pass the bystander bbox in
                # original frame coordinates as an `input_boxes` prompt and take
                # the highest-IoU mask from the (1, num_masks_per_box, H, W)
                # output. One forward pass per frame instead of 1024 candidate
                # points; deterministic mask selection by IoU, not by largest-
                # area heuristic gambling on background.
                best_mask_full = self._segment_with_sam(img, x1, y1, x2, y2)
                if best_mask_full is not None:
                    # The SAM mask is at the original frame resolution; resize
                    # to the depth map's resolution so masking aligns 1:1.
                    mask_pil = Image.fromarray(best_mask_full.astype(np.uint8) * 255).resize(
                        (dw, dh), Image.NEAREST
                    )
                    mask = np.array(mask_pil, dtype=bool)
                    # If SAM returned an empty mask (degenerate prompt), fall
                    # back to the rectangular bbox so we still produce a sample.
                    if not mask.any():
                        mask = np.zeros(depth_arr.shape[:2], dtype=bool)
                        mask[dy1:dy2, dx1:dx2] = True
                else:
                    # SAM unavailable or inference failed — fall back to the bbox.
                    mask = np.zeros(depth_arr.shape[:2], dtype=bool)
                    mask[dy1:dy2, dx1:dx2] = True

                masked_depths = depth_arr[mask]
                if masked_depths.size > 0:
                    median_depth = float(np.median(masked_depths))
                    # normalize by 255 for simplicity
                    depths.append((t, median_depth / 255.0))
            except Exception as e:
                print(f"Depth inference failed at t={t}: {e}")
                
        cap.release()

        return self._slope_span_delta(depths)

    @staticmethod
    def _slope_span_delta(depths):
        """Linear-regression slope through (t, median_depth) pairs, scaled by
        window duration to recover a unit-consistent "predicted span" delta.

        Robust to single-frame outliers at either endpoint and uses all
        collected samples — recovering the rationale for the adaptive 3 FPS
        sampling that previously fed a discarded two-point delta.
        """
        if len(depths) < 2:
            return None
        ts = np.array([t for t, _ in depths], dtype=float)
        ds = np.array([d for _, d in depths], dtype=float)
        slope, _ = np.polyfit(ts, ds, 1)
        window_duration = ts[-1] - ts[0]
        if window_duration <= 0:
            return None
        return float(slope * window_duration)

    def _segment_with_sam(self, img, x1, y1, x2, y2):
        """Run SAM with the bbox as an input-box prompt; return a boolean mask
        at the original image resolution, or None if SAM is unavailable / fails.

        The mask is selected by the SAM IoU score head rather than by largest
        area, so we don't accidentally promote a background mask just because
        it covers more pixels.
        """
        if self.sam_model is None or self.sam_processor is None:
            return None
        try:
            inputs = self.sam_processor(
                img,
                input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
                return_tensors="pt",
            )
            # Issue 2 (June 10): SamProcessor emits float64 tensors
            # (input_boxes / original_sizes) which the MPS backend cannot host
            # ("Cannot convert a MPS Tensor to float64 dtype") — previously
            # every SAM call failed here and silently fell back to the raw-bbox
            # mask. Cast float64 -> float32 BEFORE moving to device; integer
            # tensors (sizes) are left untouched.
            inputs = {
                k: (v.to(torch.float32) if torch.is_tensor(v) and v.dtype == torch.float64 else v)
                for k, v in inputs.items()
            }
            inputs = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            # masks[0] shape: (num_prompts=1, num_masks_per_prompt, H, W)
            mask_tensor = masks[0][0]
            iou_scores = outputs.iou_scores.cpu().numpy().reshape(-1)
            best_idx = int(np.argmax(iou_scores))
            return mask_tensor[best_idx].numpy().astype(bool)
        except Exception as e:
            # Issue 2 (June 10): never degrade silently for a whole batch — the
            # June 9 smell test showed SAM failing on EVERY frame (MPS float64)
            # while production quietly used diluted raw-bbox depth medians.
            if not self._sam_failure_warned:
                self._sam_failure_warned = True
                print("[03d] *** WARNING: SAM bbox-prompt inference failed — falling "
                      "back to rectangular-bbox depth medians (less precise). First "
                      f"error: {e} ***")
            else:
                print(f"SAM bbox-prompt inference failed: {e}")
            return None

    def _compute_proxemic_vector(self, bbox_delta, depth_delta):
        # Heuristic combination of bbox-area % delta and depth slope-span.
        # Decreasing depth (depth_delta < 0) -> approach -> positive vector.
        norm_bbox = max(-1.0, min(1.0, bbox_delta / self.BBOX_NORM_PCT))
        norm_depth = max(-1.0, min(1.0, -depth_delta * self.DEPTH_NORM_SCALE))
        vector = (norm_bbox * self.BBOX_WEIGHT) + (norm_depth * self.DEPTH_WEIGHT)

        # Reject jitter inside the deadband.
        if abs(vector) < self.MICROMOVEMENT_THRESHOLD:
            vector = 0.0

        if vector > self.APPROACH_THRESHOLD:
            action = "Approach_Intervention"
        elif vector < -self.APPROACH_THRESHOLD:
            action = "Avoidance"
        else:
            action = "Neutral"

        return vector, action
