import json
import cv2
import gc
import traceback
import math
import sys
import torch
import numpy as np
from pathlib import Path

# Try to import config for EGO4D_METADATA_PATH
try:
    from src.config import EGO4D_METADATA_PATH
except ImportError:
    EGO4D_METADATA_PATH = None

# Add L2CS-Net to python path
l2cs_path = Path(__file__).resolve().parent.parent.parent / "models" / "l2cs-net"
sys.path.append(str(l2cs_path))
try:
    from l2cs import Pipeline
    from l2cs.utils import select_device
except ImportError:
    Pipeline = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Face-presence gate (Resolved Issue 1) and bbox re-detection (Resolved Issue 2)
# dependencies. Imported defensively so the module still loads on hosts without
# them (the gates then degrade to no-op, logged at init).
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
try:
    from src.models_config import get_model
except ImportError:
    try:
        from models_config import get_model
    except ImportError:
        get_model = None

# Face gate tunables (env-overridable). A crop is "valid" for gaze only if a
# human face is detected with confidence >= MIN_FACE_CONF; otherwise the sample
# is scored 0.0 with target "NoFace" (rejects dogs, empty/occluded crops, and
# faces too degraded for reliable gaze — see Resolved Issue 1).
import os as _os
FACE_DETECTOR_PATH = _PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite"
MIN_FACE_CONF = float(_os.getenv("SR_03A_MIN_FACE_CONF", "0.5"))
REDETECT_IOU_THRESH = float(_os.getenv("SR_03A_REDETECT_IOU", "0.1"))
ENABLE_FACE_GATE = _os.getenv("SR_03A_FACE_GATE", "1").lower() in ("1", "true", "yes")
ENABLE_BBOX_REDETECT = _os.getenv("SR_03A_BBOX_REDETECT", "1").lower() in ("1", "true", "yes")
# Head pose (docs/03e Issue 1, Option A): emit head_pitch_rad/head_yaw_rad from a
# MediaPipe FaceLandmarker run on the SAME face crop used for gaze, so 03e can
# detect nod/shake from true head orientation instead of gaze direction. Additive
# + gated: if the asset/mediapipe is missing, no head_* fields are emitted and 03e
# transparently falls back to gaze. FaceLandmarker has its own tracking-loss on
# small faces, so a present-face crop may still yield no head pose (-> null).
HEAD_POSE_MODEL_PATH = _PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"
ENABLE_HEAD_POSE = _os.getenv("SR_03A_HEAD_POSE", "1").lower() in ("1", "true", "yes")
# Window-restricted sampling (scaling Idea 1): only ~19% of clip-time contains a
# bystander, so process just the +/-WINDOW_PAD_SEC intervals around real bystander
# detections and seek past the gaps. WINDOW_PAD_SEC must match the per-sample
# nearest-detection tolerance (the 2.0 s check in _track_and_score_batched) so the
# set of scored frames is unchanged. Toggle off with SR_03A_WINDOW_RESTRICT=0.
ENABLE_WINDOW_RESTRICT = _os.getenv("SR_03A_WINDOW_RESTRICT", "1").lower() in ("1", "true", "yes")
WINDOW_PAD_SEC = float(_os.getenv("SR_03A_WINDOW_PAD_SEC", "2.0"))
# Throughput levers (Resolved Issue 6).
# A' — clip-level face-quality gate: reuse the shared bystander_face_quality
# manifest pre-pass (03b Resolved #8) with a 03a-TUNED tier. L2CS gaze resolves
# smaller faces than HSEmotion emotion, so 03b's strict 120px/0.8/3 tier
# provably loses real attention clips (June 11 join: 6fd026d8 eng 0.94,
# 3f503d0b eng 0.68); 60px/0.6/2 lost none while skipping 54% of the 50-clip
# corpus sample. Fail-open: unscored entries always pass.
ENABLE_FACE_QUALITY_GATE = _os.getenv("SR_03A_FACE_QUALITY_GATE", "1").lower() in ("1", "true", "yes")
FQ_MIN_PX = int(_os.getenv("SR_03A_FQ_MIN_PX", "60"))
FQ_MIN_CONF = float(_os.getenv("SR_03A_FQ_MIN_CONF", "0.6"))
FQ_MIN_FRAMES = int(_os.getenv("SR_03A_FQ_MIN_FRAMES", "2"))
# B' — YOLO re-detect runs at most once per this interval. The gaze-sampling
# cadence is untouched (03e's medfilt contract needs the dense 8-32 FPS trace),
# but a person's box barely moves in 250ms relative to the +/-20px crop pad, so
# re-detecting on every 32 FPS burst frame was 8x wasted YOLO. Cached boxes are
# <=250ms stale vs the <=2s manifest staleness re-detection exists to fix.
REDETECT_MIN_INTERVAL_SEC = float(_os.getenv("SR_03A_REDETECT_INTERVAL", "0.25"))
class AttentionLayerPipeline:
    def __init__(self, input_manifest_path, output_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03a_attention_errors.json"
        self.force = force
        
        # Load Ego4D metadata if available for camera intrinsics
        self.camera_intrinsics = {}
        if EGO4D_METADATA_PATH and Path(EGO4D_METADATA_PATH).exists():
            try:
                with open(EGO4D_METADATA_PATH, 'r') as f:
                    ego4d_meta = json.load(f)
                    for vid in ego4d_meta.get('videos', []):
                        vid_id = vid.get('video_uid')
                        if vid_id and 'camera_intrinsics' in vid:
                            # Using 'camera_intrinsics' object or similar fields if they exist
                            self.camera_intrinsics[vid_id] = vid.get('camera_intrinsics')
                        elif vid_id and 'focal_length' in vid:
                            self.camera_intrinsics[vid_id] = {
                                'focal_length': vid.get('focal_length'),
                                'principal_point': vid.get('principal_point')
                            }
                print(f"Loaded camera intrinsics for {len(self.camera_intrinsics)} videos.")
            except Exception as e:
                print(f"Failed to load Ego4D metadata for intrinsics: {e}")

        self.device = select_device('') if Pipeline else torch.device('cpu')
        try:
            weights_path = l2cs_path / "models" / "L2CSNet_gaze360.pkl"
            if Pipeline and weights_path.exists():
                self.gaze_pipeline = Pipeline(
                    weights=str(weights_path),
                    arch='ResNet50',
                    device=self.device,
                    include_detector=False
                )
            else:
                self.gaze_pipeline = None
        except Exception as e:
            print(f"Failed to load L2CS-Net: {e}")
            self.gaze_pipeline = None

        # Lazy-loaded crop-validation models (Resolved Issues 1 & 2).
        self._face_detector = None
        self._pose_model = None
        self._head_pose_estimator = None  # docs/03e Issue 1
        self.enable_head_pose = ENABLE_HEAD_POSE and HEAD_POSE_MODEL_PATH.exists()
        # Resolved Issue 6 (A'): instance-level so tests (and callers running on
        # synthetic fixtures) can disable the clip gate like enable_face_gate.
        self.enable_face_quality_gate = ENABLE_FACE_QUALITY_GATE
        self.enable_face_gate = ENABLE_FACE_GATE and mp is not None and FACE_DETECTOR_PATH.exists()
        self.enable_bbox_redetect = ENABLE_BBOX_REDETECT and YOLO is not None and get_model is not None
        if ENABLE_FACE_GATE and not self.enable_face_gate:
            print("[03a] Face gate requested but unavailable (mediapipe/model missing); scoring all crops.")
        if ENABLE_BBOX_REDETECT and not self.enable_bbox_redetect:
            print("[03a] Bbox re-detect requested but unavailable (ultralytics/registry missing); using manifest boxes.")

        self.processed_ids = set()
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh.")

        # On ≥48 GB hosts (Mac Studio M4 Max), boost the adaptive burst to 32 FPS
        # so the cadence clears Layer 03e's medfilt saccade-suppression gate (36 FPS
        # in Resolved Issue 10; lowered to 32 FPS in tandem with this change).
        # Mac mini hosts retain the 16 FPS budget.
        if AttentionLayerPipeline.host_can_retain_resident():
            self.burst_stride_sec = self.BURST_STRIDE_SEC_HIGH_MEM
        else:
            self.burst_stride_sec = self.BURST_STRIDE_SEC

    # On hosts with >= 48 GB unified memory the full 03 model set (~12 GB)
    # stays resident, so the E2E orchestrator keeps the L2CS-Net instance
    # alive across the whole 03 run instead of paying a reload + MPS warm-up
    # between layers. Direct `with` callers still unload via __exit__.
    HIGH_MEMORY_HOST_BYTES = 48 * 2**30

    @staticmethod
    def host_can_retain_resident():
        """ True when the host has enough unified memory to keep L2CS-Net
        resident for the whole 03 run; the E2E orchestrator uses this to skip
        the post-run unload(). Falls back to False (unload) without psutil. """
        try:
            import psutil
        except ImportError:
            return False
        return psutil.virtual_memory().total >= AttentionLayerPipeline.HIGH_MEMORY_HOST_BYTES

    @property
    def face_detector(self):
        """ Lazy MediaPipe Tasks BlazeFace detector (Resolved Issue 1). IMAGE mode
        so one instance is reused across crops without timestamp constraints. """
        if self._face_detector is None:
            opts = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(FACE_DETECTOR_PATH)),
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=MIN_FACE_CONF,
            )
            self._face_detector = mp_vision.FaceDetector.create_from_options(opts)
        return self._face_detector

    @property
    def pose_model(self):
        """ Lazy YOLO-pose for per-frame bystander re-detection (Resolved Issue 2).
        Same yolov8n-pose.pt the Node-02 social filter uses. """
        if self._pose_model is None:
            self._pose_model = YOLO(get_model("social_presence_pose"))
        return self._pose_model

    def _crop_has_face(self, crop_bgr):
        """ True iff a human face is detected in the crop at >= MIN_FACE_CONF.
        Gate for Resolved Issue 1 — rejects dogs/empty/occluded/unresolvable crops. """
        if not self.enable_face_gate or crop_bgr.size == 0:
            return True  # gate disabled -> preserve legacy behavior (score all crops)
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            det = self.face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            return len(det.detections) > 0
        except Exception as e:
            print(f"[03a] Face gate error (passing crop through): {e}")
            return True  # an outage must not silently zero out every bystander

    def _estimate_head_pose(self, crop_bgr):
        """(head_pitch_rad, head_yaw_rad) for the crop, or None (docs/03e Issue 1).

        None means "no head pose for this sample" — the caller emits null head_*
        fields and 03e treats them as missing (NaN), never as a real 0.0.
        """
        if not self.enable_head_pose:
            return None
        if self._head_pose_estimator is None:
            try:
                from shared.head_pose import HeadPoseEstimator
            except ImportError:
                from src.shared.head_pose import HeadPoseEstimator
            self._head_pose_estimator = HeadPoseEstimator(HEAD_POSE_MODEL_PATH)
        return self._head_pose_estimator.estimate(crop_bgr)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(0.0, (ax2-ax1)*(ay2-ay1)) + max(0.0, (bx2-bx1)*(by2-by1)) - inter
        return inter / ua if ua > 0 else 0.0

    def _detect_pose_boxes(self, frame_bgr):
        """ Run YOLO-pose once on a frame; return list of person [x1,y1,x2,y2]
        (Resolved Issue 2). Empty list on error / when disabled. """
        if not self.enable_bbox_redetect:
            return []
        try:
            results = self.pose_model(frame_bgr, classes=[0], verbose=False, conf=0.5)
            boxes = []
            for r in results:
                if r.boxes is None:
                    continue
                for bx in r.boxes:
                    boxes.append([int(v) for v in bx.xyxy[0].tolist()])
            return boxes
        except Exception as e:
            print(f"[03a] Pose re-detect error (using manifest box): {e}")
            return []

    def unload(self):
        """ Free the L2CS-Net + crop-validation models from MPS / GPU memory. """
        if getattr(self, 'gaze_pipeline', None) is not None:
            print("[AttentionLayerPipeline] Unloading L2CS-Net...")
            del self.gaze_pipeline
            self.gaze_pipeline = None
        if getattr(self, '_face_detector', None) is not None:
            try:
                self._face_detector.close()
            except Exception:
                pass
            self._face_detector = None
        if getattr(self, '_pose_model', None) is not None:
            del self._pose_model
            self._pose_model = None
        if getattr(self, '_head_pose_estimator', None) is not None:
            try:
                self._head_pose_estimator.close()
            except Exception:
                pass
            self._head_pose_estimator = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    def _passes_face_quality(self, entry):
        """Resolved Issue 6 (A'): 03a-tier clip gate over the shared
        `bystander_face_quality` manifest field. Tier is deliberately looser
        than 03b's (gaze works on smaller faces than emotion). Fail-open:
        unscored entries and any import failure pass everything through."""
        if not getattr(self, 'enable_face_quality_gate', ENABLE_FACE_QUALITY_GATE):
            return True
        try:
            try:
                from shared.face_quality_prefilter import passes_face_quality
            except ImportError:
                from src.shared.face_quality_prefilter import passes_face_quality
        except ImportError:
            return True
        return passes_face_quality(entry, min_px=FQ_MIN_PX, min_conf=FQ_MIN_CONF,
                                   min_frames=FQ_MIN_FRAMES, enabled=True)

    def _atomic_write(self, results):
        temp_out = self.output_result_path.with_suffix('.tmp')
        with open(temp_out, 'w') as f:
            json.dump(results, f, indent=4)
        temp_out.replace(self.output_result_path)

    def run(self, workers=1):
        """Process every eligible clip. `workers` (Resolved Issue 6, June 13):
        **default 1 = serial** (byte-identical to the historical path, and the
        only spawn-safe mode for arbitrary library callers). `workers > 1`
        distributes whole clips across a spawn-safe process Pool — each clip's
        `process_video` is unchanged, so its output record is identical to the
        serial run (validated); only *which process* runs it differs, letting
        one clip's CPU decode overlap another's MPS inference. Use the guarded
        `python -m layer_03a_attention.pipeline … --workers N` CLI to opt in
        (a bare runner that is not `__main__`-guarded must not pass workers>1,
        or macOS 'spawn' will re-exec it)."""
        # Resolved Issue 6 (A'): annotate face quality first (idempotent
        # ~0.9s/clip one-time pre-pass, shared with 03b) so the 03a-tier gate
        # below can skip clips with no gaze-resolvable face before paying
        # decode + YOLO + L2CS. Defensive: any failure leaves the gate open.
        if self.enable_face_quality_gate:
            try:
                try:
                    from shared.face_quality_prefilter import populate_face_quality_for_manifest
                except ImportError:
                    from src.shared.face_quality_prefilter import populate_face_quality_for_manifest
                nq = populate_face_quality_for_manifest(self.input_manifest_path)
                if nq:
                    print(f"[03a] Scored bystander face quality for {nq} manifest entries.")
            except Exception as e:
                print(f"[03a] face-quality pre-filter skipped ({e}); gate fails open.")

        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)

        results = []
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
            except (json.JSONDecodeError, IOError, ValueError):
                pass

        # Build the eligible-clip list with the exact same skip rules the serial
        # loop has always applied (synthetic / flagged_invalid / already-processed
        # / below the face-quality tier), so serial and parallel see an identical
        # work set.
        todo = []
        skipped_face_quality = 0
        for entry in registry:
            if entry.get("synthetic") is True:
                continue
            # Skip clips flagged as invalid social positives (e.g. a non-human
            # bystander track) during spot-check — Node 02 Issue 2 remediation.
            if entry.get("flagged_invalid") is True:
                print(f"Skipping {entry.get('video_id', entry.get('id'))}: flagged_invalid "
                      f"({entry.get('flag_reason', 'unspecified')})")
                continue
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue
            # Resolved Issue 6 (A'): no gaze-resolvable face anywhere in the
            # clip -> the trace would be all NoFace/0.0 (and 03e rejects such
            # traces anyway), so skip before paying decode + inference.
            if not self._passes_face_quality(entry):
                skipped_face_quality += 1
                continue
            todo.append(entry)

        if workers and workers > 1 and len(todo) > 1:
            self._run_parallel(todo, results, int(workers))
        else:
            for entry in todo:
                video_id = entry.get('id', entry.get('video_id'))
                print(f"Processing Attention Layer for video: {video_id}")
                try:
                    result = self.process_video(entry)
                    if result:
                        results.append(result)
                        self.processed_ids.add(video_id)
                        self._atomic_write(results)
                except Exception as e:
                    self.log_error(video_id, e)

        if skipped_face_quality:
            print(f"[03a] Skipped {skipped_face_quality} clips below the face-quality tier "
                  f"(px>={FQ_MIN_PX}, conf>={FQ_MIN_CONF}, frames>={FQ_MIN_FRAMES}).")
        print(f"Final count: {len(results)} videos processed for Attention.")

    def _run_parallel(self, todo, results, workers):
        """Resolved Issue 6 (June 13): distribute whole clips across a spawn-safe
        Pool. Each worker builds its own AttentionLayerPipeline once (models
        loaded per process) and calls the unchanged `process_video`, so every
        emitted record is identical to the serial run; results are written
        incrementally (resumable) as they complete. Records land in completion
        order, not registry order — the Layer-04 aggregator outer-joins on
        `video_id`, so list order is not part of the contract."""
        import multiprocessing as mp
        n = min(workers, len(todo))
        print(f"[03a] Parallel decode: {n} workers over {len(todo)} clips "
              f"(per-clip output identical to serial; MPS inference serializes, "
              f"CPU decode overlaps).")
        ctx = mp.get_context("spawn")
        init_args = (str(self.input_manifest_path), str(self.output_result_path))
        with ctx.Pool(processes=n, initializer=_par_worker_init, initargs=init_args) as pool:
            for video_id, result, err in pool.imap_unordered(_par_worker_process, todo):
                if err:
                    self._record_error(video_id, err)
                elif result:
                    results.append(result)
                    self.processed_ids.add(video_id)
                    self._atomic_write(results)

    def _record_error(self, video_id, error_text):
        """Append a worker-reported error (already-formatted traceback string)
        to the error log, mirroring log_error's on-disk shape."""
        errors = []
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    errors = json.load(f)
            except (json.JSONDecodeError, IOError, ValueError):
                pass
        errors.append({"video_id": video_id, "error": error_text, "traceback": error_text})
        with open(self.error_log_path, 'w') as f:
            json.dump(errors, f, indent=4)
        print(f"Error processing {video_id} (worker): {error_text.splitlines()[-1] if error_text else ''}")

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
            except (json.JSONDecodeError, IOError, ValueError):
                pass
        
        errors.append(error_entry)
        with open(self.error_log_path, 'w') as f:
            json.dump(errors, f, indent=4)

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['video_path'])
        
        if not video_path.exists():
            print(f"File not found: {video_path}")
            return None
            
        bystanders = entry.get('bystander_detections', [])
        if not bystanders:
            print(f"No bystanders found for {video_id}.")
            return None
            
        hand_detections = entry.get('hand_detections', [])
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or total_frames == 0:
            cap.release()
            return None
            
        duration_sec = total_frames / fps
        intrinsics = self.camera_intrinsics.get(video_id)
        all_traces = self._track_and_score_batched(cap, fps, duration_sec, bystanders, intrinsics, hand_detections)
        
        per_person_results = []
        for bystander in bystanders:
            person_id = bystander.get('person_id')
            trace = all_traces.get(person_id, [])
            
            if not trace:
                continue
                
            scores = [t['score'] for t in trace]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Find peak engagement timestamp
            peak_item = max(trace, key=lambda x: x['score'])
            peak_timestamp = peak_item['t']
            
            # Variance
            variance = 0
            if len(scores) > 1:
                variance = sum((s - avg_score) ** 2 for s in scores) / (len(scores) - 1)
                
            # Sustained engagement (longest contiguous window >= 0.7)
            # Uses actual 't' timestamps for accuracy across adaptive stride changes
            sustained_windows = []
            current_start = None
            last_t = None
            for item in trace:
                if item['score'] >= 0.7:
                    if current_start is None:
                        current_start = item['t']
                    last_t = item['t']
                else:
                    if current_start is not None and last_t is not None:
                        sustained_windows.append(last_t - current_start)
                    current_start = None
                    last_t = None
            if current_start is not None and last_t is not None:
                sustained_windows.append(last_t - current_start)
                
            sustained_sec = max(sustained_windows) if sustained_windows else 0
            is_engaged = avg_score >= 0.5 or sustained_sec > 2.0
            
            # Aggregate the dominant gaze target across the trace.
            target_counts = {}
            for item in trace:
                tgt = item.get('target')
                # Exclude both "Unknown" and "NoFace" (Resolved Issue 1) so the
                # classification reflects where the bystander looked on the
                # samples where a face was actually resolved, not the gate's
                # rejections.
                if tgt and tgt not in ("Unknown", "NoFace"):
                    target_counts[tgt] = target_counts.get(tgt, 0) + 1
            if target_counts:
                gaze_target = max(target_counts.items(), key=lambda kv: kv[1])[0]
            else:
                gaze_target = "Unknown"

            # Resolved Issue 3: length-invariant engagement metrics. `average_
            # attention_score` averages over the whole trace (incl. NoFace/0.0
            # samples), so it tracks clip length, not engagement. `attended_
            # fraction` = how often the bystander looked; `engaged_attention_
            # score` = how intently when they did (mean over score>0 samples).
            attended_fraction = round(sum(1 for s in scores if s >= 0.5) / len(scores), 3) if scores else 0.0
            _nonzero = [s for s in scores if s > 0.0]
            engaged_attention_score = round(sum(_nonzero) / len(_nonzero), 2) if _nonzero else 0.0

            per_person_results.append({
                "person_id": person_id,
                "average_attention_score": round(avg_score, 2),
                "attended_fraction": attended_fraction,
                "engaged_attention_score": engaged_attention_score,
                "peak_engagement_timestamp_sec": peak_timestamp,
                "attention_variance": round(variance, 4),
                "sustained_engagement_sec": round(sustained_sec, 2),
                "is_engaged": is_engaged,
                "gaze_target_classification": gaze_target,
                "attention_trace": trace
            })
            
        cap.release()
        
        if not per_person_results:
            return None
            
        # Aggregate stats
        all_scores = [p['average_attention_score'] for p in per_person_results]
        mean_all = sum(all_scores) / len(all_scores) if all_scores else 0
        any_engaged = any(p['is_engaged'] for p in per_person_results)
        
        burst_fps = round(1.0 / self.burst_stride_sec, 1)
        return {
            "video_id": video_id,
            "layer": "03a_attention",
            "processing_meta": {
                "model_used": "l2cs_net_3d_gaze" if self.gaze_pipeline else "fallback_dummy",
                "sampling_fps_effective": 8.0,
                "sampling_fps_burst": burst_fps,
                "sampling_strategy": f"adaptive_8_to_{int(burst_fps)}_fps"
            },
            "per_person": per_person_results,
            "aggregate": {
                "num_bystanders_tracked": len(per_person_results),
                "mean_attention_all_persons": round(mean_all, 2),
                "any_person_engaged": any_engaged
            }
        }

    BASELINE_STRIDE_SEC = 0.125            # 8 FPS — preserves Layer 03e Nyquist floor (4 Hz)
    BURST_STRIDE_SEC = 0.0625              # 16 FPS — low-memory default; engaged when |Δscore| > 0.3
    BURST_STRIDE_SEC_HIGH_MEM = 0.03125    # 32 FPS — engaged on ≥48 GB hosts to clear 03e's medfilt gate
    BURST_DURATION_SEC = 2.0
    BURST_DELTA_THRESHOLD = 0.3

    def _track_and_score_batched(self, cap, fps, duration_sec, bystanders, intrinsics=None, hand_detections=None):
        traces = {}
        next_t = {}
        last_scores = {}
        burst_until_t = {}
        
        for bystander in bystanders:
            pid = bystander.get('person_id')
            timestamps_sec = bystander.get('timestamps_sec', [])
            bounding_boxes = bystander.get('bounding_boxes', [])
            if not timestamps_sec or not bounding_boxes:
                continue
            traces[pid] = []
            next_t[pid] = 0.0
            last_scores[pid] = -1.0
            burst_until_t[pid] = -1.0

        if not next_t:
            return {}

        # Idea 1 (window-restricted sampling): build the union of
        # +/-WINDOW_PAD_SEC intervals around every bystander detection timestamp.
        # Samples in a GAP (no bystander within the tolerance) are still grabbed
        # for frame-accuracy but SKIP the expensive per-frame YOLO re-detect +
        # scoring below. v2 ran YOLO on every sampled frame and then discarded
        # gap samples via the 2.0 s check anyway, so the recorded output is
        # unchanged while ~80% of the YOLO passes are eliminated. WINDOW_PAD_SEC
        # equals that 2.0 s tolerance.
        # NOTE: decoding stays fully sequential — cap.set() seeking is
        # keyframe-approximate on H.264 and desyncs frame<->timestamp (it
        # corrupted scores in an earlier seek-based attempt), so we do not seek.
        windows = None
        if ENABLE_WINDOW_RESTRICT:
            det_times = sorted({t for b in bystanders
                                for t in b.get('timestamps_sec', [])
                                if b.get('person_id') in next_t})
            windows = []
            for t in det_times:
                s, e = max(0.0, t - WINDOW_PAD_SEC), min(duration_sec, t + WINDOW_PAD_SEC)
                if windows and s <= windows[-1][1] + 1e-6:
                    windows[-1][1] = max(windows[-1][1], e)
                else:
                    windows.append([s, e])
            if not windows:
                return traces

        def _in_window(t):
            if windows is None:
                return True
            for s, e in windows:
                if s <= t <= e:
                    return True
                if t < s:          # windows are sorted; past the relevant one
                    return False
            return False

        # Prime the capture so the first cap.retrieve() has a valid buffer (Issue 5).
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret = cap.grab()
        if not ret:
            return {}
        current_frame_idx = 1

        # Resolved Issue 6 (B'): YOLO re-detect cache. Refreshed at most every
        # REDETECT_MIN_INTERVAL_SEC; between refreshes the freshest boxes are
        # reused (<=250ms stale). The gaze-sampling cadence below is unchanged.
        last_redetect_t = None
        cached_pose_boxes = []

        while any(next_t[pid] <= duration_sec for pid in next_t):
            current_t = min(next_t[pid] for pid in next_t if next_t[pid] <= duration_sec)
            target_frame_idx = int(current_t * fps)

            # Sequential skip with grab() to reach the target frame without random seeking.
            while current_frame_idx < target_frame_idx:
                ret = cap.grab()
                if not ret:
                    break
                current_frame_idx += 1

            if current_frame_idx < target_frame_idx:
                break

            ret, frame = cap.retrieve()
            current_frame_idx += 1

            if not ret:
                break

            h, w = frame.shape[:2]
            batch_pids = []
            batch_crops = []
            batch_info = []

            # Idea 1: gap sample — skip the per-frame YOLO re-detect + scoring
            # (it would record nothing here, per the 2.0 s check). Advance the
            # active tracks by baseline stride exactly as the in-loop gap path
            # does, so the recorded trace is identical to the full-clip result.
            if windows is not None and not _in_window(current_t):
                for bystander in bystanders:
                    pid = bystander.get('person_id')
                    if pid in next_t and abs(next_t[pid] - current_t) < 1e-4:
                        next_t[pid] += self.BASELINE_STRIDE_SEC
                continue

            # Re-detect bystander boxes at most once per REDETECT_MIN_INTERVAL_SEC
            # (Resolved Issue 6, B'); shared across all bystanders active at
            # current_t. Between refreshes the cached boxes are reused — far
            # fresher than the <=2s-stale manifest boxes that per-frame
            # re-detection (Resolved Issue 3) was added to replace.
            if last_redetect_t is None or (current_t - last_redetect_t) >= REDETECT_MIN_INTERVAL_SEC:
                cached_pose_boxes = self._detect_pose_boxes(frame)
                last_redetect_t = current_t
            pose_boxes = cached_pose_boxes

            for bystander in bystanders:
                pid = bystander.get('person_id')
                if pid not in next_t:
                    continue
                if next_t[pid] > duration_sec:
                    continue

                if abs(next_t[pid] - current_t) < 1e-4:
                    b_timestamps = bystander.get('timestamps_sec', [])
                    b_bboxes = bystander.get('bounding_boxes', [])

                    diffs = [abs(t - current_t) for t in b_timestamps]
                    closest_idx = diffs.index(min(diffs))

                    if diffs[closest_idx] > 2.0:
                        next_t[pid] += self.BASELINE_STRIDE_SEC
                        continue

                    bbox = b_bboxes[closest_idx]
                    # Resolved Issue 2: replace the (up to 2 s) stale manifest box
                    # with the best-IoU fresh YOLO-pose detection on this frame.
                    if pose_boxes:
                        best = max(pose_boxes, key=lambda pb: self._iou(pb, bbox))
                        if self._iou(best, bbox) >= REDETECT_IOU_THRESH:
                            bbox = best
                    x1, y1, x2, y2 = bbox

                    px1 = max(0, int(x1) - 20)
                    py1 = max(0, int(y1) - 20)
                    px2 = min(w, int(x2) + 20)
                    py2 = min(h, int(y2) + 20)

                    crop = frame[py1:py2, px1:px2]
                    if crop.size == 0:
                        next_t[pid] += self.BASELINE_STRIDE_SEC
                        continue

                    # Resolved Issue 1: face-presence gate. Crops without a
                    # detectable human face (dog, empty, occluded, unresolvable)
                    # get a NoFace/0.0 sample instead of a bogus gaze score.
                    if not self._crop_has_face(crop):
                        traces[pid].append({
                            "t": round(current_t, 2), "score": 0.0,
                            "pitch_rad": 0.0, "yaw_rad": 0.0, "target": "NoFace",
                            "head_pitch_rad": None, "head_yaw_rad": None
                        })
                        if last_scores[pid] >= 0.0 and abs(0.0 - last_scores[pid]) > self.BURST_DELTA_THRESHOLD:
                            burst_until_t[pid] = current_t + self.BURST_DURATION_SEC
                        stride = self.burst_stride_sec if current_t < burst_until_t[pid] else self.BASELINE_STRIDE_SEC
                        next_t[pid] += stride
                        last_scores[pid] = 0.0
                        continue

                    resized_crop = cv2.resize(crop, (448, 448))
                    batch_pids.append(pid)
                    batch_crops.append(resized_crop)
                    batch_info.append((pid, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x1, y1, x2, y2))

            if not batch_crops:
                continue

            batch_results = []
            if self.gaze_pipeline and batch_crops:
                try:
                    batch_input = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in batch_crops])
                    results = self.gaze_pipeline.step(batch_input)
                    
                    def to_float(val):
                        if hasattr(val, "item"):
                            return float(val.item())
                        if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                            return float(val[0])
                        return float(val)

                    for idx in range(len(batch_crops)):
                        pitch_rad = to_float(results.pitch[idx]) if results.pitch.size > idx else 0.0
                        yaw_rad = to_float(results.yaw[idx]) if results.yaw.size > idx else 0.0
                        batch_results.append((pitch_rad, yaw_rad))
                except Exception as e:
                    print(f"[03a] Batched gaze inference failed at t={current_t:.2f}s: {e}")
                    batch_results = [(0.0, 0.0)] * len(batch_crops)
            else:
                batch_results = [(0.0, 0.0)] * len(batch_crops)

            for idx, (pid, cx, cy, x1, y1, x2, y2) in enumerate(batch_info):
                pitch_rad, yaw_rad = batch_results[idx]
                # docs/03e Issue 1: true head orientation from the SAME crop the
                # gaze model used. None when FaceLandmarker finds no face (its own
                # tracking-loss) -> null head_* -> 03e treats it as missing (NaN).
                hp = self._estimate_head_pose(batch_crops[idx])
                head_pitch_rad = round(hp[0], 4) if hp else None
                head_yaw_rad = round(hp[1], 4) if hp else None
                score = 0.0
                target_label = "Unknown"

                if self.gaze_pipeline:
                    try:
                        v_look_x = -math.cos(yaw_rad) * math.sin(pitch_rad)
                        v_look_y = -math.sin(yaw_rad)
                        v_look_z = -math.cos(yaw_rad) * math.cos(pitch_rad)

                        if intrinsics and 'focal_length' in intrinsics and 'principal_point' in intrinsics:
                            fx, fy = intrinsics['focal_length'] if isinstance(intrinsics['focal_length'], (list, tuple)) else (intrinsics['focal_length'], intrinsics['focal_length'])
                            px, py = intrinsics['principal_point']
                            v_cam_x = px - cx
                            v_cam_y = py - cy
                            v_cam_z = -float(fx)
                        else:
                            v_cam_x = (w / 2.0) - cx
                            v_cam_y = (h / 2.0) - cy
                            v_cam_z = -float(w)

                        norm_cam = math.sqrt(v_cam_x**2 + v_cam_y**2 + v_cam_z**2)
                        if norm_cam > 0:
                            v_cam_x /= norm_cam
                            v_cam_y /= norm_cam
                            v_cam_z /= norm_cam

                        dot_prod_cam = (v_look_x * v_cam_x) + (v_look_y * v_cam_y) + (v_look_z * v_cam_z)
                        max_dot_prod = dot_prod_cam
                        target_label = "Camera"

                        if hand_detections:
                            h_diffs = [abs(t['timestamp_sec'] - current_t) for t in hand_detections]
                            if h_diffs:
                                closest_h_idx = h_diffs.index(min(h_diffs))
                                if h_diffs[closest_h_idx] <= 2.0:
                                    hands = hand_detections[closest_h_idx].get('hand_boxes', [])
                                    for h_box in hands:
                                        hx1, hy1, hx2, hy2 = h_box
                                        hcx = (hx1 + hx2) / 2.0
                                        hcy = (hy1 + hy2) / 2.0

                                        if intrinsics and 'focal_length' in intrinsics and 'principal_point' in intrinsics:
                                            v_hand_x = px - hcx
                                            v_hand_y = py - hcy
                                            v_hand_z = -float(fx)
                                        else:
                                            v_hand_x = (w / 2.0) - hcx
                                            v_hand_y = (h / 2.0) - hcy
                                            v_hand_z = -float(w)

                                        norm_hand = math.sqrt(v_hand_x**2 + v_hand_y**2 + v_hand_z**2)
                                        if norm_hand > 0:
                                            v_hand_x /= norm_hand
                                            v_hand_y /= norm_hand
                                            v_hand_z /= norm_hand

                                        dot_prod_hand = (v_look_x * v_hand_x) + (v_look_y * v_hand_y) + (v_look_z * v_hand_z)
                                        if dot_prod_hand > max_dot_prod:
                                            max_dot_prod = dot_prod_hand
                                            target_label = "POV_Actor_Hands"

                        mapped_score = max(0.0, (max_dot_prod - 0.5) * 2.0)
                        score = round(min(1.0, mapped_score), 2)
                    except Exception as e:
                        print(f"[03a] Gaze processing failed for person {pid} at t={current_t:.2f}s: {e}")

                traces[pid].append({
                    "t": round(current_t, 2),
                    "score": score,
                    "pitch_rad": round(pitch_rad, 4),
                    "yaw_rad": round(yaw_rad, 4),
                    "target": target_label,
                    "head_pitch_rad": head_pitch_rad,
                    "head_yaw_rad": head_yaw_rad
                })

                if last_scores[pid] >= 0.0 and abs(score - last_scores[pid]) > self.BURST_DELTA_THRESHOLD:
                    burst_until_t[pid] = current_t + self.BURST_DURATION_SEC

                stride = self.burst_stride_sec if current_t < burst_until_t[pid] else self.BASELINE_STRIDE_SEC
                next_t[pid] += stride
                last_scores[pid] = score

        return traces


# ---------------------------------------------------------------------------
# Spawn-safe parallel workers (Resolved Issue 6, June 13). Module-level so they
# are picklable under the macOS 'spawn' start method. Each worker process holds
# one AttentionLayerPipeline (models loaded once) in `_WORKER_PIPE` and only
# ever calls the unchanged `process_video`, so per-clip output is identical to
# a serial run.
# ---------------------------------------------------------------------------
_WORKER_PIPE = None


def _par_worker_init(manifest_path, output_path):
    global _WORKER_PIPE
    try:
        cv2.setNumThreads(1)  # each worker single-threaded; parallelism is across clips
    except Exception:
        pass
    # force=True: the worker never resumes/writes the result list (the parent
    # owns that) and never re-runs the face-quality gate (done in the parent) —
    # it only calls process_video.
    _WORKER_PIPE = AttentionLayerPipeline(manifest_path, output_path, force=True)
    _WORKER_PIPE.enable_face_quality_gate = False


def _par_worker_process(entry):
    """Process one clip in a worker; returns (video_id, result|None, err|None)."""
    video_id = entry.get('id', entry.get('video_id'))
    try:
        return video_id, _WORKER_PIPE.process_video(entry), None
    except Exception:
        return video_id, None, traceback.format_exc()


if __name__ == "__main__":
    # Guarded CLI — the spawn-safe entry point for parallel 03a. A runner script
    # that is NOT __main__-guarded must not request workers>1, or macOS 'spawn'
    # will re-exec it in every worker.
    #   PYTHONPATH=src python -m layer_03a_attention.pipeline <manifest> <out> --workers N
    import argparse
    ap = argparse.ArgumentParser(description="Run Layer 03a (Attention); optionally parallel across clips.")
    ap.add_argument("manifest")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    AttentionLayerPipeline(a.manifest, a.output, force=a.force).run(workers=a.workers)
