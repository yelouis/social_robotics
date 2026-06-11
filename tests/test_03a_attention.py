import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Assuming running from root directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from layer_03a_attention.pipeline import AttentionLayerPipeline

class TestAttentionLayerPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        self.input_manifest_path = self.temp_dir_path / "filtered_manifest.json"
        self.output_result_path = self.temp_dir_path / "03a_attention_result.json"
        
        # Dummy manifest
        dummy_manifest = [
            {
                "video_id": "test_video_1",
                "video_path": str(self.temp_dir_path / "dummy_video.mp4"),
                "bystander_detections": [
                    {
                        "person_id": 0,
                        "timestamps_sec": [0.0, 0.5, 1.0],
                        "bounding_boxes": [[10, 10, 50, 50], [10, 10, 50, 50], [10, 10, 50, 50]],
                        "detection_confidence": [0.9, 0.9, 0.9]
                    }
                ]
            }
        ]
        
        with open(self.input_manifest_path, 'w') as f:
            json.dump(dummy_manifest, f)
            
        # Create a dummy video file so path.exists() is true
        with open(self.temp_dir_path / "dummy_video.mp4", 'w') as f:
            f.write("dummy")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('layer_03a_attention.pipeline.cv2.VideoCapture')
    @patch('layer_03a_attention.pipeline.Pipeline')
    def test_pipeline_execution(self, mock_pipeline_class, mock_videocapture):
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 2.0 if prop == 5 else 3.0 # fps=2.0, total_frames=3.0 -> duration 1.5s
        
        # The pipeline uses cap.grab() + cap.retrieve() for sequential frame seeking
        # rather than cap.read(); mock both so unpacking succeeds.
        import numpy as np
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.grab.return_value = True
        mock_cap.retrieve.return_value = (True, dummy_frame)

        mock_videocapture.return_value = mock_cap
        
        # Mock L2CS Pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        # Create a dummy result for step()
        class DummyResult:
            def __init__(self):
                # Simulated looking straight ahead (pitch=0, yaw=0) -> should give max dot product
                self.pitch = np.array([[0.0]])
                self.yaw = np.array([[0.0]])
        
        mock_pipeline_instance.step.return_value = DummyResult()
        
        pipeline = AttentionLayerPipeline(
            input_manifest_path=self.input_manifest_path,
            output_result_path=self.output_result_path,
            force=True
        )

        # This test exercises the gaze dot-product -> attention-score math, so
        # make it independent of whether the real L2CS weights / BlazeFace / YOLO
        # assets are present on disk:
        #  - pin the mocked gaze pipeline (the __init__ guard sets it to None when
        #    the .pkl weights are absent, which would zero every score);
        #  - disable the face gate (Resolved Issue 1) and bbox re-detect (Resolved
        #    Issue 2): the synthetic all-zeros mock frame has no real face, so the
        #    gate would correctly score it NoFace/0.0 and the scoring path under
        #    test would never run. The gate's own behavior is covered separately.
        pipeline.gaze_pipeline = mock_pipeline_instance
        pipeline.enable_face_gate = False
        pipeline.enable_bbox_redetect = False
        # Resolved Issue 6: the dummy fixture has no real video, so the
        # face-quality pre-pass would score it zero-face and the clip gate
        # would (correctly) skip it — disable for this scoring-math test.
        pipeline.enable_face_quality_gate = False

        pipeline.run()

        self.assertTrue(self.output_result_path.exists())
        with open(self.output_result_path, 'r') as f:
            results = json.load(f)
            
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res['video_id'], 'test_video_1')
        self.assertEqual(res['aggregate']['num_bystanders_tracked'], 1)
        self.assertTrue(res['aggregate']['any_person_engaged'])
        
        p0 = res['per_person'][0]
        self.assertEqual(p0['person_id'], 0)
        # With pitch=0 and yaw=0, the dot product will be high but depends on bbox center vs screen center
        # Since box is [10, 10, 50, 50] (center 30,30) and screen is 100x100 (center 50,50), vector is (20, 20, 100)
        # Normalized: ~ (0.19, 0.19, 0.96)
        # V_look is (0, 0, 1). Dot product is 0.96.
        # mapped score = max(0, (0.96-0.5)*2) = 0.92
        self.assertEqual(p0['average_attention_score'], 0.92)
        self.assertTrue(p0['is_engaged'])
        self.assertTrue(len(p0['attention_trace']) > 0)
        
        # Check trace
        trace = p0['attention_trace']
        self.assertEqual(trace[0]['score'], 0.92)
        self.assertEqual(trace[0]['pitch_rad'], 0.0)
        self.assertEqual(trace[0]['yaw_rad'], 0.0)

    @patch('layer_03a_attention.pipeline.cv2.VideoCapture')
    @patch('layer_03a_attention.pipeline.Pipeline')
    def test_resumability(self, mock_pipeline_class, mock_videocapture):
        # Create a dummy result file so it thinks it's processed
        dummy_result = [
            {
                "video_id": "test_video_1",
                "layer": "03a_attention",
                "dummy": True
            }
        ]
        with open(self.output_result_path, 'w') as f:
            json.dump(dummy_result, f)
            
        pipeline = AttentionLayerPipeline(
            input_manifest_path=self.input_manifest_path,
            output_result_path=self.output_result_path,
            force=False
        )
        
        pipeline.run()
        
        # Because we didn't force, pipeline step should not have been called
        mock_pipeline_class.return_value.step.assert_not_called()
        
        # Result file should remain the same
        with open(self.output_result_path, 'r') as f:
            results = json.load(f)
        self.assertTrue(results[0].get('dummy'))

class TestThroughputLevers(unittest.TestCase):
    """Resolved Issue 6: A' face-quality clip gate at the 03a tier (60px/0.6/2,
    looser than 03b's because L2CS gaze resolves smaller faces than HSEmotion
    emotion) and B' cadence-decoupled YOLO re-detect."""

    def _bare_pipeline(self):
        with patch.object(AttentionLayerPipeline, "__init__", lambda s: None):
            return AttentionLayerPipeline()

    def test_face_quality_gate_tier(self):
        p = self._bare_pipeline()
        # The real June-11 cases: 10167fcf (57px) fails the 60px floor even at
        # high conf; 3f503d0b-like (99px/0.71/2) passes the 03a tier although it
        # fails 03b's strict tier — the clip carried real attention (eng 0.68).
        below = {"bystander_face_quality": {"best_face_px": 57, "best_face_conf": 0.84, "n_face_frames": 2}}
        above = {"bystander_face_quality": {"best_face_px": 99, "best_face_conf": 0.71, "n_face_frames": 2}}
        self.assertFalse(p._passes_face_quality(below))
        self.assertTrue(p._passes_face_quality(above))

    def test_face_quality_gate_fails_open_when_unscored(self):
        p = self._bare_pipeline()
        # Entries never pre-scored (no bystander_face_quality field) must pass.
        self.assertTrue(p._passes_face_quality({}))

    @patch('layer_03a_attention.pipeline.cv2.VideoCapture')
    @patch('layer_03a_attention.pipeline.Pipeline')
    def test_yolo_redetect_cadence_decoupled(self, mock_pipeline_class, mock_videocapture):
        """B': 13 samples at the 8 FPS baseline over 1.5s must trigger only 7
        YOLO re-detects at the 0.25s interval (t=0.0, 0.25, ..., 1.5) — the
        gaze-sampling cadence itself is unchanged (13 trace points)."""
        import numpy as np
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 2.0 if prop == 5 else 3.0  # fps=2, frames=3 -> 1.5s
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.grab.return_value = True
        mock_cap.retrieve.return_value = (True, dummy_frame)
        mock_videocapture.return_value = mock_cap

        mock_pipeline_instance = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline_instance

        class DummyResult:
            def __init__(self):
                self.pitch = np.array([[0.0]])
                self.yaw = np.array([[0.0]])
        mock_pipeline_instance.step.return_value = DummyResult()

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp = Path(td.name)
        manifest = tmp / "m.json"
        with open(manifest, 'w') as f:
            json.dump([{
                "video_id": "v_cadence",
                "video_path": str(tmp / "dummy.mp4"),
                "bystander_detections": [{
                    "person_id": 0,
                    "timestamps_sec": [0.0, 0.5, 1.0],
                    "bounding_boxes": [[10, 10, 50, 50]] * 3,
                }],
            }], f)
        (tmp / "dummy.mp4").write_text("dummy")

        pipeline = AttentionLayerPipeline(manifest, tmp / "out.json", force=True)
        pipeline.gaze_pipeline = mock_pipeline_instance
        pipeline.enable_face_gate = False
        pipeline.enable_bbox_redetect = False
        pipeline.enable_face_quality_gate = False  # dummy fixture has no real video

        calls = []
        pipeline._detect_pose_boxes = lambda frame: calls.append(1) or []

        pipeline.run()

        with open(tmp / "out.json") as f:
            results = json.load(f)
        trace = results[0]["per_person"][0]["attention_trace"]
        self.assertEqual(len(trace), 13)   # sampling cadence untouched
        self.assertEqual(len(calls), 7)    # YOLO at 4 Hz, not per frame


class TestHostMemoryGate(unittest.TestCase):
    """Issue 1: the E2E orchestrator calls host_can_retain_resident() to decide
    whether to skip the post-run unload() on high-memory hosts."""

    def test_high_memory_host_retains_resident(self):
        mock_vm = MagicMock()
        mock_vm.total = 64 * 2**30
        with patch('psutil.virtual_memory', return_value=mock_vm):
            self.assertTrue(AttentionLayerPipeline.host_can_retain_resident())

    def test_constrained_host_unloads(self):
        mock_vm = MagicMock()
        mock_vm.total = 24 * 2**30
        with patch('psutil.virtual_memory', return_value=mock_vm):
            self.assertFalse(AttentionLayerPipeline.host_can_retain_resident())


class TestBurstStrideMemoryGate(unittest.TestCase):
    """Resolved Issue 15: burst stride switches between 16 FPS (Mac mini) and
    32 FPS (Mac Studio / M4 Max) based on host_can_retain_resident()."""

    def _build(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        tmp = Path(td.name)
        manifest = tmp / "manifest.json"
        with open(manifest, 'w') as f:
            json.dump([], f)
        return AttentionLayerPipeline(manifest, tmp / "out.json", force=True)

    def test_high_memory_host_selects_32_fps_burst(self):
        mock_vm = MagicMock()
        mock_vm.total = 64 * 2**30
        with patch('psutil.virtual_memory', return_value=mock_vm):
            pipeline = self._build()
        self.assertAlmostEqual(pipeline.burst_stride_sec, 1.0 / 32.0)

    def test_constrained_host_retains_16_fps_burst(self):
        mock_vm = MagicMock()
        mock_vm.total = 24 * 2**30
        with patch('psutil.virtual_memory', return_value=mock_vm):
            pipeline = self._build()
        self.assertAlmostEqual(pipeline.burst_stride_sec, 1.0 / 16.0)


if __name__ == '__main__':
    unittest.main()
