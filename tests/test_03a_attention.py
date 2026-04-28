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
        
        # Mock cap.read() returning True and a dummy frame
        import numpy as np
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        
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

if __name__ == '__main__':
    unittest.main()
