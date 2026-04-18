import numpy as np
import cv2
import os
from debug_overlay import DebugOverlay
from saf_pipeline_utils import DistributionMonitor, ConfidenceRouter, ReactionFilter

def test_debug_overlay():
    print("--- Testing DebugOverlay ---")
    overlay = DebugOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock data (Standard COCO indices)
    # 0: nose, 5: l_shoulder, 6: r_shoulder
    keypoints = np.zeros((17, 3))
    keypoints[0] = [320, 100, 0.9]
    keypoints[5] = [280, 150, 0.9]
    keypoints[6] = [360, 150, 0.9]
    
    action_labels = "c008 1.0 5.0;c106 2.0 6.0"
    
    overlay.annotate_frame(frame, action_labels, "FOCUSED", 0.95, keypoints=keypoints)
    cv2.imwrite("debug_smoke_test.jpg", frame)
    print("  ✅ Created debug_smoke_test.jpg")

def test_saf_utils():
    print("--- Testing SAF Utils ---")
    
    # Test Distribution Monitor
    print("Testing DistributionMonitor (Collapse Check)...")
    monitor = DistributionMonitor(buffer_size=5)
    for _ in range(5): monitor.add_score(0.5)
    is_collapsed = monitor.check_score_variance_for_collapse()
    print(f"  Result (constant scores): {is_collapsed} (Expected: True)")
    
    # Test Reaction Filter
    print("Testing ReactionFilter (Temporal Lag)...")
    lag1 = ReactionFilter.validate_reaction_lag(1.0, 1.05)
    lag2 = ReactionFilter.validate_reaction_lag(1.0, 1.20)
    lag3 = ReactionFilter.validate_reaction_lag(1.0, 7.0)
    
    print(f"  Result (50ms): {lag1}")
    print(f"  Result (200ms): {lag2}")
    print(f"  Result (6s): {lag3}")

if __name__ == "__main__":
    test_debug_overlay()
    test_saf_utils()
