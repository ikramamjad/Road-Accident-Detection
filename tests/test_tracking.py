"""Unit tests for BoT-SORT tracking and trajectory buffer manager."""

import unittest
import numpy as np

from road_accident_detection.detection.classes import RoadClass
from road_accident_detection.detection.yolo_detector import DetectionResult
from road_accident_detection.tracking.tracker import BoTSORTTracker, TrackedObject, compute_iou
from road_accident_detection.tracking.trajectory_buffer import TrajectoryBufferManager


class TestTrackingModule(unittest.TestCase):

    def test_compute_iou(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (5.0, 0.0, 15.0, 10.0)
        # Inter: 5x10 = 50, Union: 100 + 100 - 50 = 150 -> IoU = 50/150 = 1/3
        iou = compute_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=3)

        # Disjoint boxes
        box3 = (20.0, 20.0, 30.0, 30.0)
        self.assertEqual(compute_iou(box1, box3), 0.0)

    def test_tracker_persistent_ids(self):
        tracker = BoTSORTTracker(track_high_thresh=0.4, new_track_thresh=0.5, fps=30.0)

        # Frame 1: Detection at (100, 100, 200, 200)
        dets_f1 = [
            DetectionResult(bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.85, class_name=RoadClass.CAR, raw_class_id=2)
        ]
        tracks_f1 = tracker.update(dets_f1, frame_idx=1, timestamp=0.033)
        self.assertEqual(len(tracks_f1), 1)
        orig_id = tracks_f1[0].track_id

        # Frame 2: Slight movement to (105, 102, 205, 202)
        dets_f2 = [
            DetectionResult(bbox=(105.0, 102.0, 205.0, 202.0), confidence=0.84, class_name=RoadClass.CAR, raw_class_id=2)
        ]
        tracks_f2 = tracker.update(dets_f2, frame_idx=2, timestamp=0.066)
        self.assertEqual(len(tracks_f2), 1)
        # Verify persistent ID maintained
        self.assertEqual(tracks_f2[0].track_id, orig_id)

    def test_trajectory_buffer_kinematics(self):
        buf_mgr = TrajectoryBufferManager(max_history_len=20, fps=30.0)

        # Add observations over time
        for i in range(5):
            cx = 100.0 + i * 15.0  # moving right at 15px/frame (450px/sec)
            cy = 200.0
            pt = buf_mgr.update_track(
                track_id=1,
                class_name=RoadClass.CAR,
                frame_idx=i,
                timestamp=i / 30.0,
                bbox=(cx - 20, cy - 20, cx + 20, cy + 20),
            )

        hist = buf_mgr.get_history(1)
        self.assertIsNotNone(hist)
        self.assertEqual(len(hist.history), 5)
        # Speed should be positive and close to 15 px/frame
        self.assertGreater(hist.latest.speed, 5.0)

        # Check feature matrix shape
        feat = hist.get_feature_matrix(img_w=1920, img_h=1080, target_len=16)
        self.assertEqual(feat.shape, (16, 12))


if __name__ == "__main__":
    unittest.main()
