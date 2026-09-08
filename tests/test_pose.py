"""Unit tests for YOLO11-Pose estimation and abnormal posture analyzer."""

import unittest
import numpy as np

from road_accident_detection.pose.pose_detector import PersonPose, YOLO11PoseDetector
from road_accident_detection.pose.posture_analyzer import PostureAnalyzer


class TestPoseModule(unittest.TestCase):

    def test_upright_pose_normal_score(self):
        analyzer = PostureAnalyzer(fall_torso_angle_thresh=35.0)

        # Construct upright keypoints (shoulders at y=100, hips at y=200)
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[5] = [100.0, 100.0, 0.9]  # left shoulder
        kpts[6] = [120.0, 100.0, 0.9]  # right shoulder
        kpts[11] = [102.0, 200.0, 0.9] # left hip
        kpts[12] = [118.0, 200.0, 0.9] # right hip

        pose = PersonPose(
            keypoints=kpts,
            bbox=(80.0, 80.0, 140.0, 260.0), # tall box: h=180, w=60, aspect ratio=3.0
            confidence=0.9,
        )

        rep = analyzer.analyze_pose(pose, track_id=1)
        self.assertFalse(rep.is_fall)
        self.assertGreater(rep.torso_angle_deg, 75.0)
        self.assertLess(rep.anomaly_score, 0.35)

    def test_horizontal_fall_pose_high_score(self):
        analyzer = PostureAnalyzer(fall_torso_angle_thresh=35.0)

        # Construct fallen/horizontal keypoints (shoulders and hips on same horizontal plane)
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[5] = [100.0, 200.0, 0.9]  # left shoulder
        kpts[6] = [100.0, 210.0, 0.9]  # right shoulder
        kpts[11] = [220.0, 202.0, 0.9] # left hip
        kpts[12] = [220.0, 212.0, 0.9] # right hip

        pose = PersonPose(
            keypoints=kpts,
            bbox=(80.0, 190.0, 260.0, 230.0), # flat box: w=180, h=40, aspect ratio=0.22
            confidence=0.88,
        )

        rep = analyzer.analyze_pose(pose, track_id=2)
        self.assertTrue(rep.is_fall)
        self.assertLess(rep.torso_angle_deg, 20.0)
        self.assertGreater(rep.anomaly_score, 0.65)


if __name__ == "__main__":
    unittest.main()
