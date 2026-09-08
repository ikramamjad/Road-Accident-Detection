"""Unit tests for MotionAnalyzer, TTCCalculator, and KinematicRiskEngine."""

import unittest
import numpy as np

from road_accident_detection.detection.classes import RoadClass
from road_accident_detection.tracking.tracker import TrackedObject
from road_accident_detection.tracking.trajectory_buffer import TrajectoryBufferManager
from road_accident_detection.kinematics.motion_analyzer import MotionAnalyzer
from road_accident_detection.kinematics.ttc_calculator import TTCCalculator
from road_accident_detection.kinematics.kinematic_risk import KinematicRiskEngine


class TestKinematicsModule(unittest.TestCase):

    def test_motion_analyzer_hard_braking(self):
        fps = 30.0
        ppm = 10.0
        analyzer = MotionAnalyzer(pixels_per_meter=ppm, fps=fps, hard_braking_threshold_m_s2=-4.0)

        buf = TrajectoryBufferManager(fps=fps)
        # Step 0: Moving at 20 px/frame (20 m/s)
        buf.update_track(1, RoadClass.CAR, 0, 0.0, (80, 80, 120, 120))
        # Step 1: Same speed
        buf.update_track(1, RoadClass.CAR, 1, 0.033, (100, 80, 140, 120))
        # Step 2: Severe drop in position advancement (severe deceleration)
        buf.update_track(1, RoadClass.CAR, 2, 0.066, (103, 80, 143, 120))

        hist = buf.get_history(1)
        state = analyzer.analyze(hist)
        self.assertIsNotNone(state)
        # Should detect negative longitudinal acceleration (braking)
        self.assertLess(state.longitudinal_accel_m_s2, -2.0)

    def test_ttc_calculator_converging_pair(self):
        fps = 30.0
        ppm = 10.0
        ttc_calc = TTCCalculator(pixels_per_meter=ppm, fps=fps, ttc_critical_thresh=2.0)

        buf = TrajectoryBufferManager(fps=fps)

        # Vehicle A at x=100, moving right at +10 px/frame
        buf.update_track(1, RoadClass.CAR, 0, 0.0, (80, 100, 120, 140))
        buf.update_track(1, RoadClass.CAR, 1, 0.033, (90, 100, 130, 140))

        # Vehicle B at x=300, moving left at -10 px/frame (head-on collision course)
        buf.update_track(2, RoadClass.CAR, 0, 0.0, (280, 100, 320, 140))
        buf.update_track(2, RoadClass.CAR, 1, 0.033, (270, 100, 310, 140))

        tracks = [
            TrackedObject(track_id=1, bbox=(90, 100, 130, 140), confidence=0.9, class_name=RoadClass.CAR, raw_class_id=2),
            TrackedObject(track_id=2, bbox=(270, 100, 310, 140), confidence=0.9, class_name=RoadClass.CAR, raw_class_id=2),
        ]

        results = ttc_calc.compute_pairwise_ttc(tracks, buf.tracks)
        self.assertEqual(len(results), 1)
        res = results[0]

        # Trajectories must be converging
        self.assertTrue(res.is_converging)
        self.assertIsNotNone(res.ttc_seconds)
        # Closing speed is (10 - (-10)) * 30 / 10 = 60 m/s
        # Distance is (290 - 110) / 10 = 18 meters -> TTC ~ 0.3s
        self.assertLess(res.ttc_seconds, 1.5)
        self.assertTrue(res.is_critical)
        self.assertGreater(res.risk_score, 0.7)


if __name__ == "__main__":
    unittest.main()
