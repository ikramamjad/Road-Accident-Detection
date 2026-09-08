"""Unit tests for Multi-Channel Fusion, Debouncing, Calibration, and Severity."""

import unittest
import numpy as np

from road_accident_detection.detection.classes import RoadClass
from road_accident_detection.tracking.tracker import TrackedObject
from road_accident_detection.pose.posture_analyzer import PostureAnomalyReport
from road_accident_detection.kinematics.motion_analyzer import KinematicState
from road_accident_detection.fusion.multi_channel_fusion import (
    ChannelScores,
    MultiChannelFusionEngine,
)
from road_accident_detection.fusion.debouncer import TemporalDebouncer, DebounceStatus
from road_accident_detection.fusion.calibrator import ConfidenceCalibrator
from road_accident_detection.fusion.severity_classifier import (
    SeverityClassifier,
    IncidentSeverity,
)


class TestFusionModule(unittest.TestCase):

    def test_multi_channel_fusion_weighted_sum(self):
        engine = MultiChannelFusionEngine(method="weighted_sum", trigger_threshold=0.68)

        # Baseline nominal conditions
        scores_nominal = ChannelScores(
            detection_conf=0.9, pose_anomaly=0.0, kinematic_risk=0.05, temporal_anomaly=0.05
        )
        res_nom = engine.fuse(scores_nominal)
        self.assertLess(res_nom.fused_score, 0.40)
        self.assertFalse(res_nom.is_candidate_trigger)

        # Severe multi-channel accident
        scores_crash = ChannelScores(
            detection_conf=0.95, pose_anomaly=0.90, kinematic_risk=0.95, temporal_anomaly=0.85
        )
        res_crash = engine.fuse(scores_crash)
        self.assertGreater(res_crash.fused_score, 0.75)
        self.assertTrue(res_crash.is_candidate_trigger)
        self.assertIn(res_crash.primary_driver, ["pose_anomaly", "kinematic_risk"])

    def test_temporal_debouncing_suppresses_spikes(self):
        debouncer = TemporalDebouncer(
            trigger_threshold=0.70,
            consecutive_frames_required=5,
            cooldown_frames=20,
        )

        # Single frame false alarm spike (e.g. tracking glitch)
        out1 = debouncer.update(0.95)
        self.assertEqual(out1.status, DebounceStatus.DEBOUNCING)
        self.assertFalse(out1.is_event_fired)

        # Drop back down to normal -> must reset without triggering
        out2 = debouncer.update(0.20)
        self.assertEqual(out2.status, DebounceStatus.MONITORING)
        self.assertFalse(out2.is_event_fired)
        self.assertEqual(debouncer.consecutive_positive, 0)

    def test_temporal_debouncing_fires_on_sustained(self):
        debouncer = TemporalDebouncer(
            trigger_threshold=0.70,
            consecutive_frames_required=4,
            cooldown_frames=20,
        )

        # Feed 3 high frames -> still debouncing
        for _ in range(3):
            out = debouncer.update(0.85)
            self.assertEqual(out.status, DebounceStatus.DEBOUNCING)
            self.assertFalse(out.is_event_fired)

        # 4th high frame -> EVENT FIRED!
        out4 = debouncer.update(0.85)
        self.assertEqual(out4.status, DebounceStatus.ACCIDENT_TRIGGERED)
        self.assertTrue(out4.is_event_fired)

        # 5th frame -> enter cooldown
        out5 = debouncer.update(0.85)
        self.assertEqual(out5.status, DebounceStatus.COOLDOWN)
        self.assertFalse(out5.is_event_fired)

    def test_confidence_calibration(self):
        calibrator = ConfidenceCalibrator()
        p_low = calibrator.calibrate(0.1)
        p_high = calibrator.calibrate(0.9)
        self.assertLess(p_low, 0.1)
        self.assertGreater(p_high, 0.9)

        # ECE check
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = ConfidenceCalibrator.compute_ece(probs, labels)
        self.assertLessEqual(ece, 0.25)

    def test_severity_classification(self):
        tracks = [
            TrackedObject(track_id=1, bbox=(0, 0, 50, 50), confidence=0.9, class_name=RoadClass.CAR, raw_class_id=2),
            TrackedObject(track_id=2, bbox=(0, 0, 50, 50), confidence=0.9, class_name=RoadClass.CAR, raw_class_id=2),
            TrackedObject(track_id=3, bbox=(0, 0, 50, 50), confidence=0.9, class_name=RoadClass.TRUCK, raw_class_id=7),
        ]

        # Multi-vehicle collision
        assess_mv = SeverityClassifier.classify(
            involved_tracks=tracks,
            kinematic_states=[],
            posture_reports=[],
            is_near_miss_trigger=False,
        )
        self.assertEqual(assess_mv.severity, IncidentSeverity.MULTI_VEHICLE)

        # Vulnerable road user fall -> Major
        ped_track = [
            TrackedObject(track_id=1, bbox=(0, 0, 50, 50), confidence=0.9, class_name=RoadClass.PEDESTRIAN, raw_class_id=0)
        ]
        fall_report = [
            PostureAnomalyReport(
                track_id=1,
                anomaly_score=0.9,
                is_fall=True,
                is_ejection=False,
                is_prolonged_ground=True,
                torso_angle_deg=10.0,
                aspect_ratio=0.3,
                consecutive_ground_frames=15,
                explanation="Fall",
            )
        ]
        assess_major = SeverityClassifier.classify(
            involved_tracks=ped_track,
            kinematic_states=[],
            posture_reports=fall_report,
            is_near_miss_trigger=False,
        )
        self.assertEqual(assess_major.severity, IncidentSeverity.MAJOR)


if __name__ == "__main__":
    unittest.main()
