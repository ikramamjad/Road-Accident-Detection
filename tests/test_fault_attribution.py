"""
Unit tests for Fault & Liability Attribution Engine (Road Accident Forensics).
"""

import unittest
from road_accident_detection.detection.classes import RoadClass
from road_accident_detection.tracking.tracker import TrackedObject
from road_accident_detection.tracking.trajectory_buffer import TrajectoryHistory
from road_accident_detection.pose.posture_analyzer import PostureAnomalyReport
from road_accident_detection.fusion.fault_attribution import (
    FaultAttributionEngine,
    ViolationType,
    FaultAttributionReport,
)


class TestFaultAttributionEngine(unittest.TestCase):
    """Test suite for causal fault attribution and legal liability reconstruction."""

    def _create_track_history(self, track_id: int, class_name: RoadClass, centers):
        history = TrajectoryHistory(track_id=track_id, class_name=class_name, fps=25.0)
        for idx, (cx, cy) in enumerate(centers):
            bbox = (cx - 20.0, cy - 20.0, cx + 20.0, cy + 20.0)
            history.add_observation(
                frame_idx=idx,
                timestamp=idx / 25.0,
                bbox=bbox,
                pose_anomaly_score=0.0,
                kinematic_risk=0.0,
            )
        return history

    def test_no_accident_returns_clean_report(self):
        """When no accident occurred, report should indicate zero liability and no fault."""
        report = FaultAttributionEngine.evaluate_fault(
            involved_tracks=[],
            trajectory_histories={},
            is_accident=False,
        )
        self.assertFalse(report.has_fault_determination)
        self.assertEqual(report.violation_category, ViolationType.NO_VIOLATION)
        self.assertEqual(report.primary_liability_percentage, 0.0)
        self.assertIn("traffic flow", report.reconstruction_narrative)

    def test_vehicle_vs_pedestrian_vru_collision(self):
        """Vehicle colliding with pedestrian should attribute primary liability to vehicle."""
        car = TrackedObject(
            track_id=1,
            class_name=RoadClass.CAR,
            raw_class_id=2,
            bbox=(100.0, 100.0, 150.0, 150.0),
            confidence=0.9,
            vx=15.0,
            vy=0.0,
        )
        ped = TrackedObject(
            track_id=2,
            class_name=RoadClass.PEDESTRIAN,
            raw_class_id=0,
            bbox=(140.0, 100.0, 160.0, 150.0),
            confidence=0.85,
            vx=1.0,
            vy=0.0,
        )

        h1 = self._create_track_history(1, RoadClass.CAR, [(50.0 + i * 15.0, 125.0) for i in range(10)])
        h2 = self._create_track_history(2, RoadClass.PEDESTRIAN, [(150.0, 125.0) for _ in range(10)])

        histories = {1: h1, 2: h2}
        report = FaultAttributionEngine.evaluate_fault(
            involved_tracks=[car, ped],
            trajectory_histories=histories,
            is_accident=True,
            pixels_per_meter=15.0,
        )

        self.assertTrue(report.has_fault_determination)
        self.assertEqual(report.primary_at_fault_id, 1)
        self.assertEqual(report.primary_at_fault_class, RoadClass.CAR)
        self.assertGreaterEqual(report.primary_liability_percentage, 80.0)
        self.assertEqual(report.secondary_involved_id, 2)
        self.assertEqual(report.secondary_involved_class, RoadClass.PEDESTRIAN)
        self.assertEqual(report.violation_category, ViolationType.FAILURE_TO_YIELD_VRU)
        self.assertIn("PRIMARY FAULT ATTRIBUTION", report.reconstruction_narrative)
        self.assertEqual(len(report.timeline_phases), 3)

    def test_unsafe_lane_cut_in(self):
        """Vehicle with high lateral velocity changing lanes abruptly into leading vehicle."""
        v1 = TrackedObject(
            track_id=10,
            class_name=RoadClass.CAR,
            raw_class_id=2,
            bbox=(100.0, 100.0, 140.0, 140.0),
            confidence=0.9,
            vx=12.0,
            vy=10.0,
        )
        v2 = TrackedObject(
            track_id=11,
            class_name=RoadClass.CAR,
            raw_class_id=2,
            bbox=(130.0, 100.0, 170.0, 140.0),
            confidence=0.9,
            vx=1.0,
            vy=10.0,
        )

        h1 = self._create_track_history(10, RoadClass.CAR, [(50.0 + i * 15.0, 100.0 + i * 10.0) for i in range(10)])
        h2 = self._create_track_history(11, RoadClass.CAR, [(150.0, 50.0 + i * 10.0) for i in range(10)])

        histories = {10: h1, 11: h2}
        report = FaultAttributionEngine.evaluate_fault(
            involved_tracks=[v1, v2],
            trajectory_histories=histories,
            is_accident=True,
            pixels_per_meter=15.0,
        )

        self.assertTrue(report.has_fault_determination)
        self.assertEqual(report.primary_at_fault_id, 10)
        self.assertEqual(report.violation_category, ViolationType.UNSAFE_LANE_CHANGE_CUT_IN)
        self.assertGreaterEqual(report.primary_liability_percentage, 80.0)

    def test_rear_end_tailgating(self):
        """Trailing vehicle striking rear of lead vehicle."""
        # Lead vehicle is ahead down-lane (y=440), trailing vehicle is behind (y=370) moving down fast
        lead = TrackedObject(
            track_id=20,
            class_name=RoadClass.BUS,
            raw_class_id=5,
            bbox=(100.0, 420.0, 160.0, 470.0),
            confidence=0.9,
            vx=0.0,
            vy=5.0,
        )
        follower = TrackedObject(
            track_id=21,
            class_name=RoadClass.CAR,
            raw_class_id=2,
            bbox=(105.0, 340.0, 155.0, 400.0),
            confidence=0.9,
            vx=0.0,
            vy=30.0,
        )

        h_lead = self._create_track_history(20, RoadClass.BUS, [(130.0, 400.0 + i * 5.0) for i in range(10)])
        h_foll = self._create_track_history(21, RoadClass.CAR, [(130.0, 100.0 + i * 30.0) for i in range(10)])

        histories = {20: h_lead, 21: h_foll}
        report = FaultAttributionEngine.evaluate_fault(
            involved_tracks=[follower, lead],
            trajectory_histories=histories,
            is_accident=True,
            pixels_per_meter=15.0,
        )

        self.assertTrue(report.has_fault_determination)
        self.assertEqual(report.primary_at_fault_id, 21)
        self.assertEqual(report.violation_category, ViolationType.TAILGATING_UNSAFE_DISTANCE)
        self.assertGreaterEqual(report.primary_liability_percentage, 85.0)

    def test_single_vehicle_rollover_or_spin(self):
        """Single vehicle spinning out or suffering sudden deceleration shock."""
        car = TrackedObject(
            track_id=5,
            class_name=RoadClass.CAR,
            raw_class_id=2,
            bbox=(200.0, 200.0, 260.0, 260.0),
            confidence=0.92,
            vx=0.0,
            vy=0.0,
        )
        centers = []
        for i in range(6):
            centers.append((100.0 + i * 30.0, 100.0))
        for i in range(4):
            centers.append((250.0, 100.0 + i * 1.0))

        history = self._create_track_history(5, RoadClass.CAR, centers)

        report = FaultAttributionEngine.evaluate_fault(
            involved_tracks=[car],
            trajectory_histories={5: history},
            is_accident=True,
            pixels_per_meter=15.0,
        )

        self.assertTrue(report.has_fault_determination)
        self.assertEqual(report.primary_at_fault_id, 5)
        self.assertEqual(report.primary_liability_percentage, 100.0)
        self.assertEqual(report.violation_category, ViolationType.SINGLE_VEHICLE_LOSS_OF_CONTROL)
        self.assertIn("100% at fault", report.reconstruction_narrative)


if __name__ == "__main__":
    unittest.main()
