"""Unit tests for detection classes and YOLO11 detector wrapper."""

import unittest
import numpy as np

from road_accident_detection.detection.classes import (
    RoadClass,
    COCO_TO_UNIFIED,
    IDD_TO_UNIFIED,
    is_vehicle,
    is_vulnerable_road_user,
    is_two_wheeler,
)
from road_accident_detection.detection.yolo_detector import YOLO11Detector, DetectionResult


class TestDetectionModule(unittest.TestCase):

    def test_class_mappings(self):
        # Test COCO mappings
        self.assertEqual(COCO_TO_UNIFIED[0], RoadClass.PEDESTRIAN)
        self.assertEqual(COCO_TO_UNIFIED[2], RoadClass.CAR)
        self.assertEqual(COCO_TO_UNIFIED[3], RoadClass.MOTORCYCLE)
        self.assertEqual(COCO_TO_UNIFIED[5], RoadClass.BUS)
        self.assertEqual(COCO_TO_UNIFIED[7], RoadClass.TRUCK)

        # Test IDD mappings
        self.assertEqual(IDD_TO_UNIFIED["autorickshaw"], RoadClass.AUTO_RICKSHAW)
        self.assertEqual(IDD_TO_UNIFIED["person"], RoadClass.PEDESTRIAN)

        # Category checks
        self.assertTrue(is_vehicle(RoadClass.CAR))
        self.assertTrue(is_vehicle(RoadClass.AUTO_RICKSHAW))
        self.assertTrue(is_vulnerable_road_user(RoadClass.PEDESTRIAN))
        self.assertTrue(is_vulnerable_road_user(RoadClass.CYCLIST))
        self.assertTrue(is_two_wheeler(RoadClass.MOTORCYCLE))
        self.assertFalse(is_vehicle(RoadClass.PEDESTRIAN))

    def test_detection_result_properties(self):
        det = DetectionResult(
            bbox=(100.0, 150.0, 250.0, 300.0),
            confidence=0.88,
            class_name=RoadClass.CAR,
            raw_class_id=2,
        )
        self.assertEqual(det.x1, 100.0)
        self.assertEqual(det.y1, 150.0)
        self.assertEqual(det.width, 150.0)
        self.assertEqual(det.height, 150.0)
        self.assertEqual(det.center, (175.0, 225.0))
        self.assertEqual(det.area, 22500.0)

    def test_detector_mock_mode(self):
        detector = YOLO11Detector(mock_mode=True)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(dummy_frame)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], DetectionResult)


if __name__ == "__main__":
    unittest.main()
