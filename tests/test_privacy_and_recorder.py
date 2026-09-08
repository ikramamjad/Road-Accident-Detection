"""Unit tests for Privacy Redactor and Circular Clip Recorder."""

import os
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from road_accident_detection.detection.classes import RoadClass
from road_accident_detection.tracking.tracker import TrackedObject
from road_accident_detection.privacy.face_plate_redaction import PrivacyRedactor
from road_accident_detection.privacy.retention_policy import RetentionPolicyManager
from road_accident_detection.logging.clip_recorder import CircularClipRecorder, RecordingTrigger


class TestPrivacyAndRecorder(unittest.TestCase):

    def test_privacy_redactor_alters_license_plate(self):
        redactor = PrivacyRedactor(redact_faces=True, redact_plates=True, blur_kernel_size=15)
        # Create non-uniform test image with high-frequency pattern
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

        # Vehicle at [100, 100, 300, 300]
        vehicle_track = TrackedObject(
            track_id=1,
            bbox=(100.0, 100.0, 300.0, 300.0),
            confidence=0.9,
            class_name=RoadClass.CAR,
            raw_class_id=2,
        )

        redacted = redactor.redact_frame(frame, [vehicle_track], poses=[])

        # Plate area (lower portion of vehicle box)
        orig_roi = frame[260:290, 150:250]
        redacted_roi = redacted[260:290, 150:250]

        # Redacted ROI must differ from original because of Gaussian blur
        diff = np.max(np.abs(orig_roi.astype(float) - redacted_roi.astype(float)))
        self.assertGreater(diff, 5.0)

        # Area outside bounding box must remain identical
        self.assertTrue(np.array_equal(frame[10:50, 10:50], redacted[10:50, 10:50]))

    def test_circular_clip_recorder_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = CircularClipRecorder(
                output_dir=tmpdir,
                pre_buffer_seconds=1,
                post_buffer_seconds=1,
                fps=10.0,
            )
            dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

            # Feed 15 frames (pre-buffer maxlen is 10)
            for _ in range(15):
                recorder.add_frame(dummy_frame)

            self.assertEqual(len(recorder.ring_buffer), 10)

            # Trigger recording
            trigger = RecordingTrigger(
                event_id="test1234",
                timestamp_str="20260908",
                severity="major",
                camera_id="CAM_TEST",
            )
            recorder.trigger_recording(trigger)

            # Feed 10 post-incident frames to complete the clip
            completed = []
            for _ in range(10):
                res = recorder.add_frame(dummy_frame)
                completed.extend(res)

            self.assertEqual(len(completed), 1)
            self.assertTrue(completed[0].endswith(".mp4"))


if __name__ == "__main__":
    unittest.main()
