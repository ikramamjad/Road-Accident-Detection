"""Integration test for full end-to-end AccidentDetectionPipeline."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from road_accident_detection.pipeline.engine import AccidentDetectionPipeline
from road_accident_detection.fusion.debouncer import DebounceStatus


class TestEndToEndPipeline(unittest.TestCase):

    def test_pipeline_execution(self):
        pipeline = AccidentDetectionPipeline(mock_mode=True)
        dummy_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)

        # Run pipeline over 10 frames
        results = []
        for i in range(10):
            res = pipeline.process_frame(dummy_frame, render_annotated=True)
            results.append(res)

        self.assertEqual(len(results), 10)
        last_res = results[-1]

        # Verify components in result
        self.assertIsNotNone(last_res.fused_risk)
        self.assertIsNotNone(last_res.debounce_output)
        self.assertIsNotNone(last_res.kinematic_report)
        self.assertIsNotNone(last_res.annotated_frame)
        self.assertEqual(last_res.annotated_frame.shape, (720, 1280, 3))

        # Verify profiler recorded stage latencies
        summary = pipeline.profiler.summary()
        self.assertGreater(len(summary), 5)
        stage_names = [s.stage_name for s in summary]
        self.assertIn("yolo_detection", stage_names)
        self.assertIn("botsort_tracking", stage_names)
        self.assertIn("pose_estimation", stage_names)
        self.assertIn("kinematics_and_ttc", stage_names)
        self.assertIn("causal_gru", stage_names)
        self.assertIn("multi_channel_fusion", stage_names)
        self.assertIn("debouncing", stage_names)

        # Throughput FPS should be positive
        self.assertGreater(pipeline.profiler.get_fps(), 0.0)


if __name__ == "__main__":
    unittest.main()
