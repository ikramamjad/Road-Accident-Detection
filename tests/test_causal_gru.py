"""Unit tests for Causal GRU temporal modeling and strict causality verification."""

import unittest
import torch
import numpy as np

from road_accident_detection.temporal.causal_gru import CausalGRUModel, TemporalInferenceEngine


class TestCausalGRUModule(unittest.TestCase):

    def test_forward_shape_and_range(self):
        model = CausalGRUModel(input_dim=12, hidden_dim=64, num_layers=2)
        model.eval()

        batch_size = 4
        seq_len = 16
        dummy_x = torch.randn(batch_size, seq_len, 12)

        score, hidden = model(dummy_x)
        self.assertEqual(score.shape, (batch_size, 1))
        self.assertEqual(hidden.shape, (2, batch_size, 64))

        # Output score must be sigmoid bounded [0.0, 1.0]
        self.assertTrue(torch.all(score >= 0.0))
        self.assertTrue(torch.all(score <= 1.0))

    def test_strict_causality_no_future_leakage(self):
        """
        Verify that perturbing future frames in a sequence does NOT affect
        the GRU hidden state at prior or current time steps.
        """
        model = CausalGRUModel(input_dim=12, hidden_dim=64, num_layers=2)
        model.eval()

        seq_len = 10
        x1 = torch.randn(1, seq_len, 12)
        # Create x2 identical to x1 up to index 6, but perturbed at indices 7, 8, 9 (future)
        x2 = x1.clone()
        x2[:, 7:, :] += 5.0

        with torch.no_grad():
            proj1 = model.input_layer(x1)
            out1, _ = model.gru(proj1)

            proj2 = model.input_layer(x2)
            out2, _ = model.gru(proj2)

        # The GRU representation at time step t=6 MUST be identical in both sequences
        diff_at_t6 = torch.max(torch.abs(out1[:, 6, :] - out2[:, 6, :])).item()
        self.assertAlmostEqual(diff_at_t6, 0.0, places=5)

    def test_inference_engine_wrapper(self):
        engine = TemporalInferenceEngine(input_dim=12, hidden_dim=64, num_layers=2)
        feat = np.zeros((16, 12), dtype=np.float32)
        score = engine.evaluate_sequence(feat)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
