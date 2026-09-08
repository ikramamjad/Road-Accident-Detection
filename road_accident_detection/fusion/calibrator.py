"""
Confidence Calibrator.
Applies Platt scaling / Logistic regression calibration to convert raw fused scores
into true empirical event probabilities.
"""

from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression


class ConfidenceCalibrator:
    """
    Calibrates continuous fused anomaly scores into well-calibrated probabilities.
    Minimizes Expected Calibration Error (ECE).
    """

    def __init__(self, default_scale: float = 8.0, default_bias: float = -4.5):
        self.scale = default_scale
        self.bias = default_bias
        self.is_fitted = False
        self._model = LogisticRegression(C=1.0, solver="lbfgs")

    def calibrate(self, raw_score: float) -> float:
        """Map raw score [0, 1] to calibrated probability [0, 1]."""
        if self.is_fitted:
            prob = self._model.predict_proba([[raw_score]])[0, 1]
            return float(np.clip(prob, 0.0, 1.0))
        # Analytical logistic sigmoid: sigma(A * s + B)
        logit = self.scale * raw_score + self.bias
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(prob, 0.0, 1.0))

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit logistic calibrator on validation predictions and binary truth labels."""
        X = scores.reshape(-1, 1)
        y = labels.ravel()
        if len(np.unique(y)) < 2:
            return  # Need both positive and negative labels
        self._model.fit(X, y)
        self.is_fitted = True

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_limits = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(probs)
        if n == 0:
            return 0.0

        for i in range(n_bins):
            bin_mask = (probs >= bin_limits[i]) & (probs < bin_limits[i + 1])
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                bin_acc = np.mean(labels[bin_mask])
                bin_conf = np.mean(probs[bin_mask])
                ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return float(ece)
