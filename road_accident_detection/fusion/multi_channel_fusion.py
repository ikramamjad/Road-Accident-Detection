"""
Multi-Channel Fusion Engine.
Combines Detection Confidence, Pose Anomaly, Kinematic Risk, and Temporal Anomaly
into an explainable, calibrated risk score.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ChannelScores:
    """Individual normalized channel scores for a single frame [0.0, 1.0]."""
    detection_conf: float
    pose_anomaly: float
    kinematic_risk: float
    temporal_anomaly: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.detection_conf,
            self.pose_anomaly,
            self.kinematic_risk,
            self.temporal_anomaly,
        ], dtype=np.float32)


@dataclass
class FusedRiskResult:
    """Consolidated multi-channel fusion output."""
    fused_score: float  # [0.0, 1.0]
    calibrated_probability: float  # [0.0, 1.0]
    channel_attributions: Dict[str, float]
    primary_driver: str
    secondary_driver: Optional[str]
    is_candidate_trigger: bool
    explanation: str


class LearnedFusionMLP(nn.Module):
    """Small neural MLP for non-linear cross-channel fusion."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiChannelFusionEngine:
    """
    Fuses multiple heterogeneous channels with support for:
    - Weighted Sum (linear, fully explainable)
    - Learned MLP (cross-channel feature interactions)
    - Rule-Gated (safety override rules)
    """

    def __init__(
        self,
        method: str = "weighted_sum",
        weights: Optional[Dict[str, float]] = None,
        trigger_threshold: float = 0.68,
    ):
        self.method = method
        self.trigger_threshold = trigger_threshold

        # Default normalized weights
        if weights is None:
            self.weights = {
                "detection_conf": 0.10,
                "pose_anomaly": 0.35,
                "kinematic_risk": 0.35,
                "temporal_anomaly": 0.20,
            }
        else:
            total = sum(weights.values())
            self.weights = {k: v / max(total, 1e-6) for k, v in weights.items()}

        self.mlp_model = LearnedFusionMLP()
        self.mlp_model.eval()

    def fuse(self, scores: ChannelScores) -> FusedRiskResult:
        """Combine channel scores into a unified risk estimate."""
        if self.method == "learned_mlp":
            fused = self._fuse_mlp(scores)
        elif self.method == "rule_gated":
            fused = self._fuse_rule_gated(scores)
        else:
            fused = self._fuse_weighted_sum(scores)

        fused = float(np.clip(fused, 0.0, 1.0))

        # Channel attribution vector (w_i * s_i)
        attributions = {
            "detection_conf": float(self.weights["detection_conf"] * scores.detection_conf),
            "pose_anomaly": float(self.weights["pose_anomaly"] * scores.pose_anomaly),
            "kinematic_risk": float(self.weights["kinematic_risk"] * scores.kinematic_risk),
            "temporal_anomaly": float(self.weights["temporal_anomaly"] * scores.temporal_anomaly),
        }

        sorted_channels = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        primary_driver = sorted_channels[0][0]
        secondary_driver = sorted_channels[1][0] if len(sorted_channels) > 1 else None

        # Simple logistic calibration default (can be updated by ConfidenceCalibrator)
        calibrated_prob = float(1.0 / (1.0 + np.exp(-7.0 * (fused - 0.5))))

        is_trigger = fused >= self.trigger_threshold

        # Human-readable explanation
        reasons = []
        if scores.kinematic_risk >= 0.6:
            reasons.append(f"High kinematic shock (score: {scores.kinematic_risk:.2f})")
        if scores.pose_anomaly >= 0.6:
            reasons.append(f"Severe posture anomaly / fall (score: {scores.pose_anomaly:.2f})")
        if scores.temporal_anomaly >= 0.55:
            reasons.append(f"Sustained temporal pattern (score: {scores.temporal_anomaly:.2f})")

        explanation = "; ".join(reasons) if reasons else f"Low anomaly risk (fused score: {fused:.2f})"

        return FusedRiskResult(
            fused_score=fused,
            calibrated_probability=calibrated_prob,
            channel_attributions=attributions,
            primary_driver=primary_driver,
            secondary_driver=secondary_driver,
            is_candidate_trigger=is_trigger,
            explanation=explanation,
        )

    def _fuse_weighted_sum(self, s: ChannelScores) -> float:
        return (
            self.weights["detection_conf"] * s.detection_conf +
            self.weights["pose_anomaly"] * s.pose_anomaly +
            self.weights["kinematic_risk"] * s.kinematic_risk +
            self.weights["temporal_anomaly"] * s.temporal_anomaly
        )

    def _fuse_mlp(self, s: ChannelScores) -> float:
        with torch.no_grad():
            inp = torch.tensor(s.to_array()).unsqueeze(0)
            out = self.mlp_model(inp)
            return float(out.item())

    def _fuse_rule_gated(self, s: ChannelScores) -> float:
        base = self._fuse_weighted_sum(s)
        # Safety rule 1: Severe rider ejection / fall with high kinematic shock overrides subtle temporal drift
        if s.pose_anomaly >= 0.85 and s.kinematic_risk >= 0.70:
            return max(base, 0.90)
        # Safety rule 2: Hard collision with zero posture change (car-on-car)
        if s.kinematic_risk >= 0.85 and s.temporal_anomaly >= 0.65:
            return max(base, 0.88)
        # Safety rule 3: Single-frame detector false positive suppression
        if s.temporal_anomaly < 0.15 and s.kinematic_risk < 0.20:
            return min(base, 0.35)
        return base
