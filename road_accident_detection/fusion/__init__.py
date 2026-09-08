"""Multi-channel fusion, temporal debouncing, calibration, and severity classification."""

from .multi_channel_fusion import ChannelScores, FusedRiskResult, MultiChannelFusionEngine
from .debouncer import TemporalDebouncer, DebounceStatus
from .calibrator import ConfidenceCalibrator
from .severity_classifier import SeverityClassifier, IncidentSeverity

__all__ = [
    "ChannelScores",
    "FusedRiskResult",
    "MultiChannelFusionEngine",
    "TemporalDebouncer",
    "DebounceStatus",
    "ConfidenceCalibrator",
    "SeverityClassifier",
    "IncidentSeverity",
]
