"""Evaluation and stratified literature benchmarking module."""

from .metrics import EventMetrics, compute_event_metrics, DetectionTrackingMetrics
from .stratified_eval import StratifiedEvaluator, StratifiedResultsReport, CCD_LITERATURE_BASELINES

__all__ = [
    "EventMetrics",
    "compute_event_metrics",
    "DetectionTrackingMetrics",
    "StratifiedEvaluator",
    "StratifiedResultsReport",
    "CCD_LITERATURE_BASELINES",
]
