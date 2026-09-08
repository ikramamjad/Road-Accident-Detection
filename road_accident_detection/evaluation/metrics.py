"""
Evaluation Metrics for Road Accident Detection.
Computes Event Precision, Recall, F1, Time-to-Detection (TTD), False Alarm Rate (FAR),
and Detection/Tracking summary metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EventMetrics:
    total_incidents: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_time_to_detection_sec: float  # Latency from collision onset to alert trigger
    false_alarm_rate_per_hour: float


@dataclass
class DetectionTrackingMetrics:
    map_50: float
    map_50_95: float
    mota: float
    idf1: float


def compute_event_metrics(
    predictions: List[Dict],  # Each dict: {"clip_id": str, "detected": bool, "detection_time_sec": float, "is_accident": bool, "onset_time_sec": float, "clip_duration_sec": float}
) -> EventMetrics:
    """Compute event-level detection accuracy and timeliness."""
    tp = 0
    fp = 0
    fn = 0
    ttd_list: List[float] = []
    total_video_seconds = 0.0

    for p in predictions:
        is_ground_truth_accident = p.get("is_accident", False)
        was_detected = p.get("detected", False)
        duration = p.get("clip_duration_sec", 10.0)
        total_video_seconds += duration

        if is_ground_truth_accident and was_detected:
            tp += 1
            onset = p.get("onset_time_sec", 0.0)
            det_t = p.get("detection_time_sec", onset)
            # Latency (can be negative if anticipated prior to impact, or positive after)
            ttd = max(0.0, det_t - onset)
            ttd_list.append(ttd)
        elif not is_ground_truth_accident and was_detected:
            fp += 1
        elif is_ground_truth_accident and not was_detected:
            fn += 1

    total_incidents = tp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)

    mean_ttd = float(np.mean(ttd_list)) if ttd_list else 0.0
    video_hours = total_video_seconds / 3600.0
    far = fp / max(video_hours, 1e-4)

    return EventMetrics(
        total_incidents=total_incidents,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        mean_time_to_detection_sec=round(mean_ttd, 3),
        false_alarm_rate_per_hour=round(far, 3),
    )
