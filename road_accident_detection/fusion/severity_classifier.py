"""
Severity and Near-Miss Classifier.
Categorizes detected road incidents into Minor, Major, Multi-Vehicle, or Near-Miss.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np

from ..detection.classes import is_vulnerable_road_user, RoadClass
from ..tracking.tracker import TrackedObject
from ..kinematics.motion_analyzer import KinematicState
from ..pose.posture_analyzer import PostureAnomalyReport


class IncidentSeverity(str, Enum):
    NEAR_MISS = "near_miss"
    MINOR = "minor"
    MAJOR = "major"
    MULTI_VEHICLE = "multi_vehicle"


@dataclass
class SeverityAssessment:
    severity: IncidentSeverity
    involved_track_ids: List[int]
    confidence: float
    justification: str


class SeverityClassifier:
    """Classifies post-trigger incident severity based on dynamics and participants."""

    @staticmethod
    def classify(
        involved_tracks: List[TrackedObject],
        kinematic_states: List[KinematicState],
        posture_reports: List[PostureAnomalyReport],
        is_near_miss_trigger: bool = False,
    ) -> SeverityAssessment:
        """Determine incident severity level."""
        track_ids = [t.track_id for t in involved_tracks]

        if is_near_miss_trigger:
            return SeverityAssessment(
                severity=IncidentSeverity.NEAR_MISS,
                involved_track_ids=track_ids,
                confidence=0.89,
                justification="High kinematic risk resolved without impact or posture loss (evasion successful)",
            )

        # Multi-vehicle check (>= 3 vehicles)
        vehicle_tracks = [t for t in involved_tracks if not is_vulnerable_road_user(t.class_name)]
        if len(vehicle_tracks) >= 3:
            return SeverityAssessment(
                severity=IncidentSeverity.MULTI_VEHICLE,
                involved_track_ids=track_ids,
                confidence=0.94,
                justification=f"Pile-up / multi-vehicle collision involving {len(vehicle_tracks)} vehicles",
            )

        # Check for vulnerable road user fall or severe ejection
        has_vulnerable_fall = any(
            p.is_fall or p.is_ejection or p.is_prolonged_ground for p in posture_reports
        )
        if has_vulnerable_fall:
            return SeverityAssessment(
                severity=IncidentSeverity.MAJOR,
                involved_track_ids=track_ids,
                confidence=0.96,
                justification="Severe collision involving pedestrian or rider fall/ejection",
            )

        # Check pre-impact speed / kinetic severity
        max_speed_m_s = 0.0
        max_decel_m_s2 = 0.0
        for ks in kinematic_states:
            if ks.speed_m_s > max_speed_m_s:
                max_speed_m_s = ks.speed_m_s
            if abs(ks.longitudinal_accel_m_s2) > max_decel_m_s2:
                max_decel_m_s2 = abs(ks.longitudinal_accel_m_s2)

        # Speed > 40 km/h (~11.1 m/s) or high deceleration > 7.0 m/s^2
        if max_speed_m_s >= 11.1 or max_decel_m_s2 >= 7.0:
            return SeverityAssessment(
                severity=IncidentSeverity.MAJOR,
                involved_track_ids=track_ids,
                confidence=0.91,
                justification=f"High-energy impact: speed {max_speed_m_s * 3.6:.1f} km/h, deceleration {max_decel_m_s2:.1f} m/s²",
            )

        # Default to minor collision
        return SeverityAssessment(
            severity=IncidentSeverity.MINOR,
            involved_track_ids=track_ids,
            confidence=0.85,
            justification="Low-speed fender bender / bumper impact with controlled kinematics",
        )
