"""
Abnormal Posture Analyzer.
Calculates fall detection, rider ejection, sudden orientation change,
and prolonged ground-level posture anomaly scores from skeletal keypoints.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .pose_detector import PersonPose
from ..tracking.tracker import TrackedObject
from ..detection.classes import is_two_wheeler, RoadClass


@dataclass
class PostureAnomalyReport:
    """Detailed report on posture abnormality for a tracked person."""
    track_id: Optional[int]
    anomaly_score: float  # Normalized [0.0, 1.0]
    is_fall: bool
    is_ejection: bool
    is_prolonged_ground: bool
    torso_angle_deg: float  # Angle relative to horizontal ground (0 = flat on ground, 90 = upright)
    aspect_ratio: float     # height / width
    consecutive_ground_frames: int
    explanation: str


class PostureAnalyzer:
    """
    Evaluates abnormal human posture for pedestrians and motorcycle/bicycle riders.
    """

    def __init__(
        self,
        fall_torso_angle_thresh: float = 35.0,  # Degrees
        prolonged_ground_frames: int = 12,
        rapid_tilt_angular_vel: float = 60.0,   # Deg/frame
        ejection_dist_threshold_px: float = 120.0,
    ):
        self.fall_torso_angle_thresh = fall_torso_angle_thresh
        self.prolonged_ground_frames = prolonged_ground_frames
        self.rapid_tilt_angular_vel = rapid_tilt_angular_vel
        self.ejection_dist_threshold_px = ejection_dist_threshold_px

        # State tracking per person track_id: (prev_angle, ground_frames_count)
        self._track_states: Dict[int, Dict] = {}

    def analyze_pose(
        self,
        pose: PersonPose,
        track_id: Optional[int] = None,
        nearby_two_wheelers: Optional[List[TrackedObject]] = None,
    ) -> PostureAnomalyReport:
        """
        Analyze a detected person pose and compute the posture anomaly score.
        """
        # 1. Torso Angle Calculation
        # Vector from hip center to shoulder center
        sc = pose.shoulder_center[:2]
        hc = pose.hip_center[:2]

        dx = sc[0] - hc[0]
        dy = sc[1] - hc[1]  # Note: y points downwards in image coordinates

        # Angle relative to horizontal plane
        hypot = math.hypot(dx, dy)
        if hypot > 1e-3:
            # Absolute angle of torso vector relative to horizontal axis
            torso_angle_deg = math.degrees(math.asin(min(1.0, abs(dy) / hypot)))
        else:
            torso_angle_deg = 90.0  # Default assume vertical if keypoints coincide

        # 2. Aspect ratio of bounding box
        bx1, by1, bx2, by2 = pose.bbox
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        aspect_ratio = bh / bw

        # 3. Fall detection criteria
        # A person is fallen if torso is near horizontal (< 35 deg) or aspect ratio is inverted (bh < bw)
        is_fall = (torso_angle_deg < self.fall_torso_angle_thresh) or (aspect_ratio < 0.85)

        # 4. Temporal ground posture and angular velocity tracking
        prev_angle = 90.0
        ground_count = 0
        angular_vel = 0.0

        if track_id is not None:
            state = self._track_states.get(track_id, {"prev_angle": torso_angle_deg, "ground_count": 0})
            prev_angle = state["prev_angle"]
            ground_count = state["ground_count"]

            if is_fall:
                ground_count += 1
            else:
                ground_count = max(0, ground_count - 1)

            angular_vel = abs(torso_angle_deg - prev_angle)
            self._track_states[track_id] = {
                "prev_angle": torso_angle_deg,
                "ground_count": ground_count,
            }

        is_prolonged_ground = (ground_count >= self.prolonged_ground_frames)

        # 5. Rider Ejection Check
        # Check if person was previously co-located with a two-wheeler and is now rapidly separating
        is_ejection = False
        if nearby_two_wheelers:
            person_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            for tw in nearby_two_wheelers:
                tw_center = tw.center
                dist = math.hypot(person_center[0] - tw_center[0], person_center[1] - tw_center[1])
                # If person is fallen or tilting and within separation distance
                if 40.0 < dist < self.ejection_dist_threshold_px and is_fall:
                    is_ejection = True
                    break

        # 6. Compute composite posture anomaly score [0.0 - 1.0]
        # Angle term: 0 deg (flat) -> 1.0; 90 deg (upright) -> 0.0
        angle_score = max(0.0, min(1.0, (90.0 - torso_angle_deg) / 90.0))
        # Aspect ratio term: < 0.7 -> 1.0; > 1.8 -> 0.0
        ar_score = max(0.0, min(1.0, (1.8 - aspect_ratio) / 1.1))
        # Angular velocity term
        tilt_score = min(1.0, angular_vel / max(self.rapid_tilt_angular_vel, 1.0))
        # Prolonged ground term
        prolonged_score = min(1.0, ground_count / max(self.prolonged_ground_frames, 1.0))

        # Weighted combination
        weights = [0.35, 0.25, 0.20, 0.20]
        anomaly_score = (
            weights[0] * angle_score +
            weights[1] * ar_score +
            weights[2] * tilt_score +
            weights[3] * prolonged_score
        )

        if is_fall:
            anomaly_score = max(anomaly_score, 0.72)

        if is_ejection:
            anomaly_score = max(anomaly_score, 0.85)

        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        # Build explainability description
        reasons = []
        if is_fall:
            reasons.append(f"Torso angle {torso_angle_deg:.1f}° (horizontal fall)")
        if is_ejection:
            reasons.append("Two-wheeler rider ejection trajectory")
        if is_prolonged_ground:
            reasons.append(f"Prolonged ground posture ({ground_count} frames)")
        if angular_vel > self.rapid_tilt_angular_vel * 0.7:
            reasons.append(f"Rapid tilt change ({angular_vel:.1f}°/frame)")

        explanation = "; ".join(reasons) if reasons else "Normal upright posture"

        return PostureAnomalyReport(
            track_id=track_id,
            anomaly_score=anomaly_score,
            is_fall=is_fall,
            is_ejection=is_ejection,
            is_prolonged_ground=is_prolonged_ground,
            torso_angle_deg=torso_angle_deg,
            aspect_ratio=aspect_ratio,
            consecutive_ground_frames=ground_count,
            explanation=explanation,
        )
