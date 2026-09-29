"""
Trajectory Buffer Manager.
Maintains sliding-window temporal kinematics and bounding box history per tracked object ID.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import math
import numpy as np

from ..detection.classes import RoadClass


@dataclass
class TrajectoryPoint:
    """State of an object at a discrete frame."""
    frame_idx: int
    timestamp: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/frame or m/s
    speed: float = 0.0
    acceleration: Tuple[float, float] = (0.0, 0.0)
    yaw_angle_rad: float = 0.0  # Heading direction in radians
    yaw_rate_rad: float = 0.0  # Change in heading per second


class TrajectoryHistory:
    """Sliding-window trajectory history for an individual tracked object."""

    def __init__(self, track_id: int, class_name: RoadClass, max_len: int = 45, fps: float = 30.0):
        self.track_id = track_id
        self.class_name = class_name
        self.max_len = max_len
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)
        self.history: Deque[TrajectoryPoint] = deque(maxlen=max_len)
        self.pose_anomaly_scores: Deque[float] = deque(maxlen=max_len)
        self.kinematic_risks: Deque[float] = deque(maxlen=max_len)

    def add_observation(
        self,
        frame_idx: int,
        timestamp: float,
        bbox: Tuple[float, float, float, float],
        pose_anomaly_score: float = 0.0,
        kinematic_risk: float = 0.0,
    ) -> TrajectoryPoint:
        """Add a new observation and compute smoothed kinematics."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        vx, vy = 0.0, 0.0
        ax, ay = 0.0, 0.0
        yaw_rad = 0.0
        yaw_rate = 0.0

        if len(self.history) >= 1:
            prev = self.history[-1]
            time_delta = max(timestamp - prev.timestamp, self.dt)
            # Velocity calculation (exponentially weighted moving average with previous velocity)
            raw_vx = (cx - prev.center[0]) / time_delta
            raw_vy = (cy - prev.center[1]) / time_delta
            alpha = 0.65
            vx = alpha * raw_vx + (1.0 - alpha) * prev.velocity[0]
            vy = alpha * raw_vy + (1.0 - alpha) * prev.velocity[1]

            # Heading (yaw) calculation
            speed = math.hypot(vx, vy)
            if speed > 2.0:  # Only compute heading if moving
                yaw_rad = math.atan2(vy, vx)
                # Angular difference accounting for -pi to pi wrap
                angle_diff = (yaw_rad - prev.yaw_angle_rad + math.pi) % (2 * math.pi) - math.pi
                yaw_rate = angle_diff / time_delta
            else:
                yaw_rad = prev.yaw_angle_rad
                yaw_rate = 0.0

            # Acceleration calculation
            if len(self.history) >= 2:
                raw_ax = (vx - prev.velocity[0]) / time_delta
                raw_ay = (vy - prev.velocity[1]) / time_delta
                ax = 0.6 * raw_ax + 0.4 * prev.acceleration[0]
                ay = 0.6 * raw_ay + 0.4 * prev.acceleration[1]
        else:
            speed = 0.0

        point = TrajectoryPoint(
            frame_idx=frame_idx,
            timestamp=timestamp,
            bbox=bbox,
            center=(cx, cy),
            velocity=(vx, vy),
            speed=math.hypot(vx, vy),
            acceleration=(ax, ay),
            yaw_angle_rad=yaw_rad,
            yaw_rate_rad=yaw_rate,
        )

        self.history.append(point)
        self.pose_anomaly_scores.append(pose_anomaly_score)
        self.kinematic_risks.append(kinematic_risk)
        return point

    @property
    def latest(self) -> Optional[TrajectoryPoint]:
        return self.history[-1] if self.history else None

    def get_feature_matrix(self, img_w: int, img_h: int, target_len: int = 16) -> np.ndarray:
        """
        Extract normalized 12-D causal feature sequence of shape (target_len, 12).
        Features: [norm_cx, norm_cy, norm_w, norm_h, norm_vx, norm_vy, norm_ax, norm_ay, yaw_rate, pose_score, kinematic_risk, speed]
        Zero-padded on left if history length < target_len.
        """
        features = np.zeros((target_len, 12), dtype=np.float32)
        n = len(self.history)
        if n == 0:
            return features

        pts = list(self.history)[-target_len:]
        pose_scores = list(self.pose_anomaly_scores)[-target_len:]
        kin_risks = list(self.kinematic_risks)[-target_len:]

        offset = target_len - len(pts)
        for i, pt in enumerate(pts):
            w = pt.bbox[2] - pt.bbox[0]
            h = pt.bbox[3] - pt.bbox[1]
            features[offset + i] = [
                pt.center[0] / max(img_w, 1.0),
                pt.center[1] / max(img_h, 1.0),
                w / max(img_w, 1.0),
                h / max(img_h, 1.0),
                pt.velocity[0] / 500.0,  # normalized velocity scale
                pt.velocity[1] / 500.0,
                pt.acceleration[0] / 500.0,
                pt.acceleration[1] / 500.0,
                pt.yaw_rate_rad / math.pi,
                pose_scores[i] if i < len(pose_scores) else 0.0,
                kin_risks[i] if i < len(kin_risks) else 0.0,
                pt.speed / 500.0,
            ]

        # Temporal overlapping: back-fill earlier slots with the earliest observation
        if offset > 0 and len(pts) > 0:
            for j in range(offset):
                features[j] = features[offset]

        return features


class TrajectoryBufferManager:
    """Manages trajectory histories across all active and recently lost tracks."""

    def __init__(self, max_history_len: int = 45, max_missing_frames: int = 30, fps: float = 30.0):
        self.max_history_len = max_history_len
        self.max_missing_frames = max_missing_frames
        self.fps = fps
        self.tracks: Dict[int, TrajectoryHistory] = {}
        self.last_seen_frame: Dict[int, int] = {}

    def update_track(
        self,
        track_id: int,
        class_name: RoadClass,
        frame_idx: int,
        timestamp: float,
        bbox: Tuple[float, float, float, float],
        pose_score: float = 0.0,
        kinematic_risk: float = 0.0,
    ) -> TrajectoryPoint:
        """Update or register a track observation."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrajectoryHistory(
                track_id=track_id,
                class_name=class_name,
                max_len=self.max_history_len,
                fps=self.fps,
            )

        point = self.tracks[track_id].add_observation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            bbox=bbox,
            pose_anomaly_score=pose_score,
            kinematic_risk=kinematic_risk,
        )
        self.last_seen_frame[track_id] = frame_idx
        return point

    def prune_inactive(self, current_frame: int) -> List[int]:
        """Evict tracks that have been inactive for more than max_missing_frames."""
        dead_ids = [
            tid for tid, last_frame in self.last_seen_frame.items()
            if (current_frame - last_frame) > self.max_missing_frames
        ]
        for tid in dead_ids:
            self.tracks.pop(tid, None)
            self.last_seen_frame.pop(tid, None)
        return dead_ids

    def get_history(self, track_id: int) -> Optional[TrajectoryHistory]:
        return self.tracks.get(track_id)
