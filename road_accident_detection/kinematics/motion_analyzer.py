"""
Motion Analyzer.
Estimates instantaneous speed, linear and longitudinal acceleration,
and yaw rate (heading change) from tracked object trajectories.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

from ..tracking.trajectory_buffer import TrajectoryHistory, TrajectoryPoint


@dataclass
class KinematicState:
    """Instantaneous kinematic metrics for a single tracked object."""
    track_id: int
    speed_px_s: float
    speed_m_s: float
    acceleration_px_s2: float
    longitudinal_accel_m_s2: float  # Negative denotes braking
    is_hard_braking: bool
    yaw_rate_deg_s: float
    is_rapid_swerve: bool


class MotionAnalyzer:
    """Computes physical velocity, acceleration, and yaw rates from object trajectories."""

    def __init__(
        self,
        pixels_per_meter: float = 15.0,
        fps: float = 30.0,
        hard_braking_threshold_m_s2: float = -4.5,
        rapid_swerve_threshold_deg_s: float = 45.0,
    ):
        self.ppm = max(pixels_per_meter, 1.0)
        self.fps = max(fps, 1.0)
        self.dt = 1.0 / self.fps
        self.hard_braking_threshold = hard_braking_threshold_m_s2
        self.rapid_swerve_threshold = rapid_swerve_threshold_deg_s

    def analyze(self, history: TrajectoryHistory) -> Optional[KinematicState]:
        """Compute kinematic metrics from trajectory history."""
        if not history or len(history.history) < 2:
            return None

        curr = history.history[-1]
        prev = history.history[-2]

        # Speed in px/s and m/s (TrajectoryPoint velocity and speed are already in px/s)
        speed_px_s = curr.speed
        speed_m_s = speed_px_s / self.ppm

        # Instantaneous acceleration vector (already in px/s^2)
        ax = curr.acceleration[0]
        ay = curr.acceleration[1]
        accel_mag_px_s2 = math.hypot(ax, ay)

        # Longitudinal acceleration (projection along velocity vector)
        vx = curr.velocity[0]
        vy = curr.velocity[1]
        v_mag = math.hypot(vx, vy)

        if v_mag > 1e-2:
            long_accel_px_s2 = (ax * vx + ay * vy) / v_mag
        else:
            long_accel_px_s2 = 0.0

        long_accel_m_s2 = long_accel_px_s2 / self.ppm
        is_hard_braking = long_accel_m_s2 <= self.hard_braking_threshold

        # Yaw rate in degrees/sec
        yaw_rate_deg_s = math.degrees(abs(curr.yaw_rate_rad))
        is_rapid_swerve = (yaw_rate_deg_s >= self.rapid_swerve_threshold) and (speed_m_s > 3.0)

        return KinematicState(
            track_id=history.track_id,
            speed_px_s=speed_px_s,
            speed_m_s=speed_m_s,
            acceleration_px_s2=accel_mag_px_s2,
            longitudinal_accel_m_s2=long_accel_m_s2,
            is_hard_braking=is_hard_braking,
            yaw_rate_deg_s=yaw_rate_deg_s,
            is_rapid_swerve=is_rapid_swerve,
        )
