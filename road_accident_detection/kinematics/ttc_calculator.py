"""
Time-to-Collision (TTC) Calculator.
Performs pairwise collision geometry calculations between moving road objects.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from ..tracking.tracker import TrackedObject
from ..tracking.trajectory_buffer import TrajectoryHistory


@dataclass
class TTCPairResult:
    """Pairwise collision geometry result between two tracks."""
    track_id_a: int
    track_id_b: int
    ttc_seconds: Optional[float]
    is_converging: bool
    closing_speed_m_s: float
    distance_meters: float
    min_miss_distance_meters: float
    is_critical: bool  # TTC < critical threshold with intersecting path
    risk_score: float  # Normalized [0.0, 1.0]


class TTCCalculator:
    """Computes pairwise Time-to-Collision for multi-vehicle & pedestrian scenes."""

    def __init__(
        self,
        pixels_per_meter: float = 15.0,
        fps: float = 30.0,
        ttc_critical_thresh: float = 1.5,   # seconds
        ttc_warning_thresh: float = 2.5,    # seconds
    ):
        self.ppm = max(pixels_per_meter, 1.0)
        self.fps = max(fps, 1.0)
        self.ttc_critical_thresh = ttc_critical_thresh
        self.ttc_warning_thresh = ttc_warning_thresh

    def compute_pairwise_ttc(
        self,
        tracks: List[TrackedObject],
        trajectory_histories: Dict[int, TrajectoryHistory],
    ) -> List[TTCPairResult]:
        """
        Evaluate all unique track pairs (A, B) for collision trajectory convergence.
        """
        results: List[TTCPairResult] = []
        n = len(tracks)
        if n < 2:
            return results

        for i in range(n):
            for j in range(i + 1, n):
                t_a = tracks[i]
                t_b = tracks[j]

                hist_a = trajectory_histories.get(t_a.track_id)
                hist_b = trajectory_histories.get(t_b.track_id)

                if not hist_a or not hist_b or not hist_a.latest or not hist_b.latest:
                    continue

                res = self._evaluate_pair(t_a, t_b, hist_a.latest, hist_b.latest)
                if res:
                    results.append(res)

        return results

    def _evaluate_pair(
        self,
        t_a: TrackedObject,
        t_b: TrackedObject,
        p_a,
        p_b,
    ) -> TTCPairResult:
        # Relative position vector r = p_a - p_b (in meters)
        rx = (p_a.center[0] - p_b.center[0]) / self.ppm
        ry = (p_a.center[1] - p_b.center[1]) / self.ppm
        dist = math.hypot(rx, ry)

        # Relative velocity vector v_rel = v_a - v_b (in m/s)
        # Note: TrajectoryPoint velocity is in px/s, so dividing by ppm yields m/s
        vx_a = p_a.velocity[0] / self.ppm
        vy_a = p_a.velocity[1] / self.ppm

        vx_b = p_b.velocity[0] / self.ppm
        vy_b = p_b.velocity[1] / self.ppm

        vrx = vx_a - vx_b
        vry = vy_a - vy_b
        v_rel_sq = vrx * vrx + vry * vry
        v_rel = math.sqrt(v_rel_sq)

        # Dot product r . v_rel
        r_dot_v = rx * vrx + ry * vry

        # Converging check: r_dot_v < 0 means distance is currently decreasing
        is_converging = (r_dot_v < -1e-3) and (v_rel > 0.5)

        ttc = None
        min_miss_dist = dist
        is_critical = False
        risk_score = 0.0

        # Characteristic collision radius (half size of bounding boxes in meters)
        radius_a = (t_a.width + t_a.height) / (4.0 * self.ppm)
        radius_b = (t_b.width + t_b.height) / (4.0 * self.ppm)
        collision_zone = radius_a + radius_b + 0.5  # Add 0.5m buffer

        if is_converging and v_rel_sq > 1e-3:
            # Analytical Time-to-Collision
            ttc_raw = -r_dot_v / v_rel_sq

            # Miss distance at closest point of approach
            d_min_sq = max(0.0, dist * dist - (ttc_raw * v_rel) ** 2)
            min_miss_dist = math.sqrt(d_min_sq)

            # Check if trajectory projected distance touches collision zone
            if min_miss_dist <= collision_zone and 0.05 <= ttc_raw <= self.ttc_warning_thresh:
                ttc = float(ttc_raw)
                is_critical = (ttc <= self.ttc_critical_thresh)

                # Normalized risk score: 1.0 when TTC -> 0, 0.0 when TTC >= ttc_warning_thresh
                ttc_factor = max(0.0, (self.ttc_warning_thresh - ttc) / self.ttc_warning_thresh)
                proximity_factor = max(0.0, (collision_zone - min_miss_dist) / max(collision_zone, 0.1))
                risk_score = float(np.clip(0.6 * ttc_factor + 0.4 * proximity_factor, 0.0, 1.0))
            elif min_miss_dist <= collision_zone * 1.5 and ttc_raw <= self.ttc_warning_thresh:
                # Close pass / Near-miss candidate
                ttc = float(ttc_raw)
                risk_score = float(np.clip(0.4 * (self.ttc_warning_thresh - ttc) / self.ttc_warning_thresh, 0.0, 0.6))

        closing_speed = -r_dot_v / max(dist, 1e-3) if dist > 1e-3 else 0.0

        return TTCPairResult(
            track_id_a=t_a.track_id,
            track_id_b=t_b.track_id,
            ttc_seconds=ttc,
            is_converging=is_converging,
            closing_speed_m_s=closing_speed,
            distance_meters=dist,
            min_miss_distance_meters=min_miss_dist,
            is_critical=is_critical,
            risk_score=risk_score,
        )
