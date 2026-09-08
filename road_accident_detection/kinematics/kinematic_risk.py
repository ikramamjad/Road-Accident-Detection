"""
Composite Kinematic Risk Engine.
Aggregates single-object acceleration shocks, swerves, and pairwise TTC hazards.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..tracking.tracker import TrackedObject
from ..tracking.trajectory_buffer import TrajectoryHistory
from .motion_analyzer import MotionAnalyzer, KinematicState
from .ttc_calculator import TTCCalculator, TTCPairResult


@dataclass
class KinematicRiskReport:
    """Consolidated kinematic hazard assessment for current frame."""
    overall_risk_score: float  # Normalized [0.0, 1.0]
    per_track_risk: Dict[int, float] = field(default_factory=dict)
    critical_pairs: List[TTCPairResult] = field(default_factory=list)
    hard_braking_tracks: List[int] = field(default_factory=list)
    rapid_swerve_tracks: List[int] = field(default_factory=list)
    min_ttc_seconds: Optional[float] = None
    explanation: str = "Nominal kinematic flow"


class KinematicRiskEngine:
    """Evaluates multi-vehicle kinematic risk."""

    def __init__(
        self,
        pixels_per_meter: float = 15.0,
        fps: float = 30.0,
        ttc_critical_thresh: float = 1.5,
        ttc_warning_thresh: float = 2.5,
        hard_braking_thresh: float = -4.5,
    ):
        self.motion_analyzer = MotionAnalyzer(
            pixels_per_meter=pixels_per_meter,
            fps=fps,
            hard_braking_threshold_m_s2=hard_braking_thresh,
        )
        self.ttc_calculator = TTCCalculator(
            pixels_per_meter=pixels_per_meter,
            fps=fps,
            ttc_critical_thresh=ttc_critical_thresh,
            ttc_warning_thresh=ttc_warning_thresh,
        )

    def evaluate(
        self,
        tracks: List[TrackedObject],
        histories: Dict[int, TrajectoryHistory],
    ) -> KinematicRiskReport:
        """Evaluate full scene kinematics across tracks and pairs."""
        if not tracks:
            return KinematicRiskReport(overall_risk_score=0.0)

        per_track_risk: Dict[int, float] = {t.track_id: 0.0 for t in tracks}
        hard_braking_tracks: List[int] = []
        rapid_swerve_tracks: List[int] = []

        # 1. Single object analysis
        for track in tracks:
            hist = histories.get(track.track_id)
            if not hist:
                continue

            kin_state = self.motion_analyzer.analyze(hist)
            if not kin_state:
                continue

            track_risk = 0.0
            if kin_state.is_hard_braking:
                hard_braking_tracks.append(track.track_id)
                # Severe deceleration risk
                decel_mag = abs(kin_state.longitudinal_accel_m_s2)
                track_risk = max(track_risk, min(1.0, decel_mag / 10.0))

            if kin_state.is_rapid_swerve:
                rapid_swerve_tracks.append(track.track_id)
                swerve_risk = min(1.0, kin_state.yaw_rate_deg_s / 90.0)
                track_risk = max(track_risk, swerve_risk * 0.75)

            per_track_risk[track.track_id] = track_risk

        # 2. Pairwise TTC analysis
        ttc_results = self.ttc_calculator.compute_pairwise_ttc(tracks, histories)
        critical_pairs: List[TTCPairResult] = []
        min_ttc: Optional[float] = None

        for pair in ttc_results:
            if pair.ttc_seconds is not None:
                if min_ttc is None or pair.ttc_seconds < min_ttc:
                    min_ttc = pair.ttc_seconds

            if pair.risk_score > 0.0:
                # Distribute pairwise risk to both participating tracks
                per_track_risk[pair.track_id_a] = max(per_track_risk[pair.track_id_a], pair.risk_score)
                per_track_risk[pair.track_id_b] = max(per_track_risk[pair.track_id_b], pair.risk_score)

            if pair.is_critical:
                critical_pairs.append(pair)

        # 3. Overall scene risk calculation
        overall_risk = max(per_track_risk.values()) if per_track_risk else 0.0
        if critical_pairs:
            overall_risk = max(overall_risk, 0.85)

        # Explanations
        reasons = []
        if critical_pairs:
            reasons.append(f"Critical TTC imminent ({min_ttc:.2f}s) between tracks {[p.track_id_a for p in critical_pairs]}")
        elif min_ttc is not None:
            reasons.append(f"Converging trajectories, minimum TTC {min_ttc:.2f}s")

        if hard_braking_tracks:
            reasons.append(f"Emergency hard braking on tracks {hard_braking_tracks}")
        if rapid_swerve_tracks:
            reasons.append(f"Violent swerving on tracks {rapid_swerve_tracks}")

        explanation = "; ".join(reasons) if reasons else "Nominal kinematic flow"

        return KinematicRiskReport(
            overall_risk_score=float(np.clip(overall_risk, 0.0, 1.0)),
            per_track_risk=per_track_risk,
            critical_pairs=critical_pairs,
            hard_braking_tracks=hard_braking_tracks,
            rapid_swerve_tracks=rapid_swerve_tracks,
            min_ttc_seconds=min_ttc,
            explanation=explanation,
        )
