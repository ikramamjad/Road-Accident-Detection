"""
Explainability and Audit Reporter.
Generates human-readable causal reasoning and attribution breakdowns for detected events.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..fusion.multi_channel_fusion import FusedRiskResult
from ..kinematics.kinematic_risk import KinematicRiskReport
from ..pose.posture_analyzer import PostureAnomalyReport


@dataclass
class ExplainabilityReport:
    primary_channel: str
    secondary_channel: Optional[str]
    channel_percentages: Dict[str, float]
    narrative_summary: str
    alert_tags: List[str]


class ExplainabilityReporter:
    """Produces explainable auditing breakdowns for alerts."""

    @staticmethod
    def generate_report(
        fusion_result: FusedRiskResult,
        kinematic_report: Optional[KinematicRiskReport] = None,
        posture_reports: Optional[List[PostureAnomalyReport]] = None,
        debounce_frames: int = 6,
    ) -> ExplainabilityReport:
        """Construct comprehensive explainability breakdown."""
        # Calculate percentage contribution per channel
        total_attrib = sum(fusion_result.channel_attributions.values())
        percentages = {}
        for ch, val in fusion_result.channel_attributions.items():
            percentages[ch] = round((val / max(total_attrib, 1e-6)) * 100.0, 1)

        tags: List[str] = []
        narrative_parts: List[str] = []

        # Pose findings
        if posture_reports:
            for p in posture_reports:
                if p.is_fall:
                    tags.append("POSTURE_FALL")
                if p.is_ejection:
                    tags.append("RIDER_EJECTION")
                if p.is_prolonged_ground:
                    tags.append("PROLONGED_GROUND_INCAPACITATION")

        # Kinematic findings
        if kinematic_report:
            if kinematic_report.critical_pairs:
                tags.append("CRITICAL_TTC_CONVERGENCE")
            if kinematic_report.hard_braking_tracks:
                tags.append("EMERGENCY_HARD_BRAKE")
            if kinematic_report.rapid_swerve_tracks:
                tags.append("HIGH_YAW_SWERVE")

        # Construct narrative
        ch_name_map = {
            "pose_anomaly": "Rider/Pedestrian Posture Analysis",
            "kinematic_risk": "Trajectory Kinematics & TTC",
            "temporal_anomaly": "Causal GRU Temporal Modeling",
            "detection_conf": "Spatial Object Detection",
        }

        prim_name = ch_name_map.get(fusion_result.primary_driver, fusion_result.primary_driver)
        narrative_parts.append(
            f"Event primarily driven by {prim_name} ({percentages.get(fusion_result.primary_driver, 0)}% influence)"
        )

        if fusion_result.secondary_driver and percentages.get(fusion_result.secondary_driver, 0) > 15.0:
            sec_name = ch_name_map.get(fusion_result.secondary_driver, fusion_result.secondary_driver)
            narrative_parts.append(
                f"with secondary corroboration from {sec_name} ({percentages.get(fusion_result.secondary_driver, 0)}%)"
            )

        narrative_parts.append(f"sustained across {debounce_frames} consecutive frames.")

        if fusion_result.explanation:
            narrative_parts.append(f"Observations: {fusion_result.explanation}.")

        narrative_summary = " ".join(narrative_parts)

        return ExplainabilityReport(
            primary_channel=fusion_result.primary_driver,
            secondary_channel=fusion_result.secondary_driver,
            channel_percentages=percentages,
            narrative_summary=narrative_summary,
            alert_tags=list(set(tags)),
        )
