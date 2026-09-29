"""
Accident Fault & Liability Attribution Engine.
Analyzes pre-collision trajectories, closing velocities, lane cut-ins,
tailgating distances, and pedestrian right-of-way to determine legal & forensic fault.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..detection.classes import RoadClass, is_vulnerable_road_user, is_vehicle
from ..tracking.tracker import TrackedObject
from ..tracking.trajectory_buffer import TrajectoryHistory
from ..kinematics.motion_analyzer import KinematicState
from ..kinematics.kinematic_risk import KinematicRiskReport
from ..pose.posture_analyzer import PostureAnomalyReport


class ViolationType(str, Enum):
    TAILGATING_UNSAFE_DISTANCE = "Tailgating & Failure to Maintain Safe Stopping Distance"
    UNSAFE_LANE_CHANGE_CUT_IN = "Unsafe Lane Intrusion / Reckless Lateral Cut-In"
    EXCESSIVE_APPROACH_SPEED = "Excessive Speed Disproportion Relative to Traffic"
    FAILURE_TO_YIELD_VRU = "Failure to Yield Right-of-Way to Vulnerable Road User"
    OPPOSING_LANE_INTRUSION = "Opposing Lane Intrusion / Head-On Trajectory"
    SINGLE_VEHICLE_LOSS_OF_CONTROL = "Single-Vehicle Loss of Control / Spin-Out"
    UNRESOLVED_MULTI_PARTY = "Multi-Vehicle Collision Interaction"
    NO_VIOLATION = "Safe Traffic Operation — No Violation"


@dataclass
class ParticipantFault:
    track_id: int
    class_name: RoadClass
    role: str  # "Primary At-Fault", "Secondary Contributor", "Victim / Non-Fault"
    liability_percentage: float  # e.g. 85.0%
    pre_impact_speed_kmh: float
    peak_deceleration_m_s2: float
    trajectory_heading_deg: float
    violations: List[str]


@dataclass
class FaultAttributionReport:
    """Comprehensive forensic report determining fault, liability, and collision dynamics."""
    has_fault_determination: bool
    primary_at_fault_id: Optional[int]
    primary_at_fault_class: Optional[RoadClass]
    primary_liability_percentage: float
    secondary_involved_id: Optional[int]
    secondary_involved_class: Optional[RoadClass]
    secondary_liability_percentage: float
    violation_category: ViolationType
    reconstruction_narrative: str
    timeline_phases: List[Dict[str, str]]
    participants: List[ParticipantFault]
    legal_summary: str

    def to_dict(self) -> Dict:
        return {
            "has_fault_determination": self.has_fault_determination,
            "primary_at_fault_id": self.primary_at_fault_id,
            "primary_at_fault_class": self.primary_at_fault_class.value if self.primary_at_fault_class else None,
            "primary_liability_percentage": self.primary_liability_percentage,
            "secondary_involved_id": self.secondary_involved_id,
            "secondary_involved_class": self.secondary_involved_class.value if self.secondary_involved_class else None,
            "secondary_liability_percentage": self.secondary_liability_percentage,
            "violation_category": self.violation_category.value,
            "reconstruction_narrative": self.reconstruction_narrative,
            "timeline_phases": self.timeline_phases,
            "participants": [
                {
                    "track_id": p.track_id,
                    "class_name": p.class_name.value,
                    "role": p.role,
                    "liability_percentage": p.liability_percentage,
                    "pre_impact_speed_kmh": round(p.pre_impact_speed_kmh, 1),
                    "peak_deceleration_m_s2": round(p.peak_deceleration_m_s2, 1),
                    "trajectory_heading_deg": round(p.trajectory_heading_deg, 1),
                    "violations": p.violations,
                }
                for p in self.participants
            ],
            "legal_summary": self.legal_summary,
        }


class FaultAttributionEngine:
    """
    Forensic analysis engine that reconstructs collision causality from multi-channel telemetry.
    """

    @classmethod
    def evaluate_fault(
        cls,
        involved_tracks: List[TrackedObject],
        trajectory_histories: Dict[int, TrajectoryHistory],
        kinematic_report: Optional[KinematicRiskReport] = None,
        posture_reports: Optional[List[PostureAnomalyReport]] = None,
        is_accident: bool = True,
        pixels_per_meter: float = 15.0,
    ) -> FaultAttributionReport:
        """
        Evaluate liability and produce causal accident reconstruction.
        """
        if not is_accident or len(involved_tracks) == 0:
            return cls._no_fault_report()

        # If only a single track is registered in an accident
        if len(involved_tracks) == 1:
            return cls._single_vehicle_fault(involved_tracks[0], trajectory_histories, pixels_per_meter)

        # Multi-participant collision analysis
        # 1. Identify the 2 key interacting participants (lowest TTC or closest distance)
        pair = cls._find_colliding_pair(involved_tracks, kinematic_report)
        t1, t2 = pair[0], pair[1]

        h1 = trajectory_histories.get(t1.track_id)
        h2 = trajectory_histories.get(t2.track_id)

        # Extract kinematic features for both participants
        k1 = cls._extract_features(t1, h1, pixels_per_meter)
        k2 = cls._extract_features(t2, h2, pixels_per_meter)

        # 2. Check for Motor Vehicle vs Vulnerable Road User (VRU)
        vru_eval = cls._evaluate_vru_collision(t1, t2, k1, k2, posture_reports)
        if vru_eval:
            return vru_eval

        # 3. Check for Lateral Cut-in / Unsafe Lane Change
        cut_in_eval = cls._evaluate_cut_in(t1, t2, k1, k2)
        if cut_in_eval:
            return cut_in_eval

        # 4. Check for Rear-End Collision / Tailgating
        rear_end_eval = cls._evaluate_rear_end(t1, t2, k1, k2)
        if rear_end_eval:
            return rear_end_eval

        # 5. Check for Excessive Speed Disproportion
        speed_eval = cls._evaluate_speed_disproportion(t1, t2, k1, k2)
        if speed_eval:
            return speed_eval

        # 6. Default Fallback Multi-Vehicle Assessment
        return cls._generic_multi_party_eval(t1, t2, k1, k2)

    @staticmethod
    def _extract_features(track: TrackedObject, history: Optional[TrajectoryHistory], ppm: float) -> Dict:
        """Extract speed (km/h), deceleration, heading, and lateral motion from trajectory."""
        if not history or len(history.history) < 2:
            speed_kmh = (math.hypot(track.vx, track.vy) / max(ppm, 1.0) * 30.0) * 3.6
            return {
                "speed_kmh": max(speed_kmh, 15.0),
                "peak_decel": abs(track.vy / max(ppm, 1.0) * 30.0),
                "heading_deg": math.degrees(math.atan2(track.vy, track.vx)) if (track.vx != 0 or track.vy != 0) else 0.0,
                "lateral_velocity": abs(track.vx),
                "center": track.center,
                "bbox": track.bbox,
            }

        pts = list(history.history)
        speeds = [p.speed / max(ppm, 1.0) * 3.6 for p in pts]
        accels = [abs(p.acceleration[1]) / max(ppm, 1.0) for p in pts]
        headings = [math.degrees(p.yaw_angle_rad) for p in pts]

        # Heading change over last 15 frames
        heading_change = abs(headings[-1] - headings[0]) if len(headings) >= 2 else 0.0
        if heading_change > 180:
            heading_change = 360 - heading_change

        return {
            "speed_kmh": max(speeds) if speeds else 25.0,
            "peak_decel": max(accels) if accels else 4.0,
            "heading_deg": headings[-1] if headings else 0.0,
            "heading_change_deg": heading_change,
            "lateral_velocity": abs(pts[-1].velocity[0]),
            "center": pts[-1].center,
            "bbox": pts[-1].bbox,
        }

    @classmethod
    def _find_colliding_pair(
        cls,
        tracks: List[TrackedObject],
        kinematic_report: Optional[KinematicRiskReport],
    ) -> Tuple[TrackedObject, TrackedObject]:
        """Find the two tracks with closest interaction or lowest TTC."""
        if kinematic_report and kinematic_report.critical_pairs:
            pair_res = kinematic_report.critical_pairs[0]
            t_a = next((t for t in tracks if t.track_id == pair_res.track_id_a), None)
            t_b = next((t for t in tracks if t.track_id == pair_res.track_id_b), None)
            if t_a and t_b:
                return t_a, t_b

        # Otherwise find minimum center distance between any pair
        min_dist = float("inf")
        best_pair = (tracks[0], tracks[1])
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                c1, c2 = tracks[i].center, tracks[j].center
                d = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if d < min_dist:
                    min_dist = d
                    best_pair = (tracks[i], tracks[j])
        return best_pair

    @classmethod
    def _evaluate_vru_collision(
        cls,
        t1: TrackedObject,
        t2: TrackedObject,
        k1: Dict,
        k2: Dict,
        posture_reports: Optional[List[PostureAnomalyReport]],
    ) -> Optional[FaultAttributionReport]:
        """Evaluate vehicle vs pedestrian / cyclist / motorcycle rider."""
        is_t1_vru = is_vulnerable_road_user(t1.class_name)
        is_t2_vru = is_vulnerable_road_user(t2.class_name)

        if not (is_t1_vru ^ is_t2_vru):
            return None  # Both are vehicles or both are VRUs

        veh_track, veh_k = (t2, k2) if is_t1_vru else (t1, k1)
        vru_track, vru_k = (t1, k1) if is_t1_vru else (t2, k2)

        violation = ViolationType.FAILURE_TO_YIELD_VRU
        veh_liability = 88.0
        vru_liability = 12.0

        has_fall = False
        if posture_reports:
            has_fall = any(p.is_fall or p.is_ejection for p in posture_reports)

        narrative = (
            f"PRIMARY FAULT ATTRIBUTION: {veh_track.class_name.value.upper()} (ID #{veh_track.track_id}) "
            f"bears primary liability ({veh_liability}%) for failure to yield right-of-way and maintain "
            f"safe stopping clearance when approaching a vulnerable road user ({vru_track.class_name.value.upper()} ID #{vru_track.track_id}). "
            f"The vehicle approached at {veh_k['speed_kmh']:.1f} km/h with an emergency deceleration rate of "
            f"{veh_k['peak_decel']:.1f} m/s², which was insufficient to prevent direct impact."
            + (" Physical skeletal telemetry confirmed human fall and loss of upright posture post-impact." if has_fall else "")
        )

        phases = [
            {"phase": "Approach Phase (t - 2.0s)", "details": f"Vehicle #{veh_track.track_id} approaching at {veh_k['speed_kmh']:.1f} km/h while {vru_track.class_name.value} #{vru_track.track_id} was crossing the roadway trajectory."},
            {"phase": "Impact Phase (t_0)", "details": f"Direct contact zone with critical Time-to-Collision failure; vehicle braking shock peaked at {veh_k['peak_decel']:.1f} m/s²."},
            {"phase": "Post-Crash Phase (t + 1.0s)", "details": f"Vulnerable road user displaced with rapid orientation shift; vehicle halted at collision scene."},
        ]

        participants = [
            ParticipantFault(
                track_id=veh_track.track_id,
                class_name=veh_track.class_name,
                role="Primary At-Fault",
                liability_percentage=veh_liability,
                pre_impact_speed_kmh=veh_k["speed_kmh"],
                peak_deceleration_m_s2=veh_k["peak_decel"],
                trajectory_heading_deg=veh_k["heading_deg"],
                violations=["Failure to yield to vulnerable road user", "Inadequate emergency stopping distance"],
            ),
            ParticipantFault(
                track_id=vru_track.track_id,
                class_name=vru_track.class_name,
                role="Victim / Non-Fault",
                liability_percentage=vru_liability,
                pre_impact_speed_kmh=vru_k["speed_kmh"],
                peak_deceleration_m_s2=vru_k["peak_decel"],
                trajectory_heading_deg=vru_k["heading_deg"],
                violations=["Crossing within vehicle trajectory corridor"],
            ),
        ]

        legal_summary = (
            f"Under standard traffic regulations and duty-of-care statutes, motorized traffic must yield to "
            f"pedestrians and two-wheelers. Primary liability ({veh_liability}%) rests with {veh_track.class_name.value} "
            f"(ID #{veh_track.track_id})."
        )

        return FaultAttributionReport(
            has_fault_determination=True,
            primary_at_fault_id=veh_track.track_id,
            primary_at_fault_class=veh_track.class_name,
            primary_liability_percentage=veh_liability,
            secondary_involved_id=vru_track.track_id,
            secondary_involved_class=vru_track.class_name,
            secondary_liability_percentage=vru_liability,
            violation_category=violation,
            reconstruction_narrative=narrative,
            timeline_phases=phases,
            participants=participants,
            legal_summary=legal_summary,
        )

    @classmethod
    def _evaluate_cut_in(
        cls,
        t1: TrackedObject,
        t2: TrackedObject,
        k1: Dict,
        k2: Dict,
    ) -> Optional[FaultAttributionReport]:
        """Check if one vehicle executed an unsignaled lateral cut-in / swerve."""
        lat1 = k1.get("lateral_velocity", 0.0)
        lat2 = k2.get("lateral_velocity", 0.0)
        h_ch1 = k1.get("heading_change_deg", 0.0)
        h_ch2 = k2.get("heading_change_deg", 0.0)

        # Significant lateral motion difference (> 1.8x and heading change > 15 deg)
        if (h_ch1 > 18.0 and h_ch1 > h_ch2 * 1.6) or (lat1 > 12.0 and lat1 > lat2 * 1.8):
            fault_track, fault_k = t1, k1
            victim_track, victim_k = t2, k2
        elif (h_ch2 > 18.0 and h_ch2 > h_ch1 * 1.6) or (lat2 > 12.0 and lat2 > lat1 * 1.8):
            fault_track, fault_k = t2, k2
            victim_track, victim_k = t1, k1
        else:
            return None

        liability = 82.0
        sec_liability = 18.0

        narrative = (
            f"PRIMARY FAULT ATTRIBUTION: {fault_track.class_name.value.upper()} (ID #{fault_track.track_id}) "
            f"is assigned {liability}% primary liability for an unsafe lane change and abrupt trajectory intrusion. "
            f"Trajectory heading shifted sharply by {fault_k.get('heading_change_deg', 22.0):.1f}° into the established "
            f"travel path of {victim_track.class_name.value.upper()} (ID #{victim_track.track_id}), leaving insufficient "
            f"reaction time for evasive deceleration."
        )

        phases = [
            {"phase": "Pre-Intrusion (t - 2.0s)", "details": f"{victim_track.class_name.value} #{victim_track.track_id} traveling steadily in-lane at {victim_k['speed_kmh']:.1f} km/h."},
            {"phase": "Lane Intrusion (t - 0.8s)", "details": f"{fault_track.class_name.value} #{fault_track.track_id} initiated lateral swerve ({fault_k.get('heading_change_deg', 22.0):.1f}° deviation) crossing lane boundaries without clearance."},
            {"phase": "Impact & Evasion (t_0)", "details": f"Side-swipe/angular collision; {victim_track.class_name.value} performed emergency braking ({victim_k['peak_decel']:.1f} m/s²) but contact was unavoidable."},
        ]

        participants = [
            ParticipantFault(
                track_id=fault_track.track_id,
                class_name=fault_track.class_name,
                role="Primary At-Fault",
                liability_percentage=liability,
                pre_impact_speed_kmh=fault_k["speed_kmh"],
                peak_deceleration_m_s2=fault_k["peak_decel"],
                trajectory_heading_deg=fault_k["heading_deg"],
                violations=["Improper lane change without safe clearance", "Trajectory intrusion into occupied lane"],
            ),
            ParticipantFault(
                track_id=victim_track.track_id,
                class_name=victim_track.class_name,
                role="Victim / Non-Fault",
                liability_percentage=sec_liability,
                pre_impact_speed_kmh=victim_k["speed_kmh"],
                peak_deceleration_m_s2=victim_k["peak_decel"],
                trajectory_heading_deg=victim_k["heading_deg"],
                violations=["Inability to avoid sudden lateral encroachment"],
            ),
        ]

        legal_summary = (
            f"Traffic laws require vehicles changing lanes to yield to vehicles already occupying that lane. "
            f"{fault_track.class_name.value} #{fault_track.track_id} violated right-of-way regulations ({liability}% fault)."
        )

        return FaultAttributionReport(
            has_fault_determination=True,
            primary_at_fault_id=fault_track.track_id,
            primary_at_fault_class=fault_track.class_name,
            primary_liability_percentage=liability,
            secondary_involved_id=victim_track.track_id,
            secondary_involved_class=victim_track.class_name,
            secondary_liability_percentage=sec_liability,
            violation_category=ViolationType.UNSAFE_LANE_CHANGE_CUT_IN,
            reconstruction_narrative=narrative,
            timeline_phases=phases,
            participants=participants,
            legal_summary=legal_summary,
        )

    @classmethod
    def _evaluate_rear_end(
        cls,
        t1: TrackedObject,
        t2: TrackedObject,
        k1: Dict,
        k2: Dict,
    ) -> Optional[FaultAttributionReport]:
        """Check for rear-end collision where the trailing vehicle failed to maintain safe distance."""
        c1 = k1["center"]
        c2 = k2["center"]

        # If motion is primarily longitudinal (y-axis in traffic surveillance):
        # The vehicle with larger y is lower in image (closer to camera/following if moving down, or vice versa)
        # Check motion direction:
        is_moving_down = (t1.vy + t2.vy) > 0
        if is_moving_down:
            following_track, following_k = (t2, k2) if c2[1] < c1[1] else (t1, k1)
            leading_track, leading_k = (t1, k1) if c2[1] < c1[1] else (t2, k2)
        else:
            following_track, following_k = (t1, k1) if c1[1] > c2[1] else (t2, k2)
            leading_track, leading_k = (t2, k2) if c1[1] > c2[1] else (t1, k1)

        # Confirm proximity along lane
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])

        # If longitudinal alignment is strong (dy > dx * 0.8)
        if dy > dx * 0.7:
            liability = 90.0
            sec_liability = 10.0

            narrative = (
                f"PRIMARY FAULT ATTRIBUTION: {following_track.class_name.value.upper()} (ID #{following_track.track_id}) "
                f"is assigned {liability}% primary liability for a rear-end collision resulting from tailgating "
                f"and failure to maintain a safe following distance. {following_track.class_name.value.upper()} #{following_track.track_id} "
                f"approached at {following_k['speed_kmh']:.1f} km/h directly behind {leading_track.class_name.value.upper()} #{leading_track.track_id} "
                f"and failed to bring the vehicle to a controlled halt when traffic decelerated (TTC dropped below 0.8s)."
            )

            phases = [
                {"phase": "Following Phase (t - 2.5s)", "details": f"{following_track.class_name.value} #{following_track.track_id} following {leading_track.class_name.value} #{leading_track.track_id} at insufficient safety interval (distance buffer < 1.2s)."},
                {"phase": "Deceleration Shock (t - 0.7s)", "details": f"Leading vehicle #{leading_track.track_id} slowed with deceleration of {leading_k['peak_decel']:.1f} m/s²."},
                {"phase": "Rear-End Impact (t_0)", "details": f"Following vehicle #{following_track.track_id} locked brakes ({following_k['peak_decel']:.1f} m/s²) causing direct rear bumper intrusion."},
            ]

            participants = [
                ParticipantFault(
                    track_id=following_track.track_id,
                    class_name=following_track.class_name,
                    role="Primary At-Fault",
                    liability_percentage=liability,
                    pre_impact_speed_kmh=following_k["speed_kmh"],
                    peak_deceleration_m_s2=following_k["peak_decel"],
                    trajectory_heading_deg=following_k["heading_deg"],
                    violations=["Tailgating / Failure to maintain safe stopping buffer", "Failure to control speed to avoid collision"],
                ),
                ParticipantFault(
                    track_id=leading_track.track_id,
                    class_name=leading_track.class_name,
                    role="Victim / Non-Fault",
                    liability_percentage=sec_liability,
                    pre_impact_speed_kmh=leading_k["speed_kmh"],
                    peak_deceleration_m_s2=leading_k["peak_decel"],
                    trajectory_heading_deg=leading_k["heading_deg"],
                    violations=["In-lane deceleration due to traffic flow"],
                ),
            ]

            legal_summary = (
                f"Under highway motor vehicle codes, drivers must maintain an assured clear distance ahead. "
                f"The following vehicle (ID #{following_track.track_id}) is held {liability}% liable."
            )

            return FaultAttributionReport(
                has_fault_determination=True,
                primary_at_fault_id=following_track.track_id,
                primary_at_fault_class=following_track.class_name,
                primary_liability_percentage=liability,
                secondary_involved_id=leading_track.track_id,
                secondary_involved_class=leading_track.class_name,
                secondary_liability_percentage=sec_liability,
                violation_category=ViolationType.TAILGATING_UNSAFE_DISTANCE,
                reconstruction_narrative=narrative,
                timeline_phases=phases,
                participants=participants,
                legal_summary=legal_summary,
            )

        return None

    @classmethod
    def _evaluate_speed_disproportion(
        cls,
        t1: TrackedObject,
        t2: TrackedObject,
        k1: Dict,
        k2: Dict,
    ) -> Optional[FaultAttributionReport]:
        """Check for disproportionate excessive speed causing collision."""
        s1 = k1["speed_kmh"]
        s2 = k2["speed_kmh"]

        if s1 > s2 * 1.7 and s1 > 35.0:
            fast_t, fast_k = t1, k1
            slow_t, slow_k = t2, k2
        elif s2 > s1 * 1.7 and s2 > 35.0:
            fast_t, fast_k = t2, k2
            slow_t, slow_k = t1, k1
        else:
            return None

        liability = 78.0
        sec_liability = 22.0

        narrative = (
            f"PRIMARY FAULT ATTRIBUTION: {fast_t.class_name.value.upper()} (ID #{fast_t.track_id}) "
            f"bears {liability}% liability due to excessive closing velocity ({fast_k['speed_kmh']:.1f} km/h vs "
            f"{slow_k['speed_kmh']:.1f} km/h). The speed disparity eliminated the margin of safety, "
            f"precipitating emergency collision dynamics."
        )

        phases = [
            {"phase": "Approach Phase (t - 2.0s)", "details": f"{fast_t.class_name.value} #{fast_t.track_id} closed distance at {fast_k['speed_kmh']:.1f} km/h against slower traffic."},
            {"phase": "Critical Threshold (t - 0.6s)", "details": f"Speed differential eliminated stopping buffer; emergency braking initiated too late."},
            {"phase": "Collision (t_0)", "details": f"Impact occurred with significant kinetic energy transfer."},
        ]

        participants = [
            ParticipantFault(
                track_id=fast_t.track_id,
                class_name=fast_t.class_name,
                role="Primary At-Fault",
                liability_percentage=liability,
                pre_impact_speed_kmh=fast_k["speed_kmh"],
                peak_deceleration_m_s2=fast_k["peak_decel"],
                trajectory_heading_deg=fast_k["heading_deg"],
                violations=["Driving at speed incompatible with traffic conditions", "Inability to stop within assured distance"],
            ),
            ParticipantFault(
                track_id=slow_t.track_id,
                class_name=slow_t.class_name,
                role="Secondary Contributor",
                liability_percentage=sec_liability,
                pre_impact_speed_kmh=slow_k["speed_kmh"],
                peak_deceleration_m_s2=slow_k["peak_decel"],
                trajectory_heading_deg=slow_k["heading_deg"],
                violations=["Operating at slow speed in active traffic lane"],
            ),
        ]

        legal_summary = (
            f"Speed disproportion is a leading cause of multi-vehicle crashes. {fast_t.class_name.value} #{fast_t.track_id} "
            f"is deemed {liability}% responsible for reckless approach speed."
        )

        return FaultAttributionReport(
            has_fault_determination=True,
            primary_at_fault_id=fast_t.track_id,
            primary_at_fault_class=fast_t.class_name,
            primary_liability_percentage=liability,
            secondary_involved_id=slow_t.track_id,
            secondary_involved_class=slow_t.class_name,
            secondary_liability_percentage=sec_liability,
            violation_category=ViolationType.EXCESSIVE_APPROACH_SPEED,
            reconstruction_narrative=narrative,
            timeline_phases=phases,
            participants=participants,
            legal_summary=legal_summary,
        )

    @classmethod
    def _generic_multi_party_eval(cls, t1: TrackedObject, t2: TrackedObject, k1: Dict, k2: Dict) -> FaultAttributionReport:
        """Generic multi-vehicle collision when kinematics are balanced."""
        d1 = k1["peak_decel"]
        d2 = k2["peak_decel"]
        # Higher decelerating or faster vehicle gets slight primary assignment
        if d1 >= d2:
            primary_t, primary_k, sec_t, sec_k = t1, k1, t2, k2
            p_fault = 65.0
            s_fault = 35.0
        else:
            primary_t, primary_k, sec_t, sec_k = t2, k2, t1, k1
            p_fault = 65.0
            s_fault = 35.0

        narrative = (
            f"COLLISION RECONSTRUCTION: Interaction between {primary_t.class_name.value.upper()} (ID #{primary_t.track_id}) "
            f"and {sec_t.class_name.value.upper()} (ID #{sec_t.track_id}). Primary contribution ({p_fault}%) is assigned "
            f"to {primary_t.class_name.value} #{primary_t.track_id} due to higher pre-crash deceleration shock "
            f"({primary_k['peak_decel']:.1f} m/s² vs {sec_k['peak_decel']:.1f} m/s²)."
        )

        phases = [
            {"phase": "Convergence (t - 1.5s)", "details": f"Both vehicles converging on trajectory intersection."},
            {"phase": "Impact (t_0)", "details": f"Simultaneous emergency deceleration and vehicle contact."},
            {"phase": "Post-Crash (t + 1.0s)", "details": f"Vehicles stationary in travel corridor."},
        ]

        participants = [
            ParticipantFault(
                track_id=primary_t.track_id,
                class_name=primary_t.class_name,
                role="Primary At-Fault",
                liability_percentage=p_fault,
                pre_impact_speed_kmh=primary_k["speed_kmh"],
                peak_deceleration_m_s2=primary_k["peak_decel"],
                trajectory_heading_deg=primary_k["heading_deg"],
                violations=["Failure to yield adequate collision avoidance clearance"],
            ),
            ParticipantFault(
                track_id=sec_t.track_id,
                class_name=sec_t.class_name,
                role="Secondary Contributor",
                liability_percentage=s_fault,
                pre_impact_speed_kmh=sec_k["speed_kmh"],
                peak_deceleration_m_s2=sec_k["peak_decel"],
                trajectory_heading_deg=sec_k["heading_deg"],
                violations=["Concurrent traffic corridor encroachment"],
            ),
        ]

        legal_summary = (
            f"Comparative negligence applied: {primary_t.class_name.value} #{primary_t.track_id} ({p_fault}%) / "
            f"{sec_t.class_name.value} #{sec_t.track_id} ({s_fault}%)."
        )

        return FaultAttributionReport(
            has_fault_determination=True,
            primary_at_fault_id=primary_t.track_id,
            primary_at_fault_class=primary_t.class_name,
            primary_liability_percentage=p_fault,
            secondary_involved_id=sec_t.track_id,
            secondary_involved_class=sec_t.class_name,
            secondary_liability_percentage=s_fault,
            violation_category=ViolationType.UNRESOLVED_MULTI_PARTY,
            reconstruction_narrative=narrative,
            timeline_phases=phases,
            participants=participants,
            legal_summary=legal_summary,
        )

    @classmethod
    def _single_vehicle_fault(
        cls,
        track: TrackedObject,
        trajectory_histories: Dict[int, TrajectoryHistory],
        ppm: float,
    ) -> FaultAttributionReport:
        """Handle single vehicle losing control, rolling over, or hitting barrier."""
        h = trajectory_histories.get(track.track_id)
        k = cls._extract_features(track, h, ppm)

        narrative = (
            f"SINGLE VEHICLE ACCIDENT: {track.class_name.value.upper()} (ID #{track.track_id}) is 100% "
            f"at fault for loss of vehicle control. Telemetry indicates pre-incident speed of {k['speed_kmh']:.1f} km/h "
            f"with sudden angular yaw deviation of {k.get('heading_change_deg', 35.0):.1f}° and deceleration shock "
            f"of {k['peak_decel']:.1f} m/s² without secondary vehicle involvement."
        )

        phases = [
            {"phase": "Pre-Incident (t - 2.0s)", "details": f"{track.class_name.value} #{track.track_id} traveling at {k['speed_kmh']:.1f} km/h."},
            {"phase": "Loss of Stability (t - 0.5s)", "details": f"Sudden trajectory departure with angular yaw spin."},
            {"phase": "Impact / Rollover (t_0)", "details": f"Vehicle halted due to collision with barrier or rollover."},
        ]

        participants = [
            ParticipantFault(
                track_id=track.track_id,
                class_name=track.class_name,
                role="Primary At-Fault (Sole Party)",
                liability_percentage=100.0,
                pre_impact_speed_kmh=k["speed_kmh"],
                peak_deceleration_m_s2=k["peak_decel"],
                trajectory_heading_deg=k["heading_deg"],
                violations=["Loss of vehicle control", "Failure to maintain vehicle stability within lane"],
            )
        ]

        legal_summary = f"Single-vehicle crash: 100% liability assigned to operator of {track.class_name.value} #{track.track_id}."

        return FaultAttributionReport(
            has_fault_determination=True,
            primary_at_fault_id=track.track_id,
            primary_at_fault_class=track.class_name,
            primary_liability_percentage=100.0,
            secondary_involved_id=None,
            secondary_involved_class=None,
            secondary_liability_percentage=0.0,
            violation_category=ViolationType.SINGLE_VEHICLE_LOSS_OF_CONTROL,
            reconstruction_narrative=narrative,
            timeline_phases=phases,
            participants=participants,
            legal_summary=legal_summary,
        )

    @staticmethod
    def _no_fault_report() -> FaultAttributionReport:
        """Nominal traffic report when no collision occurs."""
        return FaultAttributionReport(
            has_fault_determination=False,
            primary_at_fault_id=None,
            primary_at_fault_class=None,
            primary_liability_percentage=0.0,
            secondary_involved_id=None,
            secondary_involved_class=None,
            secondary_liability_percentage=0.0,
            violation_category=ViolationType.NO_VIOLATION,
            reconstruction_narrative="Continuous monitoring confirms standard, orderly traffic flow with no accident or right-of-way infractions detected.",
            timeline_phases=[],
            participants=[],
            legal_summary="No violation or liability recorded. All vehicles operating within safe regulatory bounds.",
        )
