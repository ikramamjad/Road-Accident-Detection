"""
End-to-End Real-Time Road Accident Detection Pipeline.
Integrates YOLO11 -> BoT-SORT -> Pose -> Kinematics -> Causal GRU -> Fusion -> Logging.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

from ..detection.classes import RoadClass, is_person, is_two_wheeler, is_vulnerable_road_user
from ..detection.yolo_detector import YOLO11Detector, DetectionResult
from ..tracking.tracker import BoTSORTTracker, TrackedObject
from ..pose.pose_detector import YOLO11PoseDetector, PersonPose
from ..pose.posture_analyzer import PostureAnalyzer, PostureAnomalyReport
from ..kinematics.motion_analyzer import MotionAnalyzer, KinematicState
from ..kinematics.ttc_calculator import TTCCalculator, TTCPairResult
from ..kinematics.kinematic_risk import KinematicRiskEngine, KinematicRiskReport
from ..temporal.causal_gru import TemporalInferenceEngine
from ..fusion.multi_channel_fusion import ChannelScores, FusedRiskResult, MultiChannelFusionEngine
from ..fusion.debouncer import TemporalDebouncer, DebounceOutput, DebounceStatus
from ..fusion.calibrator import ConfidenceCalibrator
from ..fusion.severity_classifier import SeverityClassifier, IncidentSeverity, SeverityAssessment
from ..privacy.face_plate_redaction import PrivacyRedactor
from ..logging.clip_recorder import CircularClipRecorder, RecordingTrigger
from ..logging.explainability import ExplainabilityReporter, ExplainabilityReport
from ..logging.event_logger import EventLogger, AccidentEventRecord
from .profiler import PipelineProfiler


@dataclass
class PipelineFrameResult:
    frame_idx: int
    timestamp: float
    tracks: List[TrackedObject]
    fused_risk: FusedRiskResult
    debounce_output: DebounceOutput
    kinematic_report: KinematicRiskReport
    posture_reports: List[PostureAnomalyReport]
    event_record: Optional[AccidentEventRecord] = None
    annotated_frame: Optional[np.ndarray] = None


class AccidentDetectionPipeline:
    """Master Orchestrator for Real-Time Road Accident Detection."""

    def __init__(
        self,
        config_path: str = "d:/Road Accident Detection/configs/pipeline_config.yaml",
        camera_config_path: str = "d:/Road Accident Detection/configs/camera_config.json",
        mock_mode: bool = False,
    ):
        self.config_path = config_path
        self.camera_config_path = camera_config_path
        self.mock_mode = mock_mode

        # Load configurations
        self.config = self._load_yaml(config_path)
        self.camera_meta = self._load_json(camera_config_path)

        fps = float(self.config.get("kinematics", {}).get("fps", 30.0))
        ppm = float(self.config.get("kinematics", {}).get("pixels_per_meter", 15.0))

        # 1. Detection Stage
        det_cfg = self.config.get("detection", {})
        self.detector = YOLO11Detector(
            model_name_or_path=det_cfg.get("model_name", "yolo11n.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.40),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            device=det_cfg.get("device", "cpu"),
            mock_mode=mock_mode,
        )

        # 2. Tracking Stage
        track_cfg = self.config.get("tracking", {})
        self.tracker = BoTSORTTracker(
            track_high_thresh=track_cfg.get("track_high_thresh", 0.5),
            track_low_thresh=track_cfg.get("track_low_thresh", 0.1),
            new_track_thresh=track_cfg.get("new_track_thresh", 0.6),
            track_buffer_frames=track_cfg.get("track_buffer", 30),
            fps=fps,
        )

        # 3. Pose Estimation & Analysis
        pose_cfg = self.config.get("pose", {})
        self.pose_detector = YOLO11PoseDetector(
            model_name_or_path=pose_cfg.get("model_name", "yolo11n-pose.pt"),
            confidence_threshold=pose_cfg.get("confidence_threshold", 0.45),
            mock_mode=mock_mode,
        )
        self.posture_analyzer = PostureAnalyzer(
            fall_torso_angle_thresh=pose_cfg.get("fall_torso_angle_threshold", 35.0),
            prolonged_ground_frames=pose_cfg.get("prolonged_ground_frames_threshold", 12),
        )

        # 4. Kinematics & TTC
        kin_cfg = self.config.get("kinematics", {})
        self.kinematic_engine = KinematicRiskEngine(
            pixels_per_meter=ppm,
            fps=fps,
            ttc_critical_thresh=kin_cfg.get("ttc_critical_threshold", 1.5),
            ttc_warning_thresh=kin_cfg.get("ttc_warning_threshold", 2.5),
            hard_braking_thresh=kin_cfg.get("sudden_deceleration_threshold", -4.5),
        )

        # 5. Temporal Modeling (Causal GRU)
        temp_cfg = self.config.get("temporal", {})
        self.temporal_engine = TemporalInferenceEngine(
            input_dim=temp_cfg.get("feature_dim", 12),
            hidden_dim=temp_cfg.get("hidden_dim", 64),
            num_layers=temp_cfg.get("num_layers", 2),
        )

        # 6. Multi-Channel Fusion & Debouncing
        fus_cfg = self.config.get("fusion", {})
        self.fusion_engine = MultiChannelFusionEngine(
            method=fus_cfg.get("method", "weighted_sum"),
            weights=fus_cfg.get("weights"),
            trigger_threshold=fus_cfg.get("trigger_threshold", 0.68),
        )
        self.debouncer = TemporalDebouncer(
            trigger_threshold=fus_cfg.get("trigger_threshold", 0.68),
            near_miss_threshold=fus_cfg.get("near_miss_lower_threshold", 0.45),
            consecutive_frames_required=fus_cfg.get("debounce_consecutive_frames", 6),
            cooldown_frames=fus_cfg.get("event_cooldown_frames", 150),
        )
        self.calibrator = ConfidenceCalibrator()

        # 7. Privacy Redactor
        priv_cfg = self.config.get("privacy", {})
        self.redactor = PrivacyRedactor(
            redact_faces=priv_cfg.get("redact_faces", True),
            redact_plates=priv_cfg.get("redact_license_plates", True),
            blur_kernel_size=priv_cfg.get("blur_kernel_size", 31),
            mode=priv_cfg.get("anonymize_mode", "gaussian_blur"),
        )

        # 8. Evidence Clip Recorder & Event Logger
        log_cfg = self.config.get("logging", {})
        self.clip_recorder = CircularClipRecorder(
            output_dir=log_cfg.get("clips_dir", "d:/Road Accident Detection/data/recordings/clips"),
            pre_buffer_seconds=log_cfg.get("pre_buffer_seconds", 10),
            post_buffer_seconds=log_cfg.get("post_buffer_seconds", 10),
            fps=fps,
        )
        self.event_logger = EventLogger(
            output_dir=log_cfg.get("events_dir", "d:/Road Accident Detection/data/recordings/events"),
            camera_metadata=self.camera_meta,
        )

        # Latency Profiler
        self.profiler = PipelineProfiler()
        self.frame_index = 0

    def process_frame(
        self,
        frame: np.ndarray,
        render_annotated: bool = True,
    ) -> PipelineFrameResult:
        """
        Execute full real-time pipeline on a single frame.
        """
        self.frame_index += 1
        curr_frame = self.frame_index
        curr_time = curr_frame / max(self.tracker.fps, 1.0)
        img_h, img_w = frame.shape[:2]

        self.profiler.start_frame()

        # Stage 1: Detection
        with self.profiler.stage("yolo_detection"):
            detections = self.detector.detect(frame)

        # Stage 2: Tracking
        with self.profiler.stage("botsort_tracking"):
            tracks = self.tracker.update(detections, frame_idx=curr_frame, timestamp=curr_time)

        # Stage 3: Pose Estimation
        with self.profiler.stage("pose_estimation"):
            needs_pose = any(
                is_person(d.class_name) or is_vulnerable_road_user(d.class_name) or is_two_wheeler(d.class_name)
                for d in detections
            ) or any(
                is_person(t.class_name) or is_vulnerable_road_user(t.class_name) or is_two_wheeler(t.class_name)
                for t in tracks
            )
            if needs_pose:
                poses = self.pose_detector.estimate_poses(frame)
            else:
                poses = []

        # Stage 4: Abnormal Posture Analysis
        posture_reports: List[PostureAnomalyReport] = []
        max_pose_score = 0.0
        with self.profiler.stage("posture_analysis"):
            two_wheelers = [t for t in tracks if is_two_wheeler(t.class_name)]
            for pose in poses:
                # Associate pose to track if bounding box overlaps
                matched_tid = None
                for t in tracks:
                    if is_person(t.class_name):
                        matched_tid = t.track_id
                        break
                rep = self.posture_analyzer.analyze_pose(pose, track_id=matched_tid, nearby_two_wheelers=two_wheelers)
                posture_reports.append(rep)
                if rep.anomaly_score > max_pose_score:
                    max_pose_score = rep.anomaly_score

        # Stage 5: Kinematic Analysis & TTC
        with self.profiler.stage("kinematics_and_ttc"):
            kin_report = self.kinematic_engine.evaluate(tracks, self.tracker.trajectory_manager.tracks)

        # Stage 6: Temporal Behavior Modeling (Causal GRU)
        max_temporal_score = 0.0
        with self.profiler.stage("causal_gru"):
            for track in tracks:
                hist = self.tracker.trajectory_manager.get_history(track.track_id)
                if hist and len(hist.history) >= 4:
                    feat_matrix = hist.get_feature_matrix(img_w, img_h, target_len=16)
                    temp_score = self.temporal_engine.evaluate_sequence(feat_matrix)
                    if temp_score > max_temporal_score:
                        max_temporal_score = temp_score

        # Stage 7: Multi-Channel Fusion
        max_det_conf = max([d.confidence for d in detections], default=0.0)
        with self.profiler.stage("multi_channel_fusion"):
            channel_scores = ChannelScores(
                detection_conf=max_det_conf,
                pose_anomaly=max_pose_score,
                kinematic_risk=kin_report.overall_risk_score,
                temporal_anomaly=max_temporal_score,
            )
            fusion_result = self.fusion_engine.fuse(channel_scores)

        # Stage 8: Temporal Debouncing
        with self.profiler.stage("debouncing"):
            debounce_out = self.debouncer.update(fusion_result.fused_score)

        # Stage 9: Incident Event Handling
        event_record: Optional[AccidentEventRecord] = None
        if debounce_out.is_event_fired or debounce_out.is_near_miss_fired:
            is_near_miss = debounce_out.is_near_miss_fired
            # Kinematic states of involved tracks
            kin_states = []
            for t in tracks:
                h = self.tracker.trajectory_manager.get_history(t.track_id)
                if h:
                    ks = self.kinematic_engine.motion_analyzer.analyze(h)
                    if ks:
                        kin_states.append(ks)

            severity_assess = SeverityClassifier.classify(
                involved_tracks=tracks,
                kinematic_states=kin_states,
                posture_reports=posture_reports,
                is_near_miss_trigger=is_near_miss,
            )

            explain_rep = ExplainabilityReporter.generate_report(
                fusion_result=fusion_result,
                kinematic_report=kin_report,
                posture_reports=posture_reports,
                debounce_frames=self.debouncer.consecutive_frames_required,
            )

            # Trigger video recording
            cam_id = self.camera_meta.get("camera_id", "CAM_01")
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            trigger = RecordingTrigger(
                event_id=ts_str,
                timestamp_str=ts_str,
                severity=severity_assess.severity.value,
                camera_id=cam_id,
            )
            self.clip_recorder.trigger_recording(trigger)

            # Structured event persistence
            event_record = self.event_logger.log_event(
                severity_assessment=severity_assess,
                fusion_result=fusion_result,
                explainability=explain_rep,
                involved_tracks=tracks,
                clip_path=str(self.clip_recorder._build_clip_path(trigger)),
            )

        # Stage 10: Privacy Redaction & Buffer Push
        with self.profiler.stage("privacy_and_buffer"):
            redacted_frame = self.redactor.redact_frame(frame, tracks, poses)
            self.clip_recorder.add_frame(redacted_frame)

        # Stage 11: HUD Overlay
        annotated = None
        if render_annotated:
            with self.profiler.stage("hud_overlay"):
                annotated = self._render_hud(
                    frame=redacted_frame.copy(),
                    tracks=tracks,
                    poses=poses,
                    kin_report=kin_report,
                    fusion_result=fusion_result,
                    debounce_out=debounce_out,
                )

        self.profiler.end_frame()

        return PipelineFrameResult(
            frame_idx=curr_frame,
            timestamp=curr_time,
            tracks=tracks,
            fused_risk=fusion_result,
            debounce_output=debounce_out,
            kinematic_report=kin_report,
            posture_reports=posture_reports,
            event_record=event_record,
            annotated_frame=annotated,
        )

    def _render_hud(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        poses: List[PersonPose],
        kin_report: KinematicRiskReport,
        fusion_result: FusedRiskResult,
        debounce_out: DebounceOutput,
    ) -> np.ndarray:
        """Render informative real-time heads-up display overlay."""
        h, w = frame.shape[:2]

        # Draw tracked objects
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            color = (0, 255, 0)
            if t.track_id in kin_report.hard_braking_tracks:
                color = (0, 0, 255)  # Red for braking
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t.track_id} {t.class_name.value} {t.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw critical TTC vector lines between colliding pairs
        for pair in kin_report.critical_pairs:
            hist_a = self.tracker.trajectory_manager.get_history(pair.track_id_a)
            hist_b = self.tracker.trajectory_manager.get_history(pair.track_id_b)
            if hist_a and hist_b and hist_a.latest and hist_b.latest:
                pt_a = (int(hist_a.latest.center[0]), int(hist_a.latest.center[1]))
                pt_b = (int(hist_b.latest.center[0]), int(hist_b.latest.center[1]))
                cv2.line(frame, pt_a, pt_b, (0, 0, 255), 2, cv2.LINE_AA)

        # Top HUD Banner: Risk Gauge & Channel Bars
        cv2.rectangle(frame, (10, 10), (450, 130), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (450, 130), (80, 80, 80), 1)

        risk_val = fusion_result.fused_score
        gauge_color = (0, 255, 0) if risk_val < 0.45 else ((0, 200, 255) if risk_val < 0.68 else (0, 0, 255))

        status_txt = f"STATUS: {debounce_out.status.value.upper()}"
        if debounce_out.consecutive_positive_frames > 0:
            status_txt += f" ({debounce_out.consecutive_positive_frames}/{debounce_out.frames_required})"
        cv2.putText(frame, status_txt, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Fused Risk bar
        cv2.putText(frame, f"Fused Risk: {risk_val:.2f}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        bar_w = int(240 * risk_val)
        cv2.rectangle(frame, (150, 44), (390, 58), (50, 50, 50), -1)
        cv2.rectangle(frame, (150, 44), (150 + bar_w, 58), gauge_color, -1)

        # Primary Channel
        cv2.putText(
            frame,
            f"Primary: {fusion_result.primary_driver}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        # FPS & Camera info
        fps_curr = self.profiler.get_fps()
        cv2.putText(
            frame,
            f"FPS: {fps_curr:.1f} | Cam: {self.camera_meta.get('camera_id', 'CAM_01')}",
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )

        # If Event Triggered, flash red alert banner across top
        if debounce_out.is_event_fired:
            cv2.rectangle(frame, (w // 4, 20), (3 * w // 4, 80), (0, 0, 220), -1)
            cv2.putText(
                frame,
                "*** ACCIDENT DETECTED - DISPATCH ALERT ***",
                (w // 4 + 20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
        elif debounce_out.is_near_miss_fired:
            cv2.rectangle(frame, (w // 4, 20), (3 * w // 4, 80), (0, 165, 255), -1)
            cv2.putText(
                frame,
                "*** NEAR-MISS EVENT LOGGED ***",
                (w // 4 + 40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2,
            )

        return frame

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
