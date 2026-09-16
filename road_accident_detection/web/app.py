"""
Real-Time Road Accident Detection System — Web Backend Server.
Flask application providing video upload, sample selection, pipeline execution,
and rich multi-modal forensic diagnostics.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional
import uuid

# Ensure project root and lib are on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_LIB = _ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np
from werkzeug.utils import secure_filename

from road_accident_detection.pipeline.engine import AccidentDetectionPipeline, PipelineFrameResult
from road_accident_detection.detection.classes import RoadClass
from road_accident_detection.detection.yolo_detector import DetectionResult
from road_accident_detection.pose.pose_detector import PersonPose


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "rads-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

UPLOAD_DIR = _ROOT / "data" / "web_uploads"
RESULTS_DIR = _ROOT / "data" / "web_results"
RECORDINGS_DIR = _ROOT / "data" / "recordings" / "clips"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

# Global cached pipeline instance to avoid reloading heavy YOLO models on every request
_CACHED_PIPELINE: Optional[AccidentDetectionPipeline] = None


def get_pipeline(mock_mode: bool = False) -> AccidentDetectionPipeline:
    """Return singleton pipeline instance, initialized on first use."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        cfg_path = str(_ROOT / "configs" / "pipeline_config.yaml")
        cam_path = str(_ROOT / "configs" / "camera_config.json")
        _CACHED_PIPELINE = AccidentDetectionPipeline(
            config_path=cfg_path,
            camera_config_path=cam_path,
            mock_mode=mock_mode,
        )
    return _CACHED_PIPELINE


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_synthetic_stream(num_frames: int = 120, width: int = 1280, height: int = 720):
    """Generates synthetic video frames simulating an approaching vehicle and crossing pedestrian collision."""
    for f in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Road lane markings
        cv2.line(frame, (0, height // 2), (width, height // 2), (255, 255, 255), 2)
        for x in range(0, width, 60):
            cv2.line(frame, (x, height * 2 // 3), (x + 30, height * 2 // 3), (0, 255, 255), 3)

        # Vehicle (Car)
        car_x = int(100 + f * 7.5)
        car_y = int(height * 0.55)
        cv2.rectangle(frame, (car_x, car_y), (car_x + 120, car_y + 60), (180, 80, 50), -1)
        cv2.putText(frame, "CAR", (car_x + 10, car_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Motorcycle / Rider
        moto_speed = 10.0 if f < 45 else 1.0
        moto_x = int(width - 150 - f * moto_speed)
        moto_y = int(height * 0.58)
        cv2.rectangle(frame, (moto_x, moto_y), (moto_x + 70, moto_y + 40), (50, 180, 80), -1)

        # Pedestrian: falls at frame 48
        ped_x = int(width * 0.5)
        if f < 48:
            ped_y = int(height * 0.35 + f * 3.5)
            cv2.rectangle(frame, (ped_x - 15, ped_y - 45), (ped_x + 15, ped_y + 45), (50, 50, 220), -1)
        else:
            ped_y = int(height * 0.52)
            cv2.rectangle(frame, (ped_x - 45, ped_y - 12), (ped_x + 45, ped_y + 12), (50, 50, 220), -1)

        yield frame
 
 
class SyntheticDetectorAdapter:
    """Provides ground-truth detection boxes matching generate_synthetic_stream for simulation."""
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self.confidence_threshold = 0.40

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        f = self.frame_idx
        self.frame_idx += 1
        car_x = int(100 + f * 7.5)
        moto_speed = 10.0 if f < 45 else 1.0
        moto_x = int(self.width - 150 - f * moto_speed)
        ped_y = int(self.height * 0.35 + f * 3.5) if f < 48 else int(self.height * 0.52)
        ped_bbox = (
            int(self.width * 0.5) - 15, ped_y - 45, int(self.width * 0.5) + 15, ped_y + 45
        ) if f < 48 else (
            int(self.width * 0.5) - 45, ped_y - 12, int(self.width * 0.5) + 45, ped_y + 12
        )

        return [
            DetectionResult(bbox=(car_x, int(self.height * 0.55), car_x + 120, int(self.height * 0.55) + 60), confidence=0.92, class_name=RoadClass.CAR, raw_class_id=2),
            DetectionResult(bbox=(moto_x, int(self.height * 0.58), moto_x + 70, int(self.height * 0.58) + 40), confidence=0.89, class_name=RoadClass.MOTORCYCLE, raw_class_id=3),
            DetectionResult(bbox=ped_bbox, confidence=0.88, class_name=RoadClass.PEDESTRIAN, raw_class_id=0),
        ]


class SyntheticPoseAdapter:
    """Provides keypoints for the pedestrian, simulating upright crossing then horizontal fall."""
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self.confidence_threshold = 0.45

    def estimate_poses(self, frame: np.ndarray) -> List[PersonPose]:
        f = self.frame_idx
        self.frame_idx += 1
        ped_x = float(self.width * 0.5)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 2] = 0.9

        if f < 48:
            ped_y = float(self.height * 0.35 + f * 3.5)
            kps[5] = [ped_x - 10, ped_y - 20, 0.9]
            kps[6] = [ped_x + 10, ped_y - 20, 0.9]
            kps[11] = [ped_x - 8, ped_y + 20, 0.9]
            kps[12] = [ped_x + 8, ped_y + 20, 0.9]
            bbox = (ped_x - 15, ped_y - 45, ped_x + 15, ped_y + 45)
        else:
            ped_y = float(self.height * 0.52)
            kps[5] = [ped_x - 30, ped_y, 0.9]
            kps[6] = [ped_x - 30, ped_y + 5, 0.9]
            kps[11] = [ped_x + 30, ped_y, 0.9]
            kps[12] = [ped_x + 30, ped_y + 5, 0.9]
            bbox = (ped_x - 45, ped_y - 12, ped_x + 45, ped_y + 12)

        return [PersonPose(keypoints=kps, bbox=bbox, confidence=0.88)]


@app.route("/")
def index():
    """Render the primary Landing Page."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Health check and model telemetry endpoint."""
    return jsonify({
        "status": "online",
        "pipeline_version": "1.0.0",
        "models": {
            "object_detector": "YOLO11 Nano",
            "pose_estimator": "YOLO11-Pose Nano (VRU Gated)",
            "tracker": "BoT-SORT (Re-ID + IoU)",
            "temporal_model": "Causal 2-Layer GRU (16-frame causal window)",
            "fusion_debouncer": "Weighted Sum + N=6 Frame Debounce",
            "privacy_redactor": "Automated Face & Plate Gaussian Blur"
        },
        "device": "cpu",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/sample-clips", methods=["GET"])
def api_sample_clips():
    """Return preloaded sample clips available for immediate testing."""
    samples = []
    # Real highway recording if exists
    real_clip = RECORDINGS_DIR / "93869-642182008_medium.mp4"
    if real_clip.exists():
        samples.append({
            "id": "real_highway",
            "title": "Real Highway Traffic (1440p 2.5K)",
            "description": "Multi-lane dusk expressway underpass with dense vehicular flow. Verifies false-alarm immunity and persistent multi-object tracking.",
            "source_type": "sample",
            "filename": "93869-642182008_medium.mp4",
            "resolution": "2560x1440",
            "duration": "29.5s",
            "badge": "Real CCTV"
        })

    # Synthetic collision simulation
    samples.append({
        "id": "synthetic_collision",
        "title": "Pedestrian & Motorcycle Collision Simulation",
        "description": "Controlled trajectory with converging vehicles, sudden deceleration shock, and pedestrian fall posture.",
        "source_type": "synthetic",
        "filename": None,
        "resolution": "1280x720",
        "duration": "4.0s (120 frames)",
        "badge": "Crash Simulation"
    })

    return jsonify({"sample_clips": samples})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a custom video file."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        ext = file.filename.rsplit(".", 1)[1].lower()
        file_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        safe_name = f"{file_id}.{ext}"
        save_path = UPLOAD_DIR / safe_name
        file.save(save_path)

        # Inspect basic metadata via OpenCV
        cap = cv2.VideoCapture(str(save_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()

        return jsonify({
            "message": "Upload successful",
            "file_id": file_id,
            "filename": safe_name,
            "width": width,
            "height": height,
            "fps": round(fps, 1),
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 1)
        })

    return jsonify({"error": "Unsupported video format. Allowed: MP4, AVI, MOV, MKV, WEBM"}), 400


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Run end-to-end accident detection pipeline on the requested video.
    Returns definitive verdict, severity, telemetry timeline, keyframe snapshots, and annotated video.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    source_type = data.get("source_type", "sample")
    filename = data.get("filename")
    max_frames = int(data.get("max_frames", 120))
    conf_threshold = float(data.get("conf_threshold", 0.40))
    redact_privacy = str(data.get("redact_privacy", "true")).lower() in ("true", "1", "yes")
    mock_mode = str(data.get("mock_mode", "false")).lower() in ("true", "1", "yes")

    analysis_id = f"ana_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    out_video_name = f"{analysis_id}_annotated.mp4"
    out_video_path = RESULTS_DIR / out_video_name
    peak_image_name = f"{analysis_id}_peak.jpg"
    peak_image_path = RESULTS_DIR / peak_image_name

    # Determine input stream
    if source_type == "synthetic":
        stream_gen = generate_synthetic_stream(num_frames=max_frames)
        fps = 30.0
        input_desc = "Synthetic Crash Simulation"
    else:
        # Locate video file
        target_path = None
        if filename:
            # Check uploads first
            if (UPLOAD_DIR / filename).exists():
                target_path = UPLOAD_DIR / filename
            # Check recordings
            elif (RECORDINGS_DIR / filename).exists():
                target_path = RECORDINGS_DIR / filename

        if not target_path or not target_path.exists():
            return jsonify({"error": f"Video source not found: {filename}"}), 404

        input_desc = target_path.name
        cap = cv2.VideoCapture(str(target_path))
        if not cap.isOpened():
            return jsonify({"error": "Failed to read video file"}), 500
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        def video_stream():
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
                count += 1
                if max_frames and count >= max_frames:
                    break
            cap.release()

        stream_gen = video_stream()

    # Get pipeline and configure runtime options
    pipeline = get_pipeline(mock_mode=mock_mode)
    pipeline.detector.confidence_threshold = conf_threshold
    pipeline.redactor.redact_faces = redact_privacy
    pipeline.redactor.redact_plates = redact_privacy

    orig_detector = pipeline.detector
    orig_pose = pipeline.pose_detector
    if source_type == "synthetic":
        pipeline.detector = SyntheticDetectorAdapter()
        pipeline.pose_detector = SyntheticPoseAdapter()

    # Reset frame state and profiler
    pipeline.frame_index = 0
    pipeline.debouncer.reset()

    # Processing loop
    start_time = time.time()
    frame_count = 0
    writer = None
    peak_risk = 0.0
    peak_frame_img = None
    peak_frame_idx = 0
    primary_driver = "normal_flow"

    accident_events: List[Dict[str, Any]] = []
    near_miss_events: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    keyframe_snapshots: List[Dict[str, Any]] = []

    channel_sums = {"detection_conf": 0.0, "pose_anomaly": 0.0, "kinematic_risk": 0.0, "temporal_anomaly": 0.0}

    try:
        for raw_frame in stream_gen:
            frame_count += 1
            result: PipelineFrameResult = pipeline.process_frame(raw_frame, render_annotated=True)

            curr_risk = result.fused_risk.fused_score
            driver = result.fused_risk.primary_driver

            # Accumulate channel scores
            attr = result.fused_risk.channel_attributions
            channel_sums["detection_conf"] += attr.get("detection_conf", 0.0)
            channel_sums["pose_anomaly"] += attr.get("pose_anomaly", 0.0)
            channel_sums["kinematic_risk"] += attr.get("kinematic_risk", 0.0)
            channel_sums["temporal_anomaly"] += attr.get("temporal_anomaly", 0.0)

            # Track peak risk
            if curr_risk > peak_risk:
                peak_risk = curr_risk
                peak_frame_idx = frame_count
                primary_driver = driver
                if result.annotated_frame is not None:
                    peak_frame_img = result.annotated_frame.copy()

            # Record events
            if result.debounce_output.is_event_fired and result.event_record:
                ev = {
                    "frame_idx": frame_count,
                    "timestamp_sec": round(frame_count / fps, 2),
                    "type": "ACCIDENT",
                    "severity": result.event_record.severity,
                    "justification": result.event_record.severity_justification,
                    "fused_risk": round(curr_risk, 3),
                    "primary_driver": driver,
                    "involved_tracks_count": len(result.tracks)
                }
                accident_events.append(ev)

                # Save keyframe snapshot
                snap_name = f"{analysis_id}_kf_{frame_count}.jpg"
                cv2.imwrite(str(RESULTS_DIR / snap_name), result.annotated_frame)
                keyframe_snapshots.append({
                    "frame_idx": frame_count,
                    "timestamp_sec": round(frame_count / fps, 2),
                    "label": f"Crash Event ({result.event_record.severity.upper()})",
                    "url": f"/media/results/{snap_name}"
                })

            elif result.debounce_output.is_near_miss_fired:
                ev = {
                    "frame_idx": frame_count,
                    "timestamp_sec": round(frame_count / fps, 2),
                    "type": "NEAR_MISS",
                    "severity": "near_miss",
                    "justification": f"High kinematic convergence or rapid braking (Risk: {curr_risk:.2f})",
                    "fused_risk": round(curr_risk, 3),
                    "primary_driver": driver,
                    "involved_tracks_count": len(result.tracks)
                }
                near_miss_events.append(ev)

            # Sample timeline every 10 frames
            if frame_count % 10 == 0:
                timeline.append({
                    "frame_idx": frame_count,
                    "time_sec": round(frame_count / fps, 2),
                    "risk_score": round(curr_risk, 3),
                    "active_tracks": len(result.tracks),
                    "status": "ALERT" if result.debounce_output.is_event_fired else ("NEAR_MISS" if result.debounce_output.is_near_miss_fired else "NORMAL")
                })

            # Video encoding
            if result.annotated_frame is not None:
                if writer is None:
                    h, w = result.annotated_frame.shape[:2]
                    # Resize to 1280x720 max for web video streaming performance
                    scale = min(1.0, 1280.0 / max(w, 1))
                    target_w = int(w * scale)
                    target_h = int(h * scale)
                    target_w = target_w if target_w % 2 == 0 else target_w - 1
                    target_h = target_h if target_h % 2 == 0 else target_h - 1

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_video_path), fourcc, 25.0, (target_w, target_h))

                if (target_w, target_h) != (w, h):
                    write_frame = cv2.resize(result.annotated_frame, (target_w, target_h))
                else:
                    write_frame = result.annotated_frame
                writer.write(write_frame)

    finally:
        if writer:
            writer.release()
        if source_type == "synthetic":
            pipeline.detector = orig_detector
            pipeline.pose_detector = orig_pose

    elapsed_time = max(0.01, time.time() - start_time)
    avg_fps = frame_count / elapsed_time

    # Save peak keyframe
    if peak_frame_img is not None:
        cv2.imwrite(str(peak_image_path), peak_frame_img)
    elif frame_count > 0:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(peak_image_path), dummy)

    # Determine grand verdict
    if len(accident_events) > 0:
        verdict = "ACCIDENT_DETECTED"
        verdict_label = "Accident Detected"
        severity = accident_events[0]["severity"].upper()
        verdict_color = "red"
        explanation = (
            f"Multi-channel fusion confirmed a sustained incident ({severity}) "
            f"primarily driven by {accident_events[0]['primary_driver'].replace('_', ' ')}."
        )
    elif len(near_miss_events) > 0 or peak_risk >= 0.65:
        verdict = "NEAR_MISS"
        verdict_label = "Near-Miss Hazard Detected"
        severity = "NEAR_MISS"
        verdict_color = "yellow"
        explanation = "Critical kinematic deceleration / near-collision detected, but no persistent structural impact or fall occurred."
    else:
        verdict = "NO_ACCIDENT"
        verdict_label = "Safe Traffic Flow — No Accident"
        severity = "NONE"
        verdict_color = "green"
        explanation = "Traffic trajectories, Time-to-Collision (TTC), and poses remained within stable, safe operational bounds."

    # Compute channel contribution percentages
    total_channel = sum(channel_sums.values()) or 1.0
    contributions = {
        "kinematic_risk": round((channel_sums["kinematic_risk"] / total_channel) * 100, 1),
        "temporal_anomaly": round((channel_sums["temporal_anomaly"] / total_channel) * 100, 1),
        "detection_conf": round((channel_sums["detection_conf"] / total_channel) * 100, 1),
        "pose_anomaly": round((channel_sums["pose_anomaly"] / total_channel) * 100, 1),
    }

    if not keyframe_snapshots:
        keyframe_snapshots.append({
            "frame_idx": peak_frame_idx,
            "timestamp_sec": round(peak_frame_idx / fps, 2),
            "label": f"Peak Risk Frame ({round(peak_risk, 2)})",
            "url": f"/media/results/{peak_image_name}"
        })

    response_payload = {
        "analysis_id": analysis_id,
        "input_source": input_desc,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "verdict_color": verdict_color,
        "severity": severity,
        "explanation": explanation,
        "max_risk_score": round(peak_risk, 3),
        "peak_frame_idx": peak_frame_idx,
        "primary_driver": primary_driver.replace("_", " ").title(),
        "channel_contributions": contributions,
        "frames_processed": frame_count,
        "duration_analyzed_sec": round(frame_count / fps, 2),
        "processing_time_sec": round(elapsed_time, 2),
        "processing_fps": round(avg_fps, 1),
        "accident_events": accident_events,
        "near_miss_events": near_miss_events,
        "timeline": timeline,
        "keyframe_url": f"/media/results/{peak_image_name}",
        "video_url": f"/media/results/{out_video_name}",
        "keyframe_snapshots": keyframe_snapshots
    }

    return jsonify(response_payload)


@app.route("/media/<category>/<path:filename>", methods=["GET"])
def serve_media(category: str, filename: str):
    """Serve media files from uploads, results, or recordings clips."""
    safe_name = secure_filename(filename)
    if category == "uploads":
        return send_from_directory(UPLOAD_DIR, safe_name)
    elif category == "results":
        return send_from_directory(RESULTS_DIR, safe_name)
    elif category == "recordings":
        return send_from_directory(RECORDINGS_DIR, safe_name)
    return jsonify({"error": "Unknown media category"}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
