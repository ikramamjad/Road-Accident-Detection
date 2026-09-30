"""
Real-Time Road Accident Detection System — Command-Line Runner.
Executes the multi-modal pipeline on live webcam, video file, or synthetic simulation.
"""

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from road_accident_detection.pipeline.engine import AccidentDetectionPipeline


def generate_synthetic_stream(num_frames: int = 120, width: int = 1280, height: int = 720):
    """
    Generate synthetic frames simulating an approaching motorcycle and crossing pedestrian,
    leading to an emergency swerve / collision and fall.
    """
    for f in range(num_frames):
        # Road asphalt background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw road lane markings
        cv2.line(frame, (0, height // 2), (width, height // 2), (255, 255, 255), 2)
        for x in range(0, width, 60):
            cv2.line(frame, (x, height * 2 // 3), (x + 30, height * 2 // 3), (0, 255, 255), 3)

        # Vehicle 1 (Car): moving rightwards
        car_x = int(100 + f * 7.5)
        car_y = int(height * 0.55)
        cv2.rectangle(frame, (car_x, car_y), (car_x + 120, car_y + 60), (180, 80, 50), -1)
        cv2.putText(frame, "CAR", (car_x + 10, car_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Vehicle 2 (Motorcycle / Rider): converging from right
        moto_speed = 10.0 if f < 45 else 1.0  # Decelerates abruptly post collision
        moto_x = int(width - 150 - f * moto_speed)
        moto_y = int(height * 0.58)
        cv2.rectangle(frame, (moto_x, moto_y), (moto_x + 70, moto_y + 40), (50, 180, 80), -1)

        # Pedestrian: crossing road, falls at frame 48
        ped_x = int(width * 0.5)
        if f < 48:
            ped_y = int(height * 0.35 + f * 3.5)
            # Upright pedestrian box (taller than wide)
            cv2.rectangle(frame, (ped_x - 15, ped_y - 45), (ped_x + 15, ped_y + 45), (50, 50, 220), -1)
        else:
            # Fallen pedestrian on ground (wider than tall)
            ped_y = int(height * 0.52)
            cv2.rectangle(frame, (ped_x - 45, ped_y - 12), (ped_x + 45, ped_y + 12), (50, 50, 220), -1)

        yield frame


def main():
    parser = argparse.ArgumentParser(description="Real-Time Road Accident Detection System")
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help="Input source: video file path, webcam device index (e.g. '0'), or 'synthetic'",
    )
    _ROOT = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--config",
        type=str,
        default=str(_ROOT / "configs" / "pipeline_config.yaml"),
        help="Pipeline configuration YAML path",
    )
    parser.add_argument(
        "--camera_config",
        type=str,
        default=str(_ROOT / "configs" / "camera_config.json"),
        help="Camera metadata JSON path",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Maximum frames to process (useful for benchmarks/tests)",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default=None,
        help="Path to save annotated output video (MP4)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display live GUI window",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Print detailed latency profiler table at completion",
    )
    parser.add_argument(
        "--mock_models",
        action="store_true",
        default=False,
        help="Use deterministic mock detector/pose for fast testing without pretrained weight downloads",
    )
    args = parser.parse_args()

    print("[Init] Initializing Accident Detection Pipeline...")
    pipeline = AccidentDetectionPipeline(
        config_path=args.config,
        camera_config_path=args.camera_config,
        mock_mode=args.mock_models,
    )

    writer = None
    frame_count = 0
    start_time = time.time()

    print(f"[Run] Processing stream from source: {args.source} (max frames: {args.duration})")

    if args.source == "synthetic":
        stream = generate_synthetic_stream(num_frames=args.duration)
    else:
        # Check if integer webcam or file
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[Error] Failed to open video source: {args.source}")
            return

        def cap_stream():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

        stream = cap_stream()

    try:
        for raw_frame in stream:
            frame_count += 1
            result = pipeline.process_frame(raw_frame, render_annotated=True)

            if result.debounce_output.is_event_fired:
                print(f"\n[ALERT] Frame {frame_count}: ACCIDENT DETECTED! Severity: {result.event_record.severity.upper()}")
                print(f"        Attribution: {result.fused_risk.primary_driver} ({result.fused_risk.fused_score:.2f})")
                print(f"        Justification: {result.event_record.severity_justification}")
            elif result.debounce_output.is_near_miss_fired:
                print(f"\n[NOTICE] Frame {frame_count}: NEAR-MISS INCIDENT LOGGED! Fused Score: {result.fused_risk.fused_score:.2f}")

            # Optional save
            if args.save_video and result.annotated_frame is not None:
                if writer is None:
                    h, w = result.annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (w, h))
                writer.write(result.annotated_frame)

            # Optional display
            if args.display and result.annotated_frame is not None:
                cv2.imshow("Real-Time Road Accident Detection", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.duration and frame_count >= args.duration:
                break
    finally:
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    fps = frame_count / max(total_time, 1e-3)
    print(f"\n[Completed] Processed {frame_count} frames in {total_time:.2f}s ({fps:.1f} FPS)")

    if args.benchmark:
        print("\n" + pipeline.profiler.format_table())


if __name__ == "__main__":
    main()
