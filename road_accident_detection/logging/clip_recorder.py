"""
Circular Clip Recorder.
Maintains a pre-incident RAM ring buffer and captures post-incident video
to generate self-contained evidence clips (e.g. 10s pre, 10s post).
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import threading
from typing import Deque, List, Optional
import cv2
import numpy as np


@dataclass
class RecordingTrigger:
    event_id: str
    timestamp_str: str
    severity: str
    camera_id: str


class CircularClipRecorder:
    """In-memory circular frame buffer and clip persistence engine."""

    def __init__(
        self,
        output_dir: str = "d:/Road Accident Detection/data/recordings/clips",
        pre_buffer_seconds: int = 10,
        post_buffer_seconds: int = 10,
        fps: float = 30.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = max(fps, 1.0)
        self.pre_buffer_frames = int(pre_buffer_seconds * self.fps)
        self.post_buffer_frames = int(post_buffer_seconds * self.fps)

        self.ring_buffer: Deque[np.ndarray] = deque(maxlen=self.pre_buffer_frames)
        self._active_sessions: List[dict] = []
        self._lock = threading.Lock()

    def add_frame(self, frame: np.ndarray) -> List[str]:
        """
        Push new frame into buffer and feed any ongoing post-trigger recordings.
        Returns paths of any completed clips saved on this frame.
        """
        if frame is None or frame.size == 0:
            return []

        completed_clips: List[str] = []

        with self._lock:
            # 1. Add copy to circular pre-buffer
            self.ring_buffer.append(frame.copy())

            # 2. Advance active post-trigger recording sessions
            remaining_sessions = []
            for sess in self._active_sessions:
                sess["post_frames"].append(frame.copy())
                sess["frames_left"] -= 1

                if sess["frames_left"] <= 0:
                    # Clip is complete! Spawn async disk save
                    clip_path = self._build_clip_path(sess["trigger"])
                    all_frames = sess["pre_frames"] + sess["post_frames"]

                    save_thread = threading.Thread(
                        target=self._save_video_worker,
                        args=(all_frames, clip_path, self.fps),
                        daemon=True,
                    )
                    save_thread.start()
                    completed_clips.append(str(clip_path))
                else:
                    remaining_sessions.append(sess)

            self._active_sessions = remaining_sessions

        return completed_clips

    def trigger_recording(self, trigger: RecordingTrigger) -> None:
        """Initiate post-incident recording session using current pre-buffer."""
        with self._lock:
            # Snapshot current pre-buffer
            pre_snapshot = list(self.ring_buffer)
            session = {
                "trigger": trigger,
                "pre_frames": pre_snapshot,
                "post_frames": [],
                "frames_left": self.post_buffer_frames,
            }
            self._active_sessions.append(session)

    def _build_clip_path(self, trigger: RecordingTrigger) -> Path:
        filename = f"{trigger.camera_id}_{trigger.severity}_{trigger.event_id[:8]}.mp4"
        return self.output_dir / filename

    @staticmethod
    def _save_video_worker(frames: List[np.ndarray], target_path: Path, fps: float) -> None:
        """Background thread worker to encode MP4 video file."""
        if not frames:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(target_path), fourcc, fps, (w, h))

        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
