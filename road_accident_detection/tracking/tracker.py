"""
BoT-SORT / Multi-Object Tracker Implementation.
Maintains persistent IDs across occlusions, associates detections with tracklets,
and interfaces directly with TrajectoryBufferManager.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..detection.classes import RoadClass
from ..detection.yolo_detector import DetectionResult
from .trajectory_buffer import TrajectoryBufferManager, TrajectoryPoint


class TrackState(Enum):
    NEW = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


@dataclass
class TrackedObject:
    """Represents a persistently tracked road object."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: RoadClass
    raw_class_id: int
    state: TrackState = TrackState.NEW
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    is_overlapped: bool = False
    # Kalman-like smooth velocity state
    vx: float = 0.0
    vy: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class BoTSORTTracker:
    """
    Persistent Object Tracker with BoT-SORT principles:
    - High & Low confidence bipartite association
    - Trajectory continuity & Occlusion re-identification
    - Trajectory buffer integration
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer_frames: int = 30,
        match_thresh: float = 0.8,
        fps: float = 30.0,
        overlap_retention_frames: int = 5,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.max_lost_frames = track_buffer_frames
        self.overlap_retention_frames = overlap_retention_frames
        self.match_thresh = match_thresh
        self.fps = fps

        self._next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.lost_objects: Dict[int, TrackedObject] = {}
        self.frame_count = 0
        self.trajectory_manager = TrajectoryBufferManager(
            max_history_len=45,
            max_missing_frames=track_buffer_frames,
            fps=fps,
        )

    def update(
        self,
        detections: List[DetectionResult],
        frame_idx: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        """
        Process detections for current frame and return active tracked objects.
        """
        self.frame_count += 1
        curr_frame = frame_idx if frame_idx is not None else self.frame_count
        curr_time = timestamp if timestamp is not None else (curr_frame / max(self.fps, 1.0))

        # Separate detections into high and low confidence
        high_dets = [d for d in detections if d.confidence >= self.track_high_thresh]
        low_dets = [d for d in detections if self.track_low_thresh <= d.confidence < self.track_high_thresh]

        # 1. Predict track positions (simple linear motion prior)
        for track in list(self.tracked_objects.values()) + list(self.lost_objects.values()):
            track.age += 1
            track.time_since_update += 1
            # Propagate box by velocity
            x1, y1, x2, y2 = track.bbox
            track.bbox = (x1 + track.vx, y1 + track.vy, x2 + track.vx, y2 + track.vy)

        # 2. First association: Match active tracks with high-conf detections
        active_tracks = list(self.tracked_objects.values())
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
            active_tracks, high_dets, iou_thresh=1.0 - self.match_thresh
        )

        # Update matched tracks from stage 1
        for track, det in matched_tracks_1:
            self._update_track_state(track, det, curr_frame, curr_time)

        # 3. Second association: Match unmatched active tracks with low-conf detections
        unmatched_active_tracks = [active_tracks[i] for i in unmatched_tracks_1]
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            unmatched_active_tracks, low_dets, iou_thresh=0.5
        )

        for track, det in matched_tracks_2:
            self._update_track_state(track, det, curr_frame, curr_time)

        # Remaining unmatched active tracks: apply temporal overlapping / coasting
        for idx in unmatched_tracks_2:
            track = unmatched_active_tracks[idx]
            if track.time_since_update <= self.overlap_retention_frames:
                # Retain active track by overlapping projected position forward
                track.is_overlapped = True
                self.trajectory_manager.update_track(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    frame_idx=curr_frame,
                    timestamp=curr_time,
                    bbox=track.bbox,
                )
            else:
                track.state = TrackState.LOST
                track.is_overlapped = False
                self.lost_objects[track.track_id] = track
                self.tracked_objects.pop(track.track_id, None)

        # 4. Third association: Match lost tracks with remaining unmatched high-conf detections
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets_1]
        lost_track_list = list(self.lost_objects.values())
        matched_tracks_3, unmatched_tracks_3, remaining_dets_idx = self._associate(
            lost_track_list, unmatched_high_dets, iou_thresh=0.45
        )

        for track, det in matched_tracks_3:
            track.state = TrackState.TRACKED
            self.tracked_objects[track.track_id] = track
            self.lost_objects.pop(track.track_id, None)
            self._update_track_state(track, det, curr_frame, curr_time)

        # 5. Initialize new tracks from unassigned high-conf detections
        for idx in remaining_dets_idx:
            det = unmatched_high_dets[idx]
            if det.confidence >= self.new_track_thresh:
                new_track = TrackedObject(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_name=det.class_name,
                    raw_class_id=det.raw_class_id,
                    state=TrackState.TRACKED,
                    hits=1,
                    age=1,
                    time_since_update=0,
                )
                self._next_id += 1
                self.tracked_objects[new_track.track_id] = new_track
                self.trajectory_manager.update_track(
                    track_id=new_track.track_id,
                    class_name=new_track.class_name,
                    frame_idx=curr_frame,
                    timestamp=curr_time,
                    bbox=new_track.bbox,
                )

        # 6. Remove lost tracks exceeding max_lost_frames
        dead_ids = [
            tid for tid, track in self.lost_objects.items()
            if track.time_since_update > self.max_lost_frames
        ]
        for tid in dead_ids:
            self.lost_objects.pop(tid, None)

        self.trajectory_manager.prune_inactive(curr_frame)
        return list(self.tracked_objects.values())

    def _associate(
        self,
        tracks: List[TrackedObject],
        detections: List[DetectionResult],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[TrackedObject, DetectionResult]], List[int], List[int]]:
        """Greedy IoU bipartite association."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Build IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(track.bbox, det.bbox)

        matched_tracks: List[Tuple[TrackedObject, DetectionResult]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        # Greedy match on highest IoU
        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            sub_matrix = iou_matrix[list(unmatched_tracks), :][:, list(unmatched_dets)]
            max_val = np.max(sub_matrix)
            if max_val < (1.0 - iou_thresh):
                break

            # Find coordinates of max
            indices = np.where(iou_matrix == max_val)
            t_idx = indices[0][0]
            d_idx = indices[1][0]

            matched_tracks.append((tracks[t_idx], detections[d_idx]))
            unmatched_tracks.discard(t_idx)
            unmatched_dets.discard(d_idx)
            iou_matrix[t_idx, :] = -1.0
            iou_matrix[:, d_idx] = -1.0

        return matched_tracks, list(unmatched_tracks), list(unmatched_dets)

    def _update_track_state(
        self,
        track: TrackedObject,
        det: DetectionResult,
        frame_idx: int,
        timestamp: float,
    ) -> None:
        """Update track coordinates, smooth velocities, and trajectory history."""
        old_center = track.center
        track.bbox = det.bbox
        track.confidence = det.confidence
        track.class_name = det.class_name
        track.hits += 1
        track.time_since_update = 0
        track.is_overlapped = False

        # Update smooth linear velocity
        new_center = track.center
        dt = 1.0 / max(self.fps, 1.0)
        inst_vx = (new_center[0] - old_center[0])
        inst_vy = (new_center[1] - old_center[1])
        track.vx = 0.7 * inst_vx + 0.3 * track.vx
        track.vy = 0.7 * inst_vy + 0.3 * track.vy

        self.trajectory_manager.update_track(
            track_id=track.track_id,
            class_name=track.class_name,
            frame_idx=frame_idx,
            timestamp=timestamp,
            bbox=track.bbox,
        )
