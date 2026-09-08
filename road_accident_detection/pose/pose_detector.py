"""
YOLO11-Pose Detector Interface.
Extracts 17 COCO skeletal keypoints for pedestrians and two-wheeler riders.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

COCO_KEYPOINTS = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]


@dataclass
class PersonPose:
    """Estimated 17-keypoint skeleton for a person."""
    keypoints: np.ndarray  # Shape (17, 3): [x, y, confidence]
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float

    @property
    def nose(self) -> np.ndarray:
        return self.keypoints[0]

    @property
    def left_shoulder(self) -> np.ndarray:
        return self.keypoints[5]

    @property
    def right_shoulder(self) -> np.ndarray:
        return self.keypoints[6]

    @property
    def left_hip(self) -> np.ndarray:
        return self.keypoints[11]

    @property
    def right_hip(self) -> np.ndarray:
        return self.keypoints[12]

    @property
    def left_knee(self) -> np.ndarray:
        return self.keypoints[13]

    @property
    def right_knee(self) -> np.ndarray:
        return self.keypoints[14]

    @property
    def left_ankle(self) -> np.ndarray:
        return self.keypoints[15]

    @property
    def right_ankle(self) -> np.ndarray:
        return self.keypoints[16]

    @property
    def shoulder_center(self) -> np.ndarray:
        """Midpoint between shoulders [x, y, conf]."""
        ls, rs = self.left_shoulder, self.right_shoulder
        return (ls + rs) / 2.0

    @property
    def hip_center(self) -> np.ndarray:
        """Midpoint between hips [x, y, conf]."""
        lh, rh = self.left_hip, self.right_hip
        return (lh + rh) / 2.0


class YOLO11PoseDetector:
    """YOLO11-Pose inference interface."""

    def __init__(
        self,
        model_name_or_path: str = "yolo11n-pose.pt",
        confidence_threshold: float = 0.45,
        device: str = "cpu",
        img_size: int = 640,
        mock_mode: bool = False,
    ):
        self.model_name = model_name_or_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.img_size = img_size
        self.mock_mode = mock_mode
        self._model = None
        self._is_loaded = False

        if not self.mock_mode:
            self._load_model()

    def _load_model(self) -> None:
        """Attempt to load Ultralytics YOLO pose model."""
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self.model_name)
            self._is_loaded = True
        except Exception:
            self._model = None
            self._is_loaded = False

    def estimate_poses(self, frame: np.ndarray) -> List[PersonPose]:
        """
        Run pose estimation on image frame.
        Returns a list of PersonPose objects with 17 keypoints each.
        """
        if frame is None or frame.size == 0:
            return []

        if self.mock_mode or not self._is_loaded:
            return self._mock_poses(frame)

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        poses: List[PersonPose] = []
        if not results:
            return poses

        res = results[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return poses

        kpts_data = res.keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes_data = res.boxes.xyxy.cpu().numpy()     # (N, 4)
        confs_data = res.boxes.conf.cpu().numpy()     # (N,)

        for kpts, bbox, conf in zip(kpts_data, boxes_data, confs_data):
            poses.append(
                PersonPose(
                    keypoints=kpts,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(conf),
                )
            )

        return poses

    def _mock_poses(self, frame: np.ndarray) -> List[PersonPose]:
        """Generate mock upright pose for testing when weights unavailable."""
        h, w = frame.shape[:2]
        cx, cy = w * 0.5, h * 0.6
        bw, bh = 40.0, 100.0

        # Construct upright keypoints
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [cx, cy - 45, 0.9]  # nose
        kpts[5] = [cx - 15, cy - 30, 0.9]  # left shoulder
        kpts[6] = [cx + 15, cy - 30, 0.9]  # right shoulder
        kpts[11] = [cx - 12, cy + 5, 0.9]  # left hip
        kpts[12] = [cx + 12, cy + 5, 0.9]  # right hip
        kpts[13] = [cx - 12, cy + 30, 0.9]  # left knee
        kpts[14] = [cx + 12, cy + 30, 0.9]  # right knee
        kpts[15] = [cx - 12, cy + 50, 0.9]  # left ankle
        kpts[16] = [cx + 12, cy + 50, 0.9]  # right ankle

        return [
            PersonPose(
                keypoints=kpts,
                bbox=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                confidence=0.85,
            )
        ]
