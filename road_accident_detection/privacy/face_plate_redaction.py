"""
Privacy Redactor for Video Streams.
Automatically obscures human faces and vehicle license plates before clip storage.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

from ..detection.classes import is_person, is_vehicle, RoadClass
from ..tracking.tracker import TrackedObject
from ..pose.pose_detector import PersonPose


@dataclass
class RedactionZone:
    x1: int
    y1: int
    x2: int
    y2: int
    category: str  # 'face' or 'license_plate'


class PrivacyRedactor:
    """Obscures faces and license plates in video frames."""

    def __init__(
        self,
        redact_faces: bool = True,
        redact_plates: bool = True,
        blur_kernel_size: int = 31,
        mode: str = "gaussian_blur",  # 'gaussian_blur', 'pixelate', 'blackout'
    ):
        self.redact_faces = redact_faces
        self.redact_plates = redact_plates
        # Kernel size must be positive odd integer
        self.kernel_size = blur_kernel_size if (blur_kernel_size % 2 == 1) else (blur_kernel_size + 1)
        self.mode = mode

    def redact_frame(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        poses: Optional[List[PersonPose]] = None,
    ) -> np.ndarray:
        """
        Apply privacy redaction in-place or return redacted copy.
        """
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()
        zones: List[RedactionZone] = []

        h, w = out.shape[:2]

        # 1. License Plate Zones (lower 25% of vehicle bounding boxes)
        if self.redact_plates:
            for track in tracks:
                if is_vehicle(track.class_name):
                    bx1, by1, bx2, by2 = track.bbox
                    bw = bx2 - bx1
                    bh = by2 - by1
                    # Bumper / Plate region: lower 25% height, center 70% width
                    plate_y1 = int(max(0, by2 - bh * 0.25))
                    plate_y2 = int(min(h, by2))
                    plate_x1 = int(max(0, bx1 + bw * 0.15))
                    plate_x2 = int(min(w, bx2 - bw * 0.15))
                    if plate_x2 > plate_x1 and plate_y2 > plate_y1:
                        zones.append(RedactionZone(plate_x1, plate_y1, plate_x2, plate_y2, "license_plate"))

        # 2. Face Zones (upper 25% of pedestrian bounding boxes or pose head keypoints)
        if self.redact_faces:
            if poses:
                for pose in poses:
                    # Head keypoints: nose(0), eyes(1,2), ears(3,4)
                    head_kpts = pose.keypoints[0:5]
                    valid_kpts = [pt for pt in head_kpts if pt[2] > 0.2]
                    if valid_kpts:
                        xs = [pt[0] for pt in valid_kpts]
                        ys = [pt[1] for pt in valid_kpts]
                        pad = 20
                        fx1 = int(max(0, min(xs) - pad))
                        fy1 = int(max(0, min(ys) - pad))
                        fx2 = int(min(w, max(xs) + pad))
                        fy2 = int(min(h, max(ys) + pad))
                        if fx2 > fx1 and fy2 > fy1:
                            zones.append(RedactionZone(fx1, fy1, fx2, fy2, "face"))
            else:
                for track in tracks:
                    if is_person(track.class_name):
                        bx1, by1, bx2, by2 = track.bbox
                        bh = by2 - by1
                        fx1 = int(max(0, bx1))
                        fy1 = int(max(0, by1))
                        fx2 = int(min(w, bx2))
                        fy2 = int(min(h, by1 + bh * 0.25))
                        if fx2 > fx1 and fy2 > fy1:
                            zones.append(RedactionZone(fx1, fy1, fx2, fy2, "face"))

        # 3. Apply redaction on zones
        for z in zones:
            sub = out[z.y1:z.y2, z.x1:z.x2]
            if sub.size == 0:
                continue

            if self.mode == "blackout":
                out[z.y1:z.y2, z.x1:z.x2] = 0
            elif self.mode == "pixelate":
                # Downsample then upsample
                zh, zw = sub.shape[:2]
                small_w = max(1, zw // 10)
                small_h = max(1, zh // 10)
                small = cv2.resize(sub, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                out[z.y1:z.y2, z.x1:z.x2] = cv2.resize(small, (zw, zh), interpolation=cv2.INTER_NEAREST)
            else:  # gaussian_blur default
                k = self.kernel_size
                # Ensure kernel is smaller than ROI dimensions
                zh, zw = sub.shape[:2]
                kx = min(k, zw if zw % 2 == 1 else zw - 1)
                ky = min(k, zh if zh % 2 == 1 else zh - 1)
                if kx >= 3 and ky >= 3:
                    out[z.y1:z.y2, z.x1:z.x2] = cv2.GaussianBlur(sub, (kx, ky), 0)

        return out
