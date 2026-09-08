"""Detection module for YOLO11 multi-class object detection."""

from .classes import (
    RoadClass,
    UNIFIED_CLASSES,
    COCO_TO_UNIFIED,
    BDD100K_TO_UNIFIED,
    IDD_TO_UNIFIED,
    is_vehicle,
    is_vulnerable_road_user,
    is_person,
    is_two_wheeler,
)
from .yolo_detector import YOLO11Detector, DetectionResult

__all__ = [
    "RoadClass",
    "UNIFIED_CLASSES",
    "COCO_TO_UNIFIED",
    "BDD100K_TO_UNIFIED",
    "IDD_TO_UNIFIED",
    "is_vehicle",
    "is_vulnerable_road_user",
    "is_person",
    "is_two_wheeler",
    "YOLO11Detector",
    "DetectionResult",
]
