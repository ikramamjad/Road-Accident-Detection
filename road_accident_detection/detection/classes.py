"""
Unified class taxonomy and cross-dataset mappings for Road Accident Detection.
Supports IDD (India Driving Dataset), BDD100K, and standard COCO datasets.
"""

from enum import Enum
from typing import Dict, List, Optional


class RoadClass(str, Enum):
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    AUTO_RICKSHAW = "auto_rickshaw"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    UNKNOWN = "unknown"


UNIFIED_CLASSES: List[str] = [
    RoadClass.CAR.value,
    RoadClass.BUS.value,
    RoadClass.TRUCK.value,
    RoadClass.MOTORCYCLE.value,
    RoadClass.AUTO_RICKSHAW.value,
    RoadClass.PEDESTRIAN.value,
    RoadClass.CYCLIST.value,
]

# COCO dataset index mapping (Ultralytics pretrained default)
COCO_TO_UNIFIED: Dict[int, RoadClass] = {
    0: RoadClass.PEDESTRIAN,     # person
    1: RoadClass.CYCLIST,        # bicycle
    2: RoadClass.CAR,            # car
    3: RoadClass.MOTORCYCLE,     # motorcycle
    5: RoadClass.BUS,            # bus
    7: RoadClass.TRUCK,          # truck
}

# BDD100K class label mapping
BDD100K_TO_UNIFIED: Dict[str, RoadClass] = {
    "pedestrian": RoadClass.PEDESTRIAN,
    "rider": RoadClass.CYCLIST,
    "car": RoadClass.CAR,
    "truck": RoadClass.TRUCK,
    "bus": RoadClass.BUS,
    "motorcycle": RoadClass.MOTORCYCLE,
    "bicycle": RoadClass.CYCLIST,
}

# IDD (India Driving Dataset) label mapping
IDD_TO_UNIFIED: Dict[str, RoadClass] = {
    "car": RoadClass.CAR,
    "bus": RoadClass.BUS,
    "truck": RoadClass.TRUCK,
    "motorcycle": RoadClass.MOTORCYCLE,
    "autorickshaw": RoadClass.AUTO_RICKSHAW,
    "auto_rickshaw": RoadClass.AUTO_RICKSHAW,
    "person": RoadClass.PEDESTRIAN,
    "pedestrian": RoadClass.PEDESTRIAN,
    "rider": RoadClass.CYCLIST,
    "bicycle": RoadClass.CYCLIST,
}

VEHICLE_CLASSES = {
    RoadClass.CAR,
    RoadClass.BUS,
    RoadClass.TRUCK,
    RoadClass.MOTORCYCLE,
    RoadClass.AUTO_RICKSHAW,
}

VULNERABLE_ROAD_USER_CLASSES = {
    RoadClass.PEDESTRIAN,
    RoadClass.CYCLIST,
}

TWO_WHEELER_CLASSES = {
    RoadClass.MOTORCYCLE,
    RoadClass.CYCLIST,
}


def is_vehicle(cls: RoadClass) -> bool:
    """Return True if class represents a motorized or non-motorized vehicle."""
    return cls in VEHICLE_CLASSES


def is_vulnerable_road_user(cls: RoadClass) -> bool:
    """Return True if class represents a pedestrian, cyclist, or rider."""
    return cls in VULNERABLE_ROAD_USER_CLASSES


def is_person(cls: RoadClass) -> bool:
    """Return True if class represents a pedestrian or human."""
    return cls == RoadClass.PEDESTRIAN


def is_two_wheeler(cls: RoadClass) -> bool:
    """Return True if class represents a motorcycle or bicycle."""
    return cls in TWO_WHEELER_CLASSES
