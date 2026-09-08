"""
Real-Time Multi-Modal Road Accident Detection System
End-to-end framework integrating YOLO11, BoT-SORT, YOLO11-Pose,
Kinematics, Causal GRU temporal modeling, and debounced multi-channel fusion.
"""

import sys
from pathlib import Path

# Ensure local lib/ with ultralytics is on sys.path
_lib_path = str(Path(__file__).resolve().parent.parent / "lib")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

__version__ = "1.0.0"
