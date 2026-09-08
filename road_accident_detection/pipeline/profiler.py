"""
Stage-by-Stage Latency Profiler.
Measures high-resolution execution time across all architectural stages:
YOLO11 -> BoT-SORT -> Pose -> Kinematics -> Causal GRU -> Fusion -> Logging.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Dict, Generator, List
import numpy as np


@dataclass
class StageProfile:
    stage_name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    percent_total: float


class PipelineProfiler:
    """Tracks latency metrics per pipeline component."""

    def __init__(self):
        self.stage_times: Dict[str, List[float]] = defaultdict(list)
        self.total_frame_times: List[float] = []
        self._frame_start: float = 0.0

    def start_frame(self) -> None:
        """Mark start of a full pipeline frame iteration."""
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        """Mark completion of current frame iteration."""
        duration_ms = (time.perf_counter() - self._frame_start) * 1000.0
        self.total_frame_times.append(duration_ms)

    @contextmanager
    def stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to measure latency of a single stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stage_times[stage_name].append(elapsed_ms)

    def get_fps(self) -> float:
        """Calculate overall pipeline throughput in Frames Per Second."""
        if not self.total_frame_times:
            return 0.0
        mean_ms = np.mean(self.total_frame_times[-100:])
        return 1000.0 / max(mean_ms, 1e-4)

    def summary(self) -> List[StageProfile]:
        """Compute statistical breakdown for each stage."""
        total_mean = np.mean(self.total_frame_times) if self.total_frame_times else 1.0
        profiles: List[StageProfile] = []

        for stage_name, times in self.stage_times.items():
            if not times:
                continue
            arr = np.array(times)
            mean_val = float(np.mean(arr))
            profiles.append(
                StageProfile(
                    stage_name=stage_name,
                    mean_ms=round(mean_val, 2),
                    std_ms=round(float(np.std(arr)), 2),
                    min_ms=round(float(np.min(arr)), 2),
                    max_ms=round(float(np.max(arr)), 2),
                    percent_total=round((mean_val / max(total_mean, 1e-4)) * 100.0, 1),
                )
            )

        return profiles

    def format_table(self) -> str:
        """Generate formatted latency budget report."""
        profiles = self.summary()
        fps = self.get_fps()
        total_mean = np.mean(self.total_frame_times) if self.total_frame_times else 0.0

        lines = [
            "+-----------------------+-----------+----------+----------+----------+---------+",
            "| Stage                 | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | % Total |",
            "+-----------------------+-----------+----------+----------+----------+---------+",
        ]
        for p in profiles:
            lines.append(
                f"| {p.stage_name:<21} | {p.mean_ms:>9.2f} | {p.std_ms:>8.2f} | {p.min_ms:>8.2f} | {p.max_ms:>8.2f} | {p.percent_total:>6.1f}% |"
            )
        lines.append("+-----------------------+-----------+----------+----------+----------+---------+")
        lines.append(f"| Total Pipeline Latency: {total_mean:>6.2f} ms/frame  |  Throughput: {fps:>6.1f} FPS              |")
        lines.append("+-------------------------------------------------------------------------+")
        return "\n".join(lines)
