"""
Temporal Debouncer State Machine.
Enforces N consecutive positive frames to eliminate single-frame false alarms,
and manages post-event cooldown periods.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DebounceStatus(Enum):
    MONITORING = "monitoring"
    DEBOUNCING = "debouncing"
    ACCIDENT_TRIGGERED = "accident_triggered"
    NEAR_MISS_TRIGGERED = "near_miss_triggered"
    COOLDOWN = "cooldown"


@dataclass
class DebounceOutput:
    status: DebounceStatus
    consecutive_positive_frames: int
    frames_required: int
    is_event_fired: bool
    is_near_miss_fired: bool
    current_fused_score: float


class TemporalDebouncer:
    """
    State machine requiring N consecutive positive frames before firing alert.
    Primary defense against sensor/tracker false positives.
    """

    def __init__(
        self,
        trigger_threshold: float = 0.68,
        near_miss_threshold: float = 0.45,
        consecutive_frames_required: int = 6,
        near_miss_min_frames: int = 4,
        cooldown_frames: int = 150,
    ):
        self.trigger_threshold = trigger_threshold
        self.near_miss_threshold = near_miss_threshold
        self.consecutive_frames_required = consecutive_frames_required
        self.near_miss_min_frames = near_miss_min_frames
        self.cooldown_frames = cooldown_frames

        self.consecutive_positive = 0
        self.consecutive_near_miss = 0
        self.cooldown_counter = 0
        self.status = DebounceStatus.MONITORING

    def update(self, fused_score: float) -> DebounceOutput:
        """
        Feed current frame's fused risk score into the state machine.
        """
        # Handle active cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.status = DebounceStatus.COOLDOWN
            return DebounceOutput(
                status=self.status,
                consecutive_positive_frames=0,
                frames_required=self.consecutive_frames_required,
                is_event_fired=False,
                is_near_miss_fired=False,
                current_fused_score=fused_score,
            )

        is_event_fired = False
        is_near_miss_fired = False

        # 1. Check Accident Trigger condition
        if fused_score >= self.trigger_threshold:
            self.consecutive_positive += 1
            self.consecutive_near_miss = 0

            if self.consecutive_positive >= self.consecutive_frames_required:
                self.status = DebounceStatus.ACCIDENT_TRIGGERED
                is_event_fired = True
                self.cooldown_counter = self.cooldown_frames
                self.consecutive_positive = 0
            else:
                self.status = DebounceStatus.DEBOUNCING

        # 2. Check Near-Miss condition
        elif fused_score >= self.near_miss_threshold:
            self.consecutive_near_miss += 1
            # If we were previously debouncing towards an accident and it dropped to near-miss
            if self.consecutive_positive >= self.near_miss_min_frames:
                self.status = DebounceStatus.NEAR_MISS_TRIGGERED
                is_near_miss_fired = True
                self.cooldown_counter = self.cooldown_frames // 2
                self.consecutive_positive = 0
                self.consecutive_near_miss = 0
            else:
                self.consecutive_positive = 0
                self.status = DebounceStatus.MONITORING
        else:
            # Score dropped below near-miss threshold
            if self.consecutive_positive >= self.near_miss_min_frames:
                # Evasion succeeded - classify as near miss
                self.status = DebounceStatus.NEAR_MISS_TRIGGERED
                is_near_miss_fired = True
                self.cooldown_counter = self.cooldown_frames // 2
            else:
                self.status = DebounceStatus.MONITORING

            self.consecutive_positive = 0
            self.consecutive_near_miss = 0

        return DebounceOutput(
            status=self.status,
            consecutive_positive_frames=self.consecutive_positive,
            frames_required=self.consecutive_frames_required,
            is_event_fired=is_event_fired,
            is_near_miss_fired=is_near_miss_fired,
            current_fused_score=fused_score,
        )

    def reset(self) -> None:
        """Reset state machine counters."""
        self.consecutive_positive = 0
        self.consecutive_near_miss = 0
        self.cooldown_counter = 0
        self.status = DebounceStatus.MONITORING
