"""
Retention Policy Manager.
Enforces GDPR / compliance data retention schedules for logged clips and telemetry.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RetentionSchedule:
    accident_days: int = 30
    near_miss_days: int = 1
    normal_frames_days: int = 0


class RetentionPolicyManager:
    """Manages clip lifecycle and auto-purges expired recordings."""

    def __init__(self, schedule: Optional[RetentionSchedule] = None):
        self.schedule = schedule or RetentionSchedule()

    def enforce_retention(self, storage_dir: str) -> Dict[str, int]:
        """
        Scan storage directory and delete files exceeding retention TTL.
        Returns count of purged files by category.
        """
        dir_path = Path(storage_dir)
        if not dir_path.exists():
            return {"purged_accidents": 0, "purged_near_misses": 0}

        now = datetime.now()
        purged = {"purged_accidents": 0, "purged_near_misses": 0}

        for file in dir_path.glob("**/*.*"):
            if not file.is_file():
                continue

            file_mtime = datetime.fromtimestamp(file.stat().st_mtime)
            age_days = (now - file_mtime).total_seconds() / 86400.0

            # Identify if near-miss or accident by filename or parent folder
            is_near_miss = "near_miss" in file.name.lower() or "near_miss" in str(file.parent).lower()

            ttl = self.schedule.near_miss_days if is_near_miss else self.schedule.accident_days
            if age_days > ttl:
                try:
                    file.unlink()
                    if is_near_miss:
                        purged["purged_near_misses"] += 1
                    else:
                        purged["purged_accidents"] += 1
                except Exception:
                    pass

        return purged
