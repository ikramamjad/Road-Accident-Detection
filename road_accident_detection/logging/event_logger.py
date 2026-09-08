"""
Structured Event Logger.
Persists immutable JSON incident records with geolocation, timestamps,
attributions, and clip references.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from .explainability import ExplainabilityReport
from ..fusion.multi_channel_fusion import FusedRiskResult
from ..fusion.severity_classifier import SeverityAssessment
from ..tracking.tracker import TrackedObject


@dataclass
class AccidentEventRecord:
    event_id: str
    timestamp_utc: str
    camera_id: str
    location: Dict[str, Any]
    severity: str
    severity_justification: str
    involved_tracks: List[Dict[str, Any]]
    channel_scores: Dict[str, float]
    fused_score: float
    calibrated_probability: float
    explainability: Dict[str, Any]
    evidence_clip_path: Optional[str] = None


class EventLogger:
    """Manages creation and disk persistence of structured incident event records."""

    def __init__(
        self,
        output_dir: str = "d:/Road Accident Detection/data/recordings/events",
        camera_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_meta = camera_metadata or {
            "camera_id": "DEFAULT_CAM_01",
            "latitude": 0.0,
            "longitude": 0.0,
            "intersection_name": "Unspecified Roadway",
        }

    def log_event(
        self,
        severity_assessment: SeverityAssessment,
        fusion_result: FusedRiskResult,
        explainability: ExplainabilityReport,
        involved_tracks: List[TrackedObject],
        clip_path: Optional[str] = None,
    ) -> AccidentEventRecord:
        """Create and write structured JSON incident record."""
        event_id = str(uuid.uuid4())
        now_utc = datetime.now(timezone.utc).isoformat()

        track_entries = [
            {
                "track_id": t.track_id,
                "class_name": t.class_name.value if hasattr(t.class_name, "value") else str(t.class_name),
                "confidence": round(t.confidence, 3),
                "bbox": [round(coord, 1) for coord in t.bbox],
            }
            for t in involved_tracks
        ]

        record = AccidentEventRecord(
            event_id=event_id,
            timestamp_utc=now_utc,
            camera_id=self.camera_meta.get("camera_id", "UNKNOWN"),
            location={
                "latitude": self.camera_meta.get("latitude", 0.0),
                "longitude": self.camera_meta.get("longitude", 0.0),
                "intersection_name": self.camera_meta.get("intersection_name", "Unknown Intersection"),
            },
            severity=severity_assessment.severity.value,
            severity_justification=severity_assessment.justification,
            involved_tracks=track_entries,
            channel_scores=fusion_result.channel_attributions,
            fused_score=round(fusion_result.fused_score, 4),
            calibrated_probability=round(fusion_result.calibrated_probability, 4),
            explainability={
                "primary_driver": explainability.primary_channel,
                "secondary_driver": explainability.secondary_channel,
                "channel_percentages": explainability.channel_percentages,
                "narrative": explainability.narrative_summary,
                "alert_tags": explainability.alert_tags,
            },
            evidence_clip_path=clip_path,
        )

        # Write to JSON
        filename = f"event_{severity_assessment.severity.value}_{event_id[:8]}.json"
        target_path = self.output_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

        return record
