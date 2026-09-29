"""
Unit and Integration tests for Web Landing Page and API endpoints.
"""

import io
import json
from pathlib import Path
import unittest

from road_accident_detection.web.app import app


class TestWebAPI(unittest.TestCase):
    """Test suite verifying Flask routes, API endpoints, and analysis pipelines."""

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_index_page_returns_html(self):
        """Verify the landing page loads successfully with HTML content."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("RADS", html)
        self.assertIn("Road Accident Detection", html)
        self.assertIn("Video Upload", html)

    def test_api_status_endpoint(self):
        """Verify health check and model registry status."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "online")
        self.assertIn("models", data)
        self.assertIn("object_detector", data["models"])
        self.assertIn("temporal_model", data["models"])

    def test_api_sample_clips_listing(self):
        """Verify sample clips listing returns available test clips."""
        response = self.client.get("/api/sample-clips")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("sample_clips", data)
        clip_ids = [c["id"] for c in data["sample_clips"]]
        self.assertIn("synthetic_collision", clip_ids)

    def test_api_upload_rejects_invalid_extension(self):
        """Verify uploading an invalid file extension returns HTTP 400."""
        data = {
            "video": (io.BytesIO(b"dummy fake content"), "test_script.py")
        }
        response = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        res = json.loads(response.data)
        self.assertIn("error", res)

    def test_api_analyze_synthetic_mock_mode(self):
        """Verify running analysis on synthetic stream returns structured diagnosis."""
        payload = {
            "source_type": "synthetic",
            "max_frames": 30,
            "conf_threshold": 0.40,
            "redact_privacy": True,
            "mock_mode": True
        }
        response = self.client.post("/api/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn("verdict", data)
        self.assertIn(data["verdict"], ["ACCIDENT_DETECTED", "NEAR_MISS", "NO_ACCIDENT"])
        self.assertIn("severity", data)
        self.assertIn("channel_contributions", data)
        self.assertIn("max_risk_score", data)
        self.assertIn("video_url", data)
        self.assertIn("keyframe_url", data)
        self.assertIn("keyframe_snapshots", data)
        self.assertIn("fault_attribution", data)
        self.assertIsInstance(data["fault_attribution"], dict)
        self.assertEqual(data["frames_processed"], 30)

    def test_health_check_endpoints(self):
        """Verify /healthz and /api/health return 200 with status healthy."""
        for path in ["/healthz", "/api/health"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            data = json.loads(res.data)
            self.assertEqual(data["status"], "healthy")

    def test_favicon_endpoint(self):
        """Verify /favicon.ico returns 204 No Content without error."""
        res = self.client.get("/favicon.ico")
        self.assertEqual(res.status_code, 204)

    def test_graceful_404_handler(self):
        """Verify unknown page route serves index.html, while /api/* returns JSON 404."""
        # Page route
        res_page = self.client.get("/non-existent-page")
        self.assertEqual(res_page.status_code, 200)
        self.assertIn(b"RADS", res_page.data)

        # API route
        res_api = self.client.get("/api/non-existent-endpoint")
        self.assertEqual(res_api.status_code, 404)
        data = json.loads(res_api.data)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
