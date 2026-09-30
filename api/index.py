"""
Vercel Serverless Function entrypoint for RADS (Road Accident Detection System).
Exposes the Flask WSGI application to Vercel's Serverless runtime.
"""

import os
import sys
from pathlib import Path

# Mark serverless environment
os.environ["RADS_SERVERLESS"] = "1"
os.environ["VERCEL"] = "1"

# Ensure repository root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from road_accident_detection.web.app import app

# Vercel WSGI entrypoint callable
application = app
