"""
RADS Web Application Entrypoint (main.py).
Provides root entrypoint for cloud providers searching for main.py (e.g. Google Cloud App Engine, Cloud Run, Render, Koyeb).
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_LIB = _ROOT / "lib"
if _LIB.exists() and str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from road_accident_detection.web.app import app

application = app


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
