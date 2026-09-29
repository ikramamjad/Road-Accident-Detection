"""
WSGI entrypoint for Gunicorn, uWSGI, Waitress, and cloud reverse proxies.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_LIB = _ROOT / "lib"
if _LIB.exists() and str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from road_accident_detection.web.app import app

# Standard WSGI targets
application = app
