"""
RADS Web Application Entrypoint (Production & Cloud Deployment).
This root entrypoint is automatically discovered by cloud platforms
(Render, Railway, Heroku, Google Cloud Run, AWS, Koyeb, Azure).
"""

import os
import sys
from pathlib import Path

# Add project root and bundled lib to sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_LIB = _ROOT / "lib"
if _LIB.exists() and str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from road_accident_detection.web.app import app

# Standard WSGI aliases for gunicorn, uWSGI, mod_wsgi, waitress
application = app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")

    print("\n" + "=" * 76)
    print(" [RADS] Real-Time Road Accident Detection System -- Cloud Server Online")
    print("=" * 76)
    print(f" >> Binding on: {host}:{port} (Production Cloud Deployment Ready)")
    print(f" >> Environment PORT: {port}")
    print("=" * 76 + "\n")

    app.run(host=host, port=port, debug=debug)
