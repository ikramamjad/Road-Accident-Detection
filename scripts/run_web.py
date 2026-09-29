"""
Real-Time Road Accident Detection System — Web Application Runner.
Launches the Flask web server and interactive landing page.
"""

import argparse
from pathlib import Path
import sys

# Ensure root and lib on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_LIB = _ROOT / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from road_accident_detection.web.app import app


import os

def main():
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", 5000))

    parser = argparse.ArgumentParser(description="Run RADS Web Landing Page & Inspection Server")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host address to bind (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in Flask debug mode")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    print("\n" + "=" * 76)
    print(" [RADS] Real-Time Road Accident Detection System -- Web Server Online")
    print("=" * 76)
    print(f" >> Access the Web Interface at: {url}")
    print(f" >> Binding on: {args.host}:{args.port} (Production & Cloud Deployment Ready)")
    print(f" >> Upload videos or test preloaded 1440p real traffic / collision clips")
    print("=" * 76 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
