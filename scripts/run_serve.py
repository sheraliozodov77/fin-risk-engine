#!/usr/bin/env python3
"""
Run the Fraud Risk Scoring API (FastAPI + uvicorn).
Loads model + calibrator from config; exposes /health, /score, /score_batch.

Usage:
  PYTHONPATH=. python scripts/run_serve.py
  PYTHONPATH=. python scripts/run_serve.py --host 0.0.0.0 --port 8000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from src.serving.app import app


def main():
    parser = argparse.ArgumentParser(description="Run fraud risk scoring API")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload on code change")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
