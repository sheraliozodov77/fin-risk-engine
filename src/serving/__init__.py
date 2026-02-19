# FastAPI scoring: score, level (HIGH/MED/LOW), top-k reason codes.
# Run: PYTHONPATH=. python scripts/run_serve.py
from src.serving.app import app, get_app

__all__ = ["app", "get_app"]
