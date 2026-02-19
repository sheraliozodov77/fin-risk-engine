#!/usr/bin/env python3
"""
Single-command pipeline: build_features_and_splits → train_model → verify_model → evaluate_model.
Run from project root with PYTHONPATH=. so all scripts resolve.

Usage:
  PYTHONPATH=. python scripts/run_pipeline.py
  PYTHONPATH=. python scripts/run_pipeline.py --skip-behavioral
  PYTHONPATH=. python scripts/run_pipeline.py --nrows 50000
  PYTHONPATH=. python scripts/run_pipeline.py --skip-evaluate   # stop after verify
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_with_pythonpath():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return env


def run_step(cmd: list[str], step: str) -> int:
    print("\n" + "=" * 60)
    print("Pipeline step: %s" % step)
    print(" ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, cwd=ROOT, env=_env_with_pythonpath())
    if result.returncode != 0:
        print("Pipeline failed at step: %s (exit %d)" % (step, result.returncode))
        sys.exit(result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline: build → train → verify → evaluate")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for build (default: all)")
    parser.add_argument("--skip-behavioral", action="store_true", help="Skip behavioral features in build")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluate_model (stop after verify)")
    parser.add_argument("--iterations", type=int, default=None, help="CatBoost iterations (default: from config)")
    args = parser.parse_args()

    # 1. Build features and splits
    build_cmd = [sys.executable, str(ROOT / "scripts" / "build_features_and_splits.py")]
    if args.nrows is not None:
        build_cmd += ["--nrows", str(args.nrows)]
    if args.skip_behavioral:
        build_cmd += ["--skip-behavioral"]
    run_step(build_cmd, "build_features_and_splits")

    # 2. Train model
    train_cmd = [sys.executable, str(ROOT / "scripts" / "train_model.py")]
    if args.iterations is not None:
        train_cmd += ["--iterations", str(args.iterations)]
    run_step(train_cmd, "train_model")

    # 3. Verify model
    run_step([sys.executable, str(ROOT / "scripts" / "verify_model.py")], "verify_model")

    # 4. Evaluate model (calibration, SHAP, reports)
    if not args.skip_evaluate:
        run_step([sys.executable, str(ROOT / "scripts" / "evaluate_model.py")], "evaluate_model")

    print("\n" + "=" * 60)
    print("Pipeline complete: build → train → verify" + (" → evaluate" if not args.skip_evaluate else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
