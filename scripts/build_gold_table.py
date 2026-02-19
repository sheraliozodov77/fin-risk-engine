#!/usr/bin/env python3
"""
Build gold table and optionally save.
Usage:
  PYTHONPATH=. python scripts/build_gold_table.py
  PYTHONPATH=. python scripts/build_gold_table.py --nrows 100000 --out data/processed/gold_sample.parquet
"""
import argparse
from pathlib import Path

# Run from project root so src is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.gold_table import build_gold_table


def main():
    parser = argparse.ArgumentParser(description="Build gold training table")
    parser.add_argument("--nrows", type=int, default=None, help="Limit transactions (default: all)")
    parser.add_argument("--out", type=str, default=None, help="Output parquet path (default: from config)")
    parser.add_argument("--no-parse", action="store_true", help="Skip static parsing")
    args = parser.parse_args()
    gold = build_gold_table(
        nrows=args.nrows,
        parse_static=not args.no_parse,
        save_path=args.out,
    )
    print("Gold table shape:", gold.shape)
    print("has_label:", gold["has_label"].sum())
    if gold["has_label"].any():
        print("is_fraud (labeled):", gold.loc[gold["has_label"], "is_fraud"].sum())


if __name__ == "__main__":
    main()
