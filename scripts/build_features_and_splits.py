#!/usr/bin/env python3
"""
Build full pre-modeling pipeline: gold table -> static (Phase 6) -> behavioral (Phase 7)
-> risk aggregates (Phase 8) -> save gold (Phase 9) -> time-based splits (Phase 10).

Output directories:
  - With --skip-behavioral: saves to data/processed/ (default, fast)
  - WITHOUT --skip-behavioral: saves to data/processed/behavioral/ (safe -- won't overwrite
    existing splits if the run fails or OOMs). On success, prints instructions to promote.

Usage:
  PYTHONPATH=. python scripts/build_features_and_splits.py   # full data with behavioral -> data/processed/behavioral/
  PYTHONPATH=. python scripts/build_features_and_splits.py --behavioral-batch 300   # smaller batches if timeout
  PYTHONPATH=. python scripts/build_features_and_splits.py --skip-behavioral   # fast: gold + risk + splits -> data/processed/
  PYTHONPATH=. python scripts/build_features_and_splits.py --nrows 50000   # sample for quick test (first N rows -- may lack val/test)
  PYTHONPATH=. python scripts/build_features_and_splits.py --sample-cards 250   # sample 250 cards, full history -> proper splits
  PYTHONPATH=. python scripts/build_features_and_splits.py --promote   # move behavioral/ -> processed/ after successful run
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_paths, get_model_config
from src.ingestion.gold_table import build_gold_table
from src.features.behavioral import add_behavioral_features
from src.features.risk_aggregates import add_risk_aggregates
from src.features.splits import (
    get_split_boundaries,
    time_based_splits,
    split_fraud_rates,
    save_splits,
)
from src.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def _resolve_out_dir(args) -> Path:
    """Resolve output directory. Behavioral runs go to a safe subfolder."""
    paths = get_paths()
    processed = paths.get("data", {}).get("processed", {})
    base_dir = args.out_dir or processed.get("dir", "data/processed")
    base = Path(base_dir)
    if not base.is_absolute():
        root = Path(__file__).resolve().parents[1]
        base = root / base_dir

    if not args.skip_behavioral and not args.out_dir and not args.promote:
        # Behavioral run: use safe subfolder
        return base / "behavioral"
    return base


def _promote(args):
    """Promote behavioral/ outputs to the main processed/ directory."""
    paths = get_paths()
    processed = paths.get("data", {}).get("processed", {})
    base_dir = processed.get("dir", "data/processed")
    root = Path(__file__).resolve().parents[1]
    base = root / base_dir
    behavioral_dir = base / "behavioral"

    if not behavioral_dir.exists():
        print(f"ERROR: {behavioral_dir} does not exist. Run without --skip-behavioral first.")
        sys.exit(1)

    files = ["gold_transactions.parquet", "train.parquet", "val.parquet", "test.parquet"]
    promoted = 0
    for f in files:
        src = behavioral_dir / f
        dst = base / f
        if src.exists():
            # Backup existing file
            if dst.exists():
                backup = base / f"{f}.backup"
                shutil.copy2(dst, backup)
                logger.info("backed_up", src=str(dst), backup=str(backup))
            shutil.copy2(src, dst)
            promoted += 1
            logger.info("promoted", src=str(src), dst=str(dst))

    print(f"\nPromoted {promoted} files from behavioral/ -> processed/")
    print("Backups saved as *.backup in data/processed/")
    print("You can now retune and retrain on the behavioral feature set.")


def main():
    parser = argparse.ArgumentParser(description="Build features and time-based splits")
    parser.add_argument("--nrows", type=int, default=None, help="Limit transactions (first N rows -- may lack val/test data)")
    parser.add_argument("--sample-cards", type=int, default=None, help="Sample N random cards with full history (preserves time range for proper splits)")
    parser.add_argument("--skip-behavioral", action="store_true", help="Skip behavioral features (fast)")
    parser.add_argument("--behavioral-batch", type=int, default=None, help="Process behavioral in batches of N entities (default: 400 for full data)")
    parser.add_argument("--skip-risk", action="store_true", help="Skip risk aggregate features")
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir (overrides auto-detection)")
    parser.add_argument("--promote", action="store_true", help="Promote behavioral/ outputs to main processed/ dir")
    args = parser.parse_args()

    setup_logging()

    # Handle --promote separately
    if args.promote:
        _promote(args)
        return

    out_dir = _resolve_out_dir(args)
    gold_path = out_dir / "gold_transactions.parquet"

    if not args.skip_behavioral:
        print(f"\n*** BEHAVIORAL MODE: outputs will be saved to {out_dir} ***")
        print(f"*** Your existing data/processed/ splits are SAFE and unchanged ***")
        print(f"*** After success, run with --promote to move to data/processed/ ***\n")

    t0 = time.time()
    print("Phase 6 + gold: building gold table with static preprocessing...")
    gold = build_gold_table(nrows=args.nrows, parse_static=True, save_path=None)
    print("  shape: %s  (%.1fs)" % (gold.shape, time.time() - t0))
    logger.info("gold_built", rows=gold.shape[0], cols=gold.shape[1])

    # Entity-based sampling: pick N random cards, keep full history (all years)
    if args.sample_cards is not None:
        import numpy as np
        all_cards = gold["card_id"].unique()
        n_sample = min(args.sample_cards, len(all_cards))
        rng = np.random.RandomState(42)
        sampled_cards = rng.choice(all_cards, size=n_sample, replace=False)
        gold = gold[gold["card_id"].isin(sampled_cards)].reset_index(drop=True)
        date_range = "%s to %s" % (gold["date"].min(), gold["date"].max())
        print("  sampled %d cards -> %d rows (date range: %s)" % (n_sample, len(gold), date_range))
        logger.info("sampled_cards", n_cards=n_sample, rows=len(gold), date_range=date_range)

    if not args.skip_behavioral:
        batch = args.behavioral_batch
        if batch is None and args.sample_cards is None and args.nrows is None:
            batch = 400  # default batching for full data to avoid timeout
        t1 = time.time()
        n_cards = gold["card_id"].nunique() if "card_id" in gold.columns else 0
        n_users = gold["client_id"].nunique() if "client_id" in gold.columns else 0
        if batch is not None:
            print("Phase 7: adding behavioral features (past-only windows, batch=%d entities)..." % batch)
        else:
            print("Phase 7: adding behavioral features (past-only windows)...")
        print("  entities: %d cards, %d users, %d rows" % (n_cards, n_users, len(gold)))
        gold = add_behavioral_features(gold, entity_batch_size=batch)
        print("  shape: %s  (%.1fs)" % (gold.shape, time.time() - t1))
        logger.info("behavioral_done", rows=gold.shape[0], cols=gold.shape[1])
    else:
        print("Phase 7: skipped (--skip-behavioral)")

    if not args.skip_risk:
        t2 = time.time()
        print("Phase 8: adding leakage-safe risk aggregates...")
        gold = add_risk_aggregates(gold)
        print("  shape: %s  (%.1fs)" % (gold.shape, time.time() - t2))
        logger.info("risk_aggregates_done", rows=gold.shape[0], cols=gold.shape[1])
    else:
        print("Phase 8: skipped (--skip-risk)")

    # Phase 9: final table in date order and save
    gold = gold.sort_values("date").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(gold_path, index=False)
    print("Phase 9: saved %s  (total elapsed: %.1fs)" % (gold_path, time.time() - t0))
    logger.info("gold_saved", path=str(gold_path))

    # Phase 10: time-based splits
    train_end, val_end, test_start = get_split_boundaries(get_model_config())
    date_min = gold["date"].min()
    date_max = gold["date"].max()
    print("Phase 10: splitting by date (train_end=%s, val_end=%s, test_start=%s)" % (train_end, val_end, test_start))
    print("  data date range: %s to %s" % (date_min, date_max))
    train, val, test = time_based_splits(gold, train_end=train_end, val_end=val_end, test_start=test_start)
    rates = split_fraud_rates(train, val, test)
    for name, s in rates.items():
        print("  %s: rows=%d labeled=%d fraud=%d fraud_rate=%.4f" % (
            name, s["rows"], s["labeled_rows"], s["fraud_count"], s["fraud_rate"]))
    if len(val) == 0 or len(test) == 0:
        print("  WARNING: val or test is empty. If your data is only from early years (e.g. 2010),"
              " all rows fall in train (<=2017). Run without --nrows on full data to get 2018+ for val/test.")
    p_train, p_val, p_test = save_splits(train, val, test, str(out_dir))
    print("  saved:", p_train, p_val, p_test)

    if not args.skip_behavioral:
        print(f"\n{'='*60}")
        print(f"  SUCCESS: Behavioral features built!")
        print(f"  Output: {out_dir}")
        print(f"  Your existing data/processed/ splits are UNCHANGED.")
        print(f"")
        print(f"  To use these for training, run:")
        print(f"    PYTHONPATH=. python scripts/build_features_and_splits.py --promote")
        print(f"  Then retune and retrain:")
        print(f"    PYTHONPATH=. python scripts/run_tuning.py --model-type catboost --n-trials 20")
        print(f"    PYTHONPATH=. python scripts/train_model.py --model-type catboost --use-tuned")
        print(f"{'='*60}")
    else:
        print("Done. Ready for modeling.")


if __name__ == "__main__":
    main()
