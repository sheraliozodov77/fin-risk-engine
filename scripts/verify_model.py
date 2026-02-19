#!/usr/bin/env python3
"""
Verify saved model loads and predicts on one row (sanity check).
Usage: PYTHONPATH=. python scripts/verify_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import get_paths
from src.modeling.train import get_feature_columns


def main():
    paths = get_paths()
    root = Path(__file__).resolve().parents[1]
    model_dir = paths.get("outputs", {}).get("models", "outputs/models")
    val_path = paths.get("data", {}).get("processed", {}).get("val", "data/processed/val.parquet")
    model_path = root / model_dir / "catboost_fraud.cbm"
    val_path = root / val_path if not Path(val_path).is_absolute() else Path(val_path)

    if not model_path.exists():
        print("Model not found:", model_path)
        sys.exit(1)
    if not val_path.exists():
        print("Val not found:", val_path)
        sys.exit(1)

    import catboost as cb
    model = cb.CatBoostClassifier()
    model.load_model(str(model_path))
    print("Loaded model:", model_path)

    val = pd.read_parquet(val_path)
    feature_cols, cat_cols = get_feature_columns(val, target_col="is_fraud")
    row = val.head(1)
    X = row.reindex(columns=feature_cols).copy()
    for c in feature_cols:
        if c in cat_cols and c in X.columns:
            X[c] = X[c].fillna("__missing__").astype(str).replace("nan", "__missing__")
        elif c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    print("One-row prediction: proba=%.4f pred=%s" % (float(proba[0]), int(pred[0])))
    print("OK: model loads and predicts.")


if __name__ == "__main__":
    main()
