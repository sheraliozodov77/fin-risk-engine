"""
Hyperparameter tuning with Optuna for CatBoost, XGBoost, and LightGBM.
Objective: maximize PR-AUC on validation set.

Performance: data preparation and encoding are done ONCE before the study,
so each trial only trains a model (no redundant copies or transforms).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger
from src.modeling.train import get_feature_columns, _fill_cat

logger = get_logger(__name__)


def _prepare_Xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    target_col: str = "is_fraud",
    has_label_col: str = "has_label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X, y from a split DataFrame (labeled rows only)."""
    if has_label_col in df.columns:
        df = df.loc[df[has_label_col]].copy()
    df = df.dropna(subset=[target_col])
    X = df[feature_cols].copy()
    for c in feature_cols:
        if c in cat_features:
            X[c] = _fill_cat(X[c])
        elif pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(-999)
    y = df[target_col].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def _catboost_objective_fast(
    trial,
    train_pool,
    eval_pool,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Optuna objective for CatBoost using pre-built pools."""
    import catboost as cb
    from sklearn.metrics import average_precision_score

    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
    }

    logger.info(
        "catboost_trial_start",
        trial=trial.number,
        iterations=params["iterations"],
        lr=round(params["learning_rate"], 4),
        depth=params["depth"],
        spw=round(params["scale_pos_weight"], 2),
    )

    model = cb.CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="PRAUC",
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        train_dir="outputs/catboost_info",
    )
    model.fit(train_pool, eval_set=eval_pool)

    best_iter = model.get_best_iteration() if hasattr(model, "get_best_iteration") else None
    logger.info("catboost_trial_trained", trial=trial.number, best_iteration=best_iter)

    proba = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, proba))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def _xgboost_objective_fast(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Optuna objective for XGBoost using pre-encoded data."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }

    logger.info(
        "xgboost_trial_start",
        trial=trial.number,
        n_estimators=params["n_estimators"],
        lr=round(params["learning_rate"], 4),
        depth=params["max_depth"],
        spw=round(params["scale_pos_weight"], 2),
    )

    model = xgb.XGBClassifier(
        **params,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        verbosity=0,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    best_iter = model.best_iteration if hasattr(model, "best_iteration") else None
    logger.info("xgboost_trial_trained", trial=trial.number, best_iteration=best_iter)

    proba = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, proba))


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def _lightgbm_objective_fast(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: list[str],
) -> float:
    """Optuna objective for LightGBM using pre-prepared data."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
    }

    model = lgb.LGBMClassifier(
        **params,
        metric="average_precision",
        verbosity=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=cat_features if cat_features else "auto",
    )

    proba = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, proba))


# ---------------------------------------------------------------------------
# Legacy objective wrappers (for backward compat if called directly)
# ---------------------------------------------------------------------------

def catboost_objective(trial, train_df, val_df, feature_cols, cat_features, target_col="is_fraud"):
    import catboost as cb
    X_train, y_train = _prepare_Xy(train_df, feature_cols, cat_features, target_col)
    X_val, y_val = _prepare_Xy(val_df, feature_cols, cat_features, target_col)
    train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
    eval_pool = cb.Pool(X_val, y_val, cat_features=cat_features)
    return _catboost_objective_fast(trial, train_pool, eval_pool, X_val, y_val)


def xgboost_objective(trial, train_df, val_df, feature_cols, cat_features, target_col="is_fraud"):
    from src.modeling.train_xgboost import _target_encode_fit, _target_encode_transform
    X_train, y_train = _prepare_Xy(train_df, feature_cols, cat_features, target_col)
    X_val, y_val = _prepare_Xy(val_df, feature_cols, cat_features, target_col)
    encoders = _target_encode_fit(X_train, y_train, cat_features)
    X_val = _target_encode_transform(X_val, cat_features, encoders)
    return _xgboost_objective_fast(trial, X_train, y_train, X_val, y_val)


def lightgbm_objective(trial, train_df, val_df, feature_cols, cat_features, target_col="is_fraud"):
    X_train, y_train = _prepare_Xy(train_df, feature_cols, cat_features, target_col)
    X_val, y_val = _prepare_Xy(val_df, feature_cols, cat_features, target_col)
    for c in cat_features:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
        if c in X_val.columns:
            X_val[c] = X_val[c].astype("category")
    return _lightgbm_objective_fast(trial, X_train, y_train, X_val, y_val, cat_features)


OBJECTIVES = {
    "catboost": catboost_objective,
    "xgboost": xgboost_objective,
    "lightgbm": lightgbm_objective,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _trial_callback(study, trial):
    """Log each completed trial for visibility."""
    logger.info(
        "trial_complete",
        trial=trial.number,
        pr_auc=round(trial.value, 6) if trial.value is not None else None,
        best_so_far=round(study.best_value, 6),
        params={k: round(v, 4) if isinstance(v, float) else v for k, v in trial.params.items()},
    )


def run_tuning(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    n_trials: int = 50,
    target_col: str = "is_fraud",
) -> dict[str, Any]:
    """
    Run Optuna study for the given model type.
    Pre-computes data once, then runs n_trials with only model training per trial.
    Returns dict with best_params, best_pr_auc, study.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if model_type not in OBJECTIVES:
        raise ValueError(f"model_type must be one of {list(OBJECTIVES.keys())}")

    # --- Pre-compute data (done ONCE, not per-trial) ---
    logger.info("preparing_data", model_type=model_type)
    t0 = time.time()
    X_train, y_train = _prepare_Xy(train_df, feature_cols, cat_features, target_col)
    X_val, y_val = _prepare_Xy(val_df, feature_cols, cat_features, target_col)

    if model_type == "xgboost":
        from src.modeling.train_xgboost import _target_encode_fit, _target_encode_transform
        encoders = _target_encode_fit(X_train, y_train, cat_features)
        X_val = _target_encode_transform(X_val, cat_features, encoders)

        def objective(trial):
            return _xgboost_objective_fast(trial, X_train, y_train, X_val, y_val)

    elif model_type == "catboost":
        import catboost as cb
        train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
        eval_pool = cb.Pool(X_val, y_val, cat_features=cat_features)

        def objective(trial):
            return _catboost_objective_fast(trial, train_pool, eval_pool, X_val, y_val)

    elif model_type == "lightgbm":
        for c in cat_features:
            if c in X_train.columns:
                X_train[c] = X_train[c].astype("category")
            if c in X_val.columns:
                X_val[c] = X_val[c].astype("category")

        def objective(trial):
            return _lightgbm_objective_fast(trial, X_train, y_train, X_val, y_val, cat_features)

    prep_time = time.time() - t0
    logger.info(
        "data_prepared",
        train_rows=len(X_train),
        val_rows=len(X_val),
        prep_seconds=round(prep_time, 1),
    )

    # --- Run study ---
    study = optuna.create_study(direction="maximize", study_name=f"{model_type}_tuning")
    study.optimize(objective, n_trials=n_trials, callbacks=[_trial_callback])

    logger.info(
        "tuning_complete",
        model_type=model_type,
        n_trials=n_trials,
        best_pr_auc=round(study.best_value, 6),
        best_params=study.best_params,
    )

    return {
        "best_params": study.best_params,
        "best_pr_auc": study.best_value,
        "study": study,
    }
