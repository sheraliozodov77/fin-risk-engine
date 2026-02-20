"""
FastAPI scoring service: load model + calibrator; accept feature vector(s);
return calibrated risk score, level (HIGH/MED/LOW), and top-k reason codes.
Gradio UI at /gradio for interactive scoring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.logging_config import get_logger

logger = get_logger(__name__)


# Pydantic models at module level so OpenAPI schema can resolve them (avoids 500 on /openapi.json)
class ScoreRequest(BaseModel):
    features: dict[str, Any]


class ScoreBatchRequest(BaseModel):
    features_list: list[dict[str, Any]]

# Lazy imports for FastAPI, CatBoost, joblib to avoid startup cost if not serving
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_models_dir() -> Path:
    from src.config import get_paths
    paths = get_paths()
    models_dir = paths.get("outputs", {}).get("models", "outputs/models")
    base = Path(models_dir) if Path(models_dir).is_absolute() else _project_root() / models_dir
    return base


def load_artifacts() -> tuple[Any, Any, list[str], list[str], float, float, int]:
    """
    Load champion model, calibrator, feature_cols, cat_cols, T_high, T_med, top_k.

    Model loading order:
    1. MLflow registry @champion alias (source of truth for champion model type)
    2. File fallback (outputs/models/) if registry unavailable or alias not set

    Calibrator and feature metadata always loaded from files (not in MLflow yet — WS4).
    Returns (model, calibrator, feature_cols, cat_cols, t_high, t_med, top_k).
    """
    import joblib
    root = _project_root()
    models_dir = _get_models_dir()
    from src.config import get_paths, get_model_config
    paths = get_paths()
    cfg = get_model_config()
    serving = cfg.get("serving", {})
    processed = paths.get("data", {}).get("processed", {})
    mlflow_cfg = cfg.get("mlflow", {})
    top_k = int(serving.get("top_k_reason_codes", 5))
    t_high = float(cfg.get("thresholds", {}).get("high", 0.7))
    t_med = float(cfg.get("thresholds", {}).get("medium", 0.3))

    # Step 1: determine champion model type from MLflow @champion alias
    champion_model_type = serving.get("champion_model", "catboost")  # config default
    model = None

    try:
        from src.tracking.mlflow_tracker import MLflowTracker
        registry_name = mlflow_cfg.get("registry_model_name", "fraud-detection")
        tracker = MLflowTracker(
            experiment_name=mlflow_cfg.get("experiment_name", "fin-risk-engine"),
            tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
        )
        champion_info = tracker.get_champion_info(registry_name)
        if champion_info and champion_info.get("model_type"):
            champion_model_type = champion_info["model_type"]
            logger.info("champion_from_registry", model_type=champion_model_type,
                        version=champion_info.get("version"))
            # Try loading model directly from registry
            try:
                model_uri = f"models:/{registry_name}@champion"
                if champion_model_type == "catboost":
                    import mlflow.catboost as mlflow_cb
                    model = mlflow_cb.load_model(model_uri)
                elif champion_model_type == "xgboost":
                    import mlflow.xgboost as mlflow_xgb
                    model = mlflow_xgb.load_model(model_uri)
                if model is not None:
                    logger.info("model_loaded_from_registry", uri=model_uri, model_type=champion_model_type)
            except Exception as reg_exc:
                logger.warning("registry_model_load_failed", error=str(reg_exc), fallback="file")
                model = None
    except Exception as mlflow_exc:
        logger.warning("mlflow_champion_lookup_failed", error=str(mlflow_exc), fallback="file")

    # Step 2: file fallback
    if model is None:
        if champion_model_type == "xgboost":
            import xgboost as xgb
            model_path = models_dir / "xgboost_fraud.json"
            if not model_path.exists():
                raise FileNotFoundError("Model not found: %s" % model_path)
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
        else:
            try:
                from catboost import CatBoostClassifier
            except ImportError:
                raise RuntimeError("catboost not installed")
            model_path = models_dir / serving.get("model_file", "catboost_fraud.cbm")
            if not model_path.exists():
                raise FileNotFoundError("Model not found: %s" % model_path)
            model = CatBoostClassifier()
            model.load_model(str(model_path))
        logger.info("model_loaded_from_file", model_type=champion_model_type)

    # Step 3: calibrator + feature metadata always from files (calibrator not in registry yet)
    calibrator_path = models_dir / f"calibrator_{champion_model_type}.joblib"
    calibrator = None
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)

    feature_cols = []
    cat_cols = []
    meta_path = models_dir / f"feature_metadata_{champion_model_type}.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta.get("feature_cols", [])
        cat_cols = meta.get("cat_cols", [])
    if not feature_cols:
        train_path = processed.get("train", "data/processed/train.parquet")
        train_path = Path(train_path) if Path(train_path).is_absolute() else root / train_path
        if train_path.exists():
            from src.modeling.train import get_feature_columns
            train_df = pd.read_parquet(train_path)
            feature_cols, cat_cols = get_feature_columns(train_df, target_col="is_fraud")
    if not feature_cols:
        raise ValueError("No feature metadata and no train.parquet to infer features")

    return model, calibrator, feature_cols, cat_cols, t_high, t_med, top_k


def _features_dict_to_row(features: dict[str, Any], feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Build one row DataFrame in feature_cols order; fill missing/numeric/cat."""
    row = {}
    for c in feature_cols:
        v = features.get(c)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            row[c] = -999 if c not in cat_cols else "__missing__"
        else:
            row[c] = v
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_cols)
    for c in feature_cols:
        if c in cat_cols:
            df[c] = df[c].fillna("__missing__").astype(str).replace("nan", "__missing__")
        elif pd.api.types.is_numeric_dtype(df[c]) or c not in cat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-999)
    return df


def _score_one(
    model: Any,
    calibrator: Any,
    features: dict[str, Any],
    feature_cols: list[str],
    cat_cols: list[str],
    t_high: float,
    t_med: float,
    top_k: int,
) -> dict[str, Any]:
    from src.modeling.calibrate import apply_calibrator
    from src.modeling.explain import get_local_reason_codes_batch

    X = _features_dict_to_row(features, feature_cols, cat_cols)
    proba_raw = model.predict_proba(X)[0, 1]
    import numpy as np
    proba = float(apply_calibrator(np.array([proba_raw]), calibrator)[0]) if calibrator else float(proba_raw)
    if proba >= t_high:
        level = "HIGH"
    elif proba >= t_med:
        level = "MEDIUM"
    else:
        level = "LOW"
    reason_codes = []
    try:
        codes = get_local_reason_codes_batch(model, X, feature_cols, cat_cols=cat_cols, top_k=top_k, max_rows=1)
        for name, v in codes.get(0, []):
            val = X[name].iloc[0] if name in X.columns else None
            if isinstance(val, (int, float)) and (val == -999 or (isinstance(val, float) and pd.isna(val))):
                val_str = "(missing)"
            else:
                val_str = str(val)[:40] + "..." if isinstance(val, str) and len(str(val)) > 40 else str(val)
            reason_codes.append({"feature": name, "impact": round(v, 6), "value": val_str})
    except Exception as e:
        logger.warning("reason_codes_computation_failed", error=str(e))
    return {"risk_score": round(proba, 6), "level": level, "reason_codes": reason_codes}


# Global state set at startup
_model = None
_calibrator = None
_feature_cols = []
_cat_cols = []
_t_high = 0.7
_t_med = 0.3
_top_k = 5


def get_app():
    """Build FastAPI app with lazy dependency injection; mount Gradio at /gradio."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Fraud Risk Scoring API", version="1.0")

    @app.on_event("startup")
    def startup():
        global _model, _calibrator, _feature_cols, _cat_cols, _t_high, _t_med, _top_k
        try:
            _model, _calibrator, _feature_cols, _cat_cols, _t_high, _t_med, _top_k = load_artifacts()
        except Exception as e:
            raise RuntimeError("Failed to load model/calibrator: %s" % e)

    @app.get("/")
    def root():
        """Root: links to docs and health. Use /health, /score, /score_batch for the API."""
        return {
            "service": "Fraud Risk Scoring API",
            "docs": "/docs",
            "health": "/health",
            "model_info": "GET /model_info",
            "sample_from_test": "GET /sample_from_test",
            "gradio": "/gradio",
            "score": "POST /score",
            "score_batch": "POST /score_batch",
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": _model is not None}

    @app.get("/model_info")
    def model_info():
        """Return loaded model feature/category counts so clients know the model is fully loaded."""
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "n_features": len(_feature_cols),
            "n_categorical": len(_cat_cols),
            "feature_names": _feature_cols,
            "thresholds": {"high": _t_high, "medium": _t_med},
        }

    def _get_one_test_sample() -> str:
        """Return one row from test.parquet (or val if test empty) as JSON. For Gradio / sample_from_test."""
        from src.config import get_paths
        root = _project_root()
        paths = get_paths()
        processed = paths.get("data", {}).get("processed", {})
        for name, rel in [("test", processed.get("test", "data/processed/test.parquet")),
                          ("val", processed.get("val", "data/processed/val.parquet"))]:
            data_path = Path(rel) if Path(rel).is_absolute() else root / rel
            if not data_path.exists():
                continue
            df = pd.read_parquet(data_path)
            if len(df) == 0:
                continue
            cols = [c for c in _feature_cols if c in df.columns]
            if not cols:
                continue
            row = df[cols].sample(n=1, random_state=None).iloc[0]
            return json.dumps(row.to_dict(), indent=2)
        return ""

    @app.get("/sample_from_test")
    def sample_from_test():
        """Return one row from test.parquet (features only) for use in UI or scripts."""
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        out = _get_one_test_sample()
        if not out:
            raise HTTPException(status_code=404, detail="Test data not found or no matching features")
        return json.loads(out)

    @app.post("/score")
    def score(req: ScoreRequest):
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            return _score_one(
                _model, _calibrator, req.features,
                _feature_cols, _cat_cols, _t_high, _t_med, _top_k,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/score_batch")
    def score_batch(req: ScoreBatchRequest):
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        out = []
        for features in req.features_list:
            try:
                out.append(_score_one(
                    _model, _calibrator, features,
                    _feature_cols, _cat_cols, _t_high, _t_med, _top_k,
                ))
            except Exception as e:
                out.append({"error": str(e), "risk_score": None, "level": None, "reason_codes": []})
        return {"scores": out}

    # Mount Gradio UI at /gradio
    try:
        import gradio as gr
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        def _make_score_charts(risk_score: float, level: str, reason_codes: list[dict]) -> plt.Figure | None:
            """Build a figure: risk gauge + SHAP horizontal bar chart. For interview/demo."""
            fig, axes = plt.subplots(2, 1, figsize=(7, 4), height_ratios=[1, 1.8])
            fig.set_facecolor("#fafafa")
            # 1) Risk score gauge (horizontal bar 0–100% with zones)
            ax0 = axes[0]
            ax0.set_xlim(0, 1)
            ax0.set_ylim(0, 1)
            ax0.axis("off")
            # Background zones
            ax0.axhspan(0, 1, 0, _t_med, facecolor="#22c55e", alpha=0.25)
            ax0.axhspan(0, 1, _t_med, _t_high, facecolor="#eab308", alpha=0.25)
            ax0.axhspan(0, 1, _t_high, 1, facecolor="#ef4444", alpha=0.25)
            ax0.barh(0.5, risk_score, height=0.4, color="#0ea5e9", edgecolor="#0369a1", linewidth=1.5)
            ax0.axvline(_t_med, color="gray", linestyle="--", linewidth=1)
            ax0.axvline(_t_high, color="gray", linestyle="--", linewidth=1)
            ax0.set_xlabel("Fraud probability")
            ax0.set_title(f"Risk score: {risk_score:.1%}  —  {level}")
            ax0.set_xticks([0, _t_med, _t_high, 1])
            ax0.set_xticklabels(["0%", f"{_t_med:.0%}", f"{_t_high:.0%}", "100%"])
            # 2) SHAP reason codes (horizontal bars)
            ax1 = axes[1]
            if reason_codes:
                names = [r["feature"] for r in reason_codes]
                impacts = [r["impact"] for r in reason_codes]
                colors = ["#22c55e" if v <= 0 else "#ef4444" for v in impacts]
                y_pos = np.arange(len(names))[::-1]
                ax1.barh(y_pos, impacts, color=colors, edgecolor="gray", linewidth=0.5)
                ax1.axvline(0, color="black", linewidth=0.8)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(names, fontsize=9)
                ax1.set_xlabel("SHAP value (impact on model output)\n(positive = higher risk, negative = lower risk)")
                ax1.set_title("Why this score? — Feature contributions (SHAP)")
            else:
                ax1.text(0.5, 0.5, "No reason codes", ha="center", va="center", transform=ax1.transAxes)
            plt.tight_layout()
            return fig

        def _shap_plain(impact: float) -> str:
            abs_impact = abs(impact)
            if abs_impact >= 2.0:
                strength = "**Strongly**"
            elif abs_impact >= 1.0:
                strength = "**Moderately**"
            else:
                strength = "**Slightly**"
            return f"{strength} {'increased' if impact > 0 else 'decreased'} risk"

        def _gradio_score(features_json: str):
            yield "⏳ **Scoring...**", None
            if not _model:
                yield "**Error:** Model not loaded. Check server startup.", None
                return
            try:
                features = json.loads(features_json)
            except json.JSONDecodeError as e:
                yield f"**Invalid JSON:** {e}", None
                return
            # Use more factors for Gradio so we list each contributing factor (official SHAP style)
            top_k_display = min(15, len(_feature_cols))
            try:
                out = _score_one(
                    _model, _calibrator, features,
                    _feature_cols, _cat_cols, _t_high, _t_med, top_k_display,
                )
            except Exception as e:
                yield f"**Scoring error:** {e}", None
                return
            level_desc = {"HIGH": "🔴 **Review or block** — high fraud probability.",
                         "MEDIUM": "🟡 **Step-up auth / watchlist** — moderate risk.",
                         "LOW": "🟢 **Allow** — low risk."}
            lines = [
                "### Risk score",
                f"**{out['risk_score']:.2%}** — estimated probability of fraud (0–100%).",
                "",
                "### Alert level",
                level_desc.get(out["level"], out["level"]),
                "",
                "### Why this score? — Each factor (SHAP value = impact on model output)",
                "Listed in order of |impact|. **Positive SHAP** = factor pushes score up (higher risk). **Negative SHAP** = factor pushes score down (lower risk).",
                ""
            ]
            for r in out["reason_codes"]:
                interp = _shap_plain(r["impact"])
                val_str = r.get("value", "")
                lines.append(f"- **{r['feature']}** = `{val_str}` — SHAP **{r['impact']:+.2f}** — {interp}")
            if not out["reason_codes"]:
                lines.append("_No reason codes computed._")
            fig = _make_score_charts(out["risk_score"], out["level"], out["reason_codes"])
            yield "\n".join(lines), fig

        def _load_test_sample():
            sample = _get_one_test_sample()
            return sample if sample else ""

        def _get_model_info_md():
            """Return model info string at runtime (after startup has set _feature_cols)."""
            n_f, n_c = len(_feature_cols), len(_cat_cols)
            return f"**Model loaded:** **{n_f}** features (**{n_c}** categorical). Thresholds: HIGH ≥ {_t_high:.0%}, MED ≥ {_t_med:.0%}."

        with gr.Blocks(title="Fraud Risk Scoring") as demo:
            gr.Markdown("""# Fraud Risk Scoring

**What this does:** This app scores a single transaction for fraud risk using a trained CatBoost model and calibrated probabilities. You get an estimated fraud probability (0–100%), an alert level (LOW / MEDIUM / HIGH), and the top factors that pushed the score up or down (SHAP reason codes).

**How to use:**
- **Paste JSON** — Paste a transaction’s features as JSON (same feature names as in training). Missing features are filled with defaults; for best accuracy, include all features.
- **Load sample from test set** — Fills the box with a random transaction from the validation/test set so you can try different examples.
- **Score** — Runs the model and shows risk score, alert level, and reason codes.

**Alert levels:** HIGH (≥70%) → review or block | MEDIUM (30–70%) → step-up / watchlist | LOW (&lt;30%) → allow.
""")
            model_info_md = gr.Markdown(value="**Model loaded:** loading...")
            with gr.Row():
                inp = gr.Textbox(
                    label="Features (JSON)",
                    placeholder="Click 'Load sample from test set' to load a full row, or paste JSON (same feature names as in training).",
                    value="",
                    lines=14,
                )
            with gr.Row():
                btn_score = gr.Button("Score", variant="primary")
                btn_load_test = gr.Button("Load sample from test set")
            out = gr.Markdown(label="Result")
            out_plot = gr.Plot(label="Risk & reason codes")
            btn_score.click(fn=_gradio_score, inputs=inp, outputs=[out, out_plot])
            btn_load_test.click(fn=_load_test_sample, inputs=[], outputs=inp)
            def _on_page_load():
                return _get_model_info_md(), _load_test_sample()
            demo.load(fn=_on_page_load, inputs=[], outputs=[model_info_md, inp])
            gr.Markdown("---\n**API:** `POST /score` with body `{\"features\": {...}}`  |  **Docs:** [/docs](/docs)")

        app = gr.mount_gradio_app(app, demo, path="/gradio")
    except ImportError:
        pass  # Gradio optional; /gradio will 404 if not installed

    return app


# For uvicorn: uvicorn src.serving.app:app --reload
app = get_app()
