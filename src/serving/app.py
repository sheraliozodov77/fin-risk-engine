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

    Calibrator and feature metadata always loaded from files (not in MLflow yet -- WS4).
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
    from fastapi.responses import RedirectResponse

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
        """Redirect root to the Gradio UI."""
        return RedirectResponse(url="/gradio")

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

    # -- Gradio UI -------------------------------------------------------------
    try:
        import math

        import gradio as gr
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # -- Helper: champion model info for the header banner -----------------
        _PERF_STATS = {
            "catboost": "PR-AUC **0.854** | ROC-AUC **0.999** | Brier **0.000527**",
            "xgboost":  "PR-AUC **0.847** | ROC-AUC **0.999** | Brier **0.000555**",
        }

        def _get_champion_banner() -> str:
            mtype = "catboost"
            version = "--"
            try:
                from src.config import get_model_config
                cfg = get_model_config()
                mlflow_cfg = cfg.get("mlflow", {})
                from src.tracking.mlflow_tracker import MLflowTracker
                tracker = MLflowTracker(
                    experiment_name=mlflow_cfg.get("experiment_name", "fin-risk-engine"),
                    tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"),
                )
                info = tracker.get_champion_info(mlflow_cfg.get("registry_model_name", "fraud-detection"))
                if info and info.get("model_type"):
                    mtype = info["model_type"].lower()
                    version = str(info.get("version", "--"))
            except Exception:
                pass
            perf = _PERF_STATS.get(mtype, _PERF_STATS["catboost"])
            n_feat = len(_feature_cols) or 99
            n_cat  = len(_cat_cols) or 9
            return (
                f"🏆 **Champion: {mtype.upper()}** (registry v{version})  *  "
                f"Test set 2020 (held-out): {perf}  *  "
                f"**{n_feat}** features ({n_cat} categorical)  *  "
                f"Thresholds: HIGH ≥ {_t_high:.0%} | MED ≥ {_t_med:.0%}"
            )

        # -- Helper: load sample row from test set -----------------------------
        def _safe_json_val(v: Any) -> Any:
            """Convert pandas/numpy scalars to plain Python for json.dumps."""
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                pass
            if hasattr(v, "item"):
                return v.item()
            return v

        def _get_sample_by_label(want_fraud: bool | None = None) -> tuple[str, str]:
            """Return (json_str, ground_truth_md) for one row from test.parquet."""
            from src.config import get_paths
            root = _project_root()
            paths = get_paths()
            processed = paths.get("data", {}).get("processed", {})
            for _, rel in [
                ("test", processed.get("test", "data/processed/test.parquet")),
                ("val",  processed.get("val",  "data/processed/val.parquet")),
            ]:
                data_path = Path(rel) if Path(rel).is_absolute() else root / rel
                if not data_path.exists():
                    continue
                df = pd.read_parquet(data_path)
                if len(df) == 0:
                    continue
                # Filter to labeled rows
                if "is_fraud" in df.columns:
                    df = df.dropna(subset=["is_fraud"])
                if want_fraud is not None and "is_fraud" in df.columns:
                    subset = df[df["is_fraud"].astype(int) == int(want_fraud)]
                    df = subset if len(subset) > 0 else df
                if len(df) == 0:
                    continue
                row = df.sample(n=1, random_state=None).iloc[0]
                # Ground truth label
                label_md = ""
                if "is_fraud" in row.index:
                    lv = int(row["is_fraud"])
                    label_md = (
                        "**Ground truth:** 🔴 FRAUD -- this is a confirmed fraudulent transaction"
                        if lv == 1 else
                        "**Ground truth:** 🟢 LEGITIMATE -- this is a normal transaction"
                    )
                # Feature JSON (only model features)
                feat_cols = [c for c in _feature_cols if c in row.index]
                feat_dict = {k: _safe_json_val(row[k]) for k in feat_cols}
                return json.dumps(feat_dict, indent=2), label_md
            return "", ""

        # -- Helper: SHAP + gauge chart ----------------------------------------
        def _make_risk_chart(risk_score: float, level: str, reason_codes: list[dict]) -> plt.Figure:
            """Professional risk gauge (top) + SHAP waterfall bar (bottom)."""
            n = max(len(reason_codes), 1)
            lc = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}.get(level, "#64748b")
            lb = {"HIGH": "#fee2e2", "MEDIUM": "#fef3c7", "LOW": "#dcfce7"}.get(level, "#f8fafc")

            fig, axes = plt.subplots(
                2, 1,
                figsize=(9.5, 2.6 + n * 0.52),
                gridspec_kw={"height_ratios": [1.5, n], "hspace": 0.55},
            )
            fig.patch.set_facecolor("#ffffff")

            # -- Gauge ---------------------------------------------------------
            ax = axes[0]
            ax.set_facecolor(lb)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            ax.axhspan(0, 1, 0,      _t_med,  facecolor="#bbf7d0", alpha=0.55, zorder=0)
            ax.axhspan(0, 1, _t_med,  _t_high, facecolor="#fde68a", alpha=0.55, zorder=0)
            ax.axhspan(0, 1, _t_high, 1,       facecolor="#fecaca", alpha=0.55, zorder=0)

            ax.barh(0.5, risk_score, height=0.52,
                    color=lc, alpha=0.92, edgecolor="white", linewidth=2, zorder=2)
            ax.axvline(_t_med,  color="#64748b", ls="--", lw=1.2, zorder=3)
            ax.axvline(_t_high, color="#64748b", ls="--", lw=1.2, zorder=3)

            # Zone labels
            for x, txt, col in [
                (_t_med / 2,              "LOW",  "#15803d"),
                ((_t_med + _t_high) / 2,  "MED",  "#92400e"),
                ((_t_high + 1) / 2,       "HIGH", "#991b1b"),
            ]:
                ax.text(x, 0.08, txt, ha="center", va="bottom",
                        fontsize=8, color=col, fontweight="bold", zorder=4)

            ax.set_xticks([0, _t_med, _t_high, 1])
            ax.set_xticklabels(["0%", f"{_t_med:.0%}", f"{_t_high:.0%}", "100%"], fontsize=9)
            ax.set_title(
                f"Fraud probability: {risk_score:.2%}     *     Alert level: {level}",
                fontsize=12, fontweight="bold", color=lc, pad=5,
            )

            # -- SHAP bar chart -------------------------------------------------
            ax2 = axes[1]
            if reason_codes:
                names   = [r["feature"]       for r in reason_codes]
                impacts = [r["impact"]         for r in reason_codes]
                values  = [r.get("value", "")  for r in reason_codes]

                y_pos  = np.arange(len(names))
                colors = ["#ef4444" if v > 0 else "#3b82f6" for v in impacts]

                ax2.barh(y_pos, impacts, color=colors, alpha=0.87,
                         edgecolor="white", linewidth=0.4, height=0.68)
                ax2.axvline(0, color="#0f172a", linewidth=1.0)

                def _label(nm: str, val: str) -> str:
                    display = nm.replace("_", " ")
                    if val and str(val) not in ("(missing)", "None", "nan", "-999", ""):
                        short = str(val)[:14]
                        s = f"{display}  [{short}]"
                    else:
                        s = display
                    # Hard-truncate so labels never overflow the left margin
                    return s if len(s) <= 30 else s[:28] + ".."

                ylabels = [_label(nm, v) for nm, v in zip(names, values)]

                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(ylabels, fontsize=8.5)
                ax2.invert_yaxis()          # Most impactful at top
                ax2.set_xlabel(
                    "SHAP value  (positive → raises fraud risk     negative → lowers fraud risk)",
                    fontsize=8.5,
                )
                ax2.set_title(
                    "Why this score? -- Top feature contributions (SHAP)",
                    fontsize=10, fontweight="bold",
                )
                ax2.grid(axis="x", alpha=0.22, linestyle="--")
                ax2.spines[["top", "right"]].set_visible(False)
            else:
                ax2.text(0.5, 0.5, "No SHAP reason codes computed",
                         ha="center", va="center", transform=ax2.transAxes,
                         fontsize=10, color="#94a3b8")
                ax2.axis("off")

            # Use explicit margins so long y-axis labels are never clipped
            fig.subplots_adjust(left=0.32, right=0.97, top=0.97, bottom=0.06, hspace=0.55)
            return fig

        # -- Helper: format result markdown -----------------------------------
        def _format_result(out: dict, ground_truth_md: str = "") -> str:
            level  = out["level"]
            score  = out["risk_score"]
            icon   = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")
            action = {
                "HIGH":   "**Recommended action:** Block transaction / route to manual review",
                "MEDIUM": "**Recommended action:** Step-up authentication / flag for watchlist",
                "LOW":    "**Recommended action:** Auto-approve",
            }.get(level, "")
            lines = [
                f"## {icon} {level} RISK -- {score:.2%}",
                "",
                action,
            ]
            if ground_truth_md:
                lines += ["", ground_truth_md]
            lines += [
                "",
                "---",
                "### Feature contributions (SHAP)",
                "Positive values raise the fraud probability; negative values lower it.",
                "",
                "| # | Feature | Observed value | SHAP impact | Direction |",
                "|---|---------|---------------|-------------|-----------|",
            ]
            for i, r in enumerate(out["reason_codes"], 1):
                direction = "⬆ raises risk" if r["impact"] > 0 else "⬇ lowers risk"
                val = r.get("value", "--")
                lines.append(
                    f"| {i} | `{r['feature']}` | {val} | `{r['impact']:+.3f}` | {direction} |"
                )
            if not out["reason_codes"]:
                lines.append("_No reason codes computed._")
            return "\n".join(lines)

        # -- Gradio event handlers ---------------------------------------------
        _last_truth: list[str] = [""]   # mutable closure container

        def _gradio_score(features_json: str):
            yield "⏳ Scoring...", None
            if not _model:
                yield "**Error:** Model not loaded. Check server startup logs.", None
                return
            try:
                features = json.loads(features_json)
            except json.JSONDecodeError as e:
                yield f"**Invalid JSON:** {e}\n\nCheck the JSON syntax and try again.", None
                return
            top_k_display = min(15, len(_feature_cols))
            try:
                out = _score_one(
                    _model, _calibrator, features,
                    _feature_cols, _cat_cols, _t_high, _t_med, top_k_display,
                )
            except Exception as e:
                yield f"**Scoring error:** {e}", None
                return
            md  = _format_result(out, _last_truth[0])
            fig = _make_risk_chart(out["risk_score"], out["level"], out["reason_codes"])
            yield md, fig

        def _load_fraud():
            j, label = _get_sample_by_label(want_fraud=True)
            _last_truth[0] = label
            return j, label

        def _load_legit():
            j, label = _get_sample_by_label(want_fraud=False)
            _last_truth[0] = label
            return j, label

        def _on_page_load():
            j, label = _get_sample_by_label()
            _last_truth[0] = label
            banner = _get_champion_banner()
            return banner, j, label

        # -- Gradio layout -----------------------------------------------------
        with gr.Blocks(title="fin-risk-engine -- Fraud Risk Scoring") as demo:

            # -- Header --------------------------------------------------------
            # -- Header --------------------------------------------------------
            gr.Markdown("# fin-risk-engine -- Real-Time Fraud Risk Scoring")
            gr.Markdown(
                "Scores a single bank transaction using calibrated ML. "
                "Returns a fraud probability, alert level, and top SHAP reason codes."
            )

            gr.Markdown("---")

            # -- Main two-column panel -----------------------------------------
            with gr.Row():
                # Left: input
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Transaction Features")
                    gr.Markdown(
                        "Load a sample or paste your own JSON. "
                        "Feature names match the training schema (99 features: 44 static + 55 behavioral rolling-window)."
                    )
                    inp = gr.Textbox(
                        label="Features (JSON)",
                        placeholder="Click a Load button below, or paste transaction JSON here.",
                        lines=22,
                    )
                    true_label = gr.Markdown(value="")
                    with gr.Row():
                        btn_fraud = gr.Button("🔴 Load Fraud Sample",      variant="secondary", size="sm")
                        btn_legit = gr.Button("🟢 Load Legitimate Sample", variant="secondary", size="sm")
                    btn_score = gr.Button("▶  Score Transaction", variant="primary", size="lg")

                # Right: results
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Risk Assessment")
                    result_md = gr.Markdown(
                        value="_Load a sample and click **Score Transaction** to see the risk assessment._"
                    )
                    out_plot = gr.Plot(show_label=False)

            gr.Markdown("---")

            # -- Champion banner + performance (below the scoring columns) -----
            champion_banner = gr.Markdown(value="⏳ Loading model info...")

            with gr.Accordion("📊 Model performance (test set 2020)", open=False):
                gr.Markdown(
                    "**CatBoost** (champion): PR-AUC **0.854**, ROC-AUC **0.999**, "
                    "Recall@P90 **0.600**, Brier **0.000527**, F1@MED **0.797**\n\n"
                    "**XGBoost**: PR-AUC **0.847**, ROC-AUC **0.999**, "
                    "Recall@P90 **0.643**, Brier **0.000555**, F1@HIGH **0.776**\n\n"
                    "Both trained on CaixaBank Tech 2024 AI Hackathon data "
                    "(13.3M transactions, 0.15% fraud rate). No val-to-test degradation."
                )

            # -- Footer --------------------------------------------------------
            gr.Markdown("""---
**Decision policy:**  🔴 HIGH (≥ 70%) → Block / route to manual review  *  🟡 MEDIUM (30-70%) → Step-up authentication  *  🟢 LOW (< 30%) → Auto-approve

**API endpoints:** `POST /score` * `POST /score_batch` * [`GET /model_info`](/model_info) * [`GET /sample_from_test`](/sample_from_test) * [`GET /health`](/health) * [`API Docs`](/docs)
""")

            # -- Event wiring --------------------------------------------------
            btn_score.click(fn=_gradio_score, inputs=inp, outputs=[result_md, out_plot])
            btn_fraud.click(fn=_load_fraud,   inputs=[], outputs=[inp, true_label])
            btn_legit.click(fn=_load_legit,   inputs=[], outputs=[inp, true_label])
            demo.load(fn=_on_page_load, inputs=[], outputs=[champion_banner, inp, true_label])

        app = gr.mount_gradio_app(app, demo, path="/gradio")
    except ImportError:
        pass  # Gradio optional; /gradio will 404 if not installed

    return app


# For uvicorn: uvicorn src.serving.app:app --reload
app = get_app()
