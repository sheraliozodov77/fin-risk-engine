"""
MLflow experiment tracking wrapper for fin-risk-engine.

Gracefully degrades if mlflow is not installed — training and evaluation
continue normally with a logged warning. No MLflow server required;
uses local file-based tracking (mlruns/) by default.

Usage:
    tracker = MLflowTracker()
    with tracker.start_run("catboost-train", tags={"stage": "training"}):
        tracker.log_params({"lr": 0.05, "depth": 9})
        tracker.log_metrics({"val_pr_auc": 0.852})
        tracker.log_model(model, "catboost_model", model_type="catboost")
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    Thin wrapper around MLflow for experiment tracking and model registry.
    Falls back to a no-op if mlflow is not installed or unavailable.
    """

    def __init__(
        self,
        experiment_name: str = "fin-risk-engine",
        tracking_uri: str = "mlruns",
    ) -> None:
        self._mlflow = None
        self._run = None
        self._experiment_name = experiment_name

        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            logger.info(
                "mlflow_initialized",
                experiment=experiment_name,
                tracking_uri=tracking_uri,
            )
        except ImportError:
            logger.warning(
                "mlflow_not_installed",
                note="pip install mlflow to enable tracking",
            )
        except Exception as exc:
            logger.warning("mlflow_init_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        """Context manager for an MLflow run. No-op if mlflow is unavailable."""
        if self._mlflow is None:
            yield self
            return

        with self._mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            self._run = run
            logger.info(
                "mlflow_run_started",
                run_id=run.info.run_id,
                run_name=run_name,
                experiment=self._experiment_name,
            )
            try:
                yield self
            finally:
                logger.info("mlflow_run_ended", run_id=run.info.run_id)
                self._run = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat dict of hyperparameters."""
        if not self._is_active:
            return
        try:
            # MLflow requires string values for params
            clean = {k: str(v) for k, v in params.items()}
            self._mlflow.log_params(clean)
        except Exception as exc:
            logger.warning("mlflow_log_params_failed", error=str(exc))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a flat dict of numeric metrics. NaN / inf values are skipped."""
        if not self._is_active:
            return
        try:
            clean = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            }
            if clean:
                self._mlflow.log_metrics(clean)
        except Exception as exc:
            logger.warning("mlflow_log_metrics_failed", error=str(exc))

    def log_artifact(self, path: str | Path) -> None:
        """Log a file or directory as a run artifact."""
        if not self._is_active:
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            logger.warning("mlflow_log_artifact_failed", path=str(path), error=str(exc))

    def set_tag(self, key: str, value: str) -> None:
        if not self._is_active:
            return
        try:
            self._mlflow.set_tag(key, value)
        except Exception as exc:
            logger.warning("mlflow_set_tag_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Model logging (creates MLflow model package for registry)
    # ------------------------------------------------------------------

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_type: str,
        extra_artifacts: list[str | Path] | None = None,
    ) -> str | None:
        """
        Log model using the appropriate MLflow flavor (catboost / xgboost).
        Returns the model URI (runs:/<run_id>/<artifact_path>) for registry use,
        or None if logging failed.
        """
        if not self._is_active:
            return None
        try:
            if model_type == "catboost":
                import mlflow.catboost as mlflow_cb
                mlflow_cb.log_model(model, artifact_path)
            elif model_type == "xgboost":
                import mlflow.xgboost as mlflow_xgb
                mlflow_xgb.log_model(model, artifact_path)
            else:
                logger.warning("mlflow_unknown_model_type", model_type=model_type)
                return None

            model_uri = f"runs:/{self.run_id}/{artifact_path}"
            logger.info("mlflow_model_logged", uri=model_uri, model_type=model_type)

            for path in extra_artifacts or []:
                self.log_artifact(path)

            return model_uri

        except Exception as exc:
            logger.warning("mlflow_log_model_failed", model_type=model_type, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: dict | None = None,
    ) -> None:
        """
        Register a logged model in the MLflow Model Registry.
        model_uri must be the return value of log_model() (runs:/<run_id>/...).
        """
        if not self._is_active or model_uri is None:
            return
        try:
            result = self._mlflow.register_model(model_uri=model_uri, name=name, tags=tags or {})
            logger.info(
                "mlflow_model_registered",
                name=name,
                version=result.version,
                status=result.status,
            )
        except Exception as exc:
            logger.warning("mlflow_register_model_failed", name=name, error=str(exc))

    def promote_champion(self, registry_name: str, champion_model_type: str) -> None:
        """
        Set '@champion' alias on the latest version matching champion_model_type.
        Set '@challenger' alias on the latest version of any other type.
        Reads model_type from version tags, falling back to run tags.
        """
        if self._mlflow is None:
            return
        try:
            client = self._mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{registry_name}'")

            # Group by model_type, keep latest version per type
            by_type: dict[str, Any] = {}
            for v in versions:
                mtype = v.tags.get("model_type")
                if not mtype and v.run_id:
                    try:
                        run = self._mlflow.get_run(v.run_id)
                        mtype = run.data.tags.get("model_type")
                    except Exception:
                        pass
                if mtype:
                    if mtype not in by_type or int(v.version) > int(by_type[mtype].version):
                        by_type[mtype] = v

            champion_v = by_type.get(champion_model_type)
            if champion_v:
                client.set_registered_model_alias(registry_name, "champion", champion_v.version)
                logger.info(
                    "promoted_champion",
                    registry=registry_name,
                    model_type=champion_model_type,
                    version=champion_v.version,
                )

            for mtype, v in by_type.items():
                if mtype != champion_model_type:
                    client.set_registered_model_alias(registry_name, "challenger", v.version)
                    logger.info(
                        "set_challenger",
                        registry=registry_name,
                        model_type=mtype,
                        version=v.version,
                    )

        except Exception as exc:
            logger.warning("promote_champion_failed", error=str(exc))

    def get_champion_info(self, registry_name: str) -> dict | None:
        """
        Return {model_type, version, run_id} for the @champion alias.
        Returns None if registry or alias not found.
        """
        if self._mlflow is None:
            return None
        try:
            client = self._mlflow.MlflowClient()
            v = client.get_model_version_by_alias(registry_name, "champion")
            mtype = v.tags.get("model_type")
            if not mtype and v.run_id:
                run = self._mlflow.get_run(v.run_id)
                mtype = run.data.tags.get("model_type")
            return {"model_type": mtype, "version": v.version, "run_id": v.run_id}
        except Exception as exc:
            logger.warning("get_champion_info_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None

    @property
    def _is_active(self) -> bool:
        return self._mlflow is not None and self._run is not None
