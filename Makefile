# =============================================================================
# fin-risk-engine Makefile
# =============================================================================
.PHONY: help install install-dev lint test train tune evaluate serve monitor \
        benchmark clean clean-models clean-data docker-build docker-up docker-down

PYTHON     := python
PYTHONPATH := PYTHONPATH=.
VENV       := .finvenv
VENV_BIN   := $(VENV)/bin

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Environment
# =============================================================================

install:  ## Install core + ML dependencies
	pip install -e ".[ml,serve,viz]"

install-dev:  ## Install all dependencies including dev tools
	pip install -e ".[all]"

# =============================================================================
# Code Quality
# =============================================================================

lint:  ## Run ruff linter
	ruff check src/ scripts/ tests/

format:  ## Auto-fix lint issues
	ruff check --fix src/ scripts/ tests/

typecheck:  ## Run mypy type checker
	mypy src/

# =============================================================================
# Tests
# =============================================================================

test:  ## Run all tests
	$(PYTHONPATH) pytest tests/ -v

test-cov:  ## Run tests with coverage report
	$(PYTHONPATH) pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# =============================================================================
# Data Pipeline
# =============================================================================

build:  ## Build features and splits (with behavioral features — slow)
	$(PYTHONPATH) $(PYTHON) scripts/build_features_and_splits.py

build-fast:  ## Build features and splits without behavioral (fast iteration)
	$(PYTHONPATH) $(PYTHON) scripts/build_features_and_splits.py --skip-behavioral

build-sample:  ## Build on 50K sample (development)
	$(PYTHONPATH) $(PYTHON) scripts/build_features_and_splits.py --skip-behavioral --nrows 50000

# =============================================================================
# Training
# =============================================================================

train:  ## Train CatBoost (champion) with tuned params
	$(PYTHONPATH) $(PYTHON) scripts/train_model.py --use-tuned --subsample-train 3000000

train-xgboost:  ## Train XGBoost with tuned params
	$(PYTHONPATH) $(PYTHON) scripts/train_model.py --model-type xgboost --use-tuned --subsample-train 3000000

train-all:  ## Train both CatBoost and XGBoost
	$(MAKE) train
	$(MAKE) train-xgboost

# =============================================================================
# Hyperparameter Tuning
# =============================================================================

tune:  ## Tune CatBoost (20 Optuna trials)
	$(PYTHONPATH) $(PYTHON) scripts/run_tuning.py --model-type catboost --n-trials 20 --subsample-train 3000000

tune-xgboost:  ## Tune XGBoost (10 Optuna trials)
	$(PYTHONPATH) $(PYTHON) scripts/run_tuning.py --model-type xgboost --n-trials 10 --subsample-train 3000000

# =============================================================================
# Evaluation
# =============================================================================

evaluate:  ## Evaluate both models on val set (head-to-head)
	$(PYTHONPATH) $(PYTHON) scripts/evaluate_model.py --compare

evaluate-test:  ## Evaluate both models on held-out test set (OOT 2020)
	$(PYTHONPATH) $(PYTHON) scripts/evaluate_model.py --compare --val data/processed/test.parquet

benchmark:  ## Run 3-model benchmark with tuned params
	$(PYTHONPATH) $(PYTHON) scripts/run_benchmark.py --models catboost,xgboost --use-tuned

# =============================================================================
# Full Pipeline
# =============================================================================

pipeline:  ## Full pipeline: build → train → verify → evaluate
	$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py

pipeline-fast:  ## Fast pipeline: build (no behavioral) → train → evaluate
	$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py --skip-behavioral

# =============================================================================
# Serving
# =============================================================================

serve:  ## Launch FastAPI + Gradio serving API (port 8000)
	$(PYTHONPATH) $(PYTHON) scripts/run_serve.py

score:  ## Batch score test data → outputs/artifacts/predictions.csv
	$(PYTHONPATH) $(PYTHON) scripts/score_test_data.py --reason-codes

# =============================================================================
# Monitoring
# =============================================================================

monitor:  ## Run drift + performance monitoring (val as current)
	$(PYTHONPATH) $(PYTHON) scripts/run_monitoring.py

monitor-test:  ## Run monitoring with test set as current period
	$(PYTHONPATH) $(PYTHON) scripts/run_monitoring.py --current data/processed/test.parquet

# =============================================================================
# Docker (WS4)
# =============================================================================

docker-build:  ## Build Docker serving image
	docker build -t fin-risk-engine:latest .

docker-up:  ## Start full stack (API + Redis + Kafka + MLflow)
	docker compose up -d

docker-down:  ## Stop all services
	docker compose down

docker-logs:  ## Stream API container logs
	docker compose logs -f api

# =============================================================================
# Clean
# =============================================================================

clean:  ## Remove Python cache files
	find . -type d -name "__pycache__" -not -path "./.finvenv/*" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -not -path "./.finvenv/*" -delete 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage; \
	echo "Cache cleaned."

clean-models:  ## Remove trained model artifacts (keeps tuning params)
	rm -f outputs/models/*.cbm outputs/models/*.json outputs/models/*.joblib
	@echo "Model artifacts removed. Re-run: make train-all"

clean-data:  ## Remove processed data (keeps raw data)
	rm -f data/processed/*.parquet
	@echo "Processed data removed. Re-run: make build"
