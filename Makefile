.PHONY: install pipeline preprocess train evaluate serve test lint docker-build clean

# ── Setup ──────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	pre-commit install

# ── Pipeline ───────────────────────────────────────────────────
pipeline: preprocess train evaluate
	@echo "✅ Full pipeline completed"

preprocess:
	python -m src.data.preprocessing --config configs/pipeline_config.yaml

train:
	python -m src.training.trainer --config configs/pipeline_config.yaml

evaluate:
	python -m src.evaluation.benchmark --config configs/pipeline_config.yaml

# ── Individual model training ──────────────────────────────────
train-baselines:
	python -m src.training.trainer --models sarima prophet xgboost

train-deep:
	python -m src.training.trainer --models nbeats nhits tft patchtst

train-tft:
	python -m src.training.trainer --models tft --config configs/model_configs/tft.yaml

train-patchtst:
	python -m src.training.trainer --models patchtst --config configs/model_configs/patchtst.yaml

# ── XAI ────────────────────────────────────────────────────────
explain:
	python -m src.explainability.shap_explainer
	python -m src.explainability.attention_viz
	python -m src.explainability.tft_interpretability

# ── Serving ────────────────────────────────────────────────────
serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-gradio:
	python -m src.serving.gradio_app

# ── Testing ────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ── Code quality ───────────────────────────────────────────────
lint:
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# ── Docker ─────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile -t ts-benchmark:latest .

docker-up:
	docker-compose -f docker/docker-compose.yml up --build

docker-down:
	docker-compose -f docker/docker-compose.yml down

# ── MLflow ─────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --port 5000 --backend-store-uri mlruns/

# ── Databricks ─────────────────────────────────────────────────
databricks-upload:
	databricks workspace import-dir databricks/notebooks /Shared/ts-benchmark --overwrite

databricks-run:
	databricks jobs run-now --job-id $(JOB_ID)

# ── Clean ──────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf .coverage htmlcov/
