# 📈 Time Series Forecasting Benchmark — Transformers + XAI on M5 Walmart

> **Enterprise-grade** deep learning forecasting pipeline on the M5 Walmart dataset (58M rows, 30,490 time series). Full end-to-end implementation on **Azure Databricks** with distributed Apache Spark processing, Delta Lake storage, MLflow experiment tracking, and a complete Explainability AI layer (SHAP, attention heatmaps, counterfactual analysis).

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Azure Databricks](https://img.shields.io/badge/Azure-Databricks-FF3621?logo=databricks)](https://azure.microsoft.com/en-us/products/databricks)
[![Apache Spark](https://img.shields.io/badge/Apache-Spark-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Delta Lake](https://img.shields.io/badge/Delta-Lake-00ADD8)](https://delta.io/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This project implements a **production-ready forecasting benchmark** comparing 4 models across 2 architectural tiers on the M5 Forecasting Competition dataset — the de facto standard for retail demand forecasting research. The full pipeline runs on **Azure Databricks** with Apache Spark handling the 58M-row dataset at scale.

### What was built

| Component | Implementation | Status |
|-----------|---------------|--------|
| Distributed EDA | PySpark + Plotly on Databricks | ✅ Validated |
| Feature Engineering (30+ features) | PySpark window functions, Delta Lake | ✅ Validated |
| Data Quality Validation | Great Expectations (22/23 checks) | ✅ Validated |
| Prophet baseline (30,490 series) | Spark pandas UDF — fully parallel | ✅ Validated |
| XGBoost (2.8M rows, 19 features) | Distributed prep + local training | ✅ Validated |
| TFT — Temporal Fusion Transformer | NeuralForecast + MLflow on Databricks | ✅ Validated |
| PatchTST | NeuralForecast + MLflow on Databricks | ✅ Validated |
| XAI — SHAP + Attention + Counterfactual | Databricks notebook 08 | ✅ Validated |
| Local Python modules (`src/`) | FastAPI serving, local preprocessing | ⚠️ Requires 32GB+ RAM locally |
| Unit & integration tests (`tests/`) | pytest suite | ⚠️ Not run in production pipeline |

> **Architecture note:** The production pipeline runs entirely through the `databricks/notebooks/` directory. The `src/` Python modules provide a clean, modular codebase for local development, CI/CD, and future FastAPI serving — but the full 58M-row dataset requires Databricks + Spark for processing. See [Running the Pipeline](#-running-the-pipeline) for details.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           AZURE ECOSYSTEM                                │
│                                                                          │
│  ┌──────────────────┐          ┌─────────────────────────────────────┐  │
│  │  Azure Data      │          │        Azure Databricks             │  │
│  │  Lake Gen2       │          │                                     │  │
│  │  (ADLS)          │  Delta   │  01 — EDA (PySpark + Plotly)        │  │
│  │                  │◄────────►│  02 — Feature Engineering           │  │
│  │  /raw            │  Lake    │  03 — Data Validation (GE)          │  │
│  │  /processed      │          │  04 — Baselines (Prophet + XGB)     │  │
│  │  /features       │          │  05 — TFT Training                  │  │
│  │  /results        │          │  06 — PatchTST Training             │  │
│  └──────────────────┘          │  07 — Benchmark Results             │  │
│                                │  08 — XAI (SHAP + Attention)        │  │
│  ┌──────────────────┐          └──────────────┬──────────────────────┘  │
│  │  MLflow          │                         │                          │
│  │  Experiment      │◄────────────────────────┘                         │
│  │  Tracking +      │   metrics · params · artifacts · model registry   │
│  │  Model Registry  │                                                    │
│  └──────────────────┘                                                    │
└──────────────────────────────────────────────────────────────────────────┘

Local Development (src/)
┌─────────────────────────────────────────────────────┐
│  src/data/          → ingestion, preprocessing       │
│  src/models/        → model wrappers (XGB, TFT, ...) │
│  src/evaluation/    → metrics, benchmark runner      │
│  src/explainability/→ SHAP, attention, counterfact.  │
│  src/serving/       → FastAPI /predict /explain      │
│  tests/             → pytest unit + integration      │
└─────────────────────────────────────────────────────┘
  ⚠️  Local pipeline requires 32GB+ RAM for full M5 dataset.
     Use Databricks notebooks for production-scale runs.
```

### Data Architecture — Medallion Pattern

```
Raw CSVs (Kaggle M5)
        │
        ▼  spark.read.csv()
┌───────────────────┐
│   Bronze / Raw    │  ADLS /raw/
│   sales, calendar │  Original files, no transformation
│   sell_prices     │
└────────┬──────────┘
         │  melt (wide→long) + merge calendar + prices
         ▼
┌───────────────────┐
│   Silver Layer    │  ADLS /processed/m5_silver/  [Delta, partitioned by dept_id]
│   59M rows        │  Cleaned, merged, date-typed
│   20 columns      │  OPTIMIZE + ZORDER (item_id, store_id, date)
└────────┬──────────┘
         │  30+ features: lags, rolling, calendar, price, target encoding
         ▼
┌───────────────────────────────────────────────────────┐
│                    Gold Layer                         │
│   ADLS /features/m5_gold/   [Delta, partitioned]     │
│                                                       │
│   m5_train  2011-01-29 → 2016-03-27   ~56M rows      │
│   m5_val    2016-03-28 → 2016-04-24   ~1.5M rows     │
│   m5_test   2016-04-25 → 2016-05-22   ~1.5M rows     │
└───────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

> **Transparency note on evaluation splits:**
> XGBoost, TFT, and Prophet were evaluated on the **validation set** (2016-03-28 → 2016-04-24).
> PatchTST predictions land on the **test set** (2016-04-25 → 2016-05-22) due to its
> rolling-window prediction mechanism. A unified single-split evaluation is left as future work.
>
> High SMAPE values (87–151%) are expected on M5 due to the high proportion of
> zero-sales days in sparse retail series — this is a known dataset characteristic,
> not a model failure. **RMSSE is the primary metric**, matching the official M5 competition.

| Model | RMSSE ↓ | RMSE ↓ | MAE ↓ | SMAPE ↓ | Series | Eval Split |
|-------|---------|--------|-------|---------|--------|-----------|
| **XGBoost** ⭐ | **0.375** | 1.928 | 0.969 | 140.6% | All (2.8M rows, 19 features) | Validation |
| TFT | 0.711 | 2.817 | 1.289 | 87.9% | 1,000 series | Validation |
| PatchTST | 0.786 | 1.216 | 0.811 | 151.1% | 1,000 series | Test |
| Prophet | 1.100 | 3.851 | 1.386 | 87.4% | 30,490 series | Validation |

**RMSSE** (Root Mean Squared Scaled Error) scales errors relative to a naïve seasonal baseline — the official M5 competition metric, robust to sparse series.

XGBoost wins on RMSSE because it was trained on the **full dataset** (2.8M rows) with 19 hand-crafted features, while TFT and PatchTST were trained on a 1,000-series CPU subset. With GPU access and full-dataset training, deep learning models are expected to close this gap significantly (literature reports ~15–25% RMSSE improvement).

---

## 🔍 Explainability AI

All XAI experiments run in **Notebook 08** on Databricks and are logged as artifacts in MLflow under the `xai_artifacts` run.

### XGBoost — SHAP Global Feature Importance

![SHAP Summary](docs/images/shap_summary.png)

**Insight:** Recent sales history dominates — `roll_mean_7` and `lag_1` are the strongest predictors with SHAP values up to 30 units. Short-term momentum is the primary driver of Walmart retail demand. Calendar events (`event_name_encoded`) and SNAP food assistance benefits have measurable but secondary effects. Price-related features contribute minimally, consistent with Walmart's EDLP (Every Day Low Price) strategy.

---

### TFT — Variable Selection Weights

![TFT Variable Importance](docs/images/tft_variable_importance.png)

**Insight:** TFT's Variable Selection Network assigns highest weight to `roll_mean_28` (0.171 — long-term trend) and `event_name_encoded` (0.161 — calendar events), revealing a complementary perspective to SHAP. The TFT attention mechanism captures longer-range seasonal patterns and event-driven spikes that XGBoost's lag features partially miss. `snap_CA` receives the lowest weight (0.008), suggesting SNAP California benefits have limited predictive value in this market.

---

### TFT — Temporal Attention Heatmap

![TFT Attention Heatmap](docs/images/tft_attention_heatmap.png)

**Insight:** Both attention heads show strong vertical bands at **t-7, t-14, t-21** — the model autonomously learned weekly seasonality without any hard-coded periodicity constraints. The average heatmap (right panel, red scale) confirms this pattern persists consistently across all query timesteps. This is a hallmark of a well-trained Transformer on retail daily data.

---

### XGBoost — Counterfactual What-If Analysis

![Counterfactual Analysis](docs/images/counterfactual_whatif.png)

**Insight:** SNAP benefit activation drives the largest demand spike (+2 units on day 5). Price elasticity is remarkably low — a +10% price increase reduces mean daily sales by only ~0.01 units (bottom panel), consistent with inelastic demand for staple food products. The Promotion -20% scenario closely tracks baseline, suggesting promotions alone have limited incremental lift for this product class.

---

### Price Sensitivity Sweep

![Price Sensitivity](docs/images/price_sensitivity.png)

**Insight:** Demand is largely **price-inelastic** — mean predicted sales stay near 2.65 units across the entire $1–$6 price range, with only a ~2% drop above the current price of $2.86. This validates Walmart's EDLP positioning and suggests limited revenue upside from dynamic pricing on commodity food items.

---

## 🛠️ Technical Design Decisions

### Why Azure Databricks + Spark?

The M5 dataset in long format is 59 million rows. Local pandas processing failed at the `dropna` step on a 16GB machine (OOM error). Spark on a `Standard_DS4_v2` node (16GB RAM, 8 cores) handles this transparently with distributed window functions computing lag and rolling features across all 30,490 series in parallel. The `OPTIMIZE` + `ZORDER BY (item_id, store_id, date)` step after Silver layer creation reduced downstream Gold layer read times by ~60%.

### Why Delta Lake?

Delta Lake provides ACID transactions, schema enforcement, time-travel, and Z-ORDER clustering. Critical benefits for this pipeline: (1) multiple notebooks read/write concurrently without corruption, (2) `overwriteSchema=true` allows iterative schema evolution during development, (3) Delta's columnar format with Parquet compression reduces storage costs vs raw CSV by ~75%.

### Why TFT?

The Temporal Fusion Transformer (Lim et al., 2021) was specifically designed for multi-horizon forecasting with mixed input types. Key advantages: (1) **Variable Selection Networks** provide native input importance without post-hoc approximation, (2) **multi-head temporal self-attention** explicitly models which past timesteps matter, (3) unified handling of static (store identity), historical (lagged sales, prices), and future-known (calendar, SNAP) covariates. No other architecture handles all three covariate types natively.

### Why PatchTST?

PatchTST (Nie et al., 2023) applies the Vision Transformer patch concept to time series — dividing the lookback window into non-overlapping patches and applying self-attention over them. This reduces attention complexity from O(L²) to O((L/P)²) and enables longer effective lookback windows (56 days here vs 28 for TFT) within the same compute budget. It achieves state-of-the-art results on ETT and Weather benchmarks with no input covariates.

### Why XGBoost as baseline?

XGBoost with manually engineered lag + rolling features remains a strong baseline for tabular time series — consistent with M5 competition findings where gradient boosting often outperforms pure deep learning on sparse retail series. Here it achieves **RMSSE=0.375**, the best result in this benchmark, trained on the full dataset in ~15 minutes on a single CPU node. This result establishes a strong production baseline and demonstrates that feature engineering quality matters as much as model architecture.

### Feature Engineering Strategy

```python
# 19 features engineered via distributed PySpark window functions

Lag features      : lag_1, lag_7, lag_14, lag_28
Rolling stats     : roll_mean_{7,14,28}, roll_std_{7,14}  (shift(1) to avoid leakage)
Calendar          : day_of_week, month, is_weekend, quarter
Price             : sell_price, price_pct_change
Government        : snap_CA, snap_TX, snap_WI  (food assistance benefits)
Events            : event_name_encoded  (ordinal: 30 event types → integer)
```

Target encoding (smoothed by group: `dept_id`, `store_id`, `cat_id`, `state_id`) is computed only on the training split to prevent data leakage into validation and test.

### MLflow Tracking Architecture

Every model run logs: hyperparameters, evaluation metrics, model artifact with input signature, and feature names. The MLflow experiment `/Shared/ts-benchmark-m5` on Databricks serves as single source of truth. Models are registered in the Databricks-managed workspace registry (`tft-m5 v1`, `patchtst-m5 v1`, `xgboost-m5`). The `xai_artifacts` run stores all 7 XAI plots as downloadable PNG artifacts.

---

## 📁 Repository Structure

```
timeseries-transformer-benchmark/
│
├── databricks/                          ← Production pipeline (Azure Databricks)
│   ├── notebooks/
│   │   ├── 01_EDA_spark.py              # EDA with PySpark + Plotly + STL decomposition
│   │   ├── 02_feature_engineering.py    # 30+ features via distributed Spark UDFs
│   │   ├── 03_data_validation.py        # Great Expectations quality gate
│   │   ├── 04_baseline_models.py        # Prophet (parallel UDF) + XGBoost (19 features)
│   │   ├── 05_deep_learning_tft.py      # Temporal Fusion Transformer + MLflow
│   │   ├── 06_deep_learning_patchtst.py # PatchTST + MLflow
│   │   ├── 07_benchmark_results.py      # Leaderboard + Plotly charts + Delta save
│   │   └── 08_explainability.py         # SHAP + TFT attention + counterfactuals
│   ├── cluster_config/
│   │   └── cpu_cluster.json             # Databricks cluster spec (Standard_DS4_v2)
│   └── jobs/
│       └── training_job.json            # Databricks Workflow DAG (8-step pipeline)
│
├── src/                                 ← Local Python modules (dev / CI / serving)
│   ├── data/
│   │   ├── ingestion.py                 # M5 loader (local CSV + ADLS)
│   │   ├── preprocessing.py             # Temporal split pipeline
│   │   └── feature_engineering.py       # Pandas feature engineering (local dev)
│   ├── models/
│   │   ├── baselines/
│   │   │   ├── xgboost_ts.py            # XGBoost wrapper + MLflow logging
│   │   │   └── prophet.py               # Prophet wrapper (multiprocess)
│   │   ├── deep/
│   │   │   ├── tft.py                   # TFT wrapper (NeuralForecast)
│   │   │   ├── patchtst.py              # PatchTST wrapper
│   │   │   └── nbeats_nhits.py          # N-BEATS / N-HiTS wrappers
│   │   └── registry.py                  # Factory: get_model("tft")
│   ├── training/
│   │   └── trainer.py                   # Unified trainer + Optuna tuning
│   ├── evaluation/
│   │   ├── metrics.py                   # RMSSE, MASE, SMAPE, WQL
│   │   ├── benchmark.py                 # Multi-model benchmark runner
│   │   └── statistical_tests.py         # Diebold-Mariano test
│   ├── explainability/
│   │   ├── shap_explainer.py            # SHAP TreeExplainer + plots
│   │   ├── tft_interpretability.py      # Variable importance + attention heatmaps
│   │   ├── attention_viz.py             # PatchTST patch attention visualization
│   │   └── counterfactual.py            # What-if analysis
│   └── serving/
│       ├── api.py                       # FastAPI /predict /explain /compare
│       └── schemas.py                   # Pydantic request/response models
│
├── tests/                               ← pytest test suite
│   ├── unit/
│   │   ├── test_metrics.py              # 30+ metric tests
│   │   ├── test_feature_engineering.py  # Feature engineering tests
│   │   └── test_preprocessing.py        # Temporal split tests
│   └── integration/
│       ├── test_api.py                  # FastAPI endpoint tests
│       └── test_pipeline_e2e.py         # End-to-end pipeline test
│
├── configs/
│   ├── pipeline_config.yaml             # Global config (splits, horizon, MLflow)
│   └── model_configs/
│       ├── tft.yaml                     # TFT hyperparameters + Optuna search space
│       ├── patchtst.yaml                # PatchTST hyperparameters
│       └── xgboost.yaml                 # XGBoost hyperparameters
│
├── docker/
│   ├── Dockerfile                       # FastAPI serving image
│   └── docker-compose.yml               # API + MLflow server
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # Lint + tests + Docker build
│
├── docs/images/                         # XAI result figures
├── .env.example                         # Credentials template
├── .gitignore
├── Makefile                             # make pipeline / serve / test / docker-up
├── pyproject.toml                       # Black, ruff, isort, pytest config
├── requirements.txt
└── README.md
```

---

## 🚀 Running the Pipeline

### Option A — Azure Databricks (Production, Recommended)

This is the validated production path. Handles the full 58M-row dataset.

#### Prerequisites
- Azure subscription ([free tier](https://azure.microsoft.com/free/) with $200 credit works)
- Azure Databricks workspace (Standard tier)
- Azure Data Lake Storage Gen2 account
- Kaggle account (for M5 dataset)

#### Step 1 — Provision Azure Resources

```bash
az login
az group create --name rg-ts-benchmark --location westeurope

# ADLS Gen2 — enable hierarchical namespace
az storage account create \
  --name yourtsstore \
  --resource-group rg-ts-benchmark \
  --location westeurope \
  --sku Standard_LRS \
  --enable-hierarchical-namespace true

az storage container create \
  --name tsdata \
  --account-name yourtsstore \
  --auth-mode login

# Retrieve storage key
az storage account keys list \
  --account-name yourtsstore \
  --resource-group rg-ts-benchmark \
  --output table
```

#### Step 2 — Download and Upload M5 Data

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/ after downloading from kaggle.com → Account → API
# Accept M5 rules at: kaggle.com/competitions/m5-forecasting-accuracy/rules

kaggle competitions download -c m5-forecasting-accuracy -p data/raw/

# Unzip (Windows)
python -c "import zipfile; zipfile.ZipFile('data/raw/m5-forecasting-accuracy.zip').extractall('data/raw/')"
# Unzip (Mac/Linux)
unzip data/raw/m5-forecasting-accuracy.zip -d data/raw/

# Upload to ADLS (repeat for calendar.csv and sell_prices.csv)
az storage blob upload \
  --account-name yourtsstore \
  --account-key YOUR_STORAGE_KEY \
  --container-name tsdata \
  --name raw/sales_train_evaluation.csv \
  --file data/raw/sales_train_evaluation.csv --overwrite
```

#### Step 3 — Configure Databricks Cluster

In Databricks UI → **Compute** → **Create compute**:

```
Cluster name     : ts-benchmark-cpu
Runtime          : 14.3 LTS (Scala 2.12, Spark 3.5.0)
Node type        : Standard_DS4_v2  (16 GB RAM, 8 cores)
Single node      : ✅ enabled
Terminate after  : 60 minutes
```

**Set environment variables** (Compute → Edit → Advanced Options → Environment Variables):

```
ADLS_ACCOUNT_NAME=yourtsstore
ADLS_KEY=your_storage_account_key
```

> ⚠️ Never hardcode credentials in notebooks. All notebooks read from environment variables via `os.environ.get("ADLS_ACCOUNT_NAME")`.

#### Step 4 — Import Notebooks

In Databricks UI → **Workspace** → your home → **Create Folder** → `ts-benchmark`

Inside the folder → **Import** → upload each `.py` file from `databricks/notebooks/` one by one.

#### Step 5 — Run Notebooks in Order

Each notebook reads from the Delta layer written by the previous one. **Do not skip steps.**

| Notebook | Duration | Output |
|----------|----------|--------|
| `01_EDA_spark.py` | ~20 min | Silver Delta layer + Plotly charts |
| `02_feature_engineering.py` | ~35 min | Gold Delta splits (train/val/test, 30+ features) |
| `03_data_validation.py` | ~5 min | 22/23 GE checks passed, logged to MLflow |
| `04_baseline_models.py` | ~40 min | Prophet + XGBoost metrics in MLflow |
| `05_deep_learning_tft.py` | ~60–90 min | TFT model `tft-m5` registered in MLflow |
| `06_deep_learning_patchtst.py` | ~60–90 min | PatchTST model `patchtst-m5` registered |
| `07_benchmark_results.py` | ~5 min | Leaderboard + interactive Plotly charts |
| `08_explainability.py` | ~15 min | 7 XAI plots logged to MLflow artifacts |

> **Note on pip installs:** Notebooks 05 and 06 install `typing_extensions==4.8.0` before `neuralforecast==1.6.4` to resolve a compatibility issue with Databricks Runtime 14.3. The kernel restarts automatically — re-run the ADLS config cell after each restart.

#### Step 6 — View Results in MLflow

Databricks sidebar → **Experiments** → `/Shared/ts-benchmark-m5`

- One run per model with hyperparameters and metrics
- `xai_artifacts` run: 7 XAI plots downloadable as PNG
- `benchmark_summary` run: full leaderboard CSV

---

### Option B — Local Development (src/ modules)

> ⚠️ **RAM requirement:** The full M5 dataset requires 32GB+ RAM for local pandas processing. For machines with less RAM, use a sampled subset or run on Databricks.

```bash
# Setup
git clone https://github.com/your-username/timeseries-transformer-benchmark
cd timeseries-transformer-benchmark
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy and configure credentials
cp .env.example .env
# Edit .env with your Kaggle credentials for data download

# Download M5 data
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/
python -c "import zipfile; zipfile.ZipFile('data/raw/m5-forecasting-accuracy.zip').extractall('data/raw/')"

# Run preprocessing (requires 32GB+ RAM for full dataset)
python -m src.data.preprocessing --config configs/pipeline_config.yaml

# Train a single model
python -m src.training.trainer --config configs/pipeline_config.yaml --models xgboost

# Launch MLflow UI
mlflow ui --port 5000 --backend-store-uri mlruns/

# Run test suite (no data required — uses synthetic data)
make test
# or: pytest tests/ -v --cov=src

# Launch FastAPI serving
make serve
# or: uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

---

### Option C — Docker (API serving)

```bash
# Build and start API + MLflow server
make docker-up
# or: docker-compose -f docker/docker-compose.yml up --build

# Test endpoints
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"series_id": "FOODS_3_090_CA_3", "model": "tft", "horizon": 28}'

curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"series_id": "FOODS_3_090_CA_3", "model": "xgboost", "horizon": 7}'

curl http://localhost:8000/compare

# Interactive API docs
open http://localhost:8000/docs
```

---

## 🧪 Running Tests

```bash
# All tests with coverage report
make test

# Unit tests only (fast, no data required — synthetic dataset)
make test-unit

# Integration tests (requires model artifacts)
make test-integration

# Specific test file
pytest tests/unit/test_metrics.py -v

# Coverage HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

The test suite covers: RMSSE/MASE/SMAPE/RMSE/MAE correctness, temporal split non-overlap, feature engineering column completeness, API endpoint schemas, and end-to-end pipeline on synthetic data.

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `neuralforecast` | 1.6.4 | TFT and PatchTST training |
| `prophet` | 1.1.5 | Baseline (30,490 series parallel via Spark UDF) |
| `xgboost` | 2.x | Gradient boosting baseline |
| `shap` | 0.45 | XGBoost SHAP TreeExplainer |
| `great-expectations` | 0.18.14 | Data quality validation |
| `mlflow` | 2.x (Databricks native) | Experiment tracking + model registry |
| `delta-spark` | 3.x (Databricks native) | Delta Lake ACID read/write |
| `torch` | 2.x | Deep learning backend (TFT, PatchTST) |
| `plotly` | 5.x | Interactive benchmark charts |
| `fastapi` | 0.111 | REST API serving layer |
| `optuna` | 3.6 | Hyperparameter tuning (configured, not run) |
| `typing_extensions` | 4.8.0 | Compatibility fix for neuralforecast on DBR 14.3 |

---

## ⚠️ Known Limitations & Roadmap

**Evaluation split inconsistency** — TFT/XGBoost/Prophet on val, PatchTST on test. A unified evaluation framework with all models on the same test split is the primary v2 improvement.

**Deep learning series coverage** — TFT and PatchTST trained on 1,000/30,490 series due to CPU constraints. GPU cluster (Standard_NC6s_v3) access would enable full-dataset training and is expected to improve deep learning RMSSE by 15–25%.

**Hyperparameter tuning** — Optuna search spaces are defined in `configs/model_configs/` and the `Trainer` class supports `--tune` flag, but tuning was not run in the current benchmark due to compute budget.

**Probabilistic forecasting** — TFT was trained with MAE loss (point forecast). Switching to `MQLoss(level=[50, 80, 95])` would add prediction intervals — directly applicable to inventory safety stock optimization.

**SARIMA baseline** — Implemented in `src/models/baselines/` but not run in the Databricks benchmark due to the per-series training overhead at 30,490 series scale.

---

## 📚 References

- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023.
- Makridakis, S. et al. (2022). *M5 accuracy competition: Results, findings, and conclusions.* International Journal of Forecasting.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Lundberg, S. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- Oreshkin, B. et al. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.* ICLR 2020.

---

## Author

**Achraf Jemali** — Data & AI Engineer.

[![GitHub](https://img.shields.io/badge/GitHub-JEMALIACHRAF-black?logo=github&style=flat-square)](https://github.com/JEMALIACHRAF)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Achraf_Jemali-0077B5?logo=linkedin&style=flat-square)](https://linkedin.com/in/achraf-jemali-54a417239)

If you found this useful or want to discuss the design choices, feel free to reach out.

---

## 📄 License

MIT — see [LICENSE](LICENSE)