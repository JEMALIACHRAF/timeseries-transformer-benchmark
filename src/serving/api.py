"""
FastAPI serving layer.

Endpoints:
  POST /predict   — generate forecast for a series
  POST /explain   — return SHAP values or attention weights
  GET  /compare   — return benchmark results table
  GET  /models    — list available models
  GET  /health    — health check
"""
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import (
    BenchmarkResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Global model cache ─────────────────────────────────────────────────────────
_models: Dict[str, Any] = {}
_benchmark_results: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Loading models from MLflow registry...")
    await _load_models()
    logger.info("API ready.")
    yield
    logger.info("Shutting down API.")


async def _load_models():
    """Load all registered models from MLflow."""
    global _models
    model_names = ["tft", "patchtst", "xgboost", "nbeats", "nhits"]
    for name in model_names:
        try:
            uri = f"models:/{name}-m5/Production"
            if name == "xgboost":
                _models[name] = mlflow.xgboost.load_model(uri)
            else:
                _models[name] = mlflow.pytorch.load_model(uri)
            logger.info(f"  Loaded model: {name}")
        except Exception as e:
            logger.warning(f"  Could not load model '{name}': {e}")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Time Series Benchmark API",
    description="Forecast + XAI API for TFT, PatchTST, N-BEATS, N-HiTS, XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — returns loaded models."""
    return HealthResponse(
        status="ok",
        loaded_models=list(_models.keys()),
        version="1.0.0",
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List all available models."""
    return {"available": list(_models.keys())}


@app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
async def predict(request: PredictRequest):
    """
    Generate a point forecast for a given series.

    Args:
        request.series_id: Item/store identifier (e.g. "FOODS_3_090_CA_3").
        request.model: Model to use (default: "tft").
        request.horizon: Forecast horizon in days (default: 28).
        request.features: Optional dict of feature values for the forecast window.
    """
    model_name = request.model.lower()

    if model_name not in _models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded. Available: {list(_models.keys())}",
        )

    try:
        model = _models[model_name]
        # Build input features DataFrame
        X = _build_input(request.series_id, request.horizon, request.features)

        # Generate predictions
        if hasattr(model, "predict"):
            raw_preds = model.predict(X)
        else:
            raw_preds = np.zeros(request.horizon)

        # Extract forecast values
        if isinstance(raw_preds, pd.DataFrame):
            pred_col = [c for c in raw_preds.columns if c not in ["unique_id", "ds"]][0]
            forecasts = raw_preds[pred_col].tolist()
        else:
            forecasts = raw_preds.tolist()

        return PredictResponse(
            series_id=request.series_id,
            model=model_name,
            horizon=request.horizon,
            forecasts=forecasts,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["XAI"])
async def explain(request: ExplainRequest):
    """
    Return model explanations for a given series.

    For XGBoost: returns SHAP values.
    For TFT: returns variable importance + attention weights.
    For PatchTST: returns patch attention weights.
    """
    model_name = request.model.lower()

    if model_name not in _models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    try:
        model = _models[model_name]
        X = _build_input(request.series_id, request.horizon, request.features)

        explanation = {}

        if model_name == "xgboost":
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            explanation = {
                "type": "shap",
                "feature_names": list(X.columns),
                "shap_values": shap_vals.mean(axis=0).tolist(),
                "base_value": float(explainer.expected_value),
            }

        elif model_name == "tft":
            from src.explainability.tft_interpretability import TFTInterpreter
            # Simplified: return variable selection weights
            explanation = {
                "type": "variable_importance",
                "weights": {},  # populated by TFTInterpreter in full implementation
            }

        elif model_name == "patchtst":
            explanation = {
                "type": "patch_attention",
                "n_patches": 12,
                "attention_per_patch": [],  # populated in full implementation
            }

        return ExplainResponse(
            series_id=request.series_id,
            model=model_name,
            explanation=explanation,
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare", response_model=BenchmarkResponse, tags=["Benchmark"])
async def compare():
    """Return benchmark comparison table (all models, all metrics)."""
    global _benchmark_results
    if _benchmark_results is None:
        try:
            _benchmark_results = pd.read_csv("data/results/benchmark.csv")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Benchmark results not found. Run: make evaluate",
            )
    return BenchmarkResponse(results=_benchmark_results.to_dict(orient="records"))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_input(
    series_id: str,
    horizon: int,
    features: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build model input from series_id + optional features.
    In production: query feature store for precomputed features.
    """
    # Stub: in production, query Delta Lake / feature store
    n_rows = horizon
    default_features = {
        "lag_7": 10.0, "lag_28": 9.5,
        "roll_mean_7": 10.2, "roll_mean_28": 10.0,
        "roll_std_7": 2.1, "roll_std_28": 2.3,
        "sell_price": 2.5, "price_pct_change": 0.0, "price_norm_store": 0.0,
        "day_of_week": 1, "month": 6, "is_weekend": 0,
        "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
        "event_name_encoded": -1,
    }
    if features:
        default_features.update(features)

    return pd.DataFrame([default_features] * n_rows)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
