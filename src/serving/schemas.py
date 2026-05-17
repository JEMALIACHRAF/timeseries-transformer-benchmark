"""
Pydantic schemas for FastAPI request/response validation.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    series_id: str = Field(..., example="FOODS_3_090_CA_3")
    model: str = Field(default="tft", example="tft")
    horizon: int = Field(default=28, ge=1, le=56, example=28)
    features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional feature overrides. If None, fetched from feature store.",
        example={"sell_price": 2.5, "is_weekend": 0},
    )


class ExplainRequest(BaseModel):
    series_id: str = Field(..., example="FOODS_3_090_CA_3")
    model: str = Field(default="tft", example="tft")
    horizon: int = Field(default=28, ge=1, le=56)
    features: Optional[Dict[str, Any]] = None


# ── Response schemas ───────────────────────────────────────────────────────────


class PredictResponse(BaseModel):
    series_id: str
    model: str
    horizon: int
    forecasts: List[float] = Field(..., description="Point forecast per day, length = horizon")


class ExplainResponse(BaseModel):
    series_id: str
    model: str
    explanation: Dict[str, Any] = Field(
        ...,
        description=(
            "XAI output. Shape depends on model: "
            "SHAP dict for XGBoost, variable importance for TFT, "
            "patch attention for PatchTST."
        ),
    )


class BenchmarkResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(
        ..., description="List of dicts, one per model, with all metric values."
    )


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    version: str
