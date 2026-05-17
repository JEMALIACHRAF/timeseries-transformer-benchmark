"""
Integration tests for the FastAPI serving layer.
Uses TestClient to test all endpoints without running a server.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

# Mock model loading before importing the app
with patch("src.serving.api._load_models"):
    from src.serving.api import app, _models

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_models():
    """Inject a mock model for all tests."""
    mock_xgb = MagicMock()
    mock_xgb.predict.return_value = np.ones(28) * 5.0

    _models["xgboost"] = mock_xgb
    _models["tft"]     = MagicMock()
    _models["patchtst"]= MagicMock()
    yield
    _models.clear()


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "loaded_models" in data
        assert "version" in data
        assert data["status"] == "ok"

    def test_loaded_models_listed(self):
        response = client.get("/health")
        loaded = response.json()["loaded_models"]
        assert "xgboost" in loaded


class TestModelsEndpoint:
    def test_list_models_200(self):
        response = client.get("/models")
        assert response.status_code == 200

    def test_list_models_returns_dict(self):
        response = client.get("/models")
        data = response.json()
        assert "available" in data
        assert isinstance(data["available"], list)


class TestPredictEndpoint:
    def test_predict_xgboost_200(self):
        payload = {
            "series_id": "FOODS_3_090_CA_3",
            "model": "xgboost",
            "horizon": 28,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_schema(self):
        payload = {
            "series_id": "FOODS_3_090_CA_3",
            "model": "xgboost",
            "horizon": 7,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "series_id" in data
        assert "model" in data
        assert "horizon" in data
        assert "forecasts" in data

    def test_predict_forecast_length(self):
        payload = {
            "series_id": "FOODS_3_090_CA_3",
            "model": "xgboost",
            "horizon": 14,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert len(data["forecasts"]) == 14

    def test_predict_unknown_model_404(self):
        payload = {
            "series_id": "FOODS_3_090_CA_3",
            "model": "unknown_model",
            "horizon": 7,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 404

    def test_predict_horizon_validation(self):
        """Horizon > 56 should be rejected by Pydantic."""
        payload = {
            "series_id": "X",
            "model": "xgboost",
            "horizon": 100,  # exceeds max=56
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_with_feature_override(self):
        payload = {
            "series_id": "FOODS_3_090_CA_3",
            "model": "xgboost",
            "horizon": 7,
            "features": {"sell_price": 3.0, "is_weekend": 1},
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


class TestExplainEndpoint:
    def test_explain_200(self):
        with patch("shap.TreeExplainer") as mock_explainer:
            mock_explainer.return_value.shap_values.return_value = np.zeros((28, 10))
            mock_explainer.return_value.expected_value = 5.0

            payload = {
                "series_id": "FOODS_3_090_CA_3",
                "model": "xgboost",
                "horizon": 7,
            }
            response = client.post("/explain", json=payload)
            assert response.status_code == 200

    def test_explain_response_schema(self):
        payload = {
            "series_id": "FOODS_3_090_CA_3",
            "model": "tft",
            "horizon": 7,
        }
        response = client.post("/explain", json=payload)
        data = response.json()
        assert "series_id" in data
        assert "model" in data
        assert "explanation" in data

    def test_explain_unknown_model_404(self):
        payload = {
            "series_id": "X",
            "model": "nonexistent",
            "horizon": 7,
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 404
