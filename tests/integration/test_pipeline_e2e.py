"""
End-to-end integration test for the full pipeline:
preprocessing → feature engineering → model training (XGBoost) → evaluation.

Uses a tiny synthetic dataset so no real data download is required.
"""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from pathlib import Path
import tempfile
import os

from src.data.feature_engineering import M5FeatureEngineer
from src.data.preprocessing import M5Preprocessor
from src.evaluation.metrics import compute_all_metrics
from src.models.baselines.xgboost_ts import XGBoostForecaster
from src.models.registry import get_model


@pytest.fixture
def synthetic_df():
    """
    Build a synthetic M5-like DataFrame (no file I/O required).
    3 series × 120 days.
    """
    np.random.seed(42)
    records = []
    for i in range(3):
        base = np.random.uniform(5, 15)
        for d in range(120):
            date = pd.Timestamp("2015-11-27") + pd.Timedelta(days=d)
            records.append({
                "item_id":  f"ITEM_{i}",
                "store_id": "CA_1",
                "dept_id":  "FOODS_1",
                "cat_id":   "FOODS",
                "state_id": "CA",
                "date":     date,
                "sales":    max(0.0, base + np.sin(d / 7) * 2 + np.random.normal(0, 1)),
                "sell_price":    2.5,
                "snap_CA":       int(d % 7 == 0),
                "snap_TX":       0,
                "snap_WI":       0,
                "event_name_encoded": -1,
                "wm_yr_wk": 11501 + d // 7,
            })
    return pd.DataFrame(records)


@pytest.fixture
def cfg_tmpdir(tmp_path):
    return OmegaConf.create({
        "data": {
            "raw_path":        str(tmp_path / "raw") + "/",
            "processed_path":  str(tmp_path / "processed") + "/",
            "features_path":   str(tmp_path / "features") + "/",
            "train_end": "2016-02-01",
            "val_end":   "2016-02-15",
            "test_end":  "2016-02-28",
            "horizon": 14,
            "freq": "D",
        },
        "features": {
            "lag_days": [1, 7],
            "rolling_windows": [7],
            "rolling_funcs": ["mean", "std"],
            "calendar_features": ["day_of_week", "month"],
            "include_price": True,
            "include_events": True,
            "include_snap": True,
        },
        "mlflow": {
            "tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "test-e2e",
            "register_best_model": False,
        },
    })


class TestFeatureEngineeringPipeline:
    def test_fe_output_has_features(self, synthetic_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(synthetic_df)
        assert "lag_7" in result.columns
        assert "roll_mean_7" in result.columns
        assert "is_weekend" in result.columns
        assert len(result) > 0

    def test_fe_no_nulls_after_dropna(self, synthetic_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(synthetic_df)
        assert result["lag_7"].isna().sum() == 0


class TestTemporalSplitPipeline:
    def test_splits_correct_dates(self, cfg_tmpdir, synthetic_df):
        preprocessor = M5Preprocessor(cfg_tmpdir)
        splits = preprocessor._temporal_split(synthetic_df)
        assert splits["train"]["date"].max() <= pd.Timestamp("2016-02-01")
        assert splits["val"]["date"].min()   >  pd.Timestamp("2016-02-01")
        assert splits["test"]["date"].min()  >  pd.Timestamp("2016-02-15")


class TestXGBoostE2E:
    def test_full_train_predict_cycle(self, synthetic_df):
        """Train XGBoost on synthetic data and verify predictions are non-negative."""
        fe = M5FeatureEngineer(lag_days=[1, 7], rolling_windows=[7], rolling_funcs=["mean"])
        df = fe.fit_transform(synthetic_df)

        train = df[df["date"] <= pd.Timestamp("2016-02-01")]
        test  = df[df["date"] >  pd.Timestamp("2016-02-01")]

        from src.data.feature_engineering import get_feature_columns
        feature_cols = get_feature_columns(lag_days=[1, 7], rolling_windows=[7],
                                           rolling_funcs=["mean"])
        available = [c for c in feature_cols if c in train.columns]

        model = XGBoostForecaster(
            n_estimators=50, feature_cols=available, horizon=14
        )
        model.fit(train)

        if len(test) > 0:
            preds = model.predict(test)
            assert len(preds) == len(test)
            assert (preds >= 0).all(), "Predictions should be non-negative"

    def test_metrics_run_on_predictions(self, synthetic_df):
        """Verify metrics compute without errors on model output."""
        fe = M5FeatureEngineer(lag_days=[1, 7], rolling_windows=[7], rolling_funcs=["mean"])
        df = fe.fit_transform(synthetic_df)

        y_true  = df["sales"].values[-14:]
        y_pred  = df["sales"].values[-14:] + np.random.normal(0, 0.5, 14)
        y_pred  = y_pred.clip(min=0)
        y_train = df["sales"].values[:-14]

        metrics = compute_all_metrics(y_true, y_pred, y_train)
        for key in ["rmsse", "mase", "smape", "rmse", "mae"]:
            assert key in metrics
            assert not np.isnan(metrics[key])
            assert metrics[key] >= 0.0


class TestModelRegistry:
    def test_get_model_xgboost(self):
        model = get_model("xgboost", horizon=28)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model")

    @pytest.mark.parametrize("model_name", ["xgboost", "nbeats", "nhits", "tft", "patchtst"])
    def test_all_models_instantiate(self, model_name):
        model = get_model(model_name, horizon=14)
        assert model is not None
