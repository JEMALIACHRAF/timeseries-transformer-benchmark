"""
XGBoost time series forecaster.
Trains one XGBoost model across all series using tabular lag features.
"""
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from src.data.feature_engineering import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostForecaster:
    """
    Global XGBoost model for multi-series time series forecasting.

    Strategy: train a single model on all series using lag/rolling/calendar features.
    Item and store identity are encoded as target-encoded numerics.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        feature_cols: Optional[List[str]] = None,
        horizon: int = 28,
        **kwargs,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
            **kwargs,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_cols = feature_cols or get_feature_columns()
        self.horizon = horizon
        self.model: Optional[xgb.XGBRegressor] = None

    def _prepare_xy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract X features and y target from DataFrame."""
        available = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available)
        if missing:
            logger.warning(f"Missing features (will be ignored): {missing}")

        X = df[available].copy()
        y = df["sales"].copy()
        return X, y

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "XGBoostForecaster":
        """Train the model."""
        logger.info("Training XGBoost forecaster...")
        X_train, y_train = self._prepare_xy(train_df)

        eval_set = None
        if val_df is not None:
            X_val, y_val = self._prepare_xy(val_df)
            eval_set = [(X_val, y_val)]

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds if val_df else None,
            verbose=100,
        )
        logger.info(
            f"Training done. Best iteration: {self.model.best_iteration}"
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        X, _ = self._prepare_xy(df)
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)  # Sales cannot be negative

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        importance = self.model.feature_importances_
        available = [c for c in self.feature_cols if c in self.model.feature_names_in_]
        return (
            pd.DataFrame({"feature": available, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """Log params, metrics, and model to MLflow."""
        mlflow.log_params(self.params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(self.model, artifact_path="xgboost_model")
        logger.info("Logged XGBoost model to MLflow.")
