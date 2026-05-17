"""
Temporal Fusion Transformer (TFT) wrapper using NeuralForecast.

The TFT is the flagship deep learning model in this benchmark:
- Combines LSTM encoder, multi-head attention, and gating mechanisms
- Natively interpretable: variable importance + temporal attention
- Handles static, historical, and future covariates

Reference: Lim et al. (2021) — "Temporal Fusion Transformers for
           Interpretable Multi-horizon Time Series Forecasting"
"""
from typing import Dict, List, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TFTForecaster:
    """
    TFT wrapper with MLflow tracking and interpretability hooks.
    """

    def __init__(
        self,
        horizon: int = 28,
        input_size: int = 28,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        max_epochs: int = 100,
        val_check_steps: int = 100,
        hist_exog_list: Optional[List[str]] = None,
        futr_exog_list: Optional[List[str]] = None,
        stat_exog_list: Optional[List[str]] = None,
        accelerator: str = "auto",
    ):
        self.horizon = horizon
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_check_steps = val_check_steps
        self.accelerator = accelerator

        # Covariates — set defaults if not provided
        self.hist_exog_list = hist_exog_list or [
            "sell_price", "lag_7", "lag_28",
            "roll_mean_7", "roll_mean_28", "roll_std_7",
        ]
        self.futr_exog_list = futr_exog_list or [
            "day_of_week", "month", "is_weekend",
            "snap_CA", "snap_TX", "snap_WI", "event_name_encoded",
        ]
        self.stat_exog_list = stat_exog_list or []

        self.nf: Optional[NeuralForecast] = None
        self._model_params: Dict = {}

    def _build_model(self) -> TFT:
        """Instantiate TFT model with current hyperparameters."""
        self._model_params = dict(
            h=self.horizon,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_steps=self.max_epochs * 100,
            val_check_steps=self.val_check_steps,
            hist_exog_list=self.hist_exog_list,
            futr_exog_list=self.futr_exog_list,
            stat_exog_list=self.stat_exog_list,
            accelerator=self.accelerator,
            enable_progress_bar=True,
        )
        return TFT(**self._model_params)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "TFTForecaster":
        """
        Fit TFT model.

        Args:
            train_df: DataFrame in NeuralForecast format:
                      [unique_id, ds, y, *hist_exog, *futr_exog, *stat_exog]
            val_df: Optional validation set.
        """
        logger.info(
            f"Training TFT | hidden_size={self.hidden_size} "
            f"| heads={self.num_attention_heads} | epochs={self.max_epochs}"
        )
        model = self._build_model()
        self.nf = NeuralForecast(models=[model], freq="D")
        self.nf.fit(df=train_df, val_size=len(val_df) if val_df is not None else 0)
        logger.info("TFT training complete.")
        return self

    def predict(self, futr_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            futr_df: Future exogenous covariates (required if futr_exog_list is non-empty).

        Returns:
            DataFrame with columns [unique_id, ds, TFT]
        """
        if self.nf is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        preds = self.nf.predict(futr_df=futr_df)
        preds["TFT"] = preds["TFT"].clip(lower=0)
        return preds

    def get_attention_weights(self, series_id: str) -> Dict[str, np.ndarray]:
        """
        Extract temporal attention weights for a specific series.

        Returns:
            Dict with 'encoder_attention' and 'decoder_attention' arrays.
            Shape: (n_heads, seq_len, seq_len)
        """
        if self.nf is None:
            raise RuntimeError("Model not trained.")
        # Access internal PyTorch model
        tft_model = self.nf.models[0]
        # The TFT stores attention weights after the last forward pass
        attention = {}
        if hasattr(tft_model, "encoder_attention_weights"):
            attention["encoder"] = tft_model.encoder_attention_weights.detach().cpu().numpy()
        if hasattr(tft_model, "decoder_attention_weights"):
            attention["decoder"] = tft_model.decoder_attention_weights.detach().cpu().numpy()
        return attention

    def get_variable_selection_weights(self) -> Dict[str, float]:
        """
        Return TFT variable selection weights (input feature importance).

        Returns:
            Dict mapping feature name → importance weight.
        """
        if self.nf is None:
            raise RuntimeError("Model not trained.")
        tft_model = self.nf.models[0]

        weights = {}
        if hasattr(tft_model, "hist_encoder_variable_selection"):
            vsn = tft_model.hist_encoder_variable_selection
            if hasattr(vsn, "flattened_grn"):
                w = vsn.flattened_grn.detach().cpu().numpy()
                for i, feat in enumerate(self.hist_exog_list):
                    weights[feat] = float(w[i]) if i < len(w) else 0.0

        if hasattr(tft_model, "futr_encoder_variable_selection"):
            vsn = tft_model.futr_encoder_variable_selection
            if hasattr(vsn, "flattened_grn"):
                w = vsn.flattened_grn.detach().cpu().numpy()
                for i, feat in enumerate(self.futr_exog_list):
                    weights[feat] = float(w[i]) if i < len(w) else 0.0

        return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """Log hyperparameters, metrics, and model artifact to MLflow."""
        mlflow.log_params(self._model_params)
        mlflow.log_metrics(metrics)
        # Log PyTorch model
        if self.nf is not None:
            mlflow.pytorch.log_model(self.nf.models[0], artifact_path="tft_model")
        logger.info("TFT logged to MLflow.")
