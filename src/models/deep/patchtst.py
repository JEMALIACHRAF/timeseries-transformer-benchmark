"""
PatchTST forecaster wrapper using NeuralForecast.

PatchTST treats time series as sequences of patches (like image patches in ViT),
enabling efficient self-attention over longer lookback windows.

Reference: Nie et al. (2023) — "A Time Series is Worth 64 Words:
           Long-term Forecasting with Transformers"
"""
from typing import Dict, List, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PatchTSTForecaster:
    """
    PatchTST wrapper with attention visualization support.
    """

    def __init__(
        self,
        horizon: int = 28,
        input_size: int = 104,       # ~3.5 months lookback
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        learning_rate: float = 0.0001,
        batch_size: int = 512,
        max_epochs: int = 100,
        accelerator: str = "auto",
    ):
        self.horizon = horizon
        self.input_size = input_size
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator

        self.nf: Optional[NeuralForecast] = None
        self._model_params: Dict = {}

        # Derived: number of patches
        self.n_patches = max(1, (input_size - patch_len) // stride + 1)
        logger.info(
            f"PatchTST: input_size={input_size}, patch_len={patch_len}, "
            f"stride={stride} → n_patches={self.n_patches}"
        )

    def _build_model(self) -> PatchTST:
        self._model_params = dict(
            h=self.horizon,
            input_size=self.input_size,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_steps=self.max_epochs * 100,
            accelerator=self.accelerator,
            enable_progress_bar=True,
        )
        return PatchTST(**self._model_params)

    def fit(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> "PatchTSTForecaster":
        """
        Train PatchTST.

        Args:
            train_df: NeuralForecast format [unique_id, ds, y]
            val_df: Optional validation set.
        """
        logger.info(
            f"Training PatchTST | d_model={self.d_model} "
            f"| n_heads={self.n_heads} | patch_len={self.patch_len}"
        )
        model = self._build_model()
        self.nf = NeuralForecast(models=[model], freq="D")
        self.nf.fit(
            df=train_df,
            val_size=len(val_df) if val_df is not None else 0,
        )
        logger.info("PatchTST training complete.")
        return self

    def predict(self) -> pd.DataFrame:
        """Generate forecasts."""
        if self.nf is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        preds = self.nf.predict()
        preds["PatchTST"] = preds["PatchTST"].clip(lower=0)
        return preds

    def get_patch_attention(self) -> Optional[np.ndarray]:
        """
        Extract patch-level attention weights.

        Returns:
            Attention matrix of shape (n_heads, n_patches, n_patches)
            or None if not available.
        """
        if self.nf is None:
            return None
        model = self.nf.models[0]
        # Access the Transformer encoder's attention
        for layer in model.modules():
            if hasattr(layer, "attn_weights") and layer.attn_weights is not None:
                return layer.attn_weights.detach().cpu().numpy()
        return None

    def get_patch_labels(self) -> List[str]:
        """Return human-readable labels for each patch."""
        labels = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            labels.append(f"t-{self.input_size - start}:t-{self.input_size - end}")
        return labels

    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        """Log to MLflow."""
        mlflow.log_params(self._model_params)
        mlflow.log_metrics(metrics)
        if self.nf is not None:
            mlflow.pytorch.log_model(self.nf.models[0], artifact_path="patchtst_model")
        logger.info("PatchTST logged to MLflow.")
