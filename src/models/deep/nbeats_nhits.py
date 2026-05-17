"""
N-BEATS and N-HiTS forecasters using NeuralForecast.

N-BEATS: Neural Basis Expansion Analysis for Time Series (Oreshkin et al., 2020)
  - Stack of fully connected blocks with residual connections
  - Basis expansion in trend + seasonality blocks

N-HiTS: Neural Hierarchical Interpolation for Time Series (Challu et al., 2023)
  - Extends N-BEATS with multi-rate data sampling
  - Better at capturing long-range dependencies efficiently
"""
from typing import Dict, List, Optional

import mlflow
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NBeatsForecaster:
    """N-BEATS forecaster wrapper."""

    def __init__(
        self,
        horizon: int = 28,
        input_size: int = 56,           # 2x horizon
        n_blocks: List[int] = [1, 1],   # per stack
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        max_epochs: int = 100,
        accelerator: str = "auto",
    ):
        self.horizon = horizon
        self.input_size = input_size
        self.n_blocks = n_blocks
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.nf: Optional[NeuralForecast] = None
        self._model_params: Dict = {}

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "NBeatsForecaster":
        logger.info("Training N-BEATS...")
        self._model_params = dict(
            h=self.horizon,
            input_size=self.input_size,
            n_harmonics=self.n_harmonics,
            n_polynomials=self.n_polynomials,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_steps=self.max_epochs * 100,
            accelerator=self.accelerator,
            enable_progress_bar=True,
        )
        model = NBEATS(**self._model_params)
        self.nf = NeuralForecast(models=[model], freq="D")
        self.nf.fit(df=train_df, val_size=len(val_df) if val_df else 0)
        logger.info("N-BEATS training complete.")
        return self

    def predict(self) -> pd.DataFrame:
        if self.nf is None:
            raise RuntimeError("Not trained.")
        preds = self.nf.predict()
        preds["NBEATS"] = preds["NBEATS"].clip(lower=0)
        return preds

    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        mlflow.log_params(self._model_params)
        mlflow.log_metrics(metrics)
        if self.nf:
            mlflow.pytorch.log_model(self.nf.models[0], artifact_path="nbeats_model")


class NHitsForecaster:
    """N-HiTS forecaster wrapper."""

    def __init__(
        self,
        horizon: int = 28,
        input_size: int = 56,
        n_blocks: List[int] = [1, 1, 1],
        n_pool_kernel_size: List[int] = [2, 2, 1],
        n_freq_downsample: List[int] = [4, 2, 1],
        learning_rate: float = 0.001,
        batch_size: int = 512,
        max_epochs: int = 100,
        accelerator: str = "auto",
    ):
        self.horizon = horizon
        self.input_size = input_size
        self.n_blocks = n_blocks
        self.n_pool_kernel_size = n_pool_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.nf: Optional[NeuralForecast] = None
        self._model_params: Dict = {}

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "NHitsForecaster":
        logger.info("Training N-HiTS...")
        self._model_params = dict(
            h=self.horizon,
            input_size=self.input_size,
            n_blocks=self.n_blocks,
            n_pool_kernel_size=self.n_pool_kernel_size,
            n_freq_downsample=self.n_freq_downsample,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_steps=self.max_epochs * 100,
            accelerator=self.accelerator,
            enable_progress_bar=True,
        )
        model = NHITS(**self._model_params)
        self.nf = NeuralForecast(models=[model], freq="D")
        self.nf.fit(df=train_df, val_size=len(val_df) if val_df else 0)
        logger.info("N-HiTS training complete.")
        return self

    def predict(self) -> pd.DataFrame:
        if self.nf is None:
            raise RuntimeError("Not trained.")
        preds = self.nf.predict()
        preds["NHITS"] = preds["NHITS"].clip(lower=0)
        return preds

    def log_to_mlflow(self, metrics: Dict[str, float]) -> None:
        mlflow.log_params(self._model_params)
        mlflow.log_metrics(metrics)
        if self.nf:
            mlflow.pytorch.log_model(self.nf.models[0], artifact_path="nhits_model")
