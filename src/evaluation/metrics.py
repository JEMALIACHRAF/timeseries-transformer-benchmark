"""
Evaluation metrics for time series forecasting.

Implements:
- RMSSE (Root Mean Squared Scaled Error) — M5 official metric
- MASE  (Mean Absolute Scaled Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)
- RMSE / MAE
- WQL   (Weighted Quantile Loss) — for probabilistic forecasts
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Root Mean Squared Scaled Error (official M5 metric).

    Scale is the naive seasonal forecast error on the training set.
    """
    n = len(y_train)
    # Denominator: MSE of seasonal naive on training set
    naive_errors = y_train[seasonality:] - y_train[:-seasonality]
    scale = np.mean(naive_errors ** 2) + 1e-8
    msse = np.mean((y_true - y_pred) ** 2) / scale
    return float(np.sqrt(msse))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error."""
    n = len(y_train)
    naive_mae = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])) + 1e-8
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def wql(
    y_true: np.ndarray,
    y_quantiles: Dict[float, np.ndarray],
) -> float:
    """
    Weighted Quantile Loss (for probabilistic forecasts).

    Args:
        y_true: Ground truth values.
        y_quantiles: Dict mapping quantile level (0-1) to predicted quantile array.

    Returns:
        Mean WQL across all quantiles.
    """
    losses = []
    for q, y_q in y_quantiles.items():
        errors = y_true - y_q
        loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
        losses.append(loss)
    return float(np.mean(losses))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 7,  # Weekly seasonality for daily M5 data
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.

    Returns:
        Dict with keys: rmsse, mase, smape, rmse, mae
    """
    return {
        "rmsse": rmsse(y_true, y_pred, y_train, seasonality),
        "mase":  mase(y_true, y_pred, y_train, seasonality),
        "smape": smape(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "mae":   mae(y_true, y_pred),
    }


def aggregate_series_metrics(
    results_df: pd.DataFrame,
    y_true_col: str = "sales",
    y_pred_col: str = "prediction",
    group_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute metrics per series and aggregate.

    Args:
        results_df: DataFrame with actual and predicted values.
        y_true_col: Column name for actuals.
        y_pred_col: Column name for predictions.
        group_cols: Grouping columns (e.g., ['dept_id', 'store_id']).

    Returns:
        Aggregated metrics DataFrame.
    """
    rows = []
    group_cols = group_cols or ["item_id", "store_id"]

    for keys, group in results_df.groupby(group_cols):
        y_true = group[y_true_col].values
        y_pred = group[y_pred_col].values

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row["rmse"] = rmse(y_true, y_pred)
        row["mae"] = mae(y_true, y_pred)
        row["smape"] = smape(y_true, y_pred)
        rows.append(row)

    return pd.DataFrame(rows)
