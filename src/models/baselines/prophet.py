"""
Prophet forecaster wrapper for M5 dataset.
Uses multiprocessing to train one model per series in parallel.
"""
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Optional

import pandas as pd
from prophet import Prophet

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _train_single_series(
    group_key: tuple,
    df_group: pd.DataFrame,
    horizon: int,
    include_regressors: bool,
) -> Dict:
    """Train a Prophet model for one series. Used by multiprocessing."""
    item_id, store_id = group_key

    # Rename to Prophet convention
    df_prophet = df_group[["date", "sales"]].rename(
        columns={"date": "ds", "sales": "y"}
    )
    df_prophet = df_prophet.sort_values("ds")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )

    if include_regressors and "sell_price" in df_group.columns:
        model.add_regressor("sell_price")
        df_prophet["sell_price"] = df_group["sell_price"].values

    if include_regressors and "snap_CA" in df_group.columns:
        for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
            if snap_col in df_group.columns:
                model.add_regressor(snap_col)
                df_prophet[snap_col] = df_group[snap_col].values

    model.fit(df_prophet, iter=300)

    future = model.make_future_dataframe(periods=horizon, freq="D")

    # Fill regressors in future (use last known value)
    if include_regressors and "sell_price" in df_group.columns:
        last_price = df_group["sell_price"].iloc[-1]
        future["sell_price"] = last_price
        for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
            if snap_col in df_group.columns:
                future[snap_col] = 0  # Conservative: no SNAP in future

    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
    result["item_id"] = item_id
    result["store_id"] = store_id
    return result


class ProphetForecaster:
    """
    Parallel Prophet trainer for multi-series M5 data.

    On Databricks, use the PySpark UDF version in
    databricks/notebooks/04_baseline_models.py for true distributed training.
    """

    def __init__(
        self,
        horizon: int = 28,
        include_regressors: bool = True,
        n_workers: int = -1,
        max_series: Optional[int] = None,
    ):
        self.horizon = horizon
        self.include_regressors = include_regressors
        self.n_workers = n_workers if n_workers > 0 else mp.cpu_count()
        self.max_series = max_series  # For debugging; set None for full run
        self.forecasts_: Optional[pd.DataFrame] = None

    def fit_predict(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Train one Prophet model per series and return all forecasts.

        Args:
            train_df: DataFrame with columns [item_id, store_id, date, sales, ...]

        Returns:
            DataFrame with columns [item_id, store_id, ds, yhat, yhat_lower, yhat_upper]
        """
        groups = list(train_df.groupby(["item_id", "store_id"]))
        if self.max_series:
            groups = groups[: self.max_series]

        logger.info(
            f"Training Prophet for {len(groups)} series "
            f"using {self.n_workers} workers..."
        )

        worker_fn = partial(
            _train_single_series_wrapper,
            horizon=self.horizon,
            include_regressors=self.include_regressors,
        )

        with mp.Pool(self.n_workers) as pool:
            results = pool.map(worker_fn, groups)

        self.forecasts_ = pd.concat(results, ignore_index=True)
        logger.info(f"Prophet forecasts shape: {self.forecasts_.shape}")
        return self.forecasts_


def _train_single_series_wrapper(args, horizon: int, include_regressors: bool):
    """Wrapper for multiprocessing (must be picklable)."""
    group_key, df_group = args
    try:
        return _train_single_series(group_key, df_group, horizon, include_regressors)
    except Exception as e:
        item_id, store_id = group_key
        logger.warning(f"Prophet failed for {item_id}/{store_id}: {e}")
        return pd.DataFrame(
            {"item_id": [item_id], "store_id": [store_id],
             "ds": [None], "yhat": [0.0], "yhat_lower": [0.0], "yhat_upper": [0.0]}
        )
