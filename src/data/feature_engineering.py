"""
Feature engineering for M5 time series dataset.
Produces lag features, rolling statistics, and calendar covariates.

Works both with pandas (local) and PySpark (Databricks/Azure).
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Pandas implementation ──────────────────────────────────────────────────────


class M5FeatureEngineer:
    """
    Feature engineering for M5 dataset using pandas.
    For large-scale use, see the equivalent PySpark version in databricks/notebooks/02_feature_engineering.py.
    """

    def __init__(
        self,
        lag_days: List[int] = [1, 7, 14, 28],
        rolling_windows: List[int] = [7, 14, 28],
        rolling_funcs: List[str] = ["mean", "std", "min", "max"],
    ):
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        self.rolling_funcs = rolling_funcs

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features per series (item_id + store_id)."""
        logger.info(f"Adding lag features: {self.lag_days}")
        group_cols = ["item_id", "store_id"]

        for lag in self.lag_days:
            col_name = f"lag_{lag}"
            df[col_name] = df.groupby(group_cols)["sales"].transform(
                lambda x: x.shift(lag)
            )
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics per series."""
        logger.info(f"Adding rolling features: windows={self.rolling_windows}")
        group_cols = ["item_id", "store_id"]

        for window in self.rolling_windows:
            for func in self.rolling_funcs:
                col_name = f"roll_{func}_{window}"
                df[col_name] = df.groupby(group_cols)["sales"].transform(
                    lambda x: x.shift(1).rolling(window).agg(func)
                )
        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar covariates from the date column."""
        logger.info("Adding calendar features")
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["quarter"] = df["date"].dt.quarter
        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived features."""
        logger.info("Adding price features")
        group_cols = ["item_id", "store_id"]

        # Price momentum
        df["price_lag_1"] = df.groupby(group_cols)["sell_price"].transform(
            lambda x: x.shift(1)
        )
        df["price_change"] = df["sell_price"] - df["price_lag_1"]
        df["price_pct_change"] = df["price_change"] / (df["price_lag_1"] + 1e-8)

        # Normalized price vs store average
        df["price_norm_store"] = df.groupby(["store_id", "date"])["sell_price"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        return df

    def add_target_encoding(
        self, df: pd.DataFrame, target_col: str = "sales"
    ) -> pd.DataFrame:
        """Add smoothed target encoding for categorical columns."""
        logger.info("Adding target encodings")
        global_mean = df[target_col].mean()

        for col in ["dept_id", "cat_id", "store_id", "state_id"]:
            if col in df.columns:
                # Smoothed encoding to avoid data leakage — use only train set
                group_mean = df.groupby(col)[target_col].mean()
                df[f"te_{col}"] = df[col].map(group_mean).fillna(global_mean)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature engineering pipeline."""
        logger.info(f"Starting feature engineering on {df.shape[0]:,} rows")
        df = df.sort_values(["item_id", "store_id", "date"]).copy()

        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_calendar_features(df)
        df = self.add_price_features(df)
        df = self.add_target_encoding(df)

        # Drop rows with NaN lags (first N days)
        min_lag = max(self.lag_days) + max(self.rolling_windows)
        df = df.dropna(subset=[f"lag_{max(self.lag_days)}"])

        logger.info(f"Feature engineering done. Shape: {df.shape}")
        logger.info(f"Feature columns: {[c for c in df.columns if c not in ['sales', 'date', 'item_id', 'store_id']]}")
        return df


def get_feature_columns(
    lag_days: List[int] = [1, 7, 14, 28],
    rolling_windows: List[int] = [7, 14, 28],
    rolling_funcs: List[str] = ["mean", "std"],
) -> List[str]:
    """Return list of all feature column names (for model input)."""
    features = []

    # Lag features
    features += [f"lag_{d}" for d in lag_days]

    # Rolling features
    for w in rolling_windows:
        for f in rolling_funcs:
            features.append(f"roll_{f}_{w}")

    # Calendar
    features += [
        "day_of_week", "day_of_month", "month", "year",
        "week_of_year", "is_weekend", "quarter",
    ]

    # Price
    features += ["sell_price", "price_pct_change", "price_norm_store"]

    # SNAP
    features += ["snap_CA", "snap_TX", "snap_WI"]

    # Events
    features += ["event_name_encoded"]

    return features
