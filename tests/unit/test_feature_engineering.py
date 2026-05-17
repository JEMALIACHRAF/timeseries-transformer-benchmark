"""
Unit tests for feature engineering module.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import M5FeatureEngineer, get_feature_columns


@pytest.fixture
def sample_df():
    """Minimal M5-like DataFrame for testing."""
    n_series = 3
    n_days = 60
    records = []
    for i in range(n_series):
        item_id  = f"ITEM_{i}"
        store_id = f"CA_{i}"
        for d in range(n_days):
            records.append({
                "item_id":  item_id,
                "store_id": store_id,
                "dept_id":  "FOODS_1",
                "cat_id":   "FOODS",
                "state_id": "CA",
                "date":     pd.Timestamp("2016-01-01") + pd.Timedelta(days=d),
                "sales":    float(np.random.randint(0, 20)),
                "sell_price":    float(np.random.uniform(1, 5)),
                "snap_CA":       int(np.random.randint(0, 2)),
                "snap_TX":       0,
                "snap_WI":       0,
                "event_name_encoded": -1,
            })
    return pd.DataFrame(records)


class TestM5FeatureEngineer:
    def test_lag_features_created(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[1, 7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        assert "lag_1" in result.columns
        assert "lag_7" in result.columns

    def test_rolling_features_created(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7, 14], rolling_funcs=["mean", "std"])
        result = fe.fit_transform(sample_df)
        assert "roll_mean_7"  in result.columns
        assert "roll_std_7"   in result.columns
        assert "roll_mean_14" in result.columns

    def test_calendar_features_created(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        for col in ["day_of_week", "month", "is_weekend", "year"]:
            assert col in result.columns, f"Missing calendar feature: {col}"

    def test_is_weekend_binary(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_no_nan_in_lag_after_dropna(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[7, 28], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        # After dropna, the max-lag column should have no NaN
        assert result["lag_28"].isna().sum() == 0

    def test_row_count_reduced_after_dropna(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[28], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        assert len(result) < len(sample_df)

    def test_sales_non_negative(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        assert (result["sales"] >= 0).all()

    def test_price_pct_change_exists(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        assert "price_pct_change" in result.columns

    def test_all_series_preserved(self, sample_df):
        fe = M5FeatureEngineer(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        result = fe.fit_transform(sample_df)
        original_series = set(zip(sample_df["item_id"], sample_df["store_id"]))
        result_series   = set(zip(result["item_id"], result["store_id"]))
        assert original_series == result_series


class TestGetFeatureColumns:
    def test_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_contains_expected_features(self):
        cols = get_feature_columns(lag_days=[7], rolling_windows=[7], rolling_funcs=["mean"])
        assert "lag_7" in cols
        assert "roll_mean_7" in cols
        assert "day_of_week" in cols
        assert "is_weekend" in cols
        assert "sell_price" in cols

    def test_no_duplicates(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols))
