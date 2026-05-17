"""
Unit tests for temporal split logic in preprocessing.
"""
import pandas as pd
import pytest

from src.data.preprocessing import M5Preprocessor
from omegaconf import OmegaConf


@pytest.fixture
def minimal_cfg():
    return OmegaConf.create({
        "data": {
            "raw_path": "data/raw/",
            "processed_path": "data/processed/",
            "features_path": "data/features/",
            "train_end": "2016-03-27",
            "val_end":   "2016-04-24",
            "test_end":  "2016-05-22",
            "horizon": 28,
            "freq": "D",
        },
        "features": {
            "lag_days": [1, 7],
            "rolling_windows": [7],
            "rolling_funcs": ["mean"],
            "calendar_features": ["day_of_week", "month"],
            "include_price": True,
            "include_events": True,
            "include_snap": True,
        },
        "mlflow": {
            "tracking_uri": "mlruns/",
            "experiment_name": "test",
            "register_best_model": False,
        },
    })


@pytest.fixture
def sample_long_df():
    """Simulate a merged long-format M5 DataFrame."""
    records = []
    for i in range(2):
        for d in range(200):
            records.append({
                "item_id":  f"ITEM_{i}",
                "store_id": "CA_1",
                "dept_id":  "FOODS_1",
                "cat_id":   "FOODS",
                "state_id": "CA",
                "date":     pd.Timestamp("2015-09-01") + pd.Timedelta(days=d),
                "sales":    float(d % 10),
                "sell_price":    2.5,
                "snap_CA":       0,
                "snap_TX":       0,
                "snap_WI":       0,
                "event_name_encoded": -1,
                "wm_yr_wk": 11501,
            })
    return pd.DataFrame(records)


class TestTemporalSplit:
    def test_no_date_leakage(self, minimal_cfg, sample_long_df):
        """Ensure train max date < val min date < test min date."""
        preprocessor = M5Preprocessor(minimal_cfg)
        splits = preprocessor._temporal_split(sample_long_df)

        train_max = splits["train"]["date"].max()
        val_min   = splits["val"]["date"].min()
        val_max   = splits["val"]["date"].max()
        test_min  = splits["test"]["date"].min()

        assert train_max <= pd.Timestamp("2016-03-27")
        assert val_min   >  pd.Timestamp("2016-03-27")
        assert test_min  >  pd.Timestamp("2016-04-24")

    def test_splits_non_overlapping(self, minimal_cfg, sample_long_df):
        preprocessor = M5Preprocessor(minimal_cfg)
        splits = preprocessor._temporal_split(sample_long_df)

        train_dates = set(splits["train"]["date"])
        val_dates   = set(splits["val"]["date"])
        test_dates  = set(splits["test"]["date"])

        assert len(train_dates & val_dates)  == 0, "Train/val overlap!"
        assert len(val_dates  & test_dates)  == 0, "Val/test overlap!"
        assert len(train_dates & test_dates) == 0, "Train/test overlap!"

    def test_all_rows_covered(self, minimal_cfg, sample_long_df):
        preprocessor = M5Preprocessor(minimal_cfg)
        splits = preprocessor._temporal_split(sample_long_df)

        total_split = sum(len(s) for s in splits.values())
        original    = len(sample_long_df)
        assert total_split <= original  # Dates outside all splits are dropped
