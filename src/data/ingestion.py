"""
Data ingestion module.
Handles loading M5 dataset locally or from Azure Data Lake Storage Gen2.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── M5 file names ─────────────────────────────────────────────────────────────
M5_FILES = {
    "sales_train": "sales_train_evaluation.csv",
    "calendar": "calendar.csv",
    "sell_prices": "sell_prices.csv",
    "sample_submission": "sample_submission.csv",
}


class M5DataLoader:
    """
    Loads and merges the M5 Forecasting dataset.

    Supports:
    - Local CSV files (data/raw/)
    - Azure Data Lake Storage Gen2 (via abfss:// paths on Databricks)
    """

    def __init__(self, raw_path: str = "data/raw/", use_adls: bool = False):
        self.raw_path = Path(raw_path)
        self.use_adls = use_adls

    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """Load all M5 raw files into a dict of DataFrames."""
        logger.info(f"Loading M5 raw data from {self.raw_path}")
        dfs = {}
        for key, filename in M5_FILES.items():
            path = self.raw_path / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing M5 file: {path}\n"
                    "Download with: kaggle competitions download -c m5-forecasting-accuracy"
                )
            dfs[key] = pd.read_csv(path)
            logger.info(f"  Loaded {key}: {dfs[key].shape}")
        return dfs

    def melt_sales(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Melt wide-format sales (d_1 ... d_1941) to long format.

        Returns:
            DataFrame with columns: [item_id, dept_id, cat_id, store_id, state_id, d, sales]
        """
        logger.info("Melting sales from wide to long format...")
        id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        d_cols = [c for c in sales_df.columns if c.startswith("d_")]

        df_long = sales_df[id_cols + d_cols].melt(
            id_vars=id_cols,
            value_vars=d_cols,
            var_name="d",
            value_name="sales",
        )
        logger.info(f"  Melted shape: {df_long.shape}")
        return df_long

    def merge_all(self) -> pd.DataFrame:
        """
        Full pipeline: load → melt → merge calendar + prices.

        Returns:
            Single enriched DataFrame ready for feature engineering.
        """
        dfs = self.load_raw()

        # Melt sales
        df = self.melt_sales(dfs["sales_train"])

        # Merge calendar (adds date, event_name, snap columns)
        logger.info("Merging calendar...")
        calendar = dfs["calendar"][
            ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
             "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
        ]
        df = df.merge(calendar, on="d", how="left")
        df["date"] = pd.to_datetime(df["date"])

        # Merge sell prices
        logger.info("Merging sell prices...")
        df = df.merge(
            dfs["sell_prices"],
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        # Encode event names as integer codes
        df["event_name_encoded"] = df["event_name_1"].astype("category").cat.codes

        logger.info(f"Final merged shape: {df.shape}")
        return df

    def load_from_adls(self, spark, adls_path: str) -> "pyspark.sql.DataFrame":
        """
        Load data from Azure Data Lake (used in Databricks notebooks).

        Args:
            spark: Active SparkSession.
            adls_path: Full abfss:// path.

        Returns:
            Spark DataFrame.
        """
        logger.info(f"Loading from ADLS: {adls_path}")
        return spark.read.format("delta").load(adls_path)

    def save_to_adls(
        self, spark_df, adls_path: str, mode: str = "overwrite"
    ) -> None:
        """Save Spark DataFrame as Delta table to ADLS."""
        logger.info(f"Saving to ADLS: {adls_path} (mode={mode})")
        spark_df.write.format("delta").mode(mode).save(adls_path)
        logger.info("  Save complete.")
