"""
Preprocessing pipeline: load → merge → feature engineer → train/val/test split → save.
"""
import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from omegaconf import DictConfig

from src.data.feature_engineering import M5FeatureEngineer
from src.data.ingestion import M5DataLoader
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class M5Preprocessor:
    """Full preprocessing pipeline for M5 dataset."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.loader = M5DataLoader(raw_path=cfg.data.raw_path)
        self.engineer = M5FeatureEngineer(
            lag_days=list(cfg.features.lag_days),
            rolling_windows=list(cfg.features.rolling_windows),
            rolling_funcs=list(cfg.features.rolling_funcs),
        )

    def run(self) -> Dict[str, pd.DataFrame]:
        """Execute full preprocessing pipeline."""
        # 1. Load and merge raw data
        df = self.loader.merge_all()

        # 2. Feature engineering
        df = self.engineer.fit_transform(df)

        # 3. Train / val / test splits (temporal)
        splits = self._temporal_split(df)

        # 4. Save processed data
        self._save(df, splits)

        return splits

    def _temporal_split(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Split by date — no data leakage."""
        train_end = pd.Timestamp(self.cfg.data.train_end)
        val_end   = pd.Timestamp(self.cfg.data.val_end)
        test_end  = pd.Timestamp(self.cfg.data.test_end)

        train = df[df["date"] <= train_end]
        val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
        test  = df[(df["date"] > val_end)   & (df["date"] <= test_end)]

        logger.info(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")
        logger.info(
            f"Train end: {train['date'].max()} | "
            f"Val end: {val['date'].max()} | "
            f"Test end: {test['date'].max()}"
        )
        return {"train": train, "val": val, "test": test}

    def _save(self, df: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> None:
        """Save processed files to disk (parquet format)."""
        processed_path = Path(self.cfg.data.processed_path)
        features_path  = Path(self.cfg.data.features_path)
        processed_path.mkdir(parents=True, exist_ok=True)
        features_path.mkdir(parents=True, exist_ok=True)

        # Save full processed
        out = processed_path / "m5_processed.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Saved full processed dataset → {out}")

        # Save splits
        for split_name, split_df in splits.items():
            out = features_path / f"m5_{split_name}.parquet"
            split_df.to_parquet(out, index=False)
            logger.info(f"Saved {split_name} split → {out}")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M5 preprocessing pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/pipeline_config.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocessor = M5Preprocessor(cfg)
    splits = preprocessor.run()
    logger.info("✅ Preprocessing complete.")
