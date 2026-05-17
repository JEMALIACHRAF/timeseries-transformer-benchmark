"""
Benchmark runner: trains and evaluates all models, produces comparison table.
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.models.registry import get_model, list_models
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """
    Orchestrates training and evaluation of all models for comparison.
    """

    def __init__(self, cfg, experiment_name: str = "ts-benchmark-m5"):
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.results: List[Dict] = []

        # Setup MLflow
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(experiment_name)

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run full benchmark across all (or selected) models.

        Returns:
            Summary DataFrame with metrics per model.
        """
        model_names = models or list_models()
        logger.info(f"Starting benchmark for models: {model_names}")

        for model_name in model_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Model: {model_name.upper()}")
            logger.info(f"{'='*60}")

            with mlflow.start_run(run_name=model_name):
                result = self._run_single_model(
                    model_name, train_df, val_df, test_df
                )
                self.results.append(result)
                mlflow.log_metrics({
                    k: v for k, v in result.items()
                    if isinstance(v, float)
                })

        return self.get_summary_table()

    def _run_single_model(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        """Train and evaluate one model."""
        t_start = time.time()
        result = {"model": model_name}

        try:
            model = get_model(model_name, horizon=self.cfg.data.horizon)

            # Train
            if hasattr(model, "fit_predict"):
                # Prophet: fit + predict in one call
                preds_df = model.fit_predict(train_df)
            else:
                model.fit(train_df, val_df)
                preds_df = model.predict() if hasattr(model, "predict") else None

            result["training_time_min"] = round((time.time() - t_start) / 60, 2)

            # Evaluate — use a sample of test series for speed
            if preds_df is not None:
                sample_series = test_df["item_id"].unique()[:100]
                test_sample = test_df[test_df["item_id"].isin(sample_series)]
                train_sample = train_df[train_df["item_id"].isin(sample_series)]

                y_true = test_sample["sales"].values
                # Align predictions with test set
                y_pred = self._align_predictions(preds_df, test_sample)
                y_train = train_sample["sales"].values

                metrics = compute_all_metrics(y_true, y_pred, y_train)
                result.update(metrics)
                logger.info(f"  Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error running {model_name}: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def _align_predictions(
        self, preds_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Align prediction DataFrame with test set."""
        # Stub — actual alignment depends on model output format
        if "yhat" in preds_df.columns:
            return preds_df["yhat"].values[: len(test_df)]
        for col in preds_df.columns:
            if col not in ["unique_id", "ds", "item_id", "store_id"]:
                return preds_df[col].values[: len(test_df)]
        return preds_df.iloc[:, -1].values[: len(test_df)]

    def get_summary_table(self) -> pd.DataFrame:
        """Return results as a formatted comparison table."""
        df = pd.DataFrame(self.results)
        numeric_cols = ["rmsse", "mase", "smape", "rmse", "mae", "training_time_min"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(4)

        if "rmsse" in df.columns:
            df = df.sort_values("rmsse")

        return df

    def save_results(self, output_path: str = "data/results/benchmark.csv") -> None:
        """Save benchmark results to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        table = self.get_summary_table()
        table.to_csv(output_path, index=False)
        logger.info(f"Benchmark results saved to {output_path}")
        print("\n" + table.to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    features_path = Path(cfg.data.features_path)

    train_df = pd.read_parquet(features_path / "m5_train.parquet")
    val_df   = pd.read_parquet(features_path / "m5_val.parquet")
    test_df  = pd.read_parquet(features_path / "m5_test.parquet")

    runner = BenchmarkRunner(cfg)
    results = runner.run(train_df, val_df, test_df, models=args.models)
    runner.save_results()
