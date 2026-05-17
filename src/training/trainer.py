"""
Unified trainer: orchestrates training for all models with MLflow tracking and Optuna tuning.
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import optuna
import pandas as pd
from omegaconf import DictConfig

from src.models.registry import get_model, list_models
from src.evaluation.metrics import compute_all_metrics
from src.utils.config import load_config, load_model_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    """
    Unified trainer for all forecasting models.

    Supports:
    - Standard training with MLflow autologging
    - Hyperparameter tuning via Optuna
    - Model registration to MLflow registry
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    # ── Main entry point ───────────────────────────────────────────────────────

    def train_all(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Train all (or selected) models and return their metrics."""
        model_names = models or list(self.cfg.models.selected)
        all_results = {}

        for model_name in model_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Training: {model_name.upper()}")
            logger.info(f"{'='*60}")
            result = self.train_single(model_name, train_df, val_df, test_df)
            all_results[model_name] = result

        logger.info("\n✅ All models trained.")
        self._print_leaderboard(all_results)
        return all_results

    def train_single(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tune: bool = False,
    ) -> Dict:
        """Train a single model, optionally with hyperparameter tuning."""
        # Load model-specific config if available
        try:
            model_cfg = load_model_config(model_name)
            hp = dict(model_cfg.hyperparameters)
        except FileNotFoundError:
            hp = {}

        if tune:
            logger.info(f"Running Optuna tuning for {model_name}...")
            hp = self._tune(model_name, train_df, val_df, model_cfg)

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_params({f"hp_{k}": v for k, v in hp.items()
                               if not isinstance(v, (list, dict))})

            t_start = time.time()

            # Instantiate model with hyperparameters
            model = get_model(model_name, horizon=self.cfg.data.horizon, **hp)

            # Train
            if hasattr(model, "fit_predict"):
                # Prophet: trains and predicts in one step
                preds_df = model.fit_predict(train_df)
                train_metrics = {}
            else:
                model.fit(train_df, val_df)
                preds_df = None
                train_metrics = {}

            training_time = round((time.time() - t_start) / 60, 2)
            mlflow.log_metric("training_time_min", training_time)
            logger.info(f"Training time: {training_time} min")

            # Evaluate on test set
            test_metrics = self._evaluate(model, test_df, train_df, preds_df)
            mlflow.log_metrics(test_metrics)
            logger.info(f"Test metrics: {test_metrics}")

            # Log model artifact
            self._log_model(model, model_name)

            # Register best model
            if self.cfg.mlflow.register_best_model:
                self._register_model(model_name)

            return {
                "training_time_min": training_time,
                **test_metrics,
            }

    # ── Hyperparameter tuning ──────────────────────────────────────────────────

    def _tune(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_cfg: DictConfig,
    ) -> Dict:
        """Run Optuna hyperparameter search."""
        search_space = dict(model_cfg.optuna.search_space)
        n_trials = model_cfg.optuna.n_trials
        direction = model_cfg.optuna.direction

        def objective(trial: optuna.Trial) -> float:
            hp = {}
            for param, bounds in search_space.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    if all(isinstance(b, int) for b in bounds):
                        hp[param] = trial.suggest_int(param, bounds[0], bounds[1])
                    else:
                        hp[param] = trial.suggest_float(param, bounds[0], bounds[1], log=True)
                elif isinstance(bounds, list):
                    hp[param] = trial.suggest_categorical(param, bounds)

            with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
                mlflow.log_params(hp)
                model = get_model(model_name, horizon=self.cfg.data.horizon, **hp)
                model.fit(train_df, val_df)
                metrics = self._evaluate(model, val_df, train_df)
                score = metrics.get("rmsse", metrics.get("rmse", 999))
                mlflow.log_metric("val_rmsse", score)
                return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        logger.info(f"Best trial: {study.best_trial.params}")
        logger.info(f"Best value: {study.best_value:.4f}")
        return study.best_trial.params

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        model,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        preds_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test set and return metrics dict."""
        try:
            sample_ids = test_df["item_id"].unique()[:200]
            test_sample = test_df[test_df["item_id"].isin(sample_ids)]
            train_sample = train_df[train_df["item_id"].isin(sample_ids)]

            y_true = test_sample["sales"].values

            if preds_df is not None:
                # Prophet / fit_predict style
                pred_cols = [c for c in preds_df.columns
                             if c not in ["unique_id", "ds", "item_id", "store_id"]]
                y_pred = preds_df[pred_cols[0]].values[: len(y_true)]
            elif hasattr(model, "predict"):
                raw = model.predict(test_sample)
                y_pred = raw if isinstance(raw, type(y_true)) else raw.values[: len(y_true)]
            else:
                return {}

            y_train = train_sample["sales"].values
            metrics = compute_all_metrics(y_true, y_pred, y_train)
            return {f"test_{k}": round(v, 4) for k, v in metrics.items()}

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {}

    # ── MLflow model logging ───────────────────────────────────────────────────

    def _log_model(self, model, model_name: str) -> None:
        """Log model artifact to MLflow."""
        try:
            if model_name == "xgboost" and hasattr(model, "model"):
                mlflow.xgboost.log_model(model.model, artifact_path="model")
            elif hasattr(model, "nf") and model.nf is not None:
                mlflow.pytorch.log_model(model.nf.models[0], artifact_path="model")
        except Exception as e:
            logger.warning(f"Could not log model artifact: {e}")

    def _register_model(self, model_name: str) -> None:
        """Register model to MLflow Model Registry."""
        try:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, name=f"{model_name}-m5")
            logger.info(f"  Registered: {model_name}-m5")
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

    # ── Reporting ──────────────────────────────────────────────────────────────

    def _print_leaderboard(self, results: Dict[str, Dict]) -> None:
        """Print a formatted leaderboard table."""
        rows = []
        for name, metrics in results.items():
            row = {"model": name}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        if "test_rmsse" in df.columns:
            df = df.sort_values("test_rmsse")

        print("\n" + "=" * 70)
        print("  BENCHMARK LEADERBOARD")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train time series models")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models to train. Default: all.")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features_path = Path(cfg.data.features_path)

    train_df = pd.read_parquet(features_path / "m5_train.parquet")
    val_df   = pd.read_parquet(features_path / "m5_val.parquet")
    test_df  = pd.read_parquet(features_path / "m5_test.parquet")

    trainer = Trainer(cfg)
    trainer.train_all(train_df, val_df, test_df, models=args.models)
