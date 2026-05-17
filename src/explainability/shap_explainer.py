"""
SHAP-based explainability for XGBoost (and other tree models).

Provides:
- Summary plot (global feature importance)
- Waterfall plot (single prediction explanation)
- Force plot (interactive)
- Dependence plot (feature interaction)
- Counterfactual analysis
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """SHAP explainer for XGBoost/tree-based forecasters."""

    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: Trained XGBoost model (or any tree model compatible with shap.TreeExplainer).
            feature_names: List of feature column names.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values_cache: Optional[np.ndarray] = None

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a dataset."""
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values_cache = self.explainer.shap_values(X)
        logger.info("SHAP computation complete.")
        return self.shap_values_cache

    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        SHAP summary (beeswarm) plot — global feature importance.

        Shows: which features matter most AND how they affect predictions.
        """
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Feature Importance — XGBoost Forecaster", fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved → {output_path}")

        return fig

    def plot_waterfall(
        self,
        X_single: pd.Series,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Waterfall plot for a single prediction.

        Shows exactly HOW each feature pushes the prediction above/below the baseline.
        """
        shap_values = self.explainer(X_single.to_frame().T)
        explanation = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=X_single.values,
            feature_names=self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title("SHAP Waterfall — Single Prediction Explanation", fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: str = "auto",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Dependence plot: how one feature affects predictions,
        colored by interaction with another feature.
        """
        shap_values = self.compute_shap_values(X)
        feat_idx = self.feature_names.index(feature)

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )
        ax.set_title(f"SHAP Dependence: {feature}", fontsize=13)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    def get_top_features(
        self, X: pd.DataFrame, n: int = 10
    ) -> pd.DataFrame:
        """
        Return top N features by mean absolute SHAP value.
        """
        shap_values = self.compute_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    @classmethod
    def load(cls, model_uri: str, feature_names: List[str]) -> "ShapExplainer":
        """Load model from MLflow and create explainer."""
        import mlflow.xgboost
        model = mlflow.xgboost.load_model(model_uri)
        return cls(model.get_booster(), feature_names)
