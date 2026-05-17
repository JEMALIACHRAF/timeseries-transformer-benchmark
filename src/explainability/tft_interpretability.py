"""
TFT native interpretability module.

The TFT architecture provides built-in interpretability via:
1. Variable Selection Networks (VSN) — which features matter
2. Multi-head Temporal Self-Attention — which past timesteps matter
3. Gated Residual Networks — feature interaction strengths

This module extracts and visualizes these components.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TFTInterpreter:
    """
    Extracts and visualizes TFT's native interpretability signals.
    """

    def __init__(self, tft_forecaster, feature_names: Optional[Dict] = None):
        """
        Args:
            tft_forecaster: Trained TFTForecaster instance.
            feature_names: Dict with keys 'hist', 'futr', 'stat' mapping to feature lists.
        """
        self.model = tft_forecaster
        self.feature_names = feature_names or {
            "hist": tft_forecaster.hist_exog_list,
            "futr": tft_forecaster.futr_exog_list,
            "stat": tft_forecaster.stat_exog_list,
        }

    def plot_variable_importance(
        self,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Bar chart of TFT variable selection weights.

        Answers: "Which input features does the model rely on?"
        """
        weights = self.model.get_variable_selection_weights()

        if not weights:
            logger.warning("No variable selection weights available.")
            return plt.figure()

        features = list(weights.keys())
        importances = list(weights.values())

        # Normalize to sum to 1
        total = sum(importances) + 1e-8
        importances = [v / total for v in importances]

        fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.4)))
        colors = ["#2196F3" if v > np.median(importances) else "#90CAF9" for v in importances]
        bars = ax.barh(features[::-1], importances[::-1], color=colors[::-1])

        ax.set_xlabel("Variable Selection Weight (normalized)", fontsize=12)
        ax.set_title("TFT Variable Importance — Input Feature Selection", fontsize=14)
        ax.axvline(x=1 / len(features), color="red", linestyle="--", alpha=0.5, label="Uniform baseline")
        ax.legend()

        # Add value labels
        for bar, val in zip(bars, importances[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Variable importance plot saved → {output_path}")

        return fig

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        series_id: str = "series",
        lookback_labels: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Heatmap of temporal attention weights.

        Answers: "Which past time steps does the model attend to?"

        Args:
            attention_weights: Array of shape (n_heads, seq_len, seq_len).
            series_id: Label for the plot title.
            lookback_labels: Optional list of timestep labels.
        """
        if attention_weights is None or len(attention_weights) == 0:
            logger.warning("No attention weights available.")
            return plt.figure()

        # Average across heads
        avg_attention = attention_weights.mean(axis=0)
        seq_len = avg_attention.shape[-1]

        if lookback_labels is None:
            lookback_labels = [f"t-{seq_len - i}" for i in range(seq_len)]

        fig, axes = plt.subplots(
            1, min(4, len(attention_weights)) + 1,
            figsize=(20, 5)
        )

        # Per-head attention
        for head_idx in range(min(4, len(attention_weights))):
            sns.heatmap(
                attention_weights[head_idx],
                ax=axes[head_idx],
                cmap="Blues",
                xticklabels=lookback_labels[::max(1, seq_len // 8)],
                yticklabels=lookback_labels[::max(1, seq_len // 8)],
                cbar=True,
            )
            axes[head_idx].set_title(f"Head {head_idx + 1}", fontsize=11)
            axes[head_idx].tick_params(axis="x", rotation=45, labelsize=7)
            axes[head_idx].tick_params(axis="y", labelsize=7)

        # Average across heads
        sns.heatmap(
            avg_attention,
            ax=axes[-1],
            cmap="Reds",
            xticklabels=lookback_labels[::max(1, seq_len // 8)],
            yticklabels=lookback_labels[::max(1, seq_len // 8)],
            cbar=True,
        )
        axes[-1].set_title("Average (all heads)", fontsize=11)
        axes[-1].tick_params(axis="x", rotation=45, labelsize=7)

        fig.suptitle(
            f"TFT Temporal Attention Heatmap — {series_id}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Attention heatmap saved → {output_path}")

        return fig

    def plot_temporal_pattern(
        self,
        attention_weights: np.ndarray,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Line chart of average attention per time step.
        Reveals which past windows (recent vs. seasonal) the model relies on.
        """
        avg_attention = attention_weights.mean(axis=(0, 1))  # Mean over heads and queries
        seq_len = len(avg_attention)
        timesteps = list(range(-seq_len, 0))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timesteps, avg_attention, color="#1565C0", linewidth=2)
        ax.fill_between(timesteps, avg_attention, alpha=0.3, color="#42A5F5")

        # Mark weekly seasonality peaks (t-7, t-14, t-21, t-28)
        for lag in [-7, -14, -21, -28]:
            if lag in timesteps:
                idx = timesteps.index(lag)
                ax.axvline(x=lag, color="red", linestyle="--", alpha=0.6, linewidth=1)
                ax.annotate(
                    f"t{lag}", xy=(lag, avg_attention[idx]),
                    xytext=(lag + 0.5, avg_attention[idx] * 1.05),
                    fontsize=9, color="red"
                )

        ax.set_xlabel("Lookback Timestep (relative to forecast origin)", fontsize=12)
        ax.set_ylabel("Average Attention Weight", fontsize=12)
        ax.set_title("TFT Temporal Attention Profile — Which Past Steps Matter Most?", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig

    @classmethod
    def from_mlflow(cls, model_uri: str) -> "TFTInterpreter":
        """Load TFT model from MLflow registry."""
        import mlflow.pytorch
        from src.models.deep.tft import TFTForecaster

        logger.info(f"Loading TFT from MLflow: {model_uri}")
        pytorch_model = mlflow.pytorch.load_model(model_uri)
        tft = TFTForecaster()
        tft.nf = type("NF", (), {"models": [pytorch_model]})()
        return cls(tft)
