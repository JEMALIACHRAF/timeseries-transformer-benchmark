"""
Counterfactual / What-If analysis for time series models.

Answers questions like:
- "If the price had been 10% higher, what would the forecast be?"
- "What if there was a promotion this week?"
- "How does removing the SNAP benefit affect predicted demand?"

Works with any model that has a predict() interface.
"""
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CounterfactualAnalyzer:
    """
    Generate and visualize counterfactual forecasts.
    """

    def __init__(
        self,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        feature_names: List[str],
    ):
        """
        Args:
            predict_fn: Function that takes a feature DataFrame and returns predictions.
            feature_names: List of feature columns used by the model.
        """
        self.predict_fn = predict_fn
        self.feature_names = feature_names

    def what_if(
        self,
        X_base: pd.DataFrame,
        perturbations: Dict[str, Any],
        series_dates: Optional[pd.DatetimeIndex] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate counterfactual forecast by modifying input features.

        Args:
            X_base: Original input features (one row per forecast step).
            perturbations: Dict of {feature_name: new_value_or_multiplier}.
                           Values are applied as:
                           - float in (-1, 1]: treated as relative change (e.g., 0.1 = +10%)
                           - other: treated as absolute replacement
            series_dates: Optional dates for x-axis labeling.

        Returns:
            Dict with 'baseline' and 'counterfactual' prediction arrays.
        """
        # Baseline prediction
        y_baseline = self.predict_fn(X_base)

        # Apply perturbations
        X_counterfactual = X_base.copy()
        perturbation_labels = []

        for feature, value in perturbations.items():
            if feature not in X_counterfactual.columns:
                logger.warning(f"Feature '{feature}' not in columns — skipping.")
                continue

            if isinstance(value, float) and -1 < value <= 1 and value != 0:
                # Relative change
                X_counterfactual[feature] = X_base[feature] * (1 + value)
                sign = "+" if value > 0 else ""
                perturbation_labels.append(f"{feature} {sign}{value*100:.0f}%")
            else:
                # Absolute replacement
                X_counterfactual[feature] = value
                perturbation_labels.append(f"{feature} = {value}")

        y_counterfactual = self.predict_fn(X_counterfactual)

        # Compute impact
        impact = y_counterfactual - y_baseline
        relative_impact = (impact / (np.abs(y_baseline) + 1e-8)) * 100

        results = {
            "baseline": y_baseline,
            "counterfactual": y_counterfactual,
            "impact_absolute": impact,
            "impact_relative_pct": relative_impact,
        }

        logger.info(f"Counterfactual analysis: {perturbation_labels}")
        logger.info(f"  Mean absolute impact: {np.abs(impact).mean():.3f}")
        logger.info(f"  Mean relative impact: {relative_impact.mean():.1f}%")

        self._plot_what_if(
            y_baseline, y_counterfactual, impact,
            perturbation_labels, series_dates, output_path
        )

        return results

    def scenario_sweep(
        self,
        X_base: pd.DataFrame,
        feature: str,
        values: List[Any],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Sweep a feature across multiple values and compare forecasts.

        Args:
            X_base: Baseline features.
            feature: Feature to sweep.
            values: List of values to test.

        Returns:
            DataFrame with one column per scenario.
        """
        baseline = self.predict_fn(X_base)
        scenario_results = {"baseline": baseline.mean()}

        for val in values:
            X_modified = X_base.copy()
            X_modified[feature] = val
            preds = self.predict_fn(X_modified)
            scenario_results[f"{feature}={val}"] = preds.mean()

        results_df = pd.DataFrame([scenario_results])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = list(scenario_results.keys())
        means = list(scenario_results.values())
        colors = ["#90CAF9"] + ["#EF5350" if m < means[0] else "#66BB6A" for m in means[1:]]
        bars = ax.bar(labels, means, color=colors)
        ax.axhline(y=means[0], color="navy", linestyle="--", alpha=0.5, label="Baseline")
        ax.set_title(f"Forecast Sensitivity to '{feature}'", fontsize=13)
        ax.set_ylabel("Mean Predicted Sales")
        ax.set_xlabel(f"{feature} scenario")
        plt.xticks(rotation=30, ha="right")

        for bar, val in zip(bars[1:], means[1:]):
            diff = val - means[0]
            sign = "+" if diff >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{sign}{diff:.2f}", ha="center", va="bottom", fontsize=9,
                    color="green" if diff >= 0 else "red")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Scenario sweep saved → {output_path}")
        plt.show()

        return results_df

    def _plot_what_if(
        self,
        y_baseline: np.ndarray,
        y_counterfactual: np.ndarray,
        impact: np.ndarray,
        labels: List[str],
        dates: Optional[pd.DatetimeIndex],
        output_path: Optional[str],
    ) -> plt.Figure:
        """Plot baseline vs counterfactual forecast with impact panel."""
        x = dates if dates is not None else np.arange(len(y_baseline))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})

        # Forecast panel
        ax1.plot(x, y_baseline, label="Baseline", color="#1565C0", linewidth=2)
        ax1.plot(x, y_counterfactual, label="Counterfactual", color="#E53935",
                 linewidth=2, linestyle="--")
        ax1.fill_between(x, y_baseline, y_counterfactual,
                         where=y_counterfactual >= y_baseline,
                         alpha=0.15, color="green", label="Increase")
        ax1.fill_between(x, y_baseline, y_counterfactual,
                         where=y_counterfactual < y_baseline,
                         alpha=0.15, color="red", label="Decrease")
        ax1.set_title(
            f"What-If Analysis: {' | '.join(labels)}", fontsize=13, fontweight="bold"
        )
        ax1.set_ylabel("Predicted Sales")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Impact panel
        colors_impact = ["#66BB6A" if v >= 0 else "#EF5350" for v in impact]
        ax2.bar(x if dates is None else range(len(impact)), impact, color=colors_impact)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title("Impact (Counterfactual − Baseline)", fontsize=11)
        ax2.set_ylabel("Δ Sales")
        ax2.set_xlabel("Forecast Horizon (days)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"What-if plot saved → {output_path}")
        plt.show()

        return fig
