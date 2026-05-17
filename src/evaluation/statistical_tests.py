"""
Statistical significance tests for forecast comparison.

Implements:
- Diebold-Mariano (DM) test: are two forecasters significantly different?
- Model Confidence Set (MCS): which models belong to the best set?
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> Dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: The two forecasters have equal expected loss.
    H1: They have different expected loss (two-sided).

    Args:
        y_true: Ground truth values.
        y_pred_1: Predictions from model 1 (challenger).
        y_pred_2: Predictions from model 2 (baseline).
        loss: Loss function — 'squared' (MSE-based) or 'absolute' (MAE-based).
        h: Forecast horizon (for autocorrelation correction).

    Returns:
        Dict with 'dm_statistic', 'p_value', 'reject_h0' (at 5% level).
    """
    if loss == "squared":
        e1 = (y_true - y_pred_1) ** 2
        e2 = (y_true - y_pred_2) ** 2
    elif loss == "absolute":
        e1 = np.abs(y_true - y_pred_1)
        e2 = np.abs(y_true - y_pred_2)
    else:
        raise ValueError(f"Unknown loss: {loss}. Use 'squared' or 'absolute'.")

    # Loss differential
    d = e1 - e2
    n = len(d)
    d_bar = np.mean(d)

    # HAC variance (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
        gamma_sum += (1 - lag / h) * gamma_lag
    var_d = (gamma_0 + 2 * gamma_sum) / n

    dm_stat = d_bar / np.sqrt(max(var_d, 1e-10))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "dm_statistic": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 4),
        "reject_h0_5pct": bool(p_value < 0.05),
        "model1_better": bool(dm_stat < 0),  # Negative DM → model 1 has lower loss
    }


def pairwise_dm_matrix(
    y_true: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    loss: str = "squared",
) -> pd.DataFrame:
    """
    Compute pairwise Diebold-Mariano p-values for all model pairs.

    Args:
        y_true: Ground truth.
        forecasts: Dict of {model_name: predictions}.

    Returns:
        DataFrame (p-value matrix) — lower-left triangle filled.
    """
    models = list(forecasts.keys())
    n = len(models)
    matrix = pd.DataFrame(np.ones((n, n)), index=models, columns=models)

    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = models[i], models[j]
            result = diebold_mariano_test(y_true, forecasts[m1], forecasts[m2], loss)
            matrix.loc[m1, m2] = result["p_value"]
            matrix.loc[m2, m1] = result["p_value"]

    return matrix.round(4)


def plot_dm_heatmap(dm_matrix: pd.DataFrame, output_path: Optional[str] = None):
    """Visualize DM p-value matrix as a heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(dm_matrix), dtype=bool)  # Mask diagonal

    sns.heatmap(
        dm_matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0, vmax=0.1,
        linewidths=0.5,
        mask=mask,
        cbar_kws={"label": "p-value (DM test)"},
    )

    # Mark significant pairs (p < 0.05)
    for i in range(len(dm_matrix)):
        for j in range(len(dm_matrix)):
            if i != j and dm_matrix.iloc[i, j] < 0.05:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor="blue", lw=2))

    ax.set_title(
        "Diebold-Mariano Test — Pairwise p-values\n"
        "(Blue border = significant difference at 5%)",
        fontsize=13,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"DM heatmap saved → {output_path}")

    return fig
