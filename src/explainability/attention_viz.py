"""
Attention visualization for PatchTST.

PatchTST divides the time series into non-overlapping (or strided) patches,
then applies self-attention over them. This module visualizes:
- Patch-level attention matrices
- Most attended patches (temporal segments)
- Attention patterns across forecast horizon
"""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PatchAttentionVisualizer:
    """
    Visualize PatchTST attention patterns.
    """

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        input_size: int = 104,
    ):
        self.patch_len = patch_len
        self.stride = stride
        self.input_size = input_size
        self.n_patches = max(1, (input_size - patch_len) // stride + 1)

    def get_patch_time_ranges(self) -> List[Tuple[int, int]]:
        """Return (start, end) day offsets for each patch."""
        ranges = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = min(start + self.patch_len, self.input_size)
            ranges.append((self.input_size - end, self.input_size - start))  # Relative to now
        return ranges

    def plot_patch_attention(
        self,
        attention_weights: np.ndarray,
        series_id: str = "series",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Heatmap of patch-level attention.

        Args:
            attention_weights: Shape (n_heads, n_patches, n_patches).
            series_id: Label for plot.
        """
        patch_ranges = self.get_patch_time_ranges()
        labels = [f"t-{end}:t-{start}" for start, end in patch_ranges]

        n_heads = attention_weights.shape[0]
        avg_attn = attention_weights.mean(axis=0)  # (n_patches, n_patches)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Per-head average
        sns.heatmap(
            avg_attn,
            ax=axes[0],
            cmap="YlOrRd",
            xticklabels=labels,
            yticklabels=labels,
            annot=False,
        )
        axes[0].set_title(f"Average Attention ({n_heads} heads) — {series_id}", fontsize=12)
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0].tick_params(axis="y", labelsize=8)
        axes[0].set_xlabel("Key Patch (attended to)")
        axes[0].set_ylabel("Query Patch")

        # Aggregate attention per patch (column sum = how much each patch is attended to)
        total_attention = avg_attn.sum(axis=0)
        colors = plt.cm.YlOrRd(total_attention / total_attention.max())
        axes[1].bar(range(len(labels)), total_attention, color=colors)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_title("Total Attention per Patch\n(Which temporal segments matter most?)", fontsize=12)
        axes[1].set_xlabel("Patch (time segment)")
        axes[1].set_ylabel("Total Attention Weight")

        # Highlight most attended patch
        top_patch = np.argmax(total_attention)
        axes[1].patches[top_patch].set_edgecolor("blue")
        axes[1].patches[top_patch].set_linewidth(2.5)
        axes[1].annotate(
            "Most attended",
            xy=(top_patch, total_attention[top_patch]),
            xytext=(top_patch + 0.5, total_attention[top_patch] * 0.95),
            color="blue", fontsize=9, fontweight="bold",
        )

        plt.suptitle("PatchTST Attention Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Patch attention plot saved → {output_path}")

        return fig

    def plot_series_with_patches(
        self,
        series: np.ndarray,
        attention_weights: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Overlay original time series with patch attention highlights.
        Most-attended patches are shown with stronger color.
        """
        avg_attn = attention_weights.mean(axis=0).sum(axis=0)  # (n_patches,)
        norm_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(14, 5))

        x = dates if dates is not None else np.arange(len(series))
        ax.plot(x, series, color="#1565C0", linewidth=1.5, zorder=5)

        # Shade each patch by attention weight
        patch_ranges = self.get_patch_time_ranges()
        cmap = plt.cm.Reds

        for i, (start_offset, end_offset) in enumerate(patch_ranges):
            patch_start = max(0, len(series) - end_offset)
            patch_end = min(len(series), len(series) - start_offset)

            if dates is not None:
                x_start = dates[patch_start] if patch_start < len(dates) else dates[-1]
                x_end = dates[patch_end - 1] if patch_end <= len(dates) else dates[-1]
            else:
                x_start, x_end = patch_start, patch_end

            alpha = float(0.2 + norm_attn[i] * 0.6)
            ax.axvspan(x_start, x_end, alpha=alpha, color=cmap(norm_attn[i]), zorder=1)

        ax.set_title("Time Series with PatchTST Attention Overlay\n(Darker = Higher Attention)", fontsize=13)
        ax.set_xlabel("Date" if dates is not None else "Timestep")
        ax.set_ylabel("Sales")

        # Legend
        patches_legend = [
            mpatches.Patch(color=cmap(0.2), alpha=0.4, label="Low attention"),
            mpatches.Patch(color=cmap(0.8), alpha=0.8, label="High attention"),
        ]
        ax.legend(handles=patches_legend, loc="upper left")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig
