"""
plotting.py
-----------
Plotting utilities for the SIGMA pipeline.

Provides reusable visualization functions for GMM analysis, UMAP plots,
and diagnostic visualizations used across multiple scripts.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scanpy as sc
from anndata import AnnData
from sklearn.mixture import GaussianMixture

from constants import GRAY_TO_RED, BLUE_GRAY_RED


# =============================================================================
# GMM Visualization
# =============================================================================


def plot_gmm_histogram(
    ax: plt.Axes,
    data: pd.Series,
    gmm: GaussianMixture,
    title: str,
    xlabel: str = "Mean expression per cell",
    ylabel: str = "Density",
    n_bins: int = 100,
) -> None:
    """Plot histogram with GMM fit overlay.

    Displays the data distribution as a histogram with the fitted GMM
    probability density function overlaid, including individual components.

    Args:
        ax: Matplotlib axes to plot on
        data: Data series to plot
        gmm: Fitted GaussianMixture model
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        n_bins: Number of histogram bins
    """
    ax.hist(data, bins=n_bins, alpha=0.6, density=True)

    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x))
    pdf_individual = gmm.predict_proba(x) * pdf[:, None]

    ax.plot(x, pdf, "-k", label="Total GMM")
    for i in range(gmm.n_components):
        ax.plot(x, pdf_individual[:, i], "--", label=f"Component {i + 1}")

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True)
    ax.legend()


def plot_probability_histogram(
    ax: plt.Axes,
    data: pd.Series,
    title: str,
    xlabel: str = "Probability",
    ylabel: str = "Number of Cells",
    n_bins: int = 100,
    color: str = "blue",
    log_scale: bool = True,
) -> None:
    """Plot histogram of probability distribution.

    Args:
        ax: Matplotlib axes to plot on
        data: Probability data to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        n_bins: Number of histogram bins
        color: Bar color
        log_scale: Whether to use log scale on y-axis
    """
    ax.hist(data, bins=n_bins, color=color, alpha=0.7)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--")


# =============================================================================
# UMAP Visualization
# =============================================================================


def plot_umap_or_placeholder(
    ax: plt.Axes,
    adata: AnnData,
    color: str,
    cmap,
    title: str,
    size: int = 50,
    placeholder_text: str = "UMAP not computed\n(Too few cells)",
) -> None:
    """Plot UMAP or placeholder if embeddings not available.

    Args:
        ax: Matplotlib axes to plot on
        adata: AnnData object (may or may not have X_umap)
        color: Column name in adata.obs to color by
        cmap: Colormap to use
        title: Plot title
        size: Point size
        placeholder_text: Text to display if UMAP not available
    """
    if "X_umap" in adata.obsm:
        sc.pl.umap(adata, color=color, cmap=cmap, size=size, ax=ax, show=False)
    else:
        ax.text(
            0.5, 0.5, placeholder_text,
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_title(title)


def plot_umap_with_score(
    ax: plt.Axes,
    adata: AnnData,
    score_column: str = "score",
    size: int = 50,
) -> None:
    """Plot UMAP colored by score with diverging colormap.

    Uses a TwoSlopeNorm centered at zero for the diverging colormap.

    Args:
        ax: Matplotlib axes to plot on
        adata: AnnData object
        score_column: Column name in adata.obs for score values
        size: Point size
    """
    if "X_umap" not in adata.obsm:
        ax.text(
            0.5, 0.5, "UMAP not computed\n(Too few cells)",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title(f"UMAP colored by {score_column}")
        return

    try:
        min_score = adata.obs[score_column].min()
        max_score = adata.obs[score_column].max()

        if min_score == max_score:
            sc.pl.umap(
                adata, color=score_column, cmap="gray",
                ax=ax, size=size, show=False
            )
        else:
            norm = mcolors.TwoSlopeNorm(vmin=min_score, vcenter=0, vmax=max_score)
            sc.pl.umap(
                adata, color=score_column, cmap=BLUE_GRAY_RED,
                norm=norm, ax=ax, size=size, show=False
            )
    except Exception:
        sc.pl.umap(
            adata, color=score_column, cmap=GRAY_TO_RED,
            ax=ax, size=size, show=False
        )

    ax.set_title(f"UMAP colored by {score_column}")


# =============================================================================
# Composite Plots for Target Identification
# =============================================================================


def save_target_plots(
    adata: AnnData,
    gmm: GaussianMixture,
    study_name: str,
    plot_folder: str,
) -> None:
    """Generate and save target cell diagnostic plots.

    Creates a 2x2 figure with:
    - UMAP colored by mean expression
    - GMM histogram fit
    - UMAP colored by target probability
    - Probability histogram

    Args:
        adata: AnnData object with target analysis results
        gmm: Fitted GMM for target genes
        study_name: Name of the study for filename
        plot_folder: Directory to save plots
    """
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, f"{study_name}_target.png")

    print("Generating and saving plots...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    plot_umap_or_placeholder(
        axes[0, 0], adata, "target_mean_expr", "viridis",
        "UMAP colored by target mean expression",
    )
    plot_umap_or_placeholder(
        axes[1, 0], adata, "proba_target", GRAY_TO_RED,
        f"UMAP colored by target probability\n(GMM with {gmm.n_components} components)",
    )

    plot_gmm_histogram(
        axes[0, 1], adata.obs["target_mean_expr"], gmm,
        "Distribution of target mean expression and GMM fit",
    )

    plot_probability_histogram(
        axes[1, 1], adata.obs["proba_target"],
        "Histogram of Target Probability",
        xlabel="Target Probability",
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Targets plots saved to: {plot_path}")
    plt.close(fig)


def save_exclude_plots(
    adata: AnnData,
    study_name: str,
    plot_folder: str,
    exclude_names: list,
    gmm_excludes: dict,
) -> None:
    """Generate and save exclusion gene diagnostic plots.

    Creates a figure with one row per exclusion category plus a score row.

    Args:
        adata: AnnData object with exclusion analysis results
        study_name: Name of the study for filename
        plot_folder: Directory to save plots
        exclude_names: List of exclusion category names
        gmm_excludes: Dict mapping category names to fitted GMMs
    """
    plot_path = os.path.join(plot_folder, f"{study_name}_exclude.png")
    n_rows = len(exclude_names) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5 * n_rows))

    # Plot each exclusion category
    for nrow, exclude in enumerate(exclude_names):
        plot_umap_or_placeholder(
            axes[nrow, 0], adata, f"{exclude}_mean_expr", "viridis",
            f"UMAP colored by {exclude} mean expression",
        )
        gmm = gmm_excludes[exclude]
        plot_gmm_histogram(
            axes[nrow, 1], adata.obs[f"{exclude}_mean_expr"], gmm,
            f"Distribution of {exclude} mean expression (GMM: {gmm.n_components} comp.)",
        )

    # Score plots
    nrow = len(exclude_names)
    plot_umap_with_score(axes[nrow, 0], adata, "score")
    plot_probability_histogram(
        axes[nrow, 1], adata.obs["score"],
        "Histogram of Score", xlabel="Score",
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Exclusion plots saved to: {plot_path}")
    plt.close(fig)
