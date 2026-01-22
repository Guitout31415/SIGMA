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
# Plots for target
# =============================================================================


def plot_gmm_fit_hist(ax: plt.Axes, adata: AnnData, gmm: GaussianMixture, mean_expr_col: str) -> None:
    ax.hist(adata.obs[mean_expr_col], bins=100, alpha=0.6, density=True)
    ax.set(
        title="Distribution of target mean expression and GMM fit",
        xlabel="Mean expression per cell",
        ylabel="Density"
    )
    ax.grid(True)

    x = np.linspace(adata.obs[mean_expr_col].min(), adata.obs[mean_expr_col].max(), 1000).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x))
    pdf_individual = gmm.predict_proba(x) * pdf[:, None]
    
    ax.plot(x, pdf, '-k', label='Total GMM')
    for i in range(gmm.n_components):
        ax.plot(x, pdf_individual[:, i], '--', label=f'Component {i+1}')
    ax.legend()

def plot_target_figures(
    adata: AnnData,
    gmm_target: GaussianMixture,
    plot_folder: str,
    study_name: str,
) -> None:
    """Generate and save target diagnostic plots.

    Args:
        adata: AnnData object with target analysis results
        gmm_target: Fitted GMM for target genes
        plot_folder: Directory to save plots
        study_name: Name of the study for filename
    """
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, f"{study_name}_target.png")

    try:
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    except:
        pass
    
    fig, ax = plt.subplots(2,2,figsize=(20, 10))
    
    if 'X_umap' in adata.obsm:
        # Plot UMAP colored by target mean expression
        sc.pl.umap(
            adata, color="target_mean_expr", cmap="viridis", size=50,
            ax=ax[0,0], title="UMAP colored by target mean expression")

        # Plot UMAP colored by probability of being a target cell
        sc.pl.umap(
            adata, color="proba_target", cmap=GRAY_TO_RED, size=50,
            ax=ax[1, 0], show=False,
            title=f"UMAP colored by probability of being a target cell\n(Gaussian Mixture Model with {gmm_target.n_components} components)")
    else:
        ax[0, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=ax[0, 0].transAxes)
        ax[0, 0].set_title("UMAP colored by target mean expression")
        ax[1, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=ax[1, 0].transAxes)
        ax[1, 0].set_title(f"UMAP colored by probability of being a target cell\n(Gaussian Mixture Model with {gmm_target.n_components} components)")
    
    # Plot GMM histogram fit
    plot_gmm_fit_hist(ax[0,1], adata, gmm_target, "target_mean_expr")
    # Ensure target_indices is iterable
    target_indices = adata.uns.get("target_indices", [])
    if isinstance(target_indices, (int, np.integer)):
        target_indices = [target_indices]
    elif isinstance(target_indices, np.ndarray):
        target_indices = target_indices.tolist()

    # Update legend to mark target components with *
    handles, labels = ax[0,1].get_legend_handles_labels()
    new_labels = []
    for i, label in enumerate(labels):
        if i == 0:
            new_labels.append(label)
            continue
        if i-1 in target_indices:
            new_labels.append(f"{label}*")
        else:
            new_labels.append(label)
    
    ax[0,1].legend(handles, new_labels)

    # Plot histogram of target probabilities
    ax[1,1].hist(adata.obs['proba_target'], bins=100)
    ax[1,1].set(
        title="Histogram of Target Probability",
        xlabel="Target Probability",
        ylabel="Number of Cells"
    )
       
    ax[1,1].set_yscale('log')
    ax[1,1].grid(axis='y', linestyle='--')

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Target plots saved to: {plot_path}")


# =============================================================================
# Plots for excludes
# =============================================================================


def plot_exclude_figures(
    adata: AnnData,
    gmm_exclude: dict,
    plot_folder: str,
    study_name: str,
) -> None:
    """Generate and save exclusion diagnostic plots.

    Args:
        adata: AnnData object with exclusion analysis results
        gmm_exclude: Fitted GMM for exclusion genes
        plot_folder: Directory to save plots
        study_name: Name of the study for filename
    """
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, f"{study_name}_exclude.png")

    try:
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    except:
        pass

    nrows = len(gmm_exclude.keys())+1
    fig, ax = plt.subplots(nrows,2, figsize=(20, 5*nrows))

    nrow = 0
    for category, gmm in gmm_exclude.items():
        # Plot UMAP colored by exclusion mean expression
        if 'X_umap' in adata.obsm:
            sc.pl.umap(
                adata, color=f"exclude_mean_expr_{category}", cmap="viridis", size=50,
                ax=ax[nrow,0], title=f"UMAP colored by {category} exclusion mean expression")
        else:
            ax[nrow, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=ax[nrow, 0].transAxes)
            ax[nrow, 0].set_title(f"UMAP colored by {category} exclusion mean expression")
        
        # Plot GMM histogram fit
        plot_gmm_fit_hist(ax[nrow, 1], adata, gmm, f"exclude_mean_expr_{category}")

        target_indices = adata.uns.get(f"exclude_indices_{category}", [])
        if isinstance(target_indices, (int, np.integer)):
            target_indices = [target_indices]
        elif isinstance(target_indices, np.ndarray):
            target_indices = target_indices.tolist()

        # Update legend to mark target components with *
        handles, labels = ax[nrow,1].get_legend_handles_labels()
        new_labels = []
        for i, label in enumerate(labels):
            if i == 0:
                new_labels.append(label)
                continue
            if i-1 in target_indices:
                new_labels.append(f"{label}*")
            else:
                new_labels.append(label)
        
        ax[nrow,1].legend(handles, new_labels)

        nrow += 1
    
    # Plot UMAP colored by score
    if 'X_umap' in adata.obsm:
        try:
            min_score = adata.obs["score"].min()
            max_score = adata.obs["score"].max()
            if min_score == max_score:
                sc.pl.umap(adata, color="score", cmap='gray', ax=ax[nrow, 0], size=50, show=False)
            else:
                norm_cmap = mcolors.TwoSlopeNorm(vmin=min_score, vcenter=0, vmax=max_score)
                sc.pl.umap(adata, color="score", cmap=BLUE_GRAY_RED, norm=norm_cmap, ax=ax[nrow, 0], size=50, show=False)
        except:
            sc.pl.umap(adata, color="score", cmap=GRAY_TO_RED, ax=ax[nrow, 0], size=50, show=False)
    else:
        ax[nrow, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=ax[nrow, 0].transAxes)
    
    ax[nrow, 0].set_title("UMAP colored by score")
    # Plot histogram of score
    ax[nrow, 1].hist(adata.obs["score"], bins=100, color='blue', alpha=0.7)
    ax[nrow, 1].set(title="Histogram of Score", xlabel="Score", ylabel="Number of Cells")
    ax[nrow, 1].set_yscale('log')
    ax[nrow, 1].grid(True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Exclusion plots saved to: {plot_path}")