"""
adata_utils.py
--------------
Utility functions for AnnData preprocessing and manipulation.

Provides shared functions used across multiple scripts in the SIGMA pipeline
for preparing, normalizing, and processing AnnData objects.
"""

import os
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from rename_genes import rename_genes
from constants import (
    MT_GENE_PATTERN,
    RIBO_GENE_PATTERN,
    HB_GENE_PATTERN,
    TARGET_SUM,
    MIN_CELLS_FOR_UMAP,
    THREAD_ENV_VARS,
)


# =============================================================================
# Thread Management
# =============================================================================


def set_thread_environment(threads: int) -> None:
    """Set environment variables for parallel thread control.

    Args:
        threads: Number of threads to use for parallel operations
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(threads)


# =============================================================================
# Data Preparation
# =============================================================================


def prepare_adata_qc(adata: AnnData, species: str) -> AnnData:
    """Prepare AnnData object for quality control pipeline.

    Creates a fresh AnnData from the count matrix, renames genes using
    Ensembl annotations, and stores raw counts in a layer.

    Args:
        adata: Input AnnData object
        species: Species identifier for gene renaming (e.g., 'hsapiens')

    Returns:
        Prepared AnnData object with renamed genes and 'raw' layer
    """
    count_matrix = adata.X
    adata = AnnData(X=count_matrix, obs=adata.obs.copy(), var=adata.var.copy())

    genes = rename_genes(adata.var_names.to_list(), species)
    adata.var_names = genes

    if adata.var_names.has_duplicates:
        adata.var_names_make_unique()

    adata.layers["raw"] = adata.X.copy()
    return adata


def prepare_adata_target(adata: AnnData, species: str) -> AnnData:
    """Prepare AnnData object for target cell identification.

    Creates a fresh AnnData from the count matrix, renames genes using
    Ensembl annotations, handles missing names, and normalizes to uppercase.

    Args:
        adata: Input AnnData object
        species: Species identifier for gene renaming (e.g., 'hsapiens')

    Returns:
        Prepared AnnData object with standardized gene names
    """
    count_matrix = adata.X
    original_var_names = pd.Index(adata.var_names)
    renamed_genes = rename_genes(original_var_names.tolist(), species)

    # Handle missing gene names
    genes = pd.Series(renamed_genes, index=original_var_names, dtype="object")
    missing_mask = genes.isna() | genes.eq("")
    genes.loc[missing_mask] = genes.index[missing_mask]
    genes = genes.astype(str).str.strip().str.upper()

    adata = AnnData(X=count_matrix, obs=adata.obs.copy(), var=adata.var.copy())
    adata.var_names = genes
    adata.var_names_make_unique()

    return adata


# =============================================================================
# Gene Identification
# =============================================================================


def identify_special_genes(adata: AnnData) -> AnnData:
    """Identify mitochondrial, ribosomal, and hemoglobin genes.

    Adds boolean columns to adata.var indicating gene types.

    Args:
        adata: AnnData object

    Returns:
        AnnData with gene type annotations in .var ('mt', 'ribo', 'hb')
    """
    adata.var["mt"] = adata.var_names.str.match(MT_GENE_PATTERN)
    adata.var["ribo"] = adata.var_names.str.match(RIBO_GENE_PATTERN)
    adata.var["hb"] = adata.var_names.str.match(HB_GENE_PATTERN)

    print(f"- # mt genes : {adata.var.mt.sum()}")
    print(f"- # ribo genes : {adata.var.ribo.sum()}")
    print(f"- # hb genes : {adata.var.hb.sum()}")

    return adata.copy()


# =============================================================================
# Normalization Utilities
# =============================================================================


def check_if_normalized(adata: AnnData) -> bool:
    """Check if data appears to be already normalized.

    Uses row sum analysis to detect if normalization has been applied.

    Args:
        adata: AnnData object to check

    Returns:
        True if data appears normalized, False if likely raw counts
    """
    row_sums = np.sum(adata.X, axis=1)
    return not np.allclose(row_sums, row_sums.astype(int))


def normalize_and_log(adata: AnnData, target_sum: float = TARGET_SUM, layer: str = "raw") -> AnnData:
    """Apply CPM normalization and log1p transformation.

    Stores raw counts in 'raw' layer and normalized counts in 'log1p' layer.

    Args:
        adata: Input AnnData object
        target_sum: Target sum for normalization (default: 1e6 for CPM)

    Returns:
        Normalized AnnData with layers
    """
    adata = adata.copy()
    adata.layers[layer] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers[f"{layer}_log1p"] = adata.X.copy()
    return adata


def preprocess_adata(adata: AnnData, layer: str = None, already_normalized: bool = False) -> AnnData:
    """Normalize data and compute dimensionality reduction.

    Args:
        adata: Input AnnData object
        already_normalized: Whether data is already normalized

    Returns:
        Processed AnnData with layers and embeddings (PCA, UMAP)
    """
    adata = adata.copy()

    if already_normalized:
        print("Data appears to be already normalized. Skipping normalization step.")
    else:
        adata = normalize_and_log(adata, layer=layer)

    # Compute embeddings only with enough cells
    if adata.shape[0] >= MIN_CELLS_FOR_UMAP:
        sc.pp.pca(adata, svd_solver="arpack", mask_var=None)
        n_neighbors = min(15, adata.shape[0] - 1)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata)

    return adata


# =============================================================================
# Cell Filtering
# =============================================================================


def find_candidate_cells(
    adata: AnnData,
    genes: set,
    min_genes: float,
    threshold: float,
) -> AnnData:
    """Filter cells based on candidate gene expression.

    Args:
        adata: Input AnnData object
        genes: Set of candidate gene names to check
        min_genes: Minimum number of genes that must be expressed
        threshold: Minimum expression value for a gene to be considered detected

    Returns:
        Filtered AnnData containing only cells passing the criteria
    """
    filtered_adata = adata.copy()
    available_genes = list(genes.intersection(filtered_adata.var_names))

    if not available_genes:
        print("No available genes to filter on. Returning empty AnnData object.")
        return AnnData(np.array([]))

    gene_expression_matrix = filtered_adata[:, available_genes].to_df()
    genes_detected_per_cell = (gene_expression_matrix >= threshold).sum(axis=1)
    is_expressed = genes_detected_per_cell >= min_genes

    return filtered_adata[is_expressed]


def remove_duplicate_genes(adata: AnnData) -> AnnData:
    """Remove duplicate gene names from AnnData.

    Args:
        adata: Input AnnData object

    Returns:
        AnnData with unique gene names (keeps first occurrence)
    """
    if not adata.var_names.is_unique:
        print("Warning: Duplicate variable names found. Removing duplicates...")
        return adata[:, ~adata.var_names.duplicated(keep="first")]
    return adata
