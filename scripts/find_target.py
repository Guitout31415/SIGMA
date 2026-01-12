"""
find_target.py
--------------
Tool for identifying target cells from AnnData objects based on gene expression.

Uses Gaussian Mixture Models (GMM) to identify cell populations expressing
specific marker genes, with optional exclusion of unwanted cell types.
"""

import json
import time
import argparse

import numpy as np
import scanpy as sc

from rename_genes import rename_genes
from constants import MIN_CELLS_FOR_UMAP
from adata_utils import (
    prepare_adata_target,
    check_if_normalized,
    preprocess_adata,
    normalize_and_log,
    find_candidate_cells,
    remove_duplicate_genes,
)
from gmm_utils import (
    fit_gmm,
    identify_target_components,
)
from plotting import plot_target_figures, plot_exclude_figures


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Identify target cells from AnnData based on gene expression."
    )
    parser.add_argument(
        "--h5ad_file",
        type=str,
        required=True,
        help="Path to the input AnnData (.h5ad) file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file for the filtered AnnData object.",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Name of the study, used for output filenames.",
    )
    parser.add_argument(
        "--candidate_genes",
        nargs="+",
        type=str,
        required=True,
        help="Genes for initial cell filtering.",
    )
    parser.add_argument(
        "--target_genes",
        nargs="+",
        type=str,
        required=True,
        help="Genes for GMM clustering.",
    )
    parser.add_argument(
        "--exclude_genes",
        help="JSON dictionary of genes to exclude from target population.",
    )
    parser.add_argument(
        "--min_genes_detected",
        type=float,
        required=True,
        help="Minimum candidate genes required per cell.",
    )
    parser.add_argument(
        "--gene_detection_threshold",
        type=float,
        required=True,
        help="Minimum expression for gene detection.",
    )
    parser.add_argument(
        "--n_components_target",
        type=str,
        default="auto",
        help="Number of GMM components for target genes, or 'auto'.",
    )
    parser.add_argument(
        "--n_components_exclu",
        type=str,
        default="auto",
        help="Number of GMM components for exclude genes, or 'auto'.",
    )
    parser.add_argument(
        "--min_mean_expression",
        type=float,
        default=2.0,
        help="Minimum mean expression for target cluster.",
    )
    parser.add_argument(
        "--do_QC",
        type=str,
        default="True",
        help="Whether to perform quality control.",
    )
    parser.add_argument(
        "--plot_folder",
        type=str,
        default=None,
        help="Directory to save plots (None to skip).",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="hsapiens",
        help="Species name for Ensembl database.",
    )
    parser.add_argument(
        "--exclude_celltypes",
        type=str,
        default="False",
        help="Exclude entire cell types (True) or specific low genes (False).",
    )
    return parser.parse_args()


# =============================================================================
# Pipeline Steps
# =============================================================================


def step1_find_gene_aliases(
    adata: sc.AnnData,
    candidate_genes: list,
    target_genes: list,
    exclude_genes: dict,
    species: str,
) -> tuple:
    """Step 1: Find gene aliases and available genes in the dataset.

    Args:
        adata: Input AnnData object
        candidate_genes: List of candidate gene names
        target_genes: List of target gene names
        exclude_genes: Dict of exclusion categories to gene lists
        species: Species for gene alias lookup

    Returns:
        Tuple of (candidate_genes_avail, target_genes_avail, exclude_genes_avail)
    """
    print("\n--- 1. Finding Gene Aliases ---")
    step_start = time.time()

    candidate_aliases = rename_genes(candidate_genes, species=species)
    target_aliases = rename_genes(target_genes, species=species)

    candidate_genes_avail = set(candidate_aliases).intersection(adata.var_names)
    target_genes_avail = set(target_aliases).intersection(adata.var_names)

    exclude_genes_avail = {}
    if exclude_genes:
        for category, genes in exclude_genes.items():
            exclude_aliases = rename_genes(genes, species=species)
            exclude_genes_avail[category] = set(exclude_aliases).intersection(adata.var_names)

    print(f"Available candidate genes: {candidate_genes_avail}")
    print(f"Available target genes: {target_genes_avail}")
    for category, genes in exclude_genes_avail.items():
        print(f"Available exclude genes for {category}: {genes}")
    print(f"Step 1 completed in {time.time() - step_start:.2f} seconds")

    return candidate_genes_avail, target_genes_avail, exclude_genes_avail

def step2_find_candidates(
    adata: sc.AnnData,
    candidate_genes_avail: set,
    min_genes_detected: float,
    gene_detection_threshold: float,
) -> tuple:
    """Step 2: Find candidate cells and apply normalization.

    Args:
        adata: Input AnnData object
        candidate_genes_avail: Set of available candidate genes
        min_genes_detected: Minimum candidate genes per cell
        gene_detection_threshold: Expression threshold for detection

    Returns:
        Tuple of (candidate_cells, is_valid) where is_valid indicates
        if there are enough cells to continue
    """
    print("\n--- 2. Find candidate cells ---")
    step_start = time.time()
    print(
        f"Keeping cells expressing >= {int(min_genes_detected)} candidate genes, "
        f"each above threshold {int(gene_detection_threshold)}."
    )

    already_normalized = check_if_normalized(adata)

    candidate_cells = None
    if already_normalized:
        print("Data appears to be already normalized.")
        raw_layer_name = None
        if hasattr(adata, "layers"):
            for lay in adata.layers:
                layer_adata = sc.AnnData(adata.layers[lay], obs=adata.obs, var=adata.var)
                if not check_if_normalized(layer_adata):
                    raw_layer_name = lay
                    break
        if raw_layer_name is not None:
            print(f"Layer '{raw_layer_name}' appears raw, using for further analysis.")
            adata.X = adata.layers[raw_layer_name]
            adata.layers["raw"] = adata.X.copy()
            adata = normalize_and_log(adata, layer="raw")
            candidate_cells = find_candidate_cells(
                adata, candidate_genes_avail, min_genes_detected, gene_detection_threshold
            )
        else:
            print("No raw layers found. Continuing with all cells.")
            candidate_cells = adata.copy()
    else:
        adata.layers["raw"] = adata.X.copy()
        candidate_cells = find_candidate_cells(
            adata, candidate_genes_avail, min_genes_detected, gene_detection_threshold
        )

    # Work on a copy to avoid modifying views
    if candidate_cells is not None:
        candidate_cells = candidate_cells.copy()

    # Check if we have enough cells
    if candidate_cells.shape[0] == 0:
        print("No candidate cells found. Returning empty AnnData.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        print(f"Step 2 completed in {time.time() - step_start:.2f} seconds")
        return candidate_cells, False

    pct = candidate_cells.shape[0] / adata.shape[0] * 100
    print(f"Number of candidate cells: {candidate_cells.shape[0]} ({pct:.2f}%)")

    if candidate_cells.shape[0] < 2:
        print("Not enough cells for GMM. Setting proba_target to 0.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        print(f"Step 2 completed in {time.time() - step_start:.2f} seconds")
        return candidate_cells, False

    print(f"Step 2 completed in {time.time() - step_start:.2f} seconds")

    if "raw" in candidate_cells.layers:
        candidate_cells.X = candidate_cells.layers["raw"].copy()

    return candidate_cells, True

def step3_fit_gmm_target(candidate_cells: sc.AnnData,
                         target_genes_avail: set,
                         exclude_genes_avail: dict,
                         n_components_target: str) -> tuple:
    
    already_normalized = check_if_normalized(candidate_cells)
    sub_candidate = candidate_cells.copy()
    # Remove exclude genes from raw data
    genes_remove = [str(g) for genes in exclude_genes_avail.values() for g in genes if g not in target_genes_avail]
    sub_candidate = sub_candidate[:, ~sub_candidate.var_names.isin(genes_remove)] # Remove exclude genes from raw data
    # sub_candidate = sub_candidate[:, list(target_genes_avail)]
    sub_candidate = preprocess_adata(sub_candidate, layer="raw_target", already_normalized=already_normalized)
    # sub_candidate.X = sub_candidate.layers["raw_target_log1p"].copy()

    # Compute mean expression of target genes
    target_df = sub_candidate[:, list(target_genes_avail)].to_df()
    candidate_cells.obs["target_mean_expr"] = target_df.mean(axis=1)

    # Fit GMM and predict probabilities for target genes
    gmm_target = fit_gmm(candidate_cells.obs["target_mean_expr"], n_components_target, "Target", "False")

    return gmm_target, candidate_cells

def step3bis_fit_gmm_exclude(candidate_cells: sc.AnnData,
                            target_genes_avail: set,
                            exclude_genes_avail: dict,
                            n_components_exclu: str,
                            exclude_celltypes: str) -> tuple:
    gmm_excludes = {}
    
    if not exclude_genes_avail:
        print("\n--- 3bis. No Exclude genes provided, skipping GMM fitting for Exclude genes ---")
        return gmm_exclude, None

    print("\n--- 3bis. Fitting GMM for Exclude genes ---")
    step_start = time.time()

    already_normalized = check_if_normalized(candidate_cells)
    for category, genes in exclude_genes_avail.items():
        sub_candidate = candidate_cells.copy()
        
        # Remove target genes from raw data
        genes_remove = [str(g) for g in target_genes_avail if g not in genes]
        sub_candidate = sub_candidate[:, ~sub_candidate.var_names.isin(genes_remove)]
        sub_candidate = preprocess_adata(sub_candidate, layer=f"raw_{category}", already_normalized=already_normalized)
        # sub_candidate.X = sub_candidate.layers[f"raw_{category}_log1p"]
        
        # Compute mean expression of exclude genes
        exclude_df = sub_candidate[:, list(genes)].X
        candidate_cells.obs[f"exclude_mean_expr_{category}"] = exclude_df.mean(axis=1)
        
        # Fit GMM and predict probabilities for exclude genes
        gmm_exclude = fit_gmm(candidate_cells.obs[f"exclude_mean_expr_{category}"], n_components_exclu, category, exclude_celltypes)

        # Store results
        gmm_excludes[category] = gmm_exclude
    
    return gmm_excludes, candidate_cells

def step4_calculate_target_probabilities(candidate_cells: sc.AnnData,
                                         gmm_target: object,
                                         min_mean_expression: float) -> sc.AnnData:
    target_indices = identify_target_components(gmm_target, min_mean_expression, "True")

    if isinstance(target_indices, int) and target_indices == -1:
        print("No valid target components identified. Setting proba_target to 0.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        return candidate_cells
    
    # Predict probabilities
    probas = gmm_target.predict_proba(candidate_cells.obs["target_mean_expr"].values.reshape(-1, 1))

    # handle single integer index (results in 1D slice) or multi-index
    if isinstance(target_indices, int):
        candidate_cells.obs["proba_target"] = probas[:, target_indices]
    else:
        candidate_cells.obs["proba_target"] = np.sum(probas[:, target_indices], axis=1)
    
    return candidate_cells

def step4bis_calculate_exclude_probabilities(candidate_cells: sc.AnnData,
                                         gmm_exclude: object,
                                         min_mean_expression: float,
                                         exclude_celltypes: str) -> sc.AnnData:
    
    for category in gmm_exclude.keys():
        print(f"\nCalculating exclusion probabilities for category: {category}")
        gmm_excl = gmm_exclude[category]
        exclude_indices = identify_target_components(gmm_excl, min_mean_expression, exclude_celltypes)
        
        if isinstance(exclude_indices, int) and exclude_indices == -1:
            print(f"No valid exclusion components identified for {category}. Setting proba_{category} to 0.")
            candidate_cells.obs[f"proba_{category}"] = np.zeros(candidate_cells.shape[0])
        else:
            # Predict probabilities
            probas = gmm_excl.predict_proba(candidate_cells.obs[f"exclude_mean_expr_{category}"].values.reshape(-1, 1))

            if exclude_celltypes == "False":
                probas = 1-probas

            # handle single integer index (results in 1D slice) or multi-index
            if isinstance(exclude_indices, int):
                candidate_cells.obs[f"proba_{category}"] = probas[:, exclude_indices]
            else:
                candidate_cells.obs[f"proba_{category}"] = np.sum(probas[:, exclude_indices], axis=1)

            if isinstance(exclude_indices, int) and exclude_indices == -1:
                print("No valid target components identified. Setting proba_target to 0.")
                candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
    
    return candidate_cells

def step5_calculate_score(candidate_cells: sc.AnnData) -> sc.AnnData:
    """Step 5: Calculate final target score for candidate cells.

    Combines target and exclusion probabilities into a final score.

    Args:
        candidate_cells: AnnData with proba_target and proba_exclude_* columns
    """
    print("\n--- 5. Calculating final target score ---")
    step_start = time.time()

    proba_exclude_cols = [col for col in candidate_cells.obs.columns if col.startswith("proba_") and col != "proba_target"]
    if proba_exclude_cols:
        proba_exclude_sum = candidate_cells.obs[proba_exclude_cols].sum(axis=1)
    else:
        proba_exclude_sum = 0

    candidate_cells.obs["score"] = candidate_cells.obs["proba_target"] - proba_exclude_sum
    candidate_cells.obs["score"] = candidate_cells.obs["score"].clip(lower=0)

    print(f"Step 5 completed in {time.time() - step_start:.2f} seconds")
    return candidate_cells


# =============================================================================
# Main Pipeline
# =============================================================================


def find_target_cells(
    adata: sc.AnnData,
    study_name: str,
    candidate_genes: list,
    target_genes: list,
    exclude_genes: dict,
    min_genes_detected: float,
    gene_detection_threshold: float,
    n_components_target: str,
    n_components_exclu: str,
    min_mean_expression: float,
    plot_folder: str,
    species: str,
    exclude_celltypes: str,
) -> sc.AnnData:
    """Identify and extract target cells from an AnnData object.

    Orchestrates the full pipeline by calling individual step functions:
    1. Find gene aliases
    2. Find candidate cells and normalize
    3. Fit GMM for target genes
    3bis. Fit GMM for exclusion genes
    4. Calculate target probabilities
    4bis & 5. Calculate exclusion probabilities and score
    6. Generate plots

    Args:
        adata: Input AnnData object
        study_name: Name of the study
        candidate_genes: Genes for initial cell filtering
        target_genes: Genes for primary GMM clustering
        exclude_genes: Dict of genes for secondary GMM filtering
        min_genes_detected: Min candidate genes per cell
        gene_detection_threshold: Expression threshold for detection
        n_components_target: GMM components for marker genes
        n_components_exclu: GMM components for exclude genes
        min_mean_expression: Min expression for target cluster
        plot_folder: Directory for plots
        species: Species for gene alias lookup
        exclude_celltypes: Whether to exclude entire cell types

    Returns:
        AnnData with 'proba_target' and other probability columns
    """
    # Step 1: Find gene aliases
    candidate_genes_avail, target_genes_avail, exclude_genes_avail = step1_find_gene_aliases(
        adata, candidate_genes, target_genes, exclude_genes, species
    )

    # Step 2: Find candidate cells and normalize
    candidate_cells, is_valid = step2_find_candidates(
        adata, candidate_genes_avail, min_genes_detected, gene_detection_threshold
    )
    if not is_valid:
        print("Insufficient candidate cells to proceed further.")

    # Step 3: Fit GMM for target genes
    gmm_target, candidate_cells = step3_fit_gmm_target(
        candidate_cells, target_genes_avail, exclude_genes_avail, n_components_target
    )

    # Step 3bis: Fit GMM for exclusion genes
    gmm_exclude, candidate_cells = step3bis_fit_gmm_exclude(
        candidate_cells, target_genes_avail, exclude_genes_avail, n_components_exclu, exclude_celltypes
    )

    # Step 4: Calculate target probabilities
    candidate_cells = step4_calculate_target_probabilities(
        candidate_cells, gmm_target, min_mean_expression
    )

    # Step 4bis: Calculate exclusion probabilities
    candidate_cells = step4bis_calculate_exclude_probabilities(
        candidate_cells, gmm_exclude, min_mean_expression, exclude_celltypes
    )

    # Step 5: Calculate final target score
    candidate_cells = step5_calculate_score(candidate_cells)

    # Plotting
    if plot_folder is not None:
        plot_target_figures(candidate_cells, gmm_target, plot_folder, study_name)
        plot_exclude_figures(candidate_cells, gmm_exclude, plot_folder, study_name)

    return candidate_cells


def main() -> None:
    """Main entry point for target cell identification."""
    start_time = time.time()

    args = parse_arguments()
    args.exclude_genes = json.loads(args.exclude_genes) if args.exclude_genes else {}

    try:
        adata = sc.read_h5ad(args.h5ad_file)
        if args.do_QC == "False":
            adata = prepare_adata_target(adata, species=args.species)
    except FileNotFoundError:
        print(f"Error: The file '{args.h5ad_file}' was not found.")
        return

    adata = remove_duplicate_genes(adata)

    print("\n====================================")
    print("Beginning target cell identification...")
    print("====================================")

    adata_target = find_target_cells(
        adata=adata,
        study_name=args.study_name,
        candidate_genes=args.candidate_genes,
        target_genes=args.target_genes,
        exclude_genes=args.exclude_genes,
        min_genes_detected=args.min_genes_detected,
        gene_detection_threshold=args.gene_detection_threshold,
        n_components_target=args.n_components_target,
        n_components_exclu=args.n_components_exclu,
        min_mean_expression=args.min_mean_expression,
        plot_folder=args.plot_folder,
        species=args.species,
        exclude_celltypes=args.exclude_celltypes,
    )

    print("\n------------------------------------")
    print("Extraction complete.")

    if adata_target.shape[0] > 0:
        print(f"Saving final AnnData object to: {args.output_file}")
    else:
        print("No target cells found. Empty file saved.")

    adata_target.write(args.output_file)

    execution_time = time.time() - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()