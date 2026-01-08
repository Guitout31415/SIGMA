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
    find_candidate_cells,
    remove_duplicate_genes,
)
from gmm_utils import (
    fit_gmm_and_predict_probas,
    identify_target_components,
)
from plotting import save_target_plots, save_exclude_plots


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

    Performs the following steps:
    1. Find gene aliases for candidate, marker, and exclude gene lists
    2. Find candidate cells and apply log1p-CPM normalization
    3. Fit GMM models
    4. Calculate target probabilities for each cell
    5. Optionally apply score to keep only most probable target cells
    6. Generate and save plots if plot folder is provided

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

    # Step 2: Find candidate cells and normalize
    print("\n--- 2. Find candidate cells and log1p-cpm normalize ---")
    step_start = time.time()
    print(
        f"Keeping cells expressing >= {int(min_genes_detected)} candidate genes, "
        f"each above threshold {int(gene_detection_threshold)}."
    )

    already_normalized = check_if_normalized(adata)

    if already_normalized:
        print("Data appears to be already normalized.")
        if hasattr(adata, "layers"):
            for lay in adata.layers:
                layer_adata = sc.AnnData(adata.layers[lay], obs=adata.obs, var=adata.var)
                if not check_if_normalized(layer_adata):
                    print(f"Layer '{lay}' appears raw, using for further analysis.")
                    adata.X = adata.layers[lay]
                    already_normalized = False
                    break
            else:
                print("No raw layers found. Continuing with all cells.")
        candidate_cells = adata.copy()
    else:
        candidate_cells = find_candidate_cells(
            adata, candidate_genes_avail, min_genes_detected, gene_detection_threshold
        )

    if candidate_cells.shape[0] == 0:
        print("No candidate cells found. Returning empty AnnData.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        return candidate_cells

    pct = candidate_cells.shape[0] / adata.shape[0] * 100
    print(f"Number of candidate cells: {candidate_cells.shape[0]} ({pct:.2f}%)")

    if candidate_cells.shape[0] < 2:
        print("Not enough cells for GMM. Setting proba_target to 0.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        return candidate_cells

    print("Normalizing and log1p transforming data...")
    candidate_cells = preprocess_adata(candidate_cells, already_normalized)

    print("Computing mean expression of target genes...")
    marker_df = candidate_cells[:, list(target_genes_avail)].to_df()
    candidate_cells.obs["target_mean_expr"] = marker_df.mean(axis=1)
    print(f"Step 2 completed in {time.time() - step_start:.2f} seconds")

    # Step 3: Fit GMM for target genes
    print("\n--- 3. Fitting GMM for Target genes ---")
    step_start = time.time()

    data_target = np.array(candidate_cells.obs["target_mean_expr"]).reshape(-1, 1)
    gmm_target, probas_target = fit_gmm_and_predict_probas(
        data_target, n_components_target, category="Target", exclude_celltypes=""
    )
    print(f"Step 3 completed in {time.time() - step_start:.2f} seconds")

    # Step 3bis: Fit GMM for exclusion genes
    gmm_exclude = {}
    probas_exclude = {}

    if exclude_genes_avail:
        print("\n--- 3bis. Fitting GMM for Exclude genes ---")
        step_start = time.time()

        for category, genes in exclude_genes_avail.items():
            exclude_df = candidate_cells[:, list(genes)].to_df()
            candidate_cells.obs[f"{category}_mean_expr"] = exclude_df.mean(axis=1)
            data_exclude = np.array(candidate_cells.obs[f"{category}_mean_expr"]).reshape(-1, 1)
            gmm_exclude[category], probas_exclude[category] = fit_gmm_and_predict_probas(
                data_exclude, n_components_exclu, category=category, exclude_celltypes=exclude_celltypes
            )
        print(f"Step 3bis completed in {time.time() - step_start:.2f} seconds")

    # Step 4: Calculate target probabilities
    print("\n--- 4. Calculate probabilities for target genes ---")
    step_start = time.time()

    candidate_cells.obs["proba_target"] = identify_target_components(
        gmm_target, probas_target, min_mean_expression
    )
    print(f"Step 4 completed in {time.time() - step_start:.2f} seconds")

    # Step 4bis & 5: Calculate exclusion probabilities and score
    if exclude_genes_avail:
        print("--- 4bis. Calculate probabilities for exclude genes ---")
        step_start = time.time()

        score = candidate_cells.obs["proba_target"].copy()

        if exclude_celltypes in ("True", True):
            print("Excluding entire cell types based on exclude genes...")
            for category in gmm_exclude:
                gmm = gmm_exclude[category]
                proba = probas_exclude[category]
                exclude_proba = identify_target_components(gmm, proba, min_mean_expression)
                candidate_cells.obs[f"proba_{category}"] = exclude_proba
                score = score - exclude_proba
        else:
            print("Excluding specific 'low' genes based on exclude genes...")
            for category in gmm_exclude:
                gmm = gmm_exclude[category]
                proba = probas_exclude[category]
                exclude_component = np.argmin(gmm.means_.flatten())

                if gmm.means_.flatten()[exclude_component] > min_mean_expression:
                    candidate_cells.obs[f"proba_{category}"] = 1
                else:
                    candidate_cells.obs[f"proba_{category}"] = 1 - proba[:, exclude_component]
                score = score - candidate_cells.obs[f"proba_{category}"]

        print("\n--- 5. Calculate score ---")
        candidate_cells.obs["score"] = score.clip(lower=0)
        print(f"Steps 4bis and 5 completed in {time.time() - step_start:.2f} seconds")

    # Generate plots
    if plot_folder:
        save_target_plots(candidate_cells, gmm_target, study_name, plot_folder)
        if exclude_genes_avail:
            save_exclude_plots(
                candidate_cells, study_name, plot_folder,
                exclude_names=list(exclude_genes.keys()),
                gmm_excludes=gmm_exclude,
            )

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