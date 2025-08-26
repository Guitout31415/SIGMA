import scanpy as sc
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
import argparse
import time
from pybiomart import Dataset

# --- Constants ---
MAX_COMPONENTS = 10
N_INIT_GMM = 10
REG_COVAR_GMM = 1e-6
ENFORCED_THREAD_COUNT = '60'
TARGET_SUM = 1e6

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments and returns them."""
    parser = argparse.ArgumentParser(
        description="A tool for identifying target cells from an AnnData object based on gene expression."
    )
    parser.add_argument("--h5ad_file", type=str, required=True, help="Path to the input AnnData (.h5ad) file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file for the filtered AnnData object.")
    parser.add_argument("--study_name", type=str, required=True, help="A name for the study, used for output filenames.")
    parser.add_argument("--candidate_genes", nargs='+', type=str, required=True, help="List of candidate genes to filter cells.")
    parser.add_argument("--marker_genes", nargs='+', type=str, required=True, help="List of genes to be used for GMM clustering.")
    parser.add_argument("--exclude_genes", nargs='*', type=str, default=[], help="Optional list of genes to exclude from the target cell population.")
    parser.add_argument("--min_genes_detected", type=float, required=True, help="Minimum number of candidate genes required to be detected in a cell.")
    parser.add_argument("--gene_detection_threshold", type=float, required=True, help="Minimum expression value for a gene to be considered detected.")
    parser.add_argument("--n_components_target", type=str, default="auto", help="Number of GMM components for target genes, or 'auto'.")
    parser.add_argument("--n_components_exclu", type=str, default="auto", help="Number of GMM components for exclude genes, or 'auto'.")
    parser.add_argument("--min_mean_expression", type=float, default=2.0, help="Minimum mean expression level for the higher component can be considered as target cluster.")
    parser.add_argument("--plot_folder", type=str, default=None, help="Directory to save plots. Plots will not be generated if None.")
    parser.add_argument("--species", type=str, default="hsapiens", help="Species name for Ensembl database.")
    return parser.parse_args()

def preprocess_adata(adata: sc.AnnData) -> sc.AnnData:
    """
    Normalizes and logs AnnData, then computes PCA and UMAP.
    
    This function modifies the input AnnData object by adding "raw" and "log1p" 
    layers, and populating the 'obsm' and 'uns' attributes with PCA and UMAP 
    results, respectively.
    
    Args:
        adata (sc.AnnData): An annotated data matrix.
        
    Returns:
        sc.AnnData: The processed AnnData object.
    """
    adata.layers["raw"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    
    sc.pp.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata

def get_gene_aliases(genes: list[str], species: str = "hsapiens") -> set[str]:
    """
    Retrieves all known aliases for a list of genes using Ensembl Biomart.
    
    Args:
        genes (list[str]): A list of gene names.
        species (str): The species name for the Ensembl database. Defaults to "hsapiens".
        
    Returns:
        set[str]: A set of all unique aliases found.
    """
    dataset = Dataset(name=f'{species}_gene_ensembl', host='http://www.ensembl.org')
    results = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'hgnc_symbol', 'external_synonym'])
    
    all_aliases = set()
    for gene in genes:
        gene_lower = gene.lower()
        # Find rows where the lowercase gene name appears in any column.
        mask = results.apply(lambda row: row.astype(str).str.lower().eq(gene_lower).any(), axis=1)
        # Add all unique values from these rows to the set of aliases.
        all_aliases.update(results[mask].values.flatten().tolist())
    
    return {str(alias).upper() for alias in all_aliases if pd.notnull(alias)}

def find_optimal_gmm_components(data: np.ndarray, max_components: int) -> tuple[GaussianMixture, int]:
    """
    Determines the optimal number of components for a Gaussian Mixture Model (GMM)
    using the Bayesian Information Criterion (BIC) and the KneeLocator algorithm.
    
    Args:
        data (np.ndarray): The input data for GMM fitting.
        max_components (int): The maximum number of components to test.
        
    Returns:
        tuple[GaussianMixture, int]: A tuple containing the best-fit GMM model and the optimal number of components.
    """
    bics, models = [], []
    n_range = range(1, max_components + 1)
    
    for n in n_range:
        gmm = GaussianMixture(n_components=n, n_init=N_INIT_GMM, reg_covar=REG_COVAR_GMM)
        gmm.fit(data)
        bics.append(gmm.bic(data))
        models.append(gmm)
        
    knee_locator = KneeLocator(n_range, bics, curve='convex', direction='decreasing')
    optimal_n = knee_locator.knee if knee_locator.knee else max_components
    
    return models[optimal_n - 1], optimal_n

def fit_gmm_and_predict_probas(data: np.ndarray, n_components: str) -> tuple[GaussianMixture, np.ndarray, int]:
    """
    Fits a Gaussian Mixture Model (GMM) to the data and predicts probabilities.
    
    Args:
        data (np.ndarray): The input data.
        n_components (str): The number of components, or "auto" to determine it automatically.
        
    Returns:
        tuple[GaussianMixture, np.ndarray, int]: The fitted GMM, the predicted probabilities, and the optimal number of components used.
    """
    if n_components == "auto":
        print("Automatically determining the number of components using BIC...")
        max_components = min(MAX_COMPONENTS, data.shape[0])
        gmm, optimal_n = find_optimal_gmm_components(data, max_components)
        print(f"Optimal number of components: {optimal_n}")
    else:
        num_components = int(n_components)
        print(f"Using specified number of components: {num_components}")
        gmm = GaussianMixture(n_components=num_components, n_init=N_INIT_GMM, reg_covar=REG_COVAR_GMM).fit(data)
        optimal_n = num_components
    
    probas = gmm.predict_proba(data)
    return gmm, probas, optimal_n

def find_candidate_cells(adata: sc.AnnData, genes: set[str], min_genes: float, threshold: float) -> sc.AnnData:
    """
    Filters cells in an AnnData object based on the expression of a set of genes.
    
    Args:
        adata (sc.AnnData): The input AnnData object.
        genes (set[str]): A set of gene names to check for expression.
        min_genes (float): The minimum number of genes from the set that must be expressed.
        threshold (float): The minimum expression level to consider a gene "detected".
        
    Returns:
        sc.AnnData: A new AnnData object containing only the cells that meet the criteria.
    """
    # Create a copy to avoid modifying the original object in place
    filtered_adata = adata.copy()
    
    # Check if any genes from the set exist in the AnnData object.
    available_genes = list(genes.intersection(filtered_adata.var_names))
    if not available_genes:
        print("No available genes to filter on. Returning empty AnnData object.")
        return sc.AnnData(np.array([]))
    
    # Get the expression data for the available genes
    gene_expression_matrix = filtered_adata[:, available_genes].to_df()
    
    # Count how many of these genes are detected in each cell.
    genes_detected_per_cell = (gene_expression_matrix >= threshold).sum(axis=1)
    
    # Filter for cells where the number of detected genes meets the minimum requirement.
    is_expressed = genes_detected_per_cell >= min_genes
    
    return filtered_adata[is_expressed]

def save_plots(adata: sc.AnnData, gmm: GaussianMixture, study_name: str, plot_folder: str, data: np.ndarray):
    """
    Generates and saves UMAP and histogram plots for the analysis.
    
    Args:
        adata (sc.AnnData): The AnnData object with analysis results.
        gmm (GaussianMixture): The fitted GMM model.
        study_name (str): The name of the study, used for the plot title and filename.
        plot_folder (str): The directory to save the plots.
        data (np.ndarray): The data used to fit the GMM.
    """
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, f"{study_name}_extracted.png")
    
    print("Generating and saving UMAP and histogram plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    # Plot UMAP colored by mean expression of candidate genes
    sc.pl.umap(
        adata,
        color="marker_mean_expr",
        cmap="viridis",
        size=50,
        ax=axes[0, 0],
        show=False,
        title=f"UMAP colored by mean expression of marker genes in {study_name}"
    )
    axes[0, 0].set_facecolor('lightgrey')
    
    # Plot histogram of mean expression of candidate genes with GMM components
    axes[0, 1].hist(data, bins=100, alpha=0.6, density=True)
    axes[0, 1].set(
        title="Distribution of mean expression and GMM fit",
        xlabel="Mean expression per cell",
        ylabel="Density"
    )
    axes[0, 1].grid(True)
    
    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x))
    pdf_individual = gmm.predict_proba(x) * pdf[:, None]
    
    axes[0, 1].plot(x, pdf, '-k', label='Total GMM')
    for i in range(gmm.n_components):
        axes[0, 1].plot(x, pdf_individual[:, i], '--', label=f'Component {i+1}')
    axes[0, 1].legend()

    # Plot UMAP colored by probability of being a target cell
    sc.pl.umap(
        adata,
        color="proba_target",
        cmap="viridis",
        size=50,
        ax=axes[1, 0],
        show=False,
        title=f"UMAP colored by probability of being a target cell\n(Gaussian Mixture Model with {gmm.n_components} components)"
    )
    axes[1, 0].collections[0].set_clim(0, 1)
    axes[1, 0].set_facecolor('lightgrey')
    
    # Plot histogram of probability of being a target cell
    axes[1, 1].hist(adata.obs["proba_target"], bins=70)
    axes[1, 1].set(title="Histogram of Target Probability", xlabel="Target Probability", ylabel="Number of Cells")
    axes[1, 1].grid(True)
    
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to: {plot_path}")

def plot_exclude(adata: sc.AnnData, study_name: str, plot_folder: str):
    plot_path = os.path.join(plot_folder, f"{study_name}_exclude_plot.png")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    sc.pl.umap(adata, color="marker_mean_expr", cmap="viridis", ax=axes[0, 0], size=50, show=False)
    sc.pl.umap(adata, color="exclude_mean_expr", cmap="viridis", ax=axes[0, 1], size=50, show=False)
    axes[0, 0].set_facecolor('lightgrey')
    axes[0, 1].set_facecolor('lightgrey')

    x = adata.obs["proba_exclu"]
    y = adata.obs["proba_target"]
    score = adata.obs["score"]
    axes[1, 0].scatter(x, y, c=score, cmap='coolwarm', s=5)
    axes[1, 0].set_xlabel("Alternative Probability")
    axes[1, 0].set_ylabel("Target Probability")
    axes[1, 0].set_xlim(-0.1, 1.1)
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].set_box_aspect(1)


    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], fraction=0.03, pad=0.04)
    cbar.set_label("Score")

    # Plot histogram of score
    axes[1, 1].hist(adata.obs["score"], bins=100, color='blue', alpha=0.7)
    axes[1, 1].set(title="Histogram of Score", xlabel="Score", ylabel="Number of Cells")
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

def find_target_cells(
    adata: sc.AnnData,
    study_name: str,
    candidate_genes: list[str],
    marker_genes: list[str],
    exclude_genes: list[str],
    min_genes_detected: float,
    gene_detection_threshold: float,
    n_components_target: str,
    n_components_exclu: str,
    min_mean_expression: float,
    plot_folder: str,
    species: str
) -> sc.AnnData:
    """
    The main function to identify and extract target cells from an AnnData object.
    
    This function performs a series of steps:
    1. Finds gene aliases for candidate, marker, and exclude gene lists.
    2. Find candidate cells and log1p-cpm normalization
    3. Fit a GMM.
    4. Calculates a 'target probability' for each cell.
    5. Optionally, apply the score to keep only the most probable target cells.
    6. Generates and saves plots if a plot folder is provided.
    7. Returns the final AnnData object containing the identified target cells and their probabilities.
    
    Args:
        adata (sc.AnnData): The initial AnnData object.
        study_name (str): The name of the study.
        candidate_genes (list[str]): Genes used to initially filter cells.
        marker_genes (list[str]): Genes used for the primary GMM clustering.
        exclude_genes (list[str]): Genes used for a secondary GMM to filter out unwanted cells.
        min_genes_detected (float): Min number of candidate genes to be expressed per cell.
        gene_detection_threshold (float): Expression threshold for a gene to be considered detected.
        n_components_target (str): Number of GMM components for the 'marker' genes.
        n_components_exclu (str): Number of GMM components for the 'exclude' genes.
        plot_folder (str): Directory for plots.
        species (str): The species for the gene alias lookup.
        
    Returns:
        sc.AnnData: The AnnData object with added 'proba_target' and 'proba_exclu' columns.
    """
    print("\n--- 1. Finding Gene Aliases ---")
    candidate_aliases = get_gene_aliases(candidate_genes, species=species)
    marker_aliases = get_gene_aliases(marker_genes, species=species)
    exclude_aliases = get_gene_aliases(exclude_genes, species=species)
    
    candidate_genes_avail = candidate_aliases.intersection(adata.var_names)
    marker_genes_avail = marker_aliases.intersection(adata.var_names)
    exclude_genes_avail = exclude_aliases.intersection(adata.var_names)
    
    print(f"Available candidate genes: {candidate_genes_avail}")
    print(f"Available marker genes: {marker_genes_avail}")
    print(f"Available exclude genes: {exclude_genes_avail}")

    print(f"\n--- 2. Find candidate cells and log1p-cpm normalize ---")
    print(f"Keeping cells that express at least {int(min_genes_detected)} candidate genes, each above a detection threshold of {int(gene_detection_threshold)}.")
    
    # Filter cells based on candidate gene expression
    candidate_cells = find_candidate_cells(adata, candidate_genes_avail, min_genes_detected, gene_detection_threshold)

    if candidate_cells.shape[0] == 0:
        print("No candidate cells found. Returning an empty AnnData object.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        return candidate_cells
    
    print(f"Number of candidate cells: {candidate_cells.shape[0]} ({candidate_cells.shape[0]/adata.shape[0]*100:.2f}%)")

    print("Normalizing and log1p transforming data...")
    candidate_cells = preprocess_adata(candidate_cells)
    
    # Compute mean expression of markered genes
    marker_df = candidate_cells[:, list(marker_genes_avail)].to_df()
    candidate_cells.obs['marker_mean_expr'] = marker_df.mean(axis=1)
    
    print("\n--- 3. Fitting GMM for Target Genes ---")
    # Set the number of threads for OPENBLAS to avoid potential conflicts with parallel processing
    os.environ['OPENBLAS_NUM_THREADS'] = ENFORCED_THREAD_COUNT
    data_target = np.array(candidate_cells.obs['marker_mean_expr']).reshape(-1, 1)
    
    gmm_target, probas_target, _ = fit_gmm_and_predict_probas(data_target, n_components_target)
    
    if exclude_genes_avail:
        print("\n--- 3bis. Fitting GMM for Exclude Genes ---")
        
        exclude_df = candidate_cells[:, list(exclude_genes_avail)].to_df()
        candidate_cells.obs['exclude_mean_expr'] = exclude_df.mean(axis=1)
        
        data_exclude = np.array(candidate_cells.obs['exclude_mean_expr']).reshape(-1, 1)
        
        gmm_exclude, probas_exclude, _ = fit_gmm_and_predict_probas(data_exclude, n_components_exclu)

    # Determine the target component as the one with the highest mean
    print("\n--- 4. Calculate probabilities for target genes ---")
    target_component = np.argmax(gmm_target.means_.flatten())
    if gmm_target.means_.flatten()[target_component] < min_mean_expression:
        print(f"Target component mean ({gmm_target.means_.flatten()[target_component]:.4f}) is below the minimum mean expression threshold ({min_mean_expression}).")
        candidate_cells.obs["proba_target"] = 0
    else:
        candidate_cells.obs["proba_target"] = probas_target[:, target_component]

    if exclude_genes_avail:
        print("--- 4bis. Calculate probabilities for exclude genes ---")
        exclude_component = np.argmax(gmm_exclude.means_.flatten())
        if gmm_exclude.means_.flatten()[exclude_component] < min_mean_expression:
            print(f"Exclude component mean ({gmm_exclude.means_.flatten()[exclude_component]:.4f}) is below the minimum mean expression threshold ({min_mean_expression}).")
            candidate_cells.obs["proba_exclu"] = 0
        else:
            candidate_cells.obs["proba_exclu"] = probas_exclude[:, exclude_component]

        print(f"\n--- 5. Calculate score and filter cells ---")
        score = candidate_cells.obs["proba_target"] - candidate_cells.obs["proba_exclu"]
        candidate_cells.obs["score"] = score[candidate_cells.obs_names]
        # candidate_cells = candidate_cells[candidate_cells.obs["score"] > score_threshold]

    # Plot results if a folder is specified
    if plot_folder:
        save_plots(candidate_cells, gmm_target, study_name, plot_folder, data_target)
        if exclude_genes_avail:
            plot_exclude(candidate_cells, study_name, plot_folder)

    return candidate_cells

def main():
    """
    Main function to orchestrate the entire cell extraction process.
    """
    start_time = time.time()
    
    args = parse_arguments()
    
    try:
        adata = sc.read_h5ad(args.h5ad_file)
    except FileNotFoundError:
        print(f"Error: The file '{args.h5ad_file}' was not found.")
        return
    
    if not adata.var_names.is_unique:
        print("Warning: Duplicate variable names found. Removing duplicates...")
        adata = adata[:, ~adata.var_names.duplicated(keep='first')]

    print("\n====================================")
    print("Beginning target cell extraction...")
    print("====================================")
    
    adata_target = find_target_cells(
        adata=adata,
        study_name=args.study_name,
        candidate_genes=args.candidate_genes,
        marker_genes=args.marker_genes,
        exclude_genes=args.exclude_genes,
        min_genes_detected=args.min_genes_detected,
        gene_detection_threshold=args.gene_detection_threshold,
        n_components_target=args.n_components_target,
        n_components_exclu=args.n_components_exclu,
        min_mean_expression=args.min_mean_expression,
        plot_folder=args.plot_folder,
        species=args.species
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