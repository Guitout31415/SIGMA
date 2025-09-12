from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import scanpy as sc
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root
import pandas as pd
import os
from scipy.optimize import brentq
from scipy.signal import argrelextrema
import json
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from scipy.stats import gaussian_kde
from pybiomart import Dataset

# --- Constants ---
TARGET_SUM = 1e6

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments and returns them."""
    parser = argparse.ArgumentParser(
        description="A tool for identifying target cells from an AnnData object based on gene expression."
    )
    parser.add_argument("--h5ad_file", type=str, required=True, help="Path to the input AnnData (.h5ad) file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file for the filtered AnnData object.")
    parser.add_argument("--study_name", type=str, required=True, help="A name for the study, used for output filenames.")
    parser.add_argument("--target_genes", nargs='+', type=str, required=True, help="List of genes to be used for GMM clustering.")
    parser.add_argument("--exclude_genes", help="Optional JSON dictionary of genes to exclude from the target cell population. Ensure keys and values are enclosed in double quotes.")
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
    adata = adata.copy()
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
    Optimized version with caching and vectorized operations.
    
    Args:
        genes (list[str]): A list of gene names.
        species (str): The species name for the Ensembl database. Defaults to "hsapiens".
        
    Returns:
        set[str]: A set of all unique aliases found.
    """
    # Cache pour éviter de refaire la même requête
    cache_key = f"{species}_biomart_data"
    if not hasattr(get_gene_aliases, 'cache'):
        get_gene_aliases.cache = {}
    
    # Récupérer ou créer les données biomart
    if cache_key not in get_gene_aliases.cache:
        print(f"Fetching gene data from Ensembl for {species}...")
        dataset = Dataset(name=f'{species}_gene_ensembl', host='http://www.ensembl.org')
        results = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'hgnc_symbol', 'external_synonym'])
        
        # Préprocesser une seule fois : convertir en lowercase et créer un index
        processed_data = []
        for idx, row in results.iterrows():
            row_values = [str(val).lower() if pd.notnull(val) else '' for val in row.values]
            processed_data.append((idx, row_values, row.values))
        
        get_gene_aliases.cache[cache_key] = (results, processed_data)
        print(f"Cached {len(results)} gene entries.")
    
    results, processed_data = get_gene_aliases.cache[cache_key]
    
    # Conversion vectorisée des gènes d'entrée
    genes_lower = [gene.lower() for gene in genes]
    genes_set = set(genes_lower)
    
    all_aliases = set()
    
    # Recherche optimisée
    for idx, row_lower, row_original in processed_data:
        # Vérifier si un des gènes recherchés est dans cette ligne
        if any(gene_lower in row_lower for gene_lower in genes_set):
            # Ajouter tous les alias de cette ligne
            all_aliases.update([str(val) for val in row_original if pd.notnull(val)])
    
    return {str(alias).upper() for alias in all_aliases if pd.notnull(alias) and str(alias).strip()}

def find_optimal_gmm_components(data: np.ndarray) -> int:
    """
    Estimate the optimal number of Gaussian components in a univariate dataset 
    using kernel density estimation and spline-based peak detection.

    Steps:
        1. Estimate the probability density function using a Gaussian kernel (KDE) with optimal bandwidth.
        2. Fit a smoothing spline to the KDE curve.
        3. Compute first and second derivatives of the spline.
        4. Solve systems to find local extrema.
        5. Filter maxima based on the second derivative.
        6. Return the number of significant peaks.

    Args:
        data (np.ndarray): 1D input data array for GMM fitting.
        
    Returns:
        int: The optimal number of components found.
    """
   
    # Étape 1 : Calculer la KDE avec bandwidth optimal via cross-validation
    expr = data.flatten()
    kde = gaussian_kde(expr)
    x_grid = np.linspace(expr.min(), expr.max(), 3000)
    kde_values = kde(x_grid)

    # Étape 2 : Ajuster un spline de lissage à l'estimation de densité
    smoothing_factor = 0.01*len(x_grid)  # Facteur de lissage basé sur la taille des données
    spline = UnivariateSpline(x_grid, kde_values, s=smoothing_factor, k=3)

    # Étape 3 : Trouver les extrema locaux en résolvant la dérivée première = 0
    deriv1 = spline.derivative()
    deriv_values = deriv1(x_grid)
    sign_changes = np.where(np.diff(np.sign(deriv_values)) != 0)[0]

    critical_points = []
    for i in sign_changes:
        try:
            root = brentq(deriv1, x_grid[i], x_grid[i+1])
            critical_points.append(root)
        except ValueError:
            pass  # Pas de racine dans cet intervalle

    # Étape 4 : Filtrer pour garder seulement les maxima locaux
    # Pour chaque point critique, vérifier la dérivée seconde (Hessienne en 1D)
    local_maxima = []
    for cp in critical_points:
        if x_grid.min() <= cp <= x_grid.max():  # S'assurer que c'est dans la plage
            second_deriv = spline.derivative(2)(cp)
            if second_deriv < 0 and abs(second_deriv) > 0.001:  # Maxima local avec seuil
                local_maxima.append(cp)

    # Étape 5 : Retourner les maxima locaux et leur nombre
    n_peaks = len(local_maxima)

    return n_peaks

def fit_gmm_and_predict_probas(data: np.ndarray, n_components: str, category: str) -> tuple[GaussianMixture, np.ndarray]:
    """
    Fits a Gaussian Mixture Model (GMM) to the data and predicts probabilities.
    
    Args:
        data (np.ndarray): The input data.
        n_components (str): The number of components, or "auto" to determine it automatically.
        
    Returns:
        tuple[GaussianMixture, np.ndarray]: The fitted GMM and the predicted probabilities.
    """
    if n_components == "auto":
        print(f"Automatically determining the number of components for {category}...")
        optimal_n = find_optimal_gmm_components(data)
        print(f"\tOptimal number of components: {optimal_n}")
    else:
        optimal_n = int(n_components)
        print(f"Using specified number of components for {category}: {optimal_n}")
    gmm = GaussianMixture(n_components=optimal_n).fit(data)
    probas = gmm.predict_proba(data)
    return gmm, probas

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

def plots_target(adata: sc.AnnData, gmm: GaussianMixture, study_name: str, plot_folder: str):
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
    plot_path = os.path.join(plot_folder, f"{study_name}_target.png")

    print("Generating and saving UMAP and histogram plots...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    # Plot UMAP colored by mean expression of candidate genes
    sc.pl.umap(
        adata,
        color="target_mean_expr",
        cmap="viridis",
        size=50,
        ax=axes[0, 0],
        show=False,
        title=f"UMAP colored by target mean expression"
    )
    axes[0, 0].set_facecolor('lightgrey')
    
    # Plot histogram of mean expression of candidate genes with GMM components
    axes[0, 1].hist(adata.obs["target_mean_expr"], bins=100, alpha=0.6, density=True)
    axes[0, 1].set(
        title="Distribution of target mean expression and GMM fit",
        xlabel="Mean expression per cell",
        ylabel="Density"
    )
    axes[0, 1].grid(True)

    x = np.linspace(adata.obs["target_mean_expr"].min(), adata.obs["target_mean_expr"].max(), 1000).reshape(-1, 1)
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
    axes[1, 0].set_facecolor('lightgrey')
    
    # Plot histogram of probability of being a target cell
    axes[1, 1].hist(adata.obs["proba_target"], bins=70)
    axes[1, 1].set(title="Histogram of Target Probability", xlabel="Target Probability", ylabel="Number of Cells")
    axes[1, 1].grid(True)
    
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Targets plots saved to: {plot_path}")

def plots_exclude(adata: sc.AnnData, study_name: str, plot_folder: str, exclude_names: list[str]):
    plot_path = os.path.join(plot_folder, f"{study_name}_exclude.png")
    n_rows = len(exclude_names) + 1
    print(f"Debug: len(exclude_names) = {len(exclude_names)}, n_rows = {n_rows}")
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5*n_rows))
    print(f"Debug: axes.shape = {axes.shape}")

    nrow = 0
    for exclude in exclude_names:
        sc.pl.umap(adata, color=f"{exclude}_mean_expr", cmap="viridis", ax=axes[nrow, 0], size=50, show=False)
        axes[nrow, 0].set_facecolor('lightgrey')

        axes[nrow, 1].hist(adata.obs[f"{exclude}_mean_expr"], bins=100, alpha=0.6, density=True)
        axes[nrow, 1].set(
            title=f"Distribution of {exclude} mean expression and GMM fit",
            xlabel="Mean expression per cell",
            ylabel="Density"
        )
        axes[nrow, 1].grid(True)
        nrow += 1

    print(f"Debug: nrow after loop = {nrow}")
    # Plot UMAP colored by score
    sc.pl.umap(adata, color="score", cmap="viridis", ax=axes[nrow, 0], size=50, show=False)
    axes[nrow, 0].set_facecolor('lightgrey')
    axes[nrow, 0].set(title="UMAP colored by score")
    # Plot histogram of score
    axes[nrow, 1].hist(adata.obs["score"], bins=100, color='blue', alpha=0.7)
    axes[nrow, 1].set(title="Histogram of Score", xlabel="Score", ylabel="Number of Cells")
    axes[nrow, 1].grid(True)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

def find_target_cells(
    adata: sc.AnnData,
    study_name: str,
    target_genes: list[str],
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
        target_genes (list[str]): Genes used to initially filter cells.
        target_genes (list[str]): Genes used for the primary GMM clustering.
        exclude_genes (dict): Genes used for a secondary GMM to filter out unwanted cells.
        min_genes_detected (float): Min number of candidate genes to be expressed per cell.
        gene_detection_threshold (float): Expression threshold for a gene to be considered detected.
        n_components_target (str): Number of GMM components for the 'marker' genes.
        n_components_exclu (str): Number of GMM components for the 'exclude' genes.
        min_mean_expression (float): Minimum mean expression level for the higher component to be considered as target cluster.
        plot_folder (str): Directory for plots.
        species (str): The species for the gene alias lookup.
        
    Returns:
        sc.AnnData: The AnnData object with added 'proba_target' and other probability columns.
    """
    print("\n--- 1. Finding Gene Aliases ---")
    candidate_aliases = get_gene_aliases(target_genes, species=species)
    marker_aliases = get_gene_aliases(target_genes, species=species)
    
    target_genes_avail = candidate_aliases.intersection(adata.var_names)
    target_genes_avail = marker_aliases.intersection(adata.var_names)

    if exclude_genes != dict():
        exclude_genes_avail = dict()
        for category, genes in exclude_genes.items():
            exclude_aliases = get_gene_aliases(genes, species=species)
            exclude_genes_avail[category] = exclude_aliases.intersection(adata.var_names)

    print(f"Available candidate genes: {target_genes_avail}")
    print(f"Available target genes: {target_genes_avail}")
    print(f"Available exclude genes: {exclude_genes_avail}")

    print(f"\n--- 2. Find candidate cells and log1p-cpm normalize ---")
    print(f"Keeping cells that express at least {int(min_genes_detected)} candidate genes, each above a detection threshold of {int(gene_detection_threshold)}.")
    
    # Filter cells based on candidate gene expression
    candidate_cells = find_candidate_cells(adata, target_genes_avail, min_genes_detected, gene_detection_threshold)

    if candidate_cells.shape[0] == 0:
        print("No candidate cells found. Returning an empty AnnData object.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        return candidate_cells
    
    print(f"Number of candidate cells: {candidate_cells.shape[0]} ({candidate_cells.shape[0]/adata.shape[0]*100:.2f}%)")

    print("Normalizing and log1p transforming data...")
    candidate_cells = preprocess_adata(candidate_cells)
    
    # Compute mean expression of markered genes
    print("Compute mean expression of target genes...")
    marker_df = candidate_cells[:, list(target_genes_avail)].to_df()
    candidate_cells.obs['target_mean_expr'] = marker_df.mean(axis=1)
    
    print("\n--- 3. Fitting GMM for Target genes ---")
    data_target = np.array(candidate_cells.obs['target_mean_expr']).reshape(-1, 1)
    
    gmm_target, probas_target = fit_gmm_and_predict_probas(data_target, n_components_target, category="Target")
    
    if exclude_genes_avail:
        print("\n--- 3bis. Fitting GMM for Exclude genes ---")
        gmm_exclude = dict()
        probas_exclude = dict()
        for category, genes in exclude_genes_avail.items():
            exclude_df = candidate_cells[:, list(genes)].to_df()
            candidate_cells.obs[f'{category}_mean_expr'] = exclude_df.mean(axis=1)
            data_exclude = np.array(candidate_cells.obs[f'{category}_mean_expr']).reshape(-1, 1)
            gmm_exclude[category], probas_exclude[category] = fit_gmm_and_predict_probas(data_exclude, n_components_exclu, category=category)

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
        score = candidate_cells.obs["proba_target"]
        for category, gmm, proba in zip(gmm_exclude.keys(), gmm_exclude.values(), probas_exclude.values()):
            exclude_component = np.argmax(gmm.means_.flatten())
            if gmm.means_.flatten()[exclude_component] < min_mean_expression:
                print(f"{category} component mean ({gmm.means_.flatten()[exclude_component]:.4f}) is below the minimum mean expression threshold ({min_mean_expression}).")
                candidate_cells.obs[f"proba_{category}"] = 0
            else:
                candidate_cells.obs[f"proba_{category}"] = proba[:, exclude_component]
            score -= candidate_cells.obs[f"proba_{category}"]

        print(f"\n--- 5. Calculate score ---")
        candidate_cells.obs["score"] = score

    # Plot results if a folder is specified
    if plot_folder:
        plots_target(candidate_cells, gmm_target, study_name, plot_folder)
        if exclude_genes_avail:
            plots_exclude(candidate_cells, study_name, plot_folder, exclude_names=list(exclude_genes.keys()))

    return candidate_cells

def main():
    start_time = time.time()
    
    args = parse_arguments()
    args.exclude_genes = json.loads(args.exclude_genes) if args.exclude_genes != "" else dict()

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
        target_genes=args.target_genes,
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