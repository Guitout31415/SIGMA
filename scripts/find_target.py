import numpy as np
from rename_genes import rename_genes
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scanpy as sc
from scipy.interpolate import UnivariateSpline
import pandas as pd
import os
import json
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from scipy.stats import gaussian_kde
from pybiomart import Dataset
import matplotlib.colors as mcolors

# --- Constants ---
TARGET_SUM = 1e6

# Custom colormap from gray to red
gray_to_red = mcolors.LinearSegmentedColormap.from_list("gray_to_red", [(0.5, 0.5, 0.5), (1, 0, 0)])
blue_gray_red = mcolors.LinearSegmentedColormap.from_list("blue_gray_red", [(0, 0, 1), (0.5, 0.5, 0.5), (1, 0, 0)])

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments and returns them."""
    parser = argparse.ArgumentParser(
        description="A tool for identifying target cells from an AnnData object based on gene expression."
    )
    parser.add_argument("--h5ad_file", type=str, required=True, help="Path to the input AnnData (.h5ad) file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file for the filtered AnnData object.")
    parser.add_argument("--study_name", type=str, required=True, help="A name for the study, used for output filenames.")
    parser.add_argument("--candidate_genes", nargs='+', type=str, required=True, help="List of genes to be used for initial cell filtering.")
    parser.add_argument("--target_genes", nargs='+', type=str, required=True, help="List of genes to be used for GMM clustering.")
    parser.add_argument("--exclude_genes", help="Optional JSON dictionary of genes to exclude from the target cell population. Ensure keys and values are enclosed in double quotes.")
    parser.add_argument("--min_genes_detected", type=float, required=True, help="Minimum number of candidate genes required to be detected in a cell.")
    parser.add_argument("--gene_detection_threshold", type=float, required=True, help="Minimum expression value for a gene to be considered detected.")
    parser.add_argument("--n_components_target", type=str, default="auto", help="Number of GMM components for target genes, or 'auto'.")
    parser.add_argument("--n_components_exclu", type=str, default="auto", help="Number of GMM components for exclude genes, or 'auto'.")
    parser.add_argument("--min_mean_expression", type=float, default=2.0, help="Minimum mean expression level for the higher component can be considered as target cluster.")
    parser.add_argument("--do_QC", type=str, default="True", help="Whether to perform quality control")
    parser.add_argument("--plot_folder", type=str, default=None, help="Directory to save plots. Plots will not be generated if None.")
    parser.add_argument("--species", type=str, default="hsapiens", help="Species name for Ensembl database.")
    return parser.parse_args()

def preprocess_adata(adata: sc.AnnData, already_normalized: bool) -> sc.AnnData:
    """
    Normalizes and logs AnnData, then computes PCA and UMAP if there are enough cells.
    
    This function modifies the input AnnData object by adding "raw" and "log1p" 
    layers, and populating the 'obsm' and 'uns' attributes with PCA and UMAP 
    results, respectively, only if there are at least 3 cells.
    
    Args:
        adata (sc.AnnData): An annotated data matrix.
        
    Returns:
        sc.AnnData: The processed AnnData object.
    """
    adata = adata.copy()

    if already_normalized:
        print("Data appears to be already normalized. Skipping normalization step.")
    else:
        adata.layers["raw"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()
    
    # Only compute PCA, neighbors, and UMAP if there are enough cells
    if adata.shape[0] >= 10:
        sc.pp.pca(adata, svd_solver='arpack', mask_var=None)
        n_neighbors = min(15, adata.shape[0] - 1)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
    return adata

def prepare_adata(adata, species):
    count_matrix = adata.X
    original_var_names = pd.Index(adata.var_names)
    renamed_genes = rename_genes(original_var_names.tolist(), species)

    # Ensure every gene name is a string and fall back to the original name when missing
    genes = pd.Series(renamed_genes, index=original_var_names, dtype="object")
    missing_mask = genes.isna() | genes.eq("")
    genes.loc[missing_mask] = genes.index[missing_mask]
    genes = genes.astype(str).str.strip().str.upper()

    adata = sc.AnnData(X=count_matrix, obs=adata.obs.copy(), var=adata.var.copy())
    adata.var_names = genes
    adata.var_names_make_unique()
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
    genes_df = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'hgnc_symbol', 'external_synonym'])
    genes_upper = [g.strip().upper() for g in genes]

    alias_name = set()
    for g in genes_upper:
        mask = genes_df.isin([g]).any(axis=1)
        rows = genes_df[mask]
        for _, row in rows.iterrows():
            alias_name.add(row['Gene stable ID'])
            alias_name.add(row['Gene name'])
            alias_name.add(row['HGNC symbol'])
            alias_name.add(row['Gene Synonym'])
    return alias_name

def kde_cross_validation(data, bw_candidates, cv_folds=5):
    """
    Perform cross-validation to find optimal bandwidth for KDE.
    
    Args:
        data: 1D array of data points
        bw_candidates: list of bandwidth values to test
        cv_folds: number of cross-validation folds
    
    Returns:
        optimal_bw: best bandwidth value
        cv_scores: dictionary with mean and std of log-likelihood for each bandwidth
    """    
    cv_scores = {}
    for bw in bw_candidates:
        fold_scores = []
        for _ in range(cv_folds):
            # Split data using train_test_split
            train_data, test_data = train_test_split(data, test_size=1/3, random_state=None)
            
            # Fit KDE on training data
            try:
                kde_train = gaussian_kde(train_data, bw_method=bw)
                # Evaluate log-likelihood on test data
                log_likelihood = np.sum(np.log(kde_train(test_data) + 1e-10))  # Add small epsilon to avoid log(0)
                fold_scores.append(log_likelihood)
            except Exception as e:
                fold_scores.append(-np.inf)
        
        cv_scores[bw] = {
            'mean': np.mean(fold_scores),
            'std': np.std(fold_scores)
        }

    # Find optimal bandwidth
    optimal_bw = max(cv_scores.keys(), key=lambda x: cv_scores[x]['mean'] - cv_scores[x]['std'])

    return optimal_bw, cv_scores

def find_optimal_gmm_components(data: np.ndarray) -> tuple[int, np.ndarray | None]:
    """
    """
    data = data.flatten()
    data = data[np.isfinite(data)]  # Remove NaN and inf values
    if len(data) < 2 or np.var(data) == 0:
        return 1, None
    bw_candidates = np.logspace(-1, 0.3, 100)  # From 0.1 to 2.0, 150 values logarithmically spaced
    print("Starting cross-validation for bandwidth selection...")
    # Perform cross-validation
    optimal_bw, _ = kde_cross_validation(data, bw_candidates, cv_folds=5)

    # Step 1: Compute KDE with optimal bandwidth
    kde_optimal = gaussian_kde(data, bw_method=optimal_bw)

    # Evaluate KDE on a dense grid for smooth approximation
    x_grid = np.linspace(data.min(), data.max(), 1500)
    y_grid = kde_optimal(x_grid)

    # Step 2: Find peaks
    peaks, _ = find_peaks(y_grid, prominence=0.01)

    if len(peaks) > 0:
        spline_interp = UnivariateSpline(x_grid, y_grid, s=0, k=3)
        deriv2 = spline_interp.derivative().derivative()

        threshold_deriv2 = -0.001

        significant_peaks = []
        for peak_idx in peaks:
            d2 = deriv2(x_grid[peak_idx])
            if d2 < threshold_deriv2:
                significant_peaks.append(peak_idx)
        
        peaks = np.array(significant_peaks)

    n_peaks = len(peaks)
    estimated_means = x_grid[peaks]

    if n_peaks == 0:
        n_peaks = 1
        estimated_means = None
    else:
        estimated_means = estimated_means.reshape(-1, 1)

    return n_peaks, estimated_means

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
        optimal_n, estimated_means = find_optimal_gmm_components(data)
        print(f"\tOptimal number of components: {optimal_n}")
    else:
        optimal_n, estimated_means = int(n_components), None
        print(f"Using specified number of components for {category}: {optimal_n}")
    gmm = GaussianMixture(n_components=optimal_n, means_init=estimated_means).fit(data)
    probas = gmm.predict_proba(data)
    return gmm, probas

def ashmann_distance(m1, m2, s1, s2):
    """
    Computes the Ashmann distance between two points in a feature space.

    Args:
        m1 (float): Mean of the first distribution.
        m2 (float): Mean of the second distribution.
        s1 (float): Standard deviation of the first distribution.
        s2 (float): Standard deviation of the second distribution.

    Return:
        float: The Ashmann distance.
    """
    return np.abs(m1 - m2) / np.sqrt(s1**2 + s2**2)

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

    print("Generating and saving plots...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    if 'X_umap' in adata.obsm:
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
        
        # Plot UMAP colored by probability of being a target cell
        sc.pl.umap(
            adata,
            color="proba_target",
            cmap=gray_to_red,
            size=50,
            ax=axes[1, 0],
            show=False,
            title=f"UMAP colored by probability of being a target cell\n(Gaussian Mixture Model with {gmm.n_components} components)"
        )
    else:
        axes[0, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("UMAP colored by target mean expression")
        axes[1, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title(f"UMAP colored by probability of being a target cell\n(Gaussian Mixture Model with {gmm.n_components} components)")
    
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

    # Plot histogram of probability of being a target cell
    axes[1, 1].hist(adata.obs["proba_target"], bins=100)
    axes[1, 1].set(title="Histogram of Target Probability", xlabel="Target Probability", ylabel="Number of Cells")
    axes[1, 1].grid(True)
    
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Targets plots saved to: {plot_path}")

def plots_exclude(adata: sc.AnnData, study_name: str, plot_folder: str, exclude_names: list[str], gmm_excludes=dict):
    plot_path = os.path.join(plot_folder, f"{study_name}_exclude.png")
    n_rows = len(exclude_names) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5*n_rows))

    nrow = 0
    for exclude in exclude_names:
        if 'X_umap' in adata.obsm:
            sc.pl.umap(adata, color=f"{exclude}_mean_expr", cmap="viridis", ax=axes[nrow, 0], size=50, 
                       show=False, title=f"UMAP colored by {exclude} mean expression")
        else:
            axes[nrow, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=axes[nrow, 0].transAxes)
            axes[nrow, 0].set_title(f"UMAP colored by {exclude} mean expression")

        gmm = gmm_excludes[exclude]
        axes[nrow, 1].hist(adata.obs[f"{exclude}_mean_expr"], bins=100, alpha=0.6, density=True)
        axes[nrow, 1].set(
            title=f"Distribution of {exclude} mean expression (GMM with {gmm.n_components} components)",
            xlabel="Mean expression per cell",
            ylabel="Density"
        )
        axes[nrow, 1].grid(True)
        nrow += 1

    # Plot UMAP colored by score
    if 'X_umap' in adata.obsm:
        try:
            min_score = adata.obs["score"].min()
            max_score = adata.obs["score"].max()
            if min_score == max_score:
                sc.pl.umap(adata, color="score", cmap='gray', ax=axes[nrow, 0], size=50, show=False)
            else:
                norm_cmap = mcolors.TwoSlopeNorm(vmin=min_score, vcenter=0, vmax=max_score)
                sc.pl.umap(adata, color="score", cmap=blue_gray_red, norm=norm_cmap, ax=axes[nrow, 0], size=50, show=False)
        except:
            sc.pl.umap(adata, color="score", cmap=gray_to_red, ax=axes[nrow, 0], size=50, show=False)
    else:
        axes[nrow, 0].text(0.5, 0.5, 'UMAP not computed\n(Too few cells)', ha='center', va='center', transform=axes[nrow, 0].transAxes)
    axes[nrow, 0].set_title("UMAP colored by score")
    # Plot histogram of score
    axes[nrow, 1].hist(adata.obs["score"], bins=100, color='blue', alpha=0.7)
    axes[nrow, 1].set(title="Histogram of Score", xlabel="Score", ylabel="Number of Cells")
    axes[nrow, 1].grid(True)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

def find_target_cells(
    adata: sc.AnnData,
    study_name: str,
    candidate_genes: list[str],
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
    candidate_aliases = get_gene_aliases(candidate_genes, species=species)
    target_aliases = get_gene_aliases(target_genes, species=species)
    
    candidate_genes_avail = candidate_aliases.intersection(adata.var_names)
    target_genes_avail = target_aliases.intersection(adata.var_names)

    if exclude_genes != dict():
        exclude_genes_avail = dict()
        for category, genes in exclude_genes.items():
            exclude_aliases = get_gene_aliases(genes, species=species)
            exclude_genes_avail[category] = exclude_aliases.intersection(adata.var_names)

    print(f"Available candidate genes: {candidate_genes_avail}")
    print(f"Available target genes: {target_genes_avail}")
    for category, genes in exclude_genes_avail.items():
        print(f"Available exclude genes for {category}: {genes}")

    print(f"\n--- 2. Find candidate cells and log1p-cpm normalize ---")
    print(f"Keeping cells that express at least {int(min_genes_detected)} candidate genes, each above a detection threshold of {int(gene_detection_threshold)}.")
    
    already_normalized = not (np.sum(adata.X, axis=1) == np.sum(adata.X, axis=1).astype(int)).all()

    # Filter cells based on candidate gene expression
    if already_normalized:
        print("Data appears to be already normalized. Impossible to find candidate cells. Continue with all cells.")
        candidate_cells = adata.copy()
    else:
        candidate_cells = find_candidate_cells(adata, candidate_genes_avail, min_genes_detected, gene_detection_threshold)

    if candidate_cells.shape[0] == 0:
        print("No candidate cells found. Returning an empty AnnData object.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        return candidate_cells
    
    print(f"Number of candidate cells: {candidate_cells.shape[0]} ({candidate_cells.shape[0]/adata.shape[0]*100:.2f}%)")

    if candidate_cells.shape[0] < 2:
        print("Not enough candidate cells for GMM fitting. Setting proba_target to 0.")
        candidate_cells.obs["proba_target"] = np.zeros(candidate_cells.shape[0])
        return candidate_cells

    print("Normalizing and log1p transforming data...")
    candidate_cells = preprocess_adata(candidate_cells, already_normalized=already_normalized)
    
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
        means_components = [float(m) for m in gmm_target.means_.flatten()]

        lst_comp = [int(target_component)]
        means_components[target_component] = -1 
        target_before = np.argmax(means_components)

        m_b, s_b = gmm_target.means_[target_before][0], np.sqrt(gmm_target.covariances_[target_before][0][0])
        m_t, s_t = gmm_target.means_[target_component][0], np.sqrt(gmm_target.covariances_[target_component][0][0])

        while ashmann_distance(m_b, m_t, s_b, s_t) <= 2 and target_before > 0 and m_b >= min_mean_expression:
            lst_comp.append(target_before)
            target_component = target_before
            means_components[target_component] = -1
            target_before = np.argmax(means_components)
        
            m_b, s_b = gmm_target.means_[target_before][0], np.sqrt(gmm_target.covariances_[target_before][0][0])
            m_t, s_t = gmm_target.means_[target_component][0], np.sqrt(gmm_target.covariances_[target_component][0][0])

        candidate_cells.obs["proba_target"] = np.sum(probas_target[:, lst_comp], axis=1)

    if exclude_genes_avail:
        print("--- 4bis. Calculate probabilities for exclude genes ---")
        score = candidate_cells.obs["proba_target"]
        for category, gmm, proba in zip(gmm_exclude.keys(), gmm_exclude.values(), probas_exclude.values()):
            exclude_component = np.argmax(gmm.means_.flatten())
            if gmm.means_.flatten()[exclude_component] < min_mean_expression:
                print(f"{category} component mean ({gmm.means_.flatten()[exclude_component]:.4f}) is below the minimum mean expression threshold ({min_mean_expression}).")
                candidate_cells.obs[f"proba_{category}"] = 0
            else:
                means_components = [float(m) for m in gmm.means_.flatten()]

                lst_comp = [int(exclude_component)]
                means_components[exclude_component] = -1 
                exclude_before = np.argmax(means_components)

                m_b, s_b = gmm.means_[exclude_before][0], np.sqrt(gmm.covariances_[exclude_before][0][0])
                m_t, s_t = gmm.means_[exclude_component][0], np.sqrt(gmm.covariances_[exclude_component][0][0])

                while ashmann_distance(m_b, m_t, s_b, s_t) <= 2 and exclude_before > 0 and m_b >= min_mean_expression:
                    lst_comp.append(exclude_before)
                    exclude_component = exclude_before
                    means_components[exclude_component] = -1
                    exclude_before = np.argmax(means_components)
                
                    m_b, s_b = gmm.means_[exclude_before][0], np.sqrt(gmm.covariances_[exclude_before][0][0])
                    m_t, s_t = gmm.means_[exclude_component][0], np.sqrt(gmm.covariances_[exclude_component][0][0])
                    
                candidate_cells.obs[f"proba_{category}"] = np.sum(proba[:, lst_comp], axis=1)
            score = score - candidate_cells.obs[f"proba_{category}"]

        print(f"\n--- 5. Calculate score ---")
        candidate_cells.obs["score"] = score

    # Plot results if a folder is specified
    if plot_folder:
        plots_target(candidate_cells, gmm_target, study_name, plot_folder)
        if exclude_genes_avail:
            plots_exclude(candidate_cells, study_name, plot_folder, exclude_names=list(exclude_genes.keys()), gmm_excludes=gmm_exclude)

    return candidate_cells

def main():
    start_time = time.time()
    
    args = parse_arguments()
    args.exclude_genes = json.loads(args.exclude_genes) if args.exclude_genes != "" else dict()

    try:
        adata = sc.read_h5ad(args.h5ad_file)
        if args.do_QC == "False":
            adata = prepare_adata(adata, species=args.species)
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