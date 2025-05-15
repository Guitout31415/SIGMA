import shutup; shutup.please()
import numpy as np
import scanpy as sc
import scrublet as scr
from scipy.stats import median_abs_deviation
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--study", "-s", type=str, help="Name of the study", required=True)
    parser.add_argument("--study_recreate_file", "-i", type=str, help="Absolute path to the folder containing the study files", required=True)
    parser.add_argument("--output_file", "-o", type=str, help="Absolute path to the preprocesses h5ad file", required=True)
    return parser.parse_args()


def identify_special_genes(adata):
    """Identify special genes (e.g., mitochondrial, ribosomal, hemoglobin) in the AnnData object.

    :param adata: (sc.AnnData) AnnData object

    :return: (sc.AnnData) AnnData object with special genes identified

    Notes:
    - The function identifies mitochondrial genes, ribosomal genes, and hemoglobin genes.
    - The function adds columns to the AnnData object for each special gene type.

    Examples:
    >>> adata = identify_special_genes(adata)
    # mt genes : 100
    # ribo genes : 20
    # hb genes : 5
    """
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.match(r"^RP[LS]\d+")
    adata.var["hb"] = adata.var_names.str.match(r"^HB[^P]")
    print(f"- # mt genes : {adata.var.mt.sum()}")
    print(f"- # ribo genes : {adata.var.ribo.sum()}")
    print(f"- # hb genes : {adata.var.hb.sum()}")

    return adata.copy()


def calculate_outlier(adata, metric, nmads):
    """Calculate outliers for a given metric in the AnnData object.

    :param adata: (sc.AnnData) AnnData object
    :param metric: (str) Metric to calculate outliers
    :param nmads: (int) Number of median absolute deviations to consider as outliers

    :return: (np.ndarray) Boolean array indicating outliers

    Notes:
    - The function calculates outliers based on the median and median absolute deviation.
    - The function returns a boolean array indicating the outliers for the given metric.

    Examples:
    >>> calculate_outlier(adata, "log1p_total_counts", 5)
    array([False, False, False, ..., False, False, False])
    """
    M = adata.obs[metric]
    med = np.median(M)
    mad = median_abs_deviation(M)
    lower_bound = med - nmads * mad
    upper_bound = med + nmads * mad
    return (M < lower_bound) | (M > upper_bound)


def quality_control(adata, n_mads=5, percent_top=20):
    """Perform quality control on the AnnData object.

    :param adata: (sc.AnnData) AnnData object
    :param n_mads: (int) Number of median absolute deviations to consider as outliers (default: 5)
    :param percent_top: (int) Percentage of top genes to consider for quality control metrics (default: 20)

    :return: (sc.AnnData) AnnData object after quality control

    Notes:
    - The function performs quality control on the AnnData object.
    - The function identifies special genes (e.g., mitochondrial, ribosomal, hemoglobin).
    - The function computes quality control metrics (e.g., total counts, number of genes, percentage of counts in top genes).
    - The function detects outliers based on the computed metrics.
    - The function filters out low-quality cells based on the outliers.
    - The function detects doublets using Scrublet.
    """
    print(f"Initial number of cells: {adata.n_obs}")

    # Identify special genes
    adata = identify_special_genes(adata)

    # Compute quality control metrics
    print("Computing quality control metrics...")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=[percent_top],
        log1p=True,
    )

    # Detect outliers
    print("Detecting outliers...")
    metrics = [
        "log1p_total_counts",
        "log1p_n_genes_by_counts",
        "pct_counts_in_top_20_genes",
    ]
    nmads = n_mads
    outlier_results = [calculate_outlier(adata, m, nmads) for m in metrics]
    adata.obs["outlier"] = np.any(outlier_results, axis=0)
    adata.obs["mt_outlier"] = calculate_outlier(adata, "pct_counts_mt", 3) | (
        adata.obs["pct_counts_mt"] > 8
    )

    # Filter low-quality cells
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    print(f" - Number of cells after filtering outliers: {adata.n_obs}")

    # Doublet detection
    print("Detecting doublets...")
    scrub = scr.Scrublet(adata.X)
    doublet_scores, _ = scrub.scrub_doublets(verbose=False)
    adata.obs["doublet_score"] = doublet_scores
    try:
        # Try automatic threshold detection first
        doublet_mask = scrub.call_doublets()
        print(f"Automatically identified doublet score threshold: {scrub.threshold_}")
    except Exception as e:
        print(f"Warning: {str(e)}")
        # Set a conservative manual threshold if automatic detection fails
        threshold = 0.25  # Conservative default threshold
        doublet_mask = scrub.call_doublets(threshold=threshold)
        print(f"Using manual doublet score threshold: {threshold}")

    adata.obs["doublet_class"] = doublet_mask
    adata.obs["doublet_class"] = (
        adata.obs["doublet_class"].astype(str).astype("category")
    )

    # Filter doublets
    adata = adata[adata.obs["doublet_class"] == "False"].copy()

    print(f"Final number of cells: {adata.n_obs}")

    return adata.copy()


if __name__ == "__main__":
    args = parse_arguments()
    adata = sc.read_h5ad(args.study_recreate_file) # Load the data

    print("\n===============================")
    print("Quality control...")
    adata_qc = quality_control(adata)
    print("-------------------------------")

    print(f"Saving in {args.output_file}")
    adata_qc.write(args.output_file) # Save the extracted platelets
